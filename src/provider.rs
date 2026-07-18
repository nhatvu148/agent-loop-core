//! A resilient chat-completions client.
//!
//! This exists because both consumers built their HTTP client with a bare
//! `reqwest::Client::new()`. **reqwest applies no default timeout**, so a stalled
//! provider connection hung the whole agent run with no upper bound, and a
//! single 429 discarded every turn of accumulated work.
//!
//! The surface is deliberately thin: request and response are both
//! `serde_json::Value`, so each consumer keeps its own typed request/response
//! structs and only shares the transport. That keeps this crate from becoming a
//! second, competing model abstraction.

use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde_json::Value;

use crate::error::AgentError;

/// How hard to retry a failed request.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Retries *after* the first attempt. `0` disables retrying.
    pub max_retries: u32,
    /// Delay before the first retry; doubles each attempt.
    pub initial_backoff: Duration,
    /// Ceiling for the doubling.
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(8),
        }
    }
}

impl RetryPolicy {
    /// No retries — useful in tests and for callers that own their own policy.
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    fn backoff_for(&self, attempt: u32) -> Duration {
        let scaled = self
            .initial_backoff
            .saturating_mul(2u32.saturating_pow(attempt));
        scaled.min(self.max_backoff)
    }
}

/// Everything needed to talk to one OpenAI-compatible endpoint.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// e.g. `https://openrouter.ai/api/v1` — no trailing slash, no path.
    pub base_url: String,
    /// Bearer token. **Empty means keyless**: no `Authorization` header is
    /// sent. Local providers (Ollama) and the Claude Code backend have no key,
    /// and callers legitimately pass an empty string for them.
    pub api_key: String,
    /// Whole-request budget, the thing that was missing.
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub retry: RetryPolicy,
    /// Provider-specific extras, e.g. OpenRouter's `HTTP-Referer` / `X-Title`.
    pub extra_headers: Vec<(String, String)>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            timeout: Duration::from_secs(120),
            connect_timeout: Duration::from_secs(10),
            retry: RetryPolicy::default(),
            extra_headers: Vec::new(),
        }
    }
}

/// A pooled HTTP client with a timeout and a retry policy attached.
///
/// Construct this **once** and clone/share it. Both consumers previously built a
/// fresh `Client` per call site, discarding reqwest's connection pool.
#[derive(Debug, Clone)]
pub struct ChatClient {
    http: Client,
    cfg: ProviderConfig,
}

impl ChatClient {
    /// # Errors
    /// If `base_url` is empty, or the HTTP client cannot be built.
    ///
    /// An empty `api_key` is **not** an error — see [`ProviderConfig::api_key`].
    /// Rejecting it here would break keyless providers, whose callers pass
    /// `unwrap_or_default()`.
    pub fn new(cfg: ProviderConfig) -> Result<Self, AgentError> {
        if cfg.base_url.trim().is_empty() {
            return Err(AgentError::Config("base_url is empty".into()));
        }
        let http = Client::builder()
            .timeout(cfg.timeout)
            .connect_timeout(cfg.connect_timeout)
            .build()
            .map_err(AgentError::Transport)?;
        Ok(Self { http, cfg })
    }

    /// POST to `{base_url}/chat/completions`, retrying transient failures.
    ///
    /// # Errors
    /// [`AgentError::RateLimited`] when 429s outlast the retry policy,
    /// [`AgentError::Timeout`] on budget exhaustion, [`AgentError::Api`] for
    /// non-retryable statuses, [`AgentError::Decode`] for a non-JSON body.
    pub async fn post_chat(&self, body: &Value) -> Result<Value, AgentError> {
        self.post("chat/completions", body).await
    }

    /// POST to an arbitrary path under `base_url`, retrying transient failures.
    ///
    /// # Errors
    /// See [`ChatClient::post_chat`].
    pub async fn post(&self, path: &str, body: &Value) -> Result<Value, AgentError> {
        let url = format!("{}/{}", self.cfg.base_url.trim_end_matches('/'), path);
        let max = self.cfg.retry.max_retries;

        let mut attempt = 0u32;
        loop {
            let err = match self.try_once(&url, body).await {
                Ok(v) => return Ok(v),
                Err(e) => e,
            };

            let attempts_made = attempt + 1;
            if !err.is_retryable() || attempt >= max {
                // Report the exhausted 429 with an accurate attempt count.
                if let AgentError::RateLimited { retry_after, .. } = err {
                    return Err(AgentError::RateLimited {
                        attempts: attempts_made,
                        retry_after,
                    });
                }
                return Err(err);
            }

            let delay = err
                .retry_after()
                .unwrap_or_else(|| self.cfg.retry.backoff_for(attempt));
            tracing::warn!(
                attempt = attempts_made,
                max_attempts = max + 1,
                delay_ms = delay.as_millis() as u64,
                error = %err,
                "retrying provider request"
            );
            tokio::time::sleep(delay).await;
            attempt += 1;
        }
    }

    async fn try_once(&self, url: &str, body: &Value) -> Result<Value, AgentError> {
        let mut req = self.http.post(url);
        if !self.cfg.api_key.trim().is_empty() {
            req = req.bearer_auth(&self.cfg.api_key);
        }
        for (k, v) in &self.cfg.extra_headers {
            req = req.header(k, v);
        }

        let res = req.json(body).send().await.map_err(|e| {
            if e.is_timeout() {
                AgentError::Timeout(self.cfg.timeout)
            } else {
                AgentError::Transport(e)
            }
        })?;

        let status = res.status();
        let retry_after = parse_retry_after(&res);
        let text = res.text().await.map_err(|e| {
            if e.is_timeout() {
                AgentError::Timeout(self.cfg.timeout)
            } else {
                AgentError::Transport(e)
            }
        })?;

        if status == StatusCode::TOO_MANY_REQUESTS {
            return Err(AgentError::RateLimited {
                attempts: 1,
                retry_after,
            });
        }

        // Classify by status BEFORE decoding. A 502/503 from a proxy is
        // typically an HTML page or empty body; decoding first would surface it
        // as a non-retryable `Decode` error and defeat the retry policy.
        if !status.is_success() {
            return Err(AgentError::Api {
                status: status.as_u16(),
                message: error_message(&text),
            });
        }

        let data: Value = serde_json::from_str(&text)
            .map_err(|e| AgentError::Decode(format!("{e}: {}", clip(&text, 300))))?;

        // Some OpenAI-compatible providers return 200 with an `error` member.
        if data.get("error").is_some() {
            return Err(AgentError::Api {
                status: status.as_u16(),
                message: error_message(&text),
            });
        }

        Ok(data)
    }
}

/// `Retry-After` in delta-seconds form. The HTTP-date form is ignored on
/// purpose — honouring it needs a date parser, and providers send seconds.
fn parse_retry_after(res: &reqwest::Response) -> Option<Duration> {
    res.headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Best-effort human message from an error body: the provider's own
/// `error.message` when the body is JSON, otherwise the clipped raw body.
fn error_message(text: &str) -> String {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|d| {
            d.pointer("/error/message")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| clip(text, 400))
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{header, method, path as pathm};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    /// Counts calls and replays a fixed script of responses.
    #[derive(Clone)]
    struct Script(Arc<ScriptInner>);
    struct ScriptInner {
        responses: Vec<ResponseTemplate>,
        calls: AtomicUsize,
    }

    impl Script {
        fn new(responses: Vec<ResponseTemplate>) -> Self {
            Script(Arc::new(ScriptInner {
                responses,
                calls: AtomicUsize::new(0),
            }))
        }
        fn calls(&self) -> usize {
            self.0.calls.load(Ordering::SeqCst)
        }
    }

    impl Respond for Script {
        fn respond(&self, _: &Request) -> ResponseTemplate {
            let i = self.0.calls.fetch_add(1, Ordering::SeqCst);
            self.0
                .responses
                .get(i)
                .cloned()
                .unwrap_or_else(|| self.0.responses.last().unwrap().clone())
        }
    }

    async fn serve(responses: Vec<ResponseTemplate>) -> (MockServer, Script) {
        let s = MockServer::start().await;
        let script = Script::new(responses);
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .respond_with(script.clone())
            .mount(&s)
            .await;
        (s, script)
    }

    /// Zero backoff so retry tests stay instant.
    fn client(base_url: &str, max_retries: u32) -> ChatClient {
        ChatClient::new(ProviderConfig {
            base_url: base_url.to_string(),
            api_key: "test-key".to_string(),
            retry: RetryPolicy {
                max_retries,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
            },
            ..ProviderConfig::default()
        })
        .unwrap()
    }

    fn ok_body() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({"choices": [], "ok": true}))
    }

    // ---- happy path --------------------------------------------------------

    #[tokio::test]
    async fn successful_response_is_returned_verbatim() {
        let (srv, script) = serve(vec![ok_body()]).await;
        let out = client(&srv.uri(), 3)
            .post_chat(&json!({"model": "m"}))
            .await
            .unwrap();
        assert_eq!(out["ok"], true);
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn auth_and_extra_headers_are_sent() {
        let srv = MockServer::start().await;
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .and(header("x-title", "vexar"))
            .respond_with(ok_body())
            .expect(1)
            .mount(&srv)
            .await;

        let c = ChatClient::new(ProviderConfig {
            base_url: srv.uri(),
            api_key: "test-key".to_string(),
            extra_headers: vec![("X-Title".into(), "vexar".into())],
            retry: RetryPolicy::none(),
            ..ProviderConfig::default()
        })
        .unwrap();
        c.post_chat(&json!({})).await.unwrap();
        // `expect(1)` is asserted on drop.
    }

    #[tokio::test]
    async fn base_url_trailing_slash_is_tolerated() {
        let (srv, _s) = serve(vec![ok_body()]).await;
        let c = client(&format!("{}/", srv.uri()), 0);
        assert!(c.post_chat(&json!({})).await.is_ok());
    }

    // ---- the bug this crate exists to fix ----------------------------------

    #[tokio::test]
    async fn a_429_is_retried_and_then_succeeds() {
        let (srv, script) = serve(vec![ResponseTemplate::new(429), ok_body()]).await;
        let out = client(&srv.uri(), 3).post_chat(&json!({})).await.unwrap();
        assert_eq!(out["ok"], true);
        assert_eq!(script.calls(), 2, "one failure, one success");
    }

    #[tokio::test]
    async fn persistent_429_exhausts_retries_and_reports_attempts() {
        let (srv, script) = serve(vec![ResponseTemplate::new(429)]).await;
        let err = client(&srv.uri(), 2)
            .post_chat(&json!({}))
            .await
            .unwrap_err();

        match err {
            AgentError::RateLimited { attempts, .. } => assert_eq!(attempts, 3),
            other => panic!("expected RateLimited, got {other:?}"),
        }
        assert_eq!(script.calls(), 3, "initial attempt + 2 retries");
    }

    #[tokio::test]
    async fn server_errors_are_retried() {
        let (srv, script) = serve(vec![
            ResponseTemplate::new(503),
            ResponseTemplate::new(502),
            ok_body(),
        ])
        .await;
        assert!(client(&srv.uri(), 3).post_chat(&json!({})).await.is_ok());
        assert_eq!(script.calls(), 3);
    }

    #[tokio::test]
    async fn a_gateway_error_with_a_non_json_body_is_still_retried() {
        // Regression: classifying by body before status made a proxy's HTML 502
        // a non-retryable `Decode` error, defeating the retry policy in exactly
        // the case it exists for.
        let (srv, script) = serve(vec![
            ResponseTemplate::new(502).set_body_string("<html>Bad Gateway</html>"),
            ok_body(),
        ])
        .await;
        assert!(client(&srv.uri(), 3).post_chat(&json!({})).await.is_ok());
        assert_eq!(script.calls(), 2);
    }

    #[tokio::test]
    async fn client_errors_are_not_retried() {
        let (srv, script) = serve(vec![
            ResponseTemplate::new(400).set_body_json(json!({"error": {"message": "bad model"}})),
        ])
        .await;
        let err = client(&srv.uri(), 3)
            .post_chat(&json!({}))
            .await
            .unwrap_err();

        assert!(matches!(err, AgentError::Api { status: 400, .. }));
        assert!(err.to_string().contains("bad model"));
        assert_eq!(script.calls(), 1, "retrying a 400 only burns quota");
    }

    #[tokio::test]
    async fn retry_after_header_is_honoured() {
        let (srv, script) = serve(vec![
            ResponseTemplate::new(429).insert_header("Retry-After", "0"),
            ok_body(),
        ])
        .await;
        assert!(client(&srv.uri(), 3).post_chat(&json!({})).await.is_ok());
        assert_eq!(script.calls(), 2);
    }

    #[tokio::test]
    async fn a_stalled_provider_times_out_instead_of_hanging_forever() {
        // The headline bug: with a bare `reqwest::Client::new()` this request
        // would hang indefinitely.
        let srv = MockServer::start().await;
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .respond_with(ok_body().set_delay(Duration::from_secs(30)))
            .mount(&srv)
            .await;

        let c = ChatClient::new(ProviderConfig {
            base_url: srv.uri(),
            api_key: "k".to_string(),
            timeout: Duration::from_millis(80),
            retry: RetryPolicy::none(),
            ..ProviderConfig::default()
        })
        .unwrap();

        let started = std::time::Instant::now();
        let err = c.post_chat(&json!({})).await.unwrap_err();
        assert!(matches!(err, AgentError::Timeout(_)), "got {err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "gave up promptly"
        );
    }

    // ---- response shape ----------------------------------------------------

    #[tokio::test]
    async fn an_error_member_on_a_200_is_still_an_error() {
        let (srv, _s) = serve(vec![
            ResponseTemplate::new(200).set_body_json(json!({"error": {"message": "no credit"}})),
        ])
        .await;
        let err = client(&srv.uri(), 0)
            .post_chat(&json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Api { status: 200, .. }));
        assert!(err.to_string().contains("no credit"));
    }

    #[tokio::test]
    async fn a_non_json_body_is_a_decode_error_not_a_panic() {
        let (srv, script) = serve(vec![
            ResponseTemplate::new(200).set_body_string("<html>502</html>"),
        ])
        .await;
        let err = client(&srv.uri(), 3)
            .post_chat(&json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Decode(_)), "got {err:?}");
        assert_eq!(script.calls(), 1, "a malformed body is not retryable");
    }

    // ---- config ------------------------------------------------------------

    #[test]
    fn an_empty_base_url_is_rejected_before_any_request() {
        assert!(matches!(
            ChatClient::new(ProviderConfig::default()),
            Err(AgentError::Config(_))
        ));
    }

    #[test]
    fn a_keyless_provider_is_accepted() {
        // Ollama and the Claude Code backend have no API key, and
        // vexar-desktop resolves it with `unwrap_or_default()`. Rejecting an
        // empty key here broke offline/local users at client construction.
        assert!(
            ChatClient::new(ProviderConfig {
                base_url: "http://localhost:11434/v1".into(),
                ..ProviderConfig::default()
            })
            .is_ok()
        );
    }

    #[tokio::test]
    async fn no_authorization_header_is_sent_when_the_key_is_empty() {
        let srv = MockServer::start().await;
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .and(wiremock::matchers::header_regex("authorization", ".*"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&srv)
            .await;
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .respond_with(ok_body())
            .mount(&srv)
            .await;

        let c = ChatClient::new(ProviderConfig {
            base_url: srv.uri(),
            api_key: String::new(),
            retry: RetryPolicy::none(),
            ..ProviderConfig::default()
        })
        .unwrap();

        // Matches the keyless mock, not the authorization-bearing one.
        assert!(c.post_chat(&json!({})).await.is_ok());
    }

    #[test]
    fn backoff_doubles_and_saturates_at_the_ceiling() {
        let p = RetryPolicy {
            max_retries: 10,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(500),
        };
        assert_eq!(p.backoff_for(0), Duration::from_millis(100));
        assert_eq!(p.backoff_for(1), Duration::from_millis(200));
        assert_eq!(p.backoff_for(2), Duration::from_millis(400));
        assert_eq!(p.backoff_for(3), Duration::from_millis(500));
        assert_eq!(p.backoff_for(30), Duration::from_millis(500));
    }
}
