//! Typed errors for the agent harness.
//!
//! Both consumers previously used `anyhow` end-to-end, which meant a caller
//! could not tell a rate-limit from an auth failure from a malformed response —
//! and therefore could not implement retry or backoff of its own. Every variant
//! here exists because some caller needs to branch on it.

use std::time::Duration;

/// An error from the provider transport layer.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The request did not complete within [`crate::provider::ProviderConfig::timeout`].
    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    /// Network/transport failure below the HTTP status layer.
    #[error("transport error: {0}")]
    Transport(#[source] reqwest::Error),

    /// The provider rate-limited us and retries were exhausted.
    #[error("rate limited after {attempts} attempt(s)")]
    RateLimited {
        attempts: u32,
        /// Value of the `Retry-After` header on the final response, if any.
        retry_after: Option<Duration>,
    },

    /// The provider returned a non-success status, or a body carrying `error`.
    #[error("provider returned {status}: {message}")]
    Api { status: u16, message: String },

    /// The response was not the JSON we expected.
    #[error("could not decode provider response: {0}")]
    Decode(String),

    /// Misconfiguration detected before any request was made.
    #[error("configuration error: {0}")]
    Config(String),
}

impl AgentError {
    /// Whether retrying this request could plausibly succeed.
    ///
    /// Deliberately conservative: 4xx other than 408/429 are caller errors and
    /// retrying them only burns quota.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            AgentError::Timeout(_) => true,
            AgentError::RateLimited { .. } => true,
            AgentError::Transport(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            AgentError::Api { status, .. } => *status == 408 || *status >= 500,
            AgentError::Decode(_) | AgentError::Config(_) => false,
        }
    }

    /// The server-suggested delay before retrying, when it gave one.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            AgentError::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// The HTTP status, when the failure carried one.
    #[must_use]
    pub fn status(&self) -> Option<u16> {
        match self {
            AgentError::Api { status, .. } => Some(*status),
            AgentError::RateLimited { .. } => Some(429),
            _ => None,
        }
    }
}

// Deliberately no `From<reqwest::Error>`: it could not know the configured
// timeout, so `?` would produce a misleading "timed out after 0ns". Callers map
// explicitly with the real budget, as `ChatClient::try_once` does.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_failures_are_retryable() {
        assert!(AgentError::Timeout(Duration::from_secs(1)).is_retryable());
        assert!(
            AgentError::RateLimited {
                attempts: 1,
                retry_after: None
            }
            .is_retryable()
        );
        for status in [500u16, 502, 503, 504, 408] {
            assert!(
                AgentError::Api {
                    status,
                    message: String::new()
                }
                .is_retryable(),
                "{status} should be retryable"
            );
        }
    }

    #[test]
    fn caller_errors_are_not_retryable() {
        for status in [400u16, 401, 403, 404, 422] {
            assert!(
                !AgentError::Api {
                    status,
                    message: String::new()
                }
                .is_retryable(),
                "{status} must not be retried — it burns quota and will fail again"
            );
        }
        assert!(!AgentError::Decode("bad".into()).is_retryable());
        assert!(!AgentError::Config("missing key".into()).is_retryable());
    }

    #[test]
    fn rate_limit_reports_status_and_retry_after() {
        let e = AgentError::RateLimited {
            attempts: 3,
            retry_after: Some(Duration::from_secs(2)),
        };
        assert_eq!(e.status(), Some(429));
        assert_eq!(e.retry_after(), Some(Duration::from_secs(2)));
    }
}
