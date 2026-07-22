//! The one backend trait, and the policy that drives the loop.
//!
//! `vexar` had `AgentBackend` (streams [`AgentEvent`]s) and `pr-review-core` had
//! `ReviewBackend` (returns a structured value). They are the same trait at two
//! altitudes: structured output is a typed terminal on a streaming run. This
//! module collapses them — [`Backend::run`] streams, [`run_structured`] adds the
//! typed ending.

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::AgentError;
use crate::events::{EventSink, StopReason};

/// How the loop should spend model calls.
///
/// The two-tier split comes from `pr-review-core`: a cheap model drives the tool
/// loop to gather context, then a stronger model produces the final answer from
/// what was gathered. That keeps the bulk of the token volume on the cheap model
/// while judgment stays on the good one.
#[derive(Debug, Clone)]
pub struct ModelPolicy {
    /// Model for the tool-calling phase.
    pub explore: String,
    /// Model for the final answer. Equal to `explore` disables tiering.
    pub synthesize: String,
    /// Maximum tool-calling turns before synthesis is forced.
    pub max_turns: u32,
    /// Stop *once cumulative usage reaches this*, checked between turns.
    ///
    /// This is a **stop threshold, not a hard ceiling**: a call's cost cannot be
    /// known before making it, so the total can exceed this by at most one
    /// call's `max_tokens`. Named to say so. `None` disables the check.
    pub stop_after_tokens: Option<u32>,
    /// Wall-clock budget for the whole run.
    pub timeout_secs: Option<u64>,
    /// Per-call output cap sent to the provider. Also bounds the overshoot above.
    pub max_tokens: u32,
    pub temperature: f32,
    /// `tool_choice` for the first exploration turn only; later turns use
    /// `"auto"`. Lets a host force a tool call up front ("required") without
    /// forcing one every turn.
    pub initial_tool_choice: String,
    /// Cap on characters of tool output carried in the conversation.
    pub max_history_chars: usize,
    /// Whether a failing tool aborts the run or is fed back to the model.
    pub continue_on_tool_error: bool,
    /// Whether to run a final tools-forbidden synthesis turn once exploration is
    /// done.
    ///
    /// This is separate from tiering. A tiered run (cheap explore, strong
    /// synthesize) obviously wants it. But a *single-model* run can want it too:
    /// a caller that needs clean structured output uses the synthesis turn to
    /// say "now produce only the JSON, no tools", even on the same model. And a
    /// caller whose ordinary loop completion already *is* the answer (an
    /// interactive assistant) wants it off, to avoid a redundant call. `single`
    /// defaults it off, `tiered` on; set it explicitly when neither fits.
    pub final_synthesis: bool,
    /// Extra top-level fields merged into every request body, applied last so
    /// they win over anything the wire format chose.
    ///
    /// The escape hatch for provider parameters this crate does not model.
    /// jpt-copilot needs `parallel_tool_calls: false` — without it the model
    /// emits a dependent PSJ call in the same turn as the call whose returned
    /// ID it needs, which is a real ordering bug, not a preference.
    pub extra_body: serde_json::Map<String, Value>,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self {
            explore: String::new(),
            synthesize: String::new(),
            max_turns: 6,
            stop_after_tokens: Some(100_000),
            timeout_secs: Some(300),
            max_tokens: 4_000,
            temperature: 0.2,
            initial_tool_choice: "auto".to_string(),
            max_history_chars: 45_000,
            continue_on_tool_error: true,
            final_synthesis: false,
            extra_body: serde_json::Map::new(),
        }
    }
}

impl ModelPolicy {
    /// Single-model policy — `explore` and `synthesize` are the same, and the
    /// loop's own completion is the answer (no separate synthesis turn).
    #[must_use]
    pub fn single(model: impl Into<String>) -> Self {
        let m = model.into();
        Self {
            explore: m.clone(),
            synthesize: m,
            final_synthesis: false,
            ..Self::default()
        }
    }

    /// Two-tier policy: cheap exploration, then a strong-model synthesis turn.
    #[must_use]
    pub fn tiered(explore: impl Into<String>, synthesize: impl Into<String>) -> Self {
        Self {
            explore: explore.into(),
            synthesize: synthesize.into(),
            final_synthesis: true,
            ..Self::default()
        }
    }

    /// Whether tiering is active.
    #[must_use]
    pub fn is_tiered(&self) -> bool {
        self.explore != self.synthesize
    }

    /// Human-readable model attribution, honest about both tiers.
    #[must_use]
    pub fn display_model(&self) -> String {
        if self.is_tiered() {
            format!("{} (explore: {})", self.synthesize, self.explore)
        } else {
            self.synthesize.clone()
        }
    }
}

/// What the loop was asked to do.
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    /// Prior conversation, if resuming.
    pub messages: Vec<Value>,
    /// Names of the tools to expose. Empty exposes all registered tools.
    ///
    /// Tool descriptions are always-on context, so scoping a planner to a few
    /// delegation tools is both cheaper and more accurate than handing it
    /// everything.
    pub tool_scope: Vec<String>,
}

/// What the loop produced.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// Final assistant text, if any.
    pub content: Option<String>,
    /// Model round-trips made, including the synthesis call on a tiered run —
    /// so this can exceed `ModelPolicy::max_turns`, which caps exploration only.
    pub turns: u32,
    pub total_tokens: u32,
    pub duration_ms: u64,
    pub stop_reason: StopReason,
    /// Full transcript, for persistence or a follow-up turn.
    pub messages: Vec<Value>,
    /// Model attribution — see [`ModelPolicy::display_model`].
    pub model: String,
}

/// Something that can run an agent turn.
///
/// Implementors either drive the loop themselves over a chat endpoint
/// ([`crate::loop_runner`]) or delegate wholesale to an external agent (the
/// Claude Code CLI). The distinction matters at construction, not here.
#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    /// Human-readable backend name, for logs and attribution.
    fn name(&self) -> &str;

    /// Whether this backend can resume a prior session by id.
    fn supports_session_resume(&self) -> bool {
        false
    }

    /// Run to completion, emitting progress to `sink`.
    ///
    /// # Errors
    /// Transport and configuration failures. A run that stops early because of
    /// a limit is *not* an error — it returns [`RunOutcome`] with the matching
    /// [`StopReason`].
    async fn run(&self, request: RunRequest, sink: EventSink) -> Result<RunOutcome, AgentError>;
}

/// Run a backend and parse its final message as `T`.
///
/// This is what `ReviewBackend` was: the same loop, with a typed ending. Callers
/// that want no streaming pass [`EventSink::none`].
///
/// # Errors
/// Whatever [`Backend::run`] returns, plus [`AgentError::Decode`] if the final
/// message doesn't contain parseable JSON for `T`.
pub async fn run_structured<T: DeserializeOwned>(
    backend: &dyn Backend,
    request: RunRequest,
    sink: EventSink,
) -> Result<(T, RunOutcome), AgentError> {
    let outcome = backend.run(request, sink).await?;
    let content = outcome.content.as_deref().unwrap_or_default();
    let json = extract_json(content).ok_or_else(|| {
        AgentError::Decode(format!(
            "no JSON object in response: {}",
            clip(content, 300)
        ))
    })?;
    let parsed = serde_json::from_str(json)
        .map_err(|e| AgentError::Decode(format!("{e}: {}", clip(json, 300))))?;
    Ok((parsed, outcome))
}

/// Pull the outermost JSON object or array out of a response that may be
/// wrapped in prose or a ```json fence.
#[must_use]
pub fn extract_json(text: &str) -> Option<&str> {
    let start = text.find(['{', '['])?;
    let open = text.as_bytes()[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (i, b) in text.as_bytes().iter().enumerate().skip(start) {
        if escaped {
            escaped = false;
            continue;
        }
        match b {
            b'\\' if in_string => escaped = true,
            b'"' => in_string = !in_string,
            b if *b == open && !in_string => depth += 1,
            b if *b == close && !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn tiering_is_reported_honestly() {
        let single = ModelPolicy::single("main/model");
        assert!(!single.is_tiered());
        assert_eq!(single.display_model(), "main/model");

        let tiered = ModelPolicy::tiered("cheap/model", "main/model");
        assert!(tiered.is_tiered());
        assert_eq!(tiered.display_model(), "main/model (explore: cheap/model)");
    }

    #[test]
    fn the_token_budget_is_named_as_a_threshold_not_a_ceiling() {
        // Deliberate: a call's cost is unknowable before making it, so the loop
        // checks between turns and can overshoot by at most one call's
        // max_tokens. The field name says so; the bound is max_tokens.
        let p = ModelPolicy::default();
        assert_eq!(p.stop_after_tokens, Some(100_000));
        assert_eq!(p.max_tokens, 4_000, "the overshoot bound");
    }

    #[test]
    fn extract_json_handles_bare_objects() {
        assert_eq!(extract_json(r#"{"a":1}"#), Some(r#"{"a":1}"#));
    }

    #[test]
    fn extract_json_ignores_surrounding_prose_and_fences() {
        assert_eq!(
            extract_json("Sure! ```json\n{\"a\":1}\n``` hope that helps"),
            Some(r#"{"a":1}"#)
        );
    }

    #[test]
    fn extract_json_respects_nesting() {
        let s = r#"prefix {"a":{"b":[1,2]},"c":3} suffix"#;
        assert_eq!(extract_json(s), Some(r#"{"a":{"b":[1,2]},"c":3}"#));
    }

    #[test]
    fn extract_json_is_not_fooled_by_braces_inside_strings() {
        let s = r#"{"a":"}{ not a brace","b":1}"#;
        assert_eq!(extract_json(s), Some(s));
    }

    #[test]
    fn extract_json_handles_escaped_quotes() {
        let s = r#"{"a":"say \"hi\" }","b":2}"#;
        assert_eq!(extract_json(s), Some(s));
    }

    #[test]
    fn extract_json_finds_arrays() {
        assert_eq!(extract_json("here: [1,2,3] done"), Some("[1,2,3]"));
    }

    #[test]
    fn extract_json_returns_none_when_unbalanced_or_absent() {
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json(r#"{"a": 1"#), None);
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Review {
        summary: String,
    }

    struct Canned(Option<String>);

    #[async_trait::async_trait]
    impl Backend for Canned {
        fn name(&self) -> &str {
            "canned"
        }
        async fn run(&self, _: RunRequest, sink: EventSink) -> Result<RunOutcome, AgentError> {
            sink.emit(crate::events::AgentEvent::TurnStart { turn: 1 });
            Ok(RunOutcome {
                content: self.0.clone(),
                turns: 1,
                total_tokens: 10,
                duration_ms: 0,
                stop_reason: StopReason::Complete,
                messages: vec![],
                model: "canned".into(),
            })
        }
    }

    #[tokio::test]
    async fn run_structured_parses_json_out_of_prose() {
        let b = Canned(Some(
            "Here you go:\n```json\n{\"summary\":\"ok\"}\n```".into(),
        ));
        let (review, outcome): (Review, _) =
            run_structured(&b, RunRequest::default(), EventSink::none())
                .await
                .unwrap();
        assert_eq!(
            review,
            Review {
                summary: "ok".into()
            }
        );
        assert_eq!(outcome.total_tokens, 10);
    }

    #[tokio::test]
    async fn run_structured_reports_a_decode_error_with_context() {
        let b = Canned(Some("I could not do that.".into()));
        let err = run_structured::<Review>(&b, RunRequest::default(), EventSink::none())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Decode(_)), "got {err:?}");
        assert!(err.to_string().contains("could not do that"), "{err}");
    }

    #[tokio::test]
    async fn run_structured_works_with_a_live_sink_too() {
        // The same call streams when given a real sink — that is the property
        // that lets one loop serve a desktop UI and a library caller.
        let (sink, mut rx) = EventSink::channel();
        let b = Canned(Some(r#"{"summary":"ok"}"#.into()));
        let _: (Review, _) = run_structured(&b, RunRequest::default(), sink)
            .await
            .unwrap();
        assert!(rx.recv().await.is_some(), "events reached the sink");
    }
}
