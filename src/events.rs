//! Agent lifecycle events and the sink they flow into.
//!
//! Ported from vexar, where one `mpsc` channel of these already drives both the
//! CLI and the desktop UI from a single loop. `pr-review-core` has no streaming
//! at all — every call is a blocking `res.text().await`, so a six-turn review is
//! minutes of silence. Building the sink into the shared loop means it gets
//! progress events for free, while callers that don't want them pass
//! [`EventSink::none`] and keep today's synchronous behaviour exactly.

use tokio::sync::mpsc;

/// Why a run stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The model produced a turn with no tool calls.
    Complete,
    /// The turn cap was reached.
    MaxTurns(u32),
    /// The token budget was reached.
    MaxTokens(u32),
    /// The wall-clock budget was reached.
    Timeout(u64),
    /// The caller set the interrupt flag.
    Interrupted,
    /// A tool failed and the policy was to stop.
    Error(String),
}

/// Something observable that happened during a run.
///
/// Deliberately serialisable-shaped (plain data, no borrowed state) so a UI can
/// forward these over IPC without translation.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A new model turn is about to start. 1-indexed.
    TurnStart { turn: u32 },
    /// The model replied. `has_tool_calls` decides whether the loop continues.
    TurnResponse {
        turn: u32,
        has_tool_calls: bool,
        content_preview: Option<String>,
    },
    /// A chunk of assistant text.
    ContentToken(String),
    /// A tool is about to run.
    ToolStart {
        turn: u32,
        tool: String,
        call_id: String,
    },
    /// A tool finished. `ok` is false for both execution failures and rejected
    /// or unparseable calls.
    ToolComplete {
        turn: u32,
        tool: String,
        call_id: String,
        ok: bool,
        duration_ms: u64,
        /// Leading characters of the result, for a UI that shows it inline.
        /// Always present — an empty result is an empty string, not absence.
        output_preview: String,
    },
    /// Every tool call in this turn has been executed.
    TurnComplete { turn: u32, tools_executed: u32 },
    /// Token usage reported by the provider for one call.
    Usage {
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        total_tokens: u32,
    },
    /// A non-fatal problem worth surfacing.
    Warning { turn: u32, message: String },
    /// The run ended.
    Finished {
        turns: u32,
        total_tokens: u32,
        reason: StopReason,
    },
}

/// Where events go. Cloning is cheap; a `none()` sink discards everything.
///
/// This is the seam that lets one loop serve a streaming desktop UI and a
/// library caller that just wants a return value.
#[derive(Debug, Clone, Default)]
pub struct EventSink(Option<mpsc::UnboundedSender<AgentEvent>>);

impl EventSink {
    /// A sink that drops every event. Zero cost beyond an `Option` check.
    #[must_use]
    pub fn none() -> Self {
        Self(None)
    }

    /// Send events to `tx`.
    #[must_use]
    pub fn new(tx: mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self(Some(tx))
    }

    /// Create a sink paired with its receiver.
    #[must_use]
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<AgentEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    /// Whether anything is listening. Use to skip building an expensive payload.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    /// Emit an event. A closed receiver is not an error — the run continues.
    pub fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.0 {
            let _ = tx.send(event);
        }
    }
}

impl From<mpsc::UnboundedSender<AgentEvent>> for EventSink {
    fn from(tx: mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self::new(tx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_none_sink_swallows_events_and_reports_inactive() {
        let sink = EventSink::none();
        assert!(!sink.is_active());
        sink.emit(AgentEvent::TurnStart { turn: 1 }); // must not panic
    }

    #[tokio::test]
    async fn a_channel_sink_delivers_in_order() {
        let (sink, mut rx) = EventSink::channel();
        assert!(sink.is_active());
        sink.emit(AgentEvent::TurnStart { turn: 1 });
        sink.emit(AgentEvent::ContentToken("hi".into()));
        sink.emit(AgentEvent::Finished {
            turns: 1,
            total_tokens: 7,
            reason: StopReason::Complete,
        });

        assert!(matches!(
            rx.recv().await.unwrap(),
            AgentEvent::TurnStart { turn: 1 }
        ));
        assert!(matches!(
            rx.recv().await.unwrap(),
            AgentEvent::ContentToken(t) if t == "hi"
        ));
        assert!(matches!(
            rx.recv().await.unwrap(),
            AgentEvent::Finished {
                reason: StopReason::Complete,
                total_tokens: 7,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn a_dropped_receiver_does_not_fail_the_run() {
        let (sink, rx) = EventSink::channel();
        drop(rx);
        sink.emit(AgentEvent::TurnStart { turn: 1 });
        assert!(sink.is_active(), "still nominally active; sends just no-op");
    }
}
