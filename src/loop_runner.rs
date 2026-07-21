//! The shared agent loop.
//!
//! One implementation replacing three: `pr-review-core`'s `agentic_review`, and
//! vexar's `openai_backend.rs` loop *and* the near-duplicate in `executor.rs`
//! whose stop conditions had already drifted apart (the executor copy omits the
//! token check entirely).
//!
//! The two-phase shape is `pr-review-core`'s: a cheap model drives tool calls to
//! gather context, then a strong model answers with tools forbidden. Set
//! `explore == synthesize` and it degenerates to the ordinary single-model loop
//! vexar runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::backend::{Backend, ModelPolicy, RunOutcome, RunRequest, clip};
use crate::error::AgentError;
use crate::events::{AgentEvent, EventSink, StopReason};
use crate::provider::ChatClient;
use crate::tools::ToolRegistry;

/// Decides whether a tool call may run.
///
/// The loop asks for **every** tool call when an approver is set; the policy of
/// which tools actually need a human lives with the implementor. vexar gates
/// `edit`/`write`/`bash`; a headless caller can approve everything. Keeping the
/// policy here rather than in the loop means the loop never has to know which
/// tool names are dangerous in a given host.
#[async_trait::async_trait]
pub trait ToolApprover: Send + Sync {
    /// Whether this tool needs a human decision at all. The default asks about
    /// everything; a host that only gates destructive tools overrides this so
    /// the loop skips the prompt (and the `ToolApprovalRequired` event) for the
    /// rest.
    fn needs_approval(&self, _tool: &str) -> bool {
        true
    }

    /// Return `false` to reject. A rejection is reported to the model as a tool
    /// result, not an error — the agent can choose another route. `call_id` is
    /// the model's id for this call, so an approver can correlate a prompt with
    /// its answer.
    async fn approve(&self, tool: &str, call_id: &str, args: &Value) -> bool;
}

/// A backend that drives the loop itself against an OpenAI-compatible endpoint.
pub struct ChatBackend {
    chat: ChatClient,
    tools: Arc<ToolRegistry>,
    policy: ModelPolicy,
    interrupt: Arc<AtomicBool>,
    approver: Option<Arc<dyn ToolApprover>>,
}

impl ChatBackend {
    /// A backend that talks to a chat endpoint.
    ///
    /// Note there is no constructor that yields a backend with *no* transport.
    /// Both blocking review findings on this crate came from a single
    /// constructor serving two roles — a real client and a placeholder for a
    /// delegated backend — and validating eagerly for a case that legitimately
    /// has no `base_url` or `api_key`. A delegated backend is a different type
    /// implementing [`Backend`], not this one built with fields missing.
    #[must_use]
    pub fn new(chat: ChatClient, tools: Arc<ToolRegistry>, policy: ModelPolicy) -> Self {
        Self {
            chat,
            tools,
            policy,
            interrupt: Arc::new(AtomicBool::new(false)),
            approver: None,
        }
    }

    /// Gate tool execution behind an approver.
    ///
    /// Without one, every tool runs. That is the right default for a library
    /// caller, and the wrong one for an interactive host — vexar must set this
    /// or it loses the confirmation gate on `edit`/`write`/`bash`.
    #[must_use]
    pub fn with_approver(mut self, approver: Arc<dyn ToolApprover>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Share an interrupt flag so a caller can cancel a run in flight.
    #[must_use]
    pub fn with_interrupt(mut self, flag: Arc<AtomicBool>) -> Self {
        self.interrupt = flag;
        self
    }

    /// The flag this backend watches.
    #[must_use]
    pub fn interrupt_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.interrupt)
    }

    fn tool_defs(&self, scope: &[String]) -> (Vec<Value>, Vec<String>) {
        if scope.is_empty() {
            (self.tools.definitions(), Vec::new())
        } else {
            let refs: Vec<&str> = scope.iter().map(String::as_str).collect();
            self.tools.definitions_for(&refs)
        }
    }

    async fn chat_once(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        tool_choice: &str,
    ) -> Result<(Value, u32), AgentError> {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": self.policy.max_tokens,
            "temperature": self.policy.temperature,
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!(tool_choice);
        }

        let data = self.chat.post_chat(&body).await?;
        let tokens = data
            .pointer("/usage/total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let message = data
            .pointer("/choices/0/message")
            .cloned()
            .ok_or_else(|| AgentError::Decode("response had no choices[0].message".into()))?;
        Ok((message, tokens))
    }
}

/// Parse a model-supplied `arguments` string once, for both the approver and
/// dispatch. Mirrors `ToolRegistry::call_raw_args`, which this replaces at the
/// call site so the string is not parsed twice.
fn parse_tool_args(tool: &str, arguments: &str) -> Result<Value, crate::tools::ToolError> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(trimmed).map_err(|e| crate::tools::ToolError::InvalidArguments {
        tool: tool.to_string(),
        reason: e.to_string(),
    })
}

/// Append placeholder responses for tool calls that never ran.
///
/// OpenAI-compatible APIs reject a request whose assistant message carries
/// `tool_calls` without a matching `role: "tool"` for every id. `RunOutcome`
/// documents its transcript as usable for persistence or a follow-up turn, so
/// an early return that leaves calls unanswered hands back something that
/// cannot actually be resumed.
fn seal_unanswered(messages: &mut Vec<Value>, pending: &[Value], reason: &str) {
    for call in pending {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": call["id"].as_str().unwrap_or_default(),
            "content": reason,
        }));
    }
}

/// Cap the tool-result text carried in the conversation: keep the newest
/// results whole up to `budget_chars`, elide older ones.
///
/// Only `role: "tool"` messages are elided — assistant reasoning is never
/// compacted, because losing it changes what the model concludes.
pub fn trim_history(messages: &mut [Value], budget_chars: usize) {
    let mut used = 0usize;
    for m in messages.iter_mut().rev() {
        if m["role"].as_str() != Some("tool") {
            continue;
        }
        let len = m["content"].as_str().map_or(0, |c| c.chars().count());
        if used + len > budget_chars {
            m["content"] = json!("[earlier tool result elided to save context]");
        } else {
            used += len;
        }
    }
}

#[async_trait::async_trait]
impl Backend for ChatBackend {
    fn name(&self) -> &str {
        "chat"
    }

    async fn run(&self, request: RunRequest, sink: EventSink) -> Result<RunOutcome, AgentError> {
        let started = Instant::now();
        // Destructured rather than borrowed so the prior transcript can be moved
        // instead of cloned — it is the largest thing in the request.
        let RunRequest {
            system_prompt,
            user_prompt,
            messages: prior,
            tool_scope,
        } = request;

        let (tools, missing) = self.tool_defs(&tool_scope);
        for name in &missing {
            // A typo in a scope list used to silently yield a smaller toolbelt.
            sink.emit(AgentEvent::Warning {
                turn: 0,
                message: format!("tool `{name}` in scope is not registered"),
            });
            tracing::warn!("tool `{name}` in scope is not registered");
        }

        // On resume the prior transcript already carries a system message. A
        // caller that supplies a new one means to update its instructions, so
        // splice it rather than silently discarding it.
        let mut messages: Vec<Value> = if prior.is_empty() {
            vec![json!({"role": "system", "content": system_prompt})]
        } else {
            let mut m = prior;
            if !system_prompt.is_empty() {
                let replacement = json!({"role": "system", "content": system_prompt});
                if m.first().is_some_and(|f| f["role"] == "system") {
                    m[0] = replacement;
                } else {
                    m.insert(0, replacement);
                }
            }
            m
        };
        messages.push(json!({"role": "user", "content": user_prompt}));

        let mut total_tokens = 0u32;
        let mut turn = 0u32;
        let mut content: Option<String> = None;

        // Emitting Finished lives here, not at the call sites: two early returns
        // (mid-tool interrupt, tool-error abort) previously skipped it, so a UI
        // treating Finished as "run ended" would hang. Folding it into the
        // constructor makes that omission unrepresentable.
        let finish =
            |turns, total_tokens, stop_reason: StopReason, messages, content: Option<String>| {
                sink.emit(AgentEvent::Finished {
                    turns,
                    total_tokens,
                    reason: stop_reason.clone(),
                });
                RunOutcome {
                    content,
                    turns,
                    total_tokens,
                    duration_ms: started.elapsed().as_millis() as u64,
                    stop_reason,
                    messages,
                    model: self.policy.display_model(),
                }
            };

        // ---- phase 1: exploration on the cheap model ----
        let stop = loop {
            if self.interrupt.load(Ordering::Relaxed) {
                break StopReason::Interrupted;
            }
            if turn >= self.policy.max_turns {
                break StopReason::MaxTurns(self.policy.max_turns);
            }
            if let Some(limit) = self.policy.stop_after_tokens
                && total_tokens >= limit
            {
                // Threshold, not ceiling — see ModelPolicy::stop_after_tokens.
                break StopReason::MaxTokens(limit);
            }
            if let Some(secs) = self.policy.timeout_secs
                && started.elapsed() > Duration::from_secs(secs)
            {
                break StopReason::Timeout(secs);
            }

            turn += 1;
            sink.emit(AgentEvent::TurnStart { turn });
            trim_history(&mut messages, self.policy.max_history_chars);

            let tool_choice = if turn == 1 {
                self.policy.initial_tool_choice.as_str()
            } else {
                "auto"
            };
            let (message, tokens) = self
                .chat_once(&self.policy.explore, &messages, &tools, tool_choice)
                .await?;
            total_tokens += tokens;
            if tokens > 0 {
                sink.emit(AgentEvent::Usage {
                    prompt_tokens: None,
                    completion_tokens: None,
                    total_tokens: tokens,
                });
            }

            let calls = message["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let text = message["content"].as_str().map(ToString::to_string);
            sink.emit(AgentEvent::TurnResponse {
                turn,
                has_tool_calls: !calls.is_empty(),
                content_preview: text.as_deref().map(|c| clip(c, 100)),
            });
            if let Some(t) = &text
                && !t.is_empty()
            {
                sink.emit(AgentEvent::ContentToken(t.clone()));
                content = Some(t.clone());
            }
            messages.push(message);

            if calls.is_empty() {
                break StopReason::Complete;
            }

            let mut executed = 0u32;
            for (idx, call) in calls.iter().enumerate() {
                if self.interrupt.load(Ordering::Relaxed) {
                    seal_unanswered(
                        &mut messages,
                        &calls[idx..],
                        "[not executed: run interrupted]",
                    );
                    return Ok(finish(
                        turn,
                        total_tokens,
                        StopReason::Interrupted,
                        messages,
                        content,
                    ));
                }
                let id = call["id"].as_str().unwrap_or_default().to_string();
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let args = call["function"]["arguments"].as_str().unwrap_or("{}");

                sink.emit(AgentEvent::ToolStart {
                    turn,
                    tool: name.clone(),
                    call_id: id.clone(),
                });
                // Parsed once and shared: parsing separately for the approver
                // meant malformed JSON reached it as `Null`, so an approver that
                // inspects arguments to decide would see nothing while dispatch
                // rejected the same call for being invalid.
                let parsed = parse_tool_args(&name, args);

                let approved = match (&self.approver, &parsed) {
                    (Some(a), Ok(v)) if a.needs_approval(&name) => {
                        sink.emit(AgentEvent::ToolApprovalRequired {
                            turn,
                            tool: name.clone(),
                            call_id: id.clone(),
                            arguments: args.to_string(),
                        });
                        a.approve(&name, &id, v).await
                    }
                    // Approver present but this tool is auto-approved, or the
                    // arguments are unparseable (dispatch rejects them below):
                    // nothing meaningful to approve.
                    (Some(_), _) | (None, _) => true,
                };

                // Timed after the approval decision: an interactive approver
                // waits on a human, and folding that into duration_ms would
                // report think-time as tool latency.
                let t0 = Instant::now();
                let result = match (approved, parsed) {
                    (true, Ok(v)) => self.tools.call(&name, v).await,
                    (true, Err(e)) => Err(e),
                    (false, _) => Ok(crate::tools::ToolOutput::error(
                        "Tool execution was rejected by the user.",
                    )),
                };
                let duration_ms = t0.elapsed().as_millis() as u64;

                let output = match result {
                    Ok(o) => o,
                    Err(e) if self.policy.continue_on_tool_error => e.to_tool_output(),
                    Err(e) => {
                        sink.emit(AgentEvent::ToolComplete {
                            turn,
                            tool: name,
                            call_id: id.clone(),
                            ok: false,
                            duration_ms,
                            output_preview: clip(&e.to_string(), 100),
                        });
                        // This call *did* run — persist its real error, so the
                        // resumed transcript says what actually went wrong. Only
                        // the calls after it never started.
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": e.to_string(),
                        }));
                        seal_unanswered(
                            &mut messages,
                            &calls[idx + 1..],
                            "[not executed: run aborted after a tool failure]",
                        );
                        return Ok(finish(
                            turn,
                            total_tokens,
                            StopReason::Error(e.to_string()),
                            messages,
                            content,
                        ));
                    }
                };

                sink.emit(AgentEvent::ToolComplete {
                    turn,
                    tool: name,
                    call_id: id.clone(),
                    ok: !output.is_error,
                    duration_ms,
                    output_preview: clip(&output.content, 100),
                });
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": output.content,
                }));
                executed += 1;
            }
            sink.emit(AgentEvent::TurnComplete {
                turn,
                tools_executed: executed,
            });
        };

        // A tiered run synthesizes once exploration is *done* — whether it
        // finished naturally or hit the turn cap. Both mean "gather no more,
        // now produce the answer from what you have"; a batch consumer still
        // wants a result at the cap. A hard stop (interrupt / token / time
        // budget) does skip synthesis: those mean "stop spending", and an
        // untiered run has no separate synthesis model to call.
        let exploration_done = matches!(stop, StopReason::Complete | StopReason::MaxTurns(_));
        if !exploration_done || !self.policy.final_synthesis {
            return Ok(finish(turn, total_tokens, stop, messages, content));
        }

        // Re-check immediately before committing to the expensive call: the
        // flag can flip while the last exploration response is in flight, and
        // "stop spending" has to mean the strong model too.
        if self.interrupt.load(Ordering::Relaxed) {
            return Ok(finish(
                turn,
                total_tokens,
                StopReason::Interrupted,
                messages,
                content,
            ));
        }

        // ---- phase 2: synthesis on the strong model, tools forbidden ----
        messages.push(json!({
            "role": "user",
            "content": "Stop investigating now. Using only what you've already gathered, \
                        produce the final answer — no prose preamble, no tool calls.",
        }));
        trim_history(&mut messages, self.policy.max_history_chars);

        // Synthesis is a model round-trip like any other, and typically the
        // slowest one. Without these a turn-based progress indicator sits
        // frozen through the most expensive part of the run.
        turn += 1;
        sink.emit(AgentEvent::TurnStart { turn });

        let (message, tokens) = self
            .chat_once(&self.policy.synthesize, &messages, &tools, "none")
            .await?;
        total_tokens += tokens;
        if tokens > 0 {
            // Mirrors phase 1: a consumer tallying Usage events would otherwise
            // under-report tiered runs even though Finished.total_tokens is right.
            sink.emit(AgentEvent::Usage {
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: tokens,
            });
        }
        sink.emit(AgentEvent::TurnResponse {
            turn,
            has_tool_calls: false,
            content_preview: message["content"].as_str().map(|c| clip(c, 100)),
        });
        if let Some(t) = message["content"].as_str()
            && !t.is_empty()
        {
            sink.emit(AgentEvent::ContentToken(t.to_string()));
            content = Some(t.to_string());
        }
        messages.push(message);
        sink.emit(AgentEvent::TurnComplete {
            turn,
            tools_executed: 0,
        });

        // Preserve why exploration stopped. A run that synthesized after the
        // turn cap still reports MaxTurns, not Complete — the answer exists, but
        // it was produced from capped context, and a caller should be able to
        // tell.
        Ok(finish(turn, total_tokens, stop, messages, content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::run_structured;
    use crate::provider::{ProviderConfig, RetryPolicy};
    use crate::tools::{Tool, ToolError, ToolOutput};
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use wiremock::matchers::{method, path as pathm};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    #[derive(Clone)]
    struct Seq(Arc<SeqInner>);
    struct SeqInner {
        queued: Mutex<VecDeque<Value>>,
        seen: Mutex<Vec<Value>>,
        /// Flipped as the response is served, simulating an interrupt that
        /// lands while the last exploration reply is in flight.
        trip_on_serve: Mutex<Option<Arc<AtomicBool>>>,
    }

    impl Seq {
        fn new(r: Vec<Value>) -> Self {
            Seq(Arc::new(SeqInner {
                queued: Mutex::new(r.into()),
                seen: Mutex::new(Vec::new()),
                trip_on_serve: Mutex::new(None),
            }))
        }

        fn trip_on_serve(self, flag: Arc<AtomicBool>) -> Self {
            *self.0.trip_on_serve.lock().unwrap() = Some(flag);
            self
        }
        fn requests(&self) -> Vec<Value> {
            self.0.seen.lock().unwrap().clone()
        }
        fn calls(&self) -> usize {
            self.0.seen.lock().unwrap().len()
        }
    }

    impl Respond for Seq {
        fn respond(&self, req: &Request) -> ResponseTemplate {
            self.0
                .seen
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&req.body).unwrap());
            if let Some(f) = self.0.trip_on_serve.lock().unwrap().as_ref() {
                f.store(true, Ordering::Relaxed);
            }
            match self.0.queued.lock().unwrap().pop_front() {
                Some(b) => ResponseTemplate::new(200).set_body_json(b),
                None => ResponseTemplate::new(500).set_body_string("no queued response"),
            }
        }
    }

    fn tool_turn(calls: &[(&str, &str, &str)], tokens: u32) -> Value {
        let tcs: Vec<Value> = calls
            .iter()
            .map(|(id, name, args)| {
                json!({"id": id, "type": "function",
                       "function": {"name": name, "arguments": args}})
            })
            .collect();
        json!({
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": tcs}}],
            "usage": {"total_tokens": tokens}
        })
    }

    fn text_turn(content: &str, tokens: u32) -> Value {
        json!({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"total_tokens": tokens}
        })
    }

    async fn server(r: Vec<Value>) -> (MockServer, Seq) {
        let s = MockServer::start().await;
        let seq = Seq::new(r);
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .respond_with(seq.clone())
            .mount(&s)
            .await;
        (s, seq)
    }

    #[derive(Deserialize, JsonSchema)]
    struct EchoArgs {
        text: String,
    }
    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        type Args = EchoArgs;
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echo."
        }
        async fn call(&self, a: Self::Args) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::ok(format!("echo: {}", a.text)))
        }
    }

    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}
    struct Boom;
    #[async_trait]
    impl Tool for Boom {
        type Args = NoArgs;
        fn name(&self) -> &'static str {
            "boom"
        }
        fn description(&self) -> &'static str {
            "Fails."
        }
        async fn call(&self, _: Self::Args) -> Result<ToolOutput, ToolError> {
            Err(ToolError::Execution {
                tool: "boom".into(),
                reason: "exploded".into(),
            })
        }
    }

    fn backend(url: &str, policy: ModelPolicy) -> ChatBackend {
        let chat = ChatClient::new(ProviderConfig {
            base_url: url.to_string(),
            api_key: "k".into(),
            retry: RetryPolicy::none(),
            ..Default::default()
        })
        .unwrap();
        let tools = ToolRegistry::new().with(Echo).with(Boom);
        ChatBackend::new(chat, Arc::new(tools), policy)
    }

    fn req() -> RunRequest {
        RunRequest {
            system_prompt: "SYS".into(),
            user_prompt: "USER".into(),
            ..Default::default()
        }
    }

    fn tiered() -> ModelPolicy {
        ModelPolicy::tiered("cheap/m", "main/m")
    }

    // ---- the two-phase contract, carried over from pr-review-core ----------

    #[tokio::test]
    async fn explore_runs_cheap_with_tools_then_synthesis_runs_strong_without() {
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"hi"}"#)], 10),
            text_turn("gathered enough", 10),
            text_turn(r#"{"summary":"done"}"#, 30),
        ])
        .await;
        let out = backend(&srv.uri(), tiered())
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert_eq!(seq.calls(), 3);
        let r = seq.requests();
        assert_eq!(r[0]["model"], "cheap/m");
        assert_eq!(r[0]["tool_choice"], "auto");
        assert_eq!(r[2]["model"], "main/m");
        assert_eq!(r[2]["tool_choice"], "none");
        assert_eq!(out.total_tokens, 50);
        assert_eq!(out.model, "main/m (explore: cheap/m)");
        assert_eq!(out.stop_reason, StopReason::Complete);
    }

    #[tokio::test]
    async fn a_single_model_policy_skips_the_synthesis_call_entirely() {
        // vexar's shape: no second tier, so no extra call and no extra spend.
        let (srv, seq) = server(vec![text_turn("done", 5)]).await;
        let out = backend(&srv.uri(), ModelPolicy::single("solo/m"))
            .run(req(), EventSink::none())
            .await
            .unwrap();
        assert_eq!(seq.calls(), 1);
        assert_eq!(out.content.as_deref(), Some("done"));
        assert_eq!(out.model, "solo/m");
    }

    // ---- termination -------------------------------------------------------

    #[tokio::test]
    async fn hitting_the_turn_cap_still_synthesises() {
        // The turn cap means "gather no more", not "abandon the run". A tiered
        // batch consumer (e.g. a PR reviewer) still wants the final answer from
        // whatever was gathered, so synthesis runs — but the outcome reports
        // MaxTurns, not Complete, so the cap is visible.
        let responses = (0..6)
            .map(|i| tool_turn(&[(&format!("c{i}"), "echo", r#"{"text":"x"}"#)], 1))
            .chain(std::iter::once(text_turn(r#"{"ok":true}"#, 1)))
            .collect();
        let (srv, seq) = server(responses).await;
        let mut p = tiered();
        p.max_turns = 2;
        let out = backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert_eq!(seq.calls(), 3, "2 explore turns capped, then 1 synthesis");
        assert_eq!(
            out.stop_reason,
            StopReason::MaxTurns(2),
            "the cap is preserved even though we synthesized"
        );
    }

    #[tokio::test]
    async fn a_hard_stop_still_skips_synthesis() {
        // The distinction the change turns on: a token/time/interrupt budget is
        // "stop spending", so synthesis does NOT run, unlike the turn cap.
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 100),
            tool_turn(&[("c2", "echo", r#"{"text":"y"}"#)], 100),
        ])
        .await;
        let mut p = tiered();
        p.stop_after_tokens = Some(50);
        let out = backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert_eq!(seq.calls(), 1, "no synthesis after a token-budget stop");
        assert_eq!(out.stop_reason, StopReason::MaxTokens(50));
    }

    #[tokio::test]
    async fn the_token_threshold_stops_between_turns_and_may_overshoot() {
        // Decided deliberately: a call's cost is unknowable beforehand, so the
        // budget is a stop threshold and overshoot is bounded by one call's
        // max_tokens. Matches what step 1 pinned in vexar.
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 100),
            tool_turn(&[("c2", "echo", r#"{"text":"y"}"#)], 100),
        ])
        .await;
        let mut p = tiered();
        p.stop_after_tokens = Some(50);
        let out = backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert_eq!(seq.calls(), 1);
        assert_eq!(out.stop_reason, StopReason::MaxTokens(50));
        assert_eq!(out.total_tokens, 100, "overshoot is expected, not a bug");
    }

    #[tokio::test]
    async fn a_preset_interrupt_stops_before_any_call() {
        let (srv, seq) = server(vec![text_turn("never", 1)]).await;
        let b = backend(&srv.uri(), tiered());
        b.interrupt_flag().store(true, Ordering::Relaxed);
        let out = b.run(req(), EventSink::none()).await.unwrap();
        assert_eq!(seq.calls(), 0);
        assert_eq!(out.stop_reason, StopReason::Interrupted);
    }

    #[tokio::test]
    async fn the_first_turn_can_force_a_tool_call() {
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let mut p = ModelPolicy::single("m");
        p.initial_tool_choice = "required".into();
        backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();

        let r = seq.requests();
        assert_eq!(r[0]["tool_choice"], "required", "first turn is forced");
        assert_eq!(r[1]["tool_choice"], "auto", "later turns are not");
    }

    // ---- tools -------------------------------------------------------------

    #[tokio::test]
    async fn tool_results_are_threaded_back_into_the_next_turn() {
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"hello"}"#)], 1),
            text_turn("ok", 1),
            text_turn("{}", 1),
        ])
        .await;
        backend(&srv.uri(), tiered())
            .run(req(), EventSink::none())
            .await
            .unwrap();

        let r = seq.requests();
        let tool_msg = r[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool result fed back");
        assert_eq!(tool_msg["tool_call_id"], "c1");
        assert_eq!(tool_msg["content"], "echo: hello");
    }

    #[tokio::test]
    async fn malformed_tool_arguments_are_reported_to_the_model_not_silently_defaulted() {
        // The defect step 1 pinned in both codebases, now closed: the model is
        // told its call was invalid instead of the tool running with defaults.
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", "{not json")], 1),
            text_turn("ok", 1),
            text_turn("{}", 1),
        ])
        .await;
        backend(&srv.uri(), tiered())
            .run(req(), EventSink::none())
            .await
            .unwrap();

        let r = seq.requests();
        let tool_msg = r[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap();
        let content = tool_msg["content"].as_str().unwrap();
        assert!(content.contains("invalid arguments"), "got: {content}");
        assert!(content.contains("echo"), "names the tool: {content}");
    }

    #[tokio::test]
    async fn a_failing_tool_aborts_when_continue_on_error_is_false() {
        let (srv, _s) = server(vec![tool_turn(&[("c1", "boom", "{}")], 1)]).await;
        let mut p = tiered();
        p.continue_on_tool_error = false;
        let out = backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();
        assert!(matches!(out.stop_reason, StopReason::Error(m) if m.contains("exploded")));
    }

    #[tokio::test]
    async fn an_unregistered_tool_in_scope_warns_rather_than_shrinking_silently() {
        let (srv, seq) = server(vec![text_turn("ok", 1)]).await;
        let (sink, mut rx) = EventSink::channel();
        let mut r = req();
        r.tool_scope = vec!["echo".into(), "ecoh".into()];
        backend(&srv.uri(), ModelPolicy::single("m"))
            .run(r, sink)
            .await
            .unwrap();

        let warned = std::iter::from_fn(|| rx.try_recv().ok())
            .any(|e| matches!(e, AgentEvent::Warning { message, .. } if message.contains("ecoh")));
        assert!(warned, "a typo in the scope list must be visible");
        assert_eq!(seq.requests()[0]["tools"].as_array().unwrap().len(), 1);
    }

    // ---- approval gate -----------------------------------------------------

    struct DenyAll;
    #[async_trait]
    impl ToolApprover for DenyAll {
        async fn approve(&self, _: &str, _: &str, _: &Value) -> bool {
            false
        }
    }

    struct RecordingApprover(Arc<Mutex<Vec<String>>>);
    #[async_trait]
    impl ToolApprover for RecordingApprover {
        async fn approve(&self, tool: &str, _call_id: &str, args: &Value) -> bool {
            self.0
                .lock()
                .unwrap()
                .push(format!("{tool}:{}", args["text"].as_str().unwrap_or("")));
            true
        }
    }

    #[tokio::test]
    async fn a_rejected_tool_is_reported_to_the_model_and_the_loop_continues() {
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"danger"}"#)], 1),
            text_turn("understood", 1),
        ])
        .await;
        let b = backend(&srv.uri(), ModelPolicy::single("m")).with_approver(Arc::new(DenyAll));
        let out = b.run(req(), EventSink::none()).await.unwrap();

        assert_eq!(out.stop_reason, StopReason::Complete);
        let tool_msg = seq.requests()[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap()
            .clone();
        assert!(
            tool_msg["content"].as_str().unwrap().contains("rejected"),
            "got {tool_msg}"
        );
    }

    struct IdRecorder(Arc<Mutex<Vec<String>>>);
    #[async_trait]
    impl ToolApprover for IdRecorder {
        async fn approve(&self, _tool: &str, call_id: &str, _args: &Value) -> bool {
            self.0.lock().unwrap().push(call_id.to_string());
            true
        }
    }

    #[tokio::test]
    async fn the_approver_receives_the_real_call_id() {
        let ids = Arc::new(Mutex::new(Vec::new()));
        let (srv, _s) = server(vec![
            tool_turn(&[("call_abc", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        backend(&srv.uri(), ModelPolicy::single("m"))
            .with_approver(Arc::new(IdRecorder(Arc::clone(&ids))))
            .run(req(), EventSink::none())
            .await
            .unwrap();
        assert_eq!(*ids.lock().unwrap(), vec!["call_abc"]);
    }

    struct GateEcho;
    #[async_trait]
    impl ToolApprover for GateEcho {
        fn needs_approval(&self, tool: &str) -> bool {
            tool == "echo"
        }
        async fn approve(&self, _: &str, _: &str, _: &Value) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn approval_required_is_emitted_only_for_gated_tools() {
        let (srv, _s) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let (sink, mut rx) = EventSink::channel();
        backend(&srv.uri(), ModelPolicy::single("m"))
            .with_approver(Arc::new(GateEcho))
            .run(req(), sink)
            .await
            .unwrap();

        let approvals: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|e| match e {
                AgentEvent::ToolApprovalRequired { tool, call_id, .. } => Some((tool, call_id)),
                _ => None,
            })
            .collect();
        assert_eq!(approvals, vec![("echo".to_string(), "c1".to_string())]);
    }

    #[tokio::test]
    async fn a_tool_not_needing_approval_emits_no_approval_event() {
        struct GateNothing;
        #[async_trait]
        impl ToolApprover for GateNothing {
            fn needs_approval(&self, _: &str) -> bool {
                false
            }
            async fn approve(&self, _: &str, _: &str, _: &Value) -> bool {
                false // would reject if consulted
            }
        }
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let (sink, mut rx) = EventSink::channel();
        backend(&srv.uri(), ModelPolicy::single("m"))
            .with_approver(Arc::new(GateNothing))
            .run(req(), sink)
            .await
            .unwrap();

        // Ran (not rejected) and no approval event fired.
        assert_eq!(
            seq.requests()[1]["messages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["role"] == "tool")
                .unwrap()["content"],
            "echo: x"
        );
        assert!(
            !std::iter::from_fn(|| rx.try_recv().ok())
                .any(|e| matches!(e, AgentEvent::ToolApprovalRequired { .. }))
        );
    }

    #[tokio::test]
    async fn the_approver_sees_the_tool_name_and_parsed_arguments() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (srv, _s) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"hi"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let b = backend(&srv.uri(), ModelPolicy::single("m"))
            .with_approver(Arc::new(RecordingApprover(Arc::clone(&seen))));
        b.run(req(), EventSink::none()).await.unwrap();
        assert_eq!(*seen.lock().unwrap(), vec!["echo:hi"]);
    }

    #[tokio::test]
    async fn without_an_approver_every_tool_runs() {
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        backend(&srv.uri(), ModelPolicy::single("m"))
            .run(req(), EventSink::none())
            .await
            .unwrap();
        let tool_msg = seq.requests()[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap()
            .clone();
        assert_eq!(tool_msg["content"], "echo: x");
    }

    struct SlowApprover;
    #[async_trait]
    impl ToolApprover for SlowApprover {
        async fn approve(&self, _: &str, _: &str, _: &Value) -> bool {
            tokio::time::sleep(Duration::from_millis(120)).await;
            true
        }
    }

    #[tokio::test]
    async fn approval_wait_is_not_counted_as_tool_latency() {
        // An interactive approver waits on a human. Folding that into
        // duration_ms would report think-time as tool latency.
        let (srv, _s) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let (sink, mut rx) = EventSink::channel();
        backend(&srv.uri(), ModelPolicy::single("m"))
            .with_approver(Arc::new(SlowApprover))
            .run(req(), sink)
            .await
            .unwrap();

        let d = std::iter::from_fn(|| rx.try_recv().ok())
            .find_map(|e| match e {
                AgentEvent::ToolComplete { duration_ms, .. } => Some(duration_ms),
                _ => None,
            })
            .expect("a ToolComplete was emitted");
        assert!(d < 100, "duration_ms {d} includes the 120ms approval wait");
    }

    #[tokio::test]
    async fn malformed_arguments_skip_approval_and_reach_the_model_as_an_error() {
        // Arguments are parsed once. Previously the approver got `Null` for
        // unparseable JSON while dispatch rejected the same call, so an
        // approver inspecting arguments saw nothing.
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (srv, seq) = server(vec![
            tool_turn(&[("c1", "echo", "{not json")], 1),
            text_turn("understood", 1),
        ])
        .await;
        backend(&srv.uri(), ModelPolicy::single("m"))
            .with_approver(Arc::new(RecordingApprover(Arc::clone(&seen))))
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert!(
            seen.lock().unwrap().is_empty(),
            "nothing meaningful to approve; dispatch rejects it"
        );
        let tool_msg = seq.requests()[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap()
            .clone();
        assert!(
            tool_msg["content"]
                .as_str()
                .unwrap()
                .contains("invalid arguments"),
            "got {tool_msg}"
        );
    }

    // ---- streaming is optional --------------------------------------------

    #[tokio::test]
    async fn the_same_run_streams_events_or_stays_silent() {
        let (srv, _s) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let (sink, mut rx) = EventSink::channel();
        backend(&srv.uri(), ModelPolicy::single("m"))
            .run(req(), sink)
            .await
            .unwrap();

        let kinds: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|e| match e {
                AgentEvent::TurnStart { .. } => "TurnStart",
                AgentEvent::TurnResponse { .. } => "TurnResponse",
                AgentEvent::ContentToken(_) => "ContentToken",
                AgentEvent::ToolApprovalRequired { .. } => "ToolApprovalRequired",
                AgentEvent::ToolStart { .. } => "ToolStart",
                AgentEvent::ToolComplete { .. } => "ToolComplete",
                AgentEvent::TurnComplete { .. } => "TurnComplete",
                AgentEvent::Usage { .. } => "Usage",
                AgentEvent::Finished { .. } => "Finished",
                AgentEvent::Warning { .. } => "Warning",
            })
            .collect();

        assert_eq!(
            kinds,
            vec![
                "TurnStart",
                "Usage",
                "TurnResponse",
                "ToolStart",
                "ToolComplete",
                "TurnComplete",
                "TurnStart",
                "Usage",
                "TurnResponse",
                "ContentToken",
                "Finished",
            ]
        );
    }

    // ---- every exit path signals the end -----------------------------------

    fn last_finished(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
    ) -> Option<StopReason> {
        std::iter::from_fn(|| rx.try_recv().ok()).fold(None, |acc, e| match e {
            AgentEvent::Finished { reason, .. } => Some(reason),
            _ => acc,
        })
    }

    /// A tool that trips the shared interrupt flag when it runs, so the
    /// mid-tool-execution branch is reached deterministically rather than by
    /// racing a timer.
    struct TripInterrupt(Arc<AtomicBool>);

    #[async_trait]
    impl Tool for TripInterrupt {
        type Args = NoArgs;
        fn name(&self) -> &'static str {
            "trip"
        }
        fn description(&self) -> &'static str {
            "Trips the interrupt."
        }
        async fn call(&self, _: Self::Args) -> Result<ToolOutput, ToolError> {
            self.0.store(true, Ordering::Relaxed);
            Ok(ToolOutput::ok("tripped"))
        }
    }

    #[tokio::test]
    async fn an_interrupt_during_tool_execution_still_emits_finished() {
        // Regression: this early return skipped Finished, so a UI treating it
        // as "run ended" would hang forever. Two calls in one turn: the first
        // trips the flag, the second hits the mid-tool interrupt check.
        let (srv, _s) = server(vec![tool_turn(
            &[("c1", "trip", "{}"), ("c2", "trip", "{}")],
            1,
        )])
        .await;
        let flag = Arc::new(AtomicBool::new(false));
        let chat = ChatClient::new(ProviderConfig {
            base_url: srv.uri(),
            api_key: "k".into(),
            retry: RetryPolicy::none(),
            ..Default::default()
        })
        .unwrap();
        let tools = ToolRegistry::new().with(TripInterrupt(Arc::clone(&flag)));
        let b = ChatBackend::new(chat, Arc::new(tools), ModelPolicy::single("m"))
            .with_interrupt(Arc::clone(&flag));

        let (sink, mut rx) = EventSink::channel();
        let out = b.run(req(), sink).await.unwrap();

        assert_eq!(out.stop_reason, StopReason::Interrupted);
        assert_eq!(
            last_finished(&mut rx),
            Some(StopReason::Interrupted),
            "the mid-tool early return must still signal the end"
        );
    }

    #[tokio::test]
    async fn a_tool_error_abort_still_emits_finished() {
        let (srv, _s) = server(vec![tool_turn(&[("c1", "boom", "{}")], 1)]).await;
        let (sink, mut rx) = EventSink::channel();
        let mut p = ModelPolicy::single("m");
        p.continue_on_tool_error = false;
        let out = backend(&srv.uri(), p).run(req(), sink).await.unwrap();

        assert!(matches!(out.stop_reason, StopReason::Error(_)));
        assert_eq!(last_finished(&mut rx), Some(out.stop_reason.clone()));
    }

    #[tokio::test]
    async fn every_stop_reason_reaches_the_sink_exactly_once() {
        for (responses, policy) in [
            (vec![text_turn("done", 1)], ModelPolicy::single("m")),
            (
                vec![tool_turn(&[("c1", "boom", "{}")], 1)],
                ModelPolicy {
                    continue_on_tool_error: false,
                    ..ModelPolicy::single("m")
                },
            ),
        ] {
            let (srv, _s) = server(responses).await;
            let (sink, mut rx) = EventSink::channel();
            backend(&srv.uri(), policy).run(req(), sink).await.unwrap();
            let n = std::iter::from_fn(|| rx.try_recv().ok())
                .filter(|e| matches!(e, AgentEvent::Finished { .. }))
                .count();
            assert_eq!(n, 1, "exactly one Finished per run");
        }
    }

    #[tokio::test]
    async fn an_interrupt_arriving_before_synthesis_skips_the_strong_model_call() {
        // The window this guards is narrow and easy to test vacuously: if the
        // flag is set early enough, the loop-top check catches it and the
        // pre-synthesis check never runs. So the flag is tripped by the server
        // *as it serves* the final exploration reply — exploration ends
        // normally (Complete), and only then is the interrupt visible.
        let srv = MockServer::start().await;
        let flag = Arc::new(AtomicBool::new(false));
        let seq = Seq::new(vec![
            text_turn("done exploring", 1),
            text_turn("SYNTHESIS SHOULD NOT HAPPEN", 1),
        ])
        .trip_on_serve(Arc::clone(&flag));
        Mock::given(method("POST"))
            .and(pathm("/chat/completions"))
            .respond_with(seq.clone())
            .mount(&srv)
            .await;

        let chat = ChatClient::new(ProviderConfig {
            base_url: srv.uri(),
            api_key: "k".into(),
            retry: RetryPolicy::none(),
            ..Default::default()
        })
        .unwrap();
        let b = ChatBackend::new(chat, Arc::new(ToolRegistry::new()), tiered())
            .with_interrupt(Arc::clone(&flag));

        let out = b.run(req(), EventSink::none()).await.unwrap();

        assert_eq!(out.stop_reason, StopReason::Interrupted);
        assert_eq!(seq.calls(), 1, "the strong model must not be paid for");
        assert_ne!(out.content.as_deref(), Some("SYNTHESIS SHOULD NOT HAPPEN"));
    }

    #[tokio::test]
    async fn usage_events_account_for_every_call_including_synthesis() {
        let (srv, _s) = server(vec![
            text_turn("done exploring", 10),
            text_turn("{\"ok\":true}", 40),
        ])
        .await;
        let (sink, mut rx) = EventSink::channel();
        let out = backend(&srv.uri(), tiered())
            .run(req(), sink)
            .await
            .unwrap();

        let tallied: u32 = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|e| match e {
                AgentEvent::Usage { total_tokens, .. } => Some(total_tokens),
                _ => None,
            })
            .sum();
        assert_eq!(out.total_tokens, 50);
        assert_eq!(
            tallied, out.total_tokens,
            "a consumer tallying Usage must match Finished.total_tokens"
        );
    }

    // ---- the returned transcript must be resumable -------------------------

    /// Every `tool_calls` id in the transcript has a matching `role: "tool"`.
    /// OpenAI-compatible APIs reject a request without this, so a transcript
    /// that fails it cannot be resumed or persisted usefully.
    fn assert_no_dangling_tool_calls(messages: &[Value]) {
        let answered: std::collections::HashSet<&str> = messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();
        for m in messages.iter().filter(|m| m["role"] == "assistant") {
            for call in m["tool_calls"].as_array().unwrap_or(&vec![]) {
                let id = call["id"].as_str().unwrap_or_default();
                assert!(
                    answered.contains(id),
                    "tool_call `{id}` has no matching tool response; transcript is unresumable"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_tool_error_abort_returns_a_resumable_transcript() {
        // Two calls: the first fails and aborts, so neither it nor the second
        // ever gets a response unless the loop seals them.
        let (srv, _s) = server(vec![tool_turn(
            &[("c1", "boom", "{}"), ("c2", "echo", r#"{"text":"x"}"#)],
            1,
        )])
        .await;
        let mut p = ModelPolicy::single("m");
        p.continue_on_tool_error = false;
        let out = backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert!(matches!(out.stop_reason, StopReason::Error(_)));
        assert_no_dangling_tool_calls(&out.messages);
    }

    #[tokio::test]
    async fn a_mid_tool_interrupt_returns_a_resumable_transcript() {
        let (srv, _s) = server(vec![tool_turn(
            &[
                ("c1", "trip", "{}"),
                ("c2", "trip", "{}"),
                ("c3", "trip", "{}"),
            ],
            1,
        )])
        .await;
        let flag = Arc::new(AtomicBool::new(false));
        let chat = ChatClient::new(ProviderConfig {
            base_url: srv.uri(),
            api_key: "k".into(),
            retry: RetryPolicy::none(),
            ..Default::default()
        })
        .unwrap();
        let tools = ToolRegistry::new().with(TripInterrupt(Arc::clone(&flag)));
        let out = ChatBackend::new(chat, Arc::new(tools), ModelPolicy::single("m"))
            .with_interrupt(Arc::clone(&flag))
            .run(req(), EventSink::none())
            .await
            .unwrap();

        assert_eq!(out.stop_reason, StopReason::Interrupted);
        assert_no_dangling_tool_calls(&out.messages);
    }

    #[tokio::test]
    async fn a_normal_run_also_leaves_no_dangling_tool_calls() {
        let (srv, _s) = server(vec![
            tool_turn(&[("c1", "echo", r#"{"text":"x"}"#)], 1),
            text_turn("done", 1),
        ])
        .await;
        let out = backend(&srv.uri(), ModelPolicy::single("m"))
            .run(req(), EventSink::none())
            .await
            .unwrap();
        assert_no_dangling_tool_calls(&out.messages);
    }

    #[tokio::test]
    async fn a_failed_tool_records_its_real_error_not_a_placeholder() {
        // The failing call did run; only the ones after it never started. A
        // resumed transcript should say what actually went wrong.
        let (srv, _s) = server(vec![tool_turn(
            &[("c1", "boom", "{}"), ("c2", "echo", r#"{"text":"x"}"#)],
            1,
        )])
        .await;
        let mut p = ModelPolicy::single("m");
        p.continue_on_tool_error = false;
        let out = backend(&srv.uri(), p)
            .run(req(), EventSink::none())
            .await
            .unwrap();

        let resp = |id: &str| -> String {
            out.messages
                .iter()
                .find(|m| m["role"] == "tool" && m["tool_call_id"] == id)
                .unwrap()["content"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert!(
            resp("c1").contains("exploded"),
            "real error lost: {}",
            resp("c1")
        );
        assert!(resp("c2").contains("not executed"), "c2 never ran");
        assert_no_dangling_tool_calls(&out.messages);
    }

    #[tokio::test]
    async fn the_synthesis_call_reports_turn_progress() {
        // Synthesis is usually the slowest call; without these a turn-based
        // progress indicator sits frozen through it.
        let (srv, _s) = server(vec![text_turn("explored", 1), text_turn("final", 1)]).await;
        let (sink, mut rx) = EventSink::channel();
        let out = backend(&srv.uri(), tiered())
            .run(req(), sink)
            .await
            .unwrap();

        let starts: Vec<u32> = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|e| match e {
                AgentEvent::TurnStart { turn } => Some(turn),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![1, 2], "exploration turn 1, synthesis turn 2");
        assert_eq!(out.turns, 2, "turns counts every model round-trip");
    }

    // ---- resuming ----------------------------------------------------------

    #[tokio::test]
    async fn a_resumed_run_can_update_its_system_prompt() {
        let (srv, seq) = server(vec![text_turn("ok", 1)]).await;
        let mut r = req();
        r.system_prompt = "NEW INSTRUCTIONS".into();
        r.messages = vec![
            json!({"role": "system", "content": "OLD INSTRUCTIONS"}),
            json!({"role": "user", "content": "earlier"}),
        ];
        backend(&srv.uri(), ModelPolicy::single("m"))
            .run(r, EventSink::none())
            .await
            .unwrap();

        let msgs = seq.requests()[0]["messages"].as_array().unwrap().clone();
        assert_eq!(msgs[0]["content"], "NEW INSTRUCTIONS", "silently dropped");
        assert_eq!(msgs[1]["content"], "earlier", "prior turns kept");
        assert_eq!(
            msgs.iter().filter(|m| m["role"] == "system").count(),
            1,
            "spliced, not duplicated"
        );
    }

    #[tokio::test]
    async fn resuming_without_a_system_prompt_keeps_the_prior_one() {
        let (srv, seq) = server(vec![text_turn("ok", 1)]).await;
        let mut r = req();
        r.system_prompt = String::new();
        r.messages = vec![json!({"role": "system", "content": "KEEP ME"})];
        backend(&srv.uri(), ModelPolicy::single("m"))
            .run(r, EventSink::none())
            .await
            .unwrap();
        assert_eq!(seq.requests()[0]["messages"][0]["content"], "KEEP ME");
    }

    // ---- structured output -------------------------------------------------

    #[tokio::test]
    async fn run_structured_over_the_real_loop_is_what_review_backend_was() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Review {
            summary: String,
        }
        let (srv, _s) = server(vec![
            text_turn("done exploring", 1),
            text_turn("Here: {\"summary\":\"looks good\"}", 1),
        ])
        .await;
        let b = backend(&srv.uri(), tiered());
        let (review, outcome): (Review, _) =
            run_structured(&b, req(), EventSink::none()).await.unwrap();
        assert_eq!(review.summary, "looks good");
        assert_eq!(outcome.stop_reason, StopReason::Complete);
    }

    // ---- history compaction ------------------------------------------------

    #[test]
    fn trim_history_elides_only_older_tool_results() {
        let mut msgs = vec![
            json!({"role":"system","content":"S"}),
            json!({"role":"assistant","content":"reasoning worth keeping"}),
            json!({"role":"tool","tool_call_id":"1","content":"old-and-long"}),
            json!({"role":"tool","tool_call_id":"2","content":"newest"}),
        ];
        trim_history(&mut msgs, 8);
        assert_eq!(msgs[0]["content"], "S");
        assert_eq!(msgs[1]["content"], "reasoning worth keeping");
        assert_eq!(
            msgs[2]["content"],
            "[earlier tool result elided to save context]"
        );
        assert_eq!(msgs[3]["content"], "newest");
    }
}
