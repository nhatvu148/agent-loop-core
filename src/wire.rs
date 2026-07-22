//! Wire formats: how a turn is put on the network and read back off it.
//!
//! The loop speaks one canonical dialect internally — OpenAI Chat Completions
//! message shape — because that is what [`crate::RunOutcome::messages`] hands
//! back for persistence and resume, and what every caller already stores. This
//! module is the translation layer at the boundary, so a different endpoint
//! schema does not leak into the loop, the transcript, or anyone's database.
//!
//! Two formats ship:
//!
//! - [`ChatCompletions`] — `POST {base}/chat/completions`. The default. The
//!   request body is byte-for-byte what this crate sent before the seam
//!   existed (absent an `extra_body`). Reading usage *did* change — see
//!   [`total_tokens`] — which is the one behavioural change on the old path.
//! - [`Responses`] — `POST {base}/responses`. Required by OpenAI's `gpt-5.6`
//!   family, which returns **400** for function tools on chat/completions
//!   unless reasoning is disabled — and disabling reasoning silently ships a
//!   different model than the one anyone benchmarked.
//!
//! The two schemas differ in more than a path. Responses replaces `messages`
//! with `input`, flattens function tools out of their `{"function": {...}}`
//! nesting, splits an assistant turn into separate `function_call` items
//! correlated by `call_id`, and returns a typed `output` array instead of
//! `choices[0].message`.

use serde_json::{Map, Value, json};

use crate::backend::ModelPolicy;
use crate::error::AgentError;

/// One turn, in canonical form, ready to be encoded for some endpoint.
#[derive(Debug, Clone, Copy)]
pub struct WireRequest<'a> {
    pub model: &'a str,
    /// Transcript in Chat Completions message shape.
    pub messages: &'a [Value],
    /// Tool definitions in Chat Completions shape (nested under `function`).
    pub tools: &'a [Value],
    pub tool_choice: &'a str,
    pub policy: &'a ModelPolicy,
}

/// Encodes a turn for one endpoint schema and decodes its reply.
///
/// `parse_response` returns the assistant turn **in canonical Chat Completions
/// shape** regardless of what came off the wire, which is what keeps the loop
/// and the persisted transcript schema-independent.
pub trait WireFormat: Send + Sync + std::fmt::Debug {
    /// Path under `base_url`, without a leading slash.
    fn path(&self) -> &str;

    fn build_request(&self, req: WireRequest<'_>) -> Value;

    /// # Errors
    /// [`AgentError::Decode`] when the reply has no assistant turn in it.
    fn parse_response(&self, resp: &Value) -> Result<(Value, u32), AgentError>;
}

/// Merge caller-supplied body fields. Applied last, so a caller can override
/// anything the format chose — including `model` or `tools` — deliberately.
///
/// Overriding a structural key (`model`, `messages`/`input`, `tools`) is almost
/// always a mistake rather than an intention: it can silently detach the
/// request from the transcript the loop is maintaining. The override still
/// happens — a caller who means it needs the escape hatch — but it warns, and
/// the overridden keys are returned so tests can assert on them without
/// scraping log output.
fn apply_extra(body: &mut Value, extra: &Map<String, Value>) -> Vec<String> {
    let mut overridden = Vec::new();
    if let Some(obj) = body.as_object_mut() {
        for (k, v) in extra {
            if obj.contains_key(k) {
                overridden.push(k.clone());
                tracing::warn!(
                    key = %k,
                    "ModelPolicy::extra_body overrides a field the wire format set"
                );
            }
            obj.insert(k.clone(), v.clone());
        }
    }
    overridden
}

/// Token usage, read tolerantly.
///
/// Chat Completions reports `prompt_tokens`/`completion_tokens`/`total_tokens`;
/// Responses reports `input_tokens`/`output_tokens`/`total_tokens`. A provider
/// that reports neither is not an error — usage is telemetry, and a run must
/// not fail because a proxy omitted it.
fn total_tokens(resp: &Value) -> u32 {
    let usage = &resp["usage"];
    if let Some(t) = usage["total_tokens"].as_u64() {
        return t as u32;
    }
    let a = usage["input_tokens"]
        .as_u64()
        .or_else(|| usage["prompt_tokens"].as_u64())
        .unwrap_or(0);
    let b = usage["output_tokens"]
        .as_u64()
        .or_else(|| usage["completion_tokens"].as_u64())
        .unwrap_or(0);
    (a + b) as u32
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

/// `POST {base_url}/chat/completions` — the default.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatCompletions;

impl WireFormat for ChatCompletions {
    fn path(&self) -> &str {
        "chat/completions"
    }

    fn build_request(&self, req: WireRequest<'_>) -> Value {
        let mut body = json!({
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.policy.max_tokens,
            "temperature": req.policy.temperature,
        });
        if !req.tools.is_empty() {
            body["tools"] = json!(req.tools);
            body["tool_choice"] = json!(req.tool_choice);
        }
        let _ = apply_extra(&mut body, &req.policy.extra_body);
        body
    }

    fn parse_response(&self, resp: &Value) -> Result<(Value, u32), AgentError> {
        let message = resp
            .pointer("/choices/0/message")
            .cloned()
            .ok_or_else(|| AgentError::Decode("response had no choices[0].message".into()))?;
        Ok((message, total_tokens(resp)))
    }
}

/// Message content for a Responses `input` item, or `None` to emit no item.
///
/// The rule that matters: **never drop content**. An earlier cut of this read
/// `content.as_str().unwrap_or_default()`, which silently deleted any message
/// whose content was a parts array — on resume that removed the user's question
/// from the transcript with no error anywhere. Absent and empty content is the
/// only thing that legitimately produces no item.
///
/// A pure-text parts array is folded into a string, which is a shape both
/// schemas agree on. Anything else (images, audio, provider extensions) is
/// passed through untranslated and warned about: Chat Completions names its
/// parts `text`/`image_url` while Responses wants `input_text`/`input_image`, so
/// a faithful translation needs per-part rules this crate cannot verify for
/// every provider. A visible 400 from the API is recoverable; a silently
/// truncated conversation is not.
fn message_content(content: &Value) -> Option<Value> {
    match content {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        Value::String(s) => Some(json!(s)),
        Value::Array(parts) => {
            let all_text = parts
                .iter()
                .all(|p| p["type"] == "text" && p["text"].is_string());
            if all_text {
                let joined: String = parts
                    .iter()
                    .filter_map(|p| p["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("");
                return (!joined.is_empty()).then(|| json!(joined));
            }
            tracing::warn!(
                "message content has non-text parts; passing through untranslated \
                 (Chat Completions part names differ from Responses)"
            );
            Some(content.clone())
        }
        other => {
            tracing::warn!("message content is not a string or parts array; passing through");
            Some(other.clone())
        }
    }
}

/// A `function_call_output.output` value.
///
/// The field takes a string (or a list of output content). A tool result that
/// is structured JSON is serialised rather than flattened to `""` — the same
/// silent-loss bug as above, in the place it would hurt most, since a tool
/// result is the entire reason the turn happened.
fn tool_output_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// A tool call's `arguments`, which both schemas carry as a JSON **string**.
///
/// Same rule as [`tool_output_text`], and it was missed here first time round:
/// `as_str().unwrap_or("{}")` turns arguments that arrive as an object into a
/// zero-argument call. That is the worst possible failure — the tool runs, with
/// nothing, and looks like it succeeded. Absent arguments legitimately mean
/// `{}` (models omit the field for zero-argument tools); anything else is
/// serialised so the call keeps its parameters.
fn tool_arguments_text(arguments: &Value) -> String {
    match arguments {
        Value::String(s) if s.trim().is_empty() => "{}".to_string(),
        Value::String(s) => s.clone(),
        Value::Null => "{}".to_string(),
        other => {
            tracing::warn!("tool arguments were not a JSON string; serialising");
            other.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// `POST {base_url}/responses` — OpenAI's typed-item endpoint.
///
/// Reasoning models reject `temperature`, so it is omitted by default. Call
/// [`Responses::with_temperature`] for a non-reasoning model on this endpoint.
///
/// # Provider support
///
/// This endpoint is **not** part of the OpenAI-compatible surface most
/// providers implement — [`ChatCompletions`] remains the portable default and
/// the right choice unless a model specifically requires otherwise.
///
/// | Provider | `/responses` | How established |
/// |---|---|---|
/// | OpenAI | yes | live, 2026-07-22 — full multi-turn request accepted, tool result honoured |
/// | OpenRouter | yes | live, 2026-07-22 — same generated body accepted for `openai/gpt-4o-mini` and `openai/gpt-5.6-luna`; identical item and usage shapes (plus `cost`/`is_byok`) |
/// | Moonshot | no | `404` — no such route |
/// | Ollama, LiteLLM, Z.ai, … | assume no | unverified; use [`ChatCompletions`] |
///
/// Both live checks sent the body produced by
/// `examples/dump_responses_request.rs`, so the *request* side is verified by
/// the same code path callers use — not by a hand-written payload.
///
/// Azure OpenAI needs an `?api-version=` query parameter that
/// [`crate::ChatClient`] cannot express in a path — a pre-existing limitation
/// of the transport, not of this format, and it affects both formats equally.
#[derive(Debug, Clone, Copy)]
pub struct Responses {
    send_temperature: bool,
}

impl Default for Responses {
    fn default() -> Self {
        Self::new()
    }
}

impl Responses {
    #[must_use]
    pub fn new() -> Self {
        Self {
            send_temperature: false,
        }
    }

    /// Send `temperature`. Only for non-reasoning models — the reasoning
    /// families 400 on it.
    #[must_use]
    pub fn with_temperature(mut self) -> Self {
        self.send_temperature = true;
        self
    }

    /// Canonical transcript → Responses `input` items.
    ///
    /// An assistant turn that both spoke and called tools becomes *several*
    /// items: Responses models a turn as typed items, not one message with a
    /// `tool_calls` array hanging off it.
    fn to_input(messages: &[Value]) -> Vec<Value> {
        let mut input = Vec::with_capacity(messages.len());
        for m in messages {
            let role = m["role"].as_str().unwrap_or_default();

            if role == "tool" {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"].as_str().unwrap_or_default(),
                    "output": tool_output_text(&m["content"]),
                }));
                continue;
            }

            if let Some(content) = message_content(&m["content"]) {
                input.push(json!({ "role": role, "content": content }));
            }

            for call in m["tool_calls"].as_array().into_iter().flatten() {
                input.push(json!({
                    "type": "function_call",
                    // `call_id` correlates with function_call_output; the item's
                    // own `id` is a different identifier and is not echoed back.
                    "call_id": call["id"].as_str().unwrap_or_default(),
                    "name": call["function"]["name"].as_str().unwrap_or_default(),
                    "arguments": tool_arguments_text(&call["function"]["arguments"]),
                }));
            }
        }
        input
    }

    /// Chat-shaped tool defs → Responses tool defs (flat, not nested).
    fn to_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                let f = &t["function"];
                json!({
                    "type": "function",
                    "name": f["name"],
                    "description": f["description"],
                    "parameters": f["parameters"],
                })
            })
            .collect()
    }
}

impl WireFormat for Responses {
    fn path(&self) -> &str {
        "responses"
    }

    fn build_request(&self, req: WireRequest<'_>) -> Value {
        let mut body = json!({
            "model": req.model,
            "input": Self::to_input(req.messages),
            "max_output_tokens": req.policy.max_tokens,
        });
        if self.send_temperature {
            body["temperature"] = json!(req.policy.temperature);
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(Self::to_tools(req.tools));
            body["tool_choice"] = json!(req.tool_choice);
        }
        let _ = apply_extra(&mut body, &req.policy.extra_body);
        body
    }

    fn parse_response(&self, resp: &Value) -> Result<(Value, u32), AgentError> {
        let output = resp["output"]
            .as_array()
            .ok_or_else(|| AgentError::Decode("response had no output array".into()))?;

        let mut text = String::new();
        let mut tool_calls = Vec::new();

        for item in output {
            match item["type"].as_str().unwrap_or_default() {
                "message" => {
                    for part in item["content"].as_array().into_iter().flatten() {
                        if part["type"] == "output_text"
                            && let Some(t) = part["text"].as_str()
                        {
                            text.push_str(t);
                        }
                    }
                }
                "function_call" => tool_calls.push(json!({
                    "id": item["call_id"].as_str().unwrap_or_default(),
                    "type": "function",
                    "function": {
                        "name": item["name"].as_str().unwrap_or_default(),
                        "arguments": tool_arguments_text(&item["arguments"]),
                    }
                })),
                // `reasoning` items and anything else the API grows are ignored
                // rather than rejected: an unknown item type must not fail a run.
                _ => {}
            }
        }

        let mut message = json!({
            "role": "assistant",
            "content": if text.is_empty() { Value::Null } else { json!(text) },
        });
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }
        Ok((message, total_tokens(resp)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ModelPolicy {
        ModelPolicy::single("m")
    }

    fn chat_tools() -> Vec<Value> {
        vec![json!({
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
            }
        })]
    }

    /// A transcript that has been through a full tool round-trip: system, user,
    /// assistant-with-tool-call, tool result.
    fn transcript() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "hi"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{\"q\":\"x\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "found"}),
        ]
    }

    // ---- chat completions is unchanged ------------------------------------

    #[test]
    fn chat_completions_body_is_what_the_crate_always_sent() {
        let msgs = transcript();
        let tools = chat_tools();
        let p = policy();
        let body = ChatCompletions.build_request(WireRequest {
            model: "m",
            messages: &msgs,
            tools: &tools,
            tool_choice: "auto",
            policy: &p,
        });
        assert_eq!(body["model"], "m");
        assert_eq!(body["messages"], json!(msgs), "transcript passes through");
        assert_eq!(body["max_tokens"], p.max_tokens);
        assert_eq!(body["temperature"], p.temperature);
        assert_eq!(body["tools"][0]["function"]["name"], "grep");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(ChatCompletions.path(), "chat/completions");
    }

    #[test]
    fn chat_completions_omits_tools_when_there_are_none() {
        let p = policy();
        let body = ChatCompletions.build_request(WireRequest {
            model: "m",
            messages: &[],
            tools: &[],
            tool_choice: "auto",
            policy: &p,
        });
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    // ---- responses request -------------------------------------------------

    #[test]
    fn responses_flattens_tool_definitions() {
        // The difference that silently 400s if you get it wrong: Responses puts
        // name/parameters at the top level, not under "function".
        let tools = chat_tools();
        let p = policy();
        let body = Responses::new().build_request(WireRequest {
            model: "m",
            messages: &[],
            tools: &tools,
            tool_choice: "auto",
            policy: &p,
        });
        let t = &body["tools"][0];
        assert_eq!(t["type"], "function");
        assert_eq!(t["name"], "grep", "name must be flat, not under .function");
        assert_eq!(t["parameters"]["required"][0], "q");
        assert!(t.get("function").is_none(), "no nested function object");
    }

    #[test]
    fn responses_splits_an_assistant_turn_into_typed_items() {
        let msgs = transcript();
        let p = policy();
        let body = Responses::new().build_request(WireRequest {
            model: "m",
            messages: &msgs,
            tools: &[],
            tool_choice: "auto",
            policy: &p,
        });
        let input = body["input"].as_array().unwrap();

        assert_eq!(input[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(input[1], json!({"role": "user", "content": "hi"}));
        // The assistant turn had null content, so it contributes only the call.
        assert_eq!(
            input[2],
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "grep",
                "arguments": "{\"q\":\"x\"}"
            })
        );
        assert_eq!(
            input[3],
            json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "found"
            })
        );
        assert_eq!(input.len(), 4);
        assert!(body.get("messages").is_none(), "Responses has no `messages`");
        assert_eq!(body["max_output_tokens"], p.max_tokens);
        assert_eq!(Responses::new().path(), "responses");
    }

    #[test]
    fn an_assistant_turn_that_both_spoke_and_called_yields_two_items() {
        let msgs = vec![json!({
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{
                "id": "call_9",
                "type": "function",
                "function": {"name": "grep", "arguments": "{}"}
            }]
        })];
        let p = policy();
        let body = Responses::new().build_request(WireRequest {
            model: "m",
            messages: &msgs,
            tools: &[],
            tool_choice: "auto",
            policy: &p,
        });
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 2, "one message item + one function_call item");
        assert_eq!(input[0]["content"], "let me check");
        assert_eq!(input[1]["type"], "function_call");
    }

    #[test]
    fn temperature_is_omitted_by_default_because_reasoning_models_reject_it() {
        let p = policy();
        let req = WireRequest {
            model: "m",
            messages: &[],
            tools: &[],
            tool_choice: "auto",
            policy: &p,
        };
        assert!(Responses::new().build_request(req).get("temperature").is_none());
        assert_eq!(
            Responses::new().with_temperature().build_request(req)["temperature"],
            p.temperature
        );
    }

    #[test]
    fn overriding_a_structural_key_via_extra_body_is_reported() {
        // Raised in review: extra_body is applied last and can silently detach
        // the request from the transcript the loop maintains. The override
        // still happens — it is an escape hatch — but it is no longer silent.
        let mut body = json!({"model": "m", "input": [], "tools": []});
        let mut extra = Map::new();
        extra.insert("model".into(), json!("something-else"));
        extra.insert("reasoning_effort".into(), json!("high"));

        let overridden = apply_extra(&mut body, &extra);
        assert_eq!(overridden, vec!["model".to_string()], "only collisions");
        assert_eq!(body["model"], "something-else", "the override still applies");
        assert_eq!(body["reasoning_effort"], "high", "additions are not collisions");
    }

    #[test]
    fn extra_body_reaches_both_formats() {
        // The reason this field exists: jpt-copilot needs
        // parallel_tool_calls=false or the model emits a mesh call before it has
        // seen the Part ID the previous call returned.
        let mut p = policy();
        p.extra_body
            .insert("parallel_tool_calls".into(), json!(false));
        let req = WireRequest {
            model: "m",
            messages: &[],
            tools: &[],
            tool_choice: "auto",
            policy: &p,
        };
        assert_eq!(ChatCompletions.build_request(req)["parallel_tool_calls"], false);
        assert_eq!(Responses::new().build_request(req)["parallel_tool_calls"], false);
    }

    // ---- responses parsing -------------------------------------------------

    #[test]
    fn responses_output_becomes_a_canonical_assistant_message() {
        let resp = json!({
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "the answer"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "grep", "arguments": "{\"q\":\"x\"}"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let (message, tokens) = Responses::new().parse_response(&resp).unwrap();

        assert_eq!(tokens, 15);
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "the answer");
        // Canonical shape: the loop reads .id and .function.name, so the
        // translation must produce exactly that, keyed on call_id not id.
        assert_eq!(message["tool_calls"][0]["id"], "call_1");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "grep");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"x\"}"
        );
    }

    #[test]
    fn a_text_only_response_has_no_tool_calls_key() {
        // The loop breaks the exploration loop on an empty tool_calls array, so
        // an absent key and an empty array must not be confused.
        let resp = json!({
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}]}],
            "usage": {"total_tokens": 3}
        });
        let (message, _) = Responses::new().parse_response(&resp).unwrap();
        assert_eq!(message["content"], "done");
        assert!(message.get("tool_calls").is_none());
    }

    #[test]
    fn a_tool_only_response_has_null_content() {
        let resp = json!({
            "output": [{"type": "function_call", "call_id": "c", "name": "grep", "arguments": "{}"}],
            "usage": {"total_tokens": 1}
        });
        let (message, _) = Responses::new().parse_response(&resp).unwrap();
        assert!(message["content"].is_null());
        assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn unknown_output_item_types_are_ignored_not_fatal() {
        let resp = json!({
            "output": [
                {"type": "something_openai_added_last_tuesday"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "ok"}]}
            ]
        });
        let (message, tokens) = Responses::new().parse_response(&resp).unwrap();
        assert_eq!(message["content"], "ok");
        assert_eq!(tokens, 0, "absent usage is telemetry loss, not a failure");
    }

    #[test]
    fn a_reply_with_no_output_array_is_a_decode_error() {
        let err = Responses::new()
            .parse_response(&json!({"error": {"message": "boom"}}))
            .unwrap_err();
        assert!(matches!(err, AgentError::Decode(_)), "got {err:?}");
    }

    // ---- usage tolerance ---------------------------------------------------

    #[test]
    fn usage_is_read_from_either_schemas_key_names() {
        assert_eq!(total_tokens(&json!({"usage": {"total_tokens": 42}})), 42);
        assert_eq!(
            total_tokens(&json!({"usage": {"input_tokens": 10, "output_tokens": 5}})),
            15,
            "Responses names"
        );
        assert_eq!(
            total_tokens(&json!({"usage": {"prompt_tokens": 7, "completion_tokens": 3}})),
            10,
            "Chat Completions names"
        );
        assert_eq!(total_tokens(&json!({})), 0, "no usage is not an error");
    }

    // ---- round trip --------------------------------------------------------

    #[test]
    fn a_parsed_responses_turn_re_encodes_to_the_same_call_id() {
        // The correlation that makes multi-turn tool use work: the call_id we
        // read out must be the call_id we send back, or the model sees an
        // orphaned result.
        let resp = json!({
            "output": [{"type": "function_call", "call_id": "call_abc",
                        "name": "grep", "arguments": "{}"}]
        });
        let (message, _) = Responses::new().parse_response(&resp).unwrap();

        let transcript = vec![
            message,
            json!({"role": "tool", "tool_call_id": "call_abc", "content": "result"}),
        ];
        let input = Responses::to_input(&transcript);
        assert_eq!(input[0]["call_id"], "call_abc");
        assert_eq!(input[1]["call_id"], "call_abc");
        assert_eq!(input[1]["output"], "result");
    }
}

#[cfg(test)]
mod live_wire_fixture {
    //! Captured verbatim from `POST https://api.openai.com/v1/responses`
    //! against `gpt-5.6-luna` on 2026-07-22. Every other Responses test uses a
    //! shape *I* wrote, which proves only that the parser matches my
    //! assumptions. This one proves it matches OpenAI.
    //!
    //! Fields the parser ignores are kept in the fixture on purpose: an
    //! unfamiliar member must never be what breaks a run.

    use super::*;
    use serde_json::json;

    fn captured() -> Value {
        json!({
            "id": "resp_0ab8a5408228d628006a6011d467b0819b8b32cba0a7b906a2",
            "object": "response",
            "status": "completed",
            "model": "gpt-5.6-luna",
            "max_output_tokens": 200,
            "parallel_tool_calls": false,
            "temperature": 1.0,
            "reasoning": {"context": "all_turns", "effort": "medium", "mode": "standard"},
            "output": [{
                "id": "fc_0ab8a5408228d628006a6011d53e48819bbe7d6f0ea2edbdb1",
                "type": "function_call",
                "status": "completed",
                "arguments": "{\"city\":\"Paris\"}",
                "call_id": "call_7Jl0dn2CmiDLVR7jXDb7Y6I9",
                "name": "get_weather"
            }],
            "usage": {
                "input_tokens": 128,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 18,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 146
            }
        })
    }

    /// The *request* this crate generates for a full multi-turn transcript,
    /// captured from `examples/dump_responses_request.rs` and sent verbatim to
    /// both providers on 2026-07-22. Both accepted it and answered from the
    /// tool result ("Paris: 18°C and raining"), which is what proves the model
    /// actually read our `function_call_output` rather than ignoring it.
    ///
    /// Fixtures alone could not establish this: they only ever proved the
    /// parser matches a reply. Nothing proved the API accepts what we *send*.
    fn generated_request() -> Value {
        let messages = vec![
            json!({"role": "system", "content": "You are a terse assistant."}),
            json!({"role": "user", "content": "What is the weather in Paris?"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_seed_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]}),
            json!({"role": "tool", "tool_call_id": "call_seed_1", "content": "18C, raining"}),
        ];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}
            }
        })];
        let mut policy = ModelPolicy::single("gpt-5.6-luna");
        policy.max_tokens = 200;
        policy
            .extra_body
            .insert("parallel_tool_calls".into(), json!(false));

        Responses::new().build_request(WireRequest {
            model: "gpt-5.6-luna",
            messages: &messages,
            tools: &tools,
            tool_choice: "auto",
            policy: &policy,
        })
    }

    #[test]
    fn the_request_accepted_by_both_providers_is_still_what_we_generate() {
        // Pins the exact bytes both APIs accepted. If translation drifts, this
        // fails here rather than as a 400 in production.
        let body = generated_request();
        let input = body["input"].as_array().unwrap();

        assert_eq!(input.len(), 4);
        assert_eq!(input[0], json!({"role": "system", "content": "You are a terse assistant."}));
        assert_eq!(input[1], json!({"role": "user", "content": "What is the weather in Paris?"}));
        assert_eq!(
            input[2],
            json!({"type": "function_call", "call_id": "call_seed_1",
                   "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"})
        );
        assert_eq!(
            input[3],
            json!({"type": "function_call_output", "call_id": "call_seed_1",
                   "output": "18C, raining"})
        );
        assert_eq!(body["max_output_tokens"], 200);
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert!(body.get("temperature").is_none());
        assert!(body.get("messages").is_none());
    }

    /// The text reply to that request, captured verbatim from OpenAI. Carries
    /// members the parser ignores (`phase`, `annotations`, `logprobs`,
    /// `status`) on purpose.
    #[test]
    fn the_real_text_reply_parses() {
        let resp = json!({
            "output": [{
                "id": "msg_00012b2ba28d1ae6006a60189ab9d081988f899c7066f93f58",
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": "Paris: 18°C and raining."
                }],
                "phase": "final_answer",
                "role": "assistant"
            }],
            "usage": {
                "input_tokens": 171,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 183
            }
        });
        let (message, tokens) = Responses::new().parse_response(&resp).unwrap();
        assert_eq!(tokens, 183);
        assert_eq!(message["content"], "Paris: 18°C and raining.");
        assert!(message.get("tool_calls").is_none());
    }

    /// OpenRouter's `function_call`, captured live from
    /// `openrouter.ai/api/v1/responses` with `openai/gpt-5.6-luna`.
    ///
    /// Same shape as OpenAI's, and it independently confirms the `id` /
    /// `call_id` distinction: `fc_tmp_…` versus `call_…`. Correlating on `id`
    /// would orphan every tool result.
    #[test]
    fn openrouters_tool_call_parses_identically() {
        let resp = json!({
            "output": [{
                "arguments": "{\"city\":\"Paris\"}",
                "call_id": "call_TuPiRQcCzLkaVq0v5BlUZSmp",
                "id": "fc_tmp_iadnucwi7ps",
                "name": "get_weather",
                "status": "completed",
                "type": "function_call"
            }],
            // OpenRouter adds `cost` / `is_byok` beside the standard counts.
            "usage": {
                "cost": 0.000123, "is_byok": false,
                "input_tokens": 171, "output_tokens": 12, "total_tokens": 183
            }
        });
        let (message, tokens) = Responses::new().parse_response(&resp).unwrap();
        assert_eq!(tokens, 183, "provider-specific usage keys must not confuse it");
        assert_eq!(
            message["tool_calls"][0]["id"], "call_TuPiRQcCzLkaVq0v5BlUZSmp",
            "correlate on call_id — `id` here is fc_tmp_… and would orphan the result"
        );
        assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn the_real_response_parses_into_a_canonical_tool_call() {
        let (message, tokens) = Responses::new().parse_response(&captured()).unwrap();

        assert_eq!(tokens, 146);
        assert_eq!(message["role"], "assistant");
        assert!(message["content"].is_null(), "a tool-only turn has no text");

        // The id the loop must echo back is `call_id`, not the item's own `id`
        // — the fixture has both, and they differ, so this discriminates.
        assert_eq!(
            message["tool_calls"][0]["id"], "call_7Jl0dn2CmiDLVR7jXDb7Y6I9",
            "must correlate on call_id, not the fc_… item id"
        );
        assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[test]
    fn the_real_usage_block_is_read_despite_its_nested_details() {
        // `input_tokens_details` / `output_tokens_details` are objects sitting
        // beside the scalars. Reading usage must not trip over them.
        assert_eq!(total_tokens(&captured()), 146);
    }

    #[test]
    fn the_captured_turn_round_trips_back_into_a_request() {
        // End to end on real data: parse the live reply, answer the tool, and
        // confirm the next request carries the same call_id on both items.
        let (message, _) = Responses::new().parse_response(&captured()).unwrap();
        let transcript = vec![
            message,
            json!({"role": "tool",
                   "tool_call_id": "call_7Jl0dn2CmiDLVR7jXDb7Y6I9",
                   "content": "18C and raining"}),
        ];
        let input = Responses::to_input(&transcript);

        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_7Jl0dn2CmiDLVR7jXDb7Y6I9");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_7Jl0dn2CmiDLVR7jXDb7Y6I9");
        assert_eq!(input[1]["output"], "18C and raining");
    }
}

#[cfg(test)]
mod no_silent_loss {
    //! Regression tests for a defect this module shipped with and did not
    //! catch: every other test used plain-string content, so translation could
    //! delete a message and 101 tests stayed green.

    use super::*;
    use serde_json::json;

    #[test]
    fn a_text_parts_array_is_folded_not_dropped() {
        let msgs = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "look at "}, {"type": "text", "text": "this"}]
        })];
        let input = Responses::to_input(&msgs);
        assert_eq!(input.len(), 1, "the user's question must not vanish");
        assert_eq!(input[0]["content"], "look at this");
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn non_text_parts_are_passed_through_rather_than_deleted() {
        // Passing this through may earn a 400 from the API. That is strictly
        // better than deleting the message and asking the model a question it
        // cannot see.
        let msgs = vec![json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
        })];
        let input = Responses::to_input(&msgs);
        assert_eq!(input.len(), 1);
        assert!(input[0]["content"].is_array());
    }

    #[test]
    fn structured_tool_arguments_are_serialised_not_reduced_to_no_args() {
        // The worst shape of this bug: the tool would run with no parameters
        // and report success. Raised in review on the first cut of this PR.
        let msgs = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "grep", "arguments": {"pattern": "fn main"}}}]
        })];
        let input = Responses::to_input(&msgs);
        assert_eq!(input[0]["arguments"], r#"{"pattern":"fn main"}"#);

        // Absent arguments legitimately mean "{}" — models omit the field for
        // zero-argument tools, so this must not become a warning-worthy case.
        let msgs = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ping"}}]
        })];
        assert_eq!(Responses::to_input(&msgs)[0]["arguments"], "{}");
    }

    #[test]
    fn structured_arguments_survive_parsing_too() {
        let resp = json!({
            "output": [{"type": "function_call", "call_id": "c1", "name": "grep",
                        "arguments": {"pattern": "fn main"}}]
        });
        let (message, _) = Responses::new().parse_response(&resp).unwrap();
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            r#"{"pattern":"fn main"}"#
        );
    }

    #[test]
    fn a_structured_tool_result_is_serialised_not_emptied() {
        let msgs = vec![json!({
            "role": "tool", "tool_call_id": "c1", "content": {"rows": 3}
        })];
        let input = Responses::to_input(&msgs);
        assert_eq!(input.len(), 1);
        assert_eq!(
            input[0]["output"], r#"{"rows":3}"#,
            "a tool result is why the turn happened; it must survive"
        );
    }

    #[test]
    fn absent_and_empty_content_still_emit_no_item() {
        // The legitimate no-item cases must keep working, or an assistant turn
        // that only called tools would gain a bogus empty message.
        let msgs = vec![
            json!({"role": "assistant", "content": null}),
            json!({"role": "assistant", "content": ""}),
        ];
        assert!(Responses::to_input(&msgs).is_empty());
    }

    #[test]
    fn an_assistant_turn_with_null_content_still_emits_its_call() {
        let msgs = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "grep", "arguments": "{}"}}]
        })];
        let input = Responses::to_input(&msgs);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call");
    }

    /// Deliberate behaviour change on the pre-existing path, pinned so it is a
    /// decision rather than an accident.
    ///
    /// 0.1.1 read `/usage/total_tokens` and reported 0 when it was absent, so a
    /// provider that reports only the component counts made every run look
    /// free — under-counting `stop_after_tokens` and every `Usage` event. The
    /// tolerant read fixes that, and it is a change: a caller relying on the
    /// old under-count will now stop earlier.
    #[test]
    fn chat_completions_usage_now_falls_back_to_component_counts() {
        let resp = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        });
        let (_, tokens) = ChatCompletions.parse_response(&resp).unwrap();
        assert_eq!(tokens, 150, "0.1.1 reported 0 here");

        // An explicit total still wins, so the common path is untouched.
        let resp = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 999}
        });
        let (_, tokens) = ChatCompletions.parse_response(&resp).unwrap();
        assert_eq!(tokens, 999);
    }
}
