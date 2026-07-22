# agent-loop-core

[![crates.io](https://img.shields.io/crates/v/agent-loop-core.svg)](https://crates.io/crates/agent-loop-core)
[![docs.rs](https://img.shields.io/docsrs/agent-loop-core)](https://docs.rs/agent-loop-core)
[![license](https://img.shields.io/crates/l/agent-loop-core.svg)](#license)

A small, hand-rolled LLM **agent loop** for Rust — resilient transport, typed
tools, streaming events, and an optional two-model cost split — over any
OpenAI-compatible endpoint. Not a framework: it owns the loop and the plumbing
worth sharing, and stays out of your way for everything else.

Extracted from two production reviewers that had independently converged on the
same architecture, then generalized.

## Why

Most "call an LLM in a loop until it stops asking for tools" code gets rewritten
per project — and gets the boring-but-critical parts wrong: no request timeout
(a stalled provider hangs forever), no retry (one 429 kills the run), tool
schemas that drift from the code that reads them. This crate is that loop, done
once:

- **A request timeout and 429/5xx retry** with backoff. `reqwest` has *no*
  default timeout; this closes the class of bug where a hung connection hangs
  the agent indefinitely, and where a single rate-limit response discards the
  whole run.
- **Typed tools that can't drift** — a tool declares its `Args` type; the JSON
  schema the model sees is *derived* from it and dispatch deserializes into it,
  so they are the same type by construction. Malformed arguments are a typed
  error, not a silent empty default.
- **An optional two-model split** — a cheap model drives the tool loop to gather
  context, then a strong model writes the final answer with tools forbidden. Set
  both to the same model for an ordinary single-model loop.
- **Streaming is optional** — one `EventSink` drives a live UI, or
  `EventSink::none()` with `run_structured::<T>()` gives you a typed value and no
  streaming at all.
- **Provider-agnostic** — request and response bodies stay `serde_json::Value`,
  so you keep your own typed structs and this crate never becomes a competing
  model abstraction.

## Install

```sh
cargo add agent-loop-core
```

## Example

Define a typed tool, run a two-phase loop, and parse the final answer:

```rust,no_run
use std::sync::Arc;

use agent_loop_core::{
    Backend, ChatBackend, ChatClient, EventSink, ModelPolicy, ProviderConfig,
    RunRequest, Tool, ToolError, ToolOutput, ToolRegistry, run_structured,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

// A tool. Its advertised schema is derived from `Args`, so the schema the model
// sees and the arguments this code reads are the same type — they can't drift.
struct Grep;

#[derive(Deserialize, JsonSchema)]
struct GrepArgs {
    /// Regex to search for across the repository.
    pattern: String,
}

#[async_trait]
impl Tool for Grep {
    type Args = GrepArgs;
    fn name(&self) -> &'static str { "grep" }
    fn description(&self) -> &'static str { "Regex search across the repository." }
    async fn call(&self, args: GrepArgs) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::ok(format!("(searched for {})", args.pattern)))
    }
}

#[derive(Deserialize)]
struct Review {
    summary: String,
}

# async fn run() -> Result<(), agent_loop_core::AgentError> {
let chat = ChatClient::new(ProviderConfig {
    base_url: "https://openrouter.ai/api/v1".into(),
    api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
    ..Default::default() // 120s timeout, 3 retries by default
})?;

let backend = ChatBackend::new(
    chat,
    Arc::new(ToolRegistry::new().with(Grep)),
    // Cheap model explores with tools; strong model writes the answer.
    ModelPolicy::tiered("openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5"),
);

// Runs the loop and parses the final message as `Review`.
// Pass `EventSink::channel()` instead of `none()` to stream progress to a UI.
let (review, outcome): (Review, _) = run_structured(
    &backend,
    RunRequest {
        system_prompt: "You are a code reviewer. Respond with JSON: {\"summary\": ...}".into(),
        user_prompt: "Review the pending change.".into(),
        ..Default::default()
    },
    EventSink::none(),
)
.await?;

println!(
    "{} — {} turns, {} tokens",
    review.summary, outcome.turns, outcome.total_tokens
);
# Ok(())
# }
```

Just want the resilient HTTP without the loop? `ChatClient::post_chat(&json)`
gives you the timeout + retry on any OpenAI-compatible `chat/completions`
endpoint.

Talking to a model that needs OpenAI's typed-item endpoint instead:

```rust,ignore
use agent_loop_core::{ChatBackend, ModelPolicy, Responses};
use std::sync::Arc;

let backend = ChatBackend::new(chat, tools, ModelPolicy::single("gpt-5.6-luna"))
    .with_wire_format(Arc::new(Responses::new()))
    .with_extra_field("parallel_tool_calls", false);
```

## What's in it

- **`ChatClient`** — the resilient transport (timeout, connect timeout, retry
  with backoff, `Retry-After` honored). Built once and cloned so the connection
  pool is reused. An empty API key means keyless (local providers); statuses are
  classified *before* the body is decoded, so a proxy's HTML `502` still retries.
- **`Tool` / `ToolRegistry`** — typed tools with derived schemas. Runtime-schema
  tools (e.g. MCP) register via `ErasedTool`.
- **`WireFormat`** — the endpoint schema. `ChatCompletions` (default) and
  `Responses` ship; the loop and the transcript stay in Chat Completions shape
  either way, so a session written by one is readable by the other. Reach for
  `Responses` when a model requires it — OpenAI's `gpt-5.6` family returns 400
  for function tools on `chat/completions` unless reasoning is disabled, and
  disabling reasoning ships a different model than the one you benchmarked.
  Anything the crate doesn't model goes in `ChatBackend::with_extra_body`.
- **`ChatBackend`** — the two-phase loop. Turn cap, token threshold, wall-clock
  timeout, and interrupt are all honored. Reaching the turn cap still runs the
  final synthesis pass (the run reports `MaxTurns`, not `Complete`) — you've
  gathered enough, so produce the answer. A *hard* stop (token budget, timeout,
  interrupt) skips synthesis, because it means "stop spending".
- **`EventSink` / `AgentEvent`** — one event stream for a live UI, or nothing.
  `run_structured::<T>()` adds a typed ending for callers that want a value.
- **`AgentError`** — `is_retryable()`, `retry_after()`, `status()`, so a caller
  can branch on failure kind.

## When to use this

Reach for `agent-loop-core` when you want a **small, legible agent loop you
control** — you can read the whole thing, and it imposes no model abstraction,
no runtime opinion, and no plugin system. If you want a batteries-included
framework (many providers behind one trait, vector stores, built-in RAG), look
at [`rig`](https://github.com/0xplaygrounds/rig) or
[`swiftide`](https://swiftide.rs/) instead.

## Used by

- [`pr-review-core`](https://github.com/nhatvu148/pr-review-core) — the engine
  behind an advisory AI PR reviewer for GitHub and Bitbucket.
- A private multi-agent coding cockpit.

## Status

Pre-1.0 (`0.x`). The API is still settling — expect breaking changes between
minor versions until `1.0`. MSRV: 1.88.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
