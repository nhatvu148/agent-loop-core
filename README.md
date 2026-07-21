# agent-loop-core

A hand-rolled LLM agent loop for Rust, extracted from two production consumers
([vexar](https://github.com/nhatvu148/vexar), a multi-agent coding cockpit, and
a PR-review bot) that had independently converged on the same architecture.

It owns the parts worth sharing — the loop, the transport, the tool contract —
and deliberately nothing else. Request and response bodies stay
`serde_json::Value`, so each consumer keeps its own typed request/response
structs; this crate does not try to be a model abstraction.

## What's in it

- **Resilient transport** (`ChatClient`) — a request timeout and 429/5xx retry
  with backoff over any OpenAI-compatible endpoint. Built once and cloned, so
  the connection pool is reused. An empty API key means keyless (local
  providers); statuses are classified before the body is decoded so a proxy's
  HTML 502 still retries.
- **Typed tools** (`Tool` / `ToolRegistry`) — a tool declares an `Args` type;
  the JSON schema is *derived* from it and dispatch deserializes into it, so the
  advertised schema and the arguments the tool reads cannot drift. Malformed
  arguments are a typed error, not a silent empty default. Runtime-schema tools
  (e.g. MCP) register via `ErasedTool`.
- **The loop** (`ChatBackend`) — a two-phase design: a cheap model drives tool
  calls to gather context, then a strong model answers with tools forbidden.
  Set both models equal for an ordinary single-model loop. Turn cap, token
  threshold, wall-clock timeout, and interrupt are all honored. Reaching the
  turn cap still runs the final synthesis pass (the run reports `MaxTurns`, not
  `Complete`) — you've gathered enough, so produce the answer. A *hard* stop —
  token budget, timeout, or interrupt — skips synthesis, because it means "stop
  spending".
- **Streaming** (`EventSink` / `AgentEvent`) — one event stream drives a live
  UI or, with `EventSink::none()`, nothing at all. `run_structured::<T>()` adds
  a typed ending for callers that want a value rather than a stream.
- **Typed errors** (`AgentError`) — `is_retryable()`, `retry_after()`,
  `status()`, so a caller can branch on failure kind.

## Example

```rust,no_run
use agent_loop_core::{ChatClient, ProviderConfig};

# async fn f() -> Result<(), agent_loop_core::AgentError> {
let client = ChatClient::new(ProviderConfig {
    base_url: "https://openrouter.ai/api/v1".into(),
    api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
    ..Default::default()
})?;

let res = client
    .post_chat(&serde_json::json!({
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .await?;
# let _ = res;
# Ok(())
# }
```

## Status

Pre-1.0. The API is still settling, so expect breaking changes between `0.x`
releases.

## License

MIT — see LICENSE-MIT.
