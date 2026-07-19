//! Shared primitives for the agent harnesses in this workspace and in
//! `pr-review-core`.
//!
//! Both codebases independently converged on the same architecture — a
//! hand-rolled tool-calling loop behind a backend trait — and independently
//! shipped the same gaps: no HTTP timeout, no retry, and `anyhow` all the way to
//! the public API so callers could not branch on failure kind. This crate is the
//! single implementation of the parts that were duplicated.
//!
//! Scope is deliberately narrow. Request and response bodies stay
//! `serde_json::Value` so each consumer keeps its own typed structs; this crate
//! owns transport and error semantics, not model abstraction.
//!
//! ```no_run
//! use agent_core::{ChatClient, ProviderConfig};
//! # async fn f() -> Result<(), agent_core::AgentError> {
//! let client = ChatClient::new(ProviderConfig {
//!     base_url: "https://openrouter.ai/api/v1".into(),
//!     api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
//!     ..Default::default()
//! })?;
//!
//! let res = client
//!     .post_chat(&serde_json::json!({
//!         "model": "anthropic/claude-sonnet-4.5",
//!         "messages": [{"role": "user", "content": "hi"}],
//!     }))
//!     .await?;
//! # let _ = res;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod backend;
pub mod error;
pub mod events;
pub mod loop_runner;
pub mod provider;
pub mod tools;

pub use backend::{Backend, ModelPolicy, RunOutcome, RunRequest, extract_json, run_structured};
pub use error::AgentError;
pub use events::{AgentEvent, EventSink, StopReason};
pub use loop_runner::{ChatBackend, trim_history};
pub use provider::{ChatClient, ProviderConfig, RetryPolicy};
pub use tools::{ErasedTool, Tool, ToolError, ToolOutput, ToolRegistry};
