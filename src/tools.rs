//! Typed tools with one source of truth for schema and dispatch.
//!
//! Both codebases advertised tool schemas as hand-written `json!` literals and
//! dispatched them through a separate hardcoded `match`. Two independent sources
//! of truth that could silently drift, and both parsed arguments with
//! `serde_json::from_str(..).unwrap_or(json!({}))` — so unparseable arguments
//! became an empty object and the tool ran with no parameters. In
//! `pr-review-core` that turned a malformed `grep` call into a repo-wide
//! empty-regex match; step 1 pinned that behaviour in
//! `malformed_tool_args_silently_become_an_empty_object`.
//!
//! Here a tool declares an `Args` type, the schema is *derived* from it, and
//! dispatch deserialises into it. Drift is unrepresentable, and bad arguments
//! are a [`ToolError::InvalidArguments`] instead of a silent default.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// Why a tool call did not produce a normal result.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    /// Arguments did not match the tool's declared schema. Previously silent.
    #[error("invalid arguments for `{tool}`: {reason}")]
    InvalidArguments { tool: String, reason: String },

    /// The model named a tool the registry doesn't have.
    #[error("unknown tool `{0}`")]
    NotFound(String),

    /// The tool ran and failed.
    #[error("`{tool}` failed: {reason}")]
    Execution { tool: String, reason: String },
}

impl ToolError {
    /// Render for the model. Tool failures are usually fed back as content so
    /// the agent can recover, rather than aborting the run.
    #[must_use]
    pub fn to_tool_output(&self) -> ToolOutput {
        ToolOutput {
            content: self.to_string(),
            is_error: true,
        }
    }
}

/// What a tool hands back to the loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    #[must_use]
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    #[must_use]
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// A tool the model can call.
///
/// Implement this, not [`ErasedTool`] — the blanket impl derives the schema from
/// `Args` and wires dispatch to the same type, which is what makes drift
/// impossible.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Arguments. `JsonSchema` produces what the model is told; `Deserialize`
    /// consumes what the model sends. One type, so they cannot disagree.
    type Args: DeserializeOwned + JsonSchema + Send + 'static;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// # Errors
    /// [`ToolError::Execution`] when the tool runs but fails.
    async fn call(&self, args: Self::Args) -> Result<ToolOutput, ToolError>;
}

/// Object-safe view of a [`Tool`], used for storage and dispatch.
#[async_trait]
pub trait ErasedTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// JSON Schema of the arguments, derived from `Tool::Args`.
    fn schema(&self) -> Value;
    /// OpenAI function-calling definition.
    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.schema(),
            }
        })
    }
    /// # Errors
    /// [`ToolError::InvalidArguments`] if `raw` doesn't match the schema.
    async fn call_erased(&self, raw: Value) -> Result<ToolOutput, ToolError>;
}

#[async_trait]
impl<T: Tool> ErasedTool for T {
    fn name(&self) -> &'static str {
        Tool::name(self)
    }

    fn description(&self) -> &'static str {
        Tool::description(self)
    }

    fn schema(&self) -> Value {
        sanitize_schema(
            serde_json::to_value(schemars::schema_for!(T::Args))
                .unwrap_or_else(|_| json!({"type": "object", "properties": {}})),
        )
    }

    async fn call_erased(&self, raw: Value) -> Result<ToolOutput, ToolError> {
        // A missing `arguments` member is an empty object, not an error — models
        // legitimately omit it for zero-argument tools. Anything *present but
        // wrong* is now an error rather than a silent default.
        let raw = if raw.is_null() { json!({}) } else { raw };
        let args: T::Args =
            serde_json::from_value(raw).map_err(|e| ToolError::InvalidArguments {
                tool: Tool::name(self).to_string(),
                reason: e.to_string(),
            })?;
        self.call(args).await
    }
}

/// Strip metadata providers reject or ignore in function schemas.
fn sanitize_schema(mut schema: Value) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }
    schema
}

/// Name → tool. The same map answers "what tools exist?" and "run this one",
/// which is the property that removes drift.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<&'static str, Arc<dyn ErasedTool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.names())
            .finish()
    }
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool. A duplicate name replaces the earlier registration.
    pub fn register<T: Tool>(&mut self, tool: T) {
        self.tools.insert(Tool::name(&tool), Arc::new(tool));
    }

    /// Builder form of [`ToolRegistry::register`].
    #[must_use]
    pub fn with<T: Tool>(mut self, tool: T) -> Self {
        self.register(tool);
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Registered names, sorted — so prompts and snapshots are deterministic.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        let mut n: Vec<_> = self.tools.keys().copied().collect();
        n.sort_unstable();
        n
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Definitions for every tool, in sorted-name order.
    #[must_use]
    pub fn definitions(&self) -> Vec<Value> {
        self.names()
            .into_iter()
            .filter_map(|n| self.tools.get(n))
            .map(|t| t.definition())
            .collect()
    }

    /// Definitions for a named subset — the tool-scoping primitive.
    ///
    /// Tool descriptions are always-on context: they sit in the prompt every
    /// turn whether called or not. Giving a planner four delegation tools
    /// instead of fifteen leaf tools is both cheaper and more accurate. Names
    /// that aren't registered are returned separately rather than dropped
    /// silently, so a typo in a scope list is visible.
    #[must_use]
    pub fn definitions_for(&self, names: &[&str]) -> (Vec<Value>, Vec<String>) {
        let mut defs = Vec::new();
        let mut missing = Vec::new();
        for n in names {
            match self.tools.get(n) {
                Some(t) => defs.push(t.definition()),
                None => missing.push((*n).to_string()),
            }
        }
        (defs, missing)
    }

    /// Dispatch a call.
    ///
    /// # Errors
    /// [`ToolError::NotFound`] for an unknown name, [`ToolError::InvalidArguments`]
    /// when `raw` doesn't match the tool's schema, or whatever the tool returns.
    pub async fn call(&self, name: &str, raw: Value) -> Result<ToolOutput, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.call_erased(raw).await
    }

    /// Dispatch, parsing `arguments` from the raw string a model sends.
    ///
    /// # Errors
    /// [`ToolError::InvalidArguments`] if the string isn't JSON — the case both
    /// codebases previously swallowed into an empty object.
    pub async fn call_raw_args(
        &self,
        name: &str,
        arguments: &str,
    ) -> Result<ToolOutput, ToolError> {
        let trimmed = arguments.trim();
        let raw: Value = if trimmed.is_empty() {
            json!({})
        } else {
            serde_json::from_str(trimmed).map_err(|e| ToolError::InvalidArguments {
                tool: name.to_string(),
                reason: e.to_string(),
            })?
        };
        self.call(name, raw).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    struct GrepArgs {
        /// Regex to search for.
        pattern: String,
        /// Cap on returned matches.
        #[serde(default)]
        limit: Option<u32>,
    }

    struct Grep;

    #[async_trait]
    impl Tool for Grep {
        type Args = GrepArgs;
        fn name(&self) -> &'static str {
            "grep"
        }
        fn description(&self) -> &'static str {
            "Regex search across the repository."
        }
        async fn call(&self, args: Self::Args) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::ok(format!(
                "grep {} limit={:?}",
                args.pattern, args.limit
            )))
        }
    }

    /// Every field defaulted — the shape where the old `unwrap_or(json!({}))`
    /// was genuinely dangerous, because the fallback *succeeds* and the tool
    /// runs with defaults instead of erroring.
    #[derive(Deserialize, JsonSchema)]
    struct LenientArgs {
        #[serde(default)]
        pattern: String,
    }

    struct Lenient;

    #[async_trait]
    impl Tool for Lenient {
        type Args = LenientArgs;
        fn name(&self) -> &'static str {
            "lenient"
        }
        fn description(&self) -> &'static str {
            "Search; pattern defaults to empty."
        }
        async fn call(&self, args: Self::Args) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::ok(format!(
                "lenient pattern={:?}",
                args.pattern
            )))
        }
    }

    #[derive(Deserialize, JsonSchema)]
    struct NoArgs {}

    struct Ping;

    #[async_trait]
    impl Tool for Ping {
        type Args = NoArgs;
        fn name(&self) -> &'static str {
            "ping"
        }
        fn description(&self) -> &'static str {
            "Returns pong."
        }
        async fn call(&self, _: Self::Args) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::ok("pong"))
        }
    }

    struct Boom;

    #[async_trait]
    impl Tool for Boom {
        type Args = NoArgs;
        fn name(&self) -> &'static str {
            "boom"
        }
        fn description(&self) -> &'static str {
            "Always fails."
        }
        async fn call(&self, _: Self::Args) -> Result<ToolOutput, ToolError> {
            Err(ToolError::Execution {
                tool: "boom".into(),
                reason: "exploded".into(),
            })
        }
    }

    fn registry() -> ToolRegistry {
        ToolRegistry::new()
            .with(Grep)
            .with(Ping)
            .with(Boom)
            .with(Lenient)
    }

    // ---- the defect this module exists to fix ------------------------------

    #[tokio::test]
    async fn malformed_arguments_are_an_error_not_a_silent_empty_object() {
        // Step 1 pinned the old behaviour: unparseable args became `{}` and the
        // tool ran with no parameters (a repo-wide empty-regex grep).
        let err = registry()
            .call_raw_args("grep", "{not json")
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidArguments { tool, .. } => assert_eq!(tool, "grep"),
            other => panic!("expected InvalidArguments, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bad_arguments_never_fall_back_to_defaults() {
        // The load-bearing regression test. A tool whose every field has a
        // default is the case where `unwrap_or(json!({}))` *succeeded* and ran
        // the tool with defaults — in pr-review-core that was `grep("")`,
        // matching every line in the repo. A tool with a required field would
        // error either way, so only this shape actually discriminates.
        let err = registry()
            .call("lenient", json!({"pattern": 42}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidArguments { .. }),
            "must not silently become pattern=\"\"; got {err:?}"
        );

        // Sanity: the defaulted call is genuinely runnable, so the assertion
        // above is testing the rejection and not an unrelated failure.
        let out = registry().call("lenient", json!({})).await.unwrap();
        assert_eq!(out.content, r#"lenient pattern="""#);
    }

    #[tokio::test]
    async fn arguments_missing_a_required_field_are_rejected() {
        let err = registry()
            .call("grep", json!({"limit": 5}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidArguments { .. }),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("pattern"),
            "names the field: {err}"
        );
    }

    #[tokio::test]
    async fn arguments_of_the_wrong_type_are_rejected() {
        let err = registry()
            .call("grep", json!({"pattern": 42}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments { .. }));
    }

    #[tokio::test]
    async fn an_unknown_tool_is_a_distinct_error() {
        let err = registry().call("nope", json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(n) if n == "nope"));
    }

    // ---- schema / dispatch cannot drift ------------------------------------

    #[test]
    fn every_advertised_tool_is_dispatchable_by_construction() {
        let r = registry();
        for def in r.definitions() {
            let name = def["function"]["name"].as_str().unwrap();
            assert!(
                r.contains(name),
                "{name} is advertised but not registered — impossible by design"
            );
        }
        assert_eq!(r.names(), vec!["boom", "grep", "lenient", "ping"]);
    }

    #[test]
    fn the_schema_is_derived_from_the_args_type() {
        let r = registry();
        let defs = r.definitions();
        let grep = defs
            .iter()
            .find(|d| d["function"]["name"] == "grep")
            .unwrap();
        let params = &grep["function"]["parameters"];

        assert_eq!(params["type"], "object");
        assert!(params["properties"]["pattern"].is_object());
        assert!(params["properties"]["limit"].is_object());
        // Derived from the type, not hand-written: `pattern` is required
        // because it is not an Option, `limit` is not because it is.
        let required: Vec<_> = params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(required, vec!["pattern"]);
        assert_eq!(
            grep["function"]["description"],
            "Regex search across the repository."
        );
    }

    #[test]
    fn schema_metadata_providers_reject_is_stripped() {
        let r = registry();
        let params = &r.definitions()[0]["function"]["parameters"];
        assert!(params.get("$schema").is_none());
        assert!(params.get("title").is_none());
    }

    // ---- dispatch ----------------------------------------------------------

    #[tokio::test]
    async fn a_valid_call_reaches_the_typed_tool() {
        let out = registry()
            .call("grep", json!({"pattern": "fn main", "limit": 5}))
            .await
            .unwrap();
        assert_eq!(out, ToolOutput::ok("grep fn main limit=Some(5)"));
    }

    #[tokio::test]
    async fn optional_fields_may_be_omitted() {
        let out = registry()
            .call("grep", json!({"pattern": "x"}))
            .await
            .unwrap();
        assert_eq!(out.content, "grep x limit=None");
    }

    #[tokio::test]
    async fn zero_argument_tools_accept_empty_null_and_absent_arguments() {
        let r = registry();
        for raw in ["", "{}", "null"] {
            let out = r.call_raw_args("ping", raw).await.unwrap();
            assert_eq!(out.content, "pong", "failed for {raw:?}");
        }
    }

    #[tokio::test]
    async fn execution_failures_are_distinguishable_from_bad_arguments() {
        let err = registry().call("boom", json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::Execution { .. }), "got {err:?}");
        // The loop can still feed it back to the model instead of aborting.
        assert!(err.to_tool_output().is_error);
    }

    // ---- scoping -----------------------------------------------------------

    #[test]
    fn scoping_exposes_a_subset_and_surfaces_typos() {
        let r = registry();
        let (defs, missing) = r.definitions_for(&["grep", "ping"]);
        assert_eq!(defs.len(), 2);
        assert!(missing.is_empty());

        // A typo in a scope list used to degrade silently to a smaller toolbelt.
        let (defs, missing) = r.definitions_for(&["grep", "grpe"]);
        assert_eq!(defs.len(), 1);
        assert_eq!(missing, vec!["grpe"]);
    }

    #[test]
    fn registering_the_same_name_twice_replaces_rather_than_duplicates() {
        let r = ToolRegistry::new().with(Ping).with(Ping);
        assert_eq!(r.len(), 1);
        assert_eq!(r.definitions().len(), 1);
    }

    #[test]
    fn definitions_are_deterministic() {
        assert_eq!(registry().definitions(), registry().definitions());
    }
}
