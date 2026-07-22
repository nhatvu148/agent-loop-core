//! Print the exact `/responses` request body this crate builds for a realistic
//! multi-turn transcript — system prompt, user turn, an assistant turn that
//! called a tool, and the tool's result.
//!
//! Used to verify the *request* side of the Responses translation against the
//! live API: pipe this straight to `curl`. Parsing is covered by fixtures, but
//! nothing else proves the API accepts what we generate.
//!
//! ```sh
//! cargo run --quiet --example dump_responses_request > /tmp/req.json
//! curl -s https://api.openai.com/v1/responses \
//!   -H "Authorization: Bearer $OPENAI_API_KEY" \
//!   -H "Content-Type: application/json" -d @/tmp/req.json
//! ```

use agent_loop_core::{ModelPolicy, Responses, WireFormat, WireRequest};
use serde_json::json;

fn main() {
    let messages = vec![
        json!({"role": "system", "content": "You are a terse assistant."}),
        json!({"role": "user", "content": "What is the weather in Paris?"}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_seed_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]
        }),
        json!({"role": "tool", "tool_call_id": "call_seed_1", "content": "18C, raining"}),
    ];

    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];

    let mut policy = ModelPolicy::single("gpt-5.6-luna");
    policy.max_tokens = 200;
    let mut extra = serde_json::Map::new();
    extra.insert("parallel_tool_calls".into(), json!(false));

    let body = Responses::new().build_request(WireRequest {
        model: "gpt-5.6-luna",
        messages: &messages,
        tools: &tools,
        tool_choice: "auto",
        policy: &policy,
        extra_body: &extra,
    });

    println!("{}", serde_json::to_string_pretty(&body).unwrap());
}
