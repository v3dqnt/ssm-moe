/*!
OpenAI-compatible HTTP server — the plug-in point for Vivianne.

Vivianne's provider layer (packages/ai/src/providers directory,
packages/ai/src/api/openai-* files) already knows how to talk to an
OpenAI-compatible chat completions endpoint against a custom base URL. This
server exposes exactly that shape, so the whole SSM MoE pipeline (Brain
sidecar → adaptive-K gate → expert router → critic) can be registered as a
Vivianne provider the same way any other OpenAI-compatible endpoint would be
— no Vivianne-side code changes needed, just point its base URL here.

One shared `MoEPipeline` instance serves every request. This is deliberate:
the expensive resources (the Brain sidecar HTTP client, the persistent
`llama-server` router process, the Critic model) are constructed once at
server startup, not per-session — see `MoEPipeline::new()`'s doc comment.
Per-conversation state (the SSM context memory) is still isolated per
session, just constructed cheaply on each call rather than held as
long-lived pipeline state.

Concurrency note: requests are served one at a time behind a
`tokio::sync::Mutex`. This isn't a corner cut for expediency — the pipeline
only ever has one expert "hot" at a time by design (`k_max: 1`, an 8GB VRAM
budget), so true parallel request handling wouldn't help throughput; it
would just contend for the same GPU. Serializing requests here is the
correct behavior for this hardware target, not a placeholder for something
better.
*/

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;

use crate::pipeline::MoEPipeline;

#[derive(Clone)]
struct AppState {
    pipeline: Arc<Mutex<MoEPipeline>>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[allow(dead_code)] // accepted for OpenAI-client compatibility, not used for routing
    model: Option<String>,
    messages: Vec<ChatMessage>,
    /// OpenAI convention for a stable per-caller/session identifier. Used
    /// here as the session id for context-memory isolation. Falls back to
    /// the `x-session-id` header, then "default", if absent.
    user: Option<String>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<ChatChoice>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessageOut,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatMessageOut {
    role: &'static str,
    content: String,
}

pub async fn serve(pipeline: MoEPipeline, port: u16) -> Result<()> {
    let state = AppState { pipeline: Arc::new(Mutex::new(pipeline)) };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("SSM MoE OpenAI-compatible server listening on http://{addr}");
    tracing::info!("Point Vivianne's provider base URL at http://127.0.0.1:{port}/v1");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn list_models() -> impl IntoResponse {
    // Static single-entry list — matches what OpenAI-compatible clients
    // (including Vivianne's model registry, if it probes this) expect to
    // see before picking a model id to request with.
    Json(json!({
        "object": "list",
        "data": [{
            "id": "ssm-moe",
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let session_id = req
        .user
        .clone()
        .or_else(|| {
            headers
                .get("x-session-id")
                .and_then(|v| v.to_str().ok())
                .map(String::from)
        })
        .unwrap_or_else(|| "default".to_string());

    // Known simplification: only the last user message becomes the prompt.
    // Multi-turn continuity is carried by the SSM context memory (per
    // session_id), not by replaying the full message history through the
    // gate/expert on every turn — that's the whole point of the fixed-size
    // recurrent state design. Revisit if an expert needs the full transcript
    // (e.g. for in-context few-shot prompting) rather than relying on state.
    let prompt = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "no user message in request".to_string()))?;

    tracing::info!("Request for session '{session_id}': {} chars", prompt.len());

    let pipeline = state.pipeline.clone();
    let output = tokio::task::spawn_blocking(move || {
        // MoEPipeline::run is synchronous (uses reqwest::blocking internally
        // for both the Brain sidecar and the expert router) — run it on a
        // blocking-pool thread rather than the async executor. The lock is
        // acquired via blocking_lock() inside the same closure since we're
        // already off the async runtime here.
        let mut guard = pipeline.blocking_lock();
        guard.run(&session_id, &prompt)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("task join error: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("pipeline error: {e}")))?;

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid_like()),
        object: "chat.completion",
        created,
        model: "ssm-moe",
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut { role: "assistant", content: output },
            finish_reason: "stop",
        }],
    }))
}

/// Not a real UUID — just enough entropy to make response ids look distinct
/// in logs, without pulling in a uuid crate dependency for this alone.
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("{nanos:x}")
}
