/*!
Brain Router — trait plus the BART sidecar implementation.

`Router` is the seam this file's own history has been pointing at since the
first commit: `BartSidecarRouter` wraps the temporary HTTP client to the
`bart-large-mnli` zero-shot classifier (`router_server.py`) — neither candle
nor llama.cpp has a BART implementation, so it runs as a small local Python
process instead of natively in Rust. `NativeRouter`
(`brain/native_router.rs`) is the trained, no-Python-required alternative
once a router head exists; `MoEConfig.router_backend` picks between them.
*/

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::MoEConfig;

/// Scores a prompt against the expert pool, returning per-expert gate
/// logits. `Vec<f32>` is a plain, framework-agnostic representation on
/// purpose — neither implementation needs to agree on a tensor type.
pub trait Router: Send {
    fn route(&mut self, prompt: &str) -> Result<Vec<f32>>;
}

#[derive(Serialize)]
struct RouteRequest<'a> {
    prompt: &'a str,
}

#[derive(Deserialize)]
struct RouteResponse {
    #[allow(dead_code)]
    labels: Vec<String>,
    scores: Vec<f32>,
}

pub struct BartSidecarRouter {
    http: reqwest::blocking::Client,
    endpoint: String,
    n_experts: usize,
}

impl BartSidecarRouter {
    /// `config` is kept as a parameter (rather than just an endpoint string)
    /// for parity with `NativeRouter::load`. Deliberately synchronous (not
    /// `async fn`): it uses `reqwest::blocking`, which must never run on a
    /// tokio executor thread.
    pub fn load(config: &MoEConfig) -> Result<Self> {
        let endpoint = std::env::var("BRAIN_ROUTER_ENDPOINT")
            .unwrap_or_else(|_| "http://127.0.0.1:8008".to_string());

        tracing::info!("Using Brain Router sidecar at {endpoint} ({})", config.brain_model_id);

        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        // fail fast if the sidecar isn't running — better than a confusing
        // timeout on the first real prompt
        let health_url = format!("{endpoint}/health");
        http.get(&health_url)
            .send()
            .with_context(|| {
                format!(
                    "Brain Router sidecar not reachable at {health_url}. \
                     Start it with: python router_server.py"
                )
            })?;

        Ok(Self {
            http,
            endpoint,
            n_experts: config.n_experts(),
        })
    }
}

impl Router for BartSidecarRouter {
    /// The BART sidecar is stateless, so unlike the experts/critic there's
    /// no cross-turn state to load or save here.
    fn route(&mut self, prompt: &str) -> Result<Vec<f32>> {
        let url = format!("{}/route", self.endpoint);
        let resp: RouteResponse = self
            .http
            .post(&url)
            .json(&RouteRequest { prompt })
            .send()
            .context("Brain Router sidecar request failed")?
            .json()
            .context("Brain Router sidecar returned malformed JSON")?;

        anyhow::ensure!(
            resp.scores.len() == self.n_experts,
            "sidecar returned {} scores, expected {}",
            resp.scores.len(),
            self.n_experts
        );

        // these are already per-label sigmoid scores (multi_label=True on the
        // sidecar), not raw logits — adaptive_k_gate applies its own softmax,
        // which is a reasonable enough approximation for gating purposes here
        Ok(resp.scores)
    }
}
