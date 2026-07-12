/*!
Brain Router — temporary HTTP client for the bart-large-mnli sidecar.

Neither candle nor llama.cpp has a BART implementation, so the zero-shot
classifier that scores prompts against the expert pool runs as a small local
Python process (`router_server.py`) instead of natively in Rust. This client
just posts the prompt and gets back per-expert scores.

This is scaffolding, not the final design: once an SSM-based router is
trained and loadable directly, this file is replaced with a native forward
pass and `router_server.py` is deleted. The `Vec<f32>` return type is a
plain, framework-agnostic gate-logits vector for exactly that reason — it
doesn't need to change when that swap happens, unlike a `Tensor` tied to a
specific inference framework.
*/

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::MoEConfig;

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

pub struct BrainRouter {
    http: reqwest::blocking::Client,
    endpoint: String,
    n_experts: usize,
}

impl BrainRouter {
    /// `config` is kept as a parameter (rather than just an endpoint string)
    /// so swapping back to a native model later is a signature-compatible
    /// change at call sites. Deliberately synchronous (not `async fn`): it uses
    /// `reqwest::blocking`, which must never run on a tokio executor thread.
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

    /// Score a prompt against the expert pool. The BART sidecar is
    /// stateless, so unlike the experts/critic there's no cross-turn state
    /// to load or save here.
    pub fn forward(&mut self, prompt: &str) -> Result<Vec<f32>> {
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
