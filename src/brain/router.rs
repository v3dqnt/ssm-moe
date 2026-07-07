/*!
Brain Router — temporary HTTP client for the bart-large-mnli sidecar.

candle has no BART implementation, so the zero-shot classifier that scores
prompts against the expert pool runs as a small local Python process
(`router_server.py`) instead of natively in Rust. This client just posts the
prompt and gets back per-expert scores.

This is scaffolding, not the final design: once an SSM-based router is
trained and loadable directly in candle, this file is replaced with a native
forward pass and `router_server.py` is deleted. Keeping the same
`(Tensor, ModelState)` return shape here means `pipeline.rs` and `gate.rs`
don't need to change when that swap happens.
*/

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::config::MoEConfig;
use crate::memory::context::ModelState;

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
    device: Device,
}

impl BrainRouter {
    /// `config` is kept as a parameter (rather than just an endpoint string)
    /// so swapping back to a native candle model later is a signature-compatible
    /// change at call sites. Deliberately synchronous (not `async fn`): it uses
    /// `reqwest::blocking`, which must never run on a tokio executor thread.
    pub fn load(config: &MoEConfig, device: Device) -> Result<Self> {
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
            device,
        })
    }

    /// Score a prompt against the expert pool. `prior_state` is accepted for
    /// interface parity with a future native router but is unused here — the
    /// BART sidecar is stateless.
    pub fn forward(
        &mut self,
        prompt: &str,
        _prior_state: Option<ModelState>,
    ) -> Result<(Tensor, ModelState)> {
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
        let gate_logits = Tensor::new(resp.scores.as_slice(), &self.device)?;

        Ok((gate_logits, ModelState::default()))
    }
}
