/*!
Critic / Verifier Layer.

Small always-CPU-resident GGUF model that reads the fused output and scores
it on:
  - coherence   (does the output hang together?)
  - completion  (does it answer the actual request?)
  - safety      (does it contain harmful content?)

Returns a composite score in [0, 1].
  score >= threshold → pass to user
  score <  threshold → signal re-route back to Brain with adjusted gates

Two scoring paths:

  - **Trained** (`reward_head: Some`): pool the fused text's embedding via
    llama.cpp (`LlamaContext::embeddings_seq_ith` with mean pooling) and run
    it through a trained `LinearHead` (see `layers/linear_head.rs`) producing
    the three scores directly. This is the real thing — a probe trained on
    actual labeled data, unlike the old candle version's `score_head`, which
    was a randomly zero-initialized linear layer that was never trained and
    produced meaningless numbers dressed up as a verdict.
  - **Heuristic** (`reward_head: None`, e.g. before a head is trained):
    coherence/completion come from mean per-token log-probability of the
    fused text under the critic model (teacher-forced perplexity — a genuine
    if coarse fluency signal), safety comes from a keyword denylist
    placeholder. The system stays runnable throughout training on this path.
*/

use std::{num::NonZeroU32, path::Path};

use anyhow::{bail, Result};
use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};

use crate::{experts::model::backend, layers::linear_head::LinearHead};

pub struct CriticModel {
    model: LlamaModel,
    n_ctx: u32,
    threshold: f32,
    reward_head: Option<LinearHead>,
}

#[derive(Debug, Clone)]
pub struct CriticVerdict {
    pub coherence: f32,
    pub completion: f32,
    pub safety: f32,
    pub composite: f32,
    pub passed: bool,
}

/// Coarse safety denylist — a placeholder, not a real classifier. Only used
/// on the heuristic (untrained) path. Swap for an actual moderation
/// model/API before relying on this for anything.
const UNSAFE_MARKERS: &[&str] = &["how to make a bomb", "kill yourself"];

impl CriticModel {
    pub async fn load(
        gguf_repo: &str,
        gguf_file: &str,
        threshold: f32,
        n_ctx: u32,
        head_path: Option<&Path>,
    ) -> Result<Self> {
        tracing::info!("Loading Critic: {gguf_repo}/{gguf_file}");

        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(gguf_repo.to_string());
        let gguf_path = repo.get(gguf_file).await?;

        // Deliberately forced to CPU (n_gpu_layers=0) — the critic runs once
        // per turn on a small model, cheap enough on CPU, and keeping it off
        // the GPU leaves that VRAM for experts.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(backend(), &gguf_path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load critic: {e}"))?;

        let reward_head = match head_path {
            Some(p) => {
                let head = LinearHead::load(p, model.n_embd() as usize)?;
                tracing::info!("Loaded trained critic reward head from {}", p.display());
                Some(head)
            }
            None => {
                tracing::info!("No critic_head_path configured — using heuristic scoring");
                None
            }
        };

        Ok(Self { model, n_ctx, threshold, reward_head })
    }

    pub fn verify(&mut self, output_text: &str) -> Result<CriticVerdict> {
        match &self.reward_head {
            Some(head) => self.verify_with_head(output_text, head),
            None => self.verify_heuristic(output_text),
        }
    }

    /// Trained path: pool a mean embedding of the text and run the trained
    /// head. A *separate* context from the heuristic path's, configured for
    /// embeddings output — combining `embeddings=true` with normal
    /// causal-logits decoding in one context isn't a combination this
    /// crate's docs confirm as supported, and the critic model is small/
    /// CPU-only, so the extra context is cheap.
    fn verify_with_head(&self, output_text: &str, head: &LinearHead) -> Result<CriticVerdict> {
        let tokens = self
            .model
            .str_to_token(output_text, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("critic tokenize failed: {e}"))?;
        if tokens.is_empty() {
            bail!("nothing to embed for critic scoring");
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        let mut ctx = self
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create critic embedding context: {e}"))?;

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        for (i, &token) in tokens.iter().enumerate() {
            // No per-token logits needed here — mean pooling covers the
            // whole sequence regardless of the logits flag.
            batch.add(token, i as i32, &[0], false)?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("critic embedding decode failed: {e}"))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("failed to read critic embedding: {e}"))?;

        let raw = head.forward(embedding);
        if raw.len() != 3 {
            bail!(
                "critic reward head produced {} outputs, expected 3 (coherence, completion, safety)",
                raw.len()
            );
        }

        let coherence = sigmoid(raw[0]);
        let completion = sigmoid(raw[1]);
        let safety = sigmoid(raw[2]);
        // safety weighted 2× — a single unsafe output is a hard fail
        let composite = (coherence + completion + 2.0 * safety) / 4.0;

        Ok(CriticVerdict {
            coherence,
            completion,
            safety,
            composite,
            passed: composite >= self.threshold,
        })
    }

    /// Heuristic fallback path — used until a reward head is trained.
    fn verify_heuristic(&mut self, output_text: &str) -> Result<CriticVerdict> {
        let tokens = self
            .model
            .str_to_token(output_text, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("critic tokenize failed: {e}"))?;

        // Too short to score a real continuation probability — soft-fail
        // rather than fabricate a number from nothing.
        if tokens.len() < 2 {
            return Ok(CriticVerdict {
                coherence: 0.5,
                completion: 0.0,
                safety: safety_heuristic(output_text),
                composite: 0.25,
                passed: false,
            });
        }

        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create critic context: {e}"))?;

        // Request logits at every position (not just the last) so we can
        // teacher-force score every token against the model's prediction
        // for it — that's what makes this a real perplexity/fluency signal.
        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], true)?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("critic decode failed: {e}"))?;

        let mut logprobs = Vec::with_capacity(tokens.len() - 1);
        for i in 0..tokens.len() - 1 {
            let logits = ctx.get_logits_ith(i as i32);
            logprobs.push(logprob_of(logits, tokens[i + 1].0));
        }

        let mean_logprob = logprobs.iter().sum::<f32>() / logprobs.len() as f32;
        let fluency = mean_logprob.exp().clamp(0.0, 1.0);

        let coherence = fluency;
        let completion = completion_heuristic(output_text);
        let safety = safety_heuristic(output_text);

        // safety weighted 2× — a single unsafe output is a hard fail
        let composite = (coherence + completion + 2.0 * safety) / 4.0;

        Ok(CriticVerdict {
            coherence,
            completion,
            safety,
            composite,
            passed: composite >= self.threshold,
        })
    }
}

fn completion_heuristic(text: &str) -> f32 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0.0;
    }
    let ends_cleanly = trimmed.ends_with(['.', '!', '?', '`', '"', ')']);
    let long_enough = trimmed.len() >= 8;
    match (ends_cleanly, long_enough) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.6,
        (false, false) => 0.3,
    }
}

fn safety_heuristic(text: &str) -> f32 {
    let lower = text.to_lowercase();
    if UNSAFE_MARKERS.iter().any(|m| lower.contains(m)) {
        0.0
    } else {
        1.0
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// log-softmax of `logits` evaluated at `token_id` — same helper as
/// `experts::model::logprob_of`, duplicated rather than shared since this
/// module has no other dependency on `experts::model`.
fn logprob_of(logits: &[f32], token_id: i32) -> f32 {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let logsumexp = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits[token_id as usize] - logsumexp
}
