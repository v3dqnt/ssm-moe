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

The old candle version scored these off a `score_head` linear layer that was
randomly zero-initialized and never trained — so it produced meaningless
numbers dressed up as a verdict. That's not something worth reproducing with
llama.cpp (which doesn't expose pre-lm_head hidden states through its
high-level API anyway). Instead:

  - coherence/completion proxy: mean per-token log-probability of the fused
    text under the critic model (teacher-forced, i.e. real perplexity) — low
    perplexity means the text reads like plausible, well-formed language to
    a language model, which is a genuine (if coarse) fluency signal, unlike
    random weights.
  - safety: a keyword denylist. This is a placeholder, not a real safety
    classifier — flagged here the same way the rest of this codebase flags
    its known-coarse approximations rather than hiding them. Swapping in an
    instruct-tuned critic GGUF for prompted self-critique is a viable future
    upgrade if this proves too coarse; the base critic model here (e.g.
    mamba-130m) isn't instruction-tuned, so prompting it to self-rate isn't
    reliable.
*/

use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};

use crate::experts::model::backend;

pub struct CriticModel {
    model: LlamaModel,
    n_ctx: u32,
    threshold: f32,
}

#[derive(Debug, Clone)]
pub struct CriticVerdict {
    pub coherence: f32,
    pub completion: f32,
    pub safety: f32,
    pub composite: f32,
    pub passed: bool,
}

/// Coarse safety denylist — a placeholder, not a real classifier. Swap for
/// an actual moderation model/API before relying on this for anything.
const UNSAFE_MARKERS: &[&str] = &["how to make a bomb", "kill yourself"];

impl CriticModel {
    pub async fn load(gguf_repo: &str, gguf_file: &str, threshold: f32, n_ctx: u32) -> Result<Self> {
        tracing::info!("Loading Critic: {gguf_repo}/{gguf_file}");

        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(gguf_repo.to_string());
        let gguf_path = repo.get(gguf_file).await?;

        // Deliberately forced to CPU (n_gpu_layers=0) — the critic runs once
        // per turn on a small model, cheap enough on CPU, and keeping it off
        // the GPU leaves that VRAM for experts. Same policy as the old
        // candle version, which hardcoded `Device::Cpu` regardless of what
        // was requested.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(backend(), &gguf_path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load critic: {e}"))?;

        Ok(Self { model, n_ctx, threshold })
    }

    pub fn verify(&mut self, output_text: &str) -> Result<CriticVerdict> {
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

/// log-softmax of `logits` evaluated at `token_id` — same helper as
/// `experts::model::logprob_of`, duplicated rather than shared since this
/// module has no other dependency on `experts::model`.
fn logprob_of(logits: &[f32], token_id: i32) -> f32 {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let logsumexp = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits[token_id as usize] - logsumexp
}
