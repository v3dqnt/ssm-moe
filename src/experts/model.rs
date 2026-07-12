/*!
Expert Model — a single standalone pretrained checkpoint, loaded through
llama.cpp (via `llama-cpp-2`) from a GGUF conversion pulled from the HF Hub.

This replaces the old candle-based `Backbone` enum (`Mamba1Model`/`Mamba2Model`)
and all its manual workarounds — llama.cpp's own GGUF conversion already
handles Mamba-1 vs Mamba-2 dispatch (`LlamaModel::is_recurrent()` confirms it
at load time), the embedding-key renames, sharded-weight loading, and
Codestral's untied output head, none of which need reimplementing here.

Cross-turn conversational memory is implemented with llama.cpp's own
sequence-state save/load (`LlamaContext::state_seq_save_file` /
`state_seq_load_file`) — the crate's own docs call out `PARTIAL_ONLY` state
flags as existing specifically for "recurrent cache (e.g. Mamba)", so this is
exactly the mechanism `memory/context.rs`'s old hand-rolled (and never
actually populated) `ModelState` format was trying to approximate.
*/

use std::{
    num::NonZeroU32,
    path::Path,
    sync::OnceLock,
};

use anyhow::{Context, Result};
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use crate::config::ExpertConfig;

/// A single global llama.cpp backend, shared by every expert and the critic
/// — llama.cpp expects at most one backend registry per process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("failed to init llama.cpp backend"))
}

pub struct GenerationOutput {
    pub text: String,
    /// Per-token log-probability of the sampled token (one per generated
    /// token, prompt excluded). llama.cpp doesn't expose pre-lm_head hidden
    /// states through the high-level API the way candle's private model
    /// internals almost did, so this — not a trained linear probe — is what
    /// `layers::confidence` scores against.
    pub token_logprobs: Vec<f32>,
}

pub struct ExpertModel {
    pub name: String,
    model: LlamaModel,
    n_ctx: u32,
}

impl ExpertModel {
    /// Synchronous by design: called from `ExpertRegistry::activate`, which
    /// itself must stay synchronous to be callable from `pipeline.rs`'s
    /// non-async turn loop. Uses `hf_hub::api::sync::Api` (blocking
    /// downloads, cached on disk) to fetch the `.gguf` file — same pattern
    /// as the old safetensors download, just one file instead of
    /// config.json + tokenizer.json + (sharded) weights.
    pub fn load(cfg: &ExpertConfig, n_gpu_layers: u32, n_ctx: u32) -> Result<Self> {
        tracing::info!(
            "Loading expert '{}': {}/{} ({:?})",
            cfg.name, cfg.gguf_repo, cfg.gguf_file, cfg.quantization
        );

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(cfg.gguf_repo.clone());
        let gguf_path = repo.get(&cfg.gguf_file).with_context(|| {
            format!("failed to fetch {} from {}", cfg.gguf_file, cfg.gguf_repo)
        })?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend(), &gguf_path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load expert '{}': {e}", cfg.name))?;

        tracing::info!(
            "Expert '{}' loaded: recurrent={} n_layer={} n_vocab={}",
            cfg.name,
            model.is_recurrent(),
            model.n_layer(),
            model.n_vocab()
        );

        Ok(Self { name: cfg.name.clone(), model, n_ctx })
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        state_path: &Path,
    ) -> Result<GenerationOutput> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create context for '{}': {e}", self.name))?;

        // Restore prior turns' recurrent state for this session, if any —
        // this is what makes cross-turn memory actually work now (the old
        // candle path always returned an empty, never-populated ModelState).
        let mut n_past: i32 = 0;
        if state_path.exists() {
            match ctx.state_seq_load_file(state_path, 0, self.n_ctx as usize) {
                Ok((prior_tokens, _bytes)) => {
                    n_past = prior_tokens.len() as i32;
                    tracing::debug!(
                        "Restored {n_past} prior token(s) of state for '{}'",
                        self.name
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to restore state for '{}': {e} — starting fresh",
                        self.name
                    );
                }
            }
        }

        // BOS only belongs at the very start of a sequence — don't re-add it
        // when we're continuing from restored state.
        let add_bos = if n_past == 0 { AddBos::Always } else { AddBos::Never };
        let prompt_tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .map_err(|e| anyhow::anyhow!("tokenize failed for '{}': {e}", self.name))?;

        let mut all_tokens: Vec<LlamaToken> = Vec::with_capacity(prompt_tokens.len() + max_new_tokens);
        all_tokens.extend_from_slice(&prompt_tokens);

        let mut batch = LlamaBatch::new(prompt_tokens.len().max(512), 1);
        let last_index = prompt_tokens.len() as i32 - 1;
        for (i, &token) in prompt_tokens.iter().enumerate() {
            batch.add(token, n_past + i as i32, &[0], i as i32 == last_index)?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("prompt decode failed for '{}': {e}", self.name))?;

        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature as f32),
                LlamaSampler::dist(rand_seed()),
            ])
        };

        let mut n_cur = n_past + prompt_tokens.len() as i32;
        let mut generated_tokens: Vec<LlamaToken> = Vec::with_capacity(max_new_tokens);
        let mut token_logprobs: Vec<f32> = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let idx = batch.n_tokens() - 1;
            let token = sampler.sample(&ctx, idx);
            token_logprobs.push(logprob_of(ctx.get_logits_ith(idx), token.0));
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            generated_tokens.push(token);
            all_tokens.push(token);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("decode failed for '{}': {e}", self.name))?;
        }

        #[allow(deprecated)]
        let text = self
            .model
            .tokens_to_str(&generated_tokens, Special::Plaintext)
            .map_err(|e| anyhow::anyhow!("detokenize failed for '{}': {e}", self.name))?;

        // Persist recurrent state for the next turn in this session.
        if let Err(e) = ctx.state_seq_save_file(state_path, 0, &all_tokens) {
            tracing::warn!("Failed to save state for '{}': {e}", self.name);
        }

        Ok(GenerationOutput { text, token_logprobs })
    }
}

/// log-softmax of `logits` evaluated at `token_id`, i.e. the true
/// log-probability of that token under the model's output distribution.
fn logprob_of(logits: &[f32], token_id: i32) -> f32 {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let logsumexp = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits[token_id as usize] - logsumexp
}

fn rand_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos()
}
