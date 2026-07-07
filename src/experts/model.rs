/*!
Expert Model — a single standalone pretrained checkpoint pulled from the HF
Hub (e.g. `mistralai/Mamba-Codestral-7B-v0.1`, `xl-zhao/PromptCoT-Mamba-Math-7B`,
`havenhq/mamba-chat`).

Each expert is now its own complete model rather than a LoRA delta on a
shared base — pretrained checkpoints are used as-is, no fine-tuning step.
`Quantization` on `ExpertConfig` currently only informs VRAM *planning*
(see `config.rs`); candle has no bitsandbytes-style runtime quantization for
arbitrary safetensors, so loading here happens at the checkpoint's native
dtype. True int4 requires a GGUF-converted variant — tracked as follow-up
work, not faked here.

Supports both Mamba-1 and Mamba-2 checkpoints, dispatched via `Backbone`.
This mix is real, not speculative: Codestral and the PromptCoT math model
both declare `"architectures": ["Mamba2ForCausalLM"]` in their `config.json`
(confirmed by fetching each model card directly), while `havenhq/mamba-chat`
— the only instruction-tuned SSM chat model found on HF at time of writing —
is Mamba-1 (its config.json has no `architectures` field at all, just bare
`d_model`/`n_layer`, the original state-spaces/mamba format). Architecture is
detected from `config.json` at load time rather than hardcoded per expert.
*/

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mamba::{Config as Mamba1Config, Model as Mamba1Model, State as Mamba1State};
use candle_transformers::models::mamba2::{Config as Mamba2Config, Model as Mamba2Model, State as Mamba2State};
use tokenizers::Tokenizer;

use crate::config::ExpertConfig;
use crate::memory::context::ModelState;

enum Backbone {
    Mamba1(Mamba1Model),
    Mamba2(Mamba2Model),
}

enum BackboneState {
    Mamba1(Mamba1State),
    Mamba2(Mamba2State),
}

/// Shape info needed for generation bookkeeping, independent of which
/// architecture variant is actually loaded.
enum BackboneConfig {
    Mamba1(Mamba1Config),
    Mamba2(Mamba2Config),
}

impl BackboneConfig {
    fn d_model(&self) -> usize {
        match self {
            BackboneConfig::Mamba1(c) => c.d_model,
            BackboneConfig::Mamba2(c) => c.d_model,
        }
    }

    fn n_layer(&self) -> usize {
        match self {
            BackboneConfig::Mamba1(c) => c.n_layer,
            BackboneConfig::Mamba2(c) => c.n_layer,
        }
    }
}

pub struct ExpertModel {
    pub name: String,
    backbone: Backbone,
    backbone_cfg: BackboneConfig,
    tokenizer: Tokenizer,
    lm_head: Tensor,
    device: Device,
    /// The checkpoint's native storage dtype — recurrent state tensors must
    /// match this or `forward()` hits dtype mismatches during matmuls.
    dtype: DType,
}

impl ExpertModel {
    /// Synchronous by design: called from `ExpertRegistry::activate`, which
    /// itself must stay synchronous to be callable from `pipeline.rs`'s
    /// non-async turn loop. Uses `hf_hub::api::sync::Api` (blocking downloads,
    /// cached on disk same as the tokio API) rather than an async client.
    pub fn load(cfg: &ExpertConfig, device: Device) -> Result<Self> {
        tracing::info!("Loading expert '{}': {} ({:?})", cfg.name, cfg.model_id, cfg.quantization);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(cfg.model_id.clone());

        let weights_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config_path = repo.get("config.json")?;
        let config_str = std::fs::read_to_string(config_path)?;

        let is_mamba2 = detect_mamba2(&config_str);

        // Load once, at whatever dtype the checkpoint actually stores (bf16
        // for Codestral/PromptCoT per their config.json, possibly f16/f32 for
        // older Mamba-1 checkpoints) — hardcoding F32 here would silently
        // upcast a 7B bf16 checkpoint to ~28GB instead of the real ~14GB.
        let tensors = candle_core::safetensors::load(&weights_path, &device)?;
        let native_dtype = tensors
            .values()
            .next()
            .map(|t| t.dtype())
            .ok_or_else(|| anyhow::anyhow!("empty safetensors file for expert '{}'", cfg.name))?;

        tracing::info!("Expert '{}' native dtype: {native_dtype:?}", cfg.name);

        let vb = VarBuilder::from_tensors(tensors.clone(), native_dtype, &device);

        let (backbone, backbone_cfg) = if is_mamba2 {
            let mamba_cfg: Mamba2Config = serde_json::from_str(&config_str)?;
            let model = Mamba2Model::new(&mamba_cfg, vb)?;
            (Backbone::Mamba2(model), BackboneConfig::Mamba2(mamba_cfg))
        } else {
            let mamba_cfg: Mamba1Config = serde_json::from_str(&config_str)?;
            let model = Mamba1Model::new(&mamba_cfg, vb)?;
            (Backbone::Mamba1(model), BackboneConfig::Mamba1(mamba_cfg))
        };

        // lm_head is usually tied to the input embedding in Mamba checkpoints
        let lm_head = tensors
            .get("lm_head.weight")
            .or_else(|| tensors.get("embedding.weight"))
            .or_else(|| tensors.get("backbone.embedding.weight"))
            .ok_or_else(|| anyhow::anyhow!("no lm_head or tied embedding weight found for expert '{}'", cfg.name))?
            .clone();

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed for '{}': {e}", cfg.name))?;

        Ok(Self {
            name: cfg.name.clone(),
            backbone,
            backbone_cfg,
            tokenizer,
            lm_head,
            device,
            dtype: native_dtype,
        })
    }

    fn init_state(&self) -> Result<BackboneState> {
        Ok(match &self.backbone_cfg {
            BackboneConfig::Mamba1(cfg) => {
                BackboneState::Mamba1(Mamba1State::new(1, cfg, self.dtype, &self.device)?)
            }
            BackboneConfig::Mamba2(cfg) => {
                BackboneState::Mamba2(Mamba2State::new(1, cfg, self.dtype, &self.device)?)
            }
        })
    }

    fn forward(&self, input: &Tensor, state: &mut BackboneState) -> Result<Tensor> {
        match (&self.backbone, state) {
            (Backbone::Mamba1(m), BackboneState::Mamba1(s)) => Ok(m.forward(input, s)?),
            (Backbone::Mamba2(m), BackboneState::Mamba2(s)) => Ok(m.forward(input, s)?),
            _ => unreachable!("backbone/state variant mismatch — init_state() always matches backbone"),
        }
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
    ) -> Result<(String, ModelState)> {
        let encoding = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = ids.len();

        let mut state = self.init_state()?;
        let eos_id = self.tokenizer.token_to_id("<|endoftext|>");

        for _ in 0..max_new_tokens {
            let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = self.forward(&input, &mut state)?;

            let last_hidden = hidden.i((.., hidden.dims()[1] - 1, ..))?;
            let logits = last_hidden.matmul(&self.lm_head.t()?)?.squeeze(0)?;

            let next_id = sample(&logits, temperature)?;
            ids.push(next_id);

            if Some(next_id) == eos_id {
                break;
            }
        }

        let generated_ids = &ids[prompt_len..];
        let text = self.tokenizer
            .decode(generated_ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        Ok((text, ModelState::default())) // TODO: serialise `state` once candle exposes fields
    }

    /// Rough VRAM footprint estimate at this expert's configured quantization,
    /// for capacity planning / logging — not an enforced runtime limit.
    pub fn estimated_vram_bytes(&self, cfg: &ExpertConfig) -> u64 {
        let param_count: u64 = self.backbone_cfg.n_layer() as u64
            * (self.backbone_cfg.d_model() as u64 * self.backbone_cfg.d_model().div_ceil(16) as u64 * 4);
        (param_count as f64 * cfg.quantization.bytes_per_param()) as u64
    }
}

/// Detect Mamba-2 vs Mamba-1 from a checkpoint's raw `config.json`. HF Mamba-2
/// checkpoints declare `"architectures": ["Mamba2ForCausalLM"]`; the original
/// state-spaces Mamba-1 format (used by e.g. `havenhq/mamba-chat`) has no
/// `architectures` field at all.
fn detect_mamba2(config_str: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(config_str) else {
        return false;
    };
    value
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .any(|v| v.as_str().is_some_and(|s| s.to_lowercase().contains("mamba2")))
        })
        .unwrap_or(false)
}

fn sample(logits: &Tensor, temperature: f64) -> Result<u32> {
    if temperature <= 0.0 {
        let idx = logits.argmax(0)?;
        return Ok(idx.to_scalar::<u32>()?);
    }

    let scaled = logits.affine(1.0 / temperature, 0.0)?;
    let probs = candle_nn::ops::softmax(&scaled, 0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    let r: f32 = rand_f32();
    let mut cum = 0.0;
    for (i, p) in probs_vec.iter().enumerate() {
        cum += p;
        if r <= cum {
            return Ok(i as u32);
        }
    }
    Ok((probs_vec.len() - 1) as u32)
}

fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    (nanos % 1_000_000) as f32 / 1_000_000.0
}
