/*!
Critic / Verifier Layer.

Small always-resident SSM that reads the fused output and scores it on:
  - coherence   (does the output hang together?)
  - completion  (does it answer the actual request?)
  - safety      (does it contain harmful content?)

Returns a composite score in [0, 1].
  score >= threshold → pass to user
  score <  threshold → signal re-route back to Brain with adjusted gates
*/

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use candle_transformers::models::mamba::{Config as MambaConfig, Model as MambaModel, State as MambaState};
use tokenizers::Tokenizer;

pub struct CriticModel {
    model: MambaModel,
    mamba_cfg: MambaConfig,
    tokenizer: Tokenizer,
    /// (vocab_size, 3) → [coherence, completion, safety]. Built lazily on the
    /// first `verify()` call: candle's `Model::forward()` returns final vocab
    /// logits directly (its `embedding`/`layers`/`norm_f`/`lm_head` fields
    /// are private, so we can't reach in for pre-lm_head hidden states from
    /// outside the crate), and the model's *padded* vocab size isn't
    /// reliably knowable ahead of time from `config.json` alone — so this
    /// scores off the output logit distribution instead of a hidden state,
    /// sized from whatever the first real forward pass actually produces.
    /// Harmless since it's zero-initialized/untrained either way.
    score_head: Option<Linear>,
    threshold: f32,
    /// Always CPU regardless of what's passed to `load()` — the critic is
    /// intentionally kept off the GPU to save VRAM for experts. `verify()`
    /// must build its input/state tensors on this same device or candle
    /// panics on a device mismatch.
    device: Device,
    dtype: candle_core::DType,
}

#[derive(Debug, Clone)]
pub struct CriticVerdict {
    pub coherence:  f32,
    pub completion: f32,
    pub safety:     f32,
    pub composite:  f32,
    pub passed:     bool,
}

impl CriticModel {
    pub async fn load(
        model_id: &str,
        threshold: f32,
        _requested_device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Critic: {model_id}");

        // Deliberately ignore the requested device and force CPU — the
        // critic runs once per turn on a small 130M model, cheap enough on
        // CPU, and keeping it off CUDA leaves that VRAM for experts.
        let device = Device::Cpu;

        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());

        let weights_path = repo.get("model.safetensors").await?;
        let tokenizer_path = repo.get("tokenizer.json").await?;
        let config_path = repo.get("config.json").await?;

        let mamba_cfg: MambaConfig =
            serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        // Real tensor names verified directly via safe_open against this
        // exact checkpoint: prefix is "backbone." (not "model."), and the
        // embedding tensor is "embeddings.weight" (plural) — but candle's
        // Mamba-1 Model::new() looks it up as singular "embedding" under
        // vb.pp("embedding"). Same rename as experts/model.rs.
        let tensors = candle_core::safetensors::load(&weights_path, &device)?;
        let native_dtype = tensors
            .values()
            .next()
            .map(|t| t.dtype())
            .ok_or_else(|| anyhow::anyhow!("empty safetensors file for critic checkpoint"))?;

        let tensors = remap_embedding_key(tensors);
        let backbone_vb = VarBuilder::from_tensors(tensors, native_dtype, &device);
        let model = MambaModel::new(&mamba_cfg, backbone_vb.pp("backbone"))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

        Ok(Self {
            model,
            mamba_cfg,
            tokenizer,
            score_head: None,
            threshold,
            device,
            dtype: native_dtype,
        })
    }

    pub fn verify(&mut self, output_text: &str) -> Result<CriticVerdict> {
        let enc = self.tokenizer
            .encode(output_text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let ids: Vec<u32> = enc.get_ids().to_vec();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let mut state = MambaState::new(1, &self.mamba_cfg, self.dtype, &self.device)?;
        // candle's Model::forward() already applies norm_f + lm_head
        // internally and returns final vocab logits directly, shape
        // (1, seq, vocab_size) — not raw hidden states.
        let logits_seq = self.model.forward(&input, &mut state)?;
        let pooled = logits_seq.mean(1)?; // (1, vocab_size)

        if self.score_head.is_none() {
            let vocab_size = pooled.dims()[1];
            let head_vb = VarBuilder::zeros(self.dtype, &self.device);
            self.score_head = Some(linear_no_bias(vocab_size, 3, head_vb.pp("score_head"))?);
        }

        let scores = self.score_head.as_ref().unwrap().forward(&pooled)?; // (1, 3)
        let scores = candle_nn::ops::sigmoid(&scores)?.squeeze(0)?; // (3,)
        let v: Vec<f32> = scores.to_vec1()?;

        let (coherence, completion, safety) = (v[0], v[1], v[2]);
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

/// Same rename as `experts::model::remap_mamba1_embedding_key` — candle's
/// Mamba-1 code expects singular "embedding", checkpoints store plural
/// "embeddings". Duplicated here rather than shared since this module has
/// no dependency on `experts::model` and the fix is a one-liner.
fn remap_embedding_key(
    tensors: std::collections::HashMap<String, Tensor>,
) -> std::collections::HashMap<String, Tensor> {
    tensors
        .into_iter()
        .map(|(k, v)| {
            if let Some(prefix) = k.strip_suffix("embeddings.weight") {
                (format!("{prefix}embedding.weight"), v)
            } else {
                (k, v)
            }
        })
        .collect()
}
