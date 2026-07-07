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
    score_head: Linear,  // (hidden_size, 3) → [coherence, completion, safety]
    threshold: f32,
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
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Critic: {model_id}");

        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());

        let weights_path = repo.get("model.safetensors").await?;
        let tokenizer_path = repo.get("tokenizer.json").await?;
        let config_path = repo.get("config.json").await?;

        let mamba_cfg: MambaConfig =
            serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let hidden_size = mamba_cfg.d_model;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                candle_core::DType::F32,
                &device,
            )?
        };
        let model = MambaModel::new(&mamba_cfg, vb.pp("model"))?;
        let score_head = linear_no_bias(hidden_size, 3, vb.pp("score_head"))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

        Ok(Self { model, mamba_cfg, tokenizer, score_head, threshold })
    }

    pub fn verify(&mut self, output_text: &str) -> Result<CriticVerdict> {
        let enc = self.tokenizer
            .encode(output_text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let ids: Vec<u32> = enc.get_ids().to_vec();
        let device = candle_core::Device::Cpu; // critic stays on CPU to save VRAM
        let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

        let mut state = MambaState::new(1, &self.mamba_cfg, candle_core::DType::F32, &device)?;
        let hidden = self.model.forward(&input, &mut state)?;
        let pooled = hidden.mean(1)?;  // (1, hidden_size)
        let scores = self.score_head.forward(&pooled)?; // (1, 3)
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
