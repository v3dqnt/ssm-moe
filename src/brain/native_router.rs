/*!
Native Router — a trained `LinearHead` (see `layers/linear_head.rs`) on top
of a small GGUF model's pooled prompt embedding, mapping directly to
per-expert gate logits.

This is the swap `router.rs`'s own doc comment has described since the first
commit: no Python sidecar, no BART, just a forward pass through llama.cpp
plus a trained probe — once you have a router head trained on real routing
decisions. Select it via `MoEConfig.router_backend = RouterBackend::Native`.
*/

use std::{num::NonZeroU32, path::Path};

use anyhow::Result;
use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};

use crate::{brain::router::Router, experts::model::backend, layers::linear_head::LinearHead};

pub struct NativeRouter {
    model: LlamaModel,
    head: LinearHead,
    n_ctx: u32,
    n_experts: usize,
}

impl NativeRouter {
    pub async fn load(
        gguf_repo: &str,
        gguf_file: &str,
        head_path: &Path,
        n_ctx: u32,
        n_experts: usize,
    ) -> Result<Self> {
        tracing::info!("Loading Native Router: {gguf_repo}/{gguf_file}");

        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(gguf_repo.to_string());
        let gguf_path = repo.get(gguf_file).await?;

        // Small embedding pass, same VRAM-conservation policy as the critic.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(backend(), &gguf_path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load native router model: {e}"))?;

        let head = LinearHead::load(head_path, model.n_embd() as usize)?;

        Ok(Self { model, head, n_ctx, n_experts })
    }
}

impl Router for NativeRouter {
    fn route(&mut self, prompt: &str) -> Result<Vec<f32>> {
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("native router tokenize failed: {e}"))?;
        anyhow::ensure!(!tokens.is_empty(), "nothing to embed for routing");

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        let mut ctx = self
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create native router context: {e}"))?;

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        for (i, &token) in tokens.iter().enumerate() {
            batch.add(token, i as i32, &[0], false)?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("native router embedding decode failed: {e}"))?;

        let embedding = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| anyhow::anyhow!("failed to read native router embedding: {e}"))?;

        let gate_logits = self.head.forward(embedding);
        anyhow::ensure!(
            gate_logits.len() == self.n_experts,
            "native router head produced {} outputs, expected {} (one per expert)",
            gate_logits.len(),
            self.n_experts
        );

        Ok(gate_logits)
    }
}
