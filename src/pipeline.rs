/*!
SSM MoE Pipeline — top-level orchestrator.

Full forward pass:
  Prompt
    → Decomposition (TODO: implement sub-task splitting)
    → Brain Router   (gate logits)
    → Adaptive-K Gate (select K experts)
    → Registry       (promote experts to hot tier)
    → Expert forward passes (llama.cpp, one GGUF model per expert)
    → Confidence check (logs low-confidence experts)
    → Fusion         (text-level weighted blend)
    → Critic         (verify or re-route)
    → Per-expert state write (session memory for next turn)
    → Output
*/

use anyhow::Result;
use tracing::{info, warn};

use crate::{
    brain::{gate::adaptive_k_gate, router::BrainRouter},
    config::MoEConfig,
    experts::registry::ExpertRegistry,
    layers::{confidence, critic::CriticModel, fusion::text_fuse},
    memory::context::ContextMemory,
};

pub struct MoEPipeline {
    config: MoEConfig,
    brain: BrainRouter,
    registry: ExpertRegistry,
    critic: CriticModel,
    context: ContextMemory,
    /// Max re-route attempts before returning best effort output
    max_retries: usize,
}

impl MoEPipeline {
    pub async fn new(config: MoEConfig, session_id: &str) -> Result<Self> {
        // llama.cpp handles CPU/GPU placement per-model via n_gpu_layers,
        // not a candle-style Device enum: 0 keeps everything on CPU, a large
        // number offloads every layer of the model to the GPU.
        let n_gpu_layers: u32 = if cfg!(feature = "cuda") { 1000 } else { 0 };
        info!("Initialising SSM MoE pipeline (n_gpu_layers={n_gpu_layers})");

        let brain = BrainRouter::load(&config)?;

        let registry = ExpertRegistry::new(
            config.adapters_dir.clone(),
            config.warm_cache_size,
            n_gpu_layers,
            config.n_ctx,
        )?;

        // Register each expert's GGUF checkpoint config. Nothing is
        // downloaded or constructed until the gate actually selects it.
        for expert in &config.experts {
            registry.register_cold(&expert.name, expert.clone())?;
        }

        // Critic is always forced to CPU inside `CriticModel::load`
        // regardless of `n_gpu_layers`, to leave GPU VRAM for experts.
        let critic = CriticModel::load(
            &config.critic_gguf_repo,
            &config.critic_gguf_file,
            config.critic_threshold,
            config.n_ctx,
        )
        .await?;

        let context = ContextMemory::new(session_id, config.sessions_dir.clone())?;

        Ok(Self {
            config,
            brain,
            registry,
            critic,
            context,
            max_retries: 2,
        })
    }

    pub fn run(&mut self, prompt: &str) -> Result<String> {
        let mut attempt = 0;

        loop {
            match self.try_run(prompt) {
                Ok((output, verdict)) if verdict.passed => {
                    info!("Critic passed (score={:.2})", verdict.composite);
                    return Ok(output);
                }
                Ok((output, verdict)) if attempt >= self.max_retries => {
                    warn!(
                        "Critic failed after {attempt} retries (score={:.2}), returning best effort",
                        verdict.composite
                    );
                    return Ok(output);
                }
                Ok((_, verdict)) => {
                    warn!(
                        "Critic failed (score={:.2}), re-routing (attempt {})",
                        verdict.composite,
                        attempt + 1
                    );
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn try_run(&mut self, prompt: &str) -> Result<(String, crate::layers::critic::CriticVerdict)> {
        // 1. Brain Router → gate logits. The BART sidecar is stateless, so
        // unlike experts/critic there's no cross-turn state to load here.
        let gate_logits = self.brain.forward(prompt)?;

        // 2. Adaptive-K Gate → selected experts
        let gate = adaptive_k_gate(&gate_logits, self.config.k_max, 0.05);
        let selected_names: Vec<&str> = gate
            .expert_indices
            .iter()
            .map(|&i| self.config.expert_names()[i])
            .collect();

        info!(
            "Gate selected {} expert(s): {:?} (entropy={:.2})",
            gate.k, selected_names, gate.entropy
        );

        // 3. Promote selected experts, hibernate rest
        self.registry.evict_except(&selected_names);
        for name in &selected_names {
            self.registry.activate(name)?;
        }

        // 4. Run each active expert. Single active expert is the common case
        // (adaptive-K biases toward k=1 for confident routing, and config
        // defaults k_max=1 for the 8GB VRAM budget) so this is usually one pass.
        let mut expert_outputs: Vec<(String, String)> = Vec::with_capacity(selected_names.len());
        let mut confidence_scores: Vec<(String, f32)> = Vec::with_capacity(selected_names.len());
        for name in &selected_names {
            let expert = self.registry.activate(name)?;
            let state_path = self.context.path(name);
            let generation = expert.generate(
                prompt,
                self.config.max_new_tokens,
                self.config.temperature,
                &state_path,
            )?;

            let score = confidence::score(&generation.token_logprobs);
            confidence_scores.push((name.to_string(), score));
            expert_outputs.push((name.to_string(), generation.text));
        }

        let low_confidence = confidence::check_confidence(&confidence_scores, self.config.confidence_threshold);
        if !low_confidence.is_empty() {
            warn!("Low-confidence expert output(s): {low_confidence:?}");
        }

        // 5. Fuse (text-level — see layers/fusion.rs doc comment)
        let fused = text_fuse(&expert_outputs, &gate.expert_weights);

        // 6. Critic verification
        let verdict = self.critic.verify(&fused)?;

        Ok((fused, verdict))
    }
}
