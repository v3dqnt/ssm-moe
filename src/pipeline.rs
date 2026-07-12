/*!
SSM MoE Pipeline — top-level orchestrator.

Full forward pass:
  Prompt
    → Decomposition (TODO: implement sub-task splitting)
    → Brain Router   (gate logits)
    → Adaptive-K Gate (select K experts)
    → Registry       (promote experts to hot tier)
    → Expert forward passes (llama.cpp, one GGUF model per expert,
                              best_of_n candidates reranked by the Critic)
    → Confidence check (activates the next-ranked unselected expert as
                        backup for any low-confidence output)
    → Fusion         (text-level weighted blend)
    → Critic         (verify or re-route)
    → Per-expert state write (session memory for next turn)
    → Output
*/

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use tracing::{info, warn};

use crate::{
    brain::{
        gate::adaptive_k_gate,
        native_router::NativeRouter,
        router::{BartSidecarRouter, Router},
    },
    config::{MoEConfig, RouterBackend},
    experts::{model::GenerationOutput, registry::ExpertRegistry},
    layers::{confidence, critic::CriticModel, fusion::text_fuse},
    memory::context::ContextMemory,
};

pub struct MoEPipeline {
    config: MoEConfig,
    brain: Box<dyn Router>,
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

        let brain: Box<dyn Router> = match config.router_backend {
            RouterBackend::BartSidecar => Box::new(BartSidecarRouter::load(&config)?),
            RouterBackend::Native => {
                let head_path = config.native_router_head_path.as_deref().context(
                    "router_backend = Native requires native_router_head_path to be set",
                )?;
                Box::new(
                    NativeRouter::load(
                        &config.native_router_gguf_repo,
                        &config.native_router_gguf_file,
                        head_path,
                        config.n_ctx,
                        config.n_experts(),
                    )
                    .await?,
                )
            }
        };

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
            config.critic_head_path.as_deref(),
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
        let gate_logits = self.brain.route(prompt)?;

        // 2. Adaptive-K Gate → selected experts
        let gate = adaptive_k_gate(&gate_logits, self.config.k_max, 0.05);
        // Owned, not `Vec<&str>` borrowing `self.config`: the main
        // generation loop below calls `self.generate_best_of_n`, which
        // needs `&mut self`, and a live borrow of `self.config` here would
        // conflict with that.
        let selected_names: Vec<String> = gate
            .expert_indices
            .iter()
            .map(|&i| self.config.expert_names()[i].to_string())
            .collect();

        info!(
            "Gate selected {} expert(s): {:?} (entropy={:.2})",
            gate.k, selected_names, gate.entropy
        );

        // 3. Promote selected experts, hibernate rest
        let selected_refs: Vec<&str> = selected_names.iter().map(String::as_str).collect();
        self.registry.evict_except(&selected_refs);
        for name in &selected_refs {
            self.registry.activate(name)?;
        }

        // 4. Run each active expert. Single active expert is the common case
        // (adaptive-K biases toward k=1 for confident routing, and config
        // defaults k_max=1 for the 8GB VRAM budget) so this is usually one pass.
        // `generate_best_of_n` reproduces a single generation exactly when
        // `config.best_of_n == 1` (the default).
        let mut expert_outputs: Vec<(String, String)> = Vec::with_capacity(selected_names.len());
        let mut fusion_weights: Vec<f32> = Vec::with_capacity(selected_names.len());
        let mut confidence_scores: Vec<(String, f32)> = Vec::with_capacity(selected_names.len());
        for (name, &weight) in selected_names.iter().zip(gate.expert_weights.iter()) {
            let generation = self.generate_best_of_n(name, prompt)?;

            let score = confidence::score(&generation.token_logprobs);
            confidence_scores.push((name.to_string(), score));
            expert_outputs.push((name.to_string(), generation.text));
            fusion_weights.push(weight);
        }

        // 4b. Backup expert(s): for anything below confidence_threshold,
        // pull the next-ranked *unselected* candidate off the gate's full
        // ranking and run it too, folding it into fusion alongside the
        // shaky output so the Critic sees both.
        let low_confidence =
            confidence::check_confidence(&confidence_scores, self.config.confidence_threshold);
        if !low_confidence.is_empty() {
            let expert_names_all = self.config.expert_names();
            let mut already_used: std::collections::HashSet<usize> =
                gate.expert_indices.iter().copied().collect();

            for low_name in &low_confidence {
                let Some(&(backup_idx, backup_weight)) =
                    gate.all_ranked.iter().find(|(idx, _)| !already_used.contains(idx))
                else {
                    warn!("Low-confidence '{low_name}' but no backup expert candidate remains");
                    continue;
                };
                already_used.insert(backup_idx);

                let backup_name = expert_names_all[backup_idx];
                warn!("Low-confidence '{low_name}' — activating backup '{backup_name}'");

                let expert = self.registry.activate(backup_name)?;
                let state_path = self.context.path(backup_name);
                let generation = expert.generate(
                    prompt,
                    self.config.max_new_tokens,
                    self.config.temperature,
                    &state_path,
                )?;

                expert_outputs.push((backup_name.to_string(), generation.text));
                fusion_weights.push(backup_weight);
            }
        }

        // 5. Fuse (text-level — see layers/fusion.rs doc comment)
        let fused = text_fuse(&expert_outputs, &fusion_weights);

        // 6. Critic verification
        let verdict = self.critic.verify(&fused)?;

        Ok((fused, verdict))
    }

    /// Generate from expert `name`, running `config.best_of_n` candidates
    /// and keeping the one the Critic scores highest when `best_of_n > 1`.
    /// `best_of_n == 1` (the default) is exactly today's single generation,
    /// same cost as before this existed.
    ///
    /// Each candidate reads from a scratch copy of the expert's session
    /// state file rather than the real one, so candidates don't clobber
    /// each other's view of prior-turn state; only the winning candidate's
    /// resulting state is copied back to the real path afterward.
    fn generate_best_of_n(&mut self, name: &str, prompt: &str) -> Result<GenerationOutput> {
        let expert = self.registry.activate(name)?;
        let state_path = self.context.path(name);
        let n = self.config.best_of_n.max(1);

        if n == 1 {
            return expert.generate(prompt, self.config.max_new_tokens, self.config.temperature, &state_path);
        }

        let mut best: Option<(f32, GenerationOutput, PathBuf)> = None;
        let mut candidate_paths: Vec<PathBuf> = Vec::with_capacity(n);

        for i in 0..n {
            let candidate_path = state_path.with_extension(format!("cand{i}"));
            if state_path.exists() {
                std::fs::copy(&state_path, &candidate_path)?;
            }
            candidate_paths.push(candidate_path.clone());

            let generation = expert.generate(
                prompt,
                self.config.max_new_tokens,
                self.config.temperature,
                &candidate_path,
            )?;
            let verdict = self.critic.verify(&generation.text)?;

            let is_better = match &best {
                Some((score, _, _)) => verdict.composite > *score,
                None => true,
            };
            if is_better {
                best = Some((verdict.composite, generation, candidate_path));
            }
        }

        let (_, generation, winner_path) = best.expect("n > 1 guarantees at least one candidate");
        std::fs::copy(&winner_path, &state_path)?;
        for p in candidate_paths {
            let _ = std::fs::remove_file(p);
        }

        Ok(generation)
    }
}
