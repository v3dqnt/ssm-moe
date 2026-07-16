/*!
SSM MoE Pipeline — top-level orchestrator.

Full forward pass:
  Prompt
    → Decomposition (TODO: implement sub-task splitting)
    → Brain Router   (gate logits + context memory read)
    → Adaptive-K Gate (select K experts)
    → Expert Router  (llama.cpp's native multi-model router — see
                       experts/expert_router.rs — loads/evicts on demand,
                       no explicit activate/hibernate call needed here
                       anymore)
    → Confidence check (activate backup if needed)
    → Fusion         (weighted blend)
    → Critic         (verify or re-route)
    → Context Memory write
    → Output
*/

use anyhow::Result;
use candle_core::Device;
use tracing::{info, warn};

use crate::{
    brain::{
        gate::{adaptive_k_gate, GateOutput},
        router::BrainRouter,
    },
    config::MoEConfig,
    experts::expert_router::ExpertRouter,
    layers::{
        critic::CriticModel,
        fusion::weighted_fuse,
    },
    memory::context::ContextMemory,
};

pub struct MoEPipeline {
    config: MoEConfig,
    brain: BrainRouter,
    experts: ExpertRouter,
    critic: CriticModel,
    context: ContextMemory,
    device: Device,
    /// Max re-route attempts before returning best effort output
    max_retries: usize,
}

impl MoEPipeline {
    pub async fn new(config: MoEConfig, session_id: &str) -> Result<Self> {
        let device = if cfg!(feature = "cuda") && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };

        info!("Initialising SSM MoE pipeline on {device:?}");

        let brain = BrainRouter::load(&config, device.clone())?;

        // One persistent router server for all experts — see
        // expert_router.rs doc comment for why this replaced per-expert
        // subprocess spawning.
        let experts = ExpertRouter::spawn(&config)?;

        let critic = CriticModel::load(
            &config.critic_model_id,
            config.critic_threshold,
            device.clone(),
        )
        .await?;

        let context = ContextMemory::new(
            session_id,
            config.sessions_dir.clone(),
        )?;

        Ok(Self {
            config,
            brain,
            experts,
            critic,
            context,
            device,
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
        // 1. Load prior context
        let prior_state = self.context.load("brain")?;
        let prior = if prior_state.is_empty() { None } else { Some(prior_state) };

        // 2. Brain Router → gate logits
        let (gate_logits, new_brain_state) = self.brain.forward(prompt, prior)?;

        // 3. Adaptive-K Gate → selected experts
        let gate = adaptive_k_gate(&gate_logits, self.config.k_max, 0.05)?;
        let selected_names: Vec<&str> = gate
            .expert_indices
            .iter()
            .map(|&i| self.config.expert_names()[i])
            .collect();

        info!(
            "Gate selected {} expert(s): {:?} (entropy={:.2})",
            gate.k, selected_names, gate.entropy
        );

        // 4. Run each selected expert. The router server loads each on
        // demand and evicts others per --models-max automatically — no
        // explicit activate/hibernate call needed here. Single active
        // expert is the common case (adaptive-K biases toward k=1 for
        // confident routing, and config defaults k_max=1 for the 8GB VRAM
        // budget) so this is usually one request.
        let mut expert_outputs: Vec<(String, String)> = Vec::with_capacity(selected_names.len());
        for name in &selected_names {
            let text = self.experts.generate(
                name,
                prompt,
                self.config.max_new_tokens,
                self.config.temperature,
            )?;
            expert_outputs.push((name.to_string(), text));
        }

        // 6. Fuse: single expert passes through untouched; multiple experts are
        // combined with their gate weight as a heading so the Critic can see
        // both contributions and their relative confidence. True tensor-level
        // fusion needs token-synchronized generation across experts, which is
        // future work — see docs/agent-harness.md style TODO tracking.
        let fused = if expert_outputs.len() == 1 {
            expert_outputs[0].1.clone()
        } else {
            expert_outputs
                .iter()
                .zip(gate.expert_weights.iter())
                .map(|((name, text), w)| format!("[{name} · weight={w:.2}]\n{text}"))
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        // 7. Critic verification
        let verdict = self.critic.verify(&fused)?;

        // 8. Persist new Brain hidden state
        self.context.save("brain", &new_brain_state)?;

        Ok((fused, verdict))
    }
}

// No custom Drop needed: `ExpertRouter` (the `experts` field) already kills
// the router process in its own Drop impl, which llama.cpp's router in turn
// shuts down its child model-processes for — Rust's default field-drop
// order handles the rest.
