use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub name: String,
    /// HuggingFace Hub repo id — kept as a human-readable label/provenance
    /// record even though inference now goes through a local GGUF file, not
    /// a live HF download of this repo.
    pub model_id: String,
    /// Local path to the GGUF checkpoint llama-server actually loads.
    /// Real quantization (unlike the candle path this replaced) — GGUF's
    /// Q4_K_M etc. are mature and verified to fit the 8GB VRAM budget.
    pub gguf_path: PathBuf,
    /// Load mode – `Gpu` (default) runs the model on the GPU via `-ngl`.
    /// `Cpu` forces a CPU‑only llama‑server (`--cpu` flag) which saves VRAM
    /// at the cost of slower generation. This is useful for rarely used
    /// experts such as the "creative" domain.
    pub load_mode: LoadMode,
}

impl ExpertConfig {
    pub fn new(name: &str, model_id: &str, gguf_path: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            model_id: model_id.into(),
            gguf_path: gguf_path.into(),
            load_mode: LoadMode::Gpu,
        }
    }
}

/// Whether an expert should be launched on the GPU or CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum LoadMode {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    // Router: bart-large-mnli zero-shot classifier — no fine-tuning needed,
    // already validated against sanity prompts during dataset labeling.
    pub brain_model_id: String,
    pub brain_labels: Vec<String>,
    pub brain_threshold: f32,

    pub critic_model_id: String,

    // expert pool — each is a full pretrained checkpoint, served locally via
    // a per-expert llama-server subprocess (see experts/llama_server.rs)
    pub experts: Vec<ExpertConfig>,

    // routing
    pub k_max: usize,
    pub confidence_threshold: f32,
    pub critic_threshold: f32,

    // llama-server process management
    pub llama_server_exe: PathBuf,
    pub llama_server_base_port: u16,
    /// -ngl passed to llama-server — layers offloaded to GPU. High value
    /// pushes as much as fits; llama.cpp caps it automatically if the model
    /// has fewer layers or VRAM runs out.
    pub n_gpu_layers: u32,

    // memory tiers
    pub adapters_dir: PathBuf,
    pub warm_cache_size: usize,
    pub sessions_dir: PathBuf,

    // inference
    pub max_new_tokens: usize,
    pub temperature: f64,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            brain_model_id: "facebook/bart-large-mnli".into(),
            brain_labels: vec![
                "coding".into(),
                "math".into(),
                "reasoning".into(),
                "creative".into(),
                "general".into(),
            ],
            brain_threshold: 0.65, // matches the threshold tuned during dataset labeling

            critic_model_id: "state-spaces/mamba-130m-hf".into(),

            experts: vec![
                // Q4_K_M GGUF, ~4GB VRAM, fits an 8GB card with headroom for
                // the critic (CPU-only) and generation buffers.
                ExpertConfig::new(
                    "coding",
                    "mistralai/Mamba-Codestral-7B-v0.1",
                    "../models/codestral-mamba-q4_k_m.gguf",
                ),
                // Real Mamba-2 math specialist (PromptCoT curriculum). No
                // pre-made GGUF existed — self-converted via llama.cpp's
                // convert_hf_to_gguf.py + llama-quantize (Mamba2Model class
                // explicitly supports this architecture by name).
                ExpertConfig::new(
                    "math",
                    "xl-zhao/PromptCoT-Mamba-Math-7B",
                    "../models/promptcot-math-q4_k_m.gguf",
                ),
                // OpenHermes-tuned Mamba-1, converted from the upstream
                // pytorch_model.bin mirror we made earlier this session.
                ExpertConfig::new(
                    "reasoning",
                    "v3dqnt/mamba-2.8b-instruct-openhermes-st",
                    "../models/openhermes-2.8b-q4_k_m.gguf",
                ),
                ExpertConfig::new(
                    "general",
                    "v3dqnt/mamba-2.8b-instruct-openhermes-st",
                    "../models/openhermes-2.8b-q4_k_m.gguf",
                ),
                // creative still on the generic mamba-chat placeholder — no
                // dedicated creative-writing SSM found yet.
                ExpertConfig {
                    name: "creative".into(),
                    model_id: "havenhq/mamba-chat".into(),
                    gguf_path: "../models/mamba-chat-q4_k_m.gguf".into(),
                    load_mode: LoadMode::Cpu,
                },
            ],

            // one expert at a time — matches the 8GB VRAM budget decision
            k_max: 1,
            confidence_threshold: 0.70,
            critic_threshold: 0.75,

            llama_server_exe: PathBuf::from("../llama.cpp/llama-server.exe"),
            llama_server_base_port: 8100,
            n_gpu_layers: 999, // offload everything; llama.cpp caps automatically

            adapters_dir: PathBuf::from("./adapters"),
            warm_cache_size: 3,
            sessions_dir: PathBuf::from("./.sessions"),

            max_new_tokens: 2048,
            temperature: 0.7,
        }
    }
}

impl MoEConfig {
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    pub fn expert_names(&self) -> Vec<&str> {
        self.experts.iter().map(|e| e.name.as_str()).collect()
    }

    pub fn get_expert(&self, name: &str) -> anyhow::Result<&ExpertConfig> {
        self.experts
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow::anyhow!("Unknown expert: {name}"))
    }
}
