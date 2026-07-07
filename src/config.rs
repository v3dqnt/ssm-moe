use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Quantization level a pretrained expert checkpoint is loaded at.
/// Each expert is now a distinct full model (pulled pretrained from HF Hub),
/// not a LoRA delta on a shared base — so quantization is chosen per-expert
/// to fit VRAM rather than derived from a shared rank/alpha config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    Fp32,
    Bf16,
    Int8,
    Int4,
}

impl Quantization {
    /// Rough bytes-per-parameter for VRAM estimation.
    pub fn bytes_per_param(&self) -> f64 {
        match self {
            Quantization::Fp32 => 4.0,
            Quantization::Bf16 => 2.0,
            Quantization::Int8 => 1.0,
            Quantization::Int4 => 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub name: String,
    /// HuggingFace Hub repo id for this expert's full pretrained checkpoint,
    /// e.g. "mistralai/Mamba-Codestral-7B-v0.1".
    pub model_id: String,
    pub quantization: Quantization,
}

impl ExpertConfig {
    pub fn new(name: &str, model_id: &str, quantization: Quantization) -> Self {
        Self {
            name: name.into(),
            model_id: model_id.into(),
            quantization,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    // Router: bart-large-mnli zero-shot classifier — no fine-tuning needed,
    // already validated against sanity prompts during dataset labeling.
    pub brain_model_id: String,
    pub brain_labels: Vec<String>,
    pub brain_threshold: f32,

    pub critic_model_id: String,

    // expert pool — each is a full pretrained checkpoint pulled from HF Hub
    pub experts: Vec<ExpertConfig>,

    // routing
    pub k_max: usize,
    pub confidence_threshold: f32,
    pub critic_threshold: f32,

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
                // int4: 7B model, ~3.5GB VRAM, fits an 8GB card with headroom
                // for router + critic + generation buffers.
                ExpertConfig::new("coding", "mistralai/Mamba-Codestral-7B-v0.1", Quantization::Int4),
                // Real Mamba-2 math specialist (PromptCoT curriculum), verified
                // against candle's Mamba2Config field aliases before wiring in —
                // see model.rs doc comment. Same 7B/4096-hidden shape as Codestral.
                ExpertConfig::new("math", "xl-zhao/PromptCoT-Mamba-Math-7B", Quantization::Int4),
                // OpenHermes-tuned Mamba-1, safetensors mirror we converted
                // from the upstream pytorch_model.bin (clibrain/mamba-2.8b-
                // instruct-openhermes) since candle can't load pickle files.
                // Broad GPT-4-distilled instruction data — reasonable fit for
                // both reasoning and general-purpose slots.
                ExpertConfig::new("reasoning", "v3dqnt/mamba-2.8b-instruct-openhermes-st", Quantization::Int4),
                ExpertConfig::new("general", "v3dqnt/mamba-2.8b-instruct-openhermes-st", Quantization::Int4),
                // creative still on the generic mamba-chat placeholder — no
                // dedicated creative-writing SSM found yet.
                ExpertConfig::new("creative", "havenhq/mamba-chat", Quantization::Int4),
            ],

            // one expert at a time — matches the 8GB VRAM budget decision
            k_max: 1,
            confidence_threshold: 0.70,
            critic_threshold: 0.75,

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
