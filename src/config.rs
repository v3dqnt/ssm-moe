use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Real GGUF quantization level. Unlike the old candle-based config (where
/// `Int4` etc. were VRAM-*planning* labels that candle never actually
/// applied — it always loaded at the checkpoint's native dtype), these map
/// directly to the quant type actually baked into the `.gguf` file llama.cpp
/// loads, so what's declared here is what's really running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    /// ~4 bits/weight, llama.cpp's baseline 4-bit quant.
    Q4_0,
    /// ~5 bits/weight, k-quant with better accuracy than Q4/Q5 legacy quants.
    Q5KM,
    /// 8 bits/weight — near-lossless, largest of the quantized options.
    Q8_0,
    /// Full 16-bit float, no quantization.
    F16,
}

impl Quantization {
    /// Rough bytes-per-parameter, for logging/capacity-planning only.
    pub fn bytes_per_param(&self) -> f64 {
        match self {
            Quantization::Q4_0 => 0.5,
            Quantization::Q5KM => 0.625,
            Quantization::Q8_0 => 1.0,
            Quantization::F16 => 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertConfig {
    pub name: String,
    /// HuggingFace Hub repo id hosting this expert's GGUF conversion,
    /// e.g. "gabriellarson/Mamba-Codestral-7B-v0.1-GGUF".
    pub gguf_repo: String,
    /// The specific `.gguf` filename to pull from `gguf_repo`.
    pub gguf_file: String,
    pub quantization: Quantization,
}

impl ExpertConfig {
    pub fn new(name: &str, gguf_repo: &str, gguf_file: &str, quantization: Quantization) -> Self {
        Self {
            name: name.into(),
            gguf_repo: gguf_repo.into(),
            gguf_file: gguf_file.into(),
            quantization,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    // Router: bart-large-mnli zero-shot classifier — no fine-tuning needed,
    // already validated against sanity prompts during dataset labeling.
    // Still an external Python sidecar (see brain/router.rs) — untouched by
    // the candle -> llama.cpp migration.
    pub brain_model_id: String,
    pub brain_labels: Vec<String>,
    pub brain_threshold: f32,

    /// HF repo hosting the critic's GGUF conversion (small, always
    /// CPU-resident base model — kept off the K_max expert budget).
    pub critic_gguf_repo: String,
    pub critic_gguf_file: String,

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

    /// Fixed llama.cpp context window (prior turns' state + this turn's
    /// prompt + generated tokens must all fit). Generous default since these
    /// are 7B-class experts with no KV-cache growth concerns beyond memory.
    pub n_ctx: u32,
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

            // devingulliver/mamba-gguf hosts community GGUF conversions of the
            // state-spaces/mamba checkpoints at several sizes — filename
            // unconfirmed against the repo's actual file listing (same
            // diligence the old candle code applied to Codestral's sharding;
            // verify before relying on this in production).
            critic_gguf_repo: "devingulliver/mamba-gguf".into(),
            critic_gguf_file: "mamba-130m-q8_0.gguf".into(),

            experts: vec![
                // Real, existing Mamba-2 GGUF conversion — confirmed present
                // on HF Hub at multiple quant levels.
                ExpertConfig::new(
                    "coding",
                    "gabriellarson/Mamba-Codestral-7B-v0.1-GGUF",
                    "Mamba-Codestral-7B-v0.1-Q4_0.gguf",
                    Quantization::Q4_0,
                ),
                // No pre-made GGUF found for xl-zhao/PromptCoT-Mamba-Math-7B at
                // time of writing — needs one-time offline conversion
                // (convert_hf_to_gguf.py + llama-quantize; llama.cpp's
                // converter confirmed supports Mamba2ForCausalLM) and upload,
                // same pattern as the openhermes safetensors mirror below.
                // Placeholder repo/file until that conversion is done.
                ExpertConfig::new(
                    "math",
                    "v3dqnt/PromptCoT-Mamba-Math-7B-GGUF",
                    "PromptCoT-Mamba-Math-7B-Q4_0.gguf",
                    Quantization::Q4_0,
                ),
                // Same story: clibrain/mamba-2.8b-instruct-openhermes (Mamba-1)
                // has no known GGUF — needs offline conversion + upload.
                // Broad GPT-4-distilled instruction data — reasonable fit for
                // both reasoning and general-purpose slots.
                ExpertConfig::new(
                    "reasoning",
                    "v3dqnt/mamba-2.8b-instruct-openhermes-gguf",
                    "mamba-2.8b-instruct-openhermes-Q4_0.gguf",
                    Quantization::Q4_0,
                ),
                ExpertConfig::new(
                    "general",
                    "v3dqnt/mamba-2.8b-instruct-openhermes-gguf",
                    "mamba-2.8b-instruct-openhermes-Q4_0.gguf",
                    Quantization::Q4_0,
                ),
                // creative still on the generic mamba-chat placeholder — no
                // dedicated creative-writing SSM found yet, and no GGUF for
                // it either — needs offline conversion + upload.
                ExpertConfig::new(
                    "creative",
                    "v3dqnt/mamba-chat-GGUF",
                    "mamba-chat-Q4_0.gguf",
                    Quantization::Q4_0,
                ),
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
            n_ctx: 8192,
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
