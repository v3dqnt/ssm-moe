/*!
Confidence / Uncertainty Layer.

Each active expert emits a scalar confidence score alongside its output.
Implemented as a small linear head (hidden_size → 1) on the last hidden state,
followed by sigmoid to produce a value in [0, 1].

Low confidence → triggers backup expert activation before Fusion.
*/

use anyhow::Result;
use candle_core::{Tensor, DType};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

pub struct ConfidenceHead {
    proj: Linear,  // (hidden_size, 1)
}

impl ConfidenceHead {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(hidden_size, 1, vb)?;
        Ok(Self { proj })
    }

    /// Score a hidden state tensor. Returns scalar in [0, 1].
    pub fn score(&self, hidden: &Tensor) -> Result<f32> {
        // hidden: (seq_len, hidden_size) — mean-pool first
        let pooled = hidden.mean(0)?;            // (hidden_size,)
        let logit  = self.proj.forward(&pooled.unsqueeze(0)?)?; // (1, 1)
        let score  = candle_nn::ops::sigmoid(&logit)?;
        Ok(score.to_scalar::<f32>()?)
    }
}

/// Check all expert scores and flag which fall below threshold.
pub fn check_confidence(
    scores: &[(String, f32)],
    threshold: f32,
) -> Vec<String> {
    scores
        .iter()
        .filter(|(_, s)| *s < threshold)
        .map(|(name, _)| name.clone())
        .collect()
}
