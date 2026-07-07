/*!
Adaptive-K Gate.

Maps Brain gate logits → a sparse set of expert indices + normalised weights.
The number of active experts K is determined by the entropy of the gate
distribution — uncertain prompts activate more experts.
*/

use candle_core::{Result, Tensor};

#[derive(Debug)]
pub struct GateOutput {
    /// Indices into the expert pool that should activate.
    pub expert_indices: Vec<usize>,
    /// Normalised fusion weights, parallel to expert_indices.
    pub expert_weights: Vec<f32>,
    /// Number of experts selected.
    pub k: usize,
    /// Normalised entropy [0, 1] — 0 = certain, 1 = uniform.
    pub entropy: f32,
}

/// Select K experts dynamically based on gate entropy.
///
/// Entropy thresholds:
///   H < 0.33  → k = 1  (clear single domain)
///   H < 0.66  → k = 2  (two domains)
///   H >= 0.66 → k_max  (ambiguous / multi-domain)
pub fn adaptive_k_gate(
    gate_logits: &Tensor,
    k_max: usize,
    min_weight: f32,
) -> Result<GateOutput> {
    let n = gate_logits.dims()[0];

    // softmax
    let probs = softmax(gate_logits)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // normalised entropy
    let h_max = (n as f32).ln();
    let entropy: f32 = probs_vec
        .iter()
        .map(|&p| if p > 1e-9 { -p * p.ln() } else { 0.0 })
        .sum::<f32>()
        / h_max;

    let k = if entropy < 0.33 {
        1
    } else if entropy < 0.66 {
        2.min(k_max)
    } else {
        k_max
    };

    // top-k indices by probability
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let topk: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

    // prune near-zero weights
    let mut selected: Vec<(usize, f32)> = topk
        .into_iter()
        .filter(|&(_, w)| w >= min_weight)
        .collect();

    if selected.is_empty() {
        // always keep at least one expert
        selected = vec![(0, 1.0)];
    }

    // renormalise
    let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
    let expert_indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
    let expert_weights: Vec<f32> = selected.iter().map(|(_, w)| w / weight_sum).collect();

    Ok(GateOutput {
        k: expert_indices.len(),
        expert_indices,
        expert_weights,
        entropy,
    })
}

fn softmax(logits: &Tensor) -> Result<Tensor> {
    let max = logits.max(0)?;
    let shifted = logits.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_all()?;
    exp.broadcast_div(&sum)
}
