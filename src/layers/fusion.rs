/*!
Fusion Layer.

Combines outputs from multiple active experts into a single tensor using
the normalised gate weights produced by the Adaptive-K Gate.

  fused = Σ weight[i] * expert_output[i]
*/

use anyhow::Result;
use candle_core::Tensor;

/// Weighted sum of expert output tensors.
///
/// `outputs` and `weights` must have the same length.
/// All tensors in `outputs` must have identical shape.
pub fn weighted_fuse(outputs: &[Tensor], weights: &[f32]) -> Result<Tensor> {
    assert_eq!(outputs.len(), weights.len(), "outputs/weights length mismatch");
    assert!(!outputs.is_empty(), "nothing to fuse");

    if outputs.len() == 1 {
        return Ok(outputs[0].clone());
    }

    let device = outputs[0].device();
    let dtype  = outputs[0].dtype();

    let mut fused = Tensor::zeros_like(&outputs[0])?;

    for (tensor, &w) in outputs.iter().zip(weights.iter()) {
        let scaled = tensor.affine(w as f64, 0.0)?;
        fused = fused.add(&scaled)?;
    }

    Ok(fused)
}
