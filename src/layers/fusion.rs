/*!
Fusion Layer.

Combines outputs from multiple active experts into a single string, weighted
by the gate weights produced by the Adaptive-K Gate.

True tensor-level fusion (`Σ weight[i] * expert_output[i]` over hidden
states) needs token-synchronized generation across experts, which none of
the candle, llama.cpp, or original designs here ever actually did — the
pipeline always generated each expert independently and joined the results
as text. So text-level fusion isn't a stopgap anymore, it's the documented
design: single expert passes through untouched; multiple experts are
combined with their gate weight as a heading so the Critic can see both
contributions and their relative confidence.
*/

/// Fuse expert outputs. `outputs` and `weights` must be parallel and
/// non-empty (the gate always selects at least one expert).
pub fn text_fuse(outputs: &[(String, String)], weights: &[f32]) -> String {
    assert_eq!(outputs.len(), weights.len(), "outputs/weights length mismatch");
    assert!(!outputs.is_empty(), "nothing to fuse");

    if outputs.len() == 1 {
        return outputs[0].1.clone();
    }

    outputs
        .iter()
        .zip(weights.iter())
        .map(|((name, text), w)| format!("[{name} · weight={w:.2}]\n{text}"))
        .collect::<Vec<_>>()
        .join("\n\n")
}
