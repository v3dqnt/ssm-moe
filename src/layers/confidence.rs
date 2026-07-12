/*!
Confidence / Uncertainty Layer.

Each active expert's generation carries per-token log-probabilities (see
`experts::model::GenerationOutput`), captured straight from llama.cpp's
`get_logits_ith` during decoding. We score confidence from those instead of
the old trained linear probe on hidden states: llama.cpp's high-level API
doesn't expose pre-lm_head hidden states the way a probe needs, so that
approach isn't reachable anymore (and, worth noting, the old probe was never
actually wired into the pipeline anyway — this replacement is).

Mean per-token log-probability of the generated continuation is a standard,
cheap confidence proxy — the same idea behind e.g. Whisper's `avg_logprob`.

Low confidence → triggers backup expert activation before Fusion.
*/

/// Score a generation's confidence in `[0, 1]` from its per-token logprobs.
/// Empty input (e.g. immediate EOS) scores `0.0` — nothing was generated to
/// be confident about.
pub fn score(token_logprobs: &[f32]) -> f32 {
    if token_logprobs.is_empty() {
        return 0.0;
    }
    let mean_logprob = token_logprobs.iter().sum::<f32>() / token_logprobs.len() as f32;
    // exp(mean logprob) is the geometric mean per-token probability, which
    // already falls naturally in (0, 1].
    mean_logprob.exp().clamp(0.0, 1.0)
}

/// Check all expert scores and flag which fall below threshold.
pub fn check_confidence(scores: &[(String, f32)], threshold: f32) -> Vec<String> {
    scores
        .iter()
        .filter(|(_, s)| *s < threshold)
        .map(|(name, _)| name.clone())
        .collect()
}
