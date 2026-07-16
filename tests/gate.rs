//! Architectural sanity-checks for the SSM-MoE engine.
//!
//! No external processes (`llama-server`, `router_server.py`) are spawned —
//! keeps CI footprint tiny while exercising the pure-Rust logic:
//!   • Adaptive-K gate entropy/weight behaviour.
//!   • The `LoadMode` field on `ExpertConfig` (CPU-only experts).
//!
//! The old port-allocation test (`ExpertRegistry::activate` returning a
//! stable per-expert port) was removed along with `ExpertRegistry` itself:
//! experts are now routed through a single persistent llama.cpp router
//! server (see `src/experts/expert_router.rs`) rather than one subprocess
//! per expert, so there's no longer a per-expert port to assert on. The
//! `LoadMode -> n-gpu-layers` mapping that test's spirit cared about is
//! covered directly in `expert_router.rs`'s own `#[cfg(test)]` module,
//! since testing it here would require real GGUF files on disk to
//! canonicalize paths.

use candle_core::{Device, Tensor};
use ssm_moe::{
    brain::gate::adaptive_k_gate,
    config::{LoadMode, MoEConfig},
};

#[test]
fn adaptive_k_gate_basic_behaviour() {
    // A 5-dim logits vector with a clear dominant expert (index 0).
    let logits = Tensor::new(&[2.0_f32, 0.5, 0.1, -1.0, -2.0], &Device::Cpu).unwrap();

    // Allow up to three experts; the gate should never exceed this bound.
    let gate = adaptive_k_gate(&logits, 3, 0.05).expect("gate failed");

    assert!(!gate.expert_indices.is_empty(), "gate returned zero experts");
    assert!(gate.k <= 3, "selected more experts than k_max");

    let weight_sum: f32 = gate.expert_weights.iter().copied().sum();
    assert!((weight_sum - 1.0).abs() < 1e-5, "weights do not sum to 1.0 (got {weight_sum})");

    assert!((0.0..=1.0).contains(&gate.entropy), "entropy out of bounds");
}

#[test]
fn config_expert_order_and_load_mode() {
    let cfg = MoEConfig::default();

    let expected = vec!["coding", "math", "reasoning", "general", "creative"];
    let actual: Vec<_> = cfg.expert_names().into_iter().collect();
    assert_eq!(expected, actual, "expert ordering mismatch");

    // The "creative" expert should be CPU-only per the default config.
    let creative_cfg = cfg.get_expert("creative").expect("creative expert missing");
    assert_eq!(creative_cfg.load_mode, LoadMode::Cpu, "creative expert is not marked as Cpu");
}
