//! SSM MoE — library surface, shared by the `ssm-moe` binary and integration
//! tests (see `tests/`). Splitting this out of `main.rs` is what lets tests
//! write `use ssm_moe::config::MoEConfig` instead of duplicating pipeline
//! setup code.

pub mod brain {
    pub mod gate;
    pub mod router;
}
pub mod config;
pub mod experts {
    pub mod expert_router;
}
pub mod layers {
    pub mod confidence;
    pub mod critic;
    pub mod fusion;
}
pub mod memory {
    pub mod context;
}
pub mod pipeline;
