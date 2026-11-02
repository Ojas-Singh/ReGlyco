//! A checkpoint's ring head is read from its weights, never assumed from its config.
//!
//! `glycoflow-core` looks weights up by name out of a `HashMap`, and `ModelConfig` ignores unknown
//! keys, so before the engine had a ring head a checkpoint carrying `head_pucker.*` loaded here with
//! no error and sampled on library ring templates. The engine now runs the head; what must not
//! happen is a config and a weight file that disagree. A flag with no head is a broken export and is
//! refused; a head with no flag is a ring-head model (checked against real weights in
//! `ring_head_parity.rs`).
//!
//! These tests need no fixtures or weights -- the check runs before any weight lookup -- so unlike
//! the parity tests they are not `#[ignore]`d and run on every `cargo test`.

use std::collections::HashMap;

use candle_core::Device;
use glycoflow_core::model::Precision;
use glycoflow_core::{ModelConfig, TorsionFlowNet};

fn fixed_ring_config() -> ModelConfig {
    ModelConfig {
        n_elements: 7,
        n_atom_names: 56,
        n_residues: 80,
        n_link_codes: 32,
        dim: 256,
        depth: 8,
        heads: 8,
        pair_dim: 32,
        rbf_bins: 32,
        rbf_max: 20.0,
        ff_mult: 2,
        torsion_freqs: 4,
        pucker: false,
    }
}

#[test]
fn pucker_flag_without_head_weights_is_refused() {
    let cfg = ModelConfig {
        pucker: true,
        ..fixed_ring_config()
    };
    let msg = match TorsionFlowNet::from_tensors(&HashMap::new(), cfg, &Device::Cpu, Precision::F32)
    {
        Ok(_) => panic!("a pucker=true config with no head_pucker weights must not load"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("head_pucker") && msg.contains("pucker"),
        "the error should say what is wrong: {msg}"
    );
}

#[test]
fn pucker_defaults_to_false_so_fixed_ring_json_still_parses() {
    // the released glycoflow.json has no "pucker" key at all
    let json = r#"{"n_elements":7,"n_atom_names":56,"n_residues":80,"n_link_codes":32,
                   "dim":256,"depth":8,"heads":8,"pair_dim":32,"rbf_bins":32,"rbf_max":20.0,
                   "ff_mult":2,"torsion_freqs":4}"#;
    let cfg: ModelConfig = serde_json::from_str(json).expect("fixed-ring config must still parse");
    assert!(!cfg.pucker);
}
