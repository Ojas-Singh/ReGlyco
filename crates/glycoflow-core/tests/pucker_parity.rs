//! Ring enumeration, the Cremer-Pople measure and its inverse against the Python reference.
//!
//! Fixtures from `scripts/export_pucker_fixtures.py` in the GlycoFlow checkout; point
//! `GLYCOFLOW_RUST_DIR` at its `rust/` directory and run with `--ignored`.
//!
//! The ring **order** is what these tests mostly guard. (q2c, q2s) is a Fourier component of the
//! ring's out-of-plane displacement, so its sign and phase depend on which atom the ring starts at
//! and which direction it runs. Rust rebuilds that convention from adjacency alone; a disagreement
//! would not crash, it would make every pucker number subtly wrong.

mod common;

use std::path::PathBuf;

use glycoflow_core::linalg::V3;
use glycoflow_core::pucker::{ring_cycles, ring_pucker, set_puckers, PUCKER_DIM};

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    elements: Vec<String>,
    bonds: Vec<[usize; 2]>,
    coords: Vec<[f64; 3]>,
    rings: Vec<Vec<usize>>,
    substituents: Vec<Vec<Vec<usize>>>,
    pucker: Vec<[f64; PUCKER_DIM]>,
    target: Vec<[f64; PUCKER_DIM]>,
    coords_set: Vec<[f64; 3]>,
    pucker_after_set: Vec<[f64; PUCKER_DIM]>,
}

#[derive(serde::Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

fn load() -> Fixtures {
    let path: PathBuf = PathBuf::from(std::env::var("GLYCOFLOW_RUST_DIR").expect(common::IGNORE))
        .join("fixtures/pucker.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture json")
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn ring_enumeration_matches_python() {
    for case in load().cases {
        let rings = ring_cycles(case.coords.len(), &case.bonds, &case.elements);
        assert_eq!(
            rings.len(),
            case.rings.len(),
            "{}: ring count (python {:?})",
            case.name,
            case.rings
        );
        for (r, (got, want)) in rings.iter().zip(&case.rings).enumerate() {
            assert_eq!(
                &got.atoms, want,
                "{} ring {r}: atom order differs, which flips the sign/phase of (q2c, q2s)",
                case.name
            );
            for (k, want_subs) in case.substituents[r].iter().enumerate() {
                assert_eq!(
                    &got.substituents[k], want_subs,
                    "{} ring {r} slot {k}: substituent set differs",
                    case.name
                );
            }
        }
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn ring_pucker_matches_python() {
    for case in load().cases {
        let rings = ring_cycles(case.coords.len(), &case.bonds, &case.elements);
        for (r, ring) in rings.iter().enumerate() {
            let got = ring_pucker(&case.coords, ring);
            for c in 0..PUCKER_DIM {
                let d = (got[c] - case.pucker[r][c]).abs();
                assert!(
                    d < 2e-3,
                    "{} ring {r} component {c}: rust {:.6} vs python {:.6}",
                    case.name,
                    got[c],
                    case.pucker[r][c]
                );
            }
        }
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn set_pucker_matches_python() {
    for case in load().cases {
        let rings = ring_cycles(case.coords.len(), &case.bonds, &case.elements);
        let mut x: Vec<V3> = case.coords.clone();
        set_puckers(&mut x, &rings, &case.target);

        // the deformation must land on the requested vector ...
        for (r, ring) in rings.iter().enumerate() {
            let got = ring_pucker(&x, ring);
            for c in 0..PUCKER_DIM {
                assert!(
                    (got[c] - case.target[r][c]).abs() < 5e-3,
                    "{} ring {r} component {c}: set_pucker gave {:.6}, asked {:.6}",
                    case.name,
                    got[c],
                    case.target[r][c]
                );
            }
            // ... and Python's own re-measurement agrees, so both inverses are the same map
            for c in 0..PUCKER_DIM {
                assert!(
                    (got[c] - case.pucker_after_set[r][c]).abs() < 5e-3,
                    "{} ring {r} component {c}: rust {:.6} vs python re-measured {:.6}",
                    case.name,
                    got[c],
                    case.pucker_after_set[r][c]
                );
            }
        }

        // every atom, not just the ring atoms: substituents must follow the same way
        let mut worst = 0.0f64;
        for (i, (a, b)) in x.iter().zip(&case.coords_set).enumerate() {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            if d > worst {
                worst = d;
            }
            assert!(
                d < 2e-2,
                "{} atom {i}: rust {:?} vs python {:?} ({d:.4} A apart)",
                case.name,
                a,
                b
            );
        }
        println!("  {:10} max atom deviation {worst:.5} A", case.name);
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn set_pucker_is_idempotent() {
    // applying the same target twice must change nothing: the map is a bijection onto the span the
    // mean plane leaves free, so re-applying it is the identity (`glycoflow/pucker.py`).
    for case in load().cases {
        let rings = ring_cycles(case.coords.len(), &case.bonds, &case.elements);
        let mut once: Vec<V3> = case.coords.clone();
        set_puckers(&mut once, &rings, &case.target);
        let mut twice = once.clone();
        set_puckers(&mut twice, &rings, &case.target);
        for (i, (a, b)) in once.iter().zip(&twice).enumerate() {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            assert!(d < 1e-9, "{} atom {i} moved {d:.3e} A on re-application", case.name);
        }
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn five_rings_have_no_q3() {
    // a furanose has no alternating mode, so its third component is identically zero and must stay
    // zero however it is deformed
    for case in load().cases {
        let rings = ring_cycles(case.coords.len(), &case.bonds, &case.elements);
        for (r, ring) in rings.iter().enumerate() {
            if ring.len() == 5 {
                assert!(
                    ring_pucker(&case.coords, ring)[2].abs() < 1e-12,
                    "{} ring {r}: a five-ring reported a non-zero q3",
                    case.name
                );
                let mut x = case.coords.clone();
                set_puckers(&mut x, &rings, &case.target);
                assert!(
                    ring_pucker(&x, ring)[2].abs() < 1e-12,
                    "{} ring {r}: q3 became non-zero after set_pucker",
                    case.name
                );
            }
        }
    }
}
