//! Parity of the ring head against the Python reference (needs a ring-head export).
//!
//! `scripts/export_rust_fixtures.py` writes the ring arrays only when the checkpoint it is given
//! has a `head_pucker`, so point `GLYCOFLOW_RUST_DIR` at a ring-head export (for the release model:
//! `export_rust_fixtures.py --checkpoint models/release/glycoflow.pt --out rust_rings`). With a
//! fixed-ring export every test here reports that it has nothing to check and passes, which is why
//! they can share `GLYCOFLOW_RUST_DIR` with `model_parity.rs`.

mod common;

use common::*;
use glycoflow_core::guidance::ClashGuidance;
use glycoflow_core::model::Precision;
use glycoflow_core::pucker::{ring_puckers, PUCKER_DIM, PUCKER_SCALE};
use glycoflow_core::sampler::{Method, SampleOptions, Sampler};
use glycoflow_core::Glycan;

fn sampler_for<'m>(
    fx: &Fixture,
    model: &'m glycoflow_core::TorsionFlowNet,
) -> (Sampler<'m>, Glycan) {
    let g = Glycan::from_builds(&[library().build(fx.sequence(), None).unwrap()]).unwrap();
    (Sampler::for_glycan(model, &g, &meta().vocab).unwrap(), g)
}

/// The ring-head cases of the export, or an empty list with a note when the export is fixed-ring.
fn ring_cases() -> Vec<Fixture> {
    let out: Vec<Fixture> = CASES
        .iter()
        .map(|n| Fixture::load(n))
        .filter(|f| f.has("fwd_vq"))
        .collect();
    if out.is_empty() {
        println!("no ring-head fixtures in this export (fixed-ring checkpoint): nothing to check");
    }
    out
}

/// The rings the engine finds, in the order the head's pooling assumes.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn ring_order_matches_python() {
    let model = model(&devices()[0].1, Precision::F32);
    for fx in ring_cases() {
        let (s, _) = sampler_for(&fx, &model);
        let want_n = fx.i64("ring_n");
        let want = fx.i64("ring_atoms_ordered"); // [R,6], padded with 0
        assert_eq!(
            s.n_rings(),
            want_n.len(),
            "{}: {} rings, Python has {}",
            fx.name,
            s.n_rings(),
            want_n.len()
        );
        for (r, ring) in s.topology.rings.iter().enumerate() {
            let n = want_n[r] as usize;
            let got: Vec<i64> = ring.atoms.iter().map(|&a| a as i64).collect();
            assert_eq!(got, want[r * 6..r * 6 + n], "{}: ring {r}", fx.name);
        }
        println!("{:20} {} rings in the same order", fx.name, s.n_rings());
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn ring_velocity_matches_python() {
    for (dev_name, dev) in devices() {
        let model = model(&dev, Precision::F32);
        assert!(
            model.has_pucker() || ring_cases().is_empty(),
            "the export has ring fixtures but the weights have no ring head"
        );
        let (mut worst_v, mut worst_q) = (0f32, 0f32);
        for fx in ring_cases() {
            let (s, _) = sampler_for(&fx, &model);
            let q = fx.f32("fwd_q");
            let (v, vq) = s
                .velocity_joint(
                    &fx.p3("fwd_coords"),
                    &fx.f32("fwd_tau"),
                    &fx.f32("fwd_t"),
                    Some(&q),
                )
                .unwrap();
            let vq = vq.expect("ring-head model returned no ring velocity");
            let e_v = max_abs(&v, &fx.f32(&reference(&fx, "fwd_v", dev_name)));
            let e_q = max_abs(&vq, &fx.f32("fwd_vq"));
            let scale = fx.f32("fwd_vq").iter().fold(0f32, |a, x| a.max(x.abs()));
            println!(
                "[{dev_name}] {:20} ring head: max |dvq| {e_q:.2e} (max |vq| {scale:.2}), torsions {e_v:.2e}",
                fx.name
            );
            worst_v = worst_v.max(e_v);
            worst_q = worst_q.max(e_q);
        }
        assert!(worst_q < 1e-4, "[{dev_name}] ring velocity off by {worst_q}");
        assert!(worst_v < 1e-4, "[{dev_name}] torsion velocity off by {worst_v}");
    }
}

/// `Sampler::sample` on a ring-head checkpoint reproduces the reference run with the head switched
/// off: the torsion velocity does not depend on the ring state, so holding the rings at the
/// template -- what fitting a known structure does -- is exact and not an approximation.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn fixed_ring_sampling_matches_python() {
    let (dev_name, dev) = devices().into_iter().next().unwrap();
    let model = model(&dev, Precision::F32);
    let mut worst = 0f32;
    for fx in ring_cases() {
        let (s, _) = sampler_for(&fx, &model);
        let tau0 = fx.f32("tau0");
        let b = tau0.len() / s.n_torsions();
        let tpl: Vec<[f32; 3]> = (0..b).flat_map(|_| fx.p3("template")).collect();
        let (x, tau) = s
            .sample(&tpl, &tau0, SampleOptions::default(), None)
            .unwrap();
        let e_tau = max_ang(&tau, &fx.f32("heun32_tau"));
        let e_x = max_dist(&x, &fx.p3("heun32_coords"));
        println!(
            "[{dev_name}] {:20} fixed-ring heun32: torsions {e_tau:.2e} rad, coords {e_x:.2e} A",
            fx.name
        );
        worst = worst.max(e_tau);
    }
    assert!(worst < 1e-3, "fixed-ring torsions differ by {worst} rad");
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn joint_sampling_matches_python() {
    let (dev_name, dev) = devices().into_iter().next().unwrap();
    let model = model(&dev, Precision::F32);
    let (mut w_tau, mut w_x, mut w_q) = (0f32, 0f32, 0f32);
    for fx in ring_cases() {
        let (s, _) = sampler_for(&fx, &model);
        let tau0 = fx.f32("tau0");
        let q0 = fx.f32("q0");
        let b = tau0.len() / s.n_torsions();
        let tpl: Vec<[f32; 3]> = (0..b).flat_map(|_| fx.p3("template")).collect();
        for (key, guide) in [("joint_heun32", 0.0f32), ("joint_guided_heun32", 0.3)] {
            let mut g = ClashGuidance::new(guide, 0.5);
            let (x, tau, q) = s
                .sample_joint(
                    &tpl,
                    &tau0,
                    &q0,
                    SampleOptions {
                        steps: 32,
                        method: Method::Heun,
                    },
                    (guide > 0.0).then_some(&mut g as &mut dyn glycoflow_core::sampler::Guidance),
                )
                .unwrap();
            let e_tau = max_ang(&tau, &fx.f32(&format!("{key}_tau")));
            let e_x = max_dist(&x, &fx.p3(&format!("{key}_coords")));
            // the integrated ring state, and the same state measured back off the coordinates it
            // was written into (which is how Python recorded it)
            let measured: Vec<f32> = x
                .chunks(s.n_atoms())
                .flat_map(|c| {
                    let xd: Vec<[f64; 3]> = c
                        .iter()
                        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                        .collect();
                    ring_puckers(&xd, &s.topology.rings)
                        .into_iter()
                        .flat_map(|v| v.map(|c| (c / PUCKER_SCALE) as f32))
                        .collect::<Vec<f32>>()
                })
                .collect();
            let e_q = max_abs(&measured, &fx.f32(&format!("{key}_q")));
            let e_state = max_abs(&measured, &q);
            assert_eq!(q.len(), b * s.n_rings() * PUCKER_DIM);
            println!(
                "[{dev_name}] {:20} {key}: torsions {e_tau:.2e} rad, coords {e_x:.2e} A, \
                 ring shapes {e_q:.2e} (state vs geometry {e_state:.2e})",
                fx.name
            );
            w_tau = w_tau.max(e_tau);
            w_x = w_x.max(e_x);
            w_q = w_q.max(e_q);
        }
    }
    assert!(w_tau < 1e-3, "joint torsions differ by {w_tau} rad");
    assert!(w_x < 1e-2, "joint coords differ by {w_x} A");
    assert!(w_q < 1e-2, "joint ring shapes differ by {w_q}");
}

/// Reference key for a device, as in `model_parity.rs`.
fn reference(fx: &Fixture, key: &str, dev_name: &str) -> String {
    let k = format!("{key}_cuda_f32");
    if dev_name == "cuda" && fx.has(&k) {
        k
    } else {
        key.to_string()
    }
}

/// A ring-head checkpoint whose config lost its `"pucker"` flag is still run as a ring-head model:
/// the head is read from the weights.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn stripped_flag_still_loads_the_head() {
    if ring_cases().is_empty() {
        return;
    }
    let bytes = std::fs::read(rust_dir().join("weights/glycoflow.safetensors")).unwrap();
    let mut cfg = meta().config;
    cfg.pucker = false;
    let m = glycoflow_core::TorsionFlowNet::from_safetensors(
        &bytes,
        cfg,
        &candle_core::Device::Cpu,
        Precision::F32,
    )
    .unwrap();
    assert!(m.has_pucker() && m.cfg.pucker);
}

/// `Sampler::encode` -- the prior point of a given conformer, which a path from an arbitrary
/// conformer starts from -- matches the reference's `density.encode`.
///
/// The map itself is lossy, and that is the model, not the port: at 64 Euler steps a prior point
/// comes back to ~0.2 rad, and a conformer near a basin boundary decodes into the neighbouring
/// basin (core_man3 sample 3 returns 11.494 A away, in PyTorch exactly as here). That is why a path
/// between two ensemble members uses their exact seeded prior points rather than encoding them.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn encode_matches_python() {
    let model = model(&candle_core::Device::Cpu, Precision::F32);
    let (mut w_tau, mut w_q) = (0f32, 0f32);
    for fx in ring_cases() {
        let (s, _) = sampler_for(&fx, &model);
        let tau = fx.f32("enc_tau");
        let b = tau.len() / s.n_torsions();
        let tpl: Vec<[f32; 3]> = (0..b).flat_map(|_| fx.p3("template")).collect();
        let (tau0, q0) = s.encode(&tpl, &tau, Some(&fx.f32("enc_q")), 16).unwrap();
        let e_tau = max_ang(&tau0, &fx.f32("enc_tau0"));
        let e_q = max_abs(&q0, &fx.f32("enc_q0"));
        println!("{:20} encode: prior torsions {e_tau:.2e} rad, ring shapes {e_q:.2e}", fx.name);
        w_tau = w_tau.max(e_tau);
        w_q = w_q.max(e_q);
    }
    assert!(w_tau < 1e-3 && w_q < 1e-3, "encode differs: {w_tau} rad, {w_q}");
}
