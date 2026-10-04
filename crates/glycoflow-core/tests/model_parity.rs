//! Network and sampler parity against the Python fp32 fixtures (needs rust/weights).
//! With `--features cuda` every test also runs on the GPU (fp32) and is compared with the Python
//! CUDA fp32 reference (`*_cuda_f32`, from `export_rust_fixtures.py --cuda`): on the GPU the K=5
//! matmul inside `torch.cdist` rounds differently, which alone moves fp32 velocities by ~1e-3, so
//! CPU and GPU results are each compared with the reference computed on the same kind of device.

mod common;

use common::*;
use glycoflow_core::guidance::ClashGuidance;
use glycoflow_core::model::{Ops, Precision};
use glycoflow_core::sampler::{Method, SampleOptions, Sampler};
use glycoflow_core::Glycan;

/// Reference key for a device: `key_cuda_f32` on CUDA when exported, else the CPU reference.
fn reference(fx: &Fixture, key: &str, dev_name: &str) -> String {
    let k = format!("{key}_cuda_f32");
    if dev_name == "cuda" && fx.has(&k) {
        k
    } else {
        key.to_string()
    }
}

fn sampler_for<'m>(
    fx: &Fixture,
    model: &'m glycoflow_core::TorsionFlowNet,
) -> (Sampler<'m>, Glycan) {
    let g = Glycan::from_builds(&[library().build(fx.sequence(), None).unwrap()]).unwrap();
    (Sampler::for_glycan(model, &g, &meta().vocab).unwrap(), g)
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn forward_matches_python() {
    for (dev_name, dev) in devices() {
        let model = model(&dev, Precision::F32);
        let mut worst = 0f32;
        for name in CASES {
            let fx = Fixture::load(name);
            let (s, _) = sampler_for(&fx, &model);
            let v = s
                .velocity(&fx.p3("fwd_coords"), &fx.f32("fwd_tau"), &fx.f32("fwd_t"))
                .unwrap();
            let key = reference(&fx, "fwd_v", dev_name);
            let want = fx.f32(&key);
            let e = max_abs(&v, &want);
            let e_cpu = max_abs(&v, &fx.f32("fwd_v"));
            let scale = want.iter().fold(0f32, |a, x| a.max(x.abs()));
            println!("[{dev_name}] {name:20} forward: max |dv| {e:.2e} vs {key} (vs CPU reference {e_cpu:.2e}; max |v| {scale:.2})");
            worst = worst.max(e);
        }
        assert!(worst < 1e-4, "[{dev_name}] forward max error {worst}");
    }
}

/// The composed (differentiable, candle ops only) path agrees with the fused kernels.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn composed_ops_match_fused() {
    for (dev_name, dev) in devices() {
        let mut model = model(&dev, Precision::F32);
        let mut worst = (0f32, 0f32);
        for name in CASES {
            let fx = Fixture::load(name);
            let want = fx.f32(&reference(&fx, "fwd_v", dev_name));
            let v = {
                let (s, _) = sampler_for(&fx, &model);
                s.velocity(&fx.p3("fwd_coords"), &fx.f32("fwd_tau"), &fx.f32("fwd_t"))
                    .unwrap()
            };
            model.ops = Ops::Composed;
            let vc = {
                let (s, _) = sampler_for(&fx, &model);
                s.velocity(&fx.p3("fwd_coords"), &fx.f32("fwd_tau"), &fx.f32("fwd_t"))
                    .unwrap()
            };
            model.ops = Ops::Fused;
            worst.0 = worst.0.max(max_abs(&v, &vc));
            worst.1 = worst.1.max(max_abs(&vc, &want));
        }
        println!(
            "[{dev_name}] fused vs composed: max |dv| {:.2e}; composed vs Python {:.2e}",
            worst.0, worst.1
        );
        assert!(worst.0 < 1e-4 && worst.1 < 1e-4);
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn sampling_matches_python() {
    for (dev_name, dev) in devices() {
        let model = model(&dev, Precision::F32);
        let (mut w_tau, mut w_x) = (0f32, 0f32);
        for name in CASES {
            let fx = Fixture::load(name);
            let (s, _) = sampler_for(&fx, &model);
            let n = s.n_atoms();
            let tau0 = fx.f32("tau0");
            let b = tau0.len() / s.n_torsions();
            let tpl: Vec<[f32; 3]> = (0..b).flat_map(|_| fx.p3("template")).collect();
            for (key, opts) in [
                (
                    "heun32",
                    SampleOptions {
                        steps: 32,
                        method: Method::Heun,
                    },
                ),
                (
                    "euler8",
                    SampleOptions {
                        steps: 8,
                        method: Method::Euler,
                    },
                ),
            ] {
                let (x, tau) = s.sample(&tpl, &tau0, opts, None).unwrap();
                let (kt, kx) = (
                    reference(&fx, &format!("{key}_tau"), dev_name),
                    reference(&fx, &format!("{key}_coords"), dev_name),
                );
                let e_tau = max_ang(&tau, &fx.f32(&kt));
                let e_x = max_dist(&x, &fx.p3(&kx));
                let e_cpu = max_ang(&tau, &fx.f32(&format!("{key}_tau")));
                assert_eq!(x.len(), b * n);
                println!("[{dev_name}] {name:20} {key}: torsions {e_tau:.2e} rad, coords {e_x:.2e} A vs {kt} (torsions vs CPU reference {e_cpu:.2e})");
                w_tau = w_tau.max(e_tau);
                w_x = w_x.max(e_x);
            }
        }
        // Some trajectories are chaotic at the fp32 noise scale: in Python itself, perturbing tau0 of
        // fruf3 sample 1 by 1e-7 rad moves its Heun-32 endpoint by 5.4e-4 rad. CPU stays within
        // 1e-3 rad; on CUDA that sample ends 1.7e-3 rad from the Python CUDA run.
        let tol = if dev_name == "cpu" { 1e-3 } else { 3e-3 };
        assert!(
            w_tau < tol,
            "[{dev_name}] sampled torsions differ by {w_tau} rad"
        );
        assert!(w_x < 1e-2, "[{dev_name}] sampled coords differ by {w_x} A");
    }
}

/// Clash guidance (analytic Jacobian here, autograd in Python): reported, loose bound.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn guided_sampling_matches_python() {
    for (dev_name, dev) in devices() {
        let model = model(&dev, Precision::F32);
        let mut worst = 0f32;
        for name in CASES {
            let fx = Fixture::load(name);
            let (s, _) = sampler_for(&fx, &model);
            let tau0 = fx.f32("tau0");
            let b = tau0.len() / s.n_torsions();
            let tpl: Vec<[f32; 3]> = (0..b).flat_map(|_| fx.p3("template")).collect();
            let mut g = ClashGuidance::new(0.3, 0.5);
            let (x, tau) = s
                .sample(&tpl, &tau0, SampleOptions::default(), Some(&mut g))
                .unwrap();
            let e_tau = max_ang(&tau, &fx.f32("guided_heun32_tau"));
            let e_x = max_dist(&x, &fx.p3("guided_heun32_coords"));
            println!("[{dev_name}] {name:20} guided heun32: torsions {e_tau:.2e} rad, coords {e_x:.2e} A");
            worst = worst.max(e_tau);
        }
        assert!(
            worst < 1e-2,
            "[{dev_name}] guided torsions differ by {worst} rad"
        );
    }
}

fn circ_mean(x: &[f32]) -> f64 {
    let (s, c) = x.iter().fold((0f64, 0f64), |(s, c), &a| {
        (s + (a as f64).sin(), c + (a as f64).cos())
    });
    s.atan2(c)
}

/// Circular W1 in degrees from 1-degree histograms (`metrics.circular_w1_deg`).
fn circ_w1_deg(a: &[f32], b: &[f32]) -> f64 {
    let hist = |x: &[f32]| {
        let mut h = vec![0f64; 360];
        for &v in x {
            let idx = (((v + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 360.0) as i64)
                .clamp(0, 359);
            h[idx as usize] += 1.0 / x.len() as f64;
        }
        h
    };
    let (ha, hb) = (hist(a), hist(b));
    let mut d = Vec::with_capacity(360);
    let mut acc = 0.0;
    for k in 0..360 {
        acc += ha[k] - hb[k];
        d.push(acc);
    }
    let mut sorted = d.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = sorted[179]; // torch.median: lower median
    d.iter().map(|x| (x - med).abs()).sum()
}

fn column(x: &[f32], nt: usize, k: usize) -> Vec<f32> {
    x.chunks(nt).map(|r| r[k]).collect()
}

fn ensemble_report(label: &str, a: &[f32], b: &[f32], nt: usize) -> (f64, f64, f64) {
    let (mut mean_dev, mut max_dev, mut mean_w1, mut max_w1) = (0f64, 0f64, 0f64, 0f64);
    for k in 0..nt {
        let (ca, cb) = (column(a, nt, k), column(b, nt, k));
        let d = (circ_mean(&ca) - circ_mean(&cb)).rem_euclid(2.0 * std::f64::consts::PI);
        let d = d.min(2.0 * std::f64::consts::PI - d).to_degrees();
        let w = circ_w1_deg(&ca, &cb);
        mean_dev += d / nt as f64;
        max_dev = max_dev.max(d);
        mean_w1 += w / nt as f64;
        max_w1 = max_w1.max(w);
    }
    let same = a
        .iter()
        .zip(b)
        .filter(|(x, y)| max_ang(&[**x], &[**y]) < 1e-3)
        .count() as f64
        / a.len() as f64;
    println!(
        "{label}: circular-mean diff mean {mean_dev:.3} / max {max_dev:.3} deg; W1 mean {mean_w1:.3} / max {max_w1:.3} deg; \
         {:.1}% of torsions within 1e-3 rad",
        100.0 * same
    );
    (max_dev, max_w1, same)
}

/// 256 conformers from the same tau0 batch in Python (fp32 CPU) and Rust.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn ensemble_statistics_match_python() {
    for (dev_name, dev) in devices() {
        let model = model(&dev, Precision::F32);
        for name in ["lacnac", "core_man3"] {
            let fx = Fixture::load(name);
            let (s, g) = sampler_for(&fx, &model);
            let nt = s.n_torsions();
            let tau0 = fx.f32("ens_tau0");
            let n_samples = tau0.len() / nt;
            let (_, tau) = s
                .sample_ensemble(
                    &g.templates,
                    n_samples,
                    &tau0,
                    SampleOptions::default(),
                    usize::MAX,
                    None,
                    |_, _| {},
                )
                .unwrap();
            let key = reference(&fx, "ens_heun32_tau", dev_name);
            let py = fx.f32(&key);
            let (max_dev, max_w1, _) = ensemble_report(
                &format!("[{dev_name}] {name} rust-f32 vs {key} ({n_samples} x {nt})"),
                &tau,
                &py,
                nt,
            );
            if dev_name == "cuda" {
                ensemble_report(
                    &format!("[{dev_name}] {name} rust-f32 vs ens_heun32_tau (CPU)"),
                    &tau,
                    &fx.f32("ens_heun32_tau"),
                    nt,
                );
            }
            if dev_name == "cpu" && fx.has("ens_heun32_tau_cuda_bf16") {
                let cpu = fx.f32("ens_heun32_tau");
                ensemble_report(
                    &format!("[ref] {name} python-cuda-bf16 vs python-cpu-f32"),
                    &fx.f32("ens_heun32_tau_cuda_bf16"),
                    &cpu,
                    nt,
                );
                ensemble_report(
                    &format!("[ref] {name} python-cuda-f32 vs python-cpu-f32"),
                    &fx.f32("ens_heun32_tau_cuda_f32"),
                    &cpu,
                    nt,
                );
            }
            assert!(
                max_dev < 0.5 && max_w1 < 2.0,
                "{name}: ensemble statistics differ"
            );
        }
    }
}

/// Rust bf16 matmuls (CUDA) vs the Python bf16-autocast ensemble: same distribution, not bitwise.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn bf16_ensemble_statistics() {
    for (dev_name, dev) in devices() {
        if dev_name != "cuda" {
            continue;
        }
        let model = model(&dev, Precision::Bf16);
        for name in ["lacnac", "core_man3"] {
            let fx = Fixture::load(name);
            if !fx.has("ens_heun32_tau_cuda_bf16") {
                continue;
            }
            let (s, g) = sampler_for(&fx, &model);
            let nt = s.n_torsions();
            let tau0 = fx.f32("ens_tau0");
            let n_samples = tau0.len() / nt;
            let (_, tau) = s
                .sample_ensemble(
                    &g.templates,
                    n_samples,
                    &tau0,
                    SampleOptions::default(),
                    usize::MAX,
                    None,
                    |_, _| {},
                )
                .unwrap();
            let (_, max_w1, _) = ensemble_report(
                &format!("[cuda] {name} rust-bf16 vs python-cuda-bf16"),
                &tau,
                &fx.f32("ens_heun32_tau_cuda_bf16"),
                nt,
            );
            ensemble_report(
                &format!("[cuda] {name} rust-bf16 vs python-cpu-f32"),
                &tau,
                &fx.f32("ens_heun32_tau"),
                nt,
            );
            assert!(max_w1 < 2.0, "{name}: bf16 ensemble differs");
        }
    }
}
