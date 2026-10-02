//! Numerical parity with the Python reference (`glycoflow/fitting/`), on fixtures written by
//! GlycoFlow's `scripts/fitting/export_fit_fixtures.py`. Needs the site models, the maps and the
//! residue library, so the tests are ignored by default:
//!
//! ```text
//! # in the GlycoFlow checkout
//! .venv/bin/python scripts/fitting/export_fit_fixtures.py --sites sites.json \
//!     --only 5KZC_A79 5GSQ_A297 5GSQ_B297 --out-dir /path/to/fixtures
//! # in ReGlyco
//! GLYCOFLOW_FIT_FIXTURES=/path/to/fixtures cargo test --release -p reglyco-glycoflow \
//!     --test parity parity_ -- --ignored --nocapture
//! # full pipeline on the reference's templates and prior samples (also GLYCOFLOW_MODEL;
//! # GLYCOFLOW_DEVICE=cuda with --features cuda)
//! ... --test parity controlled_fit -- --ignored --nocapture
//! ```
//!
//! Tolerances: environment atom count exact; likelihood constants rel. 1e-4; objective terms
//! rel. 1e-3 (the total relative to the magnitude of its terms); gradient vectors cosine > 0.999
//! and relative norm difference < 1e-3, single dE/dpsi_N, dE/dphi_N components within 2e-4 of the
//! gradient norm (relative 2e-3 with a floor of 10% of the norm: small components carry the
//! float32 noise of the reference; a sign or axis error still fails by orders of magnitude).
//!
//! Two properties of the float32 reference are accounted for:
//! * `torch.cdist` evaluates float32 distances as `|x|^2 + |y|^2 - 2 x.y`, which at ~100 A
//!   coordinates is off by ~1e-4 A. It dominates the reference error of the small self-contact
//!   term, so `E_self` is compared with the exporter's float64 recomputation (and loosely with
//!   the torch value); the same error enters `<g,g>` of the density term (largest at sigma 0.6).
//! * The trilinear interpolation of the blurred maps has gradient kinks on grid planes. Placed
//!   coordinates differ from the reference's float32 kinematics by <= 2e-4 A, so a pose with an
//!   atom within 1e-3 grid units of a plane can sit on the other side: for such poses the
//!   full-kinematics gradient is reported, and every pose is also checked with the same
//!   objective and chain rule evaluated at the reference's placed coordinates.

use std::path::{Path, PathBuf};

use glysys::{BuildOptions, ResidueId, read_pdb_str};
use reglyco_density::DensityMap;
use reglyco_density::site_likelihood::{
    SiteEnvironmentAtom, SiteLikelihood, SiteLikelihoodOptions,
};
use reglyco_glycoflow::observation::DensityObservation;
use reglyco_glycoflow::pipeline::calibrate_sigma;
use reglyco_glycoflow::prior::MarginalPrior;
use reglyco_glycoflow::problem::{Pose, ProblemOptions, SiteProblem, build_glycan, max_span};
use reglyco_glycoflow::search::attach_search;
use reglyco_glycoflow::site::{CrystalInput, SiteOptions, load_site, sequence_paths};
use reglyco_glycoflow::support::subtree_support;
use reglyco_glycoflow::symmetry::{UnitCell, parse_cryst1, parse_resolution};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(std::env::var("GLYCOFLOW_FIT_FIXTURES").expect(
        "set GLYCOFLOW_FIT_FIXTURES to the output of GlycoFlow scripts/fitting/export_fit_fixtures.py",
    ))
}

fn library_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../GlycoFlow/glycoflow/resources/residue_library.json")
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn vec_f(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(f).collect()
}

fn rel(a: f64, b: f64, floor: f64) -> f64 {
    (a - b).abs() / b.abs().max(floor)
}

thread_local! {
    static FAILURES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn fail(message: String) {
    println!("  FAIL {message}");
    FAILURES.with(|f| f.borrow_mut().push(message));
}

fn finish(name: &str) {
    let failures = FAILURES.with(|f| std::mem::take(&mut *f.borrow_mut()));
    assert!(
        failures.is_empty(),
        "{name}: {} parity failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn check(label: &str, rust: f64, python: f64, tol: f64, floor: f64) {
    let r = rel(rust, python, floor);
    println!("  {label:<28} rust {rust:>16.8} python {python:>16.8} rel {r:.2e}");
    if r > tol {
        fail(format!(
            "{label}: rust {rust} vs python {python} (rel {r:.2e} > {tol:.0e})"
        ));
    }
}

fn check_vec(label: &str, rust: &[f64], python: &[f64]) {
    let dot: f64 = rust.iter().zip(python).map(|(a, b)| a * b).sum();
    let nr = rust.iter().map(|a| a * a).sum::<f64>().sqrt();
    let np = python.iter().map(|a| a * a).sum::<f64>().sqrt();
    let diff = rust
        .iter()
        .zip(python)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = if nr == 0.0 && np == 0.0 {
        1.0
    } else {
        dot / (nr * np).max(1e-300)
    };
    let reln = diff / np.max(1e-6);
    println!(
        "  {label:<28} |rust| {nr:>14.6} |python| {np:>14.6} cos {cos:.7} rel.diff {reln:.2e}"
    );
    if !(cos > 0.999 && reln < 1e-3) {
        fail(format!("{label}: cosine {cos}, relative difference {reln}"));
    }
}

struct Loaded {
    fx: Value,
    problem: SiteProblem,
    site: reglyco_glycoflow::Site,
    map: DensityMap,
    resolution: f64,
}

fn load(name: &str) -> Loaded {
    let path = fixture_dir().join(format!("{name}.json"));
    let fx: Value = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap();
    let text = std::fs::read_to_string(fx["pdb"].as_str().unwrap()).unwrap();
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let structure = read_pdb_str(&text, &options).unwrap();
    let map = DensityMap::open(fx["map"].as_str().unwrap()).unwrap();
    let m = map.metadata();
    let crystal = CrystalInput {
        cryst1: parse_cryst1(&text),
        map_cell: UnitCell::new(
            m.cell_lengths_angstrom[0],
            m.cell_lengths_angstrom[1],
            m.cell_lengths_angstrom[2],
            m.cell_angles_degrees[0],
            m.cell_angles_degrees[1],
            m.cell_angles_degrees[2],
        ),
        map_space_group: Some(m.space_group),
        map_full_cell: map.is_full_unit_cell(),
    };
    let label = fx["site"].as_str().unwrap();
    let parts: Vec<&str> = label.split(':').collect();
    let residue = ResidueId {
        chain: parts[1].to_string(),
        number: parts[2].parse().unwrap(),
        insertion_code: None,
    };
    let sequence = fx["sequence"].as_str().unwrap().to_string();
    // the reference's sequence string fixes the template atom order
    let site = load_site(
        &structure,
        &residue,
        &crystal,
        &SiteOptions {
            sequence: Some(sequence.clone()),
            ..SiteOptions::default()
        },
    )
    .unwrap();
    let resolution = parse_resolution(&text).unwrap();
    let library =
        glycoflow_core::ResidueLibrary::from_json_slice(&std::fs::read(library_path()).unwrap())
            .unwrap();
    let vocab: glycoflow_core::Vocab = serde_json::from_value(
        serde_json::json!({"atom_names": [], "residues": [], "link_codes": []}),
    )
    .unwrap();
    let glycan = build_glycan(&library, &sequence, 16, 0).unwrap();
    let radius = max_span(&glycan) + 2.5;
    let sigma = f(&fx["sigma"]);
    let env: Vec<SiteEnvironmentAtom> = site
        .environment
        .iter()
        .map(|a| SiteEnvironmentAtom {
            position: a.position,
            atomic_number: a.atomic_number,
        })
        .collect();
    let lik = SiteLikelihood::from_map(
        &map,
        site.anchor[2],
        &env,
        SiteLikelihoodOptions {
            sigma_angstrom: sigma,
            spacing_angstrom: 0.5,
            radius_angstrom: radius,
            resolution_angstrom: resolution,
            periodic: map.is_full_unit_cell(),
            independent_volume: None,
        },
    )
    .unwrap();
    let grid_box = (lik.obs_blur.origin, lik.obs_blur.spacing, lik.obs_blur.dims);
    let problem = SiteProblem::new(
        &site,
        glycan,
        &vocab,
        Box::new(DensityObservation::new(lik)),
        grid_box,
        radius,
        &ProblemOptions::default(),
    )
    .unwrap();
    Loaded {
        fx,
        problem,
        site,
        map,
        resolution,
    }
}

fn parity(name: &str) {
    let Loaded {
        fx,
        mut problem,
        site,
        map,
        resolution,
    } = load(name);
    println!("{name}: {}", fx["sequence"]);
    // --- site environment
    let env = &fx["environment"];
    let n = site.environment.len();
    println!(
        "  environment atoms rust {n} python {} (symmetry: {:?})",
        env["count"], site.symmetry.space_group
    );
    assert_eq!(
        n as u64,
        env["count"].as_u64().unwrap(),
        "environment atom count"
    );
    let sum: [f64; 3] = [0, 1, 2].map(|k| site.environment.iter().map(|a| a.position[k]).sum());
    for k in 0..3 {
        check(
            &format!("environment sum xyz[{k}]"),
            sum[k],
            f(&env["sum_xyz"][k]),
            1e-9,
            1.0,
        );
    }
    check(
        "environment sum Z",
        site.environment.iter().map(|a| a.atomic_number).sum(),
        f(&env["sum_z"]),
        0.0,
        1.0,
    );
    let site_names: Vec<&str> = env["site_atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(problem.site_names, site_names);
    assert_eq!(
        problem.site_pairs.len() as u64,
        env["site_pairs"].as_u64().unwrap(),
        "site pairs"
    );
    // --- sequence: the deposited glycan through crabWURCS gives the same tree
    let ours = &site.sequence;
    let auto = ours
        .crabwurcs_glycam
        .clone()
        .or(ours.deposited_tree_glycam.clone())
        .unwrap();
    println!("  crabWURCS GLYCAM {auto}");
    assert_eq!(
        sequence_paths(&auto).unwrap(),
        sequence_paths(fx["sequence"].as_str().unwrap()).unwrap(),
        "sequence tree"
    );
    // --- sigma calibration
    let cal = calibrate_sigma(&site, &map, resolution).unwrap();
    for (s, pcc) in &cal.curve {
        check(
            &format!("sigma curve {s}"),
            *pcc,
            f(&fx["sigma_curve"][format!("{s:?}")]),
            1e-3,
            1e-3,
        );
    }
    assert_eq!(cal.selected, f(&fx["sigma"]), "calibrated sigma");
    // --- likelihood constants
    let d = &fx["density"];
    check("radius", problem.radius, f(&d["radius"]), 1e-6, 1.0);
    let dens = reglyco_glycoflow::pipeline::density_constants(&problem).unwrap();
    assert_eq!(
        dens.dims.to_vec(),
        d["dims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    );
    check("theta0[0]", dens.theta[0], f(&d["theta0"][0]), 1e-4, 1e-3);
    check("theta0[1]", dens.theta[1], f(&d["theta0"][1]), 1e-4, 1e-3);
    check("SSE0", dens.sse0, f(&d["sse0"]), 1e-4, 1.0);
    check(
        "noise variance",
        dens.noise_variance,
        f(&d["noise_var"]),
        1e-4,
        1e-6,
    );
    check(
        "correlation volume",
        dens.independent_volume,
        f(&d["corr_volume"]),
        1e-4,
        1e-6,
    );
    check(
        "clash grid C sum",
        problem.grids.carbon.iter().map(|v| *v as f64).sum(),
        f(&d["grid_c_sum"]),
        1e-4,
        1.0,
    );
    check(
        "clash grid polar sum",
        problem.grids.polar.iter().map(|v| *v as f64).sum(),
        f(&d["grid_p_sum"]),
        1e-4,
        1.0,
    );
    // --- glycan and templates
    let g = &fx["glycan"];
    let names: Vec<&str> = g["atom_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let paths: Vec<&str> = g["res_paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(problem.glycan.atom_names, names);
    assert_eq!(problem.glycan.res_paths, paths);
    let quads: Vec<[usize; 4]> = g["quads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| {
            let q = q.as_array().unwrap();
            [0, 1, 2, 3].map(|k| q[k].as_u64().unwrap() as usize)
        })
        .collect();
    assert_eq!(problem.glycan.topology.quads, quads);
    let templates: Vec<Vec<[f64; 3]>> = g["templates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            t.as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let p = vec_f(p);
                    [p[0], p[1], p[2]]
                })
                .collect()
        })
        .collect();
    let dev = problem.templates[0]
        .iter()
        .zip(&templates[0])
        .map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    println!("  template 0 (majority puckers) max |rust - python| = {dev:.2e} A");
    assert!(dev < 1e-3, "majority-pucker template differs by {dev} A");
    problem.set_templates(templates).unwrap();
    let pr = &fx["prior"];
    let samples: Vec<f64> = pr["samples"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(vec_f)
        .collect();
    let prior = MarginalPrior::new(samples, problem.n_torsions, f(&pr["bandwidth_deg"]));
    check("prior kappa", prior.kappa, f(&pr["kappa"]), 1e-12, 1.0);
    check(
        "prior log normaliser",
        prior.log_norm,
        f(&pr["log_norm"]),
        1e-9,
        1.0,
    );
    problem.prior = Some(prior);
    // --- objective and gradients on the poses
    for pose in fx["poses"].as_array().unwrap() {
        println!(" pose: {}", pose["label"]);
        let p = Pose {
            tau: vec_f(&pose["tau"]),
            psi: f(&pose["psi"]),
            phi: f(&pose["phi"]),
            template: pose["template"].as_u64().unwrap() as usize,
        };
        let ev = problem.evaluate(&p, true);
        let xr: Vec<[f64; 3]> = pose["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let v = vec_f(v);
                [v[0], v[1], v[2]]
            })
            .collect();
        let dx =
            ev.x.iter()
                .zip(&xr)
                .map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max))
                .fold(0.0, f64::max);
        println!("  coordinates max |rust - python| = {dx:.2e} A");
        assert!(dx < 2e-3, "placed coordinates differ by {dx} A");
        let t = &pose["terms"];
        let terms = ev.terms;
        // the total is a sum of terms of either sign: its tolerance scales with their magnitude
        let scale = f(&t["loglik"]).abs()
            + 10.0 * (f(&t["e_env"]).abs() + f(&t["e_self"]).abs())
            + f(&t["e_att"]).abs()
            + f(&t["e_prior"]).abs();
        for (label, rust, floor) in [
            ("total", terms.total, scale),
            ("loglik", terms.loglik, 1e-2),
            ("partial_cc", terms.partial_cc, 1e-3),
            ("gain", terms.gain, 1e-1),
            ("e_env", terms.e_env, 1e-2),
            ("e_att", terms.e_att, 1e-3),
            ("e_prior", terms.e_prior, 1e-2),
        ] {
            check(label, rust, f(&t[label]), 1e-3, floor);
        }
        // E_self against the reference's float64 recomputation (torch.cdist float32 error
        // dominates the reference value of this small term; also checked loosely against it)
        check(
            "e_self (float64 ref)",
            terms.e_self,
            f(&pose["e_self_f64"]),
            1e-3,
            1e-2,
        );
        check(
            "e_self (torch float32)",
            terms.e_self,
            f(&t["e_self"]),
            1e-2,
            1e-2,
        );
        check(
            "loglik (float64 <g,g> ref)",
            terms.loglik,
            f(&pose["loglik_f64gg"]),
            1e-3,
            1e-2,
        );
        let gr = ev.grad.unwrap();
        let gp = &pose["grad"]["total"];
        let mut rust_all = gr.tau.clone();
        rust_all.extend([gr.psi, gr.phi]);
        let mut py_all = vec_f(&gp["tau"]);
        py_all.extend([f(&gp["psi"]), f(&gp["phi"])]);
        // per-term breakdown (weights switched off one at a time)
        let full = |pr: &SiteProblem| {
            let g = pr.evaluate(&p, true).grad.unwrap();
            let mut v = g.tau.clone();
            v.extend([g.psi, g.phi]);
            v
        };
        let mut parts = Vec::new();
        for term in ["env", "self", "prior"] {
            let saved = (problem.w_env, problem.w_self, problem.w_prior);
            match term {
                "env" => problem.w_env = 0.0,
                "self" => problem.w_self = 0.0,
                _ => problem.w_prior = 0.0,
            }
            let without = full(&problem);
            (problem.w_env, problem.w_self, problem.w_prior) = saved;
            let part: Vec<f64> = rust_all.iter().zip(&without).map(|(a, b)| a - b).collect();
            parts.push((term, part));
        }
        let mut attach = vec![0.0; rust_all.len()];
        attach[problem.n_torsions] = -problem.amide_kappa * p.psi.sin();
        let mut density = rust_all.clone();
        for (_, part) in &parts {
            for (d, v) in density.iter_mut().zip(part) {
                *d -= v;
            }
        }
        for (d, v) in density.iter_mut().zip(&attach) {
            *d -= v;
        }
        parts.push(("density", density));
        parts.push(("attach", attach));
        for (term, part) in &parts {
            let py = &pose["grad"][*term];
            let mut pv = vec_f(&py["tau"]);
            pv.extend([f(&py["psi"]), f(&py["phi"])]);
            let dot: f64 = part.iter().zip(&pv).map(|(a, b)| a * b).sum();
            let (nr, np) = (
                part.iter().map(|a| a * a).sum::<f64>().sqrt(),
                pv.iter().map(|a| a * a).sum::<f64>().sqrt(),
            );
            let diff = part
                .iter()
                .zip(&pv)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            println!(
                "    term {term:<8} |rust| {nr:>12.5} |python| {np:>12.5} cos {:.6} |diff| {diff:.4e}",
                dot / (nr * np).max(1e-300)
            );
        }
        // atoms on a grid plane of the blurred maps: the trilinear gradient jumps there, and the
        // reference's float32 kinematics (<= 2e-4 A from ours) can sit on the other side
        let origin = problem.grids.origin;
        let h = problem.grids.spacing;
        let kink = xr.iter().any(|p| {
            (0..3).any(|k| {
                let g = (p[k] - origin[k]) / h;
                (g - g.round()).abs() < 1e-3
            })
        });
        let scale = py_all.iter().map(|v| v * v).sum::<f64>().sqrt();
        if kink {
            println!(
                "  (an atom lies within 1e-3 grid units of a grid plane: gradient kink; full-kinematics gradient reported only)"
            );
            let dot: f64 = rust_all.iter().zip(&py_all).map(|(a, b)| a * b).sum();
            let cos = dot / (rust_all.iter().map(|a| a * a).sum::<f64>().sqrt() * scale);
            println!(
                "  full kinematics: cosine {cos:.6}, dE/dpsi {:.4} vs {:.4}, dE/dphi {:.4} vs {:.4}",
                gr.psi,
                f(&gp["psi"]),
                gr.phi,
                f(&gp["phi"])
            );
        } else {
            check_vec("dE/dtau", &gr.tau, &vec_f(&gp["tau"]));
            check_vec("dE/d(tau, psi, phi)", &rust_all, &py_all);
            check("dE/dpsi_N", gr.psi, f(&gp["psi"]), 2e-3, 0.1 * scale);
            check("dE/dphi_N", gr.phi, f(&gp["phi"]), 2e-3, 0.1 * scale);
        }
        // same objective and chain rule evaluated at the reference's placed coordinates
        let at_ref = problem.evaluate_placed(xr.clone(), &p, true);
        let ga = at_ref.grad.unwrap();
        let mut ref_all = ga.tau.clone();
        ref_all.extend([ga.psi, ga.phi]);
        check(
            "total at reference x",
            at_ref.terms.total,
            f(&t["total"]),
            1e-3,
            scale.max(1.0) * 0.0 + {
                f(&t["loglik"]).abs()
                    + 10.0 * (f(&t["e_env"]).abs() + f(&t["e_self"]).abs())
                    + f(&t["e_att"]).abs()
                    + f(&t["e_prior"]).abs()
            },
        );
        check_vec("dE/dtau at reference x", &ga.tau, &vec_f(&gp["tau"]));
        check_vec("dE/d(all) at reference x", &ref_all, &py_all);
        check(
            "dE/dpsi_N at reference x",
            ga.psi,
            f(&gp["psi"]),
            2e-3,
            0.1 * scale,
        );
        check(
            "dE/dphi_N at reference x",
            ga.phi,
            f(&gp["phi"]),
            2e-3,
            0.1 * scale,
        );
    }
    // --- support of the deposited-fit pose
    let p0 = &fx["poses"][0];
    let x0: Vec<[f64; 3]> = p0["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let v = vec_f(v);
            [v[0], v[1], v[2]]
        })
        .collect();
    let config = reglyco_glycoflow::FitConfig::default();
    let support = subtree_support(
        &problem,
        &x0,
        config.support_base,
        config.support_per_torsion,
    );
    for s in &support {
        let py = &fx["support"]["subtrees"][&s.residue];
        check(
            &format!("support gain {}", s.residue),
            s.gain_loglik,
            f(&py["gain_loglik"]),
            1e-3,
            1e-1,
        );
        assert_eq!(s.n_torsions as u64, py["n_torsions"].as_u64().unwrap());
        assert_eq!(
            s.supported,
            fx["support"]["classified"][&s.residue].as_bool().unwrap(),
            "support class of {}",
            s.residue
        );
    }
    // --- attachment grid search on the deposited-fit torsions
    let (psi, phi, e) = attach_search(
        &problem,
        &[(vec_f(&p0["tau"]), p0["template"].as_u64().unwrap() as usize)],
    )[0];
    let a = &fx["attach_search"];
    check("attach search psi", psi, f(&a["psi"]), 1e-5, 1.0);
    check("attach search phi", phi, f(&a["phi"]), 1e-5, 1.0);
    check("attach search objective", e, f(&a["e"]), 2e-3, 1e-2);
    // --- restrained Cartesian refinement at a perturbed pose
    if let Some(cfx) = fx.get("cartesian") {
        use reglyco_glycoflow::cartesian::{
            RestraintOptions, Restraints, cartesian_refine, coordinate_objective,
        };
        let pts = |v: &Value| -> Vec<[f64; 3]> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let p = vec_f(p);
                    [p[0], p[1], p[2]]
                })
                .collect()
        };
        let xc = pts(&cfx["x"]);
        let t = cfx["template"].as_u64().unwrap() as usize;
        let options = RestraintOptions::default();
        let r = Restraints::new(&problem, t, &options);
        let mut g = vec![[0.0; 3]; xc.len()];
        let (terms, psi) = coordinate_objective(&problem, &xc, Some(&mut g));
        let er = r.energy(&xc, Some(&mut g));
        // totals are sums of terms of either sign: tolerance relative to their magnitude
        let magnitude = |t: &reglyco_glycoflow::Terms| {
            t.loglik.abs()
                + 10.0 * (t.e_env.abs() + t.e_self.abs())
                + t.e_att.abs()
                + t.e_prior.abs()
        };
        check(
            "cartesian objective",
            terms.total,
            f(&cfx["total"]),
            1e-3,
            magnitude(&terms),
        );
        check(
            "cartesian prior",
            terms.e_prior,
            f(&cfx["e_prior"]),
            1e-4,
            1e-2,
        );
        check(
            "cartesian restraints",
            er,
            f(&cfx["e_restraint"]),
            1e-4,
            1e-2,
        );
        check("cartesian psi_N", psi, f(&cfx["psi"]), 1e-5, 1.0);
        let gp: Vec<f64> = pts(&cfx["grad"]).into_iter().flatten().collect();
        let gr: Vec<f64> = g.into_iter().flatten().collect();
        check_vec("cartesian dE/dx", &gr, &gp);
        let steps = cfx["steps"].as_u64().unwrap() as usize;
        let fit = cartesian_refine(&problem, &xc, t, steps, f(&cfx["lr"]), &options);
        check(
            "cartesian refined total",
            fit.total,
            f(&cfx["refined_total"]),
            1e-3,
            magnitude(&fit.terms) + fit.e_restraint,
        );
        let xr = pts(&cfx["refined_x"]);
        let shift = fit
            .x
            .iter()
            .zip(&xr)
            .map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>())
            .fold(0.0f64, f64::max)
            .sqrt();
        println!("  cartesian refined max |x_rust - x_python| {shift:.2e} A");
        if shift > 1e-2 {
            fail(format!("cartesian refined coordinates differ by {shift} A"));
        }
    }
    // --- in-place RMSD of the deposited-fit pose
    let dep = site.deposited.as_ref().unwrap();
    let rec = reglyco_glycoflow::evaluation::recovery(&problem, dep, &x0, &[]);
    let ev = &fx["evaluation"];
    assert_eq!(rec.matched_atoms as u64, ev["n_matched"].as_u64().unwrap());
    check(
        "full-tree RMSD",
        rec.full_rmsd,
        f(&ev["full_rmsd"]),
        1e-4,
        1e-2,
    );
    check("core RMSD", rec.core_rmsd, f(&ev["core_rmsd"]), 1e-4, 1e-2);
    finish(name);
}

/// The full pipeline on the reference's templates and prior samples (removes the RNG
/// differences of the 15 random-pucker templates and of the prior samples), seeds 0..3.
/// Needs GLYCOFLOW_FIT_FIXTURES and GLYCOFLOW_MODEL (model directory); CUDA when built with
/// `--features cuda` and GLYCOFLOW_DEVICE=cuda.
fn controlled_fit(name: &str) {
    use reglyco_glycoflow::glycoflow_core::model::Precision;
    use reglyco_glycoflow::glycoflow_core::sampler::Sampler;
    use reglyco_glycoflow::{ComputeDevice, GlycoflowModel};
    let Loaded {
        fx,
        mut problem,
        site,
        ..
    } = load(name);
    let dir = GlycoflowModel::resolve_dir(None).expect("set GLYCOFLOW_MODEL");
    let device = if std::env::var("GLYCOFLOW_DEVICE").as_deref() == Ok("cuda") {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let model = GlycoflowModel::load(&dir, device, Precision::F32).unwrap();
    // tokens from the real vocabulary
    problem.tokens = problem.glycan.tokens(&model.meta.vocab);
    let g = &fx["glycan"];
    let templates: Vec<Vec<[f64; 3]>> = g["templates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            t.as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let p = vec_f(p);
                    [p[0], p[1], p[2]]
                })
                .collect()
        })
        .collect();
    problem.set_templates(templates).unwrap();
    let pr = &fx["prior"];
    let samples: Vec<f64> = pr["samples"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(vec_f)
        .collect();
    problem.prior = Some(MarginalPrior::new(
        samples,
        problem.n_torsions,
        f(&pr["bandwidth_deg"]),
    ));
    let sampler = Sampler::for_glycan(&model.net, &problem.glycan, &model.meta.vocab).unwrap();
    for seed in 0..3u64 {
        let config = reglyco_glycoflow::FitConfig {
            seed,
            ..Default::default()
        };
        let out = reglyco_glycoflow::fit_site(&mut problem, &sampler, &config).unwrap();
        let best = &out.basins[out.best];
        let rec = reglyco_glycoflow::evaluation::recovery(
            &problem,
            site.deposited.as_ref().unwrap(),
            &best.x,
            &[],
        );
        let t = best.terms;
        println!(
            "{name} controlled seed {seed}: total {:.2} loglik {:.2} e_env {:.2} e_self {:.2} e_att {:.2} e_prior {:.2} full {:.2} core {:.2} supported {}/{} ({:.1}s)",
            t.total,
            t.loglik,
            t.e_env,
            t.e_self,
            t.e_att,
            t.e_prior,
            rec.full_rmsd,
            rec.core_rmsd,
            out.support.iter().filter(|s| s.supported).count() + 1,
            out.support.len() + 1,
            out.wall_seconds
        );
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES, GLYCOFLOW_MODEL, the 5GSQ model and map"]
fn controlled_fit_5gsq_a297() {
    controlled_fit("5GSQ_A297");
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES, GLYCOFLOW_MODEL, the 5GSQ model and map"]
fn controlled_fit_5gsq_b297() {
    controlled_fit("5GSQ_B297");
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES, GLYCOFLOW_MODEL, the 5KZC model and map"]
fn controlled_fit_5kzc_a79() {
    controlled_fit("5KZC_A79");
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES (GlycoFlow export_fit_fixtures.py), the 5KZC model and map"]
fn parity_5kzc_a79() {
    parity("5KZC_A79");
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES (GlycoFlow export_fit_fixtures.py), the 5GSQ model and map"]
fn parity_5gsq_a297() {
    parity("5GSQ_A297");
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES (GlycoFlow export_fit_fixtures.py), the 5GSQ model and map"]
fn parity_5gsq_b297() {
    parity("5GSQ_B297");
}
