//! Offline check of the objective and its analytic gradients on a synthetic site: a small
//! protein-like environment, a density box built from a known glycan pose, a marginal prior from
//! synthetic samples. Gradients with respect to torsions and both attachment angles are compared
//! with central finite differences of the total objective. Needs only the GlycoFlow residue
//! library of the sibling GlycoFlow checkout (no model, no map, no network).

use std::path::Path;

use glycoflow_core::rng::SplitMix64;
use reglyco_density::site_likelihood::{
    SiteBox, SiteEnvironmentAtom, SiteLikelihood, SiteLikelihoodOptions, splat,
};
use reglyco_glycoflow::observation::DensityObservation;
use reglyco_glycoflow::prior::MarginalPrior;
use reglyco_glycoflow::problem::{Pose, ProblemOptions, SiteProblem, build_glycan};
use reglyco_glycoflow::site::{SequenceProvenance, Site, SymmetryInfo};
use reglyco_glycoflow::symmetry::EnvAtom;

const SEQUENCE: &str = "DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH";

fn library() -> glycoflow_core::ResidueLibrary {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../GlycoFlow/glycoflow/resources/residue_library.json");
    glycoflow_core::ResidueLibrary::from_json_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

fn atom(name: &str, residue: i32, z: f64, p: [f64; 3]) -> EnvAtom {
    EnvAtom {
        position: p,
        atomic_number: z,
        chain: "A".into(),
        residue_number: residue,
        residue_name: if residue == 10 {
            "ASN".into()
        } else {
            "ALA".into()
        },
        atom_name: name.into(),
        op: 0,
        shift: [0, 0, 0],
    }
}

fn site() -> Site {
    // Asn side chain ending at ND2 (origin), and a slab of "protein" atoms below the site
    let anchor = [[-1.0, -2.3, 0.4], [-0.6, -1.1, -0.1], [0.0, 0.0, 0.0]];
    let mut env = vec![
        atom("CA", 10, 6.0, [-2.3, -2.6, 0.2]),
        atom("CB", 10, 6.0, anchor[0]),
        atom("CG", 10, 6.0, anchor[1]),
        atom("OD1", 10, 8.0, [-0.8, -0.9, -1.3]),
        atom("ND2", 10, 7.0, anchor[2]),
    ];
    let mut rng = SplitMix64::new(7);
    for k in 0..160 {
        let p = [
            rng.uniform_f64() * 24.0 - 12.0,
            -4.5 - rng.uniform_f64() * 6.0,
            rng.uniform_f64() * 24.0 - 12.0,
        ];
        let (name, z) = if k % 3 == 0 { ("O", 8.0) } else { ("C", 6.0) };
        env.push(atom(name, 20 + k, z, p));
    }
    Site {
        residue: glysys::ResidueId {
            chain: "A".into(),
            number: 10,
            insertion_code: None,
        },
        residue_name: "ASN".into(),
        anchor_names: ["CB".into(), "CG".into(), "ND2".into()],
        anchor,
        deposited: None,
        sequence: SequenceProvenance {
            sequence: SEQUENCE.into(),
            source: "requested".into(),
            crabwurcs_glycam: None,
            deposited_tree_glycam: None,
            warnings: Vec::new(),
        },
        environment: env,
        symmetry: SymmetryInfo {
            applied: false,
            space_group: None,
            space_group_number: None,
            operators: 1,
            cell: None,
            reason: "synthetic".into(),
        },
    }
}

fn problem(site: &Site, observed: Option<SiteBox>) -> SiteProblem {
    let glycan = build_glycan(&library(), SEQUENCE, 4, 0).unwrap();
    let options = SiteLikelihoodOptions {
        sigma_angstrom: 0.9,
        spacing_angstrom: 0.5,
        radius_angstrom: 14.0,
        resolution_angstrom: 2.5,
        periodic: false,
        independent_volume: None,
    };
    let env: Vec<SiteEnvironmentAtom> = site
        .environment
        .iter()
        .map(|a| SiteEnvironmentAtom {
            position: a.position,
            atomic_number: a.atomic_number,
        })
        .collect();
    let (origin, dims) = SiteLikelihood::box_geometry(site.anchor[2], &options);
    let observed = observed.unwrap_or_else(|| {
        let mut b = SiteBox::zeros(origin, 0.5, dims);
        splat(&env, &mut b, 0.81);
        b
    });
    let lik = SiteLikelihood::from_observed_box(observed, site.anchor[2], &env, options).unwrap();
    let grid_box = (lik.obs_blur.origin, lik.obs_blur.spacing, lik.obs_blur.dims);
    let vocab: glycoflow_core::Vocab = serde_json::from_value(
        serde_json::json!({"atom_names": [], "residues": [], "link_codes": []}),
    )
    .unwrap();
    SiteProblem::new(
        site,
        glycan,
        &vocab,
        Box::new(DensityObservation::new(lik)),
        grid_box,
        14.0,
        &ProblemOptions::default(),
    )
    .unwrap()
}

#[test]
fn objective_gradients_match_finite_differences() {
    let site = site();
    let draft = problem(&site, None);
    let nt = draft.n_torsions;
    let template_tau: Vec<f64> = draft
        .glycan
        .torsions(&draft.glycan.templates[0])
        .iter()
        .map(|v| *v as f64)
        .collect();
    let truth = Pose {
        tau: template_tau.clone(),
        psi: 3.0,
        phi: -1.4,
        template: 0,
    };
    // observed map: environment + the true glycan (+ a little structured noise)
    let x_truth = draft.place(&truth);
    let mut atoms: Vec<SiteEnvironmentAtom> = site
        .environment
        .iter()
        .map(|a| SiteEnvironmentAtom {
            position: a.position,
            atomic_number: a.atomic_number,
        })
        .collect();
    for (i, p) in x_truth.iter().enumerate() {
        if draft.keep[i] {
            let z = match draft.glycan.elements[i].as_str() {
                "C" => 6.0,
                "N" => 7.0,
                _ => 8.0,
            };
            atoms.push(SiteEnvironmentAtom {
                position: *p,
                atomic_number: z,
            });
        }
    }
    let lik = draft.observation.density().unwrap();
    let mut observed = SiteBox::zeros(lik.obs_blur.origin, 0.5, lik.obs_blur.dims);
    splat(&atoms, &mut observed, 0.81);
    let mut rng = SplitMix64::new(11);
    for v in &mut observed.data {
        *v += (rng.uniform_f64() as f32 - 0.5) * 0.02;
    }
    let mut problem = problem(&site, Some(observed));
    let samples: Vec<f64> = (0..64)
        .flat_map(|_| {
            template_tau
                .iter()
                .map(|t| t + (rng.uniform_f64() - 0.5) * 0.8)
                .collect::<Vec<_>>()
        })
        .collect();
    problem.prior = Some(MarginalPrior::new(samples, nt, 12.0));

    let mut poses = vec![truth.clone()];
    for k in 0..4 {
        let spread = [0.15, 0.4, 1.0, 3.0][k];
        poses.push(Pose {
            tau: template_tau
                .iter()
                .map(|t| t + (rng.uniform_f64() - 0.5) * spread)
                .collect(),
            psi: 3.0 + (rng.uniform_f64() - 0.5) * 0.4,
            phi: -1.4 + (rng.uniform_f64() - 0.5) * spread,
            template: k % problem.n_templates(),
        });
    }
    let truth_terms = problem.evaluate(&truth, false).terms;
    assert!(
        truth_terms.loglik > 0.0,
        "the true pose explains the synthetic map"
    );
    let mut active = [false; 4];
    for (k, pose) in poses.iter().enumerate() {
        let ev = problem.evaluate(pose, true);
        let t = ev.terms;
        active[0] |= t.loglik > 0.0;
        active[1] |= t.e_env > 0.0;
        active[2] |= t.e_self > 0.0;
        active[3] |= t.e_prior != 0.0;
        if k > 0 {
            assert!(
                t.loglik < truth_terms.loglik,
                "pose {k} explains the map better than the truth"
            );
        }
        let g = ev.grad.unwrap();
        let eps = 1e-6;
        let total = |p: &Pose| problem.evaluate(p, false).terms.total;
        let check = |label: String, analytic: f64, plus: Pose, minus: Pose| {
            let fd = (total(&plus) - total(&minus)) / (2.0 * eps);
            let tol = 1e-4 * fd.abs().max(analytic.abs()).max(1.0);
            assert!(
                (analytic - fd).abs() <= tol,
                "pose {k} {label}: analytic {analytic} vs finite difference {fd} (terms {t:?})"
            );
        };
        for i in 0..nt {
            let (mut plus, mut minus) = (pose.clone(), pose.clone());
            plus.tau[i] += eps;
            minus.tau[i] -= eps;
            check(format!("dE/dtau[{i}]"), g.tau[i], plus, minus);
        }
        let (mut plus, mut minus) = (pose.clone(), pose.clone());
        plus.psi += eps;
        minus.psi -= eps;
        check("dE/dpsi_N".into(), g.psi, plus, minus);
        let (mut plus, mut minus) = (pose.clone(), pose.clone());
        plus.phi += eps;
        minus.phi -= eps;
        check("dE/dphi_N".into(), g.phi, plus, minus);
    }
    assert_eq!(
        active, [true; 4],
        "every objective term is exercised (density, env, self, prior)"
    );
}

#[test]
fn attachment_reproduces_the_requested_angles() {
    let site = site();
    let problem = problem(&site, None);
    let tau: Vec<f64> = problem
        .glycan
        .torsions(&problem.glycan.templates[1])
        .iter()
        .map(|v| *v as f64)
        .collect();
    for (psi, phi) in [(3.1, -1.2), (2.7, 0.4), (-2.9, 2.5)] {
        let x = problem.place(&Pose {
            tau: tau.clone(),
            psi,
            phi,
            template: 1,
        });
        let (p, f) = problem.attachment_angles(&x);
        assert!(
            (p - psi).abs() < 1e-9 && (f - phi).abs() < 1e-9,
            "({p}, {f}) vs ({psi}, {phi})"
        );
        let link = site.anchor[2];
        let c1 = x[problem.c1];
        let bond =
            ((c1[0] - link[0]).powi(2) + (c1[1] - link[1]).powi(2) + (c1[2] - link[2]).powi(2))
                .sqrt();
        assert!((bond - 1.45).abs() < 1e-9);
    }
}
