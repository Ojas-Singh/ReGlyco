//! Evaluation of fitted poses against the deposited glycan (`glycoflow/fitting/evaluation.py`).
//! Never used by the fitting itself.

use std::collections::BTreeMap;

use crate::cartesian::coordinate_objective;
use crate::problem::{SiteProblem, Terms, V3};
use crate::site::{DepositedGlycan, Site};

/// Core of an N-glycan: the chitobiose and the beta-mannose.
pub const CORE: [&str; 3] = ["r", "r/4", "r/4/4"];

#[derive(Debug, Clone, serde::Serialize)]
pub struct Recovery {
    /// template atoms matched to deposited atoms (residue path + atom name)
    pub matched_atoms: usize,
    pub full_rmsd: f64,
    pub core_rmsd: f64,
    /// RMSD over the root and the residues the fit itself calls supported
    pub claimed_supported_rmsd: f64,
    pub claimed_supported: Vec<String>,
    pub per_residue: BTreeMap<String, f64>,
}

/// In-place RMSD of a placed pose to the deposited glycan (no superposition), matching atoms by
/// residue tree path and GLYCAM atom name.
pub fn recovery(
    problem: &SiteProblem,
    deposited: &DepositedGlycan,
    x: &[V3],
    claimed: &[String],
) -> Recovery {
    let mut pairs: Vec<(String, f64)> = Vec::new();
    for (i, xi) in x.iter().enumerate().take(problem.n_atoms) {
        let path = &problem.glycan.res_paths[i];
        if path == "agl" {
            continue;
        }
        if let Some(r) = deposited
            .atoms
            .get(&(path.clone(), problem.glycan.atom_names[i].clone()))
        {
            let d2 = (0..3).map(|k| (xi[k] - r[k]).powi(2)).sum::<f64>();
            pairs.push((path.clone(), d2));
        }
    }
    let rmsd = |sel: &dyn Fn(&str) -> bool| {
        let v: Vec<f64> = pairs
            .iter()
            .filter(|(p, _)| sel(p))
            .map(|(_, d)| *d)
            .collect();
        if v.is_empty() {
            f64::NAN
        } else {
            (v.iter().sum::<f64>() / v.len() as f64).sqrt()
        }
    };
    let mut per_residue = BTreeMap::new();
    for (p, _) in &pairs {
        if !per_residue.contains_key(p) {
            per_residue.insert(p.clone(), rmsd(&|q| q == p));
        }
    }
    Recovery {
        matched_atoms: pairs.len(),
        full_rmsd: rmsd(&|_| true),
        core_rmsd: rmsd(&|p| CORE.contains(&p)),
        claimed_supported_rmsd: rmsd(&|p| claimed.iter().any(|c| c == p)),
        claimed_supported: claimed.to_vec(),
        per_residue,
    }
}

/// Glycan atoms closer than `cutoff` to an environment atom (`evaluation.Evaluator`,
/// `env_contacts_below_2.2A`): scored atoms against the environment without the site residue's
/// linking side chain, plus the explicit site pairs beyond three bonds.
pub fn contacts_below(problem: &SiteProblem, site: &Site, x: &[V3], cutoff: f64) -> usize {
    let c2 = cutoff * cutoff;
    let env: Vec<V3> = site
        .environment
        .iter()
        .filter(|a| !site.is_site_atom(a))
        .map(|a| a.position)
        .collect();
    let d2 = |a: V3, b: V3| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
    let grid = x
        .iter()
        .zip(&problem.keep)
        .filter(|(p, keep)| **keep && env.iter().any(|e| d2(**p, *e) < c2))
        .count();
    let pairs = problem
        .site_pairs
        .iter()
        .filter(|(i, j, _)| d2(x[*i], problem.site_xyz[*j]) < c2)
        .count();
    grid + pairs
}

/// The deposited glycan scored by the fitting objective (the problem's current weights, no
/// restraints; psi_N and prior torsions measured on the coordinates).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DepositedScore {
    pub terms: Terms,
    /// scored template atoms present in the deposit (missing ones take the fitted positions)
    pub matched_atoms: usize,
    pub scored_atoms: usize,
}

/// Score the deposited coordinates of the glycan with the same objective as the fit, so a fit
/// can be compared with the deposited model on the map (`fallback`: the fitted pose, used for
/// atoms the deposit lacks).
pub fn deposited_score(
    problem: &SiteProblem,
    deposited: &DepositedGlycan,
    fallback: &[V3],
) -> DepositedScore {
    let mut x = fallback.to_vec();
    let (mut matched, mut scored) = (0, 0);
    for (i, xi) in x.iter_mut().enumerate().take(problem.n_atoms) {
        if !problem.keep[i] {
            continue;
        }
        scored += 1;
        let key = (
            problem.glycan.res_paths[i].clone(),
            problem.glycan.atom_names[i].clone(),
        );
        if let Some(r) = deposited.atoms.get(&key) {
            *xi = *r;
            matched += 1;
        }
    }
    let (terms, _) = coordinate_objective(problem, &x, None);
    DepositedScore {
        terms,
        matched_atoms: matched,
        scored_atoms: scored,
    }
}
