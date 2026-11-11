//! Atom positions known before a fit (a deposited model, an earlier fit of a smaller glycan) as
//! starts of the fit.
//!
//! The torsions measured on them join the search basins ([`pose_from_atoms`]), and the positions
//! themselves, completed from the lowest basins ([`hybrid`]), join the restrained Cartesian
//! refinement as basins of their own. Every basin is ranked on the same objective, so the selected
//! model is never worse by it than the known model with ideal geometry.

use std::collections::{BTreeMap, BTreeSet};

use crate::ensemble::superpose;
use crate::problem::{Pose, SiteProblem, V3, dihedral};
use crate::ring::chair_theta;
use crate::support::residue_tree;

/// Known positions of glycan atoms, by residue tree path and GLYCAM atom name.
#[derive(Debug, Clone, Default)]
pub struct KnownAtoms {
    /// (residue path, GLYCAM atom name) -> position (heavy atoms)
    pub atoms: BTreeMap<(String, String), V3>,
    /// residues whose known atoms only place the residue as a rigid body, e.g. one whose anomer
    /// the fitted sequence corrects (its known atoms have the other configuration)
    pub rigid: BTreeSet<String>,
}

/// [`KnownAtoms`] on the atoms of one problem.
#[derive(Debug, Clone, Default)]
pub struct CoordinateStart {
    /// position of every glycan atom where known
    pub x: Vec<Option<V3>>,
    /// atoms that start at their known position itself
    pub exact: Vec<bool>,
}

/// An error in the geometry of a known residue, which no fit starts from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryError {
    /// the N-acetyl group is cis, its methyl and carbonyl oxygen exchanged (C2-N2-C2N-CME in
    /// degrees; 180 in every N-acetyl sugar)
    Acetyl(f64),
    /// the pyranose ring is not in its chair (Cremer-Pople theta from that chair, degrees)
    Ring(f64),
}

pub use crate::ring::CHAIR_TOLERANCE;

impl KnownAtoms {
    fn at(&self, path: &str, name: &str) -> Option<V3> {
        self.atoms.get(&(path.to_string(), name.to_string())).copied()
    }

    /// C2-N2-C2N-CME of the N-acetyl group of a residue (degrees).
    pub fn acetyl_torsion(&self, path: &str) -> Option<f64> {
        let at = |name| self.at(path, name);
        Some(dihedral(at("C2")?, at("N2")?, at("C2N")?, at("CME")?).to_degrees())
    }

    /// Cremer-Pople theta of the ring of an aldohexopyranose from its chair (degrees): 4C1 for
    /// a D sugar, 1C4 for an L sugar (`residue`: the GLYCAM residue, e.g. `DGlcpNAc`).
    pub fn chair_deviation(&self, path: &str, residue: &str) -> Option<f64> {
        if residue.as_bytes().get(4) != Some(&b'p') {
            return None;
        }
        let at = |name| self.at(path, name);
        let ring = [at("O5")?, at("C1")?, at("C2")?, at("C3")?, at("C4")?, at("C5")?];
        let bonded = (0..6).all(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % 6]);
            (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>() < 2.0f64.powi(2)
        });
        let theta = chair_theta(&ring);
        bonded.then_some(if residue.starts_with('L') { 180.0 - theta } else { theta })
    }

    /// The errors of the known geometry (`residues`: path -> GLYCAM residue).
    pub fn geometry_errors(
        &self,
        residues: &BTreeMap<String, String>,
    ) -> Vec<(String, GeometryError)> {
        let mut errors = Vec::new();
        for (path, residue) in residues {
            if let Some(t) = self.acetyl_torsion(path).filter(|t| t.abs() < 90.0) {
                errors.push((path.clone(), GeometryError::Acetyl(t)));
            }
            if let Some(t) = self.chair_deviation(path, residue).filter(|t| *t > CHAIR_TOLERANCE) {
                errors.push((path.clone(), GeometryError::Ring(t)));
            }
        }
        errors
    }

    /// The known atoms without those errors: the methyl and the oxygen of a cis N-acetyl group
    /// are left out (they follow the residue), and a residue with a distorted ring only takes
    /// its place from its atoms.
    pub fn without(mut self, errors: &[(String, GeometryError)]) -> Self {
        for (path, error) in errors {
            match error {
                GeometryError::Acetyl(_) => {
                    for name in ["CME", "O2N"] {
                        self.atoms.remove(&(path.clone(), name.to_string()));
                    }
                }
                GeometryError::Ring(_) => {
                    self.rigid.insert(path.clone());
                }
            }
        }
        self
    }

    /// The placed atoms of a fit (`paths`: of these residues only).
    pub fn of_fit(problem: &SiteProblem, x: &[V3], paths: Option<&BTreeSet<String>>) -> Self {
        let g = &problem.glycan;
        let atoms = (0..problem.n_atoms)
            .filter(|&i| problem.keep[i] && paths.is_none_or(|p| p.contains(&g.res_paths[i])))
            .map(|i| ((g.res_paths[i].clone(), g.atom_names[i].clone()), x[i]))
            .collect();
        Self {
            atoms,
            rigid: BTreeSet::new(),
        }
    }

    pub fn on(&self, problem: &SiteProblem) -> CoordinateStart {
        let g = &problem.glycan;
        let x: Vec<Option<V3>> = (0..problem.n_atoms)
            .map(|i| {
                self.atoms
                    .get(&(g.res_paths[i].clone(), g.atom_names[i].clone()))
                    .copied()
                    .filter(|_| problem.movable[i])
            })
            .collect();
        let exact = (0..problem.n_atoms)
            .map(|i| x[i].is_some() && !self.rigid.contains(&g.res_paths[i]))
            .collect();
        CoordinateStart { x, exact }
    }
}

/// The pose of `problem` closest to known atom positions: torsions measured wherever all four
/// atoms are known (others, such as hydroxyl rotamers, from `fallback`), the attachment angles
/// measured on them, and the pucker template closest to them in place.
pub fn pose_from_atoms(
    problem: &SiteProblem,
    at: impl Fn(usize) -> Option<V3>,
    fallback: &Pose,
) -> Pose {
    pose_among(problem, at, fallback, |_| true)
}

/// The pucker template of a start from known atoms: the closest to them ([`pose_from_atoms`])
/// among those that have every residue placed as a rigid body in the ring of the majority build.
/// Such a residue takes its ring from the template, and a known ring out of its chair is a
/// reason to place it so: the closest template to a boat is a boat.
pub fn start_template(problem: &SiteProblem, start: &CoordinateStart, fallback: &Pose) -> usize {
    let g = &problem.glycan;
    // a root that the site binds in both chairs keeps the choice between them
    let both = crate::anchor::anchor(&problem.residue_name).is_some_and(|a| a.both_chairs);
    let rigid: BTreeSet<&str> = (0..problem.n_atoms)
        .filter(|&i| start.x[i].is_some() && !start.exact[i])
        .map(|i| g.res_paths[i].as_str())
        .filter(|path| !(both && *path == "r"))
        .collect();
    let rings: Vec<[usize; 6]> = rigid
        .iter()
        .filter_map(|path| {
            let atom = |name: &str| {
                (0..problem.n_atoms).find(|&i| g.res_paths[i] == *path && g.atom_names[i] == name)
            };
            Some([atom("O5")?, atom("C1")?, atom("C2")?, atom("C3")?, atom("C4")?, atom("C5")?])
        })
        .collect();
    let theta = |template: usize, ring: &[usize; 6]| {
        chair_theta(&ring.map(|i| problem.templates[template][i]))
    };
    let allowed = |template: usize| {
        rings
            .iter()
            .all(|ring| (theta(template, ring) - theta(0, ring)).abs() <= CHAIR_TOLERANCE)
    };
    pose_among(problem, |i| start.x[i], fallback, allowed).template
}

fn pose_among(
    problem: &SiteProblem,
    at: impl Fn(usize) -> Option<V3>,
    fallback: &Pose,
    allowed: impl Fn(usize) -> bool,
) -> Pose {
    let tau: Vec<f64> = problem
        .glycan
        .topology
        .quads
        .iter()
        .zip(&fallback.tau)
        .map(|(q, &t)| match (at(q[0]), at(q[1]), at(q[2]), at(q[3])) {
            (Some(a), Some(b), Some(c), Some(d)) => dihedral(a, b, c, d),
            _ => t,
        })
        .collect();
    let (psi, phi) = match (at(problem.c1), at(problem.o5)) {
        (Some(c1), Some(o5)) => {
            let [a, b, l] = problem.anchor;
            (dihedral(a, b, l, c1), dihedral(b, l, c1, o5))
        }
        _ => (fallback.psi, fallback.phi),
    };
    let msd = |template: usize| {
        let x = problem.place(&Pose {
            tau: tau.clone(),
            psi,
            phi,
            template,
        });
        let (mut s, mut n) = (0.0, 0usize);
        for (i, xi) in x.iter().enumerate().take(problem.n_atoms) {
            if let (true, Some(r)) = (problem.keep[i], at(i)) {
                s += (0..3).map(|k| (xi[k] - r[k]).powi(2)).sum::<f64>();
                n += 1;
            }
        }
        s / n.max(1) as f64
    };
    let template = (0..problem.templates.len())
        .filter(|&t| allowed(t))
        .min_by(|&a, &b| msd(a).total_cmp(&msd(b)))
        .unwrap_or(0);
    Pose {
        tau,
        psi,
        phi,
        template,
    }
}

/// The placed glycan `base` moved onto the known atoms: from the root outwards, every residue
/// with at least four known atoms is superposed on them, carrying its subtree; then the exact
/// atoms take their known positions. Residues without known atoms keep the conformation of
/// `base` on their moved parent, so the bonds between the two parts stay close to ideal.
pub fn hybrid(problem: &SiteProblem, base: &[V3], start: &CoordinateStart) -> Vec<V3> {
    let mut x = base.to_vec();
    let paths = &problem.glycan.res_paths;
    for (root, members) in residue_tree(problem) {
        let pairs: Vec<(usize, V3)> = (0..problem.n_atoms)
            .filter(|&i| paths[i] == root)
            .filter_map(|i| start.x[i].map(|k| (i, k)))
            .collect();
        if pairs.len() < 4 {
            continue;
        }
        let from: Vec<V3> = pairs.iter().map(|(i, _)| x[*i]).collect();
        let to: Vec<V3> = pairs.iter().map(|(_, k)| *k).collect();
        let (r, t) = superpose(&from, &to);
        for i in 0..problem.n_atoms {
            if problem.movable[i] && members.contains(&paths[i]) {
                let v = x[i];
                x[i] = [0, 1, 2].map(|k| r[k][0] * v[0] + r[k][1] * v[1] + r[k][2] * v[2] + t[k]);
            }
        }
    }
    for i in 0..problem.n_atoms {
        if let (true, Some(k)) = (start.exact[i], start.x[i]) {
            x[i] = k;
        }
    }
    x
}
