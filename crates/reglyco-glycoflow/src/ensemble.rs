//! Past the density: an ensemble instead of a model (the extend mode of the fit;
//! `glycoflow/fitting/ensemble.py`).
//!
//! A fit of a glycan larger than its density builds the residues the map supports
//! ([`crate::infer::Gate`]) and leaves the rest to GlycoFlow: [`extend`] regenerates everything
//! beyond the built residues many times, conditioned on the built part and on the protein around
//! it, and returns the conformers together with what they say about the density that is missing:
//!
//! * members: GlycoFlow completions (pinned-torsion inpainting with clash guidance,
//!   [`crate::support::complete_on_templates`]) on every pucker template that agrees with the fit
//!   on the built residues, grafted onto the built model so that every member shares its atoms. A
//!   member that touches the protein or itself is moved off by the contact energies alone, first
//!   on its free torsions, then atom by atom with its template's geometry as restraints (no
//!   density, and no prior: minimising the prior would pull every member to its mode and remove
//!   the spread that is the point). Members are kept only when the atoms beyond the built residues
//!   clear the contact floors;
//! * clusters: representative conformers (medoids, in place: the built part anchors them) with
//!   populations: as few as bring every member within `cluster_rmsd` of one, at most
//!   [`MAX_CLUSTERS`];
//! * per residue: the spread of its atoms, `order` (the density an ensemble this wide leaves,
//!   relative to an ordered residue: <rho_mean, rho_mean> / <rho, rho> with Gaussian atoms of the
//!   fit's width; for Gaussian disorder this is also the ratio of the peak densities), and from it
//!   the density level that flexibility alone predicts (`order` times the level of the built
//!   residue the subtree hangs from, whose own motion the map already shows), next to the level
//!   the map has where the ensemble puts the residue.
//!
//! Nothing here is fitted to density and nothing here is evidence for a residue: it is the model's
//! account of what the map does not show.

use std::collections::BTreeSet;

use glycoflow_core::geometry::torsion_gradient;
use glycoflow_core::guidance::clash_energy_grad;
use glycoflow_core::sampler::Sampler;
use rayon::prelude::*;
use reglyco_density::DensityMap;
use serde::Serialize;

use crate::cartesian::{RestraintOptions, Restraints};
use crate::error::Result;
use crate::infer::{DensityLevels, Gate};
use crate::pipeline::FitOutcome;
use crate::problem::{Pose, SiteProblem, V3};
use crate::search::Adam;
use crate::support::complete_on_templates;

/// A, in place, over the atoms beyond the built residues (as `search::distinct`).
pub const CLUSTER_RMSD: f64 = 1.5;
pub const MAX_CLUSTERS: usize = 8;
/// Density levels (fraction solvent -> protein) closer than this agree.
pub const READING_MARGIN: f64 = 0.15;
/// Order parameter from which the ensemble holds a residue in place.
pub const ORDERED: f64 = 0.8;
/// Ring signatures within this are one pucker state: in the library the medoids of one state
/// differ by up to 1.5 (pyranoses), the two chairs by 5 or more.
pub const PUCKER_TOLERANCE: f64 = 2.0;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EnsembleOptions {
    /// completions generated (members are those that clear the contact floors)
    pub members: usize,
    /// Euler steps of the inpainting flow
    pub steps: usize,
    /// clash guidance of the inpainting flow
    pub guidance: f32,
    /// torsion-space steps off the contacts
    pub relax_steps: usize,
    /// Cartesian steps off the contacts that remain after grafting
    pub clear_steps: usize,
    pub cluster_rmsd: f64,
    /// raw contact energy (E_env + E_self of the atoms beyond the built residues) a member may keep
    pub contact_tolerance: f64,
    pub seed: u64,
}

impl Default for EnsembleOptions {
    fn default() -> Self {
        Self {
            members: 128,
            steps: 32,
            guidance: 0.3,
            relax_steps: 200,
            clear_steps: 150,
            cluster_rmsd: CLUSTER_RMSD,
            contact_tolerance: 0.0064,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Cluster {
    /// index of the representative member
    pub medoid: usize,
    pub members: Vec<usize>,
    pub population: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidueFlexibility {
    pub residue: String,
    /// the built residue the subtree hangs from
    pub anchor: String,
    /// spread of the residue's atoms over the members (A, in place)
    pub rmsf: f64,
    /// density the ensemble leaves, relative to an ordered residue
    pub order: f64,
    /// `order` x the density level of the anchor (None without map levels)
    pub density_predicted: Option<f64>,
    /// the map's level where the members put the residue
    pub density_observed: Option<f64>,
    pub reading: Option<&'static str>,
}

/// The ensemble beyond the built residues of a fit.
#[derive(Debug, Clone, Serialize)]
pub struct Ensemble {
    /// residue paths beyond the built ones, by depth
    pub region: Vec<String>,
    /// members that clear the contact floors: placed coordinates in the problem's atom order (the
    /// built residues are those of the fit)
    #[serde(skip)]
    pub members: Vec<Vec<V3>>,
    /// pucker template of every member
    pub templates: Vec<usize>,
    pub generated: usize,
    pub kept: usize,
    /// pucker templates in the built residues' pucker states
    pub compatible_templates: usize,
    pub clusters: Vec<Cluster>,
    /// largest distance of a member to its medoid (A)
    pub cluster_rmsd: f64,
    pub residues: Vec<ResidueFlexibility>,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn wrap(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    a - two_pi * ((a + std::f64::consts::PI) / two_pi).floor()
}

/// Shape of the ring atoms `idx`: their distances and the signed volumes of every four of them
/// (distances alone do not tell a chair from its mirror image, 4C1 from 1C4).
pub fn ring_signature(x: &[[f32; 3]], idx: &[usize]) -> Vec<f64> {
    let r: Vec<V3> = idx.iter().map(|&i| x[i].map(f64::from)).collect();
    let n = r.len();
    let mut out = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            out.push(dot(sub(r[a], r[b]), sub(r[a], r[b])).sqrt());
        }
    }
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                for d in c + 1..n {
                    out.push(dot(sub(r[b], r[a]), cross(sub(r[c], r[a]), sub(r[d], r[a]))));
                }
            }
        }
    }
    out
}

/// Pucker templates whose rings are in the pucker state of `template`'s on every built residue,
/// so that the built residues keep their shape in every member. When the fit uses a rare pucker
/// of a built residue, few templates (or only its own) qualify.
pub fn compatible_templates(problem: &SiteProblem, template: usize, built: &BTreeSet<String>, tol: f64) -> Vec<usize> {
    let g = &problem.glycan;
    templates_in_state(&g.templates, &g.res_paths, &g.topology.ring_atoms, template, built, tol)
}

/// [`compatible_templates`] on bare arrays: template conformers, residue path and ring flag of
/// every atom.
pub fn templates_in_state(
    templates: &[Vec<[f32; 3]>],
    paths: &[String],
    ring_atoms: &[bool],
    template: usize,
    built: &BTreeSet<String>,
    tol: f64,
) -> Vec<usize> {
    let rings: Vec<Vec<usize>> = built
        .iter()
        .map(|p| (0..paths.len()).filter(|&i| &paths[i] == p && ring_atoms[i]).collect::<Vec<_>>())
        .filter(|idx| idx.len() >= 4)
        .collect();
    (0..templates.len())
        .filter(|&k| {
            rings.iter().all(|idx| {
                let (a, b) = (ring_signature(&templates[k], idx), ring_signature(&templates[template], idx));
                a.iter().zip(&b).all(|(u, v)| (u - v).abs() <= tol)
            })
        })
        .collect()
}

/// Torsions that move only atoms beyond the built residues.
pub fn free_torsions(problem: &SiteProblem, built: &BTreeSet<String>) -> Vec<bool> {
    let beyond: Vec<bool> = problem.glycan.res_paths.iter().map(|p| p != "agl" && !built.contains(p)).collect();
    problem.glycan.topology.distal.iter().map(|d| !d.is_empty() && d.iter().all(|&i| beyond[i])).collect()
}

/// The contacts some atoms of the glycan take part in (with the environment, and with any atom of
/// the glycan), unweighted: `SiteProblem.contacts(x, atoms)` of the reference.
pub struct RegionContacts<'p> {
    problem: &'p SiteProblem,
    atoms: Vec<bool>,
    self_pairs: Vec<(usize, usize, f32)>,
    site_pairs: Vec<(usize, usize, f64)>,
}

impl<'p> RegionContacts<'p> {
    pub fn new(problem: &'p SiteProblem, atoms: Vec<bool>) -> Self {
        let self_pairs = problem.self_pairs.iter().copied().filter(|&(i, j, _)| atoms[i] || atoms[j]).collect();
        let site_pairs = problem.site_pairs.iter().copied().filter(|&(i, _, _)| atoms[i]).collect();
        Self {
            problem,
            atoms,
            self_pairs,
            site_pairs,
        }
    }

    /// E_env + E_self; `gx` receives the gradient (added).
    pub fn energy(&self, x: &[V3], mut gx: Option<&mut [V3]>) -> f64 {
        let p = self.problem;
        let mut e = 0.0;
        for i in (0..p.n_atoms).filter(|&i| p.keep[i] && self.atoms[i]) {
            let (v, g) = p.grids.sample(p.is_c[i], x[i]);
            e += v;
            if let Some(gx) = gx.as_deref_mut() {
                for k in 0..3 {
                    gx[i][k] += g[k];
                }
            }
        }
        for &(i, j, floor) in &self.site_pairs {
            let d = sub(x[i], p.site_xyz[j]);
            let r = dot(d, d).sqrt();
            if r < floor {
                e += (floor - r).powi(2);
                if let Some(gx) = gx.as_deref_mut() {
                    let s = -2.0 * (floor - r) / r.max(1e-12);
                    for k in 0..3 {
                        gx[i][k] += s * d[k];
                    }
                }
            }
        }
        let (e_self, g_self) = clash_energy_grad(x, &self.self_pairs);
        if let Some(gx) = gx {
            for (g, s) in gx.iter_mut().zip(&g_self) {
                for k in 0..3 {
                    g[k] += s[k];
                }
            }
        }
        e + e_self
    }
}

/// Move a member off its contacts in torsion space: Adam on the free torsions under the region's
/// contacts (a member without contacts has no gradient and stays as sampled), best iterate.
pub fn relax(problem: &SiteProblem, contacts: &RegionContacts, pose: &Pose, free: &[bool], steps: usize, lr: f64) -> Pose {
    let nt = pose.tau.len();
    let topo = &problem.glycan.topology;
    let mut delta = vec![0.0; nt];
    let mut adam = Adam::new(nt, lr);
    let mut best = (f64::INFINITY, pose.tau.clone());
    for _ in 0..=steps {
        let tau: Vec<f64> = (0..nt).map(|t| if free[t] { wrap(pose.tau[t] + delta[t]) } else { pose.tau[t] }).collect();
        let x = problem.place(&Pose {
            tau: tau.clone(),
            ..pose.clone()
        });
        let mut gx = vec![[0.0; 3]; problem.n_atoms];
        let e = contacts.energy(&x, Some(&mut gx));
        if e < best.0 {
            best = (e, tau);
        }
        if best.0 <= 0.0 {
            break;
        }
        let g = torsion_gradient(&x, &topo.quads, &topo.distal, &gx);
        let grad: Vec<f64> = (0..nt).map(|t| if free[t] { g[t] } else { 0.0 }).collect();
        adam.step(&mut delta, &grad);
    }
    Pose {
        tau: best.1,
        ..pose.clone()
    }
}

/// Move the region's atoms of a grafted member off their remaining contacts, atom by atom, with
/// the bonds, 1-3 distances and chiral volumes of the member's template as restraints; every other
/// atom stays. Returns the coordinates and the region's contact energy.
pub fn clear_contacts(
    problem: &SiteProblem,
    contacts: &RegionContacts,
    x0: &[V3],
    template: usize,
    steps: usize,
    lr: f64,
    weight: f64,
) -> (Vec<V3>, f64) {
    let n = problem.n_atoms;
    let restraints = Restraints::new(problem, template, &RestraintOptions::default());
    let mut delta = vec![0.0; 3 * n];
    let mut adam = Adam::new(3 * n, lr);
    let mut best = (f64::INFINITY, x0.to_vec());
    for _ in 0..=steps {
        let x: Vec<V3> = (0..n)
            .map(|i| if contacts.atoms[i] { [0, 1, 2].map(|k| x0[i][k] + delta[3 * i + k]) } else { x0[i] })
            .collect();
        let mut gc = vec![[0.0; 3]; n];
        let e = contacts.energy(&x, Some(&mut gc));
        if e < best.0 {
            best = (e, x.clone());
        }
        if best.0 <= 0.0 {
            break;
        }
        let mut gr = vec![[0.0; 3]; n];
        restraints.energy(&x, Some(&mut gr));
        let mut grad = vec![0.0; 3 * n];
        for i in (0..n).filter(|&i| contacts.atoms[i]) {
            for k in 0..3 {
                grad[3 * i + k] = weight * gc[i][k] + gr[i][k];
            }
        }
        adam.step(&mut delta, &grad);
    }
    (best.1, best.0)
}

/// Eigenvector of the largest eigenvalue of a symmetric 4x4 matrix (cyclic Jacobi).
fn largest_eigenvector(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..4).flat_map(|i| (i + 1..4).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
                let (s, c) = theta.sin_cos();
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let k = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap_or(0);
    [v[0][k], v[1][k], v[2][k], v[3][k]]
}

/// Rotation `r` and translation `t` with `r p + t ~ q` in the least-squares sense (Horn's
/// quaternion method).
pub fn superpose(p: &[V3], q: &[V3]) -> ([[f64; 3]; 3], V3) {
    let n = p.len() as f64;
    let centre = |x: &[V3]| [0, 1, 2].map(|k| x.iter().map(|v| v[k]).sum::<f64>() / n);
    let (pc, qc) = (centre(p), centre(q));
    let mut s = [[0.0; 3]; 3];
    for (a, b) in p.iter().zip(q) {
        let (a, b) = (sub(*a, pc), sub(*b, qc));
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += a[i] * b[j];
            }
        }
    }
    let m = [
        [s[0][0] + s[1][1] + s[2][2], s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]],
        [s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]],
        [s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]],
        [s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]],
    ];
    let [w, x, y, z] = largest_eigenvector(m);
    let r = [
        [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
    ];
    let rp = [0, 1, 2].map(|i| dot(r[i], pc));
    (r, sub(qc, rp))
}

/// A member with the built residues replaced by `x_built` and every subtree beyond them moved
/// rigidly with the built residue it hangs from (superposition of that residue).
pub fn graft(paths: &[String], member: &[V3], x_built: &[V3], built: &BTreeSet<String>) -> Vec<V3> {
    let mut out: Vec<V3> = member.to_vec();
    for (i, p) in paths.iter().enumerate() {
        if p == "agl" || built.contains(p) {
            out[i] = x_built[i];
        }
    }
    let roots: BTreeSet<&String> = paths
        .iter()
        .filter(|p| *p != "agl" && !built.contains(*p) && p.rsplit_once('/').is_some_and(|(parent, _)| built.contains(parent)))
        .collect();
    for root in roots {
        let parent = root.rsplit_once('/').map(|(p, _)| p).unwrap_or_default();
        let prefix = format!("{root}/");
        let pa: Vec<usize> = (0..paths.len()).filter(|&i| paths[i] == parent).collect();
        let (r, t) = superpose(
            &pa.iter().map(|&i| member[i]).collect::<Vec<_>>(),
            &pa.iter().map(|&i| x_built[i]).collect::<Vec<_>>(),
        );
        for i in (0..paths.len()).filter(|&i| &paths[i] == root || paths[i].starts_with(&prefix)) {
            out[i] = [0, 1, 2].map(|k| dot(r[k], member[i]) + t[k]);
        }
    }
    out
}

/// Representatives of conformers (the atoms `idx`, RMSD in place): starting from the most central
/// conformer, the one farthest from every medoid becomes a medoid until all are within `cutoff`
/// of one or `max_clusters` are reached; then members go to their nearest medoid and each medoid
/// moves to the most central member of its cluster, until nothing changes. Returns the clusters,
/// largest first, and the largest member-to-medoid distance.
pub fn cluster(x: &[Vec<V3>], idx: &[usize], cutoff: f64, max_clusters: usize) -> (Vec<Cluster>, f64) {
    let m = x.len();
    if m == 0 {
        return (Vec::new(), 0.0);
    }
    let mut d = vec![0.0; m * m];
    for a in 0..m {
        for b in a + 1..m {
            let s: f64 = idx.iter().map(|&i| dot(sub(x[a][i], x[b][i]), sub(x[a][i], x[b][i]))).sum();
            let r = (s / idx.len().max(1) as f64).sqrt();
            d[a * m + b] = r;
            d[b * m + a] = r;
        }
    }
    let nearest = |medoids: &[usize], j: usize| -> (usize, f64) {
        medoids
            .iter()
            .enumerate()
            .map(|(k, &c)| (k, d[j * m + c]))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
    };
    let first = |it: &mut dyn Iterator<Item = (usize, f64)>, largest: bool| -> usize {
        // the first index with the extreme value (as numpy's argmin / argmax)
        let mut best: Option<(usize, f64)> = None;
        for (j, v) in it {
            if best.is_none_or(|(_, b)| if largest { v > b } else { v < b }) {
                best = Some((j, v));
            }
        }
        best.map_or(0, |(j, _)| j)
    };
    let mut medoids = vec![first(&mut (0..m).map(|j| (j, (0..m).map(|k| d[j * m + k]).sum::<f64>())), false)];
    while medoids.len() < max_clusters.min(m) {
        let far = first(&mut (0..m).map(|j| (j, nearest(&medoids, j).1)), true);
        if nearest(&medoids, far).1 <= cutoff {
            break;
        }
        medoids.push(far);
    }
    for _ in 0..20 {
        let assign: Vec<usize> = (0..m).map(|j| nearest(&medoids, j).0).collect();
        let moved: Vec<usize> = (0..medoids.len())
            .map(|k| {
                let group: Vec<usize> = (0..m).filter(|&j| assign[j] == k).collect();
                let c = first(&mut group.iter().map(|&j| (j, group.iter().map(|&l| d[j * m + l]).sum::<f64>())), false);
                if group.is_empty() { medoids[k] } else { c }
            })
            .collect();
        if moved == medoids {
            break;
        }
        medoids = moved;
    }
    let assign: Vec<usize> = (0..m).map(|j| nearest(&medoids, j).0).collect();
    let mut out: Vec<Cluster> = medoids
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            let members: Vec<usize> = (0..m).filter(|&j| assign[j] == k).collect();
            Cluster {
                medoid: c,
                population: members.len() as f64 / m as f64,
                members,
            }
        })
        .collect();
    out.sort_by(|a, b| b.population.total_cmp(&a.population));
    let radius = (0..m).map(|j| nearest(&medoids, j).1).fold(0.0, f64::max);
    (out, radius)
}

/// <rho_mean, rho_mean> / <rho, rho> of the atoms `idx` over the conformers, with Gaussian atoms
/// (weights `z`, width `sigma`; two atoms overlap as exp(-d^2 / 4 sigma^2)): 1 for an ordered
/// residue, towards 1/M for conformers that do not overlap.
pub fn order_parameter(x: &[Vec<V3>], idx: &[usize], z: &[f64], sigma: f64) -> f64 {
    let m = x.len();
    let overlap = |a: &[V3], b: &[V3]| -> f64 {
        let mut s = 0.0;
        for (i, &p) in idx.iter().enumerate() {
            for (j, &q) in idx.iter().enumerate() {
                let d = sub(a[p], b[q]);
                s += z[i] * z[j] * (-dot(d, d) / (4.0 * sigma * sigma)).exp();
            }
        }
        s
    };
    let pairs: Vec<(usize, usize)> = (0..m).flat_map(|a| (a..m).map(move |b| (a, b))).collect();
    let (cross_sum, self_sum) = pairs
        .par_iter()
        .map(|&(a, b)| {
            let o = overlap(&x[a], &x[b]);
            if a == b { (o, o) } else { (2.0 * o, 0.0) }
        })
        .reduce(|| (0.0, 0.0), |u, v| (u.0 + v.0, u.1 + v.1));
    (cross_sum / (m * m) as f64) / (self_sum / m as f64)
}

/// Spread of the atoms `idx` over the conformers (A, in place): root mean square distance of an
/// atom from its mean position.
pub fn rmsf(x: &[Vec<V3>], idx: &[usize]) -> f64 {
    let m = x.len() as f64;
    let mut s = 0.0;
    for &i in idx {
        let mean = [0, 1, 2].map(|k| x.iter().map(|c| c[i][k]).sum::<f64>() / m);
        s += x.iter().map(|c| dot(sub(c[i], mean), sub(c[i], mean))).sum::<f64>();
    }
    (s / (m * idx.len().max(1) as f64)).sqrt()
}

/// What the map says about a residue beyond the built ones, given the ensemble.
pub fn reading(order: f64, predicted: f64, observed: f64) -> &'static str {
    if observed > predicted + READING_MARGIN {
        "more density than the ensemble predicts: more ordered than modelled"
    } else if observed < predicted - READING_MARGIN {
        "less density than the ensemble predicts: absent in part, or more mobile than modelled"
    } else if order >= ORDERED {
        "the density of an ordered residue, though below what the fit builds"
    } else {
        "flexibility accounts for the weak density"
    }
}

/// The ensemble beyond the built residues (`gate`) of a fit `outcome` of `problem`. `levels`: the
/// map levels of the site; without them no density levels are reported. None when nothing is
/// built or the whole glycan is built.
pub fn extend(
    problem: &SiteProblem,
    sampler: &Sampler,
    outcome: &FitOutcome,
    gate: &Gate,
    map: Option<&DensityMap>,
    levels: Option<&DensityLevels>,
    options: &EnsembleOptions,
) -> Result<Option<Ensemble>> {
    let paths = &problem.glycan.res_paths;
    let built: BTreeSet<String> = gate.built.iter().cloned().collect();
    let mut region: Vec<String> = paths.iter().filter(|p| *p != "agl" && !built.contains(*p)).cloned().collect();
    region.sort();
    region.dedup();
    region.sort_by(|a, b| (a.matches('/').count(), a).cmp(&(b.matches('/').count(), b)));
    if built.is_empty() || region.is_empty() {
        return Ok(None);
    }
    let best = &outcome.basins[outcome.best];
    let free = free_torsions(problem, &built);
    let usable = compatible_templates(problem, best.pose.template, &built, PUCKER_TOLERANCE);
    let n = options.members;
    let templates: Vec<usize> = (0..n).map(|m| usable[m % usable.len()]).collect();
    let poses = complete_on_templates(problem, sampler, &best.pose, &free, &templates, options.steps, options.guidance, options.seed)?;
    let beyond: Vec<bool> = paths.iter().map(|p| p != "agl" && !built.contains(p)).collect();
    let contacts = RegionContacts::new(problem, beyond.clone());
    let finished: Vec<(Vec<V3>, f64, usize)> = poses
        .par_iter()
        .map(|pose| {
            let pose = relax(problem, &contacts, pose, &free, options.relax_steps, 0.01);
            let grafted = graft(paths, &problem.place(&pose), &best.x, &built);
            let (x, e) = clear_contacts(problem, &contacts, &grafted, pose.template, options.clear_steps, 0.005, 100.0);
            (x, e, pose.template)
        })
        .collect();
    problem.counter.add_objective_grad(n * (options.relax_steps + options.clear_steps));
    let (mut members, mut kept_templates) = (Vec::new(), Vec::new());
    for (x, e, template) in finished {
        if e <= options.contact_tolerance {
            members.push(x);
            kept_templates.push(template);
        }
    }
    let scored: Vec<usize> = (0..problem.n_atoms).filter(|&i| beyond[i] && problem.keep[i]).collect();
    let (clusters, cluster_rmsd) = cluster(&members, &scored, options.cluster_rmsd, MAX_CLUSTERS);
    let sigma = problem.observation.density().map_or(1.0, |l| l.sigma);
    let mut residues = Vec::new();
    if !members.is_empty() {
        for p in &region {
            let mut anchor = p.as_str();
            while !built.contains(anchor) {
                anchor = anchor.rsplit_once('/').map_or("r", |(parent, _)| parent);
            }
            let idx: Vec<usize> = (0..problem.n_atoms).filter(|&i| &paths[i] == p && problem.keep[i]).collect();
            let z: Vec<f64> = idx.iter().map(|&i| crate::observation::glycan_z(&problem.glycan.elements[i]).unwrap_or(0.0)).collect();
            let order = order_parameter(&members, &idx, &z, sigma);
            let (mut predicted, mut observed, mut read) = (None, None, None);
            if let (Some(map), Some(levels), Some(reference)) = (map, levels, gate.density_fraction.get(anchor)) {
                let seen: Vec<f64> = members
                    .iter()
                    .filter_map(|x| levels.fraction(map, &idx.iter().map(|&i| x[i]).collect::<Vec<_>>()))
                    .collect();
                if !seen.is_empty() {
                    let (pr, ob) = (order * reference, seen.iter().sum::<f64>() / seen.len() as f64);
                    (predicted, observed, read) = (Some(pr), Some(ob), Some(reading(order, pr, ob)));
                }
            }
            residues.push(ResidueFlexibility {
                residue: p.clone(),
                anchor: anchor.to_string(),
                rmsf: rmsf(&members, &idx),
                order,
                density_predicted: predicted,
                density_observed: observed,
                reading: read,
            });
        }
    }
    Ok(Some(Ensemble {
        region,
        kept: members.len(),
        members,
        templates: kept_templates,
        generated: n,
        compatible_templates: usable.len(),
        clusters,
        cluster_rmsd,
        residues,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, for reproducible test coordinates.
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            (-2.0 * self.uniform().max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
        }
        fn point(&mut self, scale: f64) -> V3 {
            [self.normal() * scale, self.normal() * scale, self.normal() * scale]
        }
    }

    #[test]
    fn order_parameter_is_the_density_left_by_the_spread() {
        let mut rng = Rng(1);
        let one: Vec<V3> = (0..6).map(|_| rng.point(1.0)).collect();
        let z = [6.0, 6.0, 8.0, 6.0, 7.0, 8.0];
        let idx: Vec<usize> = (0..6).collect();
        assert!((order_parameter(&vec![one.clone(); 20], &idx, &z, 1.0) - 1.0).abs() < 1e-12);
        let apart: Vec<V3> = one.iter().map(|p| [p[0] + 100.0, p[1] + 100.0, p[2] + 100.0]).collect();
        assert!((order_parameter(&[one, apart], &idx, &z, 1.0) - 0.5).abs() < 1e-9);
        // Gaussian disorder of variance u^2 per axis: 1/M + (1 - 1/M) (s^2 / (s^2 + u^2))^1.5
        let (m, s, u) = (600usize, 1.2f64, 0.9f64);
        let x: Vec<Vec<V3>> = (0..m).map(|_| vec![rng.point(u)]).collect();
        let expected = 1.0 / m as f64 + (1.0 - 1.0 / m as f64) * (s * s / (s * s + u * u)).powf(1.5);
        assert!((order_parameter(&x, &[0], &[6.0], s) - expected).abs() < 0.02);
    }

    #[test]
    fn clusters_have_populations() {
        let mut rng = Rng(2);
        let a: Vec<V3> = (0..8).map(|_| rng.point(1.0)).collect();
        let b: Vec<V3> = (0..8).map(|_| rng.point(1.0)).map(|p| [p[0] + 6.0, p[1] + 6.0, p[2] + 6.0]).collect();
        let jitter = |base: &[V3], rng: &mut Rng| -> Vec<V3> { base.iter().map(|p| { let d = rng.point(0.1); [p[0] + d[0], p[1] + d[1], p[2] + d[2]] }).collect() };
        let mut x: Vec<Vec<V3>> = (0..30).map(|_| jitter(&a, &mut rng)).collect();
        x.extend((0..10).map(|_| jitter(&b, &mut rng)));
        let idx: Vec<usize> = (0..8).collect();
        let (c, radius) = cluster(&x, &idx, 1.5, MAX_CLUSTERS);
        assert_eq!(c.iter().map(|k| k.members.len()).collect::<Vec<_>>(), vec![30, 10]);
        assert!((c[0].population - 0.75).abs() < 1e-12 && radius < 1.5);
        assert!(c[0].medoid < 30 && c[1].medoid >= 30);
        // conformers spread without structure: at most eight representatives, and the radius they cover
        let wide: Vec<Vec<V3>> = (0..60).map(|_| (0..8).map(|_| rng.point(3.0)).collect()).collect();
        let (c, radius) = cluster(&wide, &idx, 1.5, MAX_CLUSTERS);
        assert!(c.len() == 8 && radius > 1.5);
        assert!((c.iter().map(|k| k.population).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(c.iter().all(|k| k.members.contains(&k.medoid)));
        let mut all: Vec<usize> = c.iter().flat_map(|k| k.members.clone()).collect();
        all.sort();
        assert_eq!(all, (0..60).collect::<Vec<_>>());
    }

    #[test]
    fn graft_keeps_the_built_model_and_moves_subtrees_rigidly() {
        let paths: Vec<String> = ["agl"].into_iter().chain(["r"; 4]).chain(["r/4"; 4]).chain(["r/4/4"; 3]).map(String::from).collect();
        let mut rng = Rng(3);
        let member: Vec<V3> = (0..12).map(|_| rng.point(2.0)).collect();
        let (s, c) = 0.7f64.sin_cos();
        let moved = |p: V3| [c * p[0] - s * p[1] + 3.0, s * p[0] + c * p[1] - 1.0, p[2] + 2.0];
        let built_model: Vec<V3> = member.iter().map(|p| moved(*p)).collect();
        let built: BTreeSet<String> = ["r".to_string(), "r/4".to_string()].into();
        let out = graft(&paths, &member, &built_model, &built);
        for (a, b) in out.iter().zip(&built_model) {
            assert!((0..3).all(|k| (a[k] - b[k]).abs() < 1e-9)); // the subtree follows the residue it hangs from
        }
        // a member whose subtree is bent stays bent, on the built residues
        let mut bent = member.clone();
        for p in bent.iter_mut().skip(9) {
            *p = [p[0] + 0.5, p[1] + 0.5, p[2] + 0.5];
        }
        let out = graft(&paths, &bent, &built_model, &built);
        for i in 0..9 {
            assert!((0..3).all(|k| (out[i][k] - built_model[i][k]).abs() < 1e-12));
        }
        for i in 9..12 {
            let expect = moved(bent[i]);
            assert!((0..3).all(|k| (out[i][k] - expect[k]).abs() < 1e-9));
        }
    }

    #[test]
    fn readings() {
        assert!(reading(0.4, 0.15, 0.14).starts_with("flexibility accounts"));
        assert!(reading(0.95, 0.40, 0.36).starts_with("the density of an ordered residue"));
        assert!(reading(0.95, 0.45, 0.20).starts_with("less density"));
        assert!(reading(0.30, 0.10, 0.40).starts_with("more density"));
    }

    #[test]
    #[ignore = "needs GLYCOFLOW_MODEL (residue_library.json of the licensed model)"]
    fn ring_signatures_tell_the_chairs_apart() {
        let path = std::path::Path::new(&std::env::var("GLYCOFLOW_MODEL").unwrap()).join("residue_library.json");
        let library = glycoflow_core::ResidueLibrary::from_json_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut g = crate::problem::build_glycan(&library, "DManpa1-3DManpb1-4DGlcpNAcb1-OH", 3, 0).unwrap();
        g.templates[1] = g.templates[0].clone();
        g.templates[2] = g.templates[0].clone();
        assert!(crate::ring::flip_root_chair(&mut g, |k| k == 2));
        let root: Vec<usize> = (0..g.res_paths.len()).filter(|&i| g.res_paths[i] == "r" && g.topology.ring_atoms[i]).collect();
        let d = |a: usize, b: usize| {
            let (u, v) = (ring_signature(&g.templates[a], &root), ring_signature(&g.templates[b], &root));
            u.iter().zip(&v).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
        };
        assert!(d(0, 1) < 1e-9 && d(0, 2) > PUCKER_TOLERANCE);
    }
}
