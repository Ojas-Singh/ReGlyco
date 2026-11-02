//! Whether a generated conformer is a well-formed structure, and a light repair for one that is not.
//!
//! **The check** applies ReGlyco's validator rules (`reglyco-validate`) to the heavy atoms, which is
//! all that validator looks at: a bond is off when it deviates from its reference length by more
//! than 0.10 A (a warning) or 0.20 A (an error), and two atoms more than two bonds apart overlap when
//! the sum of their van der Waals radii exceeds their distance by more than 0.4 A (advisory) or 0.6 A
//! (hard). The reference bond length is the residue template's own (GLYCAM geometry), not an element
//! table: the template carries the carboxylates, amides and carbonyls that a table gets wrong.
//! Hydrogens are left out, as in the validator, whose heavy-atom radii would read every hydrogen
//! bond as a hard overlap.
//!
//! **The repair** ([`tidy`]) is two short minimisations that leave a clean conformer where it was.
//! The first moves torsions only, held near the sample, against the hard overlaps and against
//! hydrogens closer than the sampler's own contact floor: covalent geometry stays exact, and two
//! residues that interpenetrate can still be pulled apart. The second polishes Cartesian coordinates
//! towards the template's bond lengths and angles, which ring generation stretches (it displaces ring
//! atoms along the ring normal; median 0.018 A, up to ~0.17 A), held near the conformer. A stage runs
//! only when there is something for it to fix, and then for a fixed number of steps, so the repair is
//! a continuous function of the conformer: the same structure decoded in two different batches (a
//! path's end and the ensemble member it is) comes out the same. Generated ensembles are benchmarked
//! raw; this is for structures handed to a user.

use serde::Serialize;

use crate::geometry::{rotate_torsions, torsion_gradient};
use crate::guidance::H_FLOOR;
use crate::topology::Topology;

/// Bond deviation (A) above which the validator warns, and above which it fails the structure.
pub const BOND_WARNING: f64 = 0.10;
pub const BOND_ERROR: f64 = 0.20;
/// Van der Waals overlap (A) above which the validator reports a contact, and above which it is hard.
pub const OVERLAP_ADVISORY: f64 = 0.4;
pub const OVERLAP_HARD: f64 = 0.6;

/// Adam's denominator floor. Adam scales every step to the size of its gradient, so a torsion with
/// a vanishing gradient would still take full steps steered by rounding noise; a floor well above
/// that noise keeps such torsions still and the repair a smooth function of its input.
const ADAM_EPS: f64 = 1e-2;

/// Bondi-like heavy-atom radii of the validator (`reglyco_core::steric::vdw_radius`).
pub fn vdw_radius(element: &str) -> f64 {
    match element {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "P" => 1.80,
        "S" => 1.80,
        _ => 1.70,
    }
}

/// The verdict on one conformer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Quality {
    /// largest heavy-atom bond deviation from the template (A)
    pub bond_deviation: f64,
    /// largest heavy-atom van der Waals overlap beyond 1-3 (A)
    pub overlap: f64,
    /// heavy-atom pairs overlapping by more than [`OVERLAP_HARD`]
    pub hard_contacts: u32,
    /// heavy-atom pairs overlapping by more than [`OVERLAP_ADVISORY`] but not [`OVERLAP_HARD`]
    pub advisory_contacts: u32,
}

impl Quality {
    /// No error by the validator's rules: no bond off by more than [`BOND_ERROR`], no hard overlap.
    pub fn valid(&self) -> bool {
        self.bond_deviation <= BOND_ERROR && self.hard_contacts == 0
    }
}

/// Precomputed pairs of one glycan.
#[derive(Debug, Clone)]
pub struct Checker {
    n: usize,
    heavy: Vec<bool>,
    /// (i, j, reference length): every bond
    bonds: Vec<(usize, usize, f64)>,
    /// (i, j, reference distance): atoms two bonds apart (the bond angles)
    angles: Vec<(usize, usize, f64)>,
    /// both ends of the angle in one ring: its distance belongs to the ring's shape
    angle_in_ring: Vec<bool>,
    /// (i, j, radius sum): heavy-atom pairs beyond 1-3
    contacts: Vec<(usize, usize, f64)>,
    /// (i, j): pairs with a hydrogen, four or more bonds apart (the sampler's contact pairs)
    hydrogen: Vec<(usize, usize)>,
    /// both atoms in one ring: their distance is fixed by the ring's shape, not by a torsion
    same_ring: Vec<bool>,
}

impl Checker {
    /// `template`: the residue template's coordinates, which carry the reference bond lengths and
    /// angles.
    pub fn new<T: crate::geometry::Real>(topology: &Topology, elements: &[String], template: &[[T; 3]]) -> Self {
        let n = topology.n_atoms;
        let heavy: Vec<bool> = elements.iter().map(|e| e != "H").collect();
        let x: Vec<[f64; 3]> = template.iter().map(|p| p.map(|v| v.to_f64())).collect();
        let d = |i: usize, j: usize| dist(x[i], x[j]);
        let mut ring_of = vec![usize::MAX; n];
        for (r, ring) in topology.rings.iter().enumerate() {
            for &a in &ring.atoms {
                ring_of[a] = r;
            }
        }
        let (mut bonds, mut angles, mut contacts, mut hydrogen, mut same_ring) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut angle_in_ring = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let t = topology.topo_dist[i * n + j];
                match t {
                    1 => bonds.push((i, j, d(i, j))),
                    2 => {
                        angles.push((i, j, d(i, j)));
                        angle_in_ring.push(ring_of[i] != usize::MAX && ring_of[i] == ring_of[j]);
                    }
                    _ => {
                        if heavy[i] && heavy[j] {
                            contacts.push((i, j, vdw_radius(&elements[i]) + vdw_radius(&elements[j])));
                            same_ring.push(ring_of[i] != usize::MAX && ring_of[i] == ring_of[j]);
                        } else if t >= 4 {
                            hydrogen.push((i, j));
                        }
                    }
                }
            }
        }
        Self { n, heavy, bonds, angles, angle_in_ring, contacts, hydrogen, same_ring }
    }

    /// The validator's verdict on a conformer.
    pub fn check<T: crate::geometry::Real>(&self, x: &[[T; 3]]) -> Quality {
        let x: Vec<[f64; 3]> = x.iter().map(|p| p.map(|v| v.to_f64())).collect();
        let mut q = Quality::default();
        for &(i, j, d0) in &self.bonds {
            if self.heavy[i] && self.heavy[j] {
                q.bond_deviation = q.bond_deviation.max((dist(x[i], x[j]) - d0).abs());
            }
        }
        for &(i, j, r) in &self.contacts {
            let o = r - dist(x[i], x[j]);
            q.overlap = q.overlap.max(o);
            if o > OVERLAP_HARD {
                q.hard_contacts += 1;
            } else if o > OVERLAP_ADVISORY {
                q.advisory_contacts += 1;
            }
        }
        q
    }

    /// Contact energy and its Cartesian gradient: hard heavy-atom overlaps (aimed a margin below the
    /// limit, `target`) and hydrogens inside the sampler's floor. Pairs within one ring are left out:
    /// a torsion cannot change them.
    fn contact_energy(&self, x: &[[f64; 3]], target: f64, g: &mut [[f64; 3]]) -> f64 {
        let mut e = 0.0;
        let mut push = |i: usize, j: usize, gap: f64, d: f64, v: [f64; 3], e: &mut f64| {
            *e += gap * gap;
            let s = -2.0 * gap / d.max(1e-9);
            for k in 0..3 {
                g[i][k] += s * v[k];
                g[j][k] -= s * v[k];
            }
        };
        for (c, &(i, j, r)) in self.contacts.iter().enumerate() {
            if self.same_ring[c] {
                continue;
            }
            let v = sub(x[i], x[j]);
            let d = norm(v);
            let gap = r - target - d;
            if gap > 0.0 {
                push(i, j, gap, d, v, &mut e);
            }
        }
        for &(i, j) in &self.hydrogen {
            let v = sub(x[i], x[j]);
            let d = norm(v);
            let gap = H_FLOOR as f64 - d;
            if gap > 0.0 {
                push(i, j, gap, d, v, &mut e);
            }
        }
        e
    }

    fn needs_contacts(&self, x: &[[f64; 3]], target: f64) -> bool {
        self.contacts
            .iter()
            .zip(&self.same_ring)
            .any(|(&(i, j, r), &ring)| !ring && r - dist(x[i], x[j]) > target + 0.05)
            || self.hydrogen.iter().any(|&(i, j)| dist(x[i], x[j]) < H_FLOOR as f64 - 0.05)
    }
}

/// What [`tidy`] did to one conformer.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Tidied {
    pub before: Quality,
    pub after: Quality,
    /// largest atom displacement (A)
    pub moved: f64,
}

/// Repair a conformer in place: torsions against hard contacts, then Cartesian coordinates towards
/// the template's covalent geometry. `quads`/`distal` are the conformer's torsions (`Topology`).
pub fn tidy(checker: &Checker, topology: &Topology, x: &mut [[f32; 3]]) -> Tidied {
    let before = checker.check(x);
    let x0: Vec<[f64; 3]> = x.iter().map(|p| p.map(f64::from)).collect();
    let mut y = x0.clone();
    let n = checker.n;

    // 1. torsions: contacts against a pull back to where each torsion started; a second pass with a
    //    weaker pull for the contacts the first could not clear
    const TARGET: f64 = OVERLAP_HARD - 0.15;
    if !topology.quads.is_empty() {
        let nt = topology.quads.len();
        let mut turned = vec![0.0f64; nt];
        for k_back in [0.5, 0.05] {
            if !checker.needs_contacts(&y, TARGET) {
                break;
            }
            let (mut m1, mut m2) = (vec![0.0; nt], vec![0.0; nt]);
            let lr = 0.03;
            for step in 1..=200 {
                let mut gx = vec![[0.0; 3]; n];
                checker.contact_energy(&y, TARGET, &mut gx);
                let mut gt = torsion_gradient(&y, &topology.quads, &topology.distal, &gx);
                for (g, a) in gt.iter_mut().zip(&turned) {
                    *g += 2.0 * k_back * a;
                }
                let delta: Vec<f64> = (0..nt)
                    .map(|t| {
                        m1[t] = 0.9 * m1[t] + 0.1 * gt[t];
                        m2[t] = 0.999 * m2[t] + 0.001 * gt[t] * gt[t];
                        let mh = m1[t] / (1.0 - 0.9f64.powi(step));
                        let vh = m2[t] / (1.0 - 0.999f64.powi(step));
                        -lr * mh / (vh.sqrt() + ADAM_EPS)
                    })
                    .collect();
                rotate_torsions(&mut y, &topology.quads, &topology.distal, &delta);
                for (a, d) in turned.iter_mut().zip(&delta) {
                    *a += d;
                }

            }
        }
    }

    // 2. Cartesian: bonds and angles to the template, held near the conformer, contacts kept clear
    // only where a bond is near the validator's warning: a ring shape is left as generated whenever
    // its bonds are acceptable, since restoring a stretched bond also shrinks the ring's pucker
    let bonds_off = checker.bonds.iter().any(|&(i, j, d0)| {
        checker.heavy[i] && checker.heavy[j] && (dist(y[i], y[j]) - d0).abs() > BOND_WARNING - 0.02
    });
    if bonds_off {
        let anchor = y.clone();
        // a ring's internal angles belong to the shape it was given: keep the conformer's own
        let angle_targets: Vec<f64> = checker
            .angles
            .iter()
            .zip(&checker.angle_in_ring)
            .map(|(&(i, j, d0), &ring)| if ring { dist(anchor[i], anchor[j]) } else { d0 })
            .collect();
        let (mut m1, mut m2) = (vec![[0.0; 3]; n], vec![[0.0; 3]; n]);
        let lr = 0.005;
        let (k_bond, k_angle, k_pos) = (100.0, 30.0, 1.0);
        for step in 1..=150 {
            let mut g = vec![[0.0; 3]; n];
            checker.contact_energy(&y, TARGET, &mut g);
            for ((i, j, d0), k) in checker
                .bonds
                .iter()
                .map(|&b| (b, k_bond))
                .chain(
                    checker
                        .angles
                        .iter()
                        .zip(&angle_targets)
                        .map(|(&(i, j, _), &d0)| ((i, j, d0), k_angle)),
                )
            {
                let v = sub(y[i], y[j]);
                let d = norm(v);
                let s = 2.0 * k * (d - d0) / d.max(1e-9);
                for c in 0..3 {
                    g[i][c] += s * v[c];
                    g[j][c] -= s * v[c];
                }
            }
            for a in 0..n {
                for c in 0..3 {
                    g[a][c] += 2.0 * k_pos * (y[a][c] - anchor[a][c]);
                    m1[a][c] = 0.9 * m1[a][c] + 0.1 * g[a][c];
                    m2[a][c] = 0.999 * m2[a][c] + 0.001 * g[a][c] * g[a][c];
                    let mh = m1[a][c] / (1.0 - 0.9f64.powi(step));
                    let vh = m2[a][c] / (1.0 - 0.999f64.powi(step));
                    y[a][c] -= lr * mh / (vh.sqrt() + ADAM_EPS);
                }
            }

        }
    }

    let moved = y
        .iter()
        .zip(&x0)
        .map(|(a, b)| dist(*a, *b))
        .fold(0.0, f64::max);
    for (o, p) in x.iter_mut().zip(&y) {
        *o = p.map(|v| v as f32);
    }
    Tidied { before, after: checker.check(x), moved }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}
