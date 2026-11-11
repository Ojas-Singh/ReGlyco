//! Restrained Cartesian refinement of torsion-space fits (`glycoflow/fitting/cartesian.py`).
//!
//! The torsion-space search keeps GlycoFlow's template geometry; real glycans deviate from it
//! slightly and over a branched tree the deviations add up to ~1 A. This stage moves every atom
//! freely against the same objective, with each pose's template geometry as restraints:
//!
//! `E = -loglik + w_env E_env + w_self E_self + E_amide(psi_N(x)) + w_prior E_prior(tau(x))
//!      + sum_bonds (d - d0)^2 / 2 s_b^2 + sum_1-3 (d - d0)^2 / 2 s_a^2 + sum_chiral (V - V0)^2 / 2 s_v^2`
//!
//! Torsions (for the prior) and psi_N are measured on the coordinates; the link atom stays on the
//! protein. Gradients are analytic (Adam, as `torch.optim.Adam`).

use rayon::prelude::*;

use crate::observer::CartesianStep;
use crate::problem::{SiteProblem, Terms, V3, dihedral};
use crate::search::Adam;

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
fn axpy(g: &mut V3, s: f64, v: V3) {
    for k in 0..3 {
        g[k] += s * v[k];
    }
}

/// Gradient of `dihedral(a, b, c, d)` with respect to the four points.
pub fn dihedral_gradient(a: V3, b: V3, c: V3, d: V3) -> [V3; 4] {
    // Blondel & Karplus (1996), with F = a - b, G = b - c, H = d - c
    let f = sub(a, b);
    let g = sub(b, c);
    let h = sub(d, c);
    let aa = cross(f, g);
    let bb = cross(h, g);
    let a2 = dot(aa, aa).max(1e-12);
    let b2 = dot(bb, bb).max(1e-12);
    let gn = norm(g).max(1e-12);
    let fg = dot(f, g);
    let hg = dot(h, g);
    let mut ga = [0.0; 3];
    let mut gb = [0.0; 3];
    let mut gc = [0.0; 3];
    let mut gd = [0.0; 3];
    axpy(&mut ga, -gn / a2, aa);
    axpy(&mut gd, gn / b2, bb);
    axpy(&mut gb, gn / a2 + fg / (a2 * gn), aa);
    axpy(&mut gb, -hg / (b2 * gn), bb);
    axpy(&mut gc, hg / (b2 * gn) - gn / b2, bb);
    axpy(&mut gc, -fg / (a2 * gn), aa);
    [ga, gb, gc, gd]
}

/// Bond, 1-3 and chiral-volume restraints to one template.
#[derive(Debug, Clone)]
pub struct Restraints {
    /// (i, j, d0, weight, is_bond)
    pub pairs: Vec<(usize, usize, f64, f64, bool)>,
    /// (centre, first three bonded neighbours, V0)
    pub chiral: Vec<(usize, [usize; 3], f64)>,
    pub w_volume: f64,
    /// (the four atoms, target in radians, weight): the ring torsions of the template, which
    /// hold the shape of every ring, and the torsions of every amide, which hold it planar
    pub torsions: Vec<([usize; 4], f64, f64)>,
    /// attachment angle: |A1 - C1| held at the linkage geometry (A1 = CG for Asn, CB for Ser/Thr)
    pub attachment: (usize, V3, f64, f64),
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct RestraintOptions {
    pub sd_bond: f64,
    pub sd_angle: f64,
    pub sd_volume: f64,
    /// ring torsions (degrees)
    pub sd_ring: f64,
    /// amide torsions (degrees)
    pub sd_amide: f64,
}

impl Default for RestraintOptions {
    fn default() -> Self {
        Self {
            sd_bond: 0.02,
            sd_angle: 0.04,
            sd_volume: 0.2,
            sd_ring: 15.0,
            sd_amide: 5.0,
        }
    }
}

fn volume(x: &[V3], c: usize, n: [usize; 3]) -> f64 {
    let o = x[c];
    dot(sub(x[n[0]], o), cross(sub(x[n[1]], o), sub(x[n[2]], o)))
}

impl Restraints {
    /// The restraints of the glycan itself, without its attachment. Targets come from the
    /// monomer library where it has them ([`crate::ideal`]) and from `reference` elsewhere; the
    /// rings are held in the shape they have in `shape`.
    pub fn internal(
        problem: &SiteProblem,
        reference: &[V3],
        shape: &[V3],
        options: &RestraintOptions,
    ) -> Self {
        let (wb, wa) = (
            1.0 / options.sd_bond.powi(2),
            1.0 / options.sd_angle.powi(2),
        );
        let pairs = problem
            .ideal
            .pairs
            .iter()
            .map(|&(i, j, bond, ideal)| {
                let d0 = ideal.unwrap_or_else(|| norm(sub(reference[i], reference[j])));
                (i, j, d0, if bond { wb } else { wa }, bond)
            })
            .collect();
        let chiral = problem
            .ideal
            .chiral
            .iter()
            .map(|&(c, k, ideal)| {
                // the hand of the centre is the template's
                let v = volume(reference, c, k);
                (c, k, ideal.map_or(v, |i| i.copysign(v)))
            })
            .collect();
        let weight = |sd: f64| 1.0 / sd.to_radians().powi(2);
        let (w_ring, w_amide) = (weight(options.sd_ring), weight(options.sd_amide));
        let ideal = &problem.ideal;
        let rings = ideal
            .ring_torsions
            .iter()
            .map(|&q| (q, dihedral(shape[q[0]], shape[q[1]], shape[q[2]], shape[q[3]]), w_ring));
        let amides = ideal.amide_torsions.iter().map(|&(q, t)| (q, t, w_amide));
        let torsions = rings.chain(amides).collect();
        Self {
            pairs,
            chiral,
            w_volume: 1.0 / options.sd_volume.powi(2),
            torsions,
            attachment: (problem.c1, [0.0; 3], 0.0, 0.0),
        }
    }

    pub fn new(problem: &SiteProblem, template: usize, options: &RestraintOptions) -> Self {
        // Bonds, angles and chirality without an ideal value are restrained to the template's
        // geometry. A generated ring shape is written by displacing ring atoms along the
        // mean-plane normal, which stretches ring bonds by up to ~0.1 A; those templates take
        // their covalent geometry from the library's majority build instead (template 0), which
        // bonds and angles do not depend on.
        let reference = if problem.generated_rings {
            &problem.templates[0]
        } else {
            &problem.templates[template]
        };
        let (a1, link) = (problem.anchor[1], problem.anchor[2]);
        let b0 = norm(sub(a1, link));
        let d_att = (b0 * b0 + problem.bond * problem.bond
            - 2.0 * b0 * problem.bond * problem.angle.cos())
        .sqrt();
        Self {
            attachment: (problem.c1, a1, d_att, 1.0 / options.sd_angle.powi(2)),
            ..Self::internal(problem, reference, &problem.templates[template], options)
        }
    }

    /// The coordinates nearest `x0` at the minimum of the restraint energy (L-BFGS).
    pub fn relax(&self, x0: &[V3]) -> Vec<V3> {
        self.minimise(x0, &vec![true; x0.len()], &vec![0.0; x0.len()])
    }

    /// `x0` with its covalent geometry repaired in place: the minimum of the restraint energy
    /// with every `free` atom held to its position in `x0` by a harmonic spring of standard
    /// deviation `sd` (A, by atom). Without the springs the repair of an angle near the protein
    /// would swing everything beyond it.
    pub fn regularise(&self, x0: &[V3], free: &[bool], sd: &[f64]) -> Vec<V3> {
        let tether: Vec<f64> = sd.iter().map(|sd| 1.0 / (sd * sd)).collect();
        self.minimise(x0, free, &tether)
    }

    fn minimise(&self, x0: &[V3], free: &[bool], tether: &[f64]) -> Vec<V3> {
        let n = x0.len();
        let dotv = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let eval = |p: &[f64]| -> (f64, Vec<f64>) {
            let x: Vec<V3> = p.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            let mut g = vec![[0.0; 3]; n];
            let mut e = self.energy(&x, Some(&mut g));
            for (i, g) in g.iter_mut().enumerate() {
                if !free[i] {
                    *g = [0.0; 3];
                } else if tether[i] > 0.0 {
                    let d = sub(x[i], x0[i]);
                    e += 0.5 * tether[i] * dot(d, d);
                    axpy(g, tether[i], d);
                }
            }
            (e, g.into_iter().flatten().collect())
        };
        let stiffest = self.pairs.iter().map(|p| p.3).fold(1.0, f64::max);
        let mut p: Vec<f64> = x0.iter().flatten().copied().collect();
        let (mut e, mut g) = eval(&p);
        let mut history: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> = Default::default();
        for _ in 0..1000 {
            if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) < 1e-7 {
                break;
            }
            // two-loop recursion: q = H g
            let mut q = g.clone();
            let mut alpha = Vec::with_capacity(history.len());
            for (s, y, rho) in history.iter().rev() {
                let a = rho * dotv(s, &q);
                q.iter_mut().zip(y).for_each(|(q, y)| *q -= a * y);
                alpha.push(a);
            }
            let scale = history
                .back()
                .map_or(1.0 / stiffest, |(s, y, _)| dotv(s, y) / dotv(y, y));
            q.iter_mut().for_each(|q| *q *= scale);
            for ((s, y, rho), a) in history.iter().zip(alpha.iter().rev()) {
                let b = rho * dotv(y, &q);
                q.iter_mut().zip(s).for_each(|(q, s)| *q += (a - b) * s);
            }
            let mut slope = -dotv(&g, &q);
            if slope >= 0.0 {
                history.clear();
                q = g.iter().map(|v| v / stiffest).collect();
                slope = -dotv(&g, &q);
            }
            let mut step = 1.0;
            let accepted = loop {
                let trial: Vec<f64> = p.iter().zip(&q).map(|(p, q)| p - step * q).collect();
                let (e1, g1) = eval(&trial);
                if e1 <= e + 1e-4 * step * slope {
                    break Some((trial, e1, g1));
                }
                step *= 0.5;
                if step < 1e-12 {
                    break None;
                }
            };
            let Some((trial, e1, g1)) = accepted else {
                break;
            };
            let s: Vec<f64> = trial.iter().zip(&p).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = g1.iter().zip(&g).map(|(a, b)| a - b).collect();
            let sy = dotv(&s, &y);
            if sy > 1e-14 {
                if history.len() == 12 {
                    history.pop_front();
                }
                history.push_back((s, y, 1.0 / sy));
            }
            (p, e, g) = (trial, e1, g1);
        }
        p.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
    }

    /// Restraint energy; `g` receives its gradient.
    pub fn energy(&self, x: &[V3], mut g: Option<&mut [V3]>) -> f64 {
        let mut e = 0.0;
        let (c1, a1, d0, w) = self.attachment;
        let v = sub(x[c1], a1);
        let d = norm(v);
        e += 0.5 * w * (d - d0).powi(2);
        if let Some(g) = g.as_deref_mut() {
            axpy(&mut g[c1], w * (d - d0) / d.max(1e-12), v);
        }
        for &(i, j, d0, w, _) in &self.pairs {
            let v = sub(x[i], x[j]);
            let d = norm(v);
            e += 0.5 * w * (d - d0).powi(2);
            if let Some(g) = g.as_deref_mut() {
                let s = w * (d - d0) / d.max(1e-12);
                axpy(&mut g[i], s, v);
                axpy(&mut g[j], -s, v);
            }
        }
        for &(c, k, v0) in &self.chiral {
            let o = x[c];
            let (a, b, cc) = (sub(x[k[0]], o), sub(x[k[1]], o), sub(x[k[2]], o));
            let v = dot(a, cross(b, cc));
            e += 0.5 * self.w_volume * (v - v0).powi(2);
            if let Some(g) = g.as_deref_mut() {
                let s = self.w_volume * (v - v0);
                let (da, db, dc) = (cross(b, cc), cross(cc, a), cross(a, b));
                axpy(&mut g[k[0]], s, da);
                axpy(&mut g[k[1]], s, db);
                axpy(&mut g[k[2]], s, dc);
                for d in [da, db, dc] {
                    axpy(&mut g[c], -s, d);
                }
            }
        }
        for &(q, t0, w) in &self.torsions {
            let p = q.map(|i| x[i]);
            let t = dihedral(p[0], p[1], p[2], p[3]) - t0;
            let t = t.sin().atan2(t.cos());
            e += 0.5 * w * t * t;
            if let Some(g) = g.as_deref_mut() {
                let dq = dihedral_gradient(p[0], p[1], p[2], p[3]);
                for (i, d) in q.into_iter().zip(dq) {
                    axpy(&mut g[i], w * t, d);
                }
            }
        }
        e
    }

    /// RMS deviation of bond and 1-3 distances from the template.
    pub fn deviations(&self, x: &[V3]) -> (f64, f64) {
        let (mut sb, mut nb, mut sa, mut na) = (0.0, 0usize, 0.0, 0usize);
        for &(i, j, d0, _, bond) in &self.pairs {
            let d = norm(sub(x[i], x[j])) - d0;
            if bond {
                sb += d * d;
                nb += 1;
            } else {
                sa += d * d;
                na += 1;
            }
        }
        (
            (sb / nb.max(1) as f64).sqrt(),
            (sa / na.max(1) as f64).sqrt(),
        )
    }
}

/// Bring the pucker templates of a problem to its restraint targets. The templates of the residue
/// library are frames of a simulation, with the thermal scatter of its bonds and angles; a pose
/// built on one would start the restrained refinement off its targets, and the torsion search
/// would rank poses that no deposit could match atom for atom.
pub fn regularise_templates(problem: &mut SiteProblem) {
    let options = RestraintOptions::default();
    let mut templates: Vec<Vec<V3>> = Vec::with_capacity(problem.templates.len());
    for template in &problem.templates {
        let reference = match templates.first() {
            Some(first) if problem.generated_rings => first,
            _ => template,
        };
        // First with the ring torsions of the template, which hold the shape of a ring while its
        // bonds settle; then without them, since the torsions of a ring and its ideal angles are
        // not independent and the ideal values are the ones to reach.
        let mut restraints = Restraints::internal(problem, reference, template, &options);
        let held = restraints.relax(template);
        restraints.torsions.drain(..problem.ideal.ring_torsions.len());
        templates.push(restraints.relax(&held));
    }
    problem
        .set_templates(templates)
        .expect("as many atoms as before");
}

/// Objective of placed coordinates (psi_N and torsions measured on `x`), without restraints;
/// `g` receives d(total)/dx. Returns (terms, psi_N).
pub fn coordinate_objective(
    problem: &SiteProblem,
    x: &[V3],
    mut g: Option<&mut [V3]>,
) -> (Terms, f64) {
    let [cb, cg, nd] = problem.anchor;
    let c1 = problem.c1;
    let psi = dihedral(cb, cg, nd, x[c1]);
    let (mut terms, d_att) = problem.placed_terms(x, psi, g.as_deref_mut());
    if let Some(g) = g.as_deref_mut() {
        let dg = dihedral_gradient(cb, cg, nd, x[c1]);
        axpy(&mut g[c1], d_att, dg[3]);
    }
    if problem.prior.is_some() {
        let quads = &problem.glycan.topology.quads;
        let tau: Vec<f64> = quads
            .iter()
            .map(|q| dihedral(x[q[0]], x[q[1]], x[q[2]], x[q[3]]))
            .collect();
        let mut gt = vec![0.0; tau.len()];
        terms.e_prior = problem.prior_energy(&tau, g.is_some().then_some(&mut gt[..]));
        terms.total += problem.w_prior * terms.e_prior;
        if let Some(g) = g {
            // prior_energy adds w_prior * dE_prior/dtau to gt
            for (q, &s) in quads.iter().zip(&gt) {
                if s == 0.0 {
                    continue;
                }
                let dq = dihedral_gradient(x[q[0]], x[q[1]], x[q[2]], x[q[3]]);
                for (k, d) in q.iter().zip(dq) {
                    axpy(&mut g[*k], s, d);
                }
            }
        }
    }
    (terms, psi)
}

#[derive(Debug, Clone)]
pub struct CartesianFit {
    pub x: Vec<V3>,
    pub psi: f64,
    pub terms: Terms,
    pub e_restraint: f64,
    /// objective + restraint energy
    pub total: f64,
    pub bond_rmsd: f64,
    pub one_three_rmsd: f64,
    pub max_shift: f64,
}

/// Refine placed coordinates `x0` of a pose built on `template` (Adam, `steps` x `lr`).
pub fn cartesian_refine(
    problem: &SiteProblem,
    x0: &[V3],
    template: usize,
    steps: usize,
    lr: f64,
    options: &RestraintOptions,
) -> CartesianFit {
    cartesian_refine_traced(problem, x0, template, steps, lr, options, None)
}

/// [`cartesian_refine`], reporting every iteration to the problem's observer as basin
/// `trace.0` in pass `trace.1` (nothing is reported without a trace or an observer).
pub fn cartesian_refine_traced(
    problem: &SiteProblem,
    x0: &[V3],
    template: usize,
    steps: usize,
    lr: f64,
    options: &RestraintOptions,
    trace: Option<(usize, &str)>,
) -> CartesianFit {
    let observer = problem.observer.get().zip(trace);
    let r = Restraints::new(problem, template, options);
    let n = problem.n_atoms;
    // the aglycone atom stays on the protein; hydrogens move with their bond restraints
    let free: Vec<bool> = problem.movable[..n].to_vec();
    let mut p: Vec<f64> = x0.iter().flat_map(|v| v.iter().copied()).collect();
    let mut adam = Adam::new(p.len(), lr);
    let to_x = |p: &[f64]| -> Vec<V3> { p.chunks(3).map(|c| [c[0], c[1], c[2]]).collect() };
    for iteration in 0..steps {
        let x = to_x(&p);
        let mut g = vec![[0.0; 3]; n];
        let (terms, _) = coordinate_objective(problem, &x, Some(&mut g));
        let e_restraint = r.energy(&x, Some(&mut g));
        if let Some((o, (basin, pass))) = observer {
            o.cartesian_step(&CartesianStep {
                basin,
                pass,
                iteration,
                iterations: steps,
                x: &x,
                terms: &terms,
                e_restraint,
            });
        }
        let grad: Vec<f64> = g
            .iter()
            .zip(&free)
            .flat_map(|(v, f)| if *f { *v } else { [0.0; 3] })
            .collect();
        adam.step(&mut p, &grad);
    }
    let x = to_x(&p);
    let (terms, psi) = coordinate_objective(problem, &x, None);
    let e_restraint = r.energy(&x, None);
    let (bond_rmsd, one_three_rmsd) = r.deviations(&x);
    let max_shift = x
        .iter()
        .zip(x0)
        .map(|(a, b)| norm(sub(*a, *b)))
        .fold(0.0, f64::max);
    CartesianFit {
        psi,
        total: terms.total + e_restraint,
        terms,
        e_restraint,
        bond_rmsd,
        one_three_rmsd,
        max_shift,
        x,
    }
}

/// `cartesian_refine` of every pose, in parallel.
pub fn cartesian_refine_all(
    problem: &SiteProblem,
    xs: &[Vec<V3>],
    templates: &[usize],
    steps: usize,
    lr: f64,
    options: &RestraintOptions,
) -> Vec<CartesianFit> {
    problem.counter.add_objective_grad(xs.len() * steps);
    xs.par_iter()
        .zip(templates)
        .enumerate()
        .map(|(basin, (x, &t))| {
            cartesian_refine_traced(problem, x, t, steps, lr, options, Some((basin, "refine")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dihedral_gradient_matches_finite_differences() {
        let p = [
            [0.1, 1.2, -0.3],
            [0.9, 0.2, 0.4],
            [2.1, 0.5, 0.1],
            [2.6, 1.4, 1.3],
        ];
        let g = dihedral_gradient(p[0], p[1], p[2], p[3]);
        let h = 1e-6;
        for a in 0..4 {
            for k in 0..3 {
                let mut q = p;
                q[a][k] += h;
                let up = dihedral(q[0], q[1], q[2], q[3]);
                q[a][k] -= 2.0 * h;
                let dn = dihedral(q[0], q[1], q[2], q[3]);
                let fd = (up - dn) / (2.0 * h);
                assert!(
                    (fd - g[a][k]).abs() < 1e-6,
                    "atom {a} axis {k}: fd {fd} analytic {}",
                    g[a][k]
                );
            }
        }
    }
}
