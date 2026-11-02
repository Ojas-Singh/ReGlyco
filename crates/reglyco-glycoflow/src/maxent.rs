//! Maximum-entropy density fitting: a weighted ensemble of the fit's basins where the fit puts one
//! pose (`glycoflow/fitting/maxent.py`).
//!
//! The site objective profiles the glycan's scale, so the step from one conformer to a mixture
//! `rho = sum_i w_i rho_i` is exact: with `e_i` the projection of member `i` on the residual map,
//! `q_i = (<g,1>, <g,prot>, <g,solv>)` and `G_ij = <rho_i, rho_j>`,
//!
//! ```text
//! e(w) = sum_i w_i e_i,   c(w) = w^T G w - q(w)^T G0^-1 q(w),   gain(w) = relu(e)^2 / c,
//! ```
//!
//! and `w = delta_i` gives back member `i`'s own gain, so a pose and an ensemble are scored on one
//! objective with no new constants. The weights maximise `loglik(w) - theta KL(w || w0)`, on
//! log-weights (Adam, analytic gradient), from stiff to free along a theta grid, each warm-started
//! from the last; `theta -> infinity` is `w0`, `theta -> 0` the mixture that explains the map best.
//! `w0` is uniform over the basins: the model's exact log-density ranks conformers poorly against
//! MD and as `w0` it is nearly a delta (`docs/fitting_results.md`), so it is not used.
//!
//! What this is and is not, measured on four sites in the reference: a population explains those
//! maps better than any single pose, also on held-out halves of the map; it does **not** bring the
//! model closer to the deposit, and the gain is largest where the search failed. It is a
//! description next to the single-pose fit, which stays the model to build. The held-out split of
//! the reference (`held_out_gain`) is not ported yet, so a gain reported here is in-sample.

use serde::Serialize;

use crate::error::{Result, invalid};
use crate::problem::SiteProblem;

/// The theta grid of the reference (`scripts/fitting/maxent_case.py`), stiff to free.
pub const THETAS: [f64; 11] = [
    300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003,
];
/// Effective sample size (fraction of the members) at which the ensemble is read off.
pub const READ_AT_ESS: f64 = 0.5;

/// Everything the gain of any weighted mixture of one ensemble needs (`EnsembleTerms`).
#[derive(Debug, Clone)]
pub struct EnsembleTerms {
    pub e: Vec<f64>,
    pub q: Vec<[f64; 3]>,
    /// [M*M] cross-member overlap
    pub gram: Vec<f64>,
    pub g0inv: [[f64; 3]; 3],
    /// log-likelihood per unit of gain
    pub loglik_per_gain: f64,
    pub sse0: f64,
}

impl EnsembleTerms {
    pub fn len(&self) -> usize {
        self.e.len()
    }

    pub fn is_empty(&self) -> bool {
        self.e.is_empty()
    }

    /// The terms of members placed at `members` (each [N] coordinates of the problem's glycan).
    pub fn of(problem: &SiteProblem, members: &[Vec<[f64; 3]>]) -> Result<Self> {
        let lik = problem
            .observation
            .density()
            .ok_or_else(|| invalid("max-ent fitting needs a density observation"))?;
        let z: Vec<f64> = (0..problem.n_atoms)
            .map(|i| {
                if problem.keep[i] {
                    crate::observation::glycan_z(&problem.glycan.elements[i]).unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .collect();
        let m = members.len();
        let (e, q): (Vec<f64>, Vec<[f64; 3]>) =
            members.iter().map(|x| lik.mixture_parts(x, &z)).unzip();
        let pairs: Vec<(usize, usize)> = (0..m).flat_map(|i| (i..m).map(move |j| (i, j))).collect();
        use rayon::prelude::*;
        let values: Vec<f64> = pairs
            .par_iter()
            .map(|&(i, j)| lik.overlap(&members[i], &members[j], &z))
            .collect();
        let mut gram = vec![0.0; m * m];
        for (&(i, j), v) in pairs.iter().zip(values) {
            gram[i * m + j] = v;
            gram[j * m + i] = v;
        }
        Ok(Self {
            e,
            q,
            gram,
            g0inv: lik.environment_inverse(),
            loglik_per_gain: lik.loglik_per_gain(),
            sse0: lik.sse0,
        })
    }
}

/// Gain, log-likelihood, profiled scale and partial CC of a mixture (`maxent.mixture`).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Mixture {
    pub gain: f64,
    pub loglik: f64,
    pub scale: f64,
    pub e: f64,
    pub c: f64,
    pub partial_cc: f64,
}

/// The mixture with weights `w`, and when asked d(loglik)/dw.
pub fn mixture(t: &EnsembleTerms, w: &[f64], grad: Option<&mut [f64]>) -> Mixture {
    let m = t.len();
    let ew: f64 = w.iter().zip(&t.e).map(|(a, b)| a * b).sum();
    let mut qw = [0.0; 3];
    for (wi, qi) in w.iter().zip(&t.q) {
        for k in 0..3 {
            qw[k] += wi * qi[k];
        }
    }
    let gw: Vec<f64> = (0..m)
        .map(|i| (0..m).map(|j| t.gram[i * m + j] * w[j]).sum())
        .collect();
    let g0q: [f64; 3] = [0, 1, 2].map(|i| (0..3).map(|j| t.g0inv[i][j] * qw[j]).sum());
    let wgw: f64 = w.iter().zip(&gw).map(|(a, b)| a * b).sum();
    let c = (wgw - (0..3).map(|k| qw[k] * g0q[k]).sum::<f64>()).max(1e-12);
    let gain = if ew > 0.0 { ew * ew / c } else { 0.0 };
    if let Some(g) = grad {
        for i in 0..m {
            g[i] = if ew > 0.0 {
                // dc/dw_i = 2 (G w)_i - 2 q_i . G0^-1 q(w)
                let dc = 2.0 * gw[i] - 2.0 * (0..3).map(|k| t.q[i][k] * g0q[k]).sum::<f64>();
                (2.0 * ew * t.e[i] * c - ew * ew * dc) / (c * c) * t.loglik_per_gain
            } else {
                0.0
            };
        }
    }
    Mixture {
        gain,
        loglik: gain * t.loglik_per_gain,
        scale: ew / c,
        e: ew,
        c,
        partial_cc: (gain / t.sse0).sqrt(),
    }
}

/// Kish's effective sample size as a fraction of the members.
pub fn effective_sample_size(w: &[f64]) -> f64 {
    let s: f64 = w.iter().sum();
    let s2: f64 = w.iter().map(|x| (x / s).powi(2)).sum();
    1.0 / (w.len() as f64 * s2)
}

/// One point of the trade-off.
#[derive(Debug, Clone, Serialize)]
pub struct MaxEntPoint {
    pub theta: f64,
    pub weights: Vec<f64>,
    #[serde(skip)]
    pub logits: Vec<f64>,
    pub kl: f64,
    pub ess: f64,
    pub objective: f64,
    pub mixture: Mixture,
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().copied().fold(f64::MIN, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Weights maximising `loglik(w) - theta KL(w || w0)` (`maxent.maxent_weights`): Adam on the
/// logits `l` of `w = softmax(log w0 + l)`, keeping the best iterate.
pub fn maxent_weights(
    t: &EnsembleTerms,
    theta: f64,
    log_w0: &[f64],
    init: Option<&[f64]>,
    iterations: usize,
    lr: f64,
) -> MaxEntPoint {
    let m = t.len();
    let lse = {
        let mx = log_w0.iter().copied().fold(f64::MIN, f64::max);
        mx + log_w0.iter().map(|v| (v - mx).exp()).sum::<f64>().ln()
    };
    let lw0: Vec<f64> = log_w0.iter().map(|v| v - lse).collect();
    let mut l = init.map_or_else(|| vec![0.0; m], <[f64]>::to_vec);
    let (b1, b2, eps) = (0.9, 0.999, 1e-8);
    let (mut m1, mut m2) = (vec![0.0; m], vec![0.0; m]);
    let mut best = (f64::NEG_INFINITY, l.clone());
    let mut gw = vec![0.0; m];
    for step in 1..=iterations {
        let z: Vec<f64> = lw0.iter().zip(&l).map(|(a, b)| a + b).collect();
        let w = softmax(&z);
        let mix = mixture(t, &w, Some(&mut gw));
        let kl: f64 = w
            .iter()
            .zip(&lw0)
            .map(|(wi, l0)| wi * (wi.max(1e-300).ln() - l0))
            .sum();
        let value = mix.loglik - theta * kl;
        if value > best.0 {
            best = (value, l.clone());
        }
        // d(value)/dw_i, then through the softmax: dv/dl_j = w_j (g_j - sum_i w_i g_i)
        let g: Vec<f64> = (0..m)
            .map(|i| gw[i] - theta * (w[i].max(1e-300).ln() - lw0[i] + 1.0))
            .collect();
        let mean: f64 = w.iter().zip(&g).map(|(a, b)| a * b).sum();
        for j in 0..m {
            let grad = -(w[j] * (g[j] - mean)); // ascend the value: minimise its negative
            m1[j] = b1 * m1[j] + (1.0 - b1) * grad;
            m2[j] = b2 * m2[j] + (1.0 - b2) * grad * grad;
            let mh = m1[j] / (1.0 - b1.powi(step as i32));
            let vh = m2[j] / (1.0 - b2.powi(step as i32));
            l[j] -= lr * mh / (vh.sqrt() + eps);
        }
    }
    let l = best.1;
    let z: Vec<f64> = lw0.iter().zip(&l).map(|(a, b)| a + b).collect();
    let w = softmax(&z);
    let mixture = mixture(t, &w, None);
    let kl = w
        .iter()
        .zip(&lw0)
        .map(|(wi, l0)| wi * (wi.max(1e-300).ln() - l0))
        .sum();
    MaxEntPoint {
        theta,
        ess: effective_sample_size(&w),
        weights: w,
        logits: l,
        kl,
        objective: best.0,
        mixture,
    }
}

/// The trade-off curve over `thetas`, stiff to free, each point warm-started from the last
/// (`maxent.maxent_curve`), with a uniform `w0`.
pub fn maxent_curve(t: &EnsembleTerms, thetas: &[f64]) -> Vec<MaxEntPoint> {
    let m = t.len();
    let lw0 = vec![-(m as f64).ln(); m];
    let mut sorted = thetas.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut init: Option<Vec<f64>> = None;
    let mut rows = Vec::with_capacity(sorted.len());
    for theta in sorted {
        let r = maxent_weights(t, theta, &lw0, init.as_deref(), 1500, 0.05);
        init = Some(r.logits.clone());
        rows.push(r);
    }
    rows
}

/// The point of a curve nearest an effective sample size (`maxent.at_ess`).
pub fn at_ess(rows: &[MaxEntPoint], ess: f64) -> Option<&MaxEntPoint> {
    rows.iter()
        .min_by(|a, b| (a.ess - ess).abs().total_cmp(&(b.ess - ess).abs()))
}

/// A fit's basins as a weighted ensemble.
#[derive(Debug, Clone, Serialize)]
pub struct MaxEntEnsemble {
    /// basin index of each member (into `FitOutcome::basins`)
    pub members: Vec<usize>,
    /// log-likelihood of each member alone (`w = delta_i`, the fit's own number)
    pub member_loglik: Vec<f64>,
    /// the best single member's log-likelihood
    pub best_pose_loglik: f64,
    /// the equal-weight average of the members (no fitted parameters)
    pub flat: Mixture,
    /// the ensemble read off at `READ_AT_ESS`
    pub at_half: MaxEntPoint,
    pub curve: Vec<MaxEntPoint>,
    /// `at_half` over the best pose; in-sample (the held-out split is not ported)
    pub gain_ratio: f64,
}

/// Max-ent weights over placed members (coordinates of the problem's glycan, e.g. the fit's
/// basins).
pub fn maxent_ensemble(
    problem: &SiteProblem,
    members: &[Vec<[f64; 3]>],
    index: Vec<usize>,
) -> Result<MaxEntEnsemble> {
    if members.len() < 2 {
        return Err(invalid("a max-ent ensemble needs at least two members"));
    }
    let t = EnsembleTerms::of(problem, members)?;
    let m = t.len();
    let member_loglik: Vec<f64> = (0..m)
        .map(|i| {
            let mut w = vec![0.0; m];
            w[i] = 1.0;
            mixture(&t, &w, None).loglik
        })
        .collect();
    let best_pose_loglik = member_loglik.iter().copied().fold(f64::MIN, f64::max);
    let flat = mixture(&t, &vec![1.0 / m as f64; m], None);
    let curve = maxent_curve(&t, &THETAS);
    let at_half = at_ess(&curve, READ_AT_ESS)
        .cloned()
        .expect("non-empty curve");
    Ok(MaxEntEnsemble {
        members: index,
        gain_ratio: at_half.mixture.loglik / best_pose_loglik.max(1e-12),
        member_loglik,
        best_pose_loglik,
        flat,
        at_half,
        curve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms() -> EnsembleTerms {
        // three members on a 1-D "map": two overlapping, one apart
        let gram = vec![1.0, 0.6, 0.1, 0.6, 1.0, 0.1, 0.1, 0.1, 1.0];
        EnsembleTerms {
            e: vec![0.8, 0.7, 0.2],
            q: vec![[1.0, 0.1, 0.0], [1.0, 0.12, 0.0], [1.0, 0.05, 0.0]],
            gram,
            g0inv: [[0.01, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            loglik_per_gain: 10.0,
            sse0: 4.0,
        }
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let t = terms();
        let w = [0.5, 0.3, 0.2];
        let mut g = [0.0; 3];
        mixture(&t, &w, Some(&mut g));
        for i in 0..3 {
            let h = 1e-6;
            let (mut a, mut b) = (w, w);
            a[i] += h;
            b[i] -= h;
            let fd = (mixture(&t, &a, None).loglik - mixture(&t, &b, None).loglik) / (2.0 * h);
            assert!(
                (fd - g[i]).abs() < 1e-5 * fd.abs().max(1.0),
                "{i}: {fd} vs {}",
                g[i]
            );
        }
    }

    #[test]
    fn stiff_theta_keeps_the_prior_and_free_theta_concentrates() {
        let t = terms();
        let curve = maxent_curve(&t, &THETAS);
        assert!(
            curve[0].ess > 0.99,
            "theta 300 should stay uniform: ess {}",
            curve[0].ess
        );
        let free = curve.last().unwrap();
        assert!(free.ess < curve[0].ess);
        // the free end explains the map at least as well as the best single member
        let best = (0..3)
            .map(|i| {
                let mut w = [0.0; 3];
                w[i] = 1.0;
                mixture(&t, &w, None).loglik
            })
            .fold(f64::MIN, f64::max);
        assert!(
            free.mixture.loglik >= best - 1e-6,
            "{} < {best}",
            free.mixture.loglik
        );
    }
}

#[cfg(test)]
mod parity {
    use super::*;

    /// The reference's curve on the same terms (`maxent.maxent_curve`, PyTorch, float64).
    #[test]
    fn curve_matches_the_reference() {
        let t = tests_terms();
        let curve = maxent_curve(&t, &THETAS);
        let want: [(f64, f64, [f64; 3]); 4] = [
            (300.0, 6.511383, [0.33755, 0.33505, 0.3274]),
            (3.0, 7.331326, [0.4702, 0.34341, 0.18639]),
            (0.3, 7.433039, [0.54411, 0.33023, 0.12566]),
            (0.003, 7.436201, [0.56023, 0.32761, 0.11216]),
        ];
        for (theta, loglik, w) in want {
            let r = curve.iter().find(|r| r.theta == theta).unwrap();
            println!(
                "theta {theta}: loglik {:.6} (ref {loglik}), w {:?}",
                r.mixture.loglik, r.weights
            );
            assert!((r.mixture.loglik - loglik).abs() < 1e-4, "theta {theta}");
            for k in 0..3 {
                assert!(
                    (r.weights[k] - w[k]).abs() < 2e-3,
                    "theta {theta} w{k}: {} vs {}",
                    r.weights[k],
                    w[k]
                );
            }
        }
    }

    fn tests_terms() -> EnsembleTerms {
        EnsembleTerms {
            e: vec![0.8, 0.7, 0.2],
            q: vec![[1.0, 0.1, 0.0], [1.0, 0.12, 0.0], [1.0, 0.05, 0.0]],
            gram: vec![1.0, 0.6, 0.1, 0.6, 1.0, 0.1, 0.1, 0.1, 1.0],
            g0inv: [[0.01, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            loglik_per_gain: 10.0,
            sse0: 4.0,
        }
    }
}
