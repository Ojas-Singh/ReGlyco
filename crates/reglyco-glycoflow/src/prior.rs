//! A cheap, differentiable GlycoFlow prior energy for one glycan (`glycoflow/fitting/prior.py`).
//!
//! GlycoFlow samples of the glycan are drawn once and each torsion's marginal density is
//! estimated with a von Mises kernel:
//!
//! ```text
//! -log p(tau) ~ - sum_t [ log mean_j vM(tau_t; tau_jt, kappa) ]
//! ```
//!
//! A product of GlycoFlow's marginals: it keeps every torsion inside the populations the model
//! learned for this glycan but ignores the couplings between torsions (an approximation, stated
//! as such in reports).

/// `log(I0(x) exp(-x))` for x >= 0 (power series of I0, adequate for kappa up to ~700).
fn ln_i0e(x: f64) -> f64 {
    // I0(x) = sum_k (x^2/4)^k / (k!)^2, summed with terms scaled by exp(-x) in log space
    let q = x * x / 4.0;
    let mut log_term = 0.0f64; // k = 0
    let mut terms = vec![0.0f64];
    let mut k = 1.0f64;
    loop {
        log_term += q.ln() - 2.0 * k.ln();
        terms.push(log_term);
        if log_term < terms.iter().cloned().fold(f64::MIN, f64::max) - 40.0 && k > x {
            break;
        }
        k += 1.0;
        if k > 10_000.0 {
            break;
        }
    }
    let m = terms.iter().cloned().fold(f64::MIN, f64::max);
    m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln() - x
}

#[derive(Debug, Clone)]
pub struct MarginalPrior {
    /// GlycoFlow samples [S * T] (radians)
    pub samples: Vec<f64>,
    pub n_samples: usize,
    pub n_torsions: usize,
    pub kappa: f64,
    pub log_norm: f64,
    pub bandwidth_deg: f64,
    /// [T] torsions the energy sums over (`MarginalPrior.tmask`): every torsion by default;
    /// `with_torsions` restricts it, e.g. to those heavy atoms define
    pub on: Vec<bool>,
}

impl MarginalPrior {
    /// From samples `[S * T]` and the kernel bandwidth (degrees).
    pub fn new(samples: Vec<f64>, n_torsions: usize, bandwidth_deg: f64) -> Self {
        let kappa = 1.0 / bandwidth_deg.to_radians().powi(2);
        let log_norm = (2.0 * std::f64::consts::PI).ln() + kappa + ln_i0e(kappa);
        let n_samples = samples.len().checked_div(n_torsions).unwrap_or(0);
        Self {
            samples,
            n_samples,
            n_torsions,
            kappa,
            log_norm,
            bandwidth_deg,
            on: vec![true; n_torsions],
        }
    }

    /// The same prior summed over the torsions `on` selects only.
    pub fn with_torsions(mut self, on: Vec<bool>) -> Self {
        assert_eq!(on.len(), self.n_torsions);
        self.on = on;
        self
    }

    /// `-log p(tau)` (same constant for every candidate) and, on request, its gradient.
    pub fn energy(&self, tau: &[f64], mut grad: Option<&mut [f64]>) -> f64 {
        let (s, t) = (self.n_samples, self.n_torsions);
        if s == 0 {
            return 0.0;
        }
        let ln_s = (s as f64).ln();
        let mut total = 0.0;
        let mut logits = vec![0.0; s];
        for k in (0..t).filter(|&k| self.on[k]) {
            let mut m = f64::MIN;
            for (j, logit) in logits.iter_mut().enumerate() {
                *logit = self.kappa * (tau[k] - self.samples[j * t + k]).cos();
                m = m.max(*logit);
            }
            for l in logits.iter_mut() {
                *l = (*l - m).exp();
            }
            let sum = logits.iter().sum::<f64>();
            let lse = m + sum.ln();
            total -= lse - ln_s - self.log_norm;
            if let Some(g) = grad.as_deref_mut() {
                // d/dtau of -lse = kappa * sum_j w_j sin(tau - s_j)
                let mut acc = 0.0;
                for (j, weight) in logits.iter().enumerate() {
                    let w = weight / sum;
                    acc += w * (tau[k] - self.samples[j * t + k]).sin();
                }
                g[k] += self.kappa * acc;
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0e_matches_known_values() {
        // scipy.special.i0e(1.0) = 0.46575960759364043, i0e(22.8) = 0.0841139... (asymptotic check)
        assert!((ln_i0e(1.0) - 0.465_759_607_593_640_4f64.ln()).abs() < 1e-12);
        let x = 22.8f64;
        let asym = -(2.0 * std::f64::consts::PI * x).sqrt().ln()
            + (1.0 + 1.0 / (8.0 * x) + 9.0 / (128.0 * x * x)).ln();
        assert!((ln_i0e(x) - asym).abs() < 1e-4);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let samples = vec![0.1, -2.0, 0.3, -1.8, 2.9, 1.0, -3.0, 1.2];
        let prior = MarginalPrior::new(samples, 2, 12.0);
        let tau = [0.2, -1.9];
        let mut g = vec![0.0; 2];
        prior.energy(&tau, Some(&mut g));
        for k in 0..2 {
            let mut p = tau;
            let mut m = tau;
            p[k] += 1e-6;
            m[k] -= 1e-6;
            let fd = (prior.energy(&p, None) - prior.energy(&m, None)) / 2e-6;
            assert!(
                (fd - g[k]).abs() < 1e-5 * fd.abs().max(1.0),
                "{fd} vs {}",
                g[k]
            );
        }
    }
}

/// Percentile below which a torsion counts as one GlycoFlow essentially never generates.
pub const RARE_PERCENTILE: f64 = 1.0;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RareTorsion {
    /// "residue path:atom-atom-atom-atom"
    pub torsion: String,
    pub degrees: f64,
    /// percentile of GlycoFlow's marginal density among the densities of its own samples
    pub percentile: f64,
}

/// How far a pose is from what GlycoFlow generates for this glycan (`prior.prior_deviation`):
/// a real but unusual conformation where the map supports it, a suspect one where it does not.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PriorDeviation {
    pub e_prior: f64,
    pub rare_torsions: Vec<RareTorsion>,
    pub n_torsions: usize,
    /// rarity in [0, 1] of the whole conformation within the prior's free-glycan ensemble: the
    /// fraction of GlycoFlow's samples in a more populated region, by a kNN density over all
    /// torsions jointly (`glycoflow_core::landscape`, the estimator validated against MD). Unlike
    /// the per-torsion percentiles it sees correlated torsions; like them it describes the free
    /// glycan, so a bound conformation may be rare for good reason.
    pub rarity: f64,
}

impl MarginalPrior {
    /// The marginal energy summed over the torsions `on` selects (`energy` over a subset).
    pub fn energy_on(&self, tau: &[f64], on: &[bool]) -> f64 {
        let t = self.n_torsions;
        let cols: Vec<usize> = (0..t).filter(|&k| on[k]).collect();
        let sub = MarginalPrior {
            samples: (0..self.n_samples)
                .flat_map(|j| cols.iter().map(move |&k| self.samples[j * t + k]))
                .collect(),
            n_samples: self.n_samples,
            n_torsions: cols.len(),
            kappa: self.kappa,
            log_norm: self.log_norm,
            bandwidth_deg: self.bandwidth_deg,
            on: vec![true; cols.len()],
        };
        sub.energy(&cols.iter().map(|&k| tau[k]).collect::<Vec<_>>(), None)
    }

    /// Rarity in [0, 1] of a conformation among the prior's samples over the torsions `on` selects
    /// (see [`PriorDeviation::rarity`]).
    pub fn rarity_on(&self, tau: &[f64], on: &[bool]) -> f64 {
        let t = self.n_torsions;
        let cols: Vec<usize> = (0..t).filter(|&k| on[k]).collect();
        let cloud: Vec<f32> = (0..self.n_samples)
            .flat_map(|j| cols.iter().map(move |&k| self.samples[j * t + k] as f32))
            .collect();
        let query: Vec<f32> = cols.iter().map(|&k| tau[k] as f32).collect();
        glycoflow_core::landscape::Rarity::new(&cloud, cols.len())
            .rarity(&query)
            .first()
            .copied()
            .unwrap_or(f64::NAN)
    }

    fn density(&self, k: usize, v: f64) -> f64 {
        let t = self.n_torsions;
        (0..self.n_samples)
            .map(|j| (self.kappa * (v - self.samples[j * t + k]).cos()).exp())
            .sum::<f64>()
            / self.n_samples.max(1) as f64
    }

    /// Per torsion: percentile (0-100) of the marginal density at `tau` among the densities at the
    /// prior's own samples.
    pub fn percentiles(&self, tau: &[f64]) -> Vec<f64> {
        let (s, t) = (self.n_samples, self.n_torsions);
        (0..t)
            .map(|k| {
                let d = self.density(k, tau[k]);
                let below = (0..s)
                    .filter(|&j| self.density(k, self.samples[j * t + k]) < d)
                    .count();
                100.0 * below as f64 / s.max(1) as f64
            })
            .collect()
    }
}
