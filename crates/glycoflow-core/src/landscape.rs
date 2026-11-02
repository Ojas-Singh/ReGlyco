//! Reading a glycan's landscape off a sampled ensemble: how rare a conformer is, and a continuous
//! path between two conformers.
//!
//! **Rarity is measured in the ensemble, not by the model's likelihood.** The exact log-density of
//! the probability-flow ODE is calibrated but ranks conformers poorly against MD (median Spearman
//! ~0.56 against a kNN MD density whose own self-agreement is 0.97), while a kNN density over the
//! model's own samples ranks the same conformers at 0.954 -- at that ceiling, 20/20 glycans better,
//! paired t = +21.8 (`GlycoFlow/scripts/benchmark/likelihood_mc_vs_ode.py`, check E5 in
//! `docs/ring_generation.md`). [`Rarity`] is that estimator as validated: Euclidean distance on the
//! torsion torus over every torsion, the k-th neighbour with k = 16, against an ensemble of about a
//! thousand samples. Only the *ranking* was validated, so the score handed out is a rank within the
//! ensemble ([`Rarity::rarity`]), not a free energy: differences of kNN log-densities in tens of
//! dimensions carry a bias that grows with the dimension, and quoting them in kT would claim more
//! than was measured.
//!
//! **A path is a walk through the model's prior.** The flow maps prior points to conformers
//! diffeomorphically, so interpolating two prior points and decoding the walk stays on the model's
//! manifold and passes through populated regions (`paths.morph` in the reference). When both ends
//! are members of a sampled ensemble their prior points are known exactly (the sampler is seeded),
//! so the path ends on the very conformers that were picked, and [`interpolate_prior`] is all that
//! is needed; [`crate::sampler::Sampler::encode`] finds the prior point of any other conformer. A
//! path is a visualisation of a connection, not a claim about the route or its barrier: the rarity
//! of each image says how populated the region it crosses is, as far as the ensemble resolves it.

use crate::geometry::wrap;

/// The neighbour rank of the validated estimator.
pub const RARITY_K: usize = 16;

/// Squared distance between two torsion vectors on the torus.
fn torus_d2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x as f64 - *y as f64).rem_euclid(2.0 * std::f64::consts::PI);
            let d = d.min(2.0 * std::f64::consts::PI - d);
            d * d
        })
        .sum()
}

/// kNN log-density of each query among `cloud` ([M*T] torsions), up to a constant:
/// `-T log r_k`, where `r_k` is the distance to the (k+1)-th nearest member
/// (`likelihood_reference_ceiling.knn_log_density`). `skip_self` drops one exact match, for a query
/// that is itself a member of the cloud.
pub fn knn_log_density(cloud: &[f32], t: usize, query: &[f32], k: usize, skip_self: bool) -> Vec<f64> {
    if t == 0 {
        return Vec::new();
    }
    let m = cloud.len() / t;
    let k = k.min(m.saturating_sub(1 + skip_self as usize));
    query
        .chunks(t)
        .map(|q| {
            let mut d: Vec<f64> = cloud.chunks(t).map(|c| torus_d2(q, c)).collect();
            let idx = k + skip_self as usize;
            let (_, kth, _) = d.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
            -(t as f64) * kth.sqrt().max(1e-6).ln()
        })
        .collect()
}

/// An ensemble's density, as a reference to rank conformers against.
pub struct Rarity {
    cloud: Vec<f32>,
    t: usize,
    k: usize,
    /// every member's own (leave-one-out) log-density, ascending
    members: Vec<f64>,
}

impl Rarity {
    /// `torsions` [M*T] of an ensemble of M samples.
    pub fn new(torsions: &[f32], t: usize) -> Self {
        Self::with_k(torsions, t, RARITY_K)
    }

    pub fn with_k(torsions: &[f32], t: usize, k: usize) -> Self {
        let mut members = knn_log_density(torsions, t, torsions, k, true);
        members.sort_by(|a, b| a.total_cmp(b));
        Self {
            cloud: torsions.to_vec(),
            t,
            k,
            members,
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Log-density of conformers outside the ensemble ([Q*T] torsions).
    pub fn log_density(&self, query: &[f32]) -> Vec<f64> {
        knn_log_density(&self.cloud, self.t, query, self.k, false)
    }

    /// Rarity in [0, 1] of a log-density: the fraction of the ensemble that sits in a more
    /// populated region. 0 is the centre of the most populated basin, 1 is rarer than every sample.
    pub fn rarity_of(&self, log_density: f64) -> f64 {
        if self.members.is_empty() {
            return 0.0;
        }
        let below = self.members.partition_point(|&m| m < log_density);
        let at = self.members[below..].partition_point(|&m| m <= log_density);
        // ties split evenly, as the rank correlation that validated the estimator counts them
        1.0 - (below as f64 + 0.5 * at as f64) / self.members.len() as f64
    }

    /// Rarity of conformers outside the ensemble ([Q*T] torsions).
    pub fn rarity(&self, query: &[f32]) -> Vec<f64> {
        self.log_density(query)
            .into_iter()
            .map(|l| self.rarity_of(l))
            .collect()
    }

    /// Rarity of every member of the ensemble, in sample order.
    pub fn member_rarity(&self) -> Vec<f64> {
        knn_log_density(&self.cloud, self.t, &self.cloud, self.k, true)
            .into_iter()
            .map(|l| self.rarity_of(l))
            .collect()
    }
}

/// `images` prior points from (`tau_a`, `q_a`) to (`tau_b`, `q_b`), ends included: the shortest
/// arc on each torsion and a straight line on each ring-shape component (`paths.interpolate`).
/// Returns (torsions [images*T], ring shapes [images*R*3]); the ring halves may be empty.
pub fn interpolate_prior(
    tau_a: &[f32],
    q_a: &[f32],
    tau_b: &[f32],
    q_b: &[f32],
    images: usize,
) -> (Vec<f32>, Vec<f32>) {
    let images = images.max(2);
    let mut tau = Vec::with_capacity(images * tau_a.len());
    let mut q = Vec::with_capacity(images * q_a.len());
    for i in 0..images {
        let u = i as f32 / (images - 1) as f32;
        tau.extend(tau_a.iter().zip(tau_b).map(|(a, b)| wrap(a + u * wrap(b - a))));
        q.extend(q_a.iter().zip(q_b).map(|(a, b)| a + u * (b - a)));
    }
    (tau, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{uniform_torsions, SplitMix64};

    #[test]
    fn a_dense_point_is_less_rare_than_an_isolated_one() {
        // two torsions: a tight cluster at the origin plus a uniform background
        let mut rng = SplitMix64::new(3);
        let mut cloud = Vec::new();
        for _ in 0..600 {
            cloud.extend([(rng.uniform_f32() - 0.5) * 0.3, (rng.uniform_f32() - 0.5) * 0.3]);
        }
        cloud.extend(uniform_torsions(&mut rng, 2 * 400));
        let r = Rarity::new(&cloud, 2);
        let got = r.rarity(&[0.0, 0.0, 3.0, -3.0]);
        assert!(got[0] < 0.2, "cluster centre rarity {}", got[0]);
        assert!(got[1] > 0.6, "background rarity {}", got[1]);
        let members = r.member_rarity();
        assert_eq!(members.len(), 1000);
        let mean_cluster = members[..600].iter().sum::<f64>() / 600.0;
        let mean_bg = members[600..].iter().sum::<f64>() / 400.0;
        assert!(mean_cluster < mean_bg);
    }

    #[test]
    fn rarity_spans_zero_to_one_over_the_members() {
        let mut rng = SplitMix64::new(9);
        let cloud = uniform_torsions(&mut rng, 3 * 200);
        let m = Rarity::new(&cloud, 3).member_rarity();
        let (lo, hi) = m.iter().fold((1f64, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
        assert!(lo < 0.01 && hi > 0.99, "{lo} {hi}");
    }

    #[test]
    fn interpolation_takes_the_short_way_round_and_keeps_its_ends() {
        let (t, q) = interpolate_prior(&[3.0], &[0.0, 1.0], &[-3.0], &[2.0, -1.0], 5);
        assert_eq!(t.len(), 5);
        assert!((t[0] - 3.0).abs() < 1e-6 && (t[4] + 3.0).abs() < 1e-5);
        // through pi, not through 0
        assert!(t[2].abs() > 3.1, "midpoint {}", t[2]);
        assert_eq!(&q[..2], &[0.0, 1.0]);
        assert_eq!(&q[8..], &[2.0, -1.0]);
        assert!((q[4] - 1.0).abs() < 1e-6 && q[5].abs() < 1e-6);
    }
}
