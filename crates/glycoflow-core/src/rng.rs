//! Small deterministic RNG (SplitMix64) so the core needs no OS entropy (wasm-friendly).

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x6A09_E667_F3BC_C909,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 random bits.
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Index drawn with probabilities proportional to `weights`.
    pub fn categorical(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut u = self.uniform_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if u < *w {
                return i;
            }
            u -= w;
        }
        weights.len().saturating_sub(1)
    }
}

/// Uniform prior torsions in [-pi, pi), as `flow.uniform_torsions`: `(u * 2 - 1) * pi` in f32.
pub fn uniform_torsions(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| (rng.uniform_f32() * 2.0 - 1.0) * std::f32::consts::PI)
        .collect()
}

/// Standard normal draws (Box-Muller), as many as `n`.
pub fn normal_f32(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u = rng.uniform_f64().max(1e-300);
        let v = rng.uniform_f64();
        let r = (-2.0 * u.ln()).sqrt();
        let a = 2.0 * std::f64::consts::PI * v;
        out.push((r * a.cos()) as f32);
        out.push((r * a.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Ring-shape prior draws for `samples` conformers (`flow.sample`): standard normal on the
/// Cremer-Pople components each ring has (`chan`, [R*3], from `Sampler::pucker_channels`), zero on
/// the rest.
pub fn ring_prior(rng: &mut SplitMix64, chan: &[f32], samples: usize) -> Vec<f32> {
    let mut q = normal_f32(rng, samples * chan.len());
    for (k, v) in q.iter_mut().enumerate() {
        *v *= chan[k % chan.len().max(1)];
    }
    q
}
