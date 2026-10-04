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
