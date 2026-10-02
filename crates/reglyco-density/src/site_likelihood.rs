//! Profiled least-squares site likelihood for glycan fitting.
//!
//! Within a fixed ball `V` around a glycosylation site (chosen before any
//! candidate is seen, so every candidate is judged on the same voxels) the
//! observed map is modelled as
//!
//! ```text
//! rho_obs(r) ~ a + b * rho_protein(r) + s * rho_glycan(r; x),   s >= 0
//! ```
//!
//! with normalised Gaussian atoms `rho(r) = sum_i Z_i N(r - x_i; sigma^2)`.
//! `a`, `b` and `s` are profiled out by least squares; the glycan's *gain* is
//! the drop in the residual sum of squares relative to the protein-only fit:
//!
//! ```text
//! gain = max(e, 0)^2 / c,  e = <g, obs> - a0 <g, 1> - b0 <g, prot>,
//!                          c = <g, g> - q^T G0^-1 q,  q = (<g, 1>, <g, prot>)
//! ```
//!
//! Every glycan inner product is analytic: `<g, obs>` and `<g, prot>` are
//! lookups in pre-blurred boxes and `<g, g>` is a sum over atom pairs, so one
//! evaluation costs O(atoms^2) and the gradient with respect to every atom is
//! exact (up to trilinear interpolation of the blurred boxes). This is the
//! Rust counterpart of `glycoflow/fitting/density.py` in GlycoFlow; both use
//! the same resampling (trilinear), blur (separable sampled Gaussian) and
//! splatting rules so values agree to floating-point precision.

use crate::{DensityError, DensityMap, Result};

/// One environment (protein, symmetry mate, other glycan) heavy atom.
#[derive(Debug, Clone, Copy)]
pub struct SiteEnvironmentAtom {
    pub position: [f64; 3],
    pub atomic_number: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SiteLikelihoodOptions {
    /// Gaussian atom width (calibrated on the nearby protein).
    pub sigma_angstrom: f64,
    /// Box spacing.
    pub spacing_angstrom: f64,
    /// Radius of the fixed scoring ball around the site.
    pub radius_angstrom: f64,
    /// Map resolution. Sets the largest lag of the residual autocorrelation
    /// (`max(3 A, resolution)`) behind the independent-sample volume.
    pub resolution_angstrom: f64,
    /// Wrap map lookups through the unit cell (full-cell crystallographic maps).
    pub periodic: bool,
    /// Volume of one independent sample of the map noise (A^3), converting
    /// squared-error gains into log-likelihood units. `None` (the default
    /// rule) measures it from the residual map, see
    /// [`noise_correlation_volume`]; `Some(v)` overrides the measurement.
    pub independent_volume: Option<f64>,
}

/// A Cartesian box of values: `data[(ix * ny + iy) * nz + iz]` at
/// `origin + (ix, iy, iz) * spacing`.
#[derive(Debug, Clone)]
pub struct SiteBox {
    pub origin: [f64; 3],
    pub spacing: f64,
    pub dims: [usize; 3],
    pub data: Vec<f32>,
}

impl SiteBox {
    pub fn zeros(origin: [f64; 3], spacing: f64, dims: [usize; 3]) -> Self {
        Self {
            origin,
            spacing,
            dims,
            data: vec![0.0; dims[0] * dims[1] * dims[2]],
        }
    }

    #[inline]
    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.dims[1] + j) * self.dims[2] + k
    }

    pub fn position(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        [
            self.origin[0] + i as f64 * self.spacing,
            self.origin[1] + j as f64 * self.spacing,
            self.origin[2] + k as f64 * self.spacing,
        ]
    }

    /// Trilinear value and Cartesian gradient; positions outside the box are
    /// clamped to the border (the scoring ball lies well inside the box).
    pub fn sample(&self, p: [f64; 3]) -> (f64, [f64; 3]) {
        let mut base = [0usize; 3];
        let mut frac = [0f64; 3];
        for a in 0..3 {
            let g = ((p[a] - self.origin[a]) / self.spacing).clamp(0.0, (self.dims[a] - 1) as f64);
            let f = g.floor().min((self.dims[a] - 2) as f64);
            base[a] = f as usize;
            frac[a] = g - f;
        }
        let mut value = 0.0;
        let mut grad = [0.0; 3];
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let v = self.data[self.index(base[0] + dx, base[1] + dy, base[2] + dz)] as f64;
                    let wx = if dx == 0 { 1.0 - frac[0] } else { frac[0] };
                    let wy = if dy == 0 { 1.0 - frac[1] } else { frac[1] };
                    let wz = if dz == 0 { 1.0 - frac[2] } else { frac[2] };
                    let sx = if dx == 0 { -1.0 } else { 1.0 };
                    let sy = if dy == 0 { -1.0 } else { 1.0 };
                    let sz = if dz == 0 { -1.0 } else { 1.0 };
                    value += v * wx * wy * wz;
                    grad[0] += v * sx * wy * wz;
                    grad[1] += v * wx * sy * wz;
                    grad[2] += v * wx * wy * sz;
                }
            }
        }
        for g in &mut grad {
            *g /= self.spacing;
        }
        (value, grad)
    }
}

#[inline]
fn gauss_norm(r2: f64, s2: f64) -> f64 {
    (-r2 / (2.0 * s2)).exp() / (2.0 * std::f64::consts::PI * s2).powf(1.5)
}

/// Sum of normalised Gaussians `Z_i N(r - x_i; s2)` on a box, truncated at
/// `4 sqrt(s2)` (same rule as the Python reference).
pub fn splat(atoms: &[SiteEnvironmentAtom], target: &mut SiteBox, s2: f64) {
    let sigma = s2.sqrt();
    let h = target.spacing;
    let rad = (4.0 * sigma / h).ceil() as i64 + 1;
    let cutoff2 = 16.0 * s2;
    for atom in atoms {
        let base = [0, 1, 2].map(|a| ((atom.position[a] - target.origin[a]) / h).round() as i64);
        for di in -rad..=rad {
            let i = base[0] + di;
            if i < 0 || i >= target.dims[0] as i64 {
                continue;
            }
            for dj in -rad..=rad {
                let j = base[1] + dj;
                if j < 0 || j >= target.dims[1] as i64 {
                    continue;
                }
                for dk in -rad..=rad {
                    let k = base[2] + dk;
                    if k < 0 || k >= target.dims[2] as i64 {
                        continue;
                    }
                    let p = target.position(i as usize, j as usize, k as usize);
                    let r2 = (0..3)
                        .map(|a| (p[a] - atom.position[a]).powi(2))
                        .sum::<f64>();
                    if r2 < cutoff2 {
                        let idx = target.index(i as usize, j as usize, k as usize);
                        target.data[idx] += (atom.atomic_number * gauss_norm(r2, s2)) as f32;
                    }
                }
            }
        }
    }
}

/// Convolution with a unit-integral Gaussian, as a separable sum over the
/// sampled kernel `n(u) h` truncated at `4 sigma` (zero outside the box).
pub fn gaussian_blur(input: &SiteBox, sigma: f64) -> SiteBox {
    let h = input.spacing;
    let rad = (4.0 * sigma / h).ceil() as i64;
    let kernel = (-rad..=rad)
        .map(|k| {
            let u = k as f64 * h;
            (-u * u / (2.0 * sigma * sigma)).exp() / ((2.0 * std::f64::consts::PI).sqrt() * sigma)
                * h
        })
        .collect::<Vec<_>>();
    let mut current = input.data.iter().map(|v| *v as f64).collect::<Vec<_>>();
    let dims = input.dims;
    for axis in 0..3 {
        let mut next = vec![0.0f64; current.len()];
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let pos = [i as i64, j as i64, k as i64];
                    let mut acc = 0.0;
                    for (t, w) in kernel.iter().enumerate() {
                        let mut q = pos;
                        q[axis] += t as i64 - rad;
                        if q[axis] < 0 || q[axis] >= dims[axis] as i64 {
                            continue;
                        }
                        acc += w * current
                            [(q[0] as usize * dims[1] + q[1] as usize) * dims[2] + q[2] as usize];
                    }
                    next[(i * dims[1] + j) * dims[2] + k] = acc;
                }
            }
        }
        current = next;
    }
    SiteBox {
        origin: input.origin,
        spacing: h,
        dims,
        data: current.into_iter().map(|v| v as f32).collect(),
    }
}

/// Constants of the protein-only fit over the scoring ball (integrals, A^3).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct SiteRegionSums {
    pub volume: f64,
    pub obs: f64,
    pub prot: f64,
    pub obs_obs: f64,
    pub prot_prot: f64,
    pub obs_prot: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct SiteLikelihoodScore {
    /// Drop in the residual sum of squares explained by the glycan (>= 0).
    pub gain: f64,
    /// `gain / (2 noise_variance * independent_volume)`.
    pub log_likelihood: f64,
    /// Correlation of the glycan model with the map after the protein fit.
    pub partial_correlation: f64,
    /// Fitted glycan scale `s` (may be negative before clamping).
    pub scale: f64,
}

#[derive(Debug, Clone)]
pub struct SiteLikelihood {
    pub center: [f64; 3],
    pub radius: f64,
    pub sigma: f64,
    pub obs_blur: SiteBox,
    pub prot_blur: SiteBox,
    pub region: SiteRegionSums,
    /// Protein-only fit (a0, b0).
    pub theta: [f64; 2],
    g0_inverse: [[f64; 2]; 2],
    pub sse0: f64,
    pub noise_variance: f64,
    pub independent_volume: f64,
}

impl SiteLikelihood {
    /// Box geometry used for a site: half width `radius + 4 sigma + 2`.
    pub fn box_geometry(
        center: [f64; 3],
        options: &SiteLikelihoodOptions,
    ) -> ([f64; 3], [usize; 3]) {
        let half = options.radius_angstrom + 4.0 * options.sigma_angstrom + 2.0;
        let n = (2.0 * half / options.spacing_angstrom).ceil() as usize + 1;
        (
            [center[0] - half, center[1] - half, center[2] - half],
            [n, n, n],
        )
    }

    /// Resample the map (trilinear, z-scored with whole-map statistics) onto
    /// the site box and prepare every candidate-independent quantity.
    pub fn from_map(
        map: &DensityMap,
        center: [f64; 3],
        environment: &[SiteEnvironmentAtom],
        options: SiteLikelihoodOptions,
    ) -> Result<Self> {
        let values = map.values();
        let n = values.len() as f64;
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        if !(std.is_finite() && std > 0.0) {
            return Err(DensityError::Geometry("map has zero variance".into()));
        }
        let (origin, dims) = Self::box_geometry(center, &options);
        let mut observed = SiteBox::zeros(origin, options.spacing_angstrom, dims);
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let p = observed.position(i, j, k);
                    let v = map.value_at_cartesian(p, options.periodic).ok_or_else(|| {
                        DensityError::Geometry(format!(
                            "site box point {p:?} lies outside the map; use a periodic full-cell map or a larger map"
                        ))
                    })?;
                    let idx = observed.index(i, j, k);
                    observed.data[idx] = ((v - mean) / std) as f32;
                }
            }
        }
        Self::from_observed_box(observed, center, environment, options)
    }

    /// Prepare from an already resampled observed box (synthetic maps, tests).
    pub fn from_observed_box(
        observed: SiteBox,
        center: [f64; 3],
        environment: &[SiteEnvironmentAtom],
        options: SiteLikelihoodOptions,
    ) -> Result<Self> {
        let sigma = options.sigma_angstrom;
        let h = observed.spacing;
        let half = (observed.dims[0] - 1) as f64 * h / 2.0;
        let box_center = [
            observed.origin[0] + half,
            observed.origin[1] + half,
            observed.origin[2] + half,
        ];
        let reach = half * 3f64.sqrt() + 6.0 * sigma;
        let near = environment
            .iter()
            .filter(|a| {
                (0..3)
                    .map(|i| (a.position[i] - box_center[i]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    < reach
            })
            .copied()
            .collect::<Vec<_>>();
        let mut prot = SiteBox::zeros(observed.origin, h, observed.dims);
        splat(&near, &mut prot, sigma * sigma);
        let mut prot_blur = SiteBox::zeros(observed.origin, h, observed.dims);
        splat(&near, &mut prot_blur, 2.0 * sigma * sigma);
        let obs_blur = gaussian_blur(&observed, sigma);
        let dv = h * h * h;
        let r2max = options.radius_angstrom * options.radius_angstrom;
        let mut inside = vec![false; observed.data.len()];
        let mut s = SiteRegionSums {
            volume: 0.0,
            obs: 0.0,
            prot: 0.0,
            obs_obs: 0.0,
            prot_prot: 0.0,
            obs_prot: 0.0,
        };
        for i in 0..observed.dims[0] {
            for j in 0..observed.dims[1] {
                for k in 0..observed.dims[2] {
                    let p = observed.position(i, j, k);
                    // single-precision distance, as in the reference (positions are float32 there)
                    let r2 = (0..3)
                        .map(|a| ((p[a] as f32 - center[a] as f32) as f64).powi(2))
                        .sum::<f64>();
                    if r2 >= r2max {
                        continue;
                    }
                    let idx = observed.index(i, j, k);
                    inside[idx] = true;
                    let o = observed.data[idx] as f64;
                    let q = prot.data[idx] as f64;
                    s.volume += dv;
                    s.obs += o * dv;
                    s.prot += q * dv;
                    s.obs_obs += o * o * dv;
                    s.prot_prot += q * q * dv;
                    s.obs_prot += o * q * dv;
                }
            }
        }
        let det = s.volume * s.prot_prot - s.prot * s.prot;
        if !(det.is_finite() && det.abs() > 0.0) {
            return Err(DensityError::Geometry(
                "degenerate protein-only fit over the site region".into(),
            ));
        }
        let g0_inverse = [
            [s.prot_prot / det, -s.prot / det],
            [-s.prot / det, s.volume / det],
        ];
        let theta = [
            g0_inverse[0][0] * s.obs + g0_inverse[0][1] * s.obs_prot,
            g0_inverse[1][0] * s.obs + g0_inverse[1][1] * s.obs_prot,
        ];
        let sse0 = s.obs_obs - (theta[0] * s.obs + theta[1] * s.obs_prot);
        let independent_volume = match options.independent_volume {
            Some(volume) => volume,
            None => {
                // residual on the protein shell (the signal the protein model
                // explains), lags up to the resolution: local correlated
                // noise / model error without long-range structure
                let prot_max = prot.data.iter().copied().fold(f32::MIN, f32::max);
                let shell = prot
                    .data
                    .iter()
                    .zip(&inside)
                    .map(|(p, inside)| *inside && *p > 0.05 * prot_max)
                    .collect::<Vec<_>>();
                let residual = observed
                    .data
                    .iter()
                    .zip(&prot.data)
                    .map(|(o, p)| *o as f64 - theta[0] - theta[1] * *p as f64)
                    .collect::<Vec<_>>();
                noise_correlation_volume(
                    &residual,
                    &shell,
                    observed.dims,
                    h,
                    options.resolution_angstrom.max(3.0),
                )
            }
        };
        if !(independent_volume.is_finite() && independent_volume > 0.0) {
            return Err(DensityError::Geometry(format!(
                "independent-sample volume {independent_volume} is not positive"
            )));
        }
        Ok(Self {
            center,
            radius: options.radius_angstrom,
            sigma,
            obs_blur,
            prot_blur,
            region: s,
            theta,
            g0_inverse,
            sse0,
            noise_variance: sse0 / s.volume,
            independent_volume,
        })
    }

    /// Score one glycan (positions and atomic numbers; zero weight excludes
    /// an atom). When `gradient` is given it receives d(log_likelihood)/dx.
    pub fn evaluate(
        &self,
        positions: &[[f64; 3]],
        atomic_numbers: &[f64],
        gradient: Option<&mut [[f64; 3]]>,
    ) -> SiteLikelihoodScore {
        let n = positions.len();
        let s2 = 2.0 * self.sigma * self.sigma;
        let mut go = 0.0;
        let mut gp = 0.0;
        let mut g1 = 0.0;
        let mut gg = 0.0;
        let mut d_obs = vec![[0.0; 3]; n];
        let mut d_prot = vec![[0.0; 3]; n];
        for a in 0..n {
            let z = atomic_numbers[a];
            if z == 0.0 {
                continue;
            }
            let (vo, go_grad) = self.obs_blur.sample(positions[a]);
            let (vp, gp_grad) = self.prot_blur.sample(positions[a]);
            go += z * vo;
            gp += z * vp;
            g1 += z;
            d_obs[a] = go_grad.map(|g| z * g);
            d_prot[a] = gp_grad.map(|g| z * g);
        }
        // <g, g>: symmetric pair sum (self terms once, each unordered pair twice)
        let mut d_gg = vec![[0.0; 3]; n];
        let norm0 = 1.0 / (2.0 * std::f64::consts::PI * s2).powf(1.5);
        for a in 0..n {
            let za = atomic_numbers[a];
            if za == 0.0 {
                continue;
            }
            gg += za * za * norm0;
            for b in a + 1..n {
                let zb = atomic_numbers[b];
                if zb == 0.0 {
                    continue;
                }
                let d = [0, 1, 2].map(|i| positions[a][i] - positions[b][i]);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let k = za * zb * (-r2 / (2.0 * s2)).exp() * norm0;
                gg += 2.0 * k;
                // d/dx_a of the (a,b) and (b,a) terms: 2 k (-(x_a - x_b) / s2)
                for i in 0..3 {
                    let g = -2.0 * k * d[i] / s2;
                    d_gg[a][i] += g;
                    d_gg[b][i] -= g;
                }
            }
        }
        let gi = &self.g0_inverse;
        let q = [g1, gp];
        let g0q = [
            gi[0][0] * q[0] + gi[0][1] * q[1],
            gi[1][0] * q[0] + gi[1][1] * q[1],
        ];
        let e = go - (self.theta[0] * g1 + self.theta[1] * gp);
        let c = (gg - (q[0] * g0q[0] + q[1] * g0q[1])).max(1e-12);
        let gain = if e > 0.0 { e * e / c } else { 0.0 };
        let scale_ll = 1.0 / (2.0 * self.noise_variance * self.independent_volume);
        if let Some(out) = gradient {
            for (a, slot) in out.iter_mut().enumerate().take(n) {
                if e <= 0.0 || atomic_numbers[a] == 0.0 {
                    *slot = [0.0; 3];
                    continue;
                }
                for i in 0..3 {
                    let de = d_obs[a][i] - self.theta[1] * d_prot[a][i];
                    // dc = d<g,g> - 2 (G0^-1 q)_prot d<g,prot>   (<g,1> is constant)
                    let dc = d_gg[a][i] - 2.0 * g0q[1] * d_prot[a][i];
                    slot[i] = scale_ll * (2.0 * e / c * de - e * e / (c * c) * dc);
                }
            }
        }
        SiteLikelihoodScore {
            gain,
            log_likelihood: gain * scale_ll,
            partial_correlation: (gain / self.sse0).sqrt(),
            scale: e / c,
        }
    }

    /// Blurred residual (observed minus protein-only fit) at a point.
    pub fn residual_at(&self, p: [f64; 3]) -> f64 {
        self.obs_blur.sample(p).0 - self.theta[0] - self.theta[1] * self.prot_blur.sample(p).0
    }
}

/// Volume of one independent sample of a residual map (A^3): the integral of
/// its normalised autocorrelation within `mask` (corrected for the mask
/// overlap at every lag) over lags up to `max_lag`, positive part only.
/// Converts squared-error gains into log-likelihood units.
///
/// `residual` and `mask` are boxes laid out as [`SiteBox`] data (`dims`,
/// spacing `h`). Lags whose mask overlap is below half the mask size are
/// skipped. Direct sum over lag offsets (the Python reference
/// `density.noise_correlation_volume` uses zero-padded FFTs; the sums are the
/// same). On blurred white noise this recovers the analytic value
/// `(4 pi sigma^2)^(3/2)` once `max_lag >= 3 A`.
pub fn noise_correlation_volume(
    residual: &[f64],
    mask: &[bool],
    dims: [usize; 3],
    h: f64,
    max_lag: f64,
) -> f64 {
    use rayon::prelude::*;
    let count = mask.iter().filter(|m| **m).count();
    if count == 0 {
        return f64::NAN;
    }
    let mean = residual
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(r, _)| *r)
        .sum::<f64>()
        / count as f64;
    let r = residual
        .iter()
        .zip(mask)
        .map(|(v, m)| if *m { v - mean } else { 0.0 })
        .collect::<Vec<_>>();
    let masked = mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m)
        .map(|(i, _)| {
            let k = i % dims[2];
            let j = (i / dims[2]) % dims[1];
            [i / (dims[1] * dims[2]), j, k]
        })
        .collect::<Vec<_>>();
    let rad = (max_lag / h).floor() as i64;
    let mut lags = Vec::new();
    for dx in -rad..=rad {
        for dy in -rad..=rad {
            for dz in -rad..=rad {
                let d2 = ((dx * dx + dy * dy + dz * dz) as f64) * h * h;
                if d2 <= max_lag * max_lag {
                    lags.push([dx, dy, dz]);
                }
            }
        }
    }
    // (autocorrelation, mask overlap) per lag
    let sums = lags
        .par_iter()
        .map(|lag| {
            let mut ac = 0.0;
            let mut am = 0usize;
            for p in &masked {
                let q = [0, 1, 2].map(|a| p[a] as i64 + lag[a]);
                if (0..3).any(|a| q[a] < 0 || q[a] >= dims[a] as i64) {
                    continue;
                }
                let qi = (q[0] as usize * dims[1] + q[1] as usize) * dims[2] + q[2] as usize;
                if mask[qi] {
                    am += 1;
                    let pi = (p[0] * dims[1] + p[1]) * dims[2] + p[2];
                    ac += r[pi] * r[qi];
                }
            }
            (ac, am as f64)
        })
        .collect::<Vec<_>>();
    let zero = lags.iter().position(|l| *l == [0, 0, 0]).unwrap_or(0);
    let (ac0, am0) = sums[zero];
    let norm0 = ac0 / am0.max(1.0);
    if norm0.is_nan() || norm0 <= 0.0 {
        return f64::NAN;
    }
    sums.iter()
        .filter(|(_, am)| *am > 0.5 * am0)
        .map(|(ac, am)| (ac / am.max(1.0) / norm0).max(0.0))
        .sum::<f64>()
        * h
        * h
        * h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(offset: [f64; 3]) -> Vec<[f64; 3]> {
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.2, 0.1],
            [2.1, 1.4, -0.2],
            [1.4, 2.6, 0.1],
            [0.0, 2.4, -0.1],
            [-0.6, 1.2, 0.2],
        ]
        .iter()
        .map(|p| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]])
        .collect()
    }

    fn synthetic(truth: &[[f64; 3]], env: &[SiteEnvironmentAtom], sigma: f64) -> SiteLikelihood {
        let options = SiteLikelihoodOptions {
            sigma_angstrom: sigma,
            spacing_angstrom: 0.5,
            radius_angstrom: 8.0,
            resolution_angstrom: 2.0,
            periodic: false,
            independent_volume: None,
        };
        let center = [0.0, 0.0, 0.0];
        let (origin, dims) = SiteLikelihood::box_geometry(center, &options);
        let mut obs = SiteBox::zeros(origin, 0.5, dims);
        splat(env, &mut obs, sigma * sigma);
        let glycan = truth
            .iter()
            .map(|p| SiteEnvironmentAtom {
                position: *p,
                atomic_number: 6.0,
            })
            .collect::<Vec<_>>();
        splat(&glycan, &mut obs, sigma * sigma);
        SiteLikelihood::from_observed_box(obs, center, env, options).unwrap()
    }

    #[test]
    fn truth_scores_best_and_gradient_matches_finite_differences() {
        let env = (0..12)
            .map(|k| SiteEnvironmentAtom {
                position: [-5.0 + k as f64 * 0.9, -4.0, 1.0 + (k % 3) as f64],
                atomic_number: 6.0,
            })
            .collect::<Vec<_>>();
        let truth = ring([0.5, 0.5, 0.0]);
        let lik = synthetic(&truth, &env, 0.8);
        let z = vec![6.0; truth.len()];
        let best = lik.evaluate(&truth, &z, None);
        for shift in [[0.7, 0.0, 0.0], [0.0, -0.9, 0.3], [0.0, 0.0, 1.5]] {
            let moved = truth
                .iter()
                .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
                .collect::<Vec<_>>();
            assert!(best.log_likelihood > lik.evaluate(&moved, &z, None).log_likelihood);
        }
        // the true glycan in protein density is penalised: put it on the environment atoms
        let into_protein = (0..6).map(|k| env[k].position).collect::<Vec<_>>();
        assert!(best.log_likelihood > lik.evaluate(&into_protein, &z, None).log_likelihood);

        // off-grid probe: trilinear interpolation has kinks at grid nodes
        let probe = ring([0.83, 0.37, 0.21]);
        let mut grad = vec![[0.0; 3]; probe.len()];
        lik.evaluate(&probe, &z, Some(&mut grad));
        let eps = 1e-5;
        for a in [0usize, 3] {
            for i in 0..3 {
                let mut plus = probe.clone();
                let mut minus = probe.clone();
                plus[a][i] += eps;
                minus[a][i] -= eps;
                let fd = (lik.evaluate(&plus, &z, None).log_likelihood
                    - lik.evaluate(&minus, &z, None).log_likelihood)
                    / (2.0 * eps);
                assert!(
                    (grad[a][i] - fd).abs() < 1e-4 * fd.abs().max(1.0),
                    "atom {a} axis {i}: {} vs {fd}",
                    grad[a][i]
                );
            }
        }
    }

    #[test]
    fn correlation_volume_of_blurred_white_noise() {
        // white noise blurred with sigma: autocorrelation integral (4 pi sigma^2)^(3/2)
        let dims = [48, 48, 48];
        let mut noise = SiteBox::zeros([0.0; 3], 0.5, dims);
        let mut state = 12345u64;
        for v in &mut noise.data {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32;
        }
        let sigma = 0.8;
        let blurred = gaussian_blur(&noise, sigma);
        let residual = blurred.data.iter().map(|v| *v as f64).collect::<Vec<_>>();
        let mask = (0..residual.len())
            .map(|i| {
                let (x, y, z) = (i / (48 * 48), (i / 48) % 48, i % 48);
                [x, y, z].iter().all(|c| (8..40).contains(c))
            })
            .collect::<Vec<_>>();
        let volume = noise_correlation_volume(&residual, &mask, dims, 0.5, 4.0);
        let analytic = (4.0 * std::f64::consts::PI * sigma * sigma).powf(1.5);
        assert!(
            (volume - analytic).abs() < 0.1 * analytic,
            "correlation volume {volume} vs analytic {analytic}"
        );
    }
}
