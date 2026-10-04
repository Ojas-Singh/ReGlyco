//! Clash guidance of the Python sampler (`flow.sample(guidance=...)`) as a [`Guidance`] hook.
//!
//! Soft contact energy E = sum_{pairs >= 4 bonds apart} relu(floor - d)^2 on the predicted endpoint
//! x_end = rotate_torsions(x, (1 - t) v); the correction is -scale * clamp(dE/d(ahead), -10, 10),
//! with dE/d(ahead) from the analytic torsion Jacobian at x_end (instead of autograd).

use crate::error::Result;
use crate::geometry::{rotate_torsions, torsion_gradient, Real};
use crate::sampler::{Guidance, GuidanceContext};
use crate::topology::ELEMENTS;

/// Contact floors (`flow.CC_FLOOR`, `C_POLAR_FLOOR`, `POLAR_FLOOR`).
pub const CC_FLOOR: f32 = 3.0;
pub const C_POLAR_FLOOR: f32 = 2.8;
pub const POLAR_FLOOR: f32 = 2.5;

pub struct ClashGuidance {
    pub scale: f32,
    /// active for step * dt >= start
    pub start: f64,
    /// (i, j, floor) for i < j, topological distance >= 4
    pairs: Option<Vec<(usize, usize, f32)>>,
}

impl ClashGuidance {
    pub fn new(scale: f32, start: f64) -> Self {
        Self {
            scale,
            start,
            pairs: None,
        }
    }

    fn pairs(&mut self, ctx: &GuidanceContext) -> &[(usize, usize, f32)] {
        self.pairs.get_or_insert_with(|| {
            let n = ctx.topology.n_atoms;
            let c_tok = ELEMENTS.iter().position(|e| *e == "C").unwrap() as u32;
            let mut out = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    if ctx.topology.topo_dist[i * n + j] >= 4 {
                        let (ci, cj) = (ctx.tokens[i][0] == c_tok, ctx.tokens[j][0] == c_tok);
                        let floor = if ci && cj {
                            CC_FLOOR
                        } else if ci ^ cj {
                            C_POLAR_FLOOR
                        } else {
                            POLAR_FLOOR
                        };
                        out.push((i, j, floor));
                    }
                }
            }
            out
        })
    }
}

/// Contact energy of one conformer and its Cartesian gradient (computed in f64).
pub fn clash_energy_grad<T: Real>(
    x: &[[T; 3]],
    pairs: &[(usize, usize, f32)],
) -> (f64, Vec<[f64; 3]>) {
    let mut e = 0f64;
    let mut g = vec![[0f64; 3]; x.len()];
    for &(i, j, floor) in pairs {
        let d = [
            x[i][0].to_f64() - x[j][0].to_f64(),
            x[i][1].to_f64() - x[j][1].to_f64(),
            x[i][2].to_f64() - x[j][2].to_f64(),
        ];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let floor = floor as f64;
        if r < floor {
            let gap = floor - r;
            e += gap * gap;
            let s = -2.0 * gap / r.max(1e-12);
            for k in 0..3 {
                g[i][k] += s * d[k];
                g[j][k] -= s * d[k];
            }
        }
    }
    (e, g)
}

impl Guidance for ClashGuidance {
    fn correction(&mut self, ctx: &GuidanceContext) -> Result<Option<Vec<f32>>> {
        if self.scale <= 0.0 || (ctx.step as f64) * (1.0 / ctx.steps as f64) < self.start {
            return Ok(None);
        }
        let n = ctx.topology.n_atoms;
        let nt = ctx.topology.n_torsions();
        let scale = self.scale;
        let pairs = self.pairs(ctx).to_vec();
        let mut out = Vec::with_capacity(ctx.batch * nt);
        let omt = 1.0 - ctx.t;
        for s in 0..ctx.batch {
            let ahead: Vec<f32> = ctx.v[s * nt..(s + 1) * nt]
                .iter()
                .map(|v| omt * v)
                .collect();
            let mut x = ctx.coords[s * n..(s + 1) * n].to_vec();
            rotate_torsions(&mut x, &ctx.topology.quads, &ctx.topology.distal, &ahead);
            let (_, gx) = clash_energy_grad(&x, &pairs);
            let gt = torsion_gradient(&x, &ctx.topology.quads, &ctx.topology.distal, &gx);
            out.extend(gt.iter().map(|&g| -(scale * (g as f32).clamp(-10.0, 10.0))));
        }
        Ok(Some(out))
    }
}
