//! Search methods (`glycoflow/fitting/methods.py`): observation-guided GlycoFlow generation
//! (method B), attachment grid search, Adam refinement, distinct basins.

use web_time::Instant;

use glycoflow_core::geometry::{P3, wrap};
use glycoflow_core::rng::{SplitMix64, uniform_torsions};
use glycoflow_core::sampler::{Guidance, GuidanceContext, Method, SampleOptions, Sampler};
use rayon::prelude::*;

use crate::error::Result;
use crate::observer::{FitObserver, GuidedStep};
use crate::problem::{Pose, SiteProblem, V3};

/// Attachment grid: the link's psi values (for Asn {180, 165, -165} deg; every 30 deg for links
/// without a planar preference) x `n_phi` values of phi.
pub fn attachment_grid(psis: &[f64], n_phi: usize) -> Vec<(f64, f64)> {
    let mut grid = Vec::with_capacity(psis.len() * n_phi);
    for &psi in psis {
        for k in 0..n_phi {
            grid.push((
                psi,
                -std::f64::consts::PI + 2.0 * std::f64::consts::PI * k as f64 / n_phi as f64,
            ));
        }
    }
    grid
}

/// Relative slack of the quick grid scores of `attach_search`, far above their rounding error.
const RIGID_SLACK: f64 = 1e-6;

/// Best attachment (psi_N, phi_N, objective) per conformer on the grid (`methods.attach_search`).
/// The torsions are set once per conformer; every grid point is a rigid placement.
pub fn attach_search(
    problem: &SiteProblem,
    conformers: &[(Vec<f64>, usize)],
) -> Vec<(f64, f64, f64)> {
    let psis = if problem.full_psi_grid {
        crate::anchor::LinkTorsion::Free.grid()
    } else {
        problem.link_torsion.grid()
    };
    let grid = attachment_grid(&psis, 36);
    problem.counter.add_objective(conformers.len() * grid.len());
    conformers
        .par_iter()
        .map(|(tau, tpl)| {
            let local = problem.local(tau, *tpl);
            let e_prior = problem.prior_energy(tau, None);
            let exact = |psi: f64, phi: f64| {
                let x = problem.attach(&local, psi, phi);
                let (terms, _) = problem.placed_terms(&x, psi, None);
                terms.total + problem.w_prior * e_prior
            };
            // every grid point is the same conformer moved as a rigid body, so the part of the
            // observation such a motion leaves unchanged is computed once; it differs from the
            // per-placement value in the last bits only, and the grid points within reach of the
            // lowest are scored again in full, which keeps the choice and its energy exact
            let rigid = problem
                .observation
                .rigid_term(&local, problem.score_only.as_deref());
            let quick: Option<Vec<f64>> = rigid.map(|rigid| {
                grid.iter()
                    .map(|&(psi, phi)| {
                        let x = problem.attach(&local, psi, phi);
                        problem.placed_terms_rigid(&x, psi, rigid).total + problem.w_prior * e_prior
                    })
                    .collect()
            });
            let low = quick
                .as_ref()
                .map(|q| q.iter().copied().fold(f64::INFINITY, f64::min));
            let mut best = (0.0, 0.0, f64::INFINITY);
            for (k, &(psi, phi)) in grid.iter().enumerate() {
                if let (Some(quick), Some(low)) = (&quick, low)
                    && low.is_finite()
                    && quick[k] > low + RIGID_SLACK * (1.0 + low.abs())
                {
                    continue;
                }
                let e = exact(psi, phi);
                if e < best.2 {
                    best = (psi, phi, e);
                }
            }
            best
        })
        .collect()
}

/// Steering of the flow by the objective (`methods.guided_sample`): from `t >= start` the
/// velocity gets `-scale * g / rms(g)` with `g = dE/dtau` at the predicted endpoint
/// `tau_hat = tau + (1 - t) v`; the attachment angles are grid-searched on the predicted
/// endpoint every `attach_every` steps and moved by a normalised gradient step in between.
pub struct ObjectiveGuidance<'p> {
    pub problem: &'p SiteProblem,
    pub templates: Vec<usize>,
    pub psi: Vec<f64>,
    pub phi: Vec<f64>,
    pub scale: f64,
    pub start: f64,
    pub attach_every: usize,
    pub attach_step: f64,
}

impl Guidance for ObjectiveGuidance<'_> {
    fn correction(&mut self, ctx: &GuidanceContext) -> glycoflow_core::Result<Option<Vec<f32>>> {
        let t = ctx.step as f64 * (1.0 / ctx.steps as f64);
        if t < self.start {
            return Ok(None);
        }
        let nt = ctx.topology.n_torsions();
        let b = ctx.batch;
        let tau_hat: Vec<Vec<f64>> = (0..b)
            .map(|s| {
                ctx.tau_hat[s * nt..(s + 1) * nt]
                    .iter()
                    .map(|&v| v as f64)
                    .collect()
            })
            .collect();
        let k = ctx.step as i64 - (self.start * ctx.steps as f64).ceil() as i64;
        if k % self.attach_every as i64 == 0 {
            let conformers: Vec<(Vec<f64>, usize)> = tau_hat
                .iter()
                .cloned()
                .zip(self.templates.iter().copied())
                .collect();
            for (s, (psi, phi, _)) in attach_search(self.problem, &conformers)
                .into_iter()
                .enumerate()
            {
                self.psi[s] = psi;
                self.phi[s] = phi;
            }
        }
        let problem = self.problem;
        let n_heavy = problem
            .heavy_torsions()
            .iter()
            .filter(|h| **h)
            .count()
            .max(1);
        let link_turn = problem.link_turn();
        let results: Vec<(Vec<f32>, f64, f64)> = (0..b)
            .into_par_iter()
            .map(|s| {
                let pose = Pose {
                    tau: tau_hat[s].clone(),
                    psi: self.psi[s],
                    phi: self.phi[s],
                    template: self.templates[s],
                };
                let g = problem.evaluate(&pose, true).grad.expect("gradient");
                // rms over the torsions the objective can move (`SiteProblem::heavy_torsions`), with
                // the turn about the link bond (dE/dphi) in the norm: a pose pushed mainly as a whole
                // about the link takes a smaller torsion step and is turned by the attachment step
                let norm = (g.tau.iter().map(|v| v * v).sum::<f64>() + g.phi * g.phi)
                    .sqrt()
                    .max(1e-6)
                    / (n_heavy as f64).sqrt();
                let mut corr: Vec<f32> = g
                    .tau
                    .iter()
                    .map(|v| (-self.scale * v / norm) as f32)
                    .collect();
                // the flow's own aglycone torsion follows the turn about the link bond, so the
                // sample the network conditions on turns with the attachment
                if let Some(k) = link_turn {
                    corr[k] = (-self.scale * g.phi / norm) as f32;
                }
                let gn = (g.psi * g.psi + g.phi * g.phi).sqrt().max(1e-6);
                (
                    corr,
                    wrap(self.psi[s] - self.attach_step * g.psi / gn),
                    wrap(self.phi[s] - self.attach_step * g.phi / gn),
                )
            })
            .collect();
        problem.counter.add_objective_grad(b);
        let mut out = Vec::with_capacity(b * nt);
        for (s, (corr, psi, phi)) in results.into_iter().enumerate() {
            out.extend(corr);
            self.psi[s] = psi;
            self.phi[s] = phi;
        }
        Ok(Some(out))
    }
}

/// Uniform initial torsions for `n` conformers.
pub fn initial_torsions(seed: u64, n: usize, nt: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    uniform_torsions(&mut rng, n * nt)
}

fn templates_for(problem: &SiteProblem, templates: &[usize]) -> Vec<P3> {
    templates
        .iter()
        .flat_map(|&k| problem.glycan.templates[k].iter().copied())
        .collect()
}

/// Conformers per guided-sampling batch (`methods.guided_sample(chunk=256)`); the default of
/// `ProblemOptions::batch`.
pub const GUIDED_CHUNK: usize = 256;

/// Passes every step of the guided flow to an observer, then defers to the objective guidance.
struct Recorded<'a, 'p> {
    inner: ObjectiveGuidance<'p>,
    observer: &'a dyn FitObserver,
    first: usize,
}

impl Guidance for Recorded<'_, '_> {
    fn correction(&mut self, ctx: &GuidanceContext) -> glycoflow_core::Result<Option<Vec<f32>>> {
        let correction = self.inner.correction(ctx)?;
        self.observer.guided_step(&GuidedStep {
            first: self.first,
            step: ctx.step,
            steps: ctx.steps,
            t: ctx.t as f64,
            templates: &self.inner.templates,
            tau: ctx.tau,
            tau_hat: ctx.tau_hat,
            psi: &self.inner.psi,
            phi: &self.inner.phi,
            guided: correction.is_some(),
        });
        Ok(correction)
    }
}

/// Guided generation of `n` conformers (template `s % K` for conformer `s`), in batches of
/// `problem.batch`; returns the poses.
pub fn guided_sample(
    problem: &SiteProblem,
    sampler: &Sampler,
    n: usize,
    steps: usize,
    scale: f64,
    start: f64,
    seed: u64,
) -> Result<Vec<Pose>> {
    let k = problem.n_templates();
    let nt = problem.n_torsions;
    let tau0 = initial_torsions(seed, n, nt);
    let opts = SampleOptions {
        steps,
        method: Method::Heun,
    };
    let mut poses = Vec::with_capacity(n);
    let observer = problem.observer.get();
    let mut s0 = 0;
    while s0 < n {
        let m = problem.batch.unwrap_or(GUIDED_CHUNK).min(n - s0);
        let templates: Vec<usize> = (s0..s0 + m).map(|s| s % k).collect();
        let guidance = ObjectiveGuidance {
            problem,
            templates: templates.clone(),
            psi: vec![problem.link_torsion.start(); m],
            phi: vec![0.0; m],
            scale,
            start,
            attach_every: 4,
            attach_step: 0.05,
        };
        let templates_xyz = templates_for(problem, &templates);
        let tau_start = &tau0[s0 * nt..(s0 + m) * nt];
        let (guidance, tau) = match observer {
            None => {
                let mut guidance = guidance;
                let (_, tau) =
                    sampler.sample(&templates_xyz, tau_start, opts, Some(&mut guidance))?;
                (guidance, tau)
            }
            Some(observer) => {
                let mut recorded = Recorded {
                    inner: guidance,
                    observer,
                    first: s0,
                };
                let (_, tau) =
                    sampler.sample(&templates_xyz, tau_start, opts, Some(&mut recorded))?;
                observer.guided_step(&GuidedStep {
                    first: s0,
                    step: steps,
                    steps,
                    t: 1.0,
                    templates: &recorded.inner.templates,
                    tau: &tau,
                    tau_hat: &tau,
                    psi: &recorded.inner.psi,
                    phi: &recorded.inner.phi,
                    guided: false,
                });
                (recorded.inner, tau)
            }
        };
        problem.counter.add_nfe(m * opts.nfe());
        poses.extend((0..m).map(|s| {
            Pose {
                tau: tau[s * nt..(s + 1) * nt]
                    .iter()
                    .map(|&v| v as f64)
                    .collect(),
                psi: guidance.psi[s],
                phi: guidance.phi[s],
                template: templates[s],
            }
        }));
        s0 += m;
        if let Some(observer) = observer {
            observer.progress("guided sample", s0, n);
        }
    }
    Ok(poses)
}

/// Free-glycan GlycoFlow samples (no observation): torsions [n][T] (`methods.sample_prior`).
pub fn sample_free(
    problem: &SiteProblem,
    sampler: &Sampler,
    n: usize,
    steps: usize,
    method: Method,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    let k = problem.n_templates();
    let nt = problem.n_torsions;
    let tau0 = initial_torsions(seed, n, nt);
    let opts = SampleOptions { steps, method };
    // in batches of `problem.batch` (memory); conformers are independent
    let mut tau = Vec::with_capacity(n * nt);
    let mut s0 = 0;
    while s0 < n {
        let m = problem.batch.unwrap_or(n).min(n - s0);
        let templates: Vec<usize> = (s0..s0 + m).map(|s| s % k).collect();
        let (_, t) = sampler.sample(
            &templates_for(problem, &templates),
            &tau0[s0 * nt..(s0 + m) * nt],
            opts,
            None,
        )?;
        tau.extend(t);
        s0 += m;
        if let Some(observer) = problem.observer.get() {
            observer.progress("prior", s0, n);
        }
    }
    Ok((0..n)
        .map(|s| {
            tau[s * nt..(s + 1) * nt]
                .iter()
                .map(|&v| v as f64)
                .collect()
        })
        .collect())
}

/// Adam (PyTorch defaults: betas 0.9 / 0.999, eps 1e-8).
pub(crate) struct Adam {
    lr: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl Adam {
    pub(crate) fn new(n: usize, lr: f64) -> Self {
        Self {
            lr,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    pub(crate) fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        self.t += 1;
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            let denom = self.v[i].sqrt() / bc2.sqrt() + eps;
            params[i] -= self.lr / bc1 * self.m[i] / denom;
        }
    }
}

/// Continuous refinement of torsions and attachment angles (Adam), returning the best pose
/// visited and its objective (`methods.refine`).
pub fn refine(problem: &SiteProblem, poses: &[Pose], steps: usize, lr: f64) -> Vec<(Pose, f64)> {
    problem.counter.add_objective_grad(poses.len() * steps);
    poses
        .par_iter()
        .map(|start| {
            let nt = start.tau.len();
            let mut params: Vec<f64> = start
                .tau
                .iter()
                .copied()
                .chain([start.psi, start.phi])
                .collect();
            let mut adam = Adam::new(params.len(), lr);
            let mut best: Option<(Vec<f64>, f64)> = None;
            for _ in 0..steps {
                let pose = Pose {
                    tau: params[..nt].to_vec(),
                    psi: params[nt],
                    phi: params[nt + 1],
                    template: start.template,
                };
                let ev = problem.evaluate(&pose, true);
                let e = ev.terms.total;
                if best.as_ref().is_none_or(|(_, b)| e < *b) {
                    best = Some((params.clone(), e));
                }
                let g = ev.grad.expect("gradient");
                let grad: Vec<f64> = g.tau.iter().copied().chain([g.psi, g.phi]).collect();
                adam.step(&mut params, &grad);
            }
            let (p, e) = best.unwrap_or((params, f64::INFINITY));
            (
                Pose {
                    tau: p[..nt].iter().map(|v| wrap(*v)).collect(),
                    psi: wrap(p[nt]),
                    phi: wrap(p[nt + 1]),
                    template: start.template,
                },
                e,
            )
        })
        .collect()
}

/// In-place RMSD over the scored atoms.
pub fn rmsd(problem: &SiteProblem, a: &[V3], b: &[V3]) -> f64 {
    let mut s = 0.0;
    let mut n = 0usize;
    let scored = |i: usize| problem.score_only.as_ref().is_none_or(|m| m[i]);
    for i in (0..problem.n_atoms).filter(|&i| problem.keep[i] && scored(i)) {
        s += (0..3).map(|k| (a[i][k] - b[i][k]).powi(2)).sum::<f64>();
        n += 1;
    }
    (s / n.max(1) as f64).sqrt()
}

/// Greedy lowest-objective selection of up to `k` poses at least `min_rmsd` apart
/// (`methods.distinct`).
pub fn distinct(
    problem: &SiteProblem,
    x: &[Vec<V3>],
    e: &[f64],
    k: usize,
    min_rmsd: f64,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..e.len()).collect();
    order.sort_by(|&a, &b| e[a].total_cmp(&e[b]));
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        if !e[i].is_finite() {
            continue;
        }
        if keep
            .iter()
            .all(|&j| rmsd(problem, &x[i], &x[j]) >= min_rmsd)
        {
            keep.push(i);
        }
        if keep.len() >= k {
            break;
        }
    }
    keep
}

/// Refined basins of method B.
pub struct MethodB {
    pub poses: Vec<Pose>,
    pub energies: Vec<f64>,
    /// objective of each basin before refinement
    pub unrefined: Vec<f64>,
    /// the guided sample each basin started from
    pub sources: Vec<usize>,
}

/// Method B (`methods.method_b`): guided generation -> attachment grid search -> distinct
/// basins -> Adam refinement.
#[allow(clippy::too_many_arguments)]
pub fn method_b(
    problem: &SiteProblem,
    sampler: &Sampler,
    n_samples: usize,
    n_basins: usize,
    refine_steps: usize,
    refine_lr: f64,
    steps: usize,
    scale: f64,
    start: f64,
    seed: u64,
    pre_refine: usize,
    pre_refine_steps: usize,
) -> Result<MethodB> {
    let c = &problem.counter;
    let observer = problem.observer.get();
    let t0 = Instant::now();
    if let Some(o) = observer {
        o.stage("guided sample");
    }
    let poses = guided_sample(problem, sampler, n_samples, steps, scale, start, seed)?;
    c.add_stage("guided sample", t0);
    let t0 = Instant::now();
    if let Some(o) = observer {
        o.stage("select");
    }
    let conformers: Vec<(Vec<f64>, usize)> =
        poses.iter().map(|p| (p.tau.clone(), p.template)).collect();
    let grid = attach_search(problem, &conformers);
    let guided: Vec<f64> = poses
        .par_iter()
        .map(|p| problem.evaluate(p, false).terms.total)
        .collect();
    c.add_objective(poses.len());
    let mut chosen = Vec::with_capacity(poses.len());
    let mut energies = Vec::with_capacity(poses.len());
    for ((pose, (psi, phi, e_grid)), e_guided) in poses.into_iter().zip(grid).zip(guided) {
        if e_grid < e_guided {
            chosen.push(Pose { psi, phi, ..pose });
        } else {
            chosen.push(pose);
        }
        energies.push(e_grid.min(e_guided));
    }
    if let Some(o) = observer {
        o.samples(&chosen, &energies);
    }
    // Optionally refine the best few candidates briefly first: a sample near the right basin with a
    // contact still to resolve ranks low unrefined and would never be kept.
    let (candidates, energies, index): (Vec<Pose>, Vec<f64>, Vec<usize>) = if pre_refine > 0 {
        let mut order: Vec<usize> = (0..chosen.len()).collect();
        order.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]));
        order.truncate((pre_refine * n_basins).min(chosen.len()));
        let top: Vec<Pose> = order.iter().map(|&i| chosen[i].clone()).collect();
        let refined = refine(problem, &top, pre_refine_steps, refine_lr);
        let (p, e): (Vec<Pose>, Vec<f64>) = refined.into_iter().unzip();
        (p, e, order)
    } else {
        let n = chosen.len();
        (chosen, energies, (0..n).collect())
    };
    let chosen = candidates;
    let x: Vec<Vec<V3>> = chosen.par_iter().map(|p| problem.place(p)).collect();
    let local = distinct(problem, &x, &energies, n_basins, 1.5);
    // `sel` indexes the guided samples (for replays); `local` the candidates
    let sel: Vec<usize> = local.iter().map(|&i| index[i]).collect();
    c.add_stage("select", t0);
    let t0 = Instant::now();
    let starts: Vec<Pose> = local.iter().map(|&i| chosen[i].clone()).collect();
    let unrefined: Vec<f64> = local.iter().map(|&i| energies[i]).collect();
    if let Some(o) = observer {
        o.basins("select", &sel, &starts, &unrefined);
        o.stage("refine");
    }
    let refined = refine(problem, &starts, refine_steps, refine_lr);
    c.add_stage("refine", t0);
    let poses: Vec<Pose> = refined.iter().map(|(p, _)| p.clone()).collect();
    let energies: Vec<f64> = refined.iter().map(|(_, e)| *e).collect();
    if let Some(o) = observer {
        o.basins("refine", &sel, &poses, &energies);
    }
    Ok(MethodB {
        poses,
        energies,
        unrefined,
        sources: sel,
    })
}

/// Re-generate around the best `basins` poses (SDEdit): each one's torsions are taken back to time
/// `t` on the straight path from fresh noise, `wrap(tau0 + t wrap(tau - tau0))`, and integrated to
/// t = 1 again under the objective's guidance, `per_basin` times; the results get the attachment
/// grid and the same refinement as the search's basins. The model proposes local alternatives (a
/// branch turned, a linkage in its other basin) that a gradient step from the pose cannot reach.
#[allow(clippy::too_many_arguments)]
pub fn renoise(
    problem: &SiteProblem,
    sampler: &Sampler,
    poses: &[Pose],
    energies: &[f64],
    basins: usize,
    per_basin: usize,
    t: f64,
    steps: usize,
    scale: f64,
    start: f64,
    refine_steps: usize,
    refine_lr: f64,
    seed: u64,
) -> Result<(Vec<Pose>, Vec<f64>)> {
    let nt = problem.n_torsions;
    let mut order: Vec<usize> = (0..poses.len()).collect();
    order.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]));
    order.truncate(basins.min(poses.len()));
    let n = order.len() * per_basin;
    if n == 0 || nt == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let k = ((t * steps as f64).round() as usize).min(steps.saturating_sub(1));
    let tk = k as f64 / steps as f64;
    let noise = initial_torsions(seed ^ 0x5D3D_17A5, n, nt);
    let mut tau_t = Vec::with_capacity(n * nt);
    let mut templates = Vec::with_capacity(n);
    let mut psi = Vec::with_capacity(n);
    let mut phi = Vec::with_capacity(n);
    for (j, &b) in order.iter().enumerate() {
        for r in 0..per_basin {
            let s = j * per_basin + r;
            for (a, &x0) in poses[b].tau.iter().zip(&noise[s * nt..(s + 1) * nt]) {
                let x0 = x0 as f64;
                tau_t.push(wrap(x0 + tk * wrap(a - x0)) as f32);
            }
            templates.push(poses[b].template);
            psi.push(poses[b].psi);
            phi.push(poses[b].phi);
        }
    }
    let opts = SampleOptions {
        steps,
        method: Method::Heun,
    };
    let mut out = Vec::with_capacity(n);
    let mut s0 = 0;
    while s0 < n {
        let m = problem.batch.unwrap_or(GUIDED_CHUNK).min(n - s0);
        let tpl: Vec<usize> = templates[s0..s0 + m].to_vec();
        let mut guidance = ObjectiveGuidance {
            problem,
            templates: tpl.clone(),
            psi: psi[s0..s0 + m].to_vec(),
            phi: phi[s0..s0 + m].to_vec(),
            scale,
            start,
            attach_every: 4,
            attach_step: 0.05,
        };
        let (_, tau) = sampler.sample_from(
            &templates_for(problem, &tpl),
            &tau_t[s0 * nt..(s0 + m) * nt],
            k,
            opts,
            Some(&mut guidance),
        )?;
        problem
            .counter
            .add_nfe(m * (2 * (steps - k)).saturating_sub(1));
        for s in 0..m {
            out.push(Pose {
                tau: tau[s * nt..(s + 1) * nt]
                    .iter()
                    .map(|&v| v as f64)
                    .collect(),
                psi: guidance.psi[s],
                phi: guidance.phi[s],
                template: tpl[s],
            });
        }
        s0 += m;
    }
    // the attachment grid, as for the guided samples, then the search's refinement
    let conformers: Vec<(Vec<f64>, usize)> =
        out.iter().map(|p| (p.tau.clone(), p.template)).collect();
    let grid = attach_search(problem, &conformers);
    let starts: Vec<Pose> = out
        .into_iter()
        .zip(grid)
        .map(|(p, (psi, phi, e_grid))| {
            let e_now = problem.evaluate(&p, false).terms.total;
            if e_grid < e_now {
                Pose { psi, phi, ..p }
            } else {
                p
            }
        })
        .collect();
    let refined = refine(problem, &starts, refine_steps, refine_lr);
    Ok(refined.into_iter().unzip())
}

/// Inpainting under the objective's guidance: the `free` torsions of each conformer are generated
/// from noise with the map's guidance, the others follow the straight path from their noise to the
/// base pose's value and end on it exactly (as `support::complete_on_templates` pins them). The
/// attachment angles stay the base pose's: with the held torsions they place the part of the
/// glycan that is kept, and a grid search over half-generated branches would move it.
struct InpaintGuidance<'p> {
    inner: ObjectiveGuidance<'p>,
    /// [B*T]
    free: Vec<bool>,
    /// straight-path velocity of the pinned torsions [B*T]
    delta_pin: Vec<f32>,
    /// the base poses' attachment angles [B]
    psi: Vec<f64>,
    phi: Vec<f64>,
}

impl Guidance for InpaintGuidance<'_> {
    fn correction(&mut self, ctx: &GuidanceContext) -> glycoflow_core::Result<Option<Vec<f32>>> {
        let mut c = self
            .inner
            .correction(ctx)?
            .unwrap_or_else(|| vec![0.0; ctx.v.len()]);
        for (i, ci) in c.iter_mut().enumerate() {
            if !self.free[i] {
                *ci = self.delta_pin[i] - ctx.v[i];
            }
        }
        self.inner.psi.clone_from(&self.psi);
        self.inner.phi.clone_from(&self.phi);
        Ok(Some(c))
    }
}

/// Generate the `free[s]` torsions of `bases[s]` again under the map's guidance, the rest held
/// (`InpaintGuidance`); the results get the attachment grid and the search's refinement.
#[allow(clippy::too_many_arguments)]
pub fn inpaint(
    problem: &SiteProblem,
    sampler: &Sampler,
    bases: &[Pose],
    free: &[Vec<bool>],
    steps: usize,
    scale: f64,
    start: f64,
    refine_steps: usize,
    refine_lr: f64,
    seed: u64,
) -> Result<(Vec<Pose>, Vec<f64>)> {
    let nt = problem.n_torsions;
    let n = bases.len();
    if n == 0 || nt == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let tau0 = initial_torsions(seed ^ 0x1A9A_1A7E, n, nt);
    let opts = SampleOptions {
        steps,
        method: Method::Heun,
    };
    let mut out = Vec::with_capacity(n);
    let mut s0 = 0;
    while s0 < n {
        let m = problem.batch.unwrap_or(GUIDED_CHUNK).min(n - s0);
        let chunk = &bases[s0..s0 + m];
        let templates: Vec<usize> = chunk.iter().map(|p| p.template).collect();
        let start_tau = &tau0[s0 * nt..(s0 + m) * nt];
        let mut mask = Vec::with_capacity(m * nt);
        let mut delta_pin = Vec::with_capacity(m * nt);
        for (j, base) in chunk.iter().enumerate() {
            mask.extend_from_slice(&free[s0 + j]);
            for t in 0..nt {
                delta_pin.push(wrap(base.tau[t] as f32 - start_tau[j * nt + t]));
            }
        }
        let mut guidance = InpaintGuidance {
            inner: ObjectiveGuidance {
                problem,
                templates: templates.clone(),
                psi: chunk.iter().map(|p| p.psi).collect(),
                phi: chunk.iter().map(|p| p.phi).collect(),
                scale,
                start,
                attach_every: 4,
                attach_step: 0.05,
            },
            free: mask,
            delta_pin,
            psi: chunk.iter().map(|p| p.psi).collect(),
            phi: chunk.iter().map(|p| p.phi).collect(),
        };
        let (_, tau) = sampler.sample(
            &templates_for(problem, &templates),
            start_tau,
            opts,
            Some(&mut guidance),
        )?;
        problem.counter.add_nfe(m * opts.nfe());
        for (j, base) in chunk.iter().enumerate() {
            out.push(Pose {
                tau: (0..nt)
                    .map(|t| {
                        if free[s0 + j][t] {
                            tau[j * nt + t] as f64
                        } else {
                            base.tau[t]
                        }
                    })
                    .collect(),
                psi: base.psi,
                phi: base.phi,
                template: base.template,
            });
        }
        s0 += m;
    }
    let refined = refine(problem, &out, refine_steps, refine_lr);
    Ok(refined.into_iter().unzip())
}

/// Refinement steps of every core-first sample on the core's objective before the cores are ranked.
pub const CORE_REFINE_STEPS: usize = 50;

/// Residues of the core: the root and the two residues above it on any path (an N-glycan's
/// chitobiose and beta-mannose).
pub fn core_atoms(problem: &SiteProblem) -> Vec<bool> {
    problem
        .glycan
        .res_paths
        .iter()
        .map(|p| p != "agl" && p.matches('/').count() <= 2)
        .collect()
}

/// Core-first search: guided samples scored on the core's atoms only, so the map places the core
/// without the branches pulling it; the `cores` best distinct ones (by core RMSD) then have
/// everything beyond the core generated again under the full objective, `per_core` times
/// (`inpaint`). Empty when the glycan is no larger than its core.
#[allow(clippy::too_many_arguments)]
pub fn core_first(
    problem: &mut SiteProblem,
    sampler: &Sampler,
    n_samples: usize,
    cores: usize,
    per_core: usize,
    contact_weight: f64,
    steps: usize,
    scale: f64,
    start: f64,
    refine_steps: usize,
    refine_lr: f64,
    seed: u64,
) -> Result<(Vec<Pose>, Vec<f64>)> {
    let core = core_atoms(problem);
    let beyond: Vec<bool> = problem
        .glycan
        .res_paths
        .iter()
        .zip(&core)
        .map(|(p, c)| p != "agl" && !c)
        .collect();
    let topology = &problem.glycan.topology;
    let free = crate::support::torsions_moving_only(&topology.quads, &topology.distal, &beyond);
    if !free.iter().any(|f| *f) || n_samples == 0 || cores == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // the core on its own: score its atoms only, at the final objective's contact weight (a
    // wrong core in contact looks better at the search's), with psi searched on the whole circle
    // (here and in the completions below)
    let search_weights = (problem.w_env, problem.w_self);
    // the observer follows the search's own guided samples; these would mix into their trace
    let observer = std::mem::take(&mut problem.observer);
    problem.score_only = Some(core);
    problem.full_psi_grid = true;
    (problem.w_env, problem.w_self) = (contact_weight, contact_weight);
    let picked = (|| -> Result<Vec<Pose>> {
        let problem: &SiteProblem = problem;
        let poses = guided_sample(
            problem,
            sampler,
            n_samples,
            steps,
            scale,
            start,
            seed ^ 0xC07E,
        )?;
        let conformers: Vec<(Vec<f64>, usize)> =
            poses.iter().map(|p| (p.tau.clone(), p.template)).collect();
        let grid = attach_search(problem, &conformers);
        let chosen: Vec<Pose> = poses
            .into_iter()
            .zip(grid)
            .map(|(p, (psi, phi, e_grid))| {
                if e_grid < problem.evaluate(&p, false).terms.total {
                    Pose { psi, phi, ..p }
                } else {
                    p
                }
            })
            .collect();
        // refined on the core's own objective first: a sample a couple of angstroms off the
        // right core ranks below a wrong core that happens to fit, and is never kept unrefined
        let refined = refine(problem, &chosen, CORE_REFINE_STEPS, refine_lr);
        let chosen: Vec<Pose> = refined.iter().map(|(p, _)| p.clone()).collect();
        // ranked without the link energy: this step looks for where the map puts the core; the
        // full objective, link energy included, judges the completed poses
        let energies: Vec<f64> = refined
            .par_iter()
            .map(|(p, e)| e - problem.evaluate(p, false).terms.e_att)
            .collect();
        let x: Vec<Vec<V3>> = chosen.par_iter().map(|p| problem.place(p)).collect();
        let keep = distinct(problem, &x, &energies, cores, 1.0);
        let picked: Vec<Pose> = keep.iter().map(|&i| chosen[i].clone()).collect();
        if let Some(o) = problem.observer.get() {
            let e: Vec<f64> = keep.iter().map(|&i| energies[i]).collect();
            o.basins("core first", &vec![usize::MAX; picked.len()], &picked, &e);
        }
        Ok(picked)
    })();
    problem.score_only = None;
    (problem.w_env, problem.w_self) = search_weights;
    problem.observer = observer;
    let completed = picked.and_then(|picked| {
        let bases: Vec<Pose> = picked
            .iter()
            .flat_map(|p| std::iter::repeat_n(p.clone(), per_core))
            .collect();
        let masks = vec![free; bases.len()];
        let (poses, energies) = inpaint(
            problem,
            sampler,
            &bases,
            &masks,
            steps,
            scale,
            start,
            refine_steps,
            refine_lr,
            seed,
        )?;
        // the best completions of every core, never a global best few: a right core still paying
        // for a strained link or poor branches ranks below wrong ones here, and the final
        // objective must get to see it
        let mut out = (Vec::new(), Vec::new());
        for group in (0..poses.len()).collect::<Vec<_>>().chunks(per_core.max(1)) {
            let mut g = group.to_vec();
            g.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]));
            for &i in g.iter().take(COMPLETIONS_KEPT) {
                out.0.push(poses[i].clone());
                out.1.push(energies[i]);
            }
        }
        if let Some(o) = problem.observer.get() {
            o.basins(
                "core completions",
                &vec![usize::MAX; out.0.len()],
                &out.0,
                &out.1,
            );
        }
        Ok(out)
    });
    problem.full_psi_grid = false;
    completed
}

/// Completions kept per core in the core-first search (`core_first`).
pub const COMPLETIONS_KEPT: usize = 2;

/// The `k` lowest-energy poses at least `min_rmsd` apart over the core's atoms ([`core_atoms`]).
pub fn distinct_cores(
    problem: &SiteProblem,
    poses: &[Pose],
    energies: &[f64],
    k: usize,
    min_rmsd: f64,
) -> Vec<usize> {
    let core = core_atoms(problem);
    let x: Vec<Vec<V3>> = poses.par_iter().map(|p| problem.place(p)).collect();
    let d = |a: &[V3], b: &[V3]| {
        let (mut s, mut n) = (0.0, 0usize);
        for i in (0..problem.n_atoms).filter(|&i| problem.keep[i] && core[i]) {
            s += (0..3).map(|j| (a[i][j] - b[i][j]).powi(2)).sum::<f64>();
            n += 1;
        }
        (s / n.max(1) as f64).sqrt()
    };
    let mut order: Vec<usize> = (0..poses.len()).collect();
    order.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]));
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        if energies[i].is_finite() && keep.iter().all(|&j| d(&x[i], &x[j]) >= min_rmsd) {
            keep.push(i);
        }
        if keep.len() >= k {
            break;
        }
    }
    keep
}

/// Branch re-noising: for each of the best `basins` poses with distinct cores and each subtree off the root (a residue
/// and everything above it), that subtree's torsions are generated again from noise under the
/// map's guidance with the rest of the pose held, `per_branch` times (`inpaint`). Whole-pose
/// re-noising (`renoise`) moves every torsion a little; this moves one branch as far as it likes.
#[allow(clippy::too_many_arguments)]
pub fn branch_renoise(
    problem: &SiteProblem,
    sampler: &Sampler,
    poses: &[Pose],
    energies: &[f64],
    basins: usize,
    per_branch: usize,
    steps: usize,
    scale: f64,
    start: f64,
    refine_steps: usize,
    refine_lr: f64,
    seed: u64,
) -> Result<(Vec<Pose>, Vec<f64>)> {
    let topology = &problem.glycan.topology;
    let masks: Vec<Vec<bool>> = crate::support::residue_tree(problem)
        .into_iter()
        .filter(|(root, _)| root != "r")
        .map(|(_, members)| {
            let atoms: Vec<bool> = problem
                .glycan
                .res_paths
                .iter()
                .map(|p| members.contains(p))
                .collect();
            crate::support::torsions_moving_only(&topology.quads, &topology.distal, &atoms)
        })
        .filter(|m| m.iter().any(|f| *f))
        .collect();
    // the best poses with distinct cores: a right core losing on its branches still gets them
    // generated again
    let order = distinct_cores(problem, poses, energies, basins, 1.0);
    let mut bases = Vec::new();
    let mut free = Vec::new();
    for &b in order.iter() {
        for m in &masks {
            for _ in 0..per_branch {
                bases.push(poses[b].clone());
                free.push(m.clone());
            }
        }
    }
    inpaint(
        problem,
        sampler,
        &bases,
        &free,
        steps,
        scale,
        start,
        refine_steps,
        refine_lr,
        seed,
    )
}
