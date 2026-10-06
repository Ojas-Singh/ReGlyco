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

/// Attachment grid: psi_N in {180, 165, -165} deg x `n_phi` values of phi_N.
pub fn attachment_grid(n_phi: usize) -> Vec<(f64, f64)> {
    let mut grid = Vec::with_capacity(3 * n_phi);
    for psi in [180.0f64, 165.0, -165.0] {
        for k in 0..n_phi {
            grid.push((
                psi.to_radians(),
                -std::f64::consts::PI + 2.0 * std::f64::consts::PI * k as f64 / n_phi as f64,
            ));
        }
    }
    grid
}

/// Best attachment (psi_N, phi_N, objective) per conformer on the grid (`methods.attach_search`).
/// The torsions are set once per conformer; every grid point is a rigid placement.
pub fn attach_search(
    problem: &SiteProblem,
    conformers: &[(Vec<f64>, usize)],
) -> Vec<(f64, f64, f64)> {
    let grid = attachment_grid(36);
    problem.counter.add_objective(conformers.len() * grid.len());
    conformers
        .par_iter()
        .map(|(tau, tpl)| {
            let local = problem.local(tau, *tpl);
            let e_prior = problem.prior_energy(tau, None);
            let mut best = (0.0, 0.0, f64::INFINITY);
            for &(psi, phi) in &grid {
                let x = problem.attach(&local, psi, phi);
                let (terms, _) = problem.placed_terms(&x, psi, None);
                let e = terms.total + problem.w_prior * e_prior;
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
                let norm =
                    g.tau.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6) / (nt as f64).sqrt();
                let corr = g
                    .tau
                    .iter()
                    .map(|v| (-self.scale * v / norm) as f32)
                    .collect();
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
            psi: vec![std::f64::consts::PI; m],
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
                let (_, tau) = sampler.sample(&templates_xyz, tau_start, opts, Some(&mut guidance))?;
                (guidance, tau)
            }
            Some(observer) => {
                let mut recorded = Recorded {
                    inner: guidance,
                    observer,
                    first: s0,
                };
                let (_, tau) = sampler.sample(&templates_xyz, tau_start, opts, Some(&mut recorded))?;
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
    for i in (0..problem.n_atoms).filter(|&i| problem.keep[i]) {
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
    let x: Vec<Vec<V3>> = chosen.par_iter().map(|p| problem.place(p)).collect();
    let sel = distinct(problem, &x, &energies, n_basins, 1.5);
    c.add_stage("select", t0);
    let t0 = Instant::now();
    let starts: Vec<Pose> = sel.iter().map(|&i| chosen[i].clone()).collect();
    let unrefined: Vec<f64> = sel.iter().map(|&i| energies[i]).collect();
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
