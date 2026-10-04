//! ODE sampler (`flow.sample`): Euler or Heun integration of the torsion flow from explicit initial
//! torsions on fixed-geometry templates, with an optional guidance hook.

use candle_core::Tensor;

use crate::error::{Error, Result};
use crate::geometry::{rotate_torsions, set_torsions, wrap, P3};
use crate::model::{GraphBatch, Prepared, TorsionFlowNet};
use crate::topology::{Glycan, Topology};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Euler,
    /// 2nd order, two network calls per step (one on the last step).
    Heun,
}

impl std::str::FromStr for Method {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "euler" => Ok(Method::Euler),
            "heun" => Ok(Method::Heun),
            _ => Err(Error::Invalid(format!(
                "unknown method {s:?} (euler | heun)"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SampleOptions {
    pub steps: usize,
    pub method: Method,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            steps: 32,
            method: Method::Heun,
        }
    }
}

impl SampleOptions {
    /// Network evaluations per sample.
    pub fn nfe(&self) -> usize {
        match self.method {
            Method::Euler => self.steps,
            Method::Heun => 2 * self.steps - 1,
        }
    }
}

/// State handed to a [`Guidance`] hook once per integration step, after the (Heun-averaged)
/// velocity is known and before the step is taken. Arrays are row-major over the batch.
pub struct GuidanceContext<'a> {
    pub step: usize,
    pub steps: usize,
    /// time of this step (`i * dt`, as fed to the network)
    pub t: f32,
    pub batch: usize,
    pub topology: &'a Topology,
    pub tokens: &'a [[u32; 5]],
    /// current conformers [B*N]
    pub coords: &'a [P3],
    /// current torsions [B*T]
    pub tau: &'a [f32],
    /// velocity [B*T]
    pub v: &'a [f32],
    /// predicted endpoint torsions wrap(tau + (1 - t) v) [B*T]
    pub tau_hat: &'a [f32],
}

/// Guidance hook: returns a torsion-velocity correction [B*T] added to the velocity, or None.
pub trait Guidance {
    fn correction(&mut self, ctx: &GuidanceContext) -> Result<Option<Vec<f32>>>;
}

/// Sampler for one glycan (graph prepared once, reused for every batch).
pub struct Sampler<'m> {
    pub model: &'m TorsionFlowNet,
    pub topology: Topology,
    pub tokens: Vec<[u32; 5]>,
    prep: Prepared,
}

impl<'m> Sampler<'m> {
    pub fn new(
        model: &'m TorsionFlowNet,
        topology: Topology,
        tokens: Vec<[u32; 5]>,
    ) -> Result<Self> {
        let graph = GraphBatch::single(&tokens, &topology, model.device())?;
        let prep = model.prepare(graph)?;
        Ok(Self {
            model,
            topology,
            tokens,
            prep,
        })
    }

    pub fn for_glycan(
        model: &'m TorsionFlowNet,
        glycan: &Glycan,
        vocab: &crate::topology::Vocab,
    ) -> Result<Self> {
        Self::new(model, glycan.topology.clone(), glycan.tokens(vocab))
    }

    pub fn n_atoms(&self) -> usize {
        self.topology.n_atoms
    }

    pub fn n_torsions(&self) -> usize {
        self.topology.n_torsions()
    }

    /// One network evaluation: coords [B*N], tau [B*T], t [B] -> velocities [B*T].
    /// On the CPU (feature `parallel`) the batch is split into sub-batches evaluated on rayon
    /// threads, since candle's CPU elementwise kernels are single-threaded.
    pub fn velocity(&self, coords: &[P3], tau: &[f32], t: &[f32]) -> Result<Vec<f32>> {
        if self.n_torsions() == 0 {
            return Ok(Vec::new());
        }
        #[cfg(feature = "parallel")]
        if self.model.device().is_cpu() && t.len() > 1 {
            let (b, n, nt) = (t.len(), self.n_atoms(), self.n_torsions());
            use rayon::prelude::*;
            let parts = rayon::current_num_threads().min(b);
            let chunk = b.div_ceil(parts);
            let out: Vec<Vec<f32>> = (0..b.div_ceil(chunk))
                .into_par_iter()
                .map(|k| {
                    let (s, e) = (k * chunk, ((k + 1) * chunk).min(b));
                    self.velocity_batch(&coords[s * n..e * n], &tau[s * nt..e * nt], &t[s..e])
                })
                .collect::<Result<_>>()?;
            return Ok(out.concat());
        }
        self.velocity_batch(coords, tau, t)
    }

    fn velocity_batch(&self, coords: &[P3], tau: &[f32], t: &[f32]) -> Result<Vec<f32>> {
        let b = t.len();
        let n = self.n_atoms();
        let nt = self.n_torsions();
        let dev = self.model.device();
        let flat: Vec<f32> = coords.iter().flat_map(|p| p.iter().copied()).collect();
        let x = Tensor::from_vec(flat, (b, n, 3), dev)?;
        let tau_t = Tensor::from_slice(tau, (b, nt), dev)?;
        let tt = Tensor::from_slice(t, b, dev)?;
        let v = self.model.forward(&self.prep, &x, &tau_t, &tt)?;
        Ok(v.flatten_all()?.to_vec1::<f32>()?)
    }

    fn rotate_batch(&self, coords: &mut [P3], delta: &[f32]) {
        let n = self.n_atoms();
        let nt = self.n_torsions();
        let (q, d) = (&self.topology.quads, &self.topology.distal);
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            coords
                .par_chunks_mut(n)
                .zip(delta.par_chunks(nt))
                .for_each(|(x, dl)| rotate_torsions(x, q, d, dl));
        }
        #[cfg(not(feature = "parallel"))]
        for (x, dl) in coords.chunks_mut(n).zip(delta.chunks(nt)) {
            rotate_torsions(x, q, d, dl);
        }
    }

    /// Integrate the flow for a batch: `templates` [B*N] (one local geometry per conformer),
    /// `tau0` [B*T]. Returns (coords [B*N], torsions [B*T]), as `flow.sample(..., tau0=tau0)`.
    pub fn sample<'g>(
        &self,
        templates: &[P3],
        tau0: &[f32],
        opts: SampleOptions,
        mut guidance: Option<&mut (dyn Guidance + 'g)>,
    ) -> Result<(Vec<P3>, Vec<f32>)> {
        let n = self.n_atoms();
        let nt = self.n_torsions();
        if !templates.len().is_multiple_of(n) {
            return Err(Error::Invalid(
                "templates length is not a multiple of the atom count".into(),
            ));
        }
        let b = templates.len() / n;
        if tau0.len() != b * nt {
            return Err(Error::Invalid(format!(
                "tau0 has {} values, expected {}",
                tau0.len(),
                b * nt
            )));
        }
        if opts.steps == 0 {
            return Err(Error::Invalid("steps must be > 0".into()));
        }
        let mut coords = templates.to_vec();
        let mut tau = tau0.to_vec();
        if nt == 0 {
            return Ok((coords, tau));
        }
        for (x, tg) in coords.chunks_mut(n).zip(tau.chunks(nt)) {
            set_torsions(x, &self.topology.quads, &self.topology.distal, tg);
        }
        let dt = 1.0f64 / opts.steps as f64;
        let dt32 = dt as f32;
        for i in 0..opts.steps {
            let t_i = (i as f64 * dt) as f32;
            let t = vec![t_i; b];
            let mut v = self.velocity(&coords, &tau, &t)?;
            if opts.method == Method::Heun && i + 1 < opts.steps {
                let trial: Vec<f32> = v.iter().map(|vi| vi * dt32).collect();
                let mut c2 = coords.clone();
                self.rotate_batch(&mut c2, &trial);
                let tau2: Vec<f32> = tau.iter().zip(&trial).map(|(a, d)| wrap(a + d)).collect();
                let t2 = vec![t_i + dt32; b];
                let v2 = self.velocity(&c2, &tau2, &t2)?;
                v = v.iter().zip(&v2).map(|(a, c)| (a + c) * 0.5).collect();
            }
            if let Some(g) = guidance.as_deref_mut() {
                let omt = 1.0 - t_i;
                let tau_hat: Vec<f32> = tau
                    .iter()
                    .zip(&v)
                    .map(|(a, vi)| wrap(a + omt * vi))
                    .collect();
                let ctx = GuidanceContext {
                    step: i,
                    steps: opts.steps,
                    t: t_i,
                    batch: b,
                    topology: &self.topology,
                    tokens: &self.tokens,
                    coords: &coords,
                    tau: &tau,
                    v: &v,
                    tau_hat: &tau_hat,
                };
                if let Some(corr) = g.correction(&ctx)? {
                    if corr.len() != v.len() {
                        return Err(Error::Invalid(
                            "guidance correction has the wrong length".into(),
                        ));
                    }
                    for (vi, c) in v.iter_mut().zip(&corr) {
                        *vi += c;
                    }
                }
            }
            let step: Vec<f32> = v.iter().map(|vi| vi * dt32).collect();
            self.rotate_batch(&mut coords, &step);
            for (a, d) in tau.iter_mut().zip(&step) {
                *a = wrap(*a + d);
            }
        }
        Ok((coords, tau))
    }

    /// Sample `n_samples` conformers in chunks of at most `atom_budget / N^2` (as
    /// `evaluate.sample_entry`); sample `s` uses template `s % templates.len()` and torsions
    /// `tau0[s*T..(s+1)*T]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_ensemble<'g>(
        &self,
        templates: &[Vec<P3>],
        n_samples: usize,
        tau0: &[f32],
        opts: SampleOptions,
        atom_budget: usize,
        mut guidance: Option<&mut (dyn Guidance + 'g)>,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<(Vec<P3>, Vec<f32>)> {
        let n = self.n_atoms();
        if tau0.len() != n_samples * self.n_torsions() {
            return Err(Error::Invalid(
                "tau0 length != n_samples * n_torsions".into(),
            ));
        }
        if templates.is_empty() {
            return Err(Error::Invalid("no templates".into()));
        }
        let chunk = (atom_budget / (n * n)).clamp(1, n_samples.max(1));
        let mut coords = Vec::with_capacity(n_samples * n);
        let mut tors = Vec::with_capacity(n_samples * self.n_torsions());
        let mut s = 0;
        while s < n_samples {
            let m = chunk.min(n_samples - s);
            let tpl: Vec<P3> = (s..s + m)
                .flat_map(|j| templates[j % templates.len()].iter().copied())
                .collect();
            let t0 = &tau0[s * self.n_torsions()..(s + m) * self.n_torsions()];
            let (x, t) = self.sample(&tpl, t0, opts, guidance.as_deref_mut())?;
            coords.extend(x);
            tors.extend(t);
            s += m;
            progress(s, n_samples);
        }
        Ok((coords, tors))
    }
}
