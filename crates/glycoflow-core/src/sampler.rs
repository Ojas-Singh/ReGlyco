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
    /// current ring shapes [B*R*3] in [`crate::pucker::PUCKER_SCALE`] units, empty unless rings are
    /// being generated
    pub q: &'a [f32],
    /// ring-shape velocity [B*R*3], empty unless rings are being generated
    pub vq: &'a [f32],
}

/// Guidance hook: returns a torsion-velocity correction [B*T] added to the velocity, or None.
pub trait Guidance {
    fn correction(&mut self, ctx: &GuidanceContext) -> Result<Option<Vec<f32>>>;

    /// A ring-shape velocity correction [B*R*3] added to `vq` while rings are being generated, in
    /// [`crate::pucker::PUCKER_SCALE`] units. The default is none, which leaves ring generation
    /// unguided -- the reference's default too (`flow.sample(guidance_pucker=0)`).
    fn correction_pucker(&mut self, _ctx: &GuidanceContext) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
}

/// Sampler for one glycan (graph prepared once, reused for every batch).
pub struct Sampler<'m> {
    pub model: &'m TorsionFlowNet,
    pub topology: Topology,
    pub tokens: Vec<[u32; 5]>,
    prep: Prepared,
    /// the glycan's registration with an external network backend (WebGPU), if any
    external: Option<crate::backend::Registration>,
}

impl<'m> Sampler<'m> {
    pub fn new(
        model: &'m TorsionFlowNet,
        topology: Topology,
        tokens: Vec<[u32; 5]>,
    ) -> Result<Self> {
        let graph = GraphBatch::single(&tokens, &topology, model.device())?;
        let prep = model.prepare(graph)?;
        let external = crate::backend::register(&topology, &tokens);
        Ok(Self {
            model,
            topology,
            tokens,
            prep,
            external,
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
        if let Some(r) = &self.external {
            if let Some(v) = r.backend.velocity(r.glycan, coords, tau, t) {
                if v.len() == tau.len() {
                    return Ok(v);
                }
            }
        }
        self.velocity_candle(coords, tau, t)
    }

    /// Whether network evaluations go to an external backend.
    pub fn uses_backend(&self) -> bool {
        self.external.is_some()
    }

    /// [`Sampler::velocity`] on candle, whatever backend is installed (reference for backends).
    pub fn velocity_candle(&self, coords: &[P3], tau: &[f32], t: &[f32]) -> Result<Vec<f32>> {
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

    /// candle only: `velocity` has already offered the batch to the backend
    fn velocity_batch(&self, coords: &[P3], tau: &[f32], t: &[f32]) -> Result<Vec<f32>> {
        Ok(self.velocity_joint_on(coords, tau, t, None, false)?.0)
    }

    /// Number of sugar rings whose shape a ring-head model generates.
    pub fn n_rings(&self) -> usize {
        self.topology.rings.len()
    }

    /// Whether this sampler can generate ring shapes (a ring-head checkpoint, and rings to apply it
    /// to). [`Sampler::sample`] holds rings at the template even then; [`Sampler::sample_joint`]
    /// generates them.
    pub fn generates_rings(&self) -> bool {
        self.model.has_pucker() && self.n_rings() > 0
    }

    /// [R*3]: which Cremer-Pople components each ring has (`flow.pucker_channels`). A five-ring has
    /// no q3, so that channel stays at zero in the prior and in the integration.
    pub fn pucker_channels(&self) -> Vec<f32> {
        self.topology
            .rings
            .iter()
            .flat_map(|r| [1.0, 1.0, if r.len() == 6 { 1.0 } else { 0.0 }])
            .collect()
    }

    /// One network evaluation with ring shapes: `q` [B*R*3] in [`PUCKER_SCALE`] units ->
    /// (torsion velocities [B*T], ring-shape velocities [B*R*3]).
    pub fn velocity_joint(
        &self,
        coords: &[P3],
        tau: &[f32],
        t: &[f32],
        q: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        if q.is_none() {
            return Ok((self.velocity(coords, tau, t)?, None));
        }
        if self.n_torsions() == 0 && self.n_rings() == 0 {
            return Ok((Vec::new(), Some(Vec::new())));
        }
        #[cfg(feature = "parallel")]
        if self.model.device().is_cpu() && t.len() > 1 {
            let (b, n, nt, nq) = (
                t.len(),
                self.n_atoms(),
                self.n_torsions(),
                self.n_rings() * crate::pucker::PUCKER_DIM,
            );
            use rayon::prelude::*;
            let parts = rayon::current_num_threads().min(b);
            let chunk = b.div_ceil(parts);
            let qq = q.unwrap();
            let out: Vec<(Vec<f32>, Option<Vec<f32>>)> = (0..b.div_ceil(chunk))
                .into_par_iter()
                .map(|k| {
                    let (s, e) = (k * chunk, ((k + 1) * chunk).min(b));
                    self.velocity_joint_batch(
                        &coords[s * n..e * n],
                        &tau[s * nt..e * nt],
                        &t[s..e],
                        Some(&qq[s * nq..e * nq]),
                    )
                })
                .collect::<Result<_>>()?;
            let mut v = Vec::with_capacity(b * nt);
            let mut vq = Vec::with_capacity(b * nq);
            for (a, c) in out {
                v.extend(a);
                vq.extend(c.unwrap_or_default());
            }
            return Ok((v, Some(vq)));
        }
        self.velocity_joint_batch(coords, tau, t, q)
    }

    /// [`Sampler::velocity_joint`] on candle, whatever backend is installed (reference for backends).
    pub fn velocity_joint_candle(
        &self,
        coords: &[P3],
        tau: &[f32],
        t: &[f32],
        q: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        self.velocity_joint_on(coords, tau, t, q, false)
    }

    fn velocity_joint_batch(
        &self,
        coords: &[P3],
        tau: &[f32],
        t: &[f32],
        q: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        self.velocity_joint_on(coords, tau, t, q, true)
    }

    fn velocity_joint_on(
        &self,
        coords: &[P3],
        tau: &[f32],
        t: &[f32],
        q: Option<&[f32]>,
        use_backend: bool,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let b = t.len();
        let n = self.n_atoms();
        let nt = self.n_torsions();
        if nt == 0 && q.is_none() {
            return Ok((Vec::new(), None));
        }
        // an external backend first; candle when there is none or it declines
        if let Some(r) = self.external.as_ref().filter(|_| use_backend) {
            match q.filter(|_| self.n_rings() > 0 && self.model.has_pucker()) {
                None => {
                    if let Some(v) = r.backend.velocity(r.glycan, coords, tau, t) {
                        if v.len() == tau.len() {
                            return Ok((v, None));
                        }
                    }
                }
                Some(qq) => {
                    if let Some((v, vq)) = r.backend.velocity_joint(r.glycan, coords, tau, t, qq) {
                        if v.len() == tau.len() && vq.len() == qq.len() {
                            return Ok((v, Some(vq)));
                        }
                    }
                }
            }
        }
        let dev = self.model.device();
        let flat: Vec<f32> = coords.iter().flat_map(|p| p.iter().copied()).collect();
        let x = Tensor::from_vec(flat, (b, n, 3), dev)?;
        let tau_t = Tensor::from_slice(tau, (b, nt), dev)?;
        let tt = Tensor::from_slice(t, b, dev)?;
        let qt = match q.filter(|_| self.n_rings() > 0) {
            Some(qq) => Some(Tensor::from_slice(
                qq,
                (b, self.n_rings(), crate::pucker::PUCKER_DIM),
                dev,
            )?),
            None => None,
        };
        let (v, vq) = self
            .model
            .forward_joint(&self.prep, &x, &tau_t, &tt, qt.as_ref())?;
        Ok((
            v.flatten_all()?.to_vec1::<f32>()?,
            match vq {
                Some(g) => Some(g.flatten_all()?.to_vec1::<f32>()?),
                None => None,
            },
        ))
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

    /// Conformers from templates [B*N], torsions [B*T] and, when given, ring shapes [B*R*3] in
    /// [`crate::pucker::PUCKER_SCALE`] units, written into `x` [B*N]: each ring is deformed to its
    /// shape first, then the torsions are set (`density._coords_of`).
    pub fn rebuild(&self, x: &mut [P3], templates: &[P3], tau: &[f32], q: Option<&[f32]>) {
        let (n, nt) = (self.n_atoms(), self.n_torsions());
        let nq = self.n_rings() * crate::pucker::PUCKER_DIM;
        for (s, x) in x.chunks_mut(n).enumerate() {
            x.copy_from_slice(&templates[s * n..(s + 1) * n]);
            if let Some(q) = q.filter(|_| nq > 0) {
                let target: Vec<[f64; crate::pucker::PUCKER_DIM]> = q[s * nq..(s + 1) * nq]
                    .chunks(crate::pucker::PUCKER_DIM)
                    .map(|c| {
                        let s = crate::pucker::PUCKER_SCALE;
                        [c[0] as f64 * s, c[1] as f64 * s, c[2] as f64 * s]
                    })
                    .collect();
                // the Cremer-Pople inverse is computed in f64, as the parity fixtures measure it
                let mut xd: Vec<[f64; 3]> = x
                    .iter()
                    .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                    .collect();
                crate::pucker::set_puckers(&mut xd, &self.topology.rings, &target);
                for (o, d) in x.iter_mut().zip(&xd) {
                    *o = [d[0] as f32, d[1] as f32, d[2] as f32];
                }
            }
            set_torsions(x, &self.topology.quads, &self.topology.distal, &tau[s * nt..(s + 1) * nt]);
        }
    }

    /// The prior point a conformer comes from: integrate the flow back from t = 1 to t = 0 with
    /// Euler steps (`density.encode`). `tau` [B*T] and, for a ring-head model generating rings, `q`
    /// [B*R*3] describe the conformers on `templates` [B*N]. Returns (tau0, q0); decoding them with
    /// Euler at the same `steps` returns the conformer to O(1/steps).
    pub fn encode(
        &self,
        templates: &[P3],
        tau: &[f32],
        q: Option<&[f32]>,
        steps: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = self.n_atoms();
        let b = templates.len() / n.max(1);
        let chan = self.pucker_channels();
        let nq = chan.len();
        let mut tau = tau.to_vec();
        let mut q = q.map(<[f32]>::to_vec);
        let dt = 1.0f32 / steps.max(1) as f32;
        let mut x = templates.to_vec();
        for i in (0..steps.max(1)).rev() {
            let t = vec![i as f32 * dt; b];
            self.rebuild(&mut x, templates, &tau, q.as_deref());
            let (v, vq) = self.velocity_joint(&x, &tau, &t, q.as_deref())?;
            for (a, d) in tau.iter_mut().zip(&v) {
                *a = wrap(*a - d * dt);
            }
            if let (Some(q), Some(vq)) = (q.as_mut(), vq) {
                for (k, (a, d)) in q.iter_mut().zip(&vq).enumerate() {
                    *a = (*a - d * dt) * chan[k % nq];
                }
            }
        }
        Ok((tau, q.unwrap_or_default()))
    }

    /// Integrate the flow for a batch: `templates` [B*N] (one local geometry per conformer),
    /// `tau0` [B*T]. Returns (coords [B*N], torsions [B*T]), as `flow.sample(..., tau0=tau0)`.
    ///
    /// Ring shapes are those of the templates and do not move, which is what fitting a known
    /// structure wants. With a ring-head checkpoint this is still the exact torsion velocity (see
    /// [`TorsionFlowNet::forward`](crate::model::TorsionFlowNet::forward)); to generate ring shapes
    /// as well, call [`Sampler::sample_joint`].
    pub fn sample<'g>(
        &self,
        templates: &[P3],
        tau0: &[f32],
        opts: SampleOptions,
        guidance: Option<&mut (dyn Guidance + 'g)>,
    ) -> Result<(Vec<P3>, Vec<f32>)> {
        let (x, tau, _) = self.sample_inner(templates, tau0, None, opts, 0, guidance)?;
        Ok((x, tau))
    }

    /// Integrate from step `start` of an `opts.steps` grid (t = start / steps) to t = 1, with
    /// `tau_t` [B*T] the torsions at that time: the second half of a run, whose first half is
    /// replaced by a point on the straight path from noise to a conformer,
    /// `tau_t = wrap(tau0 + t wrap(tau1 - tau0))` (SDEdit). Rings are held at the templates, as in
    /// [`Sampler::sample`]; guidance sees the same step indices as a full run.
    pub fn sample_from<'g>(
        &self,
        templates: &[P3],
        tau_t: &[f32],
        start: usize,
        opts: SampleOptions,
        guidance: Option<&mut (dyn Guidance + 'g)>,
    ) -> Result<(Vec<P3>, Vec<f32>)> {
        let (x, tau, _) = self.sample_inner(templates, tau_t, None, opts, start.min(opts.steps), guidance)?;
        Ok((x, tau))
    }

    /// Integrate torsions and ring shapes together (`flow.sample` with a ring head): `q0` [B*R*3]
    /// is the ring-shape prior draw, in [`crate::pucker::PUCKER_SCALE`] units -- standard normal on
    /// the components a ring has ([`Sampler::pucker_channels`]), zero on the rest. Returns
    /// (coords [B*N], torsions [B*T], ring shapes [B*R*3]).
    pub fn sample_joint<'g>(
        &self,
        templates: &[P3],
        tau0: &[f32],
        q0: &[f32],
        opts: SampleOptions,
        guidance: Option<&mut (dyn Guidance + 'g)>,
    ) -> Result<(Vec<P3>, Vec<f32>, Vec<f32>)> {
        if !self.model.has_pucker() {
            return Err(Error::Invalid(
                "this checkpoint has no ring head (head_pucker), so it cannot generate ring shapes"
                    .into(),
            ));
        }
        self.sample_inner(templates, tau0, Some(q0), opts, 0, guidance)
    }

    fn sample_inner<'g>(
        &self,
        templates: &[P3],
        tau0: &[f32],
        q0: Option<&[f32]>,
        opts: SampleOptions,
        start: usize,
        mut guidance: Option<&mut (dyn Guidance + 'g)>,
    ) -> Result<(Vec<P3>, Vec<f32>, Vec<f32>)> {
        let n = self.n_atoms();
        let nt = self.n_torsions();
        if templates.len() % n != 0 {
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
        let nq = self.n_rings() * crate::pucker::PUCKER_DIM;
        if let Some(q) = q0 {
            if q.len() != b * nq {
                return Err(Error::Invalid(format!(
                    "q0 has {} values, expected {}",
                    q.len(),
                    b * nq
                )));
            }
        }
        let mut coords = templates.to_vec();
        let mut tau = tau0.to_vec();
        let mut q = q0.unwrap_or_default().to_vec();
        if nt == 0 && q.is_empty() {
            return Ok((coords, tau, q));
        }
        // With rings in the state the conformer is rebuilt from the template each step (deforming a
        // ring moves its substituents, so torsions cannot be applied incrementally on top);
        // without them the torsions are rotated in place, as before.
        let chan = self.pucker_channels();
        let joint = q0.is_some();
        let build = |x: &mut [P3], tau: &[f32], q: &[f32]| {
            self.rebuild(x, templates, tau, joint.then_some(q));
        };
        build(&mut coords, &tau, &q);
        let advance = |q: &[f32], vq: &[f32], h: f32| -> Vec<f32> {
            q.iter()
                .zip(vq)
                .enumerate()
                .map(|(k, (a, d))| (a + h * d) * chan[k % nq])
                .collect()
        };
        let dt = 1.0f64 / opts.steps as f64;
        let dt32 = dt as f32;
        for i in start..opts.steps {
            let t_i = (i as f64 * dt) as f32;
            let t = vec![t_i; b];
            let (mut v, vq) = self.velocity_joint(&coords, &tau, &t, q0.map(|_| q.as_slice()))?;
            let mut vq = vq.unwrap_or_default();
            if opts.method == Method::Heun && i + 1 < opts.steps {
                let trial: Vec<f32> = v.iter().map(|vi| vi * dt32).collect();
                let tau2: Vec<f32> = tau.iter().zip(&trial).map(|(a, d)| wrap(a + d)).collect();
                let mut c2 = coords.clone();
                let q2 = advance(&q, &vq, dt32);
                if joint {
                    build(&mut c2, &tau2, &q2);
                } else {
                    self.rotate_batch(&mut c2, &trial);
                }
                let t2 = vec![t_i + dt32; b];
                let (v2, vq2) =
                    self.velocity_joint(&c2, &tau2, &t2, q0.map(|_| q2.as_slice()))?;
                v = v.iter().zip(&v2).map(|(a, c)| (a + c) * 0.5).collect();
                if let Some(vq2) = vq2 {
                    vq = vq.iter().zip(&vq2).map(|(a, c)| (a + c) * 0.5).collect();
                }
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
                    q: &q,
                    vq: &vq,
                };
                // both corrections are read off the same state, then applied
                let corr = g.correction(&ctx)?;
                let corr_q = if joint {
                    g.correction_pucker(&ctx)?
                } else {
                    None
                };
                if let Some(corr) = corr {
                    if corr.len() != v.len() {
                        return Err(Error::Invalid(
                            "guidance correction has the wrong length".into(),
                        ));
                    }
                    for (vi, c) in v.iter_mut().zip(&corr) {
                        *vi += c;
                    }
                }
                if let Some(corr) = corr_q {
                    if corr.len() != vq.len() {
                        return Err(Error::Invalid(
                            "guidance ring correction has the wrong length".into(),
                        ));
                    }
                    for (vi, c) in vq.iter_mut().zip(&corr) {
                        *vi += c;
                    }
                }
            }
            let step: Vec<f32> = v.iter().map(|vi| vi * dt32).collect();
            for (a, d) in tau.iter_mut().zip(&step) {
                *a = wrap(*a + d);
            }
            if joint {
                q = advance(&q, &vq, dt32);
                build(&mut coords, &tau, &q);
            } else {
                self.rotate_batch(&mut coords, &step);
            }
        }
        Ok((coords, tau, q))
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
        guidance: Option<&mut (dyn Guidance + 'g)>,
        progress: impl FnMut(usize, usize),
    ) -> Result<(Vec<P3>, Vec<f32>)> {
        let (x, t, _) = self.ensemble_inner(
            templates,
            n_samples,
            tau0,
            None,
            opts,
            atom_budget,
            guidance,
            progress,
        )?;
        Ok((x, t))
    }

    /// [`Sampler::sample_ensemble`] generating ring shapes as well: `q0` [n_samples*R*3] is the
    /// ring-shape prior draw. Returns (coords, torsions, ring shapes).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_ensemble_joint<'g>(
        &self,
        templates: &[Vec<P3>],
        n_samples: usize,
        tau0: &[f32],
        q0: &[f32],
        opts: SampleOptions,
        atom_budget: usize,
        guidance: Option<&mut (dyn Guidance + 'g)>,
        progress: impl FnMut(usize, usize),
    ) -> Result<(Vec<P3>, Vec<f32>, Vec<f32>)> {
        if !self.model.has_pucker() {
            return Err(Error::Invalid(
                "this checkpoint has no ring head (head_pucker), so it cannot generate ring shapes"
                    .into(),
            ));
        }
        self.ensemble_inner(
            templates,
            n_samples,
            tau0,
            Some(q0),
            opts,
            atom_budget,
            guidance,
            progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ensemble_inner<'g>(
        &self,
        templates: &[Vec<P3>],
        n_samples: usize,
        tau0: &[f32],
        q0: Option<&[f32]>,
        opts: SampleOptions,
        atom_budget: usize,
        mut guidance: Option<&mut (dyn Guidance + 'g)>,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<(Vec<P3>, Vec<f32>, Vec<f32>)> {
        let n = self.n_atoms();
        let nt = self.n_torsions();
        let nq = self.n_rings() * crate::pucker::PUCKER_DIM;
        if tau0.len() != n_samples * nt {
            return Err(Error::Invalid(
                "tau0 length != n_samples * n_torsions".into(),
            ));
        }
        if let Some(q) = q0 {
            if q.len() != n_samples * nq {
                return Err(Error::Invalid("q0 length != n_samples * 3 * n_rings".into()));
            }
        }
        if templates.is_empty() {
            return Err(Error::Invalid("no templates".into()));
        }
        let chunk = (atom_budget / (n * n)).clamp(1, n_samples.max(1));
        let mut coords = Vec::with_capacity(n_samples * n);
        let mut tors = Vec::with_capacity(n_samples * nt);
        let mut puck = Vec::with_capacity(q0.map_or(0, |_| n_samples * nq));
        let mut s = 0;
        while s < n_samples {
            let m = chunk.min(n_samples - s);
            let tpl: Vec<P3> = (s..s + m)
                .flat_map(|j| templates[j % templates.len()].iter().copied())
                .collect();
            let (x, t, qq) = self.sample_inner(
                &tpl,
                &tau0[s * nt..(s + m) * nt],
                q0.map(|q| &q[s * nq..(s + m) * nq]),
                opts,
                0,
                guidance.as_deref_mut(),
            )?;
            coords.extend(x);
            tors.extend(t);
            puck.extend(qq);
            s += m;
            progress(s, n_samples);
        }
        Ok((coords, tors, puck))
    }
}
