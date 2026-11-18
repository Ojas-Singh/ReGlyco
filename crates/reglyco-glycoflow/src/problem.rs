//! One fitting problem (`glycoflow/fitting/problem.py::SiteProblem`): GlycoFlow templates of the
//! site's glycan, their placement on the protein, and the single objective used for search and
//! for selection:
//!
//! ```text
//! E = E_obs(x) + w_env E_env(x) + w_self E_self(x) + E_attach(psi_N) + w_prior E_prior(tau)
//! ```
//!
//! * `E_obs`: the observation (density: `-log-likelihood` of the profiled site likelihood);
//! * `E_env`: soft overlap with protein / symmetry-mate / other-glycan heavy atoms (two
//!   precomputed penalty grids, carbon and polar probe, sampled trilinearly) plus explicit pairs
//!   with the site residue more than three bonds away;
//! * `E_self`: GlycoFlow's contact energy over glycan atom pairs >= 4 bonds apart;
//! * `E_attach = kappa (1 + cos psi_N)`, kappa = 1/(10 deg)^2 (Asn amide trans);
//! * `E_prior`: GlycoFlow marginal prior ([`crate::prior`]).
//!
//! Gradients are analytic: `dE/dx` for every term, `dE/dtau` through the torsion Jacobian,
//! `dE/dpsi_N` and `dE/dphi_N` as rigid rotations of the glycan about CG->ND2 and ND2->C1.

use glycoflow_core::geometry::P3;
use glycoflow_core::geometry::{set_torsions, torsion_gradient};
use glycoflow_core::guidance::{C_POLAR_FLOOR, CC_FLOOR, clash_energy_grad};
use glycoflow_core::rng::{SplitMix64, uniform_torsions};
use glycoflow_core::{Glycan, ResidueLibrary, Vocab};

use crate::counter::Counter;
use crate::error::{Result, invalid};
use crate::observation::{Observation, ObservedAtoms};
use crate::prior::MarginalPrior;
use crate::site::Site;
use crate::symmetry::EnvAtom;

pub type V3 = [f64; 3];

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
fn unit(a: V3) -> V3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Linkage geometry (bond length A, angle at the link atom in degrees) of a site residue; see
/// [`crate::anchor`].
pub fn linkage(residue: &str) -> (f64, f64) {
    crate::anchor::anchor(residue).map_or((1.45, 123.0), |a| (a.bond, a.angle))
}

/// Place d with |cd| = bond, angle bcd = angle, dihedral abcd = torsion (`attach.nerf`).
pub fn nerf(a: V3, b: V3, c: V3, bond: f64, angle: f64, torsion: f64) -> V3 {
    let bc = unit(sub(c, b));
    let n = unit(cross(sub(b, a), bc));
    let m = cross(n, bc);
    let d = [
        -bond * angle.cos(),
        bond * angle.sin() * torsion.cos(),
        bond * angle.sin() * torsion.sin(),
    ];
    [0, 1, 2].map(|i| c[i] + d[0] * bc[i] + d[1] * m[i] + d[2] * n[i])
}

/// Orthonormal frame (columns e1, e2, e3) at `origin`: e1 towards `axis`, e2 in the plane of
/// `plane`.
fn frame(origin: V3, axis: V3, plane: V3) -> [[f64; 3]; 3] {
    let e1 = unit(sub(axis, origin));
    let v = sub(plane, origin);
    let d = dot(v, e1);
    let e2 = unit([v[0] - d * e1[0], v[1] - d * e1[1], v[2] - d * e1[2]]);
    let e3 = cross(e1, e2);
    [
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]],
    ]
}

/// Dihedral a-b-c-d (radians).
pub fn dihedral(a: V3, b: V3, c: V3, d: V3) -> f64 {
    glycoflow_core::geometry::dihedral(a, b, c, d)
}

/// Polar-polar contact floor of the fitting objective (A). GlycoFlow's sampler guidance uses 2.5
/// (MD H-bonded minimum); the fit uses 2.6 so refined contacts stay clear of ReGlyco's
/// validator limit (heavy-atom van der Waals overlap > 0.6 A: O-O < 2.44, N-O < 2.47 A).
pub const FIT_POLAR_FLOOR: f64 = 2.6;
/// Floor for carbon-oxygen/nitrogen pairs three bonds apart within the glycan (eclipsed
/// hydroxymethyl and glycosidic geometries; the validator checks these pairs too).
pub const FIT_ONE_FOUR_FLOOR: f64 = 2.7;
/// 1-4 pairs across a rotatable bond (glycosidic phi/psi, exocyclic omega): carbon-carbon and
/// oxygen-oxygen floors 0.1-0.16 A above the validator's hard limits (vdW overlap 0.6 A: C-C 2.80,
/// O-O 2.44); eclipsed torsions put these pairs closest (C-O pairs use `FIT_ONE_FOUR_FLOOR`).
pub const FIT_ONE_FOUR_CC_FLOOR: f64 = 2.9;
pub const FIT_ONE_FOUR_POLAR_FLOOR: f64 = 2.6;

/// Distance of the 1-4 carbon-polar and carbon-carbon floors above the validator's limits (A).
pub const ONE_FOUR_FLOOR_MARGIN: f64 = 0.08;
/// Least distance of the other floors above the validator's limits (A): polar pairs 2.6 against
/// N-O 2.47, carbon-polar 2.8 against C-N 2.65, carbon-carbon 3.0 against 2.80.
pub const FAR_FLOOR_MARGIN: f64 = 0.12;

/// Penalty grids `sum_j relu(floor - d)^2` over environment atoms for a carbon probe and a polar
/// probe (`problem.clash_grids`), on the density box.
#[derive(Debug, Clone)]
pub struct ClashGrids {
    pub origin: V3,
    pub spacing: f64,
    pub dims: [usize; 3],
    pub carbon: Vec<f32>,
    pub polar: Vec<f32>,
}

impl ClashGrids {
    pub fn build(atoms: &[EnvAtom], origin: V3, spacing: f64, dims: [usize; 3]) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        let mut carbon = vec![0f64; n];
        let mut polar = vec![0f64; n];
        let rad = (CC_FLOOR as f64 / spacing).ceil() as i64 + 1;
        for atom in atoms {
            let is_c = atom.atomic_number == 6.0;
            let (floor_c, floor_p) = if is_c {
                (CC_FLOOR as f64, C_POLAR_FLOOR as f64)
            } else {
                (C_POLAR_FLOOR as f64, FIT_POLAR_FLOOR)
            };
            let base = [0, 1, 2]
                .map(|a| ((atom.position[a] - origin[a]) / spacing).round_ties_even() as i64);
            for di in -rad..=rad {
                let i = base[0] + di;
                if i < 0 || i >= dims[0] as i64 {
                    continue;
                }
                for dj in -rad..=rad {
                    let j = base[1] + dj;
                    if j < 0 || j >= dims[1] as i64 {
                        continue;
                    }
                    for dk in -rad..=rad {
                        let k = base[2] + dk;
                        if k < 0 || k >= dims[2] as i64 {
                            continue;
                        }
                        let p = [
                            origin[0] + i as f64 * spacing,
                            origin[1] + j as f64 * spacing,
                            origin[2] + k as f64 * spacing,
                        ];
                        let d = norm(sub(p, atom.position));
                        let idx = (i as usize * dims[1] + j as usize) * dims[2] + k as usize;
                        if d < floor_c {
                            carbon[idx] += (floor_c - d).powi(2);
                        }
                        if d < floor_p {
                            polar[idx] += (floor_p - d).powi(2);
                        }
                    }
                }
            }
        }
        Self {
            origin,
            spacing,
            dims,
            carbon: carbon.into_iter().map(|v| v as f32).collect(),
            polar: polar.into_iter().map(|v| v as f32).collect(),
        }
    }

    /// Trilinear value and gradient; outside the box the position is clamped to the border and
    /// the gradient along that axis is zero (`grid_sample(padding_mode="border")`).
    pub fn sample(&self, carbon_probe: bool, p: V3) -> (f64, V3) {
        let data = if carbon_probe {
            &self.carbon
        } else {
            &self.polar
        };
        let mut base = [0usize; 3];
        let mut frac = [0f64; 3];
        let mut live = [true; 3];
        for a in 0..3 {
            let top = (self.dims[a] - 1) as f64;
            let g = (p[a] - self.origin[a]) / self.spacing;
            live[a] = (0.0..=top).contains(&g);
            let g = g.clamp(0.0, top);
            let f = g.floor().min(top - 1.0);
            base[a] = f as usize;
            frac[a] = g - f;
        }
        let (d1, d2) = (self.dims[1], self.dims[2]);
        let mut value = 0.0;
        let mut grad = [0.0; 3];
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let v = data[((base[0] + dx) * d1 + base[1] + dy) * d2 + base[2] + dz] as f64;
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
        for a in 0..3 {
            grad[a] = if live[a] { grad[a] / self.spacing } else { 0.0 };
        }
        (value, grad)
    }
}

/// Objective terms of one pose (`SiteProblem.terms`).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Terms {
    pub total: f64,
    /// observation log-likelihood (density: gain / (2 noise variance x independent volume))
    pub loglik: f64,
    pub partial_cc: f64,
    pub gain: f64,
    pub scale: f64,
    pub e_env: f64,
    pub e_self: f64,
    pub e_att: f64,
    pub e_prior: f64,
}

/// Gradient of the total objective with respect to the pose parameters.
#[derive(Debug, Clone)]
pub struct PoseGradient {
    pub tau: Vec<f64>,
    pub psi: f64,
    pub phi: f64,
}

/// A pose: torsions, attachment angles (psi_N, phi_N) and pucker template.
#[derive(Debug, Clone)]
pub struct Pose {
    pub tau: Vec<f64>,
    pub psi: f64,
    pub phi: f64,
    pub template: usize,
}

#[derive(Debug, Clone)]
pub struct ProblemOptions {
    pub w_env: f64,
    pub w_self: f64,
    pub w_prior: f64,
    pub amide_sd_deg: f64,
    /// pucker templates (first: majority puckers; others drawn from `ring_source`)
    pub n_templates: usize,
    /// where the ring shapes of templates 2..n come from
    pub ring_source: RingSource,
    pub template_seed: u64,
    /// scoring-ball radius; default: bound on the glycan's reach + 2.5 A
    pub region_radius: Option<f64>,
    /// measure the density likelihood's empirical noise inflation (`null_inflation`)
    pub null_calibration: bool,
    /// conformers per network batch in guided and prior sampling (memory: the network's pair
    /// activations take about [`PAIR_BYTES`] N^2 bytes per conformer). None: batches of
    /// [`crate::search::GUIDED_CHUNK`] for guided sampling and one batch for the prior.
    /// Conformers are independent, so this changes nothing but memory and floating-point
    /// summation order.
    pub batch: Option<usize>,
    /// cap on the network activations of one batch (MB): the batch shrinks for a large glycan so
    /// that `batch * PAIR_BYTES * N^2` stays within it (at least one conformer). None: no cap.
    /// The browser sets it, since its heap never shrinks and holds the peak.
    pub batch_memory_mb: Option<f64>,
    /// bond lengths, angles and chiral volumes from the monomer library ([`crate::ideal`]): the
    /// pucker templates are brought to them and the Cartesian restraints hold them. Off: the
    /// geometry of the residue library's templates, as the Python reference.
    pub ideal_geometry: bool,
    /// six-rings only in the chair of the majority build ([`crate::ring::chair_templates`])
    pub chair_rings: bool,
}

/// Conformers per network batch for a glycan of `n_atoms` (`ProblemOptions::batch`, capped by
/// `batch_memory_mb`).
pub fn network_batch(options: &ProblemOptions, n_atoms: usize) -> Option<usize> {
    let cap = options
        .batch_memory_mb
        .map(|mb| ((mb * 1048576.0 / (PAIR_BYTES * (n_atoms * n_atoms) as f64)) as usize).max(1));
    match (options.batch, cap) {
        (Some(b), Some(c)) => Some(b.min(c).max(1)),
        (Some(b), None) => Some(b.max(1)),
        (None, c) => c,
    }
}

/// Peak network activation memory per conformer and atom pair, CPU, f32 (measured: 4IUG A:914 and
/// 5MGR B:169, `examples/memprobe.rs`).
pub const PAIR_BYTES: f64 = 400.0;

impl Default for ProblemOptions {
    fn default() -> Self {
        Self {
            w_env: 10.0,
            w_self: 10.0,
            // weak: GlycoFlow proposes and keeps density-insensitive torsions plausible; the map
            // decides where it has evidence (Python problem.SiteProblem.w_prior)
            w_prior: 0.25,
            amide_sd_deg: 10.0,
            n_templates: 16,
            ring_source: RingSource::Auto,
            template_seed: 0,
            region_radius: None,
            null_calibration: true,
            batch: None,
            batch_memory_mb: None,
            ideal_geometry: true,
            chair_rings: true,
        }
    }
}

/// Where the ring shapes of a fit's pucker templates come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RingSource {
    /// generated when the model has a ring head, else the library's (the benchmark's better
    /// choice for each model: `docs/fitting_results.md` section 16)
    #[default]
    Auto,
    /// the residue library's MD pucker states, drawn with their populations (each ring on its own)
    Library,
    /// generated by a ring-head GlycoFlow model together with the torsions: every template's rings
    /// are one joint draw from the model, so they carry the correlations between rings and with
    /// the conformation that a per-residue library draw cannot
    Generated,
}

impl RingSource {
    /// whether the fit's templates take generated ring shapes with this model
    pub fn generates(self, model: &crate::GlycoflowModel) -> bool {
        match self {
            RingSource::Auto => model.net.has_pucker(),
            RingSource::Library => false,
            RingSource::Generated => true,
        }
    }
}

/// Replace templates `1..` of `glycan` by the majority build with model-generated ring shapes:
/// `n - 1` unguided joint samples (Heun, `steps`), of which only the ring shapes are kept -- the
/// fit sets the torsions itself. Template 0 (majority puckers) stays, so the reference pose of the
/// library fit is still there. Fails if the model has no ring head.
pub fn generate_ring_templates(
    glycan: &mut Glycan,
    model: &crate::GlycoflowModel,
    steps: usize,
    seed: u64,
) -> Result<()> {
    let k = glycan.templates.len();
    if k < 2 {
        return Ok(());
    }
    let tokens = glycan.tokens(&model.meta.vocab);
    let sampler =
        glycoflow_core::sampler::Sampler::new(&model.net, glycan.topology.clone(), tokens)?;
    if !sampler.generates_rings() {
        return Err(invalid(
            "ring shapes can only be generated by a GlycoFlow model with a ring head",
        ));
    }
    let n = k - 1;
    let mut rng = SplitMix64::new(seed ^ 0x51A4_9C3D);
    let tau0 = uniform_torsions(&mut rng, n * sampler.n_torsions());
    let q0 = glycoflow_core::rng::ring_prior(&mut rng, &sampler.pucker_channels(), n);
    let base = glycan.templates[0].clone();
    let tpl: Vec<P3> = (0..n).flat_map(|_| base.iter().copied()).collect();
    let opts = glycoflow_core::sampler::SampleOptions {
        steps: steps.max(1),
        method: glycoflow_core::sampler::Method::Heun,
    };
    let (_, _, q) = sampler.sample_joint(&tpl, &tau0, &q0, opts, None)?;
    let nq = sampler.n_rings() * glycoflow_core::PUCKER_DIM;
    for s in 0..n {
        let mut x = vec![[0f32; 3]; base.len()];
        // the torsions of the template are irrelevant (the fit sets them); keep the majority's
        let keep_tau = glycoflow_core::geometry::dihedrals(&base, &glycan.topology.quads);
        sampler.rebuild(&mut x, &base, &keep_tau, Some(&q[s * nq..(s + 1) * nq]));
        glycan.templates[s + 1] = x;
    }
    Ok(())
}

/// GlycoFlow templates of a sequence: the majority-pucker build and `n - 1` builds with puckers
/// drawn from the MD populations (`evaluate.build_templates`, Rust RNG).
pub fn build_glycan(
    library: &ResidueLibrary,
    sequence: &str,
    n_templates: usize,
    seed: u64,
) -> Result<Glycan> {
    let mut rng = SplitMix64::new(seed);
    let mut builds = vec![library.build(sequence, None)?];
    for _ in 1..n_templates.max(1) {
        builds.push(library.build_sampled(sequence, None, &mut rng)?);
    }
    Ok(Glycan::from_builds(&builds)?)
}

/// Upper bound on the distance of any glycan atom from the aglycone oxygen (`_max_span`): BFS
/// path length from O1 over template-0 bond lengths, x 0.85 (realistic extension) + 1.5 A.
pub fn max_span(glycan: &Glycan) -> f64 {
    let x = &glycan.templates[0];
    let n = x.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &[a, b] in &glycan.bonds {
        let d = [0, 1, 2].map(|k| x[a][k] - x[b][k]);
        let w = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f64;
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    let mut dist = vec![f64::NAN; n];
    // from the aglycone oxygen (atom 0 unless an all-atom build put HO1 first)
    let o1 = (0..n)
        .find(|&i| glycan.res_paths[i] == "agl" && glycan.atom_names[i] == "O1")
        .unwrap_or(0);
    dist[o1] = 0.0;
    let mut queue = std::collections::VecDeque::from([o1]);
    while let Some(u) = queue.pop_front() {
        for &(v, w) in &adj[u] {
            if dist[v].is_nan() {
                dist[v] = dist[u] + w;
                queue.push_back(v);
            }
        }
    }
    0.85 * dist
        .iter()
        .copied()
        .filter(|d| !d.is_nan())
        .fold(0.0, f64::max)
        + 1.5
}

pub struct SiteProblem {
    pub sequence: String,
    pub glycan: Glycan,
    pub tokens: Vec<[u32; 5]>,
    /// pucker templates (f64 copies of `glycan.templates`)
    pub templates: Vec<Vec<V3>>,
    pub n_atoms: usize,
    pub n_torsions: usize,
    /// aglycone oxygen, root C1 and O5
    pub o1: usize,
    pub c1: usize,
    pub o5: usize,
    /// scored atoms: the glycan's heavy atoms. An all-atom model also places the polar hydrogens
    /// (it needs them as input), but the map, the environment and the deposits a fit is judged
    /// against are heavy-atom, so the hydrogens ride along with their torsions and score nothing.
    pub keep: Vec<bool>,
    /// atoms the Cartesian stage may move: everything but the aglycone placeholder
    pub movable: Vec<bool>,
    /// templates `1..` carry ring shapes generated by GlycoFlow (`RingSource::Generated`)
    pub generated_rings: bool,
    /// restrained pairs and centres, with their ideal values (`ProblemOptions::ideal_geometry`)
    pub ideal: crate::ideal::Targets,
    pub is_c: Vec<bool>,
    /// anchor atoms (A, B, link), e.g. CB, CG, ND2
    pub anchor: [V3; 3],
    pub residue_name: String,
    pub bond: f64,
    pub angle: f64,
    pub observation: Box<dyn Observation>,
    pub grids: ClashGrids,
    pub site_xyz: Vec<V3>,
    pub site_names: Vec<String>,
    /// (glycan atom, site atom, floor)
    pub site_pairs: Vec<(usize, usize, f64)>,
    /// (glycan atom i < j, floor) >= 4 bonds apart
    pub self_pairs: Vec<(usize, usize, f32)>,
    pub w_env: f64,
    pub w_self: f64,
    pub w_prior: f64,
    /// stiffness of the link torsion psi (from `amide_sd_deg`); planar links only
    pub amide_kappa: f64,
    /// preference of psi = A-B-link-C1 (the Asn amide, the Trp ring plane, or none)
    pub link_torsion: crate::anchor::LinkTorsion,
    pub prior: Option<MarginalPrior>,
    /// scoring-ball radius
    pub radius: f64,
    pub counter: Counter,
    /// conformers per network batch (`ProblemOptions::batch`)
    pub batch: Option<usize>,
    /// while set, the density and contact terms score only these atoms (of `keep`): the
    /// core-first search places the core on its own (`search::core_first`); None otherwise
    pub score_only: Option<Vec<bool>>,
    /// while set, the attachment grid covers psi on the whole circle (every 30 deg) and not only
    /// near the link's planar preference, which the link energy still charges: the core-first
    /// search can then reach a strained link the map asks for (deposits with twisted Asn amides)
    pub full_psi_grid: bool,
    pub observer: crate::observer::Observer,
}

/// One evaluated pose.
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub x: Vec<V3>,
    pub terms: Terms,
    pub grad: Option<PoseGradient>,
}

impl SiteProblem {
    /// Set up the problem. `observation` is prepared here; `grid_box` = (origin, spacing, dims)
    /// of the clash grids (the density box for density fitting).
    pub fn new(
        site: &Site,
        glycan: Glycan,
        vocab: &Vocab,
        observation: Box<dyn Observation>,
        grid_box: (V3, f64, [usize; 3]),
        radius: f64,
        options: &ProblemOptions,
    ) -> Result<Self> {
        Self::with_rings(
            site,
            glycan,
            vocab,
            observation,
            grid_box,
            radius,
            options,
            false,
        )
    }

    /// [`SiteProblem::new`]; `generated_rings`: templates `1..` of the glycan carry ring shapes
    /// generated by GlycoFlow.
    #[allow(clippy::too_many_arguments)]
    pub fn with_rings(
        site: &Site,
        glycan: Glycan,
        vocab: &Vocab,
        mut observation: Box<dyn Observation>,
        grid_box: (V3, f64, [usize; 3]),
        radius: f64,
        options: &ProblemOptions,
        generated_rings: bool,
    ) -> Result<Self> {
        let n = glycan.n_atoms();
        let nt = glycan.n_torsions();
        // The aglycone oxygen stands in for the link atom. It is atom 0 of a heavy-atom build; an
        // all-atom build also carries the reducing end's hydroxyl hydrogen (HO1) and roots the
        // topology there, so it is found by name.
        let o1 = (0..n)
            .find(|&i| glycan.res_paths[i] == "agl" && glycan.atom_names[i] == "O1")
            .ok_or_else(|| invalid("the template has no aglycone oxygen (O1)"))?;
        let find = |name: &str| {
            (0..n)
                .find(|&i| glycan.res_paths[i] == "r" && glycan.atom_names[i] == name)
                .ok_or_else(|| invalid(format!("root residue has no {name}")))
        };
        let (c1, o5) = (find("C1")?, find("O5")?);
        for (q, d) in glycan.topology.quads.iter().zip(&glycan.topology.distal) {
            // The turn about O1-C1 itself (HO1-O1-C1-O5 in an all-atom build) moves C1 and O5
            // rigidly about the bond to O1, and the attachment re-derives the frame from exactly
            // those three atoms, so it cancels: only HO1, an unscored aglycone atom, moves.
            let about_link = (q[1] == o1 && q[2] == c1) || (q[1] == c1 && q[2] == o1);
            if !about_link && (d.contains(&o1) || d.contains(&c1) || d.contains(&o5)) {
                return Err(invalid(
                    "a rotatable torsion moves the root attachment frame (O1, C1, O5)",
                ));
            }
        }
        let movable: Vec<bool> = glycan.res_paths.iter().map(|p| p != "agl").collect();
        let keep: Vec<bool> = movable
            .iter()
            .zip(&glycan.elements)
            .map(|(m, e)| *m && e != "H")
            .collect();
        let is_c: Vec<bool> = glycan.elements.iter().map(|e| e == "C").collect();
        observation.prepare(&ObservedAtoms {
            elements: glycan.elements.clone(),
            scored: keep.clone(),
        })?;
        let templates: Vec<Vec<V3>> = glycan
            .templates
            .iter()
            .map(|t| t.iter().map(|p| p.map(|v| v as f64)).collect())
            .collect();
        // site residue atoms: explicit pairs; everything else: clash grids
        let (site_atoms, grid_atoms): (Vec<&EnvAtom>, Vec<&EnvAtom>) =
            site.environment.iter().partition(|a| site.is_site_atom(a));
        let grid_atoms: Vec<EnvAtom> = grid_atoms.into_iter().cloned().collect();
        let grids = ClashGrids::build(&grid_atoms, grid_box.0, grid_box.1, grid_box.2);
        let site_xyz: Vec<V3> = site_atoms.iter().map(|a| a.position).collect();
        let site_names: Vec<String> = site_atoms.iter().map(|a| a.atom_name.clone()).collect();
        let site_anchor = crate::anchor::anchor(&site.residue_name);
        let link_bonds = |name: &str| -> Option<u32> { site_anchor.and_then(|a| a.bonds_to(name)) };
        let topo = &glycan.topology.topo_dist;
        // a free link torsion (O- and S-links) has nothing else against eclipsing it: its 1-4
        // pairs across the link (O5/C2 with the side chain's B atom) take the glycan's 1-4 floors
        let free_link =
            site_anchor.is_some_and(|a| matches!(a.torsion, crate::anchor::LinkTorsion::Free));
        let site_carbon: Vec<bool> = site_atoms.iter().map(|a| a.atomic_number == 6.0).collect();
        let mut site_pairs = Vec::new();
        for (j, name) in site_names.iter().enumerate() {
            for i in (0..n).filter(|&i| keep[i]) {
                let bonds = link_bonds(name).map(|b| b + topo[c1 * n + i] as u32);
                if free_link && bonds == Some(3) {
                    let floor = match (is_c[i], site_carbon[j]) {
                        (true, true) => FIT_ONE_FOUR_CC_FLOOR,
                        (false, false) => FIT_ONE_FOUR_POLAR_FLOOR,
                        _ => FIT_ONE_FOUR_FLOOR,
                    };
                    site_pairs.push((i, j, floor));
                    continue;
                }
                // the floors of the clash grids (`ClashGrids::build`), by both elements
                let far = bonds.is_none_or(|b| b > 3);
                if far {
                    let floor = match (is_c[i], site_carbon[j]) {
                        (true, true) => CC_FLOOR as f64,
                        (false, false) => FIT_POLAR_FLOOR,
                        _ => C_POLAR_FLOOR as f64,
                    };
                    site_pairs.push((i, j, floor));
                }
            }
        }
        // 1-4 pairs across rotatable bonds: neighbours of the two central atoms of every torsion
        let mut across = vec![false; n * n];
        for q in &glycan.topology.quads {
            let (b, c) = (q[1], q[2]);
            for i in (0..n).filter(|&i| i != c && topo[b * n + i] == 1) {
                for j in (0..n).filter(|&j| j != b && topo[c * n + j] == 1) {
                    across[i * n + j] = true;
                    across[j * n + i] = true;
                }
            }
        }
        let mut self_pairs = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                if !(keep[i] && keep[j]) {
                    continue;
                }
                let t = topo[i * n + j];
                if t >= 4 {
                    let floor = match (is_c[i], is_c[j]) {
                        (true, true) => CC_FLOOR as f64,
                        (false, false) => FIT_POLAR_FLOOR,
                        _ => C_POLAR_FLOOR as f64,
                    };
                    self_pairs.push((i, j, floor as _));
                } else if t == 3 && is_c[i] != is_c[j] {
                    self_pairs.push((i, j, FIT_ONE_FOUR_FLOOR as _));
                } else if t == 3 && across[i * n + j] {
                    let floor = if is_c[i] {
                        FIT_ONE_FOUR_CC_FLOOR
                    } else {
                        FIT_ONE_FOUR_POLAR_FLOOR
                    };
                    self_pairs.push((i, j, floor as _));
                }
            }
        }
        let (bond, angle) = linkage(&site.residue_name);
        let tokens = glycan.tokens(vocab);
        let geometry = options
            .ideal_geometry
            .then(|| crate::ideal::Geometry::new(&glycan, &site.residue_name, bond));
        let ideal = crate::ideal::Targets::new(&glycan, geometry.as_ref());
        let mut problem = Self {
            sequence: site.sequence.sequence.clone(),
            tokens,
            templates,
            n_atoms: n,
            n_torsions: nt,
            o1,
            c1,
            o5,
            keep,
            movable,
            generated_rings,
            ideal,
            is_c,
            anchor: site.anchor,
            residue_name: site.residue_name.clone(),
            bond,
            angle: angle.to_radians(),
            observation,
            grids,
            site_xyz,
            site_names,
            site_pairs,
            self_pairs,
            w_env: options.w_env,
            w_self: options.w_self,
            w_prior: options.w_prior,
            amide_kappa: 1.0 / options.amide_sd_deg.to_radians().powi(2),
            link_torsion: site_anchor
                .map_or(crate::anchor::LinkTorsion::Planar { centre: 180.0 }, |a| {
                    a.torsion
                }),
            prior: None,
            radius,
            counter: Counter::default(),
            batch: network_batch(options, n),
            score_only: None,
            full_psi_grid: false,
            observer: crate::observer::Observer::default(),
            glycan,
        };
        if options.ideal_geometry {
            crate::cartesian::regularise_templates(&mut problem);
        }
        if options.null_calibration {
            let inflation = null_inflation(&problem);
            problem.observation.set_noise_inflation(inflation);
        }
        Ok(problem)
    }

    /// dE/dtau from the atom gradients `gx` at placed coordinates `x`. The turn about O1-C1
    /// (HO1-O1-C1-O5 of an all-atom build) cancels in the attachment (see `new`), so its
    /// derivative is zero, not the torque of the glycan about the link bond (that is dE/dphi).
    pub fn torsion_gradient(&self, x: &[V3], gx: &[V3]) -> Vec<f64> {
        let topo = &self.glycan.topology;
        let mut g = torsion_gradient(x, &topo.quads, &topo.distal, gx);
        if let Some(k) = self.link_turn() {
            g[k] = 0.0;
        }
        g
    }

    /// Index of the torsion that turns about O1-C1 (HO1-O1-C1-O5 of an all-atom build), if any:
    /// the model's own aglycone torsion, which the attachment angle phi replaces.
    pub fn link_turn(&self) -> Option<usize> {
        self.glycan.topology.quads.iter().position(|q| {
            (q[1] == self.o1 && q[2] == self.c1) || (q[1] == self.c1 && q[2] == self.o1)
        })
    }

    /// Replace the pucker templates (e.g. by the reference's, for parity tests).
    pub fn set_templates(&mut self, templates: Vec<Vec<V3>>) -> Result<()> {
        if templates.is_empty() || templates.iter().any(|t| t.len() != self.n_atoms) {
            return Err(invalid("templates do not match the glycan"));
        }
        self.glycan.templates = templates
            .iter()
            .map(|t| t.iter().map(|p| p.map(|v| v as f32)).collect())
            .collect();
        self.templates = templates;
        Ok(())
    }

    pub fn n_templates(&self) -> usize {
        self.templates.len()
    }

    /// [T]: torsions defined by heavy atoms alone. The others turn a polar hydrogen of an all-atom
    /// model; a map does not see them and a deposit does not have them.
    pub fn heavy_torsions(&self) -> Vec<bool> {
        let el = &self.glycan.elements;
        self.glycan
            .topology
            .quads
            .iter()
            .map(|q| q.iter().all(|&i| el[i] != "H"))
            .collect()
    }

    /// Glycan in the template frame with torsions `tau` (`geometry.set_torsions`).
    pub fn local(&self, tau: &[f64], template: usize) -> Vec<V3> {
        let mut x = self.templates[template].clone();
        set_torsions(
            &mut x,
            &self.glycan.topology.quads,
            &self.glycan.topology.distal,
            tau,
        );
        x
    }

    /// Attach a glycan (any frame) to the site with psi_N = A-B-link-C1 and phi_N = B-link-C1-O5
    /// (`attach.attach`): C1 at the linkage bond length and angle, O5 with the template's
    /// O1-C1-O5 angle and C1-O5 bond, rigid frame alignment.
    pub fn attach(&self, x: &[V3], psi: f64, phi: f64) -> Vec<V3> {
        let [a, b, l] = self.anchor;
        let c1_t = nerf(a, b, l, self.bond, self.angle, psi);
        let (xc1, xo1, xo5) = (x[self.c1], x[self.o1], x[self.o5]);
        let (u, w) = (sub(xo1, xc1), sub(xo5, xc1));
        let ang = (dot(u, w) / (norm(u) * norm(w))).clamp(-1.0, 1.0).acos();
        let o5_t = nerf(b, l, c1_t, norm(w), ang, phi);
        let src = frame(xc1, xo1, xo5);
        let dst = frame(c1_t, l, o5_t);
        // rot = dst src^T
        let mut rot = [[0.0; 3]; 3];
        for (i, row) in rot.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| dst[i][k] * src[j][k]).sum();
            }
        }
        x.iter()
            .map(|p| {
                let d = sub(*p, xc1);
                [0, 1, 2].map(|i| rot[i][0] * d[0] + rot[i][1] * d[1] + rot[i][2] * d[2] + c1_t[i])
            })
            .collect()
    }

    pub fn place(&self, pose: &Pose) -> Vec<V3> {
        self.attach(&self.local(&pose.tau, pose.template), pose.psi, pose.phi)
    }

    /// (psi_N, phi_N) of a placed glycan.
    pub fn attachment_angles(&self, x: &[V3]) -> (f64, f64) {
        let [a, b, l] = self.anchor;
        (
            dihedral(a, b, l, x[self.c1]),
            dihedral(b, l, x[self.c1], x[self.o5]),
        )
    }

    /// Terms that depend on the placed coordinates (no prior); `gx` receives d(total)/dx and
    /// the return carries dE_attach/dpsi.
    pub fn placed_terms(&self, x: &[V3], psi: f64, gx: Option<&mut [V3]>) -> (Terms, f64) {
        self.placed_terms_weighted(x, psi, gx, 1.0, None)
    }

    /// `placed_terms` with the gradient of the observation and of the contacts multiplied by
    /// `data_weight` (the terms are those of weight 1), on the atoms of `weighted` when given.
    /// The contacts are walls, not knowledge to be weighed against the map, so they keep their
    /// strength against it.
    pub fn placed_terms_weighted(
        &self,
        x: &[V3],
        psi: f64,
        mut gx: Option<&mut [V3]>,
        data_weight: f64,
        weighted: Option<&[bool]>,
    ) -> (Terms, f64) {
        let obs = match gx.as_deref_mut() {
            Some(g) => self
                .observation
                .evaluate(x, self.score_only.as_deref(), Some(g)),
            None => self
                .observation
                .evaluate(x, self.score_only.as_deref(), None),
        };
        let out = self.terms_with(obs, x, psi, gx.as_deref_mut());
        if let Some(g) = gx.filter(|_| data_weight != 1.0) {
            g.iter_mut()
                .enumerate()
                .filter(|(i, _)| weighted.is_none_or(|w| w[*i]))
                .flat_map(|(_, v)| v)
                .for_each(|v| *v *= data_weight);
        }
        out
    }

    /// `placed_terms` without gradient, given the observation's `rigid_term` of this conformer:
    /// the same terms up to rounding, for searches over rigid placements of one conformer.
    pub fn placed_terms_rigid(&self, x: &[V3], psi: f64, rigid: f64) -> Terms {
        let obs = self
            .observation
            .evaluate_rigid(x, self.score_only.as_deref(), rigid);
        self.terms_with(obs, x, psi, None).0
    }

    fn terms_with(
        &self,
        obs: crate::observation::ObservationValue,
        x: &[V3],
        psi: f64,
        gx: Option<&mut [V3]>,
    ) -> (Terms, f64) {
        let (e_env, e_self) = self.clash_terms(x, gx);
        let (e_att, d_att) = self.link_torsion.energy(psi, self.amide_kappa);
        let total = obs.energy + self.w_env * e_env + self.w_self * e_self + e_att;
        (
            Terms {
                total,
                loglik: obs.log_likelihood,
                partial_cc: obs.partial_correlation,
                gain: obs.gain,
                scale: obs.scale,
                e_env,
                e_self,
                e_att,
                e_prior: 0.0,
            },
            d_att,
        )
    }

    /// (E_env, E_self) of a placed glycan; `gx` receives d(w_env E_env + w_self E_self)/dx.
    pub fn clash_terms(&self, x: &[V3], mut gx: Option<&mut [V3]>) -> (f64, f64) {
        let scored = |i: usize| self.score_only.as_ref().is_none_or(|m| m[i]);
        let mut e_env = 0.0;
        for i in (0..self.n_atoms).filter(|&i| self.keep[i] && scored(i)) {
            let (v, g) = self.grids.sample(self.is_c[i], x[i]);
            e_env += v;
            if let Some(gx) = gx.as_deref_mut() {
                for k in 0..3 {
                    gx[i][k] += self.w_env * g[k];
                }
            }
        }
        for &(i, j, floor) in self.site_pairs.iter().filter(|p| scored(p.0)) {
            let d = sub(x[i], self.site_xyz[j]);
            let r = norm(d);
            if r < floor {
                e_env += (floor - r).powi(2);
                if let Some(gx) = gx.as_deref_mut() {
                    let s = -2.0 * (floor - r) / r.max(1e-12) * self.w_env;
                    for k in 0..3 {
                        gx[i][k] += s * d[k];
                    }
                }
            }
        }
        let (e_self, g_self) = match &self.score_only {
            None => clash_energy_grad(x, &self.self_pairs),
            Some(_) => {
                let pairs: Vec<_> = self
                    .self_pairs
                    .iter()
                    .copied()
                    .filter(|p| scored(p.0) && scored(p.1))
                    .collect();
                clash_energy_grad(x, &pairs)
            }
        };
        if let Some(gx) = gx {
            for (g, s) in gx.iter_mut().zip(&g_self) {
                for k in 0..3 {
                    g[k] += self.w_self * s[k];
                }
            }
        }
        (e_env, e_self)
    }

    /// The deepest single contact of a placed glycan: the largest squared floor violation of one
    /// atom against the environment or of one atom pair (the terms `clash_terms` sums), each on
    /// the scale of the 1-4 carbon-polar pairs, whose floor is the closest to the validator's
    /// limit (0.08 A above it; the other floors sit at least [`FAR_FLOOR_MARGIN`] above theirs).
    pub fn worst_contact(&self, x: &[V3]) -> f64 {
        let scored = |i: usize| self.score_only.as_ref().is_none_or(|m| m[i]);
        let far = (ONE_FOUR_FLOOR_MARGIN / FAR_FLOOR_MARGIN).powi(2);
        let scale = |floor: f64| {
            let tight = (floor - FIT_ONE_FOUR_FLOOR).abs() < 1e-6
                || (floor - FIT_ONE_FOUR_CC_FLOOR).abs() < 1e-6;
            if tight { 1.0 } else { far }
        };
        let mut worst = 0f64;
        for i in (0..self.n_atoms).filter(|&i| self.keep[i] && scored(i)) {
            worst = worst.max(far * self.grids.sample(self.is_c[i], x[i]).0);
        }
        for &(i, j, floor) in self.site_pairs.iter().filter(|p| scored(p.0)) {
            let r = norm(sub(x[i], self.site_xyz[j]));
            worst = worst.max(scale(floor) * (floor - r).max(0.0).powi(2));
        }
        for &(i, j, floor) in self
            .self_pairs
            .iter()
            .filter(|p| scored(p.0) && scored(p.1))
        {
            let (r, floor) = (norm(sub(x[i], x[j])), floor as f64);
            worst = worst.max(scale(floor) * (floor - r).max(0.0).powi(2));
        }
        worst
    }

    /// Prior energy (unweighted) of torsions, with its gradient added (weighted) to `g`.
    pub fn prior_energy(&self, tau: &[f64], g: Option<&mut [f64]>) -> f64 {
        let Some(prior) = &self.prior else {
            return 0.0;
        };
        match g {
            Some(g) => {
                let mut gp = vec![0.0; tau.len()];
                let e = prior.energy(tau, Some(&mut gp));
                for (a, b) in g.iter_mut().zip(&gp) {
                    *a += self.w_prior * b;
                }
                e
            }
            None => prior.energy(tau, None),
        }
    }

    /// Full objective of a placed pose whose torsions are `tau`.
    pub fn terms_of(&self, x: &[V3], psi: f64, tau: &[f64]) -> Terms {
        let (mut terms, _) = self.placed_terms(x, psi, None);
        terms.e_prior = self.prior_energy(tau, None);
        terms.total += self.w_prior * terms.e_prior;
        terms
    }

    /// Evaluate a pose; with `want_grad` also dE/dtau, dE/dpsi_N, dE/dphi_N.
    pub fn evaluate(&self, pose: &Pose, want_grad: bool) -> Evaluation {
        self.evaluate_placed(self.place(pose), pose, want_grad)
    }

    /// Evaluate a pose whose placed coordinates `x` are already known (`x = place(pose)`).
    pub fn evaluate_placed(&self, x: Vec<V3>, pose: &Pose, want_grad: bool) -> Evaluation {
        if !want_grad {
            let terms = self.terms_of(&x, pose.psi, &pose.tau);
            return Evaluation {
                x,
                terms,
                grad: None,
            };
        }
        let mut gx = vec![[0.0; 3]; self.n_atoms];
        let (mut terms, d_att) = self.placed_terms(&x, pose.psi, Some(&mut gx));
        let mut g_tau = self.torsion_gradient(&x, &gx);
        terms.e_prior = self.prior_energy(&pose.tau, Some(&mut g_tau));
        terms.total += self.w_prior * terms.e_prior;
        let [_, b, l] = self.anchor;
        let rigid = |axis: V3, pivot: V3| -> f64 {
            let u = unit(axis);
            let mut s = [0.0; 3];
            for (p, g) in x.iter().zip(&gx) {
                let c = cross(sub(*p, pivot), *g);
                for k in 0..3 {
                    s[k] += c[k];
                }
            }
            dot(u, s)
        };
        let psi = rigid(sub(l, b), l) + d_att;
        let phi = rigid(sub(x[self.c1], l), x[self.c1]);
        Evaluation {
            x,
            terms,
            grad: Some(PoseGradient {
                tau: g_tau,
                psi,
                phi,
            }),
        }
    }
}

/// Decoys of the empirical null (`null_inflation`).
pub const NULL_DECOYS: usize = 2048;
pub const NULL_MIN_DECOYS: usize = 100;
/// Minimum decoy distance from the site residue's linking atoms (A).
pub const NULL_CLEARANCE: f64 = 2.8;

/// Deterministic decoy design (`density.decoy_poses`): unit direction (Fibonacci sphere),
/// cube-root radial fraction and Shoemake rotation quaternion (x, y, z, w) from low-discrepancy
/// sequences, identical in the Python reference.
pub fn decoy_poses(n: usize) -> Vec<(V3, f64, [f64; 4])> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let tau = 2.0 * std::f64::consts::PI;
    (0..n)
        .map(|k| {
            let kk = k as f64 + 0.5;
            let y = 1.0 - 2.0 * kk / n as f64;
            let rho = (1.0 - y * y).max(0.0).sqrt();
            let phi = k as f64 * golden;
            let d = [rho * phi.cos(), y, rho * phi.sin()];
            let u = (kk * 0.6180339887498949) % 1.0;
            let a1 = (kk * 0.8191725133961645) % 1.0;
            let a2 = (kk * 0.6710436067037893) % 1.0;
            let a3 = (kk * 0.5497004779019703) % 1.0;
            let q = [
                (1.0 - a1).sqrt() * (tau * a2).sin(),
                (1.0 - a1).sqrt() * (tau * a2).cos(),
                a1.sqrt() * (tau * a3).sin(),
                a1.sqrt() * (tau * a3).cos(),
            ];
            (d, u.cbrt(), q)
        })
        .collect()
}

fn quat_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn lower_median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[(v.len() - 1) / 2]
}

/// Empirical noise inflation of a density observation (`problem.null_inflation`): the root
/// residue (template 0) at `NULL_DECOYS` deterministic poses in the scoring ball, kept where it
/// touches no environment atom (carbon-probe clash grid zero, > `NULL_CLEARANCE` from the site
/// residue); its standardised projection is N(0, 1) under the noise model. Returns
/// max(1, (1.4826 MAD)^2), or 1 with fewer than `NULL_MIN_DECOYS` decoys or no density.
pub fn null_inflation(problem: &SiteProblem) -> f64 {
    let Some(lik) = problem.observation.density() else {
        return 1.0;
    };
    let sel: Vec<usize> = (0..problem.n_atoms)
        .filter(|&i| problem.glycan.res_paths[i] == "r" && problem.keep[i])
        .collect();
    let tpl = &problem.templates[0];
    let n = sel.len() as f64;
    let mean = [0, 1, 2].map(|k| sel.iter().map(|&i| tpl[i][k]).sum::<f64>() / n);
    let res: Vec<V3> = sel
        .iter()
        .map(|&i| [0, 1, 2].map(|k| tpl[i][k] - mean[k]))
        .collect();
    let z: Vec<f64> = sel
        .iter()
        .map(|&i| crate::observation::glycan_z(&problem.glycan.elements[i]).unwrap_or(0.0))
        .collect();
    let link = problem.anchor[2];
    let mut t: Vec<f64> = decoy_poses(NULL_DECOYS)
        .into_iter()
        .filter_map(|(d, u, q)| {
            let r = quat_matrix(q);
            let centre = [0, 1, 2].map(|k| link[k] + d[k] * (problem.radius - 3.0) * u);
            let x: Vec<V3> = res
                .iter()
                .map(|a| [0, 1, 2].map(|i| (0..3).map(|j| r[i][j] * a[j]).sum::<f64>() + centre[i]))
                .collect();
            // float32 positions, as the reference
            let x: Vec<V3> = x.iter().map(|p| p.map(|v| v as f32 as f64)).collect();
            let pen: f64 = x.iter().map(|p| problem.grids.sample(true, *p).0).sum();
            if pen > 1e-9 {
                return None;
            }
            let near_site = x.iter().any(|p| {
                problem
                    .site_xyz
                    .iter()
                    .any(|s| norm(sub(*p, *s)) <= NULL_CLEARANCE)
            });
            if near_site {
                return None;
            }
            Some(lik.projection(&x, &z))
        })
        .collect();
    if t.len() < NULL_MIN_DECOYS {
        return 1.0;
    }
    let med = lower_median(&mut t.clone());
    let mut dev: Vec<f64> = t.iter_mut().map(|v| (*v - med).abs()).collect();
    let mad = lower_median(&mut dev) * 1.4826;
    (mad * mad).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_batch_keeps_a_large_glycan_within_its_memory_cap() {
        let capped = |batch, mb| ProblemOptions {
            batch,
            batch_memory_mb: mb,
            ..ProblemOptions::default()
        };
        // no cap: the batch as given
        assert_eq!(network_batch(&capped(Some(32), None), 170), Some(32));
        assert_eq!(network_batch(&capped(None, None), 170), None);
        // 96 MB: a typical N-glycan keeps 32, a 170-atom one drops to 8, one atom pair never 0
        assert_eq!(network_batch(&capped(Some(32), Some(96.0)), 90), Some(31));
        assert_eq!(network_batch(&capped(Some(32), Some(96.0)), 170), Some(8));
        assert_eq!(network_batch(&capped(Some(32), Some(0.001)), 170), Some(1));
        // the cap alone sets the batch
        assert_eq!(network_batch(&capped(None, Some(96.0)), 170), Some(8));
    }
}
