//! Development WebGPU driver. Only the coordinating task touches GPU handles.
use super::*;
use glysys_energy::scoring::{
    Boundary, EvaluationRequest, Pose, PoseBatch, PreparedScene, ScoreModel,
};
use glysys_energy::{AtomSelection, EnergyComponents, EnergyResult};
use glysys_gpu::{GpuContext, GpuContextOptions};
use glysys_opt::{genetic_state::GeneticState, resumable::LbfgsState};
use glysys_runtime::{ExecutionOptions, ScoringSession};
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct StageReport {
    gpu_evaluations: usize,
    cpu_evaluations: usize,
    validation_evaluations: usize,
    gpu_seconds: f64,
    cpu_seconds: f64,
    fallback_reason: Option<String>,
}
#[derive(Default, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeReport {
    stages: std::collections::BTreeMap<String, StageReport>,
    requested: String,
    actual_backend: String,
    fallback_reason: Option<String>,
    gpu_evaluations: usize,
    cpu_evaluations: usize,
    gpu_seconds: f64,
    cpu_seconds: f64,
    initialization_seconds: f64,
    adapter: Option<String>,
}
pub(super) struct Runtime {
    report: ComputeReport,
    context: Option<GpuContext>,
    scoring: Option<ScoringSession>,
    scoring_key: Option<(bool, bool)>,
    rejected: HashMap<(bool, bool), String>,
    disabled: bool,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            report: ComputeReport {
                requested: "cpu".into(),
                actual_backend: "cpu".into(),
                ..Default::default()
            },
            context: None,
            scoring: None,
            scoring_key: None,
            rejected: HashMap::new(),
            disabled: true,
        }
    }
}
thread_local! { static OBSERVER:RefCell<Option<Box<dyn Fn(&str)>>>=RefCell::new(None); static RUNTIME:RefCell<Runtime>=RefCell::new(Runtime::default()); }
pub fn set_progress(callback: impl Fn(&str) + 'static) {
    OBSERVER.with(|o| *o.borrow_mut() = Some(Box::new(callback)));
}
fn notify(backend: &str) {
    OBSERVER.with(|o| {
        if let Some(f) = &*o.borrow() {
            f(backend);
        }
    });
}
pub fn configure(backend: &str) {
    RUNTIME.with(|r| {
        *r.borrow_mut() = Runtime {
            report: ComputeReport {
                requested: backend.into(),
                actual_backend: "cpu".into(),
                ..Default::default()
            },
            disabled: backend == "cpu",
            ..Default::default()
        }
    });
}
pub fn finish() -> serde_json::Value {
    OBSERVER.with(|o| *o.borrow_mut() = None);
    RUNTIME.with(|r| {
        let mut runtime = std::mem::take(&mut *r.borrow_mut());
        if runtime.report.gpu_evaluations == 0
            && runtime.report.fallback_reason.is_none()
            && runtime.report.requested != "cpu"
        {
            runtime.report.fallback_reason =
                Some("No eligible GPU batch was used; this run completed on CPU.".into());
        }
        let gpu = runtime
            .report
            .stages
            .values()
            .any(|s| s.gpu_evaluations > 0);
        let cpu = runtime
            .report
            .stages
            .values()
            .any(|s| s.cpu_evaluations > 0);
        runtime.report.actual_backend = if gpu && cpu {
            "mixed"
        } else if gpu {
            "webgpu"
        } else {
            "cpu"
        }
        .into();
        serde_json::to_value(runtime.report).unwrap_or_default()
    })
}
pub(super) async fn start_compute_stage_async() {
    RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        r.scoring = None;
        r.scoring_key = None;
        r.rejected.clear();
    });
}
/// Return the coordinator-owned context for this search. Geometry and scoring
/// sessions clone this handle; only the first caller creates the adapter and
/// device for the job.
pub(super) async fn shared_context() -> std::result::Result<GpuContext, String> {
    if let Some(context) = RUNTIME.with(|r| r.borrow().context.clone()) {
        return Ok(context);
    }
    let context = GpuContext::new(GpuContextOptions::default())
        .await
        .map_err(|error| error.to_string())?;
    RUNTIME.with(|r| {
        let mut runtime = r.borrow_mut();
        if runtime.context.is_none() {
            runtime.context = Some(context.clone());
        }
    });
    Ok(context)
}
fn options(c: &EnergySearchContext) -> EnergyOptions {
    EnergyOptions {
        cutoff: Some(c.cutoff),
        obc2: c.use_obc2.then(Obc2Options::default),
        ..Default::default()
    }
}
fn component_array(c: EnergyComponents) -> [f64; 9] {
    [
        c.bonds,
        c.angles,
        c.proper_torsions,
        c.improper_torsions,
        c.van_der_waals,
        c.electrostatics,
        c.generalized_born,
        c.surface_area,
        c.restraints,
    ]
}
fn close(a: f64, b: f64, relative: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3 + relative * b.abs()
}
fn parity(a: &EnergyResult, b: &EnergyResult) -> bool {
    component_array(a.components)
        .into_iter()
        .zip(component_array(b.components))
        .all(|(a, b)| close(a, b, 1e-4))
        && match (&a.gradients, &b.gradients) {
            (Some(a), Some(b)) => a.iter().zip(b).all(|(a, b)| {
                [a.x, a.y, a.z]
                    .into_iter()
                    .zip([b.x, b.y, b.z])
                    .all(|(a, b)| close(a, b, 1e-3))
            }),
            (None, None) => true,
            _ => false,
        }
}
fn cpu(
    c: &EnergySearchContext,
    coordinates: &[Vec3],
    active: &[usize],
    gradient: bool,
    interaction: bool,
) -> Result<EnergyResult> {
    if gradient {
        let mut evaluator = EnergyEvaluator::new(&c.system, options(c))?;
        evaluator = evaluator.with_active_terms(AtomSelection::from_indices(
            c.system.atom_count(),
            active.iter().copied(),
        ))?;
        if interaction {
            let (protein, glycan, _) = c.masks();
            let e = evaluator.interaction_energy(coordinates, protein, glycan)?;
            Ok(EnergyResult {
                components: EnergyComponents {
                    van_der_waals: e.van_der_waals,
                    electrostatics: e.electrostatics,
                    ..Default::default()
                },
                gradients: None,
            })
        } else {
            Ok(evaluator.energy_and_gradient(coordinates)?)
        }
    } else {
        let evaluator = c.evaluator()?;
        if interaction {
            let (protein, glycan, _) = c.masks();
            let e = evaluator.interaction_energy(coordinates, protein, glycan)?;
            Ok(EnergyResult {
                components: EnergyComponents {
                    van_der_waals: e.van_der_waals,
                    electrostatics: e.electrostatics,
                    ..Default::default()
                },
                gradients: None,
            })
        } else {
            Ok(evaluator.energy(coordinates)?)
        }
    }
}
impl Runtime {
    pub(super) fn for_job() -> Self {
        let requested = requested_backend();
        Self {
            disabled: requested == "cpu",
            report: ComputeReport {
                requested,
                actual_backend: "cpu".into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }
    pub(super) fn scores_enabled(&self, interaction: bool) -> bool {
        !self.disabled && !self.rejected.contains_key(&(false, interaction))
    }
    fn reject_plan(&mut self, gradient: bool, interaction: bool, reason: String) {
        self.rejected
            .insert((gradient, interaction), reason.clone());
        self.report.fallback_reason = Some(reason);
        notify("cpu");
    }
    pub(super) fn publish(&mut self) {
        let report = std::mem::take(&mut self.report);
        RUNTIME.with(|r| {
            let mut r = r.borrow_mut();
            for (name, stage) in report.stages {
                merge_stage(r.report.stages.entry(name).or_default(), stage);
            }
            r.report.gpu_evaluations += report.gpu_evaluations;
            r.report.cpu_evaluations += report.cpu_evaluations;
            r.report.gpu_seconds += report.gpu_seconds;
            r.report.cpu_seconds += report.cpu_seconds;
            r.report.initialization_seconds += report.initialization_seconds;
            if report.adapter.is_some() {
                r.report.adapter = report.adapter;
            }
            if report.fallback_reason.is_some() {
                r.report.fallback_reason = report.fallback_reason;
            }
            let has_gpu = r.report.gpu_evaluations > 0;
            let has_cpu = r.report.cpu_evaluations > 0;
            r.report.actual_backend = if has_gpu && has_cpu {
                "mixed"
            } else if has_gpu {
                "webgpu"
            } else {
                "cpu"
            }
            .into();
        });
    }
    fn fallback(&mut self, reason: String) {
        notify("cpu");
        self.report.actual_backend = if self.report.gpu_evaluations > 0 {
            "mixed".into()
        } else {
            "cpu".into()
        };
        self.disabled = true;
        self.scoring = None;
        self.scoring_key = None;
        self.report.fallback_reason = Some(reason);
    }
    fn cpu_batch(
        &mut self,
        c: &EnergySearchContext,
        points: &[Vec<Vec3>],
        active: &[Vec<usize>],
        gradient: bool,
        interaction: bool,
    ) -> Result<Vec<EnergyResult>> {
        let start = Instant::now();
        let result = points
            .par_iter()
            .zip(active)
            .map(|(p, a)| cpu(c, p, a, gradient, interaction))
            .collect();
        self.report.cpu_seconds += start.elapsed().as_secs_f64();
        self.report.cpu_evaluations += points.len();
        result
    }
    pub(super) async fn evaluate(
        &mut self,
        c: &EnergySearchContext,
        points: &[Vec<Vec3>],
        active: &[Vec<usize>],
        gradient: bool,
        interaction: bool,
    ) -> Result<Vec<EnergyResult>> {
        let before_gpu = self.report.gpu_evaluations;
        let before_cpu = self.report.cpu_evaluations;
        let before_gs = self.report.gpu_seconds;
        let before_cs = self.report.cpu_seconds;
        let result = self
            .evaluate_inner(c, points, active, gradient, interaction)
            .await;
        let stage = self
            .report
            .stages
            .entry(format!(
                "{}_{}",
                if interaction { "interaction" } else { "energy" },
                if gradient { "gradient" } else { "score" }
            ))
            .or_default();
        stage.gpu_evaluations += self.report.gpu_evaluations - before_gpu;
        let count = self.report.cpu_evaluations - before_cpu;
        // CPU work can be a normal fallback even when an earlier part of the
        // same batch reached the GPU.  Count execution by the actual deltas;
        // validation is recorded separately by the explicit parity harness.
        stage.cpu_evaluations += count;
        stage.gpu_seconds += self.report.gpu_seconds - before_gs;
        stage.cpu_seconds += self.report.cpu_seconds - before_cs;
        if self.disabled || self.rejected.contains_key(&(gradient, interaction)) {
            stage.fallback_reason = self.report.fallback_reason.clone();
        }
        result
    }
    async fn evaluate_inner(
        &mut self,
        c: &EnergySearchContext,
        points: &[Vec<Vec3>],
        active: &[Vec<usize>],
        gradient: bool,
        interaction: bool,
    ) -> Result<Vec<EnergyResult>> {
        if self.disabled || self.rejected.contains_key(&(gradient, interaction)) {
            return self.cpu_batch(c, points, active, gradient, interaction);
        }
        let key = (false, interaction);
        if self.scoring.is_none() || self.scoring_key != Some(key) {
            let start = Instant::now();
            if self.context.is_none() {
                match shared_context().await {
                    Ok(context) => self.context = Some(context),
                    Err(error) => self.fallback(error),
                }
            }
            if !self.disabled {
                let scene = PreparedScene::new(
                    Arc::new(c.system.clone()),
                    options(c),
                    Boundary::NonPeriodic,
                )
                .map_err(|error| EnsembleError::Metadata(error.to_string()))?;
                let model = if interaction {
                    ScoreModel::interaction("receptor", "glycan")
                } else {
                    ScoreModel::amber()
                };
                let mut execution = ExecutionOptions::default();
                execution.backend = match self.report.requested.as_str() {
                    "webgpu" => glysys_runtime::BackendPreference::Gpu,
                    "cpu" => glysys_runtime::BackendPreference::Cpu,
                    _ => glysys_runtime::BackendPreference::Auto,
                };
                let session =
                    ScoringSession::new_with_context(scene, model, execution, self.context.clone())
                        .await;
                match session {
                    Ok(session) => {
                        self.report.adapter = session
                            .diagnostics()
                            .adapter_identity
                            .clone()
                            .or_else(|| Some("CPU reference".into()));
                        if session
                            .diagnostics()
                            .actual_backend
                            .eq_ignore_ascii_case("CPU")
                        {
                            let reason = session
                                .diagnostics()
                                .fallback_reason
                                .clone()
                                .unwrap_or_else(|| "GPU scoring is unavailable".into());
                            self.fallback(reason);
                        } else {
                            self.scoring = Some(session);
                            self.scoring_key = Some(key);
                        }
                    }
                    Err(error) if self.report.requested == "auto" => {
                        self.fallback(error.to_string());
                    }
                    Err(error) => {
                        return Err(EnsembleError::Metadata(error.to_string()));
                    }
                }
            }
            self.report.initialization_seconds += start.elapsed().as_secs_f64();
            if self.disabled {
                return self.cpu_batch(c, points, active, gradient, interaction);
            }
        }
        let start = Instant::now();
        let poses = PoseBatch {
            poses: points
                .iter()
                .enumerate()
                .map(|(id, coordinates)| Pose::cartesian(id as u64, coordinates.clone()))
                .collect(),
        };
        let request = EvaluationRequest {
            gradients: gradient,
            ..Default::default()
        };
        let result = match self.scoring.as_mut() {
            Some(scoring) => match scoring.evaluate(&poses, &request).await {
                Ok(result) => result,
                Err(error) if self.report.requested == "auto" => {
                    self.fallback(error.to_string());
                    return self.cpu_batch(c, points, active, gradient, interaction);
                }
                Err(error) => return Err(EnsembleError::Metadata(error.to_string())),
            },
            None => {
                let reason = "GPU scoring session disappeared before evaluation";
                if self.report.requested == "auto" {
                    self.fallback(reason.into());
                    return self.cpu_batch(c, points, active, gradient, interaction);
                }
                return Err(EnsembleError::Metadata(reason.into()));
            }
        };
        let (actual_gpu, fallback_reason) = {
            let Some(session) = self.scoring.as_ref() else {
                let reason = "GPU scoring session disappeared after evaluation";
                if self.report.requested == "auto" {
                    self.fallback(reason.into());
                    return self.cpu_batch(c, points, active, gradient, interaction);
                }
                return Err(EnsembleError::Metadata(reason.into()));
            };
            (
                session
                    .diagnostics()
                    .actual_backend
                    .eq_ignore_ascii_case("GPU"),
                session.diagnostics().fallback_reason.clone(),
            )
        };
        if actual_gpu {
            notify("webgpu");
            self.report.gpu_seconds += start.elapsed().as_secs_f64();
            self.report.gpu_evaluations += points.len();
            self.report.actual_backend = "webgpu".into();
        } else {
            self.report.cpu_seconds += start.elapsed().as_secs_f64();
            self.report.cpu_evaluations += points.len();
            self.fallback(
                fallback_reason
                    .unwrap_or_else(|| "GPU scoring fell back to the CPU reference".into()),
            );
        }
        Ok(result
            .into_iter()
            .map(|value| {
                let term = |name: &str| value.terms.get(name).map_or(0., |v| v.raw);
                EnergyResult {
                    components: EnergyComponents {
                        bonds: term("bonds"),
                        angles: term("angles"),
                        proper_torsions: term("proper_torsions"),
                        improper_torsions: term("improper_torsions"),
                        van_der_waals: term("van_der_waals"),
                        electrostatics: term("electrostatics"),
                        generalized_born: term("generalized_born"),
                        surface_area: term("surface_area"),
                        restraints: term("restraints"),
                        dispersion_correction: 0.,
                    },
                    gradients: value.gradients,
                }
            })
            .collect())
    }
}

async fn evaluate_candidates(
    c: &EnergySearchContext,
    structures: &[Structure],
    verify_scores: bool,
) -> Result<Vec<CandidateEnergy>> {
    let mut runtime = RUNTIME.with(|r| std::mem::take(&mut *r.borrow_mut()));
    let result = evaluate_candidates_owned(&mut runtime, c, structures, verify_scores).await;
    RUNTIME.with(|r| *r.borrow_mut() = runtime);
    result
}
async fn evaluate_candidates_owned(
    runtime: &mut Runtime,
    c: &EnergySearchContext,
    structures: &[Structure],
    verify_scores: bool,
) -> Result<Vec<CandidateEnergy>> {
    let mut coordinates = structures
        .iter()
        .map(|s| c.coordinates_for(s))
        .collect::<Result<Vec<_>>>()?;
    let selections = if c.minimize {
        let (_, glycan, residues) = c.masks();
        coordinates
            .iter()
            .map(|p| c.active_selection(p, glycan, residues))
            .collect::<Vec<_>>()
    } else {
        // Active-region discovery is only needed by minimization.  Energy and
        // interaction ranking evaluate complete prepared coordinates, so do
        // not scan every protein residue or allocate selection vectors in the
        // ordinary score-only loop.
        (0..coordinates.len())
            .map(|_| (Vec::new(), Vec::new()))
            .collect::<Vec<_>>()
    };
    let active = selections.iter().map(|s| s.0.clone()).collect::<Vec<_>>();
    if c.minimize && !runtime.disabled {
        let config = LbfgsConfig {
            max_iterations: c.min_iterations.max(1),
            ..Default::default()
        };
        let mut states = coordinates
            .iter()
            .zip(&active)
            .map(|(p, a)| {
                LbfgsState::new(
                    &a.iter()
                        .flat_map(|&i| [p[i].x, p[i].y, p[i].z])
                        .collect::<Vec<_>>(),
                    &config,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        loop {
            let mut indices = Vec::new();
            let mut points = Vec::new();
            let mut masks = Vec::new();
            for (i, state) in states.iter_mut().enumerate() {
                if let Some(p) = state.request() {
                    indices.push(i);
                    let mut candidate = coordinates[i].clone();
                    for (&atom, v) in active[i].iter().zip(p.chunks_exact(3)) {
                        candidate[atom] = Vec3 {
                            x: v[0],
                            y: v[1],
                            z: v[2],
                        };
                    }
                    points.push(candidate);
                    masks.push(active[i].clone());
                }
            }
            if indices.is_empty() {
                break;
            }
            let values = runtime.evaluate(c, &points, &masks, true, false).await?;
            if values.len() != indices.len() {
                return Err(EnsembleError::Metadata(format!(
                    "energy evaluator returned {} results for {} minimization trials",
                    values.len(),
                    indices.len()
                )));
            }
            for (slot, mut value) in values.into_iter().enumerate() {
                let index = indices[slot];
                // Convergence is decided from f64 CPU gradients, never solely f32.
                let gradients = value.gradients.as_ref().ok_or_else(|| {
                    EnsembleError::Metadata(
                        "energy evaluator omitted gradients for a minimization trial".into(),
                    )
                })?;
                if gradients
                    .iter()
                    .flat_map(|g| [g.x, g.y, g.z])
                    .all(|g| g.abs() <= config.gradient_tolerance)
                {
                    value = cpu(c, &points[slot], &masks[slot], true, false)?;
                }
                let gradient_values = value.gradients.as_ref().ok_or_else(|| {
                    EnsembleError::Metadata(
                        "energy evaluator omitted gradients for a minimization trial".into(),
                    )
                })?;
                let gradient = active[index]
                    .iter()
                    .map(|&i| {
                        gradient_values.get(i).ok_or_else(|| {
                            EnsembleError::Metadata(format!(
                                "energy evaluator returned no gradient for active atom {i}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flat_map(|g| [g.x, g.y, g.z])
                    .collect();
                states[index].submit(value.total(), gradient)?;
            }
        }
        for ((p, state), mask) in coordinates.iter_mut().zip(states).zip(&active) {
            let Some(outcome) = state.outcome() else {
                return Err(EnsembleError::Metadata(
                    "minimizer ended without an outcome".into(),
                ));
            };
            for (&i, v) in mask.iter().zip(outcome.point.chunks_exact(3)) {
                p[i] = Vec3 {
                    x: v[0],
                    y: v[1],
                    z: v[2],
                };
            }
        }
        let final_gpu = runtime
            .evaluate(c, &coordinates, &active, true, false)
            .await?;
        let final_cpu = runtime.cpu_batch(c, &coordinates, &active, true, false)?;
        if !final_gpu.iter().zip(&final_cpu).all(|(a, b)| parity(a, b)) {
            runtime.reject_plan(
                true,
                false,
                "Final minimizer gradient validation failed; reevaluating candidates on CPU."
                    .into(),
            );
            return structures.iter().map(|s| c.evaluate(s)).collect();
        }
        c.minimizations
            .fetch_add(structures.len(), Ordering::Relaxed);
    } else if c.minimize {
        // CPU override preserves the existing minimizer's selection and diagnostics.
        return structures.iter().map(|s| c.evaluate(s)).collect();
    }
    // Sampling decisions and minimized candidates use f64 energies. GA score
    // batches use qualified GPU energies, with ambiguous rankings checked below.
    let interaction = c.mode == SearchScoringMode::ProteinGlycanInteraction;
    let values = if c.minimize || verify_scores {
        runtime.cpu_batch(c, &coordinates, &active, false, interaction)?
    } else {
        runtime
            .evaluate(c, &coordinates, &active, false, interaction)
            .await?
    };
    c.evaluations.fetch_add(structures.len(), Ordering::Relaxed);
    Ok(coordinates
        .into_iter()
        .zip(values)
        .zip(selections)
        .map(
            |((coordinates, e), (active, active_residues))| CandidateEnergy {
                score: e.total(),
                components: e.components,
                interaction: interaction.then_some(glysys_energy::InteractionEnergyComponents {
                    van_der_waals: e.components.van_der_waals,
                    electrostatics: e.components.electrostatics,
                }),
                coordinates,
                active_atoms: active.len(),
                active_indices: active,
                active_residues,
                neighbor_pairs: 0,
            },
        )
        .collect())
}
pub(super) async fn gpu_evaluate_candidate(
    c: &EnergySearchContext,
    s: &Structure,
) -> Result<CandidateEnergy> {
    // A normal GPU search must not turn every candidate into a synchronous
    // CPU reference evaluation. Validation belongs to the explicit harness;
    // production scoring stays on the qualified GPU plan.
    let mut result = evaluate_candidates(c, std::slice::from_ref(s), false)
        .await?
        .remove(0);
    if c.mode == SearchScoringMode::ProteinGlycanInteraction {
        let (protein, glycan, _) = c.masks();
        result.neighbor_pairs = c
            .evaluator()?
            .interaction_energy_with_pair_count(&result.coordinates, protein, glycan)?
            .1;
    }
    Ok(result)
}
pub(super) async fn gpu_optimize<F, C>(
    problem: &SearchProblem<'_>,
    config: &GeneticAlgorithmConfig,
    mut progress: F,
    mut cancelled: C,
) -> glysys_opt::Result<GeneticAlgorithmOutcome<Vec<Gene>>>
where
    F: FnMut(&GenerationRecord),
    C: FnMut() -> bool,
{
    let mut checkpoint = GeneticState::new(problem, config)?;
    loop {
        if cancelled() {
            return Err(glysys_opt::OptimizationError::Cancelled);
        }
        let mut scores = vec![0.; checkpoint.population().len()];
        // Bound materialized CPU structures too; the GPU buffer cap alone does
        // not protect browser memory while preparing a large population.
        let batch_size = problem.energy_context.map_or(1, |c| {
            (64 * 1024 * 1024 / (c.system.atom_count().max(1) * 256)).clamp(1, 32)
        });
        for (batch, states) in checkpoint.population().chunks(batch_size).enumerate() {
            if cancelled() {
                return Err(glysys_opt::OptimizationError::Cancelled);
            }
            let mut indices = Vec::new();
            let mut structures = Vec::new();
            for (local, state) in states.iter().enumerate() {
                let i = batch * batch_size + local;
                let eligible = problem.energy_context.is_some()
                    && problem
                        .prepared
                        .evaluate(state, problem.clash_distance)
                        .is_ok_and(|p| p.score <= 1.1);
                if eligible {
                    if let Ok(s) =
                        build_state(problem.protein, problem.sites, state, problem.builder)
                    {
                        indices.push(i);
                        structures.push(s);
                        continue;
                    }
                }
                scores[i] = problem.evaluate(state);
            }
            if !structures.is_empty() {
                let Some(context) = problem.energy_context else {
                    for (i, _) in indices.iter().zip(&structures) {
                        scores[*i] = problem.evaluate(&checkpoint.population()[*i]);
                    }
                    continue;
                };
                match evaluate_candidates(context, &structures, false).await {
                    Ok(values) => {
                        for (i, value) in indices.into_iter().zip(values) {
                            scores[i] = value.score
                                + if problem.scoring_mode == SearchScoringMode::FullEnergy {
                                    0.0
                                } else {
                                    problem.prior(&checkpoint.population()[i]) * 1e-6
                                };
                        }
                    }
                    Err(_) => {
                        for i in indices {
                            scores[i] = problem.evaluate(&checkpoint.population()[i]);
                        }
                    }
                }
            }
        }
        // f32 cannot decide near-equal fitness rankings reliably.
        let mut ordered = (0..scores.len()).collect::<Vec<_>>();
        ordered.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        let mut ambiguous = HashSet::new();
        for pair in ordered.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if scores[a] < 1e12
                && (scores[a] - scores[b]).abs()
                    <= 2e-3 + 1e-4 * (scores[a].abs() + scores[b].abs())
            {
                ambiguous.insert(a);
                ambiguous.insert(b);
            }
        }
        for i in ambiguous {
            scores[i] = problem.evaluate(&checkpoint.population()[i]);
        }
        checkpoint.submit(problem, scores, &mut progress, &mut cancelled)?;
        if let Some(outcome) = checkpoint.outcome() {
            return Ok(outcome.clone());
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a Vulkan adapter; software is correctness evidence only"]
    fn attached_energy_batches_and_minimization_are_cpu_verified() {
        pollster::block_on(async {
            let options = dry_options();
            let builder = glysys::SystemBuilder::new(options.clone()).unwrap();
            let protein = read_pdb_str(
                include_str!("../../../tests/fixtures/protein.pdb"),
                &options,
            )
            .unwrap();
            let query = GlycanQuery {
                source: GlycanSource::LocalBundle(PathBuf::from("fixture.pdb")),
                anomer: Anomer::Beta,
                format: "PDB".into(),
                level: "2".into(),
            };
            let ensemble = ensemble_from_pdb(
                include_str!("../../../tests/fixtures/glycan.pdb"),
                None,
                query,
                "test",
            )
            .unwrap();
            let sites = vec![SearchSite {
                site: reglyco_core::GlycosylationSite::new("A", 1),
                ensemble,
            }];
            let structure = build_state(
                &protein,
                &sites,
                &[Gene {
                    conformer: 0,
                    phi: 0.,
                    psi: 0.,
                    rotamer: None,
                }],
                &builder,
            )
            .unwrap();
            for (mode, minimize) in [
                (SearchScoringMode::FullEnergy, false),
                (SearchScoringMode::ProteinGlycanInteraction, false),
                (SearchScoringMode::FullEnergy, true),
            ] {
                let c = EnergySearchContext {
                    system: builder.prepare_structure(&structure).unwrap(),
                    evaluator: OnceLock::new(),
                    mode,
                    use_obc2: mode == SearchScoringMode::FullEnergy,
                    minimize,
                    min_iterations: 2,
                    min_radius: 5.,
                    cutoff: 12.,
                    cache: Mutex::new(HashMap::new()),
                    atom_masks: OnceLock::new(),
                    atom_mapping: OnceLock::new(),
                    evaluations: AtomicUsize::new(0),
                    minimizations: AtomicUsize::new(0),
                    cache_hits: AtomicUsize::new(0),
                    steric_rejections: AtomicUsize::new(0),
                    failures: AtomicUsize::new(0),
                    started: Instant::now(),
                    topology_seconds: 0.,
                };
                configure("webgpu");
                let results =
                    evaluate_candidates(&c, &[structure.clone(), structure.clone()], false)
                        .await
                        .unwrap();
                for r in results {
                    let reference = cpu(
                        &c,
                        &r.coordinates,
                        &r.active_indices,
                        false,
                        mode == SearchScoringMode::ProteinGlycanInteraction,
                    )
                    .unwrap();
                    assert!(
                        close(r.score, reference.total(), 1e-4),
                        "GPU={} CPU={}",
                        r.score,
                        reference.total()
                    );
                }
                let report = finish();
                assert!(report["gpuEvaluations"].as_u64().unwrap() > 0, "{report}");
                configure("cpu");
                let result = gpu_evaluate_candidate(&c, &structure).await.unwrap();
                let expected = c.evaluate(&structure).unwrap();
                assert_eq!(result.coordinates, expected.coordinates);
                assert_eq!(result.score, expected.score);
                finish();
            }
        });
    }
}

/// Initialization may use the compatibility GA driver. Its buffers must be
/// released before the sampler acquires its independently owned energy session.
pub(super) fn release_compatibility_resources() {
    RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        r.scoring = None;
        r.scoring_key = None;
    });
}
pub(super) fn requested_backend() -> String {
    RUNTIME.with(|r| r.borrow().report.requested.clone())
}
fn merge_stage(a: &mut StageReport, b: StageReport) {
    a.gpu_evaluations += b.gpu_evaluations;
    a.cpu_evaluations += b.cpu_evaluations;
    a.validation_evaluations += b.validation_evaluations;
    a.gpu_seconds += b.gpu_seconds;
    a.cpu_seconds += b.cpu_seconds;
    if b.fallback_reason.is_some() {
        a.fallback_reason = b.fallback_reason;
    }
}
pub(super) fn record_geometry(count: usize, seconds: f64) {
    RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        r.report.gpu_evaluations += count;
        r.report.gpu_seconds += seconds;
        let stage = r.report.stages.entry("sterics".into()).or_default();
        stage.gpu_evaluations += count;
        stage.gpu_seconds += seconds;
    });
    notify("webgpu");
}
pub(super) fn record_reference(count: usize, seconds: f64, validation: bool) {
    RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        let stage = r
            .report
            .stages
            .entry("sampling_reference".into())
            .or_default();
        if validation {
            stage.validation_evaluations += count;
        } else {
            stage.cpu_evaluations += count;
        }
        stage.cpu_seconds += seconds;
    });
}
pub(super) fn record_cpu_geometry(count: usize, seconds: f64) {
    RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        let stage = r.report.stages.entry("sterics".into()).or_default();
        stage.cpu_evaluations += count;
        stage.cpu_seconds += seconds;
    });
}
pub(super) fn geometry_fallback(reason: String) {
    RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        r.report
            .stages
            .entry("sterics".into())
            .or_default()
            .fallback_reason = Some(reason.clone());
        r.report.fallback_reason = Some(reason);
    });
}
