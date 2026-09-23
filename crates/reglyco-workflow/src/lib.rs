//! Pure, in-memory workflow boundary shared by native and browser clients.
#[cfg(feature = "webgpu")]
use reglyco_ensemble::{
    sample_attached_ensemble_with_progress_cancel_async, search_with_progress_cancelled_async,
};

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
/// `web_time` mirrors `std::time::Instant` on native targets while
/// remaining functional on wasm32-unknown-unknown, where the standard
/// clock traps. The ensemble crate already uses it for the same reason.
use web_time::Instant;

use glysys::{BuildOptions, ResidueId, Structure, SystemBuilder, read_pdb_str};
use reglyco_build::{
    GlycosylationSite, hydroxylate_proline, remove_glycan_at_site, scan_n_linked_sequons,
};
use reglyco_core::{
    Anomer, ClashStatus, ConformerPopulationSource, EnergySearchDiagnostics, GlycanQuery,
    GlycanSource, ReGlycoError, ResidueNameFormat, SamplingTarget, SearchBudgetMode,
    SearchBudgetResolution, SearchConfig, SearchOutcome, SearchScoringMode, SearchSelectionPolicy,
    SearchSite, SearchSiteResult, VonMisesComponent,
};
use reglyco_ensemble::{
    EnsembleError, SearchPhase, SearchProgress, StrictSearchDiagnostics, build_from_outcome,
    calculate_sasa, ensemble_from_pdb, linkage_priors_for_glycan, resolve_search_budget,
    sample_attached_ensemble_with_progress_cancel, search_with_progress_cancelled,
    steric_site_scores,

};
use reglyco_relax::{MovableSelection, RelaxOptions, RelaxProgress, relax_with_progress};
use reglyco_validate::{Severity, StericPolicy, ValidationFinding as NativeFinding, validate};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[cfg(feature = "full")]
use reglyco_density::{DensityMap, DensityScoreOptions, DensityScorer, DensityTarget};
#[cfg(feature = "full")]
use reglyco_refine::{
    DensityEffort as NativeDensityEffort, DensityRefinementConfig, RefineObjective, RefineProgress,
    RefineRequest, refine_with_progress,
};
#[cfg(feature = "full")]
use reglyco_saxs::{
    FitOptions, MaximumEntropyOptions, PrOptions, adapt_structure, experimental_curve_from_str,
    fit_single_models, fit_unbiased_ensemble, reweight_ensemble,
};

pub const SCHEMA_VERSION: u32 = 1;
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

fn deserialize_seed<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    // Browser requests created by older CE builds may encode an omitted
    // optional seed as null/unit.  Keep the historical native default of 0
    // instead of rejecting the entire scan request.
    Ok(Option::<u64>::deserialize(deserializer)?.unwrap_or(0))
}

pub type Result<T> = std::result::Result<T, WorkflowError>;

#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("request is invalid: {0}")]
    Invalid(String),
    #[error("required input asset {0:?} is missing")]
    MissingAsset(String),
    #[error("asset {0:?} must contain UTF-8 text")]
    TextAsset(String),
    #[error("workflow was cancelled")]
    Cancelled,
    #[error("ReGlyco build failed: {0}")]
    Build(#[from] reglyco_core::ReGlycoError),
    #[error("GlySys failed: {0}")]
    GlySys(#[from] glysys::BuildError),
    #[error("ensemble search failed: {0}")]
    Ensemble(#[from] reglyco_ensemble::EnsembleError),
    #[error("ensemble analysis failed: {0}")]
    Sasa(#[from] reglyco_ensemble::SasaError),
    #[error("relaxation failed: {0}")]
    Relax(#[from] reglyco_relax::RelaxError),
    #[error("serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[cfg(feature = "full")]
    #[error("refinement failed: {0}")]
    Refine(#[from] reglyco_refine::RefineError),
    #[cfg(feature = "full")]
    #[error("density input failed: {0}")]
    Density(#[from] reglyco_density::DensityError),
    #[cfg(feature = "full")]
    #[error("SAXS fitting failed: {0}")]
    Saxs(#[from] reglyco_saxs::SaxsError),
    #[error("this workflow is not compiled into the current profile: {0:?}")]
    Capability(WorkflowId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReGlycoProfile {
    Public,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowId {
    Uniprot,
    NScan,
    SiteBuild,
    Ensemble,
    Relax,
    Validate,
    Refine,
    Density,
    Saxs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SiteKey {
    #[serde(default = "default_model")]
    pub model: usize,
    pub chain: String,
    pub residue_number: i32,
    #[serde(default)]
    pub insertion_code: Option<char>,
}

impl SiteKey {
    pub fn residue_id(&self) -> ResidueId {
        ResidueId {
            chain: self.chain.clone(),
            number: self.residue_number,
            insertion_code: self.insertion_code,
        }
    }
}

fn default_model() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentProvenance {
    Uniprot,
    Scan,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SiteAssignment {
    pub id: String,
    pub site: SiteKey,
    pub residue_name: String,
    pub glycan_id: String,
    #[serde(default)]
    pub anomer: Anomer,
    #[serde(default = "default_level")]
    pub level: u8,
    pub provenance: AssignmentProvenance,
    #[serde(default)]
    pub source_description: Option<String>,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub glycan_asset: Option<String>,
    #[serde(default)]
    pub excluded: bool,
    #[serde(default)]
    pub unresolved: bool,
    /// Explicitly remove the deposited glycan at this site before attaching
    /// the selected replacement asset.
    #[serde(default)]
    pub replace_existing: bool,
    /// Identifier of the deposited glycan, retained for provenance only.
    #[serde(default)]
    pub existing_glycan_id: Option<String>,
    /// Explicit deposited-tree residue references used when parser metadata
    /// does not contain a GlycanTree for the attachment site.
    #[serde(default)]
    pub existing_glycan_residues: Vec<SiteKey>,
}

fn default_level() -> u8 {
    2
}

fn default_ensemble_temperature_k() -> f64 {
    300.0
}

fn default_ensemble_mh_chains() -> usize {
    8
}

fn default_ensemble_burn_in_sweeps() -> usize {
    250
}

fn default_ensemble_thinning_accepted() -> usize {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DensityEffort {
    Fast,
    Adaptive,
    Deep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaxsMode {
    Model,
    Ensemble,
    Reweight,
    Occupancy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ReGlycoOptions {
    pub ensemble_mode: Option<String>,
    pub ensemble_burn_in_steps: Option<usize>,
    pub ensemble_thinning_steps: Option<usize>,
    pub compute_backend: String,
    /// Optional numerical target for sampled energy/interaction ensembles.
    /// Omitted requests retain the historical CPU-reference behavior unless
    /// this is a new full-profile GPU/Auto request.
    #[serde(default)]
    pub sampling_target: Option<SamplingTarget>,
    pub pre_minimization: bool,
    #[serde(default, deserialize_with = "deserialize_seed")]
    pub seed: u64,
    /// Residue-name convention used for fetched Build/Ensemble assets.
    /// This is a representation choice and does not alter force-field names.
    #[serde(default)]
    pub output_format: ResidueNameFormat,
    pub post_relax: bool,
    /// Try Dunbrack sidechain rotamers at attachment residues when a pose
    /// needs local sidechain repair. This is opt-in because it can change
    /// protein coordinates.
    #[serde(default)]
    pub scan_rotamers: bool,
    pub ensemble_frames: usize,
    /// Temperature used by energy/interaction-weighted ensemble acceptance.
    #[serde(default = "default_ensemble_temperature_k")]
    pub ensemble_temperature_k: f64,
    /// Number of independent chains used by constrained-MH fallback sampling.
    #[serde(default = "default_ensemble_mh_chains")]
    pub ensemble_mh_chains: usize,
    /// Burn-in sweeps per constrained-MH chain.
    #[serde(default = "default_ensemble_burn_in_sweeps")]
    pub ensemble_burn_in_sweeps: usize,
    /// Accepted native proposals between emitted fallback frames.
    #[serde(default = "default_ensemble_thinning_accepted")]
    pub ensemble_thinning_accepted: usize,
    pub calculate_sasa: bool,
    pub calculate_hotspots: bool,
    pub scoring_mode: SearchScoringMode,
    pub use_obc2: bool,
    pub local_radius: f64,
    /// New clients send `auto`. An omitted mode is normalized as historical
    /// manual mode so persisted requests keep their original budget.
    #[serde(default)]
    pub search_budget_mode: Option<SearchBudgetMode>,
    pub population_size: usize,
    pub generations: usize,
    pub density_effort: DensityEffort,
    pub density_support_threshold: f64,
    pub density_credible_mass: f64,
    pub density_max_alternates: usize,
    pub density_post_relax: bool,
    pub saxs_mode: SaxsMode,
}

impl Default for ReGlycoOptions {
    fn default() -> Self {
        Self {
            compute_backend: "auto".into(),
            sampling_target: None,
            ensemble_mode: None,
            ensemble_burn_in_steps: None,
            ensemble_thinning_steps: None,
            pre_minimization: false,
            seed: 0,
            output_format: ResidueNameFormat::Pdb,
            post_relax: false,
            scan_rotamers: false,
            ensemble_frames: 50,
            ensemble_temperature_k: default_ensemble_temperature_k(),
            ensemble_mh_chains: default_ensemble_mh_chains(),
            ensemble_burn_in_sweeps: default_ensemble_burn_in_sweeps(),
            ensemble_thinning_accepted: default_ensemble_thinning_accepted(),
            calculate_sasa: true,
            calculate_hotspots: true,
            scoring_mode: SearchScoringMode::StericPrior,
            use_obc2: false,
            local_radius: 5.0,
            search_budget_mode: Some(SearchBudgetMode::Auto),
            population_size: 128,
            generations: 100,
            density_effort: DensityEffort::Adaptive,
            density_support_threshold: 0.60,
            density_credible_mass: 0.95,
            density_max_alternates: 10,
            density_post_relax: false,
            saxs_mode: SaxsMode::Model,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProteinInputKind {
    Uniprot,
    PdbId,
    Upload,
    Job,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProteinInput {
    pub kind: ProteinInputKind,
    pub label: String,
    #[serde(default)]
    pub source_id: Option<String>,
    pub asset: String,
    pub sha256: String,
    #[serde(default)]
    pub source_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReGlycoRunRequestV1 {
    pub schema_version: u32,
    pub workflow: WorkflowId,
    pub profile: ReGlycoProfile,
    pub input: ProteinInput,
    #[serde(default)]
    pub assignments: Vec<SiteAssignment>,
    #[serde(default)]
    pub options: ReGlycoOptions,
    #[serde(default)]
    pub parent_job_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssetData {
    Text(String),
    Bytes(Vec<u8>),
}

impl AssetData {
    fn text(&self, name: &str) -> Result<&str> {
        match self {
            Self::Text(value) => Ok(value),
            Self::Bytes(value) => {
                std::str::from_utf8(value).map_err(|_| WorkflowError::TextAsset(name.into()))
            }
        }
    }

    #[cfg(feature = "full")]
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Text(value) => value.as_bytes(),
            Self::Bytes(value) => value,
        }
    }
}

pub type InputAssets = BTreeMap<String, AssetData>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub stage: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStatus {
    Succeeded,
    Partial,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationFinding {
    pub code: String,
    pub severity: FindingSeverity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub site: Option<SiteKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glycan_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glycan: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linkage: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub involved_atoms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationSummary {
    pub valid: bool,
    pub findings: Vec<ValidationFinding>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_dictionary_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steric_policy: Option<StericPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steric_summary: Option<reglyco_core::steric::ContactSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportSite {
    pub site: SiteKey,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glycan_id: Option<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(default)]
    pub details: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowReport {
    pub workflow: WorkflowId,
    pub title: String,
    pub summary: String,
    pub generated_at: String,
    pub engine_version: String,
    pub input_sha256: String,
    pub sites: Vec<ReportSite>,
    pub validation: ValidationSummary,
    pub analysis: Value,
    pub provenance: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRole {
    Input,
    Structure,
    Analysis,
    Report,
    Provenance,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowArtifact {
    pub name: String,
    pub media_type: String,
    pub role: ArtifactRole,
    pub data: AssetData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowBundle {
    pub schema_version: u32,
    pub status: WorkflowStatus,
    pub workflow: WorkflowId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_structure: Option<String>,
    pub report: WorkflowReport,
    pub artifacts: Vec<WorkflowArtifact>,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineCapabilities {
    pub schema_version: u32,
    pub engine_version: String,
    pub profile: ReGlycoProfile,
    pub workflows: Vec<WorkflowId>,
    pub threaded: bool,
    pub pdf: bool,
}

pub trait WorkflowControl {
    fn progress(&mut self, event: ProgressEvent);
    fn cancelled(&self) -> bool {
        false
    }
}

pub struct NoopControl;
impl WorkflowControl for NoopControl {
    fn progress(&mut self, _: ProgressEvent) {}
}

pub fn capabilities(profile: ReGlycoProfile, threaded: bool) -> EngineCapabilities {
    let mut workflows = vec![
        WorkflowId::Uniprot,
        WorkflowId::NScan,
        WorkflowId::SiteBuild,
        WorkflowId::Ensemble,
        WorkflowId::Relax,
        WorkflowId::Validate,
    ];
    #[cfg(feature = "refine")]
    workflows.push(WorkflowId::Refine);
    #[cfg(feature = "full")]
    if profile == ReGlycoProfile::Full {
        workflows.extend([WorkflowId::Density, WorkflowId::Saxs]);
    }
    EngineCapabilities {
        schema_version: SCHEMA_VERSION,
        engine_version: ENGINE_VERSION.into(),
        profile,
        workflows,
        threaded,
        pdf: false,
    }
}

fn emit(
    control: &mut impl WorkflowControl,
    stage: &str,
    message: impl Into<String>,
    current: Option<usize>,
    total: Option<usize>,
) -> Result<()> {
    if control.cancelled() {
        return Err(WorkflowError::Cancelled);
    }
    control.progress(ProgressEvent {
        stage: stage.into(),
        message: message.into(),
        current,
        total,
        fraction: current
            .zip(total)
            .filter(|(_, total)| *total > 0)
            .map(|(current, total)| current as f64 / total as f64),
    });
    Ok(())
}

fn build_options(request: &ReGlycoRunRequestV1) -> BuildOptions {
    BuildOptions {
        add_water: false,
        add_ions: false,
        seed: request.options.seed,
        ..BuildOptions::default()
    }
}

fn search_config(request: &ReGlycoRunRequestV1) -> SearchConfig {
    SearchConfig {
        seed: request.options.seed,
        ensemble_size: request.options.ensemble_frames,
        population_size: request.options.population_size,
        generations: request.options.generations,
        // Generic energy/interaction searches retain their historical
        // inspectable best-candidate behavior.  Steric-prior searches use the
        // dedicated strict solver and fail with typed diagnostics before this
        // flag is consulted.
        require_clash_free: false,
        // Rotamer repair is deliberately enabled only by the attachment
        // Build/Ensemble adapters below. Refine and scan retain their
        // historical fixed-protein behavior.
        scan_rotamers: false,
        scoring_mode: request.options.scoring_mode,
        use_obc2: request.options.use_obc2,
        minimization_radius: request.options.local_radius,
        ..SearchConfig::default()
    }
}

/// Fixed small search budget for GlcNAc accessibility scans.
///
/// A scan asks one cheap question per sequon (can a single GlcNAc sit here
/// clash-free?) and must answer it for every sequon on the chain. Running the
/// full Build/Ensemble budget -- or the large Auto budget -- once per sequon
/// plus once per joint trial makes many-site scans unusable, while the extra
/// work does not change the accessibility answer: feasible single-site poses
/// are normally found in the initial population, and blocked sites exhaust
/// any budget. Scan results therefore use this fixed first-feasible budget
/// and never inherit the request's Build/Ensemble population, generations,
/// or Auto mode.
fn scan_search_config(request: &ReGlycoRunRequestV1) -> SearchConfig {
    let mut config = search_config(request);
    config.population_size = 32;
    config.generations = 25;
    config.require_clash_free = false;
    config.scan_rotamers = false;
    config.polish_attachment_vmm = false;
    config.selection_policy = SearchSelectionPolicy::CookbookFirstFeasible;
    config
}

/// Directly verify that previously accepted per-site scan results are jointly
/// clash-free by materializing them together and re-scoring. This is one
/// structure build plus one steric traversal, versus a complete search.
fn scan_trial_compatible(
    protein: &Structure,
    sites: &[SearchSite],
    results: &[SearchSiteResult],
    config: &SearchConfig,
    builder: &SystemBuilder,
) -> Result<bool> {
    let trial = SearchOutcome {
        sites: results.to_vec(),
        total_score: results.iter().map(|result| result.steric_score).sum(),
        seed: config.seed,
        generations: 0,
        clash_status: ClashStatus::ClashFree,
        complete_output: true,
        vmm_gate_satisfied: true,
        termination_reason: "scan_direct_combination".into(),
        clash_partners: Vec::new(),
        history: Vec::new(),
        warnings: Vec::new(),
        scoring_mode: SearchScoringMode::StericPrior,
        selected_energy_kcal_per_mol: None,
        interaction_energy_kcal_per_mol: None,
        energy_evaluations: 0,
        energy_cutoff_angstrom: config.energy_cutoff,
        minimization_radius_angstrom: config.minimization_radius,
        interaction_vdw_kcal_per_mol: None,
        interaction_coulomb_kcal_per_mol: None,
        energy_diagnostics: EnergySearchDiagnostics::default(),
        minimized_coordinates: Vec::new(),
        timings: Default::default(),
        vmm_polish: Default::default(),
        energy_analysis: None,
    };
    let product = build_from_outcome(protein, sites, &trial, builder, false)?;
    Ok(
        steric_site_scores(&product.structure, config.clash_distance)
            .iter()
            .all(|score| *score <= 1.1),
    )

}

fn attachment_search_config(request: &ReGlycoRunRequestV1) -> SearchConfig {
    let mut config = search_config(request);
    config.scan_rotamers = request.options.scan_rotamers;
    config.pre_minimization = request.options.pre_minimization;
    config.ensemble_mode = request.options.ensemble_mode.clone().or_else(|| {
        (request.options.post_relax || request.options.pre_minimization)
            .then(|| "conformer_collection".into())
    });
    config.burn_in_steps = request.options.ensemble_burn_in_steps;
    config.thinning_steps = request.options.ensemble_thinning_steps;
    config.polish_attachment_vmm = true;
    if config.scoring_mode == SearchScoringMode::StericPrior {
        config.selection_policy = SearchSelectionPolicy::JointPriorV1;
    }
    config
}

fn attachment_search_config_with_budget(
    request: &ReGlycoRunRequestV1,
    budget: &SearchBudgetResolution,
) -> SearchConfig {
    let mut config = attachment_search_config(request);
    config.population_size = budget.population_size;
    config.generations = budget.generations;
    config
}

fn resolved_attachment_budget(
    request: &ReGlycoRunRequestV1,
    protein: &Structure,
    sites: &[SearchSite],
) -> Result<SearchBudgetResolution> {
    resolve_search_budget(
        protein,
        sites,
        request.options.scan_rotamers,
        request.options.search_budget_mode,
        request.options.population_size,
        request.options.generations,
    )
    .map_err(WorkflowError::from)
}

fn ensemble_search_config(request: &ReGlycoRunRequestV1) -> SearchConfig {
    let mut config = attachment_search_config(request);
    // Pulling each accepted frame toward its attachment VMM mode would
    // collapse the ensemble distribution. The bounded polish is Build-only.
    config.polish_attachment_vmm = false;
    // Ensemble frames retain the sampler's declared target and proposal
    // distribution.  The probability-ranked incumbent policy is specific to
    // steric Build; enabling it here would silently turn sampling into an
    // optimizer even when the scoring mode is StericPrior.
    config.selection_policy = SearchSelectionPolicy::CookbookFirstFeasible;
    config.temperature_k = request.options.ensemble_temperature_k;
    config.mh_chains = request.options.ensemble_mh_chains;
    config.mh_burn_in_sweeps = request.options.ensemble_burn_in_sweeps;
    config.mh_thinning_accepted = request.options.ensemble_thinning_accepted;
    config.sampling_target = Some(resolve_sampling_target(request));
    config
}

fn resolve_sampling_target(request: &ReGlycoRunRequestV1) -> SamplingTarget {
    request.options.sampling_target.unwrap_or_else(|| {
        if request.profile == ReGlycoProfile::Full && request.options.compute_backend != "cpu" {
            SamplingTarget::WebgpuF32V1
        } else {
            SamplingTarget::CpuReferenceV1
        }
    })
}

fn validate_options(request: &ReGlycoRunRequestV1) -> Result<()> {
    if !["auto", "cpu", "webgpu"].contains(&request.options.compute_backend.as_str()) {
        return Err(WorkflowError::Invalid("unsupported compute backend".into()));
    }
    // Ensemble sampler controls are intentionally scoped to Process Ensemble.
    // Build, scan, relax, validate, and refine requests may carry the
    // additive fields for schema compatibility, but invalid/unused ensemble
    // values must not prevent those workflows from running.
    if request.workflow != WorkflowId::Ensemble {
        return Ok(());
    }
    let options = &request.options;
    if options
        .ensemble_mode
        .as_deref()
        .is_some_and(|m| !["sampled", "conformer_collection"].contains(&m))
    {
        return Err(WorkflowError::Invalid("unknown ensemble mode".into()));
    }
    if options.ensemble_mode.as_deref() == Some("sampled")
        && (options.pre_minimization || options.post_relax)
    {
        return Err(WorkflowError::Invalid(
            "sampled ensembles cannot minimize frames; choose conformer collection".into(),
        ));
    }
    if options.ensemble_thinning_steps == Some(0) {
        return Err(WorkflowError::Invalid(
            "ensemble thinning steps must be positive".into(),
        ));
    }
    if !(1..=500).contains(&options.ensemble_frames) {
        return Err(WorkflowError::Invalid(
            "ensembleFrames must be between 1 and 500".into(),
        ));
    }
    if options.ensemble_mh_chains == 0 {
        return Err(WorkflowError::Invalid(
            "ensembleMhChains must be at least 1".into(),
        ));
    }
    if options.ensemble_thinning_accepted == 0 {
        return Err(WorkflowError::Invalid(
            "ensembleThinningAccepted must be at least 1".into(),
        ));
    }
    if !options.ensemble_temperature_k.is_finite() || options.ensemble_temperature_k <= 0.0 {
        return Err(WorkflowError::Invalid(
            "ensembleTemperatureK must be a finite positive number".into(),
        ));
    }
    Ok(())
}

fn effective_rotamer_setting(request: &ReGlycoRunRequestV1) -> bool {
    matches!(
        request.workflow,
        WorkflowId::Uniprot | WorkflowId::SiteBuild | WorkflowId::Ensemble
    ) && request.options.scan_rotamers
}

fn effective_output_format(request: &ReGlycoRunRequestV1) -> ResidueNameFormat {
    if matches!(
        request.workflow,
        WorkflowId::Uniprot | WorkflowId::SiteBuild | WorkflowId::Ensemble
    ) {
        request.options.output_format
    } else {
        ResidueNameFormat::Pdb
    }
}

fn effective_selection_policy(request: &ReGlycoRunRequestV1) -> &'static str {
    if matches!(
        request.workflow,
        WorkflowId::Uniprot | WorkflowId::SiteBuild
    ) && request.options.scoring_mode == SearchScoringMode::StericPrior
    {
        "joint_prior_v1"
    } else if request.workflow == WorkflowId::Ensemble {
        // Ensemble sampling has its own target/proposal semantics and must
        // never be presented as a probability optimizer.
        "cookbook_first_feasible"
    } else {
        "cookbook_first_feasible"
    }
}

fn method_metadata(request: &ReGlycoRunRequestV1) -> Value {
    let mut method = serde_json::Map::from_iter([
        (
            "scanRotamers".into(),
            Value::Bool(effective_rotamer_setting(request)),
        ),
        (
            "scoringMode".into(),
            serde_json::to_value(request.options.scoring_mode).unwrap_or(Value::Null),
        ),
        ("seed".into(), json!(request.options.seed)),
        (
            "selectionPolicy".into(),
            json!(effective_selection_policy(request)),
        ),
    ]);
    if matches!(
        request.workflow,
        WorkflowId::Uniprot | WorkflowId::SiteBuild | WorkflowId::Ensemble
    ) {
        method.insert(
            "molecularScoringModelVersion".into(),
            json!("amber-glycam-v2"),
        );
        method.insert(
            "priorModel".into(),
            json!("conformer_population_x_attachment_vmm_v1"),
        );
        method.insert(
            "outputFormat".into(),
            json!(effective_output_format(request)),
        );
        method.insert(
            "searchBudgetMode".into(),
            json!(request.options.search_budget_mode),
        );
    }
    if request.workflow == WorkflowId::Ensemble {
        let mode = request.options.ensemble_mode.as_deref().unwrap_or(
            if request.options.pre_minimization || request.options.post_relax {
                "conformer_collection"
            } else {
                "sampled"
            },
        );
        method.insert(
            "ensembleSampler".into(),
            json!({
                "frames": request.options.ensemble_frames,
                "temperatureK": request.options.ensemble_temperature_k,
                "mhChains": request.options.ensemble_mh_chains,
                "burnInSweeps": request.options.ensemble_burn_in_sweeps,
                "thinningAccepted": request.options.ensemble_thinning_accepted,
                "samplerVersion": if mode=="sampled" {
                    match resolve_sampling_target(request) {
                        SamplingTarget::WebgpuF32V1 => "gpu-f32-mh-v1",
                        SamplingTarget::CpuReferenceV1 => "model-da-mh-v3",
                    }
                } else {"conformer-collection-v2"},
                "samplingTarget": serde_json::to_value(resolve_sampling_target(request)).unwrap_or(Value::Null),
                "modelVersion": "amber-glycam-v2",
                "ensembleMode": request.options.ensemble_mode.as_deref().unwrap_or(if request.options.pre_minimization || request.options.post_relax { "conformer_collection" } else { "sampled" }),
                "burnInSteps": request.options.ensemble_burn_in_steps.unwrap_or(request.options.ensemble_burn_in_sweeps.saturating_mul(request.assignments.iter().filter(|a|!a.excluded).count())),
                "thinningSteps": request.options.ensemble_thinning_steps.unwrap_or(request.options.ensemble_thinning_accepted),
                "calculateSasa": request.options.calculate_sasa,
                // Hotspots are derived from SASA; report the effective value
                // rather than an inconsistent stale toggle from an older
                // request/history record.
                "calculateHotspots": request.options.calculate_sasa
                    && request.options.calculate_hotspots,
            }),
        );
    }
    Value::Object(method)
}

fn blocked_scan_outcome(error: &EnsembleError) -> bool {
    matches!(
        error,
        EnsembleError::ReGlyco(ReGlycoError::ClashFreeRequired)
            | EnsembleError::StrictVmmFailure { .. }
    )
}

fn relax_options(request: &ReGlycoRunRequestV1) -> RelaxOptions {
    RelaxOptions {
        movable: MovableSelection::Glycans,
        include_local_sidechains: true,
        local_radius: request.options.local_radius,
        ..RelaxOptions::default()
    }
}

fn input_text<'a>(request: &ReGlycoRunRequestV1, assets: &'a InputAssets) -> Result<&'a str> {
    assets
        .get(&request.input.asset)
        .ok_or_else(|| WorkflowError::MissingAsset(request.input.asset.clone()))?
        .text(&request.input.asset)
}

fn prepare_proline_sites(
    mut protein: Structure,
    assignments: &[SiteAssignment],
    options: &BuildOptions,
) -> Result<Structure> {
    let mut converted = BTreeSet::new();
    for assignment in assignments.iter().filter(|assignment| !assignment.excluded) {
        let site = assignment.site.residue_id();
        let is_proline = protein
            .residues()
            .into_iter()
            .find(|residue| residue.id == site)
            .is_some_and(|residue| residue.name.eq_ignore_ascii_case("PRO"));
        if is_proline && converted.insert(site.clone()) {
            protein = hydroxylate_proline(&protein, &site, options)?;
        }
    }
    Ok(protein)
}

fn replacement_workflow(workflow: WorkflowId) -> bool {
    matches!(
        workflow,
        WorkflowId::Uniprot
            | WorkflowId::SiteBuild
            | WorkflowId::Ensemble
            | WorkflowId::Refine
            | WorkflowId::Density
    )
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReplacementRecord {
    site: SiteKey,
    existing_glycan_id: Option<String>,
    removed_residues: Vec<SiteKey>,
    replacement_glycan_id: String,
}

fn site_key_for_residue(model: usize, residue: &ResidueId) -> SiteKey {
    SiteKey {
        model,
        chain: residue.chain.clone(),
        residue_number: residue.number,
        insertion_code: residue.insertion_code,
    }
}

fn display_site(site: &SiteKey) -> String {
    format!(
        "{}:{}{}",
        site.chain,
        site.residue_number,
        site.insertion_code
            .map(|code| code.to_string())
            .unwrap_or_default()
    )
}

fn is_protein_residue_name(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "CYS"
            | "GLN"
            | "GLU"
            | "GLY"
            | "HIS"
            | "HID"
            | "HIE"
            | "HIP"
            | "ILE"
            | "LEU"
            | "LYS"
            | "MET"
            | "MSE"
            | "PHE"
            | "PRO"
            | "HYP"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "NLN"
            | "OLS"
            | "OLT"
            | "OLP"
    )
}

fn requested_replacement_metadata(request: &ReGlycoRunRequestV1) -> Value {
    Value::Array(
        request
            .assignments
            .iter()
            .filter(|assignment| !assignment.excluded && assignment.replace_existing)
            .map(|assignment| {
                json!({
                    "site": assignment.site,
                    "existingGlycanId": assignment.existing_glycan_id,
                    "removedResidues": assignment.existing_glycan_residues,
                    "replacementGlycanId": assignment.glycan_id,
                })
            })
            .collect(),
    )
}

/// Attach the exact records produced while preparing a replacement to a
/// workflow analysis object.  Full-profile adapters (refine and density) are
/// entered after scaffold preparation, so they receive these records
/// explicitly rather than having to reconstruct them from the request.
fn add_replacement_analysis(mut analysis: Value, replacements: &[ReplacementRecord]) -> Value {
    if replacements.is_empty() {
        return analysis;
    }
    let value = serde_json::to_value(replacements).unwrap_or_else(|_| Value::Array(Vec::new()));
    if let Some(object) = analysis.as_object_mut() {
        object.insert("replacements".into(), value);
    } else {
        analysis = json!({ "workflow": analysis, "replacements": value });
    }
    analysis
}

/// Remove only the deposited trees explicitly selected for replacement. The
/// browser inspection persists residue references so this remains deterministic
/// even when a PDB parser cannot reconstruct its GlycanTree metadata.
fn prepare_replacements(
    mut protein: Structure,
    request: &ReGlycoRunRequestV1,
) -> Result<(Structure, Vec<ReplacementRecord>)> {
    // Relax, validation, and the GlcNAc scan inspect or prepare structures but
    // never attach a replacement. Leave their input untouched, including any
    // ordinary assignment rows that may be present in a restored request.
    if !replacement_workflow(request.workflow) {
        return Ok((protein, Vec::new()));
    }

    let mut seen_sites = BTreeSet::<ResidueId>::new();
    let mut removed_all = BTreeSet::<ResidueId>::new();
    let mut records = Vec::new();
    let available = protein
        .residues()
        .into_iter()
        .map(|residue| (residue.id, residue.name))
        .collect::<BTreeMap<_, _>>();
    for assignment in request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded)
    {
        let site = assignment.site.residue_id();
        if !available.contains_key(&site) {
            return Err(WorkflowError::Invalid(format!(
                "attachment site {} is not present in the input structure",
                display_site(&assignment.site)
            )));
        }
        let matching_trees = protein
            .metadata()
            .glycan_trees
            .iter()
            .filter(|tree| tree.attachment_site.as_ref() == Some(&site))
            .collect::<Vec<_>>();
        if matching_trees.len() > 1 {
            return Err(WorkflowError::Invalid(format!(
                "multiple deposited glycans were found at {}; replacement target is ambiguous",
                display_site(&assignment.site)
            )));
        }
        let has_tree = !matching_trees.is_empty();
        let persisted_references = assignment
            .existing_glycan_residues
            .iter()
            .map(SiteKey::residue_id)
            .collect::<BTreeSet<_>>();

        if !assignment.replace_existing {
            // Never silently append a second tree to an occupied attachment
            // site. The UI marks occupied rows as replacements, but this gate
            // also protects native/CLI callers.
            if assignment.glycan_id != "existing" && (has_tree || !persisted_references.is_empty())
            {
                return Err(WorkflowError::Invalid(format!(
                    "{} already has a deposited glycan; mark the assignment as a replacement",
                    display_site(&assignment.site)
                )));
            }
            continue;
        }
        if assignment.glycan_id.trim().is_empty() || assignment.glycan_id == "existing" {
            return Err(WorkflowError::Invalid(format!(
                "replacement at {} needs a new glycan mapping",
                display_site(&assignment.site)
            )));
        }
        if !seen_sites.insert(site.clone()) {
            return Err(WorkflowError::Invalid(format!(
                "multiple replacement assignments target {}",
                display_site(&assignment.site)
            )));
        }

        let tree_residues = matching_trees
            .first()
            .map(|tree| tree.residue_ids.iter().cloned().collect::<BTreeSet<_>>())
            .unwrap_or_default();
        let tree_complete = !tree_residues.is_empty();
        if tree_complete {
            let missing_tree_residues = tree_residues
                .iter()
                .filter(|residue| !available.contains_key(*residue))
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if !missing_tree_residues.is_empty() {
                return Err(WorkflowError::Invalid(format!(
                    "deposited glycan at {} references missing residue(s): {}",
                    display_site(&assignment.site),
                    missing_tree_residues.join(", ")
                )));
            }
            let protein_tree_residues = tree_residues
                .iter()
                .filter_map(|residue| available.get(residue).map(|name| (residue, name)))
                .filter(|(_, name)| is_protein_residue_name(name))
                .map(|(residue, _)| residue.to_string())
                .collect::<Vec<_>>();
            if !protein_tree_residues.is_empty() {
                return Err(WorkflowError::Invalid(format!(
                    "deposited glycan at {} references protein residue(s): {}",
                    display_site(&assignment.site),
                    protein_tree_residues.join(", ")
                )));
            }
        }
        // Prefer persisted references when they cover the parser's tree. This
        // repairs older/incomplete parser metadata without allowing a stale
        // subset to leave part of an otherwise complete deposited tree behind.
        let use_persisted_references = !persisted_references.is_empty()
            && (!tree_complete || tree_residues.is_subset(&persisted_references));
        let (stripped, removed) = if tree_complete && !use_persisted_references {
            remove_glycan_at_site(&protein, &site).map_err(|error| {
                WorkflowError::Invalid(format!(
                    "could not remove the deposited glycan at {}: {error}",
                    display_site(&assignment.site)
                ))
            })?
        } else {
            let references = persisted_references;
            if references.is_empty() {
                return Err(WorkflowError::Invalid(format!(
                    "no deposited glycan was found at {}",
                    display_site(&assignment.site)
                )));
            }
            if references.contains(&site) {
                return Err(WorkflowError::Invalid(format!(
                    "replacement residue references at {} include the protein attachment residue",
                    display_site(&assignment.site)
                )));
            }
            let protein_references = references
                .iter()
                .filter_map(|residue| available.get(residue).map(|name| (residue, name)))
                .filter(|(_, name)| is_protein_residue_name(name))
                .map(|(residue, _)| residue.to_string())
                .collect::<Vec<_>>();
            if !protein_references.is_empty() {
                return Err(WorkflowError::Invalid(format!(
                    "replacement at {} references protein residue(s): {}",
                    display_site(&assignment.site),
                    protein_references.join(", ")
                )));
            }
            let protected = protein
                .metadata()
                .glycan_trees
                .iter()
                .filter(|tree| tree.attachment_site.as_ref() != Some(&site))
                .flat_map(|tree| tree.residue_ids.iter())
                .collect::<BTreeSet<_>>();
            let overlaps_other_tree = references
                .iter()
                .filter(|residue| protected.contains(residue))
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if !overlaps_other_tree.is_empty() {
                return Err(WorkflowError::Invalid(format!(
                    "replacement at {} would remove residue(s) from another deposited glycan: {}",
                    display_site(&assignment.site),
                    overlaps_other_tree.join(", ")
                )));
            }
            let missing = references
                .iter()
                .filter(|residue| !available.contains_key(*residue))
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(WorkflowError::Invalid(format!(
                    "deposited glycan at {} references missing residue(s): {}",
                    display_site(&assignment.site),
                    missing.join(", ")
                )));
            }
            let mut stripped = protein.clone();
            stripped.remove_residues(&references);
            (stripped, references.into_iter().collect::<Vec<_>>())
        };

        if removed.iter().any(|residue| removed_all.contains(residue)) {
            return Err(WorkflowError::Invalid(format!(
                "replacement trees at {} overlap",
                display_site(&assignment.site)
            )));
        }
        removed_all.extend(removed.iter().cloned());
        let removed_residues = removed
            .iter()
            .map(|residue| site_key_for_residue(assignment.site.model, residue))
            .collect();
        protein = stripped;
        records.push(ReplacementRecord {
            site: assignment.site.clone(),
            existing_glycan_id: assignment.existing_glycan_id.clone(),
            removed_residues,
            replacement_glycan_id: assignment.glycan_id.clone(),
        });
    }
    Ok((protein, records))
}

fn workflow_converts_proline(workflow: WorkflowId) -> bool {
    !matches!(workflow, WorkflowId::Validate | WorkflowId::NScan)
}

fn converted_proline_sites(
    request: &ReGlycoRunRequestV1,
    input: &str,
    output: &Structure,
) -> Vec<SiteKey> {
    let Ok(original) = read_pdb_str(input, &build_options(request)) else {
        return Vec::new();
    };
    let mut seen = BTreeSet::new();
    request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded)
        .filter_map(|assignment| {
            let site = assignment.site.residue_id();
            let was_pro = original
                .residues()
                .into_iter()
                .find(|residue| residue.id == site)
                .is_some_and(|residue| residue.name.eq_ignore_ascii_case("PRO"));
            let is_hyp = output
                .residues()
                .into_iter()
                .find(|residue| residue.id == site)
                .is_some_and(|residue| residue.name.eq_ignore_ascii_case("HYP"));
            (was_pro && is_hyp && seen.insert(site.clone())).then_some(SiteKey {
                model: assignment.site.model,
                chain: assignment.site.chain.clone(),
                residue_number: assignment.site.residue_number,
                insertion_code: assignment.site.insertion_code,
            })
        })
        .collect()
}

fn glycan_terminal_sugar(metadata: &Value) -> Option<String> {
    let iupac = metadata
        .get("archetype")
        .and_then(|archetype| archetype.get("iupac"))
        .and_then(Value::as_str)?;
    let tail = iupac
        .trim_end_matches(|character| character == ')' || character == ']' || character == ' ');
    [
        "Neu5Ac", "Neu5Gc", "GlcNAc", "GalNAc", "Araf", "Fuc", "Man", "Glc", "Gal", "Xyl",
    ]
    .iter()
    .find(|token| tail.ends_with(**token))
    .map(|token| (*token).to_string())
}

fn metadata_asset_name(asset_name: &str) -> String {
    asset_name
        .strip_suffix(".pdb")
        .or_else(|| asset_name.strip_suffix(".PDB"))
        .map_or_else(
            || format!("{asset_name}.json"),
            |stem| format!("{stem}.json"),
        )
}

fn search_sites(
    request: &ReGlycoRunRequestV1,
    assets: &InputAssets,
    protein: &Structure,
) -> Result<Vec<SearchSite>> {
    // UniProt one-shot requests commonly attach the same terminal glycan to
    // many sites. Parse each browser-fetched multi-model asset once; cloning
    // the parsed ensemble is considerably cheaper and avoids repeatedly
    // allocating its model parser buffers before the search starts.
    let mut ensemble_cache = HashMap::<String, reglyco_core::GlycanEnsemble>::new();
    let requested_format = effective_output_format(request).api_segment();
    request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded && assignment.glycan_id != "existing")
        .map(|assignment| {
            let asset_name = assignment.glycan_asset.as_ref().ok_or_else(|| {
                WorkflowError::Invalid(format!("{} has no glycan asset", assignment.id))
            })?;
            let query = GlycanQuery {
                source: GlycanSource::LocalBundle(PathBuf::from(asset_name)),
                anomer: assignment.anomer.clone(),
                format: requested_format.into(),
                level: assignment.level.to_string(),
            };
            let metadata_name = metadata_asset_name(asset_name);
            let metadata = assets
                .get(&metadata_name)
                .and_then(|asset| asset.text(&metadata_name).ok())
                .and_then(|text| serde_json::from_str::<Value>(text).ok());
            let mut ensemble = if let Some(cached) = ensemble_cache.get(asset_name) {
                cached.clone()
            } else {
                let pdb = assets
                    .get(asset_name)
                    .ok_or_else(|| WorkflowError::MissingAsset(asset_name.clone()))?
                    .text(asset_name)?;
                // The multi-model asset contains GlycoShape cluster medoids;
                // its metadata contains their source-population weights and
                // Level-1 parents. Preserve both so browser Build/Ensemble
                // sample the same conformational population as native runs.
                // `ensemble_from_pdb` already falls back to equal weights
                // when those metadata fields are genuinely unavailable.
                let parsed = ensemble_from_pdb(
                    pdb,
                    metadata.as_ref(),
                    query.clone(),
                    format!("browser:{asset_name}"),
                )?;
                ensemble_cache.insert(asset_name.clone(), parsed.clone());
                parsed
            };
            // Glycan-specific terminal-sugar priors are persisted by the web
            // client beside each PDB asset. Missing metadata keeps the
            // residue-level compatibility fallback for old/offline jobs.
            let terminal_sugar = metadata.as_ref().and_then(glycan_terminal_sugar);
            let residue_name = protein
                .residues()
                .into_iter()
                .find(|residue| residue.id == assignment.site.residue_id())
                .map(|residue| residue.name)
                .unwrap_or_else(|| assignment.residue_name.clone());
            let prior = linkage_priors_for_glycan(&residue_name, terminal_sugar.as_deref());
            for conformer in &mut ensemble.conformers {
                // GlycoShape metadata may carry a conformer-specific
                // attachment distribution. Preserve it; use the residue /
                // terminal-sugar table only for assets that do not provide
                // one. This keeps whole-conformer correlations intact while
                // retaining the historical fallback for old assets.
                if conformer.priors.phi.is_empty() || conformer.priors.psi.is_empty() {
                    conformer.priors = prior.clone();
                }
            }
            Ok(SearchSite {
                site: GlycosylationSite {
                    residue: assignment.site.residue_id(),
                },
                ensemble,
            })
        })
        .collect()
}

fn progress_search(event: SearchProgress) -> ProgressEvent {
    match event {
        SearchProgress::PreparingTopology => ProgressEvent {
            stage: "search".into(),
            message: "Preparing energy topology…".into(),
            current: None,
            total: None,
            fraction: None,
        },
        SearchProgress::TopologyReady { atoms, .. } => ProgressEvent {
            stage: "search".into(),
            message: format!("Energy topology ready ({atoms} atoms)."),
            current: None,
            total: None,
            fraction: None,
        },
        SearchProgress::Generation {
            phase, generation, ..
        } => ProgressEvent {
            stage: "search".into(),
            message: format!(
                "{} · generation {generation}",
                match phase {
                    SearchPhase::Feasibility => "Finding a clash-free pose",
                    SearchPhase::Compatibility => "Resolving neighboring glycans",
                    SearchPhase::ProbabilityImprovement => "Improving conformer probability",
                }
            ),
            current: Some(generation),
            total: None,
            fraction: None,
        },
    }
}

fn progress_relax(event: RelaxProgress) -> ProgressEvent {
    match event {
        RelaxProgress::InitialEnergyStarted => ProgressEvent {
            stage: "relax".into(),
            message: "Calculating initial energy…".into(),
            current: None,
            total: None,
            fraction: None,
        },
        RelaxProgress::InitialEnergy { .. } => ProgressEvent {
            stage: "relax".into(),
            message: "Initial energy calculated.".into(),
            current: None,
            total: None,
            fraction: None,
        },
        RelaxProgress::StageStarted { name, .. } => ProgressEvent {
            stage: "relax".into(),
            message: format!("Relaxing {name}…"),
            current: None,
            total: None,
            fraction: None,
        },
        RelaxProgress::Iteration {
            stage, iteration, ..
        } => ProgressEvent {
            stage: "relax".into(),
            message: format!("Relaxing {stage} · iteration {iteration}"),
            current: Some(iteration),
            total: None,
            fraction: None,
        },
        RelaxProgress::StageFinished { name, .. } => ProgressEvent {
            stage: "relax".into(),
            message: format!("Finished {name}."),
            current: None,
            total: None,
            fraction: None,
        },
    }
}

/// Compare the completed structure with the original input using stable
/// finding keys.  This keeps inherited depositional issues separate from
/// chemistry introduced by a build/minimize/optimize stage and makes the
/// distinction reproducible in report.json and the browser report.
fn validation_summary_with_baseline(
    request: &ReGlycoRunRequestV1,
    input: &str,
    structure: &Structure,
) -> ValidationSummary {
    let mut final_native = validate(structure);
    // Validation findings are attached to the accepted structure that was
    // just produced.  Older validator reports leave stage unset because they
    // are also used for stand-alone inspection; fill it here at the workflow
    // boundary so lineage and the browser report can distinguish input,
    // Build, Minimize, Optimize, and Ensemble observations consistently.
    let accepted_stage = workflow_stage(request.workflow).to_string();
    for finding in &mut final_native.findings {
        if finding.stage.is_none() {
            finding.stage = Some(accepted_stage.clone());
        }
    }
    let baseline = read_pdb_str(input, &build_options(request))
        .ok()
        .map(|value| validate(&value));
    let Some(baseline) = baseline else {
        return validation_summary_from_native(final_native);
    };
    let baseline_keys = baseline
        .findings
        .iter()
        .map(finding_key)
        .collect::<BTreeSet<_>>();
    let final_keys = final_native
        .findings
        .iter()
        .map(finding_key)
        .collect::<BTreeSet<_>>();
    let mut findings = final_native.findings;
    for finding in &mut findings {
        let key = finding_key(finding);
        finding.origin = Some(
            if baseline_keys.contains(&key) {
                "inherited from input"
            } else {
                "introduced"
            }
            .into(),
        );
    }
    // Keep a compact, machine-readable record for input findings that no
    // longer occur after the workflow.  The informational finding is useful
    // for lineage without changing report validity.
    for old in baseline.findings {
        let key = finding_key(&old);
        if final_keys.contains(&key) {
            continue;
        }
        findings.push(NativeFinding {
            code: "validation.resolved".into(),
            severity: Severity::Info,
            message: format!("{} resolved in the completed structure", old.code),
            site: old.site,
            residue: old.residue,
            atom: old.atom,
            observed: old.observed,
            expected: old.expected,
            domain: old.domain,
            origin: Some("resolved".into()),
            stage: Some("input/parent".into()),
            frame: old.frame,
            glycan_index: old.glycan_index,
            glycan: old.glycan,
            linkage: old.linkage,
            involved_atoms: old.involved_atoms,
            metric: old.metric,
            policy: old.policy,
            policy_version: old.policy_version,
        });
    }
    findings.sort_by(|left, right| {
        severity_rank(left.severity)
            .cmp(&severity_rank(right.severity))
            .then_with(|| left.code.cmp(&right.code))
    });
    let warnings = findings
        .iter()
        .filter(|finding| finding.severity == Severity::Warning)
        .map(|finding| finding.message.clone())
        .collect();
    let errors = findings
        .iter()
        .filter(|finding| finding.severity == Severity::Error)
        .map(|finding| finding.message.clone())
        .collect();
    let valid = !findings
        .iter()
        .any(|finding| finding.severity == Severity::Error);
    ValidationSummary {
        valid: valid && !structure.atoms().is_empty(),
        findings: findings.into_iter().map(convert_finding).collect(),
        warnings,
        errors,
        component_dictionary_version: Some(final_native.component_dictionary_version),
        steric_policy: final_native.steric_policy,
        steric_summary: final_native.steric_summary,
    }
}

fn validation_summary_from_native(native: reglyco_validate::ValidationReport) -> ValidationSummary {
    ValidationSummary {
        valid: native.valid,
        findings: native.findings.into_iter().map(convert_finding).collect(),
        warnings: native.warnings,
        errors: native.errors,
        component_dictionary_version: Some(native.component_dictionary_version),
        steric_policy: native.steric_policy,
        steric_summary: native.steric_summary,
    }
}

/// Keep source GlycoShape diagnostics alongside the final structure without
/// confusing them with newly introduced chemistry. Search already rejects a
/// malformed source asset before assembly; this pass retains non-fatal source
/// findings (for example an advisory bond geometry) in the report and marks
/// their provenance for offline inspection.
fn append_asset_validation(
    validation: &mut ValidationSummary,
    analysis: &mut Value,
    artifacts: &[WorkflowArtifact],
    request: &ReGlycoRunRequestV1,
) {
    if !matches!(
        request.workflow,
        WorkflowId::Uniprot
            | WorkflowId::SiteBuild
            | WorkflowId::Ensemble
            | WorkflowId::Relax
            | WorkflowId::Refine
    ) {
        return;
    }
    let mut seen = BTreeSet::new();
    let mut asset_reports = Vec::new();
    for artifact in artifacts {
        let lower = artifact.name.to_ascii_lowercase();
        if !matches!(artifact.role, ArtifactRole::Input)
            || !lower.ends_with(".pdb")
            || !seen.insert(artifact.name.clone())
        {
            continue;
        }
        let Ok(text) = artifact.data.text(&artifact.name) else {
            continue;
        };
        let Ok(structure) = read_pdb_str(text, &build_options(request)) else {
            continue;
        };
        let report = validate(&structure);
        let mut report_value = serde_json::to_value(&report).unwrap_or(Value::Null);
        if let Some(object) = report_value.as_object_mut() {
            object.insert("asset".into(), Value::String(artifact.name.clone()));
            object.insert(
                "origin".into(),
                Value::String("inherited from glycan asset".into()),
            );
        }
        asset_reports.push(report_value);
        for finding in report.findings {
            let severity = finding.severity;
            let mut converted = convert_finding(finding);
            converted.origin = Some("inherited from glycan asset".into());
            converted.stage = Some("glycan asset".into());
            converted.policy = converted
                .policy
                .or_else(|| Some("glycan-asset-validation-v1".into()));
            if severity == Severity::Warning {
                validation.warnings.push(converted.message.clone());
            } else if severity == Severity::Error {
                validation.errors.push(converted.message.clone());
                validation.valid = false;
            }
            validation.findings.push(converted);
        }
    }
    if let Some(object) = analysis.as_object_mut() {
        if !asset_reports.is_empty() {
            object.insert("assetValidation".into(), Value::Array(asset_reports));
        }
    }
}

/// Attachment VMM compliance is a hard gate only for strict steric
/// workflows. Generic energy/interaction paths still expose measured
/// torsions as diagnostics without turning a non-strict result into a failed
/// workflow.
fn append_attachment_gate_findings(
    validation: &mut ValidationSummary,
    analysis: &Value,
    request: &ReGlycoRunRequestV1,
) {
    if !matches!(
        request.workflow,
        WorkflowId::Uniprot | WorkflowId::SiteBuild | WorkflowId::Ensemble
    ) || request.options.scoring_mode != SearchScoringMode::StericPrior
    {
        return;
    }
    let Some(observations) = analysis
        .get("attachmentObservations")
        .or_else(|| analysis.get("attachment_observations"))
        .and_then(Value::as_array)
    else {
        return;
    };
    for observation in observations {
        let Some(object) = observation.as_object() else {
            continue;
        };
        let assessment = object
            .get("assessment")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let phi_within = object
            .get("phiWithinVmm95")
            .or_else(|| object.get("phi_within_vmm95"))
            .and_then(Value::as_bool);
        let psi_within = object
            .get("psiWithinVmm95")
            .or_else(|| object.get("psi_within_vmm95"))
            .and_then(Value::as_bool);
        let has_component = object
            .get("selectedPhiComponent")
            .or_else(|| object.get("selected_phi_component"))
            .is_some_and(Value::is_number)
            || object
                .get("selectedPsiComponent")
                .or_else(|| object.get("selected_psi_component"))
                .is_some_and(Value::is_number);
        // New observations carry explicit per-axis gate results.  Keep the
        // assessment fallback for old reports, but never turn a missing
        // reference into an error.
        let outside_gate = phi_within == Some(false)
            || psi_within == Some(false)
            || (phi_within.is_none() && psi_within.is_none() && assessment == "outlier");
        if !outside_gate || !has_component {
            continue;
        }
        let site_text = object
            .get("site")
            .map(residue_to_string)
            .unwrap_or_default();
        let site = parse_site_key(&site_text);
        let phi = object
            .get("phiDegrees")
            .or_else(|| object.get("phi_degrees"))
            .and_then(Value::as_f64);
        let psi = object
            .get("psiDegrees")
            .or_else(|| object.get("psi_degrees"))
            .and_then(Value::as_f64);
        let axes = match (phi_within == Some(false), psi_within == Some(false)) {
            (true, true) => "φ and ψ",
            (true, false) => "φ",
            (false, true) => "ψ",
            (false, false) => "φ/ψ",
        };
        let message = format!(
            "attachment {axes} torsion at {} lies outside the selected VMM circular 95% bounds",
            if site_text.is_empty() {
                "unknown site"
            } else {
                &site_text
            }
        );
        validation.errors.push(message.clone());
        validation.findings.push(ValidationFinding {
            code: "torsion.attachment_vmm_outlier".into(),
            severity: FindingSeverity::Error,
            message,
            site,
            observed: phi.or(psi),
            expected: Some("selected component-conditioned circular 95% bounds".into()),
            domain: Some("torsion".into()),
            origin: Some("introduced".into()),
            stage: Some(workflow_stage(request.workflow).into()),
            frame: object
                .get("frame")
                .and_then(Value::as_u64)
                .map(|value| value as usize),
            glycan_index: object
                .get("glycanIndex")
                .or_else(|| object.get("glycan_index"))
                .and_then(Value::as_u64)
                .map(|value| value as usize),
            glycan: object
                .get("glycan")
                .or_else(|| object.get("glycanId"))
                .or_else(|| object.get("glycan_id"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            linkage: object
                .get("linkage")
                .and_then(Value::as_str)
                .map(str::to_owned),
            involved_atoms: object
                .get("involvedAtoms")
                .or_else(|| object.get("involved_atoms"))
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            metric: Some("attachment_vmm_95_gate".into()),
            policy: Some("glycoshape-attachment-vmm-v1".into()),
            policy_version: Some("glycoshape-attachment-vmm-v1".into()),
        });
        validation.valid = false;
    }
}

/// Promote the strict solver's per-site diagnostics into the common
/// validation contract.  These findings are deliberately marked as
/// introduced: the candidate was generated during this run and is not an
/// inherited input or source-asset issue.
fn append_strict_diagnostic_findings(
    validation: &mut ValidationSummary,
    diagnostics: &StrictSearchDiagnostics,
) {
    let mut push = |finding: ValidationFinding| {
        match finding.severity {
            FindingSeverity::Warning => validation.warnings.push(finding.message.clone()),
            FindingSeverity::Error => {
                validation.errors.push(finding.message.clone());
                validation.valid = false;
            }
            FindingSeverity::Info => {}
        }
        validation.findings.push(finding);
    };
    for site in &diagnostics.sites {
        if site.steric_score > 1.1 {
            push(ValidationFinding {
                code: "geometry.search_steric_score".into(),
                severity: FindingSeverity::Error,
                message: format!(
                    "strict search candidate retains steric score {:.3} at {}",
                    site.steric_score, site.site
                ),
                site: Some(SiteKey {
                    model: 1,
                    chain: site.site.chain.clone(),
                    residue_number: site.site.number,
                    insertion_code: site.site.insertion_code,
                }),
                observed: Some(site.steric_score),
                expected: Some("<= 1.1 steric score".into()),
                domain: Some("sterics".into()),
                origin: Some("introduced".into()),
                stage: Some("strict search".into()),
                frame: None,
                glycan_index: None,
                glycan: None,
                linkage: None,
                involved_atoms: Vec::new(),
                metric: Some("prepared_steric_score".into()),
                policy: Some("cookbook-steric-v1".into()),
                policy_version: Some("cookbook-steric-v1".into()),
            });
        }
        if !site.phi_within_95 {
            push(ValidationFinding {
                code: "torsion.search_phi_vmm_outlier".into(),
                severity: FindingSeverity::Error,
                message: format!(
                    "strict search candidate φ is outside the selected VMM 95% bounds at {}",
                    site.site
                ),
                site: Some(SiteKey {
                    model: 1,
                    chain: site.site.chain.clone(),
                    residue_number: site.site.number,
                    insertion_code: site.site.insertion_code,
                }),
                observed: Some(site.phi_degrees),
                expected: Some(format!(
                    "component {} circular 95% interval",
                    site.phi_component
                )),
                domain: Some("torsion".into()),
                origin: Some("introduced".into()),
                stage: Some("strict search".into()),
                frame: None,
                glycan_index: None,
                glycan: None,
                linkage: None,
                involved_atoms: Vec::new(),
                metric: Some("attachment_phi_vmm_95".into()),
                policy: Some("glycoshape-attachment-vmm-v1".into()),
                policy_version: Some("glycoshape-attachment-vmm-v1".into()),
            });
        }
        if !site.psi_within_95 {
            push(ValidationFinding {
                code: "torsion.search_psi_vmm_outlier".into(),
                severity: FindingSeverity::Error,
                message: format!(
                    "strict search candidate ψ is outside the selected VMM 95% bounds at {}",
                    site.site
                ),
                site: Some(SiteKey {
                    model: 1,
                    chain: site.site.chain.clone(),
                    residue_number: site.site.number,
                    insertion_code: site.site.insertion_code,
                }),
                observed: Some(site.psi_degrees),
                expected: Some(format!(
                    "component {} circular 95% interval",
                    site.psi_component
                )),
                domain: Some("torsion".into()),
                origin: Some("introduced".into()),
                stage: Some("strict search".into()),
                frame: None,
                glycan_index: None,
                glycan: None,
                linkage: None,
                involved_atoms: Vec::new(),
                metric: Some("attachment_psi_vmm_95".into()),
                policy: Some("glycoshape-attachment-vmm-v1".into()),
                policy_version: Some("glycoshape-attachment-vmm-v1".into()),
            });
        }
    }
    if diagnostics.vdw_hard_contacts > 0 {
        push(ValidationFinding {
            code: "geometry.search_vdw_hard_overlap".into(),
            severity: FindingSeverity::Error,
            message: format!(
                "strict search candidate retains {} hard van der Waals overlap contact{}",
                diagnostics.vdw_hard_contacts,
                if diagnostics.vdw_hard_contacts == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            site: None,
            observed: Some(diagnostics.vdw_max_overlap_angstrom),
            expected: Some("hard overlap <= 0.6 Å".into()),
            domain: Some("sterics".into()),
            origin: Some("introduced".into()),
            stage: Some("strict search".into()),
            frame: None,
            glycan_index: None,
            glycan: None,
            linkage: None,
            involved_atoms: diagnostics.vdw_contacts.clone(),
            metric: Some("vdw_overlap_angstrom".into()),
            policy: Some("heavy_atom_vdw_overlap_v1".into()),
            policy_version: Some("heavy_atom_vdw_overlap_v1".into()),
        });
        for residue in &diagnostics.vdw_outlier_sites {
            push(ValidationFinding {
                code: "geometry.search_vdw_outlier_site".into(),
                severity: FindingSeverity::Error,
                message: format!(
                    "strict search candidate has a hard van der Waals contact involving {}",
                    residue
                ),
                site: Some(SiteKey {
                    model: 1,
                    chain: residue.chain.clone(),
                    residue_number: residue.number,
                    insertion_code: residue.insertion_code,
                }),
                observed: Some(diagnostics.vdw_max_overlap_angstrom),
                expected: Some("hard overlap <= 0.6 Å".into()),
                domain: Some("sterics".into()),
                origin: Some("introduced".into()),
                stage: Some("strict search".into()),
                frame: None,
                glycan_index: None,
                glycan: None,
                linkage: None,
                involved_atoms: diagnostics.vdw_contacts.clone(),
                metric: Some("vdw_overlap_angstrom".into()),
                policy: Some("heavy_atom_vdw_overlap_v1".into()),
                policy_version: Some("heavy_atom_vdw_overlap_v1".into()),
            });
        }
    }
}

fn parse_site_key(value: &str) -> Option<SiteKey> {
    let (chain, residue) = value.split_once(':')?;
    let mut digits = residue.trim();
    let mut insertion_code = None;
    if let Some(last) = digits.chars().last()
        && last.is_ascii_alphabetic()
    {
        insertion_code = Some(last);
        digits = &digits[..digits.len().saturating_sub(last.len_utf8())];
    }
    let residue_number = digits.parse::<i32>().ok()?;
    Some(SiteKey {
        model: 1,
        chain: chain.to_string(),
        residue_number,
        insertion_code,
    })
}

fn finding_key(finding: &NativeFinding) -> String {
    let residue = finding
        .site
        .as_ref()
        .or(finding.residue.as_ref())
        .map(ToString::to_string)
        .unwrap_or_default();
    format!(
        "{}|{}|{}|{}|{}|{}",
        finding.code,
        residue,
        finding.atom.as_deref().unwrap_or_default(),
        finding.linkage.as_deref().unwrap_or_default(),
        finding.metric.as_deref().unwrap_or_default(),
        finding.involved_atoms.join(";"),
    )
}

fn severity_rank(severity: Severity) -> u8 {
    match severity {
        Severity::Error => 0,
        Severity::Warning => 1,
        Severity::Info => 2,
    }
}

fn convert_finding(finding: NativeFinding) -> ValidationFinding {
    let residue_for_site = finding.site.clone().or_else(|| finding.residue.clone());
    let glycan = finding
        .glycan
        .clone()
        .or_else(|| finding.residue.as_ref().map(ToString::to_string));
    let policy_version = finding
        .policy_version
        .clone()
        .or_else(|| finding.policy.clone());
    ValidationFinding {
        code: finding.code,
        severity: match finding.severity {
            Severity::Info => FindingSeverity::Info,
            Severity::Warning => FindingSeverity::Warning,
            Severity::Error => FindingSeverity::Error,
        },
        message: finding.message,
        site: residue_for_site.map(|residue| SiteKey {
            model: 1,
            chain: residue.chain,
            residue_number: residue.number,
            insertion_code: residue.insertion_code,
        }),
        observed: finding.observed,
        expected: finding.expected,
        domain: finding.domain,
        origin: finding.origin,
        stage: finding.stage,
        frame: finding.frame,
        glycan_index: finding.glycan_index,
        glycan,
        linkage: finding.linkage,
        involved_atoms: finding.involved_atoms,
        metric: finding.metric,
        policy_version,
        policy: finding.policy,
    }
}

fn multi_model_pdb(structures: &[Structure]) -> String {
    let mut output = String::new();
    for (index, structure) in structures.iter().enumerate() {
        output.push_str(&format!("MODEL     {:>4}\n", index + 1));
        for line in structure
            .to_pdb_string()
            .lines()
            .filter(|line| !line.starts_with("END"))
        {
            output.push_str(line);
            output.push('\n');
        }
        output.push_str("ENDMDL\n");
    }
    output.push_str("END\n");
    output
}

fn text_artifact(
    name: &str,
    media_type: &str,
    role: ArtifactRole,
    data: impl Into<String>,
) -> WorkflowArtifact {
    WorkflowArtifact {
        name: name.into(),
        media_type: media_type.into(),
        role,
        data: AssetData::Text(data.into()),
    }
}

fn input_support_artifacts(assets: &InputAssets, primary_input: &str) -> Vec<WorkflowArtifact> {
    assets
        .iter()
        .filter(|(name, _)| name.as_str() != primary_input)
        .map(|(name, data)| {
            let media_type = if name.ends_with(".pdb") {
                "chemical/x-pdb"
            } else if name.ends_with(".dat") {
                "text/plain"
            } else {
                "application/octet-stream"
            };
            WorkflowArtifact {
                name: name.clone(),
                media_type: media_type.into(),
                role: ArtifactRole::Input,
                data: data.clone(),
            }
        })
        .collect()
}

fn is_transient_glycoshape_source(name: &str) -> bool {
    let normalized = name.replace('\\', "/").to_ascii_lowercase();
    let normalized = normalized.trim_start_matches("./");
    let basename = normalized.rsplit('/').next().unwrap_or(normalized);
    normalized.starts_with("glycans/")
        || basename == "scan-glycan.pdb"
        || basename == "torsion-reference.json"
        || basename.ends_with("-reference.json")
        || basename.ends_with("_reference.json")
}

/// User-owned inputs remain reproducible artifacts, while GlycoShape source
/// conformers and reference grids are deliberately execution-only.
fn exportable_input_support_artifacts(
    assets: &InputAssets,
    primary_input: &str,
) -> Vec<WorkflowArtifact> {
    input_support_artifacts(assets, primary_input)
        .into_iter()
        .filter(|artifact| !is_transient_glycoshape_source(&artifact.name))
        .collect()
}

fn report_title(workflow: WorkflowId) -> &'static str {
    match workflow {
        WorkflowId::Uniprot => "UniProt one-shot glycosylation",
        WorkflowId::NScan => "N-glycosylation GlcNAc scan",
        WorkflowId::SiteBuild => "Site-by-site glycan build",
        WorkflowId::Ensemble => "Glycoprotein ensemble",
        WorkflowId::Relax => "Glycoprotein relaxation",
        WorkflowId::Validate => "Glycan torsion analysis",
        WorkflowId::Refine => "Objective-driven refinement",
        WorkflowId::Density => "Density-guided refinement",
        WorkflowId::Saxs => "SAXS fitting",
    }
}

pub fn execute(request: &ReGlycoRunRequestV1, assets: &InputAssets) -> Result<WorkflowBundle> {
    execute_with_control(request, assets, &mut NoopControl)
}

/// Calculate report-only torsion observations after a workflow output has
/// already been returned. This function is intentionally independent of
/// search acceptance: malformed or unavailable reference data can make the
/// analysis fail, but can never invalidate or suppress the output structure.
pub fn analyze_torsions(
    request: &ReGlycoRunRequestV1,
    assets: &InputAssets,
    output_pdb: &str,
    mut analysis: Value,
) -> Result<Value> {
    if !analysis.is_object() {
        analysis = json!({ "workflow": analysis });
    }
    if let Some(object) = analysis.as_object_mut() {
        object.remove("torsionObservations");
        object.remove("ensembleTorsion");
        object.remove("clusterDistributions");
    }
    let support = input_support_artifacts(assets, &request.input.asset);
    append_torsion_reference_assets(&mut analysis, &support);

    if let Some(input) = assets
        .get(&request.input.asset)
        .and_then(|asset| asset.text(&request.input.asset).ok())
        .and_then(|pdb| read_pdb_str(pdb, &build_options(request)).ok())
    {
        append_torsion_analysis(&mut analysis, &input, "input/parent");
    }

    let models = split_pdb_models(output_pdb);
    let stage = final_structure_stage(request, &analysis);
    let has_attachment_references = analysis
        .get("attachmentReferences")
        .or_else(|| analysis.get("attachment_references"))
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty());
    if !has_attachment_references {
        // Saved browser reports retain measured observations but deliberately
        // omit private GlycoShape reference grids. Rebuild the small VMM
        // description from newly fetched, in-memory source assets; this does
        // not run search or alter the accepted coordinates.
        if let Some(model) = models.first() {
            if let Ok(structure) = read_pdb_str(model, &build_options(request)) {
                if let Ok(sites) = search_sites(request, assets, &structure) {
                    append_attachment_reference_data_from_observations(&mut analysis, &sites);
                }
            }
        }
    }
    let torsion_references = analysis
        .get("torsionReferences")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // Older/development-era reports can contain the compact search `sites`
    // records without the derived attachment observation array.  Rehydrate
    // that inexpensive adapter before calculating output torsions so a build
    // still exposes its current-model attachment point and selected VMM
    // components.  This is report enrichment only; it never participates in
    // search acceptance.
    let has_attachment_observations = analysis
        .get("attachmentObservations")
        .or_else(|| analysis.get("attachment_observations"))
        .and_then(Value::as_array)
        .is_some_and(|values| !values.is_empty());
    if !has_attachment_observations {
        append_attachment_analysis(&mut analysis, &stage);
    }
    let ensemble = models.len() > 1;
    let mut observations = Vec::new();
    for (index, model) in models.iter().enumerate() {
        let structure = read_pdb_str(model, &build_options(request))?;
        observations.extend(glycosidic_torsion_observations(
            &structure,
            stage,
            ensemble.then_some(index),
            &torsion_references,
        ));
    }
    if let Some(object) = analysis.as_object_mut() {
        object
            .entry("torsionObservations")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(items) = object
            .get_mut("torsionObservations")
            .and_then(Value::as_array_mut)
        {
            items.extend(observations);
        }
    }
    annotate_torsion_observations(&mut analysis);
    append_ensemble_torsion_summary(&mut analysis);
    append_glycan_cluster_distributions(&mut analysis, request);
    append_cluster_distributions(&mut analysis);
    Ok(analysis)
}

fn split_pdb_models(pdb: &str) -> Vec<String> {
    if !pdb.lines().any(|line| line.starts_with("MODEL")) {
        return vec![pdb.to_string()];
    }
    let mut models = Vec::new();
    let mut current = String::new();
    let mut inside = false;
    for line in pdb.lines() {
        if line.starts_with("MODEL") {
            current.clear();
            inside = true;
            continue;
        }
        if line.starts_with("ENDMDL") {
            if inside && !current.trim().is_empty() {
                current.push_str("END\n");
                models.push(current.clone());
            }
            inside = false;
            continue;
        }
        if inside {
            current.push_str(line);
            current.push('\n');
        }
    }
    if inside && !current.trim().is_empty() {
        current.push_str("END\n");
        models.push(current);
    }
    models
}

#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        search_with_progress_cancelled(sync, async = "search_with_progress_cancelled_async"),
        sample_attached_ensemble_with_progress_cancel(
            sync,
            async = "sample_attached_ensemble_with_progress_cancel_async"
        )
    )
)]
pub async fn execute_with_control(
    request: &ReGlycoRunRequestV1,
    assets: &InputAssets,
    control: &mut impl WorkflowControl,
) -> Result<WorkflowBundle> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(WorkflowError::Invalid(format!(
            "schema {} is not supported",
            request.schema_version
        )));
    }
    validate_options(request)?;
    if request.assignments.iter().any(|assignment| {
        !assignment.excluded && (assignment.unresolved || assignment.glycan_id.trim().is_empty())
    }) {
        return Err(WorkflowError::Invalid(
            "every included site needs an explicit glycan mapping".into(),
        ));
    }
    let input = input_text(request, assets)?;
    let options = build_options(request);
    let builder = SystemBuilder::new(options.clone())?;
    emit(control, "parse", "Reading the input structure…", None, None)?;
    let parsed = read_pdb_str(input, &options)?;
    let (parsed, replacements) = prepare_replacements(parsed, request)?;
    let mut protein = if workflow_converts_proline(request.workflow) {
        prepare_proline_sites(parsed, &request.assignments, &options)?
    } else {
        parsed
    };
    let mut warnings = Vec::new();
    let mut status = WorkflowStatus::Succeeded;
    let mut workflow_sites = Vec::new();
    let mut analysis = if replacements.is_empty() {
        json!({})
    } else {
        json!({ "replacements": serde_json::to_value(&replacements)? })
    };
    let mut extra_artifacts = exportable_input_support_artifacts(assets, &request.input.asset);

    match request.workflow {
        WorkflowId::Validate => {
            emit(
                control,
                "validate",
                "Preparing the inspected structure for torsion analysis…",
                None,
                None,
            )?;
        }
        WorkflowId::Relax => {
            emit(
                control,
                "parameterize",
                "Preparing the glycoprotein force field…",
                None,
                None,
            )?;
            let models = parse_pdb_models(input, &options)?;
            if models.len() > 1 {
                let mut relaxed_frames = Vec::with_capacity(models.len());
                let mut diagnostics = Vec::with_capacity(models.len());
                for (index, model) in models.iter().enumerate() {
                    emit(
                        control,
                        "relax",
                        "Relaxing ensemble frames…",
                        Some(index + 1),
                        Some(models.len()),
                    )?;
                    let model =
                        prepare_proline_sites(model.clone(), &request.assignments, &options)?;
                    let system = builder.prepare_structure(&model)?;
                    let relaxation =
                        relax_with_progress(&model, &system, &relax_options(request), |event| {
                            control.progress(progress_relax(event))
                        })?;
                    diagnostics.push(relaxation.diagnostics);
                    relaxed_frames.push(relaxation.structure);
                }
                protein = relaxed_frames
                    .first()
                    .cloned()
                    .ok_or_else(|| WorkflowError::Invalid("ensemble contains no frames".into()))?;
                extra_artifacts.push(text_artifact(
                    "ensemble.pdb",
                    "chemical/x-pdb",
                    ArtifactRole::Structure,
                    multi_model_pdb(&relaxed_frames),
                ));
                analysis = json!({ "frames": relaxed_frames.len(), "diagnostics": diagnostics });
            } else {
                let system = builder.prepare_structure(&protein)?;
                let relaxation =
                    relax_with_progress(&protein, &system, &relax_options(request), |event| {
                        control.progress(progress_relax(event))
                    })?;
                analysis = serde_json::to_value(&relaxation.diagnostics)?;
                protein = relaxation.structure;
            }
        }
        WorkflowId::Uniprot | WorkflowId::SiteBuild => {
            let sites = search_sites(request, assets, &protein)?;
            let search_budget = resolved_attachment_budget(request, &protein, &sites)?;
            emit(
                control,
                "search_budget_resolved",
                format!(
                    "Resolved {} search budget: {} candidates × {} generations{}",
                    match search_budget.requested_mode {
                        SearchBudgetMode::Auto => "Auto",
                        SearchBudgetMode::Manual => "manual",
                    },
                    search_budget.population_size,
                    search_budget.generations,
                    if search_budget.capped {
                        " (Auto ceiling reached)"
                    } else {
                        ""
                    }
                ),
                None,
                None,
            )?;
            let config = attachment_search_config_with_budget(request, &search_budget);
            emit(
                control,
                "search",
                "Searching compatible GlycoShape conformers…",
                None,
                None,
            )?;
            let cancelled = std::cell::Cell::new(control.cancelled());
            let outcome = match search_with_progress_cancelled(
                &protein,
                &sites,
                &config,
                &builder,
                |event| {
                    control.progress(progress_search(event));
                    cancelled.set(control.cancelled());
                },
                || cancelled.get(),
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(EnsembleError::StrictVmmFailure { diagnostics }) => {
                    return finish_strict_failure(
                        request,
                        input,
                        protein.clone(),
                        warnings,
                        assignment_report_sites(request, "strict-search-failed"),
                        *diagnostics,
                        &replacements,
                        extra_artifacts,
                        control,
                    );
                }
                Err(EnsembleError::Cancelled) => return Err(WorkflowError::Cancelled),
                Err(error) => return Err(error.into()),
            };
            if outcome.clash_status == ClashStatus::BestCompleteClashing
                || !outcome.vmm_gate_satisfied
            {
                status = WorkflowStatus::Partial;
                if outcome.warnings.is_empty() {
                    warnings.push(if outcome.clash_status == ClashStatus::BestCompleteClashing {
                        "The requested search budget ended with the best complete structure still showing steric clashes; the structure and diagnostics are available for review.".into()
                    } else {
                        "The complete structure is outside the requested VMM acceptance gate; the structure and diagnostics are available for review.".into()
                    });
                }
            }
            // Preserve the engine's structured steric diagnostics, including
            // the advisory VDW band, in the workflow-level report.  Without
            // this adapter the search could correctly classify a result but
            // the browser would silently discard the explanation.
            warnings.extend(outcome.warnings.clone());
            if sites.iter().any(|site| {
                matches!(
                    site.ensemble.population_source,
                    ConformerPopulationSource::EqualFallback
                )
            }) {
                warnings.push(
                    "Conformer population metadata was unavailable for at least one asset; equal conformer weights were used for probability ranking.".into(),
                );
            }
            let built = build_from_outcome(
                &protein,
                &sites,
                &outcome,
                &builder,
                request.options.post_relax,
            )?;
            workflow_sites = request
                .assignments
                .iter()
                .filter(|assignment| !assignment.excluded)
                .map(|assignment| ReportSite {
                    site: assignment.site.clone(),
                    glycan_id: Some(assignment.glycan_id.clone()),
                    status: outcome
                        .sites
                        .iter()
                        .find(|result| result.site.residue == assignment.site.residue_id())
                        .map(|result| {
                            if result.steric_score <= 1.1
                                && result.phi_within_vmm95.unwrap_or(true)
                                && result.psi_within_vmm95.unwrap_or(true)
                            {
                                "built"
                            } else {
                                "unresolved"
                            }
                        })
                        .unwrap_or("missing")
                        .into(),
                    score: None,
                    details: outcome
                        .sites
                        .iter()
                        .find(|result| result.site.residue == assignment.site.residue_id())
                        .map(|result| {
                            let result_index = outcome.sites.iter().position(|candidate| {
                                candidate.site.residue == result.site.residue
                            });
                            BTreeMap::from([
                                ("stericScore".into(), json!(result.steric_score)),
                                ("phiWithinVmm95".into(), json!(result.phi_within_vmm95)),
                                ("psiWithinVmm95".into(), json!(result.psi_within_vmm95)),
                                ("conformerId".into(), json!(result.conformer_id)),
                                (
                                    "clashPartners".into(),
                                    json!(
                                        result_index
                                            .and_then(|index| outcome.clash_partners.get(index))
                                            .cloned()
                                            .unwrap_or_default()
                                    ),
                                ),
                            ])
                        })
                        .unwrap_or_default(),
                })
                .collect();
            analysis = serde_json::to_value(&outcome)?;
            analysis["searchBudget"] = serde_json::to_value(&search_budget)?;
            append_build_search_diagnostics(
                &mut analysis,
                &outcome,
                &config,
                &sites,
                &outcome.clash_partners,
            );
            append_attachment_reference_data(&mut analysis, &protein, &sites, &outcome.sites);
            protein = built.structure;
            append_attachment_observations_from_results(
                &mut analysis,
                &outcome.sites,
                "Build",
                None,
            );
            if request.options.post_relax {
                if let Some(system) = built.system {
                    match relax_with_progress(&protein, &system, &relax_options(request), |event| {
                        control.progress(progress_relax(event))
                    }) {
                        Ok(relaxed) => {
                            extra_artifacts.push(text_artifact(
                                "pre-relax.pdb",
                                "chemical/x-pdb",
                                ArtifactRole::Structure,
                                protein.to_pdb_string(),
                            ));
                            protein = relaxed.structure;
                            analysis["relaxation"] = serde_json::to_value(relaxed.diagnostics)?;
                            append_attachment_analysis(&mut analysis, "Minimize");
                        }
                        Err(error) => {
                            status = WorkflowStatus::Partial;
                            warnings
                                .push(format!("Build succeeded but relaxation failed: {error}"));
                        }
                    }
                }
            }
            // Analyze the coordinates that will actually be exported.  The
            // search outcome may have been scored before a requested final
            // relaxation, so reusing that decomposition would describe a
            // different geometry.
            if config.scoring_mode != SearchScoringMode::StericPrior {
                let backend = match request.options.compute_backend.as_str() {
                    "webgpu" => "GPU",
                    "cpu" => "CPU",
                    _ => analysis
                        .get("energyAnalysis")
                        .and_then(|value| value.get("backend"))
                        .and_then(Value::as_str)
                        .unwrap_or("CPU"),
                };
                match reglyco_ensemble::analyze_energy_structure(
                    &sites,
                    &outcome.sites,
                    &protein,
                    &builder,
                    config.scoring_mode,
                    config.use_obc2,
                    config.energy_cutoff,
                    backend,
                ) {
                    Ok(energy_analysis) => {
                        analysis["energyAnalysis"] = serde_json::to_value(energy_analysis)?;
                    }
                    Err(error) => warnings.push(format!(
                        "Energy contribution analysis was unavailable for the exported build: {error}"
                    )),
                }
            }
        }
        WorkflowId::Ensemble => {
            let sites = search_sites(request, assets, &protein)?;
            let search_budget = resolved_attachment_budget(request, &protein, &sites)?;
            emit(
                control,
                "search_budget_resolved",
                format!(
                    "Resolved {} search budget: {} candidates × {} generations{}",
                    match search_budget.requested_mode {
                        SearchBudgetMode::Auto => "Auto",
                        SearchBudgetMode::Manual => "manual",
                    },
                    search_budget.population_size,
                    search_budget.generations,
                    if search_budget.capped {
                        " (Auto ceiling reached)"
                    } else {
                        ""
                    }
                ),
                None,
                None,
            )?;
            let config = {
                let mut config = ensemble_search_config(request);
                config.population_size = search_budget.population_size;
                config.generations = search_budget.generations;
                config
            };
            emit(
                control,
                "ensemble",
                "Sampling sterically compatible ensemble frames…",
                Some(0),
                Some(request.options.ensemble_frames),
            )?;
            let control_cell = RefCell::new(&mut *control);
            let sampling_result = sample_attached_ensemble_with_progress_cancel(
                &protein,
                &sites,
                request.options.ensemble_frames,
                &config,
                &builder,
                || control_cell.borrow().cancelled(),
                |step, total_steps, frames_complete, frames_requested| {
                    let total = total_steps.max(1);
                    let message = format!(
                        "Sampling {}-site ensemble: step {step}/{total_steps}; {frames_complete}/{frames_requested} frames",
                        sites.len()
                    );
                    control_cell.borrow_mut().progress(ProgressEvent {
                        stage: "ensemble".into(),
                        message,
                        current: Some(step),
                        total: Some(total),
                        fraction: Some(step as f64 / total as f64),
                    });
                },
            )
            .await;
            drop(control_cell);
            let (mut frames, diagnostics) = match sampling_result {
                Ok(result) => result,
                Err(EnsembleError::StrictVmmFailure { diagnostics }) => {
                    return finish_strict_failure(
                        request,
                        input,
                        protein.clone(),
                        warnings,
                        assignment_report_sites(request, "strict-search-failed"),
                        *diagnostics,
                        &replacements,
                        extra_artifacts,
                        control,
                    );
                }
                Err(EnsembleError::Cancelled) => return Err(WorkflowError::Cancelled),
                Err(error) => return Err(error.into()),
            };
            let original = frames
                .iter()
                .map(|frame| frame.structure.clone())
                .collect::<Vec<_>>();
            let mut final_frames = original.clone();
            if request.options.post_relax {
                let frame_template =
                    reglyco_ensemble::prepare_frame_topology(&protein, &sites, &builder)?;
                let coordinate_map = glysys_energy::geometry::CoordinateMap::new(&frame_template);
                let mut relaxed = Vec::with_capacity(original.len());
                for (index, frame) in original.iter().enumerate() {
                    emit(
                        control,
                        "relax",
                        "Relaxing ensemble frames…",
                        Some(index + 1),
                        Some(original.len()),
                    )?;
                    let mut frame_system = frame_template.clone();
                    let coordinates = coordinate_map
                        .coordinates(frame)
                        .map_err(|e| WorkflowError::Invalid(e.to_string()))?;
                    frame_system.set_coordinates(&coordinates)?;
                    match relax_with_progress(frame, &frame_system, &relax_options(request), |_| {})
                    {
                        Ok(mut result) => {
                            result
                                .structure
                                .update_with_parameterized_hydrogens(&result.system)?;
                            relaxed.push(result.structure);
                        }
                        Err(error) => {
                            warnings
                                .push(format!("Frame {} relaxation failed: {error}", index + 1));
                            break;
                        }
                    }
                }
                if relaxed.len() == original.len() {
                    final_frames = relaxed;
                } else {
                    status = WorkflowStatus::Partial;
                    if !relaxed.is_empty() {
                        extra_artifacts.push(text_artifact(
                            "partial-relaxed-ensemble.pdb",
                            "chemical/x-pdb",
                            ArtifactRole::Structure,
                            multi_model_pdb(&relaxed),
                        ));
                    }
                }
                extra_artifacts.push(text_artifact(
                    "pre-relax-ensemble.pdb",
                    "chemical/x-pdb",
                    ArtifactRole::Structure,
                    multi_model_pdb(&original),
                ));
            }
            for (frame, final_structure) in frames.iter_mut().zip(&final_frames) {
                frame.structure = final_structure.clone();
            }
            reglyco_ensemble::refresh_frames(&mut frames, &protein, &sites, &config, &builder)?;
            protein = final_frames
                .first()
                .cloned()
                .ok_or_else(|| WorkflowError::Invalid("ensemble returned no frames".into()))?;
            extra_artifacts.push(text_artifact(
                "ensemble.pdb",
                "chemical/x-pdb",
                ArtifactRole::Structure,
                multi_model_pdb(&final_frames),
            ));
            if request.options.calculate_sasa {
                let sasa = calculate_sasa(final_frames.iter())?;
                extra_artifacts.push(text_artifact(
                    "sasa.pdb",
                    "chemical/x-pdb",
                    ArtifactRole::Analysis,
                    sasa.sasa_pdb_string(&protein)?,
                ));
                extra_artifacts.push(text_artifact(
                    "real_sasa.pdb",
                    "chemical/x-pdb",
                    ArtifactRole::Analysis,
                    sasa.real_sasa_pdb_string(&protein)?,
                ));
                if request.options.calculate_hotspots {
                    extra_artifacts.push(text_artifact(
                        "hotspots.pdb",
                        "chemical/x-pdb",
                        ArtifactRole::Analysis,
                        sasa.hotspots_pdb_string(&protein)?,
                    ));
                }
            }
            workflow_sites = request
                .assignments
                .iter()
                .filter(|assignment| !assignment.excluded)
                .map(|assignment| ReportSite {
                    site: assignment.site.clone(),
                    glycan_id: Some(assignment.glycan_id.clone()),
                    status: "sampled".into(),
                    score: None,
                    details: BTreeMap::new(),
                })
                .collect();
            analysis = serde_json::to_value(&diagnostics)?;
            analysis["searchBudget"] = serde_json::to_value(&search_budget)?;
            // Energy decomposition is a post-sampling diagnostic.  It uses
            // the emitted (and, when requested, relaxed) coordinates and is
            // kept out of the acceptance loop.  Per-frame scalar energies
            // remain available even when the richer first-frame explanation
            // cannot be built.
            if config.scoring_mode != SearchScoringMode::StericPrior {
                let backend = match diagnostics.segments.as_slice() {
                    [] => "CPU",
                    [segment] => segment.backend.as_str(),
                    _ => "Mixed",
                };
                if let (Some(first), Some(first_frame)) = (final_frames.first(), frames.first()) {
                    match reglyco_ensemble::analyze_energy_structure(
                        &sites,
                        &first_frame.sites,
                        first,
                        &builder,
                        config.scoring_mode,
                        config.use_obc2,
                        config.energy_cutoff,
                        backend,
                    ) {
                        Ok(energy_analysis) => {
                            analysis["energyAnalysis"] = serde_json::to_value(energy_analysis)?;
                        }
                        Err(error) => warnings.push(format!(
                            "Energy contribution analysis was unavailable for the ensemble output: {error}"
                        )),
                    }
                }
                analysis["energyAnalysisFrames"] = json!(
                    frames
                        .iter()
                        .enumerate()
                        .map(|(index, frame)| json!({
                            "frame": index,
                            "energyKcalPerMol": frame.selected_energy_kcal_per_mol,
                            "source": frame.source,
                            "multiplicity": 1,
                        }))
                        .collect::<Vec<_>>()
                );
            }
            analysis["ensembleMode"] = json!(config.ensemble_mode.as_deref().unwrap_or(
                if config.pre_minimization {
                    "conformer_collection"
                } else {
                    "sampled"
                }
            ));
            analysis["samplerVersion"] = json!(if analysis["ensembleMode"] == "sampled" {
                match diagnostics.sampling_target {
                    Some(SamplingTarget::WebgpuF32V1) => "gpu-f32-mh-v1",
                    _ => "model-da-mh-v3",
                }
            } else {
                "conformer-collection-v2"
            });
            analysis["samplingTarget"] = serde_json::to_value(diagnostics.sampling_target)?;
            analysis["samplingSegments"] = serde_json::to_value(&diagnostics.segments)?;
            analysis["modelVersion"] = json!("amber-glycam-v2");
            analysis["probabilityConvention"] = json!(
                "normalized circular density per radian; discrete conformer counting measure"
            );
            analysis["target"] = json!(match config.scoring_mode {
                SearchScoringMode::StericPrior =>
                    "native prior conditioned on steric compatibility",
                SearchScoringMode::ProteinGlycanInteraction =>
                    "native prior times exp(-beta * cross interaction energy), conditioned on sterics",
                SearchScoringMode::FullEnergy =>
                    "exp(-beta * full energy) over represented states, conditioned on sterics; native prior is proposal only",
            });
            analysis["burnInSteps"] = json!(
                config
                    .burn_in_steps
                    .unwrap_or(config.mh_burn_in_sweeps.saturating_mul(sites.len()))
            );
            analysis["thinningSteps"] =
                json!(config.thinning_steps.unwrap_or(config.mh_thinning_accepted));
            analysis["residencePolicy"] =
                json!("repeated states retained after rejected proposals");
            if analysis["ensembleMode"] == "conformer_collection" {
                analysis["target"] = json!(
                    "optimized conformer collection; no population or equilibrium interpretation"
                );
                analysis["residencePolicy"] = json!("not a statistical trajectory");
                analysis["burnInSteps"] = serde_json::Value::Null;
                analysis["thinningSteps"] = serde_json::Value::Null;
            }
            analysis["frameRecords"] = json!(frames.iter().enumerate().map(|(index,frame)|json!({
                "frame":index,"source":frame.source,"proposalIndex":frame.proposal_index,
                "energyKcalPerMol":frame.selected_energy_kcal_per_mol,
                "conformationalLogPrior":frame.log_native_probability,
                "stericScores":frame.sites.iter().map(|s|s.steric_score).collect::<Vec<_>>(),
                "energyBackend":diagnostics.segments.first().map(|segment| segment.backend.clone()).unwrap_or_else(|| "CPU".into()),
                "samplingTarget":diagnostics.sampling_target,
                "coordinatePrecision":"energy uses full precision; PDB coordinates round to 0.001 angstrom"
            })).collect::<Vec<_>>());
            if let Some(frame) = frames.first() {
                append_attachment_reference_data(&mut analysis, &protein, &sites, &frame.sites);
            }
            // Keep attachment torsions for every accepted frame.  The
            // ensemble diagnostics intentionally contain the sampler
            // counters but not a transient GA population; these compact site
            // observations are the accepted frame-level records used by the
            // report and by offline history reopening.
            for (frame_index, frame) in frames.iter().enumerate() {
                append_attachment_observations_from_results(
                    &mut analysis,
                    &frame.sites,
                    "Ensemble",
                    Some(frame_index),
                );
            }
        }
        WorkflowId::NScan => {
            emit(
                control,
                "scan",
                "Finding unoccupied N-X-S/T sequons…",
                None,
                None,
            )?;
            let sequons = scan_n_linked_sequons(&protein);
            if let Some(asset) = assets.get("scan-glycan.pdb") {
                let query = GlycanQuery {
                    source: GlycanSource::LocalBundle(PathBuf::from("scan-glycan.pdb")),
                    anomer: Anomer::Beta,
                    format: "pdb".into(),
                    level: "2".into(),
                };
                let ensemble = ensemble_from_pdb(
                    asset.text("scan-glycan.pdb")?,
                    None,
                    query,
                    "browser:scan-glycan",
                )?;
                let scan_sites = sequons
                    .iter()
                    .map(|sequon| SearchSite {
                        site: GlycosylationSite {
                            residue: sequon.asparagine.clone(),
                        },
                        ensemble: ensemble.clone(),
                    })
                    .collect::<Vec<_>>();
                let mut independent = Vec::with_capacity(scan_sites.len());
                let mut compatible: Vec<SearchSite> = Vec::new();
                let mut compatible_results: Vec<SearchSiteResult> = Vec::new();
                let config = scan_search_config(request);
                let scan_started = Instant::now();
                let mut scan_seconds_per_site = Vec::with_capacity(scan_sites.len());
                let mut joint_seconds = 0.0;
                for (index, site) in scan_sites.iter().enumerate() {
                    emit(
                        control,
                        "scan",
                        "Testing GlcNAc structural accessibility…",
                        Some(index + 1),
                        Some(scan_sites.len()),
                    )?;
                    let mut site_config = config.clone();
                    site_config.seed = config.seed.wrapping_add(index as u64);
                    let site_started = Instant::now();
                    let outcome = search_with_progress_cancelled(
                        &protein,
                        std::slice::from_ref(site),
                        &site_config,
                        &builder,
                        |_| {},
                        || control.cancelled(),
                    )
                    .await;
                    scan_seconds_per_site.push(site_started.elapsed().as_secs_f64());
                    let (accessible, score, accepted) = match outcome {
                        Ok(outcome) => (
                            outcome.clash_status == ClashStatus::ClashFree,
                            outcome.sites.first().map(|result| result.steric_score),
                            outcome.sites.into_iter().next(),
                        ),
                        Err(EnsembleError::Cancelled) => return Err(WorkflowError::Cancelled),
                        Err(error) if blocked_scan_outcome(&error) => (false, None, None),
                        Err(error) => return Err(error.into()),
                    };
                    independent.push((accessible, score));
                    if accessible {
                        let Some(accepted) = accepted else {
                            continue;
                        };
                        // Fast path: directly combine the already accepted
                        // per-site results and verify joint sterics without
                        // another full search. This keeps many-site scans
                        // proportional to the site count instead of quadratic
                        // in full searches.
                        let mut trial_sites = compatible.clone();
                        trial_sites.push(site.clone());
                        let mut trial_results = compatible_results.clone();
                        trial_results.push(accepted);
                        if scan_trial_compatible(
                            &protein,
                            &trial_sites,
                            &trial_results,
                            &config,
                            &builder,
                        )? {
                            compatible = trial_sites;
                            compatible_results = trial_results;
                        } else {
                            // Direct combination clashed; spend one bounded
                            // search on the trial set before giving up, so a
                            // different jointly compatible pose set can still
                            // be found.
                            let joint_started = Instant::now();
                            let mut trial_config = config.clone();
                            trial_config.seed = config
                                .seed
                                .wrapping_add(0x9e37_79b9_7f4a_7c15u64.wrapping_add(index as u64));

                            let joint = search_with_progress_cancelled(
                                &protein,
                                &trial_sites,
                                &trial_config,
                                &builder,
                                |_| {},
                                || control.cancelled(),
                            )
                            .await;
                            joint_seconds += joint_started.elapsed().as_secs_f64();
                            match joint {
                                Ok(joint) if joint.clash_status == ClashStatus::ClashFree => {

                                    compatible = trial_sites;
                                    compatible_results = joint.sites;
                                }
                                Err(EnsembleError::Cancelled) => {
                                    return Err(WorkflowError::Cancelled);
                                }
                                Err(error) if blocked_scan_outcome(&error) => {}
                                Err(error) => return Err(error.into()),
                                Ok(_) => {}
                            }
                        }
                    }
                }
                workflow_sites = sequons
                    .iter()
                    .enumerate()
                    .map(|(index, sequon)| {
                        let jointly_compatible = compatible
                            .iter()
                            .any(|site| site.site.residue == sequon.asparagine);
                        let accessible = independent[index].0;
                        ReportSite {
                            site: SiteKey {
                                model: 1,
                                chain: sequon.asparagine.chain.clone(),
                                residue_number: sequon.asparagine.number,
                                insertion_code: sequon.asparagine.insertion_code,
                            },
                            glycan_id: Some("G14843DJ".into()),
                            status: if jointly_compatible {
                                "compatible"
                            } else if accessible {
                                "accessible"
                            } else {
                                "blocked"
                            }
                            .into(),
                            score: independent[index].1,
                            details: BTreeMap::from([
                                ("motif".into(), json!(sequon.motif)),
                                ("context".into(), json!(sequon.context)),
                                ("accessible".into(), json!(accessible)),
                                ("jointlyCompatible".into(), json!(jointly_compatible)),
                            ]),
                        }
                    })
                    .collect();
                analysis = json!({
                    "sequons": sequons,
                    "structuralAccessibilityComputed": true,
                    "independentAccessibleCount": independent.iter().filter(|entry| entry.0).count(),
                    "jointlyCompatibleCount": compatible.len(),
                    "scanBudget": {"populationSize": config.population_size, "generations": config.generations},
                    "timings": {
                        "totalSeconds": scan_started.elapsed().as_secs_f64(),
                        "perSiteSeconds": scan_seconds_per_site,
                        "jointSearchSeconds": joint_seconds,
                    },
                    "interpretation": "Structural accessibility only; not evidence of biological glycosylation."
                });
            } else {
                workflow_sites = sequons
                    .iter()
                    .map(|sequon| ReportSite {
                        site: SiteKey {
                            model: 1,
                            chain: sequon.asparagine.chain.clone(),
                            residue_number: sequon.asparagine.number,
                            insertion_code: sequon.asparagine.insertion_code,
                        },
                        glycan_id: Some("G14843DJ".into()),
                        status: "candidate".into(),
                        score: None,
                        details: BTreeMap::from([
                            ("motif".into(), json!(sequon.motif)),
                            ("context".into(), json!(sequon.context)),
                        ]),
                    })
                    .collect();
                analysis = json!({ "sequons": sequons, "structuralAccessibilityComputed": false });
                warnings.push("Sequons were identified, but the GlcNAc scan ensemble was unavailable; accessibility was not estimated.".into());
            }
        }
        WorkflowId::Refine => {
            #[cfg(feature = "full")]
            if request.profile == ReGlycoProfile::Full {
                return execute_full(request, assets, protein, &builder, &replacements, control);
            }
            #[cfg(feature = "refine")]
            {
                return execute_refine(request, assets, protein, &builder, &replacements, control);
            }
            #[cfg(not(feature = "refine"))]
            {
                return Err(WorkflowError::Capability(request.workflow));
            }
        }
        WorkflowId::Density | WorkflowId::Saxs => {
            #[cfg(feature = "full")]
            {
                if request.profile == ReGlycoProfile::Full {
                    return execute_full(
                        request,
                        assets,
                        protein,
                        &builder,
                        &replacements,
                        control,
                    );
                }
                return Err(WorkflowError::Capability(request.workflow));
            }
            #[cfg(not(feature = "full"))]
            {
                return Err(WorkflowError::Capability(request.workflow));
            }
        }
    }

    if !replacements.is_empty() {
        if let Some(object) = analysis.as_object_mut() {
            object.insert("replacements".into(), serde_json::to_value(&replacements)?);
        }
    }
    finish_workflow(
        request,
        input,
        protein,
        status,
        warnings,
        workflow_sites,
        analysis,
        extra_artifacts,
        control,
    )
}

fn finish_workflow(
    request: &ReGlycoRunRequestV1,
    input: &str,
    protein: Structure,
    status: WorkflowStatus,
    mut warnings: Vec<String>,
    workflow_sites: Vec<ReportSite>,
    mut analysis: Value,
    extra_artifacts: Vec<WorkflowArtifact>,
    control: &mut impl WorkflowControl,
) -> Result<WorkflowBundle> {
    let conversions = converted_proline_sites(request, input, &protein);
    if !conversions.is_empty() {
        warnings.push(format!(
            "Converted {} PRO attachment site{} to HYP for O-glycosylation.",
            conversions.len(),
            if conversions.len() == 1 { "" } else { "s" },
        ));
        if let Some(object) = analysis.as_object_mut() {
            object.insert("prolineConversions".into(), json!(conversions));
        } else {
            analysis = json!({
                "workflow": analysis,
                "prolineConversions": conversions,
            });
        }
    }
    // Structure validation and torsion analysis are deliberately not part of
    // workflow acceptance. Browser clients run torsion analysis after the
    // output has been displayed; no reporting bug may suppress a valid
    // Cookbook search result.
    let validation = ValidationSummary {
        valid: !protein.atoms().is_empty(),
        findings: Vec::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
        component_dictionary_version: None,
        steric_policy: None,
        steric_summary: None,
    };
    let final_status = status;
    let replacement_metadata = analysis
        .get("replacements")
        .filter(|value| value.as_array().is_some_and(|items| !items.is_empty()))
        .cloned()
        .unwrap_or_else(|| {
            if replacement_workflow(request.workflow) {
                requested_replacement_metadata(request)
            } else {
                Value::Array(Vec::new())
            }
        });
    let mut method = method_metadata(request);
    if let Some(polish) = analysis
        .get("vmm_polish")
        .or_else(|| analysis.get("vmmPolish"))
        .filter(|value| value.get("applied").and_then(Value::as_bool) == Some(true))
        .cloned()
    {
        if let Some(object) = method.as_object_mut() {
            object.insert("attachmentVmmPolish".into(), polish);
        }
    }
    if replacement_metadata
        .as_array()
        .is_some_and(|items| !items.is_empty())
    {
        if let Some(object) = method.as_object_mut() {
            object.insert("replacements".into(), replacement_metadata.clone());
        }
    }
    if let Some(search_budget) = analysis.get("searchBudget").cloned() {
        if let Some(object) = method.as_object_mut() {
            object.insert("searchBudget".into(), search_budget);
        }
    }
    if let Some(object) = analysis.as_object_mut() {
        object.insert("method".into(), method.clone());
        if replacement_metadata
            .as_array()
            .is_some_and(|items| !items.is_empty())
        {
            object
                .entry("replacements")
                .or_insert_with(|| replacement_metadata.clone());
        }
    } else {
        analysis = json!({
            "workflow": analysis,
            "method": method.clone(),
            "replacements": replacement_metadata.clone(),
        });
    }
    let generated_at = request.created_at.clone();
    let summary = match &final_status {
        WorkflowStatus::Succeeded => "Workflow completed successfully.",
        WorkflowStatus::Partial => "Workflow completed partially; retained outputs are available.",
        WorkflowStatus::Failed => "Workflow did not produce an accepted search result.",
        WorkflowStatus::Cancelled => {
            "Workflow was cancelled; retained artifacts are available for review."
        }
    };
    let report = WorkflowReport {
        workflow: request.workflow,
        title: report_title(request.workflow).into(),
        summary: summary.into(),
        generated_at,
        engine_version: ENGINE_VERSION.into(),
        input_sha256: format!("{:x}", Sha256::digest(input.as_bytes())),
        sites: workflow_sites,
        validation,
        analysis: analysis.clone(),
        provenance: json!({
            "request": request,
            "engineVersion": ENGINE_VERSION,
            "schemaVersion": SCHEMA_VERSION,
            "method": method,
            "replacements": replacement_metadata,
            "prolineConversions": conversions,
            "searchBudget": analysis
                .get("searchBudget")
                .cloned()
                .unwrap_or(Value::Null),
        }),
    };
    emit(
        control,
        "report",
        "Assembling the interactive report…",
        None,
        None,
    )?;
    let primary = protein.to_pdb_string();
    let primary_structure =
        (!matches!(final_status, WorkflowStatus::Failed)).then_some(primary.clone());
    let mut artifacts = vec![
        text_artifact("input.pdb", "chemical/x-pdb", ArtifactRole::Input, input),
        text_artifact(
            "request.json",
            "application/json",
            ArtifactRole::Provenance,
            request_artifact_json(request, &analysis)?,
        ),
        text_artifact(
            "report.json",
            "application/json",
            ArtifactRole::Report,
            serde_json::to_string_pretty(&report)?,
        ),
        text_artifact(
            "validation.json",
            "application/json",
            ArtifactRole::Analysis,
            serde_json::to_string_pretty(&report.validation)?,
        ),
    ];
    // A failed generated workflow may still be useful for diagnosis, but it
    // must never look like an accepted parent to browser/history adapters.
    // Keep that distinction in the artifact name/role as well as in the
    // optional primary_structure field.
    artifacts.insert(
        1,
        text_artifact(
            if matches!(final_status, WorkflowStatus::Failed) {
                "diagnostic-result.pdb"
            } else {
                "result.pdb"
            },
            "chemical/x-pdb",
            if matches!(final_status, WorkflowStatus::Failed) {
                ArtifactRole::Analysis
            } else {
                ArtifactRole::Structure
            },
            primary.clone(),
        ),
    );
    // Keep these two compact files in every report bundle, including scans
    // and structures with no known reference.  An empty CSV/reference object
    // is explicit and easier for offline consumers to handle than a missing
    // file whose absence could mean an incomplete download.
    artifacts.push(text_artifact(
        "torsions.csv",
        "text/csv",
        ArtifactRole::Analysis,
        torsion_csv(
            analysis
                .get("torsionObservations")
                .and_then(Value::as_array)
                .map_or(&[] as &[Value], |values| values.as_slice()),
            analysis
                .get("attachmentObservations")
                .and_then(Value::as_array)
                .map_or(&[] as &[Value], |values| values.as_slice()),
        ),
    ));
    if analysis
        .get("energyAnalysis")
        .or_else(|| analysis.get("energy_analysis"))
        .is_some()
    {
        artifacts.push(text_artifact(
            "energy.csv",
            "text/csv",
            ArtifactRole::Analysis,
            energy_analysis_csv(&analysis),
        ));
    }
    let reference = json!({
        "schemaVersion": "glycoshape-torsion-v1",
        "sourceDatabase": "GlycoShape",
        "references": analysis.get("torsionReferences").cloned().unwrap_or_else(|| json!([])),
        "attachmentReferences": analysis
            .get("attachmentReferences")
            .cloned()
            .unwrap_or_else(|| json!([])),
    });
    artifacts.push(text_artifact(
        "torsion-reference.json",
        "application/json",
        ArtifactRole::Analysis,
        serde_json::to_string_pretty(&reference)?,
    ));
    artifacts.extend(extra_artifacts);
    Ok(WorkflowBundle {
        schema_version: SCHEMA_VERSION,
        status: final_status,
        workflow: request.workflow,
        primary_structure,
        report,
        artifacts,
        warnings,
        error: None,
    })
}

/// Serialize the submitted request together with the resolved attachment
/// budget. The submitted numeric placeholders remain intact for replay, while
/// this additive top-level field tells offline consumers exactly what Auto
/// selected after assets and the conflict graph were loaded.
fn request_artifact_json(request: &ReGlycoRunRequestV1, analysis: &Value) -> Result<String> {
    let mut value = serde_json::to_value(request)?;
    if let Some(search_budget) = analysis.get("searchBudget") {
        value["resolvedSearchBudget"] = search_budget.clone();
    }
    Ok(serde_json::to_string_pretty(&value)?)
}

/// Assemble a strict-search fallback bundle. Build workflows retain the
/// complete candidate as a partial primary result; statistical Ensemble keeps
/// its historical failed status because its sampler contract requires a
/// valid target state.
fn finish_strict_failure(
    request: &ReGlycoRunRequestV1,
    input: &str,
    protein: Structure,
    mut warnings: Vec<String>,
    workflow_sites: Vec<ReportSite>,
    diagnostics: StrictSearchDiagnostics,
    replacement_records: &[ReplacementRecord],
    mut extra_artifacts: Vec<WorkflowArtifact>,
    control: &mut impl WorkflowControl,
) -> Result<WorkflowBundle> {
    let build_partial = matches!(
        request.workflow,
        WorkflowId::Uniprot | WorkflowId::SiteBuild
    ) && !diagnostics.best_candidate_pdb.trim().is_empty();
    emit(
        control,
        "validate",
        "Capturing strict steric diagnostics…",
        None,
        None,
    )?;
    warnings.push(if build_partial {
        "No candidate satisfied both clash-free sterics and the selected VMM circular 95% gate; the best complete candidate is retained as a partial Build result.".into()
    } else {
        "No candidate satisfied both clash-free sterics and the selected VMM circular 95% gate. The best candidate is available as a diagnostic artifact only.".into()
    });
    let diagnostics_value = serde_json::to_value(&diagnostics)?;
    let conversions = converted_proline_sites(request, input, &protein);
    if !conversions.is_empty() {
        warnings.push(format!(
            "Converted {} PRO attachment site{} to HYP for O-glycosylation.",
            conversions.len(),
            if conversions.len() == 1 { "" } else { "s" },
        ));
    }
    let mut analysis = json!({ "strictSearch": diagnostics_value });
    if let Some(object) = analysis.as_object_mut() {
        object.insert("prolineConversions".into(), json!(conversions.clone()));
    }
    let validation = ValidationSummary {
        valid: false,
        findings: Vec::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
        component_dictionary_version: None,
        steric_policy: None,
        steric_summary: None,
    };
    let mut method = method_metadata(request);
    let replacement_metadata = if replacement_records.is_empty() {
        requested_replacement_metadata(request)
    } else {
        serde_json::to_value(replacement_records)?
    };
    if replacement_metadata
        .as_array()
        .is_some_and(|items| !items.is_empty())
    {
        if let Some(object) = method.as_object_mut() {
            object.insert("replacements".into(), replacement_metadata.clone());
        }
    }
    if let Some(search_budget) = analysis.get("searchBudget").cloned() {
        if let Some(object) = method.as_object_mut() {
            object.insert("searchBudget".into(), search_budget.clone());
        }
        if let Some(object) = analysis.as_object_mut() {
            object.insert("method".into(), method.clone());
        }
    }
    if let Some(object) = analysis.as_object_mut() {
        object.insert("method".into(), method.clone());
        object.insert("replacements".into(), replacement_metadata.clone());
    }
    let report = WorkflowReport {
        workflow: request.workflow,
        title: report_title(request.workflow).into(),
        summary: if build_partial {
            "Build completed partially; the best complete structure was retained with unresolved steric/VMM diagnostics.".into()
        } else {
            "Strict steric search failed; no valid final structure was produced.".into()
        },
        generated_at: request.created_at.clone(),
        engine_version: ENGINE_VERSION.into(),
        input_sha256: format!("{:x}", Sha256::digest(input.as_bytes())),
        sites: workflow_sites,
        validation,
        analysis: analysis.clone(),
        provenance: json!({
            "request": request,
            "engineVersion": ENGINE_VERSION,
            "schemaVersion": SCHEMA_VERSION,
            "method": method,
            "replacements": replacement_metadata,
            "strictSearch": diagnostics,
            "prolineConversions": conversions,
        }),
    };
    emit(
        control,
        "report",
        "Assembling the strict-search diagnostic report…",
        None,
        None,
    )?;
    let diagnostic_json = serde_json::to_string_pretty(&report.analysis["strictSearch"])?;
    let mut artifacts = vec![
        text_artifact("input.pdb", "chemical/x-pdb", ArtifactRole::Input, input),
        text_artifact(
            "strict-search-diagnostics.json",
            "application/json",
            ArtifactRole::Analysis,
            diagnostic_json,
        ),
        text_artifact(
            "request.json",
            "application/json",
            ArtifactRole::Provenance,
            request_artifact_json(request, &analysis)?,
        ),
        text_artifact(
            "report.json",
            "application/json",
            ArtifactRole::Report,
            serde_json::to_string_pretty(&report)?,
        ),
        text_artifact(
            "validation.json",
            "application/json",
            ArtifactRole::Analysis,
            serde_json::to_string_pretty(&report.validation)?,
        ),
        text_artifact(
            "torsions.csv",
            "text/csv",
            ArtifactRole::Analysis,
            torsion_csv(
                analysis
                    .get("torsionObservations")
                    .and_then(Value::as_array)
                    .map_or(&[] as &[Value], |values| values.as_slice()),
                analysis
                    .get("attachmentObservations")
                    .and_then(Value::as_array)
                    .map_or(&[] as &[Value], |values| values.as_slice()),
            ),
        ),
        text_artifact(
            "torsion-reference.json",
            "application/json",
            ArtifactRole::Analysis,
            serde_json::to_string_pretty(&json!({
                "schemaVersion": "glycoshape-torsion-v1",
                "sourceDatabase": "GlycoShape",
                "references": analysis
                    .get("torsionReferences")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
                "attachmentReferences": analysis
                    .get("attachmentReferences")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            }))?,
        ),
    ];
    if !diagnostics.best_candidate_pdb.trim().is_empty() {
        artifacts.push(text_artifact(
            if build_partial {
                "result.pdb"
            } else {
                "strict-search-best-candidate.pdb"
            },
            "chemical/x-pdb",
            if build_partial {
                ArtifactRole::Structure
            } else {
                ArtifactRole::Analysis
            },
            diagnostics.best_candidate_pdb.clone(),
        ));
    }
    artifacts.append(&mut extra_artifacts);
    Ok(WorkflowBundle {
        schema_version: SCHEMA_VERSION,
        status: if build_partial {
            WorkflowStatus::Partial
        } else {
            WorkflowStatus::Failed
        },
        workflow: request.workflow,
        primary_structure: build_partial.then_some(diagnostics.best_candidate_pdb.clone()),
        report,
        artifacts,
        warnings,
        error: (!build_partial).then_some(
            "Strict steric search failed: no VMM-95%-compliant clash-free structure was found."
                .into(),
        ),
    })
}

fn workflow_stage(workflow: WorkflowId) -> &'static str {
    match workflow {
        WorkflowId::Uniprot | WorkflowId::SiteBuild => "Build",
        WorkflowId::Ensemble => "Ensemble",
        WorkflowId::Relax => "Minimize",
        WorkflowId::Refine => "Optimize",
        WorkflowId::Validate => "input/parent",
        WorkflowId::NScan | WorkflowId::Density | WorkflowId::Saxs => "input/parent",
    }
}

/// Return the accepted stage represented by the structure passed to the
/// common finalization hook. Composite Build/Refine workflows may retain an
/// unrelaxed structure as `Build`/`Optimize` and promote the relaxed structure
/// to the separate `Minimize` stage.
fn final_structure_stage(request: &ReGlycoRunRequestV1, analysis: &Value) -> &'static str {
    match request.workflow {
        WorkflowId::Uniprot | WorkflowId::SiteBuild | WorkflowId::Refine
            if request.options.post_relax
                && analysis.get("relaxation").is_some()
                && analysis
                    .get("relaxation")
                    .is_some_and(|value| !value.is_null()) =>
        {
            "Minimize"
        }
        _ => workflow_stage(request.workflow),
    }
}

fn append_torsion_analysis(analysis: &mut Value, structure: &Structure, stage: &str) {
    let references = analysis
        .get("torsionReferences")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut observations = glycosidic_torsion_observations(structure, stage, None, &references);
    if observations.is_empty() {
        return;
    }
    // References are loaded from persisted GlycoShape assets before this
    // function runs. Annotate each accepted observation against the complete
    // periodic mixture now, rather than forcing the browser to repeat the
    // percentile calculation or accidentally leaving every point as
    // `no_reference`.
    for observation in &mut observations {
        annotate_torsion_observation(observation, &references);
    }
    if let Some(object) = analysis.as_object_mut() {
        let existing = object
            .entry("torsionObservations")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(items) = existing {
            // Accepted stages can be attached both at the point a workflow
            // produces them and during the common finalization hook. Keep a
            // compact one-record-per-stage/linkage/frame stream while still
            // retaining a changed torsion after a successful relaxation.
            for observation in observations {
                let duplicate = items.iter().any(|existing| {
                    torsion_observation_key(existing) == torsion_observation_key(&observation)
                });
                if !duplicate {
                    items.push(observation);
                }
            }
            // Ensemble adapters may have inserted one observation per
            // accepted frame before the common finalization hook. Annotate
            // those records too so all points use the same reference policy.
            for item in items {
                annotate_torsion_observation(item, &references);
            }
        }
    } else {
        *analysis = json!({ "torsionObservations": observations, "workflow": analysis.clone() });
    }
}

fn torsion_observation_key(
    value: &Value,
) -> (String, Option<u64>, String, Option<i64>, Option<i64>) {
    let object = value.as_object();
    let string = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| object.and_then(|object| object.get(*name)))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let number = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| object.and_then(|object| object.get(*name)))
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_i64().map(|value| value as f64))
                    .or_else(|| value.as_u64().map(|value| value as f64))
            })
            .map(|value| (value * 1.0e6).round() as i64)
    };
    (
        string(&["stage"]),
        object
            .and_then(|object| object.get("frame"))
            .and_then(Value::as_u64),
        string(&["linkage"]),
        number(&["phiDegrees", "phi_degrees"]),
        number(&["psiDegrees", "psi_degrees"]),
    )
}

fn annotate_torsion_observation(observation: &mut Value, references: &[Value]) {
    let reference_value = torsion_reference_for_observation(observation, references);
    let Some(object) = observation.as_object_mut() else {
        return;
    };
    let Some(reference_value) = reference_value else {
        object.insert("assessment".into(), json!("no_reference"));
        object.insert("policyVersion".into(), json!("glycoshape-torsion-v1"));
        return;
    };
    let reference = reference_value.as_object();
    let grid_value = reference
        .and_then(|value| value.get("phi_psi").or_else(|| value.get("phiPsi")))
        .and_then(Value::as_object);
    let grid = grid_value
        .and_then(|value| value.get("counts").or_else(|| value.get("grid")))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_f64().or_else(|| value.as_u64().map(|v| v as f64)))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let bins = grid_value
        .and_then(|value| value.get("bins"))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .filter(|value| *value > 0)
        .unwrap_or_else(|| (grid.len() as f64).sqrt().round() as usize);
    let phi = object
        .get("phiDegrees")
        .and_then(Value::as_f64)
        .or_else(|| object.get("phi_degrees").and_then(Value::as_f64));
    let psi = object
        .get("psiDegrees")
        .and_then(Value::as_f64)
        .or_else(|| object.get("psi_degrees").and_then(Value::as_f64));
    let percentile = match (phi, psi) {
        (Some(phi), Some(psi)) if bins > 0 && grid.len() >= bins.saturating_mul(bins) => {
            let bin = |angle: f64| {
                ((angle + 180.0).rem_euclid(360.0) / 360.0 * bins as f64)
                    .floor()
                    .clamp(0.0, bins.saturating_sub(1) as f64) as usize
            };
            let index = bin(phi) + bin(psi) * bins;
            let density = grid.get(index).copied().unwrap_or(0.0).max(0.0);
            let total = grid
                .iter()
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .sum::<f64>();
            (total > 0.0).then(|| {
                grid.iter()
                    .copied()
                    .filter(|value| value.is_finite() && *value + f64::EPSILON >= density)
                    .sum::<f64>()
                    / total
                    * 100.0
            })
        }
        _ => None,
    };
    if let (Some(phi), Some(psi)) = (phi, psi) {
        let nearest = level_one_cluster_at(reference_value, phi, psi).or_else(|| {
            reference
                .and_then(|reference| reference.get("populations"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|population| {
                    population
                        .get("level")
                        .and_then(Value::as_u64)
                        .is_none_or(|level| level == 1)
                })
                .filter_map(|population| {
                    let population_phi = population
                        .get("phiMean")
                        .or_else(|| population.get("phi_mean"))
                        .or_else(|| population.get("medoidPhi"))
                        .or_else(|| population.get("medoid_phi"))
                        .and_then(value_as_f64)?;
                    let population_psi = population
                        .get("psiMean")
                        .or_else(|| population.get("psi_mean"))
                        .or_else(|| population.get("medoidPsi"))
                        .or_else(|| population.get("medoid_psi"))
                        .and_then(value_as_f64)?;
                    let index = population.get("index").and_then(|value| {
                        value
                            .as_u64()
                            .or_else(|| value.as_i64().map(|value| value as u64))
                    })?;
                    let distance = circular_distance_degrees(phi, population_phi).powi(2)
                        + circular_distance_degrees(psi, population_psi).powi(2);
                    Some((distance, index))
                })
                .min_by(|left, right| left.0.total_cmp(&right.0))
                .map(|(_, index)| index)
        });
        if let Some(nearest) = nearest {
            object.insert("nearestPopulation".into(), json!(nearest));
        }
    }
    if let Some(percentile) = percentile {
        let assessment = if percentile <= 50.0 {
            "core"
        } else if percentile <= 80.0 {
            "allowed"
        } else if percentile <= 95.0 {
            "tail"
        } else {
            "outlier"
        };
        object.insert("populationPercentile".into(), json!(percentile));
        object.insert("assessment".into(), json!(assessment));
    } else {
        object.insert("assessment".into(), json!("no_reference"));
    }
    object.insert("policyVersion".into(), json!("glycoshape-torsion-v1"));
}

fn circular_distance_degrees(first: f64, second: f64) -> f64 {
    (first - second + 180.0).rem_euclid(360.0) - 180.0
}

/// Assign an accepted point using the authoritative periodic Level-1 cluster
/// grids. Empty bins deliberately return `None`, allowing the caller to fall
/// back to the nearest circular medoid only when the grid has no observation
/// at that coordinate.
fn level_one_cluster_at(reference: &Value, phi: f64, psi: f64) -> Option<u64> {
    reference
        .get("clusters")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|cluster| {
            let id = cluster
                .get("id")
                .or_else(|| cluster.get("index"))
                .and_then(|value| value.as_u64().or_else(|| value.as_i64().map(|v| v as u64)))?;
            let contour = cluster.get("phi_psi").or_else(|| cluster.get("phiPsi"))?;
            let values = contour
                .get("counts")
                .or_else(|| contour.get("grid"))
                .and_then(Value::as_array)?;
            let bins = contour
                .get("bins")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .filter(|value| *value > 0)
                .unwrap_or_else(|| (values.len() as f64).sqrt().round() as usize);
            if bins == 0 || values.len() < bins.saturating_mul(bins) {
                return None;
            }
            let index = periodic_bin(phi, bins) + periodic_bin(psi, bins) * bins;
            let density = values
                .get(index)
                .and_then(|value| {
                    value
                        .as_f64()
                        .or_else(|| value.as_u64().map(|v| v as f64))
                        .or_else(|| value.as_i64().map(|v| v as f64))
                })
                .unwrap_or(0.0);
            (density > 0.0).then_some((density, id))
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, id)| id)
}

/// Summarize accepted ensemble points without persisting any transient GA
/// population coordinates.  The maps are keyed by canonical linkage so the
/// same report can be reopened offline and compared across stages.
fn append_ensemble_torsion_summary(analysis: &mut Value) {
    let Some(object) = analysis.as_object_mut() else {
        return;
    };
    let Some(observations) = object
        .get("torsionObservations")
        .and_then(Value::as_array)
        .cloned()
    else {
        return;
    };
    let ensemble = observations
        .iter()
        .filter(|observation| observation.get("frame").and_then(Value::as_u64).is_some())
        .collect::<Vec<_>>();
    if ensemble.is_empty() {
        return;
    }
    let mut frames = BTreeSet::new();
    let mut by_linkage = BTreeMap::<String, Vec<(usize, f64, f64, String, Option<usize>)>>::new();
    for observation in ensemble {
        let Some(frame) = observation.get("frame").and_then(Value::as_u64) else {
            continue;
        };
        let Some(linkage) = observation.get("linkage").and_then(Value::as_str) else {
            continue;
        };
        let Some(phi) = observation
            .get("phiDegrees")
            .and_then(Value::as_f64)
            .or_else(|| observation.get("phi_degrees").and_then(Value::as_f64))
        else {
            continue;
        };
        let Some(psi) = observation
            .get("psiDegrees")
            .and_then(Value::as_f64)
            .or_else(|| observation.get("psi_degrees").and_then(Value::as_f64))
        else {
            continue;
        };
        frames.insert(frame as usize);
        let nearest_population = observation
            .get("nearestPopulation")
            .or_else(|| observation.get("nearest_population"))
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_i64().map(|value| value as u64))
            })
            .map(|value| value as usize);
        by_linkage.entry(linkage.into()).or_default().push((
            frame as usize,
            phi,
            psi,
            observation
                .get("assessment")
                .and_then(Value::as_str)
                .unwrap_or("no_reference")
                .into(),
            nearest_population,
        ));
    }
    if frames.is_empty() {
        return;
    }
    let mut outlier_frames = BTreeSet::new();
    let mut linkage_outlier_rates = serde_json::Map::new();
    let mut circular_means = serde_json::Map::new();
    let mut circular_dispersion = serde_json::Map::new();
    let mut cluster_coverage = serde_json::Map::new();
    for (linkage, values) in by_linkage {
        let outliers = values
            .iter()
            .filter(|(_, _, _, assessment, _)| assessment == "outlier")
            .map(|(frame, _, _, _, _)| {
                outlier_frames.insert(*frame);
                *frame
            })
            .collect::<BTreeSet<_>>();
        linkage_outlier_rates.insert(
            linkage.clone(),
            json!(outliers.len() as f64 / values.len().max(1) as f64),
        );
        let mean = |axis: usize| {
            let (sin_sum, cos_sum) = values.iter().fold((0.0, 0.0), |(s, c), value| {
                let angle = if axis == 0 { value.1 } else { value.2 }.to_radians();
                (s + angle.sin(), c + angle.cos())
            });
            sin_sum.atan2(cos_sum).to_degrees()
        };
        let phi_mean = mean(0);
        let psi_mean = mean(1);
        let resultant = |axis: usize| {
            let (sin_sum, cos_sum) = values.iter().fold((0.0, 0.0), |(s, c), value| {
                let angle = if axis == 0 { value.1 } else { value.2 }.to_radians();
                (s + angle.sin(), c + angle.cos())
            });
            (sin_sum.hypot(cos_sum) / values.len().max(1) as f64).clamp(0.0, 1.0)
        };
        circular_means.insert(linkage.clone(), json!([phi_mean, psi_mean]));
        circular_dispersion.insert(
            linkage.clone(),
            json!([1.0 - resultant(0), 1.0 - resultant(1)]),
        );
        let observed_clusters = values
            .iter()
            .filter_map(|value| value.4)
            .collect::<BTreeSet<_>>();
        // `nearestPopulation` is emitted only when a reference contains
        // Level-1 medoids.  Coverage is therefore a conservative fraction of
        // those broad populations represented by accepted frames; when an
        // older reference has no population list we still report whether any
        // population marker was observed.
        let denominator = object
            .get("torsionReferences")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|reference| {
                reference
                    .get("canonicalLinkage")
                    .or_else(|| reference.get("canonical_linkage"))
                    .or_else(|| reference.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| name == linkage)
            })
            .and_then(|reference| reference.get("populations"))
            .and_then(Value::as_array)
            .map(|populations| {
                populations
                    .iter()
                    .filter(|population| {
                        population
                            .get("level")
                            .and_then(Value::as_u64)
                            .is_none_or(|level| level == 1)
                    })
                    .count()
            })
            .unwrap_or(0);
        let coverage = if denominator > 0 {
            observed_clusters.len() as f64 / denominator as f64
        } else if observed_clusters.is_empty() {
            0.0
        } else {
            1.0
        };
        cluster_coverage.insert(linkage, json!(coverage.clamp(0.0, 1.0)));
    }
    object.insert(
        "ensembleTorsion".into(),
        json!({
            "frames": frames.len(),
            "observedFrames": frames,
            "outlierFrames": outlier_frames,
            "linkageOutlierRates": linkage_outlier_rates,
            "circularMeans": circular_means,
            "circularDispersion": circular_dispersion,
            "clusterCoverage": cluster_coverage,
        }),
    );
}

/// Compare the selected GlycoShape conformer at each attachment site with
/// the authoritative Level-1 conformer populations. This is intentionally
/// separate from internal linkage populations: a one-sugar glycan has no
/// internal linkage to plot, but it still has a meaningful selected cluster.
fn append_glycan_cluster_distributions(analysis: &mut Value, request: &ReGlycoRunRequestV1) {
    let Some(object) = analysis.as_object_mut() else {
        return;
    };
    let references = object
        .get("glycanClusterReferences")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let observations = object
        .get("attachmentObservations")
        .or_else(|| object.get("attachment_observations"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if references.is_empty() || observations.is_empty() {
        return;
    }

    // Attachment observations use the active (non-excluded) assignment
    // order, so map their 1-based glycanIndex to the same order here.
    let glycan_ids = request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded)
        .map(|assignment| assignment.glycan_id.as_str())
        .collect::<Vec<_>>();
    let reference_for = |glycan_index: usize| {
        let requested = glycan_ids.get(glycan_index.saturating_sub(1)).copied();
        references.iter().find(|reference| {
            let value = reference
                .get("glycanId")
                .or_else(|| reference.get("glycan_id"))
                .and_then(Value::as_str);
            requested.is_some_and(|requested| value == Some(requested))
        })
    };
    let mut grouped = BTreeMap::<String, Vec<&Value>>::new();
    for observation in &observations {
        let stage = observation
            .get("stage")
            .and_then(Value::as_str)
            .unwrap_or("result");
        let normalized = stage.trim().to_ascii_lowercase();
        if normalized.contains("input") || normalized.contains("parent") {
            continue;
        }
        let site = observation
            .get("site")
            .map(residue_to_string)
            .unwrap_or_default();
        let glycan_index = observation
            .get("glycanIndex")
            .or_else(|| observation.get("glycan_index"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let cluster = observation
            .get("clusterIndex")
            .or_else(|| observation.get("cluster_index"))
            .or_else(|| observation.get("mainCluster"))
            .or_else(|| observation.get("main_cluster"))
            .and_then(|value| value.as_u64().or_else(|| value.as_i64().map(|v| v as u64)));
        if site.is_empty() || glycan_index == 0 || cluster.is_none() {
            continue;
        }
        let key = format!("{stage}|{site}|{glycan_index}");
        grouped.entry(key).or_default().push(observation);
    }
    if grouped.is_empty() {
        return;
    }

    let mut distributions = Vec::new();
    for (key, values) in grouped {
        let first = values[0];
        let site = first.get("site").map(residue_to_string).unwrap_or_default();
        let glycan_index = first
            .get("glycanIndex")
            .or_else(|| first.get("glycan_index"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let Some(reference) = reference_for(glycan_index) else {
            continue;
        };
        let clusters = reference
            .get("clusters")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|cluster| {
                let index = cluster
                    .get("id")
                    .or_else(|| cluster.get("index"))
                    .and_then(|value| {
                        value.as_u64().or_else(|| value.as_i64().map(|v| v as u64))
                    })?;
                let weight = cluster
                    .get("weight")
                    .or_else(|| cluster.get("pct"))
                    .and_then(value_as_f64)
                    .unwrap_or(0.0);
                Some((
                    index,
                    weight,
                    cluster.get("color").cloned().unwrap_or(Value::Null),
                ))
            })
            .collect::<Vec<_>>();
        if clusters.is_empty() {
            continue;
        }
        let reference_total = clusters
            .iter()
            .map(|(_, weight, _)| weight.max(0.0))
            .sum::<f64>();
        let mut observed = BTreeMap::<u64, usize>::new();
        for value in &values {
            if let Some(index) = value
                .get("clusterIndex")
                .or_else(|| value.get("cluster_index"))
                .or_else(|| value.get("mainCluster"))
                .or_else(|| value.get("main_cluster"))
                .and_then(|value| value.as_u64().or_else(|| value.as_i64().map(|v| v as u64)))
            {
                *observed.entry(index).or_default() += 1;
            }
        }
        let values_len = values.len().max(1) as f64;
        let cluster_values = clusters
            .into_iter()
            .map(|(index, weight, color)| {
                let observed_frames = observed.get(&index).copied().unwrap_or(0);
                let reference_fraction = if reference_total > 0.0 {
                    weight.max(0.0) / reference_total
                } else {
                    0.0
                };
                let observed_fraction = observed_frames as f64 / values_len;
                json!({
                    "index": index,
                    "color": color,
                    "referenceFraction": reference_fraction,
                    "observedFraction": observed_fraction,
                    "observedFrames": observed_frames,
                    "differencePercentagePoints": (observed_fraction - reference_fraction) * 100.0,
                })
            })
            .collect::<Vec<_>>();
        distributions.push(json!({
            "key": key,
            "site": site,
            "glycanIndex": glycan_index,
            "stage": first.get("stage").cloned().unwrap_or_else(|| json!("result")),
            "frames": values.len(),
            "clusters": cluster_values,
        }));
    }
    if !distributions.is_empty() {
        object.insert(
            "glycanClusterDistributions".into(),
            Value::Array(distributions),
        );
    }
}

/// Compare the authoritative GlycoShape Level-1 population weights with the
/// populations visited by accepted ensemble frames. Identical glycan
/// linkages at different protein sites remain separate report rows.
fn append_cluster_distributions(analysis: &mut Value) {
    let Some(object) = analysis.as_object_mut() else {
        return;
    };
    let references = object
        .get("torsionReferences")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let observations = object
        .get("torsionObservations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut grouped = BTreeMap::<String, Vec<Value>>::new();
    // Compare the accepted output stage, not the input/parent baseline.  A
    // single Build/Minimize/Optimize structure is still a meaningful
    // one-observation distribution; previously this function filtered out
    // every frame-less Build observation, leaving the browser with no
    // per-site cluster information at all.  Ensemble observations retain
    // their frame index and are grouped in the same representation.
    for observation in observations.into_iter().filter(|value| {
        value
            .get("stage")
            .and_then(Value::as_str)
            .is_none_or(|stage| {
                let normalized = stage.trim().to_ascii_lowercase();
                !normalized.contains("input") && !normalized.contains("parent")
            })
    }) {
        let site = observation
            .get("site")
            .and_then(Value::as_str)
            .unwrap_or("unassigned");
        let glycan = observation
            .get("glycanIndex")
            .or_else(|| observation.get("glycan_index"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let linkage = observation
            .get("linkage")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let stage = observation
            .get("stage")
            .and_then(Value::as_str)
            .unwrap_or("result");
        let branch = observation
            .get("branchPath")
            .or_else(|| observation.get("branch_path"))
            .cloned()
            .unwrap_or_else(|| json!([]));
        grouped
            .entry(format!("{stage}|{site}|{glycan}|{linkage}|{branch}"))
            .or_default()
            .push(observation);
    }
    let mut distributions = Vec::new();
    for (key, values) in grouped {
        let first = &values[0];
        let linkage = first
            .get("linkage")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let reference = references.iter().find(|reference| {
            reference
                .get("canonicalLinkage")
                .or_else(|| reference.get("canonical_linkage"))
                .or_else(|| reference.get("name"))
                .and_then(Value::as_str)
                .is_some_and(|name| name == linkage)
        });
        let populations = reference
            .and_then(|reference| reference.get("populations"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|population| {
                population
                    .get("level")
                    .and_then(Value::as_u64)
                    .is_none_or(|level| level == 1)
            })
            .collect::<Vec<_>>();
        let reference_total = populations
            .iter()
            .filter_map(|population| population.get("weight").and_then(Value::as_f64))
            .sum::<f64>();
        let mut observed = BTreeMap::<u64, usize>::new();
        for value in &values {
            if let Some(cluster) = value
                .get("nearestPopulation")
                .or_else(|| value.get("nearest_population"))
                .and_then(Value::as_u64)
            {
                *observed.entry(cluster).or_default() += 1;
            }
        }
        let clusters = populations
            .iter()
            .filter_map(|population| {
                let index = population.get("index")?.as_u64()?;
                let weight = population
                    .get("weight")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let reference_fraction = if reference_total > 0.0 {
                    weight / reference_total
                } else {
                    0.0
                };
                let count = observed.get(&index).copied().unwrap_or(0);
                let observed_fraction = count as f64 / values.len().max(1) as f64;
                Some(json!({
                    "index": index,
                    "color": population.get("color").cloned().unwrap_or(Value::Null),
                    "referenceFraction": reference_fraction,
                    "observedFraction": observed_fraction,
                    "observedFrames": count,
                    "differencePercentagePoints": (observed_fraction - reference_fraction) * 100.0,
                }))
            })
            .collect::<Vec<_>>();
        distributions.push(json!({
            "key": key,
            "site": first.get("site").cloned().unwrap_or(Value::Null),
            "glycanIndex": first.get("glycanIndex").or_else(|| first.get("glycan_index")).cloned().unwrap_or(Value::Null),
            "stage": first.get("stage").cloned().unwrap_or_else(|| json!("result")),
            "branchPath": first.get("branchPath").or_else(|| first.get("branch_path")).cloned().unwrap_or_else(|| json!([])),
            "linkage": linkage,
            "frames": values.len(),
            "unassignedFrames": values.len().saturating_sub(observed.values().sum()),
            "clusters": clusters,
        }));
    }
    object.insert("clusterDistributions".into(), Value::Array(distributions));
}

fn append_attachment_analysis(analysis: &mut Value, stage: &str) {
    let Some(object) = analysis.as_object_mut() else {
        return;
    };
    // Build and older optimized reports place the compact search records at
    // the top level; some composite reports retain them under `search`.
    // Clone the small metadata array so the later insertion of derived
    // observations does not hold a conflicting mutable borrow.
    let sites = object
        .get("sites")
        .or_else(|| object.get("search").and_then(|value| value.get("sites")))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if sites.is_empty() {
        return;
    }
    let observations = sites
        .iter()
        .filter_map(|site| {
            let site_object = site.as_object()?;
            let key = site_object.get("site")?;
            let site_label = if let Some(residue) = key.get("residue") {
                residue_to_string(residue)
            } else {
                residue_to_string(key)
            };
            Some(json!({
                "domain": "attachment",
                "origin": "accepted",
                "site": site_label,
                "stage": stage,
                "frame": Value::Null,
                "glycanIndex": site_object.get("glycan_index").or_else(|| site_object.get("glycanIndex")),
                "linkage": site_object.get("linkage"),
                "involvedAtoms": site_object.get("involved_atoms").or_else(|| site_object.get("involvedAtoms")).cloned().unwrap_or_else(|| json!([])),
                "phiDegrees": site_object.get("phi_degrees").or_else(|| site_object.get("phiDegrees")),
                "psiDegrees": site_object.get("psi_degrees").or_else(|| site_object.get("psiDegrees")),
                "selectedPhiComponent": site_object.get("phi_component").or_else(|| site_object.get("phiComponent")),
                "selectedPsiComponent": site_object.get("psi_component").or_else(|| site_object.get("psiComponent")),
                "assessment": attachment_assessment_json(
                    site_object.get("phi_within_vmm95").and_then(Value::as_bool),
                    site_object.get("psi_within_vmm95").and_then(Value::as_bool),
                ),
            }))
        })
        .collect::<Vec<_>>();
    if !observations.is_empty() {
        let entry = object
            .entry("attachmentObservations")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(existing) = entry {
            for observation in observations {
                if !existing.iter().any(|current| {
                    attachment_observation_key(current) == attachment_observation_key(&observation)
                }) {
                    existing.push(observation);
                }
            }
        }
    }
}

/// Add the compact Build search record consumed by the browser report. The
/// transient GA population is intentionally not serialized; these fields are
/// enough to explain the selected probability-ranked pose and to reproduce
/// the bounded search decision from the request and model version.
fn append_build_search_diagnostics(
    analysis: &mut Value,
    outcome: &SearchOutcome,
    config: &SearchConfig,
    search_sites: &[SearchSite],
    clash_partners: &[Vec<String>],
) {
    if config.selection_policy != SearchSelectionPolicy::JointPriorV1 {
        return;
    }
    let polish = &outcome.vmm_polish;
    let final_score = polish.final_prior_score.or(Some(outcome.total_score));
    let sites = outcome
        .sites
        .iter()
        .enumerate()
        .map(|(index, site)| {
            let population_source = search_sites
                .get(index)
                .map(|search_site| match search_site.ensemble.population_source {
                    ConformerPopulationSource::AssetMetadata => "asset_metadata",
                    ConformerPopulationSource::EqualFallback => "equal_weight_fallback",
                })
                .unwrap_or("unknown");
            json!({
                "site": site.site,
                "conformerId": site.conformer_id,
                "conformerProbability": site.conformer_probability,
                "attachmentLogDensity": site.attachment_log_density,
                "jointPriorScore": site.joint_prior_score,
                "phiDegrees": site.phi_degrees,
                "psiDegrees": site.psi_degrees,
                "phiWithinVmm95": site.phi_within_vmm95,
                "psiWithinVmm95": site.psi_within_vmm95,
                "stericScore": site.steric_score,
                "clashPartners": clash_partners.get(index).cloned().unwrap_or_default(),
                "status": if site.steric_score <= 1.1
                    && site.phi_within_vmm95.unwrap_or(true)
                    && site.psi_within_vmm95.unwrap_or(true) { "clear" } else { "unresolved" },
                "populationSource": population_source,
            })
        })
        .collect::<Vec<_>>();
    if let Some(object) = analysis.as_object_mut() {
        object.insert(
            "searchDiagnostics".into(),
            json!({
                "selectionPolicy": "joint_prior_v1",
                "priorModel": "conformer_population_x_attachment_vmm_v1",
                "populationSources": search_sites.iter().map(|site| match site.ensemble.population_source {
                    ConformerPopulationSource::AssetMetadata => "asset_metadata",
                    ConformerPopulationSource::EqualFallback => "equal_weight_fallback",
                }).collect::<Vec<_>>(),
                "description": if outcome.complete_output && (outcome.clash_status == ClashStatus::BestCompleteClashing || !outcome.vmm_gate_satisfied) {
                    "best complete result within the configured search budget; unresolved steric/VMM sites are reported below"
                } else {
                    "best found within the configured search budget"
                },
                "populationSize": config.population_size,
                "generationLimit": config.generations,
                "searchBudget": polish.search_budget,
                "evaluations": polish.evaluations,
                "validCandidates": polish.valid_candidates,
                "completeOutput": outcome.complete_output,
                "vmmGateSatisfied": outcome.vmm_gate_satisfied,
                "clashStatus": serde_json::to_value(outcome.clash_status).unwrap_or(Value::Null),
                "firstFeasibleScore": polish.first_feasible_score,
                "finalPriorScore": final_score,
                "terminationReason": if outcome.termination_reason.is_empty() { "unknown" } else { outcome.termination_reason.as_str() },
                "geometryGpuEvaluations": polish.geometry_gpu_evaluations,
                "geometryCpuEvaluations": polish.geometry_cpu_evaluations,
                "geometryGpuSeconds": polish.geometry_gpu_seconds,
                "geometryCpuSeconds": polish.geometry_cpu_seconds,
                "geometryTransformSeconds": polish.geometry_transform_seconds,
                "compatibility": {
                    "algorithm": polish.compatibility_algorithm,
                    "seededStates": polish.compatibility_seeded_states,
                    "poseAttempts": polish.compatibility_pose_attempts,
                    "poolSizes": polish.compatibility_pool_sizes,
                    "poolExpansions": polish.compatibility_pool_expansions,
                    "checks": polish.compatibility_checks,
                    "backtracks": polish.compatibility_backtracks,
                    "graphEdges": polish.compatibility_graph_edges,
                    "attemptsPerSite": polish.compatibility_attempts_per_site,
                    "proposalBudget": polish.compatibility_proposal_budget,
                },
                "stoppingRule": "10 completed generations without >1e-4 joint-score improvement or configured budget",
                "timings": {
                    "preparationSeconds": outcome.timings.preparation_seconds,
                    "proposalSeconds": outcome.timings.proposal_seconds,
                    "scoringSeconds": outcome.timings.scoring_seconds,
                    "gaSeconds": outcome.timings.ga_seconds,
                    "materializationSeconds": outcome.timings.materialization_seconds,
                    "statesPerSecond": outcome.timings.states_per_second,
                },
                "firstFeasibleVsFinal": {
                    "first": polish.first_feasible_score,
                    "final": final_score,
                },
                "sites": sites,
            }),
        );
    }
}

fn attachment_observation_key(
    value: &Value,
) -> (String, Option<u64>, String, Option<i64>, Option<i64>) {
    let object = value.as_object();
    let string = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| object.and_then(|object| object.get(*name)))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let number = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| object.and_then(|object| object.get(*name)))
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_i64().map(|value| value as f64))
                    .or_else(|| value.as_u64().map(|value| value as f64))
            })
            .map(|value| (value * 1.0e6).round() as i64)
    };
    (
        string(&["stage"]),
        object
            .and_then(|object| object.get("frame"))
            .and_then(Value::as_u64),
        string(&["site"]),
        number(&["phiDegrees", "phi_degrees"]),
        number(&["psiDegrees", "psi_degrees"]),
    )
}

fn attachment_assessment_json(phi_within: Option<bool>, psi_within: Option<bool>) -> &'static str {
    match (phi_within, psi_within) {
        (Some(true), Some(true)) => "allowed",
        (Some(false), _) | (_, Some(false)) => "outlier",
        _ => "no_reference",
    }
}

/// Convert the compact per-site result emitted by the search/ensemble engine
/// into the report's stage/frame observation contract.  Keeping this adapter
/// here means native and WASM executions serialize the same values while the
/// sampler remains free to keep its internal chromosome/cache state private.
fn append_attachment_observations_from_results(
    analysis: &mut Value,
    selected: &[SearchSiteResult],
    stage: &str,
    frame: Option<usize>,
) {
    let observations = selected
        .iter()
        .enumerate()
        .map(|(index, result)| {
            json!({
                "domain": "attachment",
                "origin": "accepted",
                "site": result.site.residue,
                "stage": stage,
                "frame": frame,
                "glycanIndex": index + 1,
                // `main_cluster` is the Level-1 parent of the selected
                // conformer.  Keep only that broad cluster in the report;
                // Level-2/3 IDs remain an implementation detail of the
                // sampler and must not create extra colours in the UI.
                "clusterIndex": result.main_cluster.unwrap_or(result.cluster_index),
                "clusterLevel": 1,
                "conformerId": result.conformer_id,
                "phiDegrees": result.phi_degrees,
                "psiDegrees": result.psi_degrees,
                "selectedPhiComponent": result.phi_component,
                "selectedPsiComponent": result.psi_component,
                "phiWithinVmm95": result.phi_within_vmm95,
                "psiWithinVmm95": result.psi_within_vmm95,
                "assessment": attachment_assessment_json(
                    result.phi_within_vmm95,
                    result.psi_within_vmm95,
                ),
                // Attachment torsions are protein-site observations, not
                // internal glycosidic linkages.  Do not hard-code C1 here:
                // keto sugars such as SIA/KDO attach through C2, and
                // furanoses use a different ring anchor.  The selected
                // component/angle data below carries the exact VMM context.
                "linkage": format!("{}:attachment", result.site.residue),
                "involvedAtoms": [format!("{}:protein", result.site.residue), format!("{}:glycan", result.site.residue)],
                "policyVersion": "glycoshape-attachment-vmm-v2",
            })
        })
        .collect::<Vec<_>>();
    if observations.is_empty() {
        return;
    }
    if let Some(object) = analysis.as_object_mut() {
        let entry = object
            .entry("attachmentObservations")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(existing) = entry {
            for observation in observations {
                if !existing.iter().any(|current| {
                    attachment_observation_key(current) == attachment_observation_key(&observation)
                }) {
                    existing.push(observation);
                }
            }
        }
    }
}

/// Persist the complete attachment-site VMM mixture used by the search.  The
/// browser report renders these as a separate contour/marginal view so the
/// protein--glycan attachment gate is never confused with an internal
/// glycosidic linkage distribution.  Keeping the means, concentrations,
/// weights, and circular 95% bounds beside the accepted result also makes an
/// offline report reproducible without re-running the engine.
fn append_attachment_reference_data(
    analysis: &mut Value,
    protein: &Structure,
    sites: &[SearchSite],
    selected: &[SearchSiteResult],
) {
    let references = selected
        .iter()
        .enumerate()
        .filter_map(|(index, result)| {
            let site = sites.get(index)?;
            let conformer = site.ensemble.conformers.get(result.conformer_index)?;
            let residue = site.site.residue.clone();
            let priors = reglyco_ensemble::resolved_priors(protein, site, &conformer.priors);
            let components = |values: &[VonMisesComponent]| {
                values
                    .iter()
                    .enumerate()
                    .map(|(component, value)| {
                        let bound = vmm_component_bounds_95(value);
                        json!({
                            "index": component,
                            "meanDegrees": value.mean_degrees,
                            "concentration": value.concentration,
                            "probabilityRegionVersion": "circular-integrated-v2",
                            "weight": value.weight,
                            "lower95Degrees": bound.map(|(lower, _)| lower),
                            "upper95Degrees": bound.map(|(_, upper)| upper),
                        })
                    })
                    .collect::<Vec<_>>()
            };
            Some(json!({
                "site": residue,
                "phi": components(&priors.phi),
                "psi": components(&priors.psi),
                "selectedPhiComponent": result.phi_component,
                "selectedPsiComponent": result.psi_component,
                "phiWithinVmm95": result.phi_within_vmm95,
                "psiWithinVmm95": result.psi_within_vmm95,
            }))
        })
        .collect::<Vec<_>>();
    if let Some(object) = analysis.as_object_mut() {
        object.insert("attachmentReferences".into(), Value::Array(references));
    }
}

fn append_attachment_reference_data_from_observations(analysis: &mut Value, sites: &[SearchSite]) {
    let observations = analysis
        .get("attachmentObservations")
        .or_else(|| analysis.get("attachment_observations"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let component = |observation: &Value, camel: &str, snake: &str| {
        observation
            .get(camel)
            .or_else(|| observation.get(snake))
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_i64().map(|value| value as u64))
            })
            .map(|value| value as usize)
    };
    let observation_site = |observation: &Value| {
        observation.get("site").map(|site| {
            site.as_str()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| residue_to_string(site))
        })
    };
    let components = |values: &[VonMisesComponent]| {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let bound = legacy_vmm_component_bounds_95(value);
                json!({
                    "index": index,
                    "meanDegrees": value.mean_degrees,
                    "concentration": value.concentration,
                    "weight": value.weight,
                    "lower95Degrees": bound.map(|(lower, _)| lower),
                    "upper95Degrees": bound.map(|(_, upper)| upper),
                })
            })
            .collect::<Vec<_>>()
    };

    let references = sites
        .iter()
        .filter_map(|site| {
            let conformer = site.ensemble.conformers.first()?;
            let site_value = serde_json::to_value(&site.site.residue).ok()?;
            let site_label = residue_to_string(&site_value);
            let selected = observations
                .iter()
                .find(|observation| observation_site(observation).as_deref() == Some(&site_label));
            Some(json!({
                "site": site.site.residue,
                "phi": components(&conformer.priors.phi),
                "psi": components(&conformer.priors.psi),
                "selectedPhiComponent": selected.and_then(|value| component(value, "selectedPhiComponent", "selected_phi_component")),
                "selectedPsiComponent": selected.and_then(|value| component(value, "selectedPsiComponent", "selected_psi_component")),
            }))
        })
        .collect::<Vec<_>>();
    if let Some(object) = analysis.as_object_mut() {
        object.insert("attachmentReferences".into(), Value::Array(references));
    }
}

fn vmm_component_bounds_95(component: &VonMisesComponent) -> Option<(f64, f64)> {
    let half = glysys_energy::prior::credible_half_width(component.concentration, 0.95)
        .ok()?
        .to_degrees();
    Some((
        wrap_degrees(component.mean_degrees - half),
        wrap_degrees(component.mean_degrees + half),
    ))
}

fn legacy_vmm_component_bounds_95(component: &VonMisesComponent) -> Option<(f64, f64)> {
    if !component.concentration.is_finite() || component.concentration <= 1e-8 {
        return None;
    }
    let half = 1.96 * (1. / component.concentration).sqrt().to_degrees();
    Some((
        wrap_degrees(component.mean_degrees - half),
        wrap_degrees(component.mean_degrees + half),
    ))
}

fn wrap_degrees(value: f64) -> f64 {
    (value + 180.0).rem_euclid(360.0) - 180.0
}

fn append_torsion_reference_assets(analysis: &mut Value, artifacts: &[WorkflowArtifact]) {
    // A run can retain the same reference in more than one fetched asset
    // (for example an archetype and a cached branch-specific copy).  Keep a
    // deterministic, provenance-aware set rather than emitting duplicate
    // contour payloads in report.json.
    let mut references = BTreeMap::<String, Value>::new();
    let mut attachment_references = BTreeMap::<String, Value>::new();
    let mut glycan_cluster_references = BTreeMap::<String, Value>::new();
    for artifact in artifacts {
        if !artifact.name.to_ascii_lowercase().contains("reference")
            || !artifact.name.to_ascii_lowercase().ends_with(".json")
        {
            continue;
        }
        let Ok(raw) = artifact.data.text(&artifact.name) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        // The envelope's top-level clusters describe the GlycoShape
        // Level-1 conformer populations for the whole glycan. Preserve them
        // separately from linkage populations so a Build can show the
        // selected output conformer per attachment site, even when a glycan
        // has no internal glycosidic bond (for example a single GlcNAc).
        if let Some(clusters) = value.get("clusters").and_then(Value::as_array) {
            let glycan_id = artifact
                .name
                .strip_prefix("glycans/")
                .and_then(|name| name.split('-').next())
                .filter(|name| !name.is_empty())
                .or_else(|| value.get("gs_id").and_then(Value::as_str))
                .unwrap_or("unknown");
            glycan_cluster_references
                .entry(glycan_id.to_string())
                .or_insert_with(|| {
                    json!({
                        "glycanId": glycan_id,
                        "clusters": clusters,
                    })
                });
        }
        // New report bundles carry attachment VMMs beside internal linkage
        // references.  Parse them independently because an envelope can
        // legitimately contain both arrays.
        if let Some(items) = value
            .get("attachmentReferences")
            .or_else(|| value.get("attachment_references"))
            .and_then(Value::as_array)
        {
            for item in items {
                let key = item
                    .get("site")
                    .or_else(|| item.get("residue"))
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| item.to_string());
                attachment_references
                    .entry(key)
                    .or_insert_with(|| item.clone());
            }
        }
        if let Some(linkages) = value.get("linkages").and_then(Value::as_array) {
            for linkage in linkages {
                let linkage = enrich_torsion_reference(linkage, &value);
                references
                    .entry(torsion_reference_asset_key(&linkage))
                    .or_insert(linkage);
            }
        } else if let Some(items) = value.get("references").and_then(Value::as_array) {
            for item in items {
                references
                    .entry(torsion_reference_asset_key(item))
                    .or_insert_with(|| item.clone());
            }
        } else if value.get("canonicalLinkage").is_some()
            || value.get("canonical_linkage").is_some()
        {
            references
                .entry(torsion_reference_asset_key(&value))
                .or_insert(value);
        }
    }
    if references.is_empty()
        && attachment_references.is_empty()
        && glycan_cluster_references.is_empty()
    {
        return;
    }
    if let Some(object) = analysis.as_object_mut() {
        if !references.is_empty() {
            object.insert(
                "torsionReferences".into(),
                Value::Array(references.into_values().collect()),
            );
        }
        if !attachment_references.is_empty() {
            object.insert(
                "attachmentReferences".into(),
                Value::Array(attachment_references.into_values().collect()),
            );
        }
        if !glycan_cluster_references.is_empty() {
            object.insert(
                "glycanClusterReferences".into(),
                Value::Array(glycan_cluster_references.into_values().collect()),
            );
        }
    }
}

/// Copy version/source/hash metadata from a reference envelope onto each
/// linkage record.  The API returns one envelope containing many linkages,
/// while the browser report persists one self-contained reference per branch.
fn enrich_torsion_reference(linkage: &Value, envelope: &Value) -> Value {
    let Some(linkage_object) = linkage.as_object() else {
        return linkage.clone();
    };
    let Some(envelope_object) = envelope.as_object() else {
        return linkage.clone();
    };
    let mut object = linkage_object.clone();
    for (target, aliases) in [
        ("version", ["version", "referenceVersion"]),
        ("sourceDatabase", ["sourceDatabase", "source_database"]),
        ("source_database", ["source_database", "sourceDatabase"]),
        ("contentHash", ["contentHash", "content_hash"]),
        ("content_hash", ["content_hash", "contentHash"]),
    ] {
        if object.contains_key(target) {
            continue;
        }
        if let Some(value) = aliases
            .iter()
            .find_map(|key| envelope_object.get(*key))
            .cloned()
        {
            object.insert(target.into(), value);
        }
    }
    Value::Object(object)
}

/// Add the stable, reference-backed assessment to accepted internal linkage
/// points. Internal population tails remain advisory; this function only
/// enriches report data and never changes workflow status.
fn annotate_torsion_observations(analysis: &mut Value) {
    let Some(object) = analysis.as_object_mut() else {
        return;
    };
    let references = object
        .get("torsionReferences")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if references.is_empty() {
        return;
    }
    let Some(observations) = object
        .get_mut("torsionObservations")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for observation in observations {
        let reference = torsion_reference_for_observation(observation, &references);
        let Some(observation_object) = observation.as_object_mut() else {
            continue;
        };
        let Some(reference) = reference else {
            continue;
        };
        let reference_object = reference.as_object();
        let contour = reference_object.and_then(|item| {
            item.get("phiPsi")
                .or_else(|| item.get("phi_psi"))
                .or_else(|| item.get("phi_psi_grid"))
        });
        let Some(contour) = contour.and_then(Value::as_object) else {
            continue;
        };
        let bins = contour
            .get("bins")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .or_else(|| {
                contour
                    .get("grid")
                    .or_else(|| contour.get("counts"))
                    .and_then(Value::as_array)
                    .map(|values| (values.len() as f64).sqrt() as usize)
            })
            .unwrap_or(0);
        let grid = contour
            .get("grid")
            .or_else(|| contour.get("counts"))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| {
                        value
                            .as_f64()
                            .or_else(|| value.as_i64().map(|number| number as f64))
                            .or_else(|| value.as_u64().map(|number| number as f64))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let phi = observation_object
            .get("phiDegrees")
            .or_else(|| observation_object.get("phi_degrees"))
            .and_then(value_as_f64);
        let psi = observation_object
            .get("psiDegrees")
            .or_else(|| observation_object.get("psi_degrees"))
            .and_then(value_as_f64);
        let (Some(phi), Some(psi), Some(bins)) = (phi, psi, (bins > 0).then_some(bins)) else {
            continue;
        };
        let index = periodic_bin(phi, bins) + periodic_bin(psi, bins) * bins;
        let density = grid.get(index).copied().unwrap_or(0.0).max(0.0);
        let total = grid
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .sum::<f64>();
        if total <= 0.0 {
            continue;
        }
        let mass = grid
            .iter()
            .filter(|value| value.is_finite() && **value + f64::EPSILON >= density)
            .copied()
            .sum::<f64>();
        let percentile = (mass / total * 100.0).clamp(0.0, 100.0);
        let assessment = if percentile <= 50.0 {
            "core"
        } else if percentile <= 80.0 {
            "allowed"
        } else if percentile <= 95.0 {
            "tail"
        } else {
            "outlier"
        };
        observation_object.insert("populationPercentile".into(), json!(percentile));
        observation_object.insert("assessment".into(), json!(assessment));
        let nearest = level_one_cluster_at(reference, phi, psi).or_else(|| {
            reference_object
                .and_then(|item| item.get("populations"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|population| {
                    population
                        .get("level")
                        .and_then(Value::as_u64)
                        .is_none_or(|level| level == 1)
                })
                .filter_map(|population| {
                    let item = population.as_object()?;
                    let population_phi = item
                        .get("medoidPhi")
                        .or_else(|| item.get("medoid_phi"))
                        .and_then(value_as_f64)?;
                    let population_psi = item
                        .get("medoidPsi")
                        .or_else(|| item.get("medoid_psi"))
                        .and_then(value_as_f64)?;
                    let index = item.get("index").and_then(Value::as_u64)?;
                    let distance = circular_distance(phi, population_phi).powi(2)
                        + circular_distance(psi, population_psi).powi(2);
                    Some((distance, index))
                })
                .min_by(|left, right| left.0.total_cmp(&right.0))
                .map(|(_, index)| index)
        });
        if let Some(nearest) = nearest {
            observation_object.insert("nearestPopulation".into(), json!(nearest));
        }
    }
}

fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|number| number as f64))
        .or_else(|| value.as_u64().map(|number| number as f64))
}

fn periodic_bin(angle: f64, bins: usize) -> usize {
    ((angle + 180.0).rem_euclid(360.0) / 360.0 * bins as f64)
        .floor()
        .clamp(0.0, bins.saturating_sub(1) as f64) as usize
}

fn circular_distance(first: f64, second: f64) -> f64 {
    (first - second + 180.0).rem_euclid(360.0) - 180.0
}

fn torsion_reference_asset_key(value: &Value) -> String {
    let object = value.as_object();
    let string = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| object.and_then(|object| object.get(*name)))
            .and_then(Value::as_str)
            .unwrap_or_default()
    };
    let branch = object
        .and_then(|object| {
            object
                .get("branchPath")
                .or_else(|| object.get("branch_path"))
        })
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| serde_json::to_string(value).unwrap_or_default())
                })
                .collect::<Vec<_>>()
                .join("/")
        })
        .unwrap_or_default();
    let canonical = string(&["canonicalLinkage", "canonical_linkage", "name"]);
    let version = string(&["version", "sourceDatabase", "source_database"]);
    let content_hash = string(&["contentHash", "content_hash"]);
    // Older assets do not carry a content hash.  Their canonical serialized
    // value still gives us a stable key while preserving distinct branch
    // paths and versions.
    let payload = if content_hash.is_empty() {
        serde_json::to_string(value).unwrap_or_default()
    } else {
        content_hash.to_string()
    };
    format!("{canonical}|{branch}|{version}|{payload}")
}

fn residue_to_string(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    let object = value.as_object();
    let chain = object
        .and_then(|object| object.get("chain"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let number = object
        .and_then(|object| {
            object
                .get("residue_number")
                .or_else(|| object.get("residueNumber"))
                .or_else(|| object.get("number"))
        })
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let insertion = object
        .and_then(|object| {
            object
                .get("insertion_code")
                .or_else(|| object.get("insertionCode"))
        })
        .and_then(Value::as_str)
        .unwrap_or("");
    format!("{chain}:{number}{insertion}")
}

fn torsion_csv(values: &[Value], attachments: &[Value]) -> String {
    let mut output = String::from(
        "domain,origin,stage,frame,glycan_index,glycan,site,branch_path,linkage,phi_degrees,psi_degrees,omega_degrees,population_percentile,assessment,selected_population,selected_phi_component,selected_psi_component,involved_atoms,policy_version\n",
    );
    for value in values {
        let object = value.as_object();
        let field = |names: &[&str]| {
            names
                .iter()
                .find_map(|name| object.and_then(|object| object.get(*name)))
        };
        let text = |value: Option<&Value>| match value {
            Some(Value::String(value)) => value.replace(',', " "),
            Some(Value::Array(values)) => values
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(";"),
            Some(value) => value.to_string().replace(',', " "),
            None => String::new(),
        };
        let text_default = |value: Option<&Value>, fallback: &str| {
            let value = text(value);
            if value.is_empty() {
                fallback.to_string()
            } else {
                value
            }
        };
        let numeric = |value: Option<&Value>| {
            value
                .and_then(Value::as_f64)
                .or_else(|| value.and_then(Value::as_i64).map(|value| value as f64))
                .or_else(|| value.and_then(Value::as_u64).map(|value| value as f64))
                .map(|value| value.to_string())
                .unwrap_or_default()
        };
        let row = [
            text_default(field(&["domain"]), "internal_glycosidic"),
            text_default(field(&["origin"]), "accepted"),
            text(field(&["stage"])),
            field(&["frame"])
                .and_then(Value::as_u64)
                .map(|value| value.to_string())
                .unwrap_or_default(),
            field(&["glycanIndex", "glycan_index"])
                .and_then(Value::as_u64)
                .map(|value| value.to_string())
                .unwrap_or_default(),
            text(field(&["glycan", "glycanId", "glycan_id"])),
            text(field(&["site"])),
            text(field(&["branchPath", "branch_path"])),
            text(field(&["linkage"])),
            numeric(field(&["phiDegrees", "phi_degrees"])),
            numeric(field(&["psiDegrees", "psi_degrees"])),
            numeric(field(&["omegaDegrees", "omega_degrees"])),
            numeric(field(&["populationPercentile", "population_percentile"])),
            text(field(&["assessment"])),
            numeric(field(&["selectedPopulation", "selected_population"])),
            numeric(field(&["selectedPhiComponent", "selected_phi_component"])),
            numeric(field(&["selectedPsiComponent", "selected_psi_component"])),
            text(field(&["involvedAtoms", "involved_atoms"])),
            text(field(&["policyVersion", "policy_version", "policy"])),
        ];
        output.push_str(&row.join(","));
        output.push('\n');
    }
    for value in attachments {
        let object = value.as_object();
        let field = |names: &[&str]| {
            names
                .iter()
                .find_map(|name| object.and_then(|object| object.get(*name)))
        };
        let text = |value: Option<&Value>| match value {
            Some(Value::String(value)) => value.replace(',', " "),
            Some(Value::Array(values)) => values
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(";"),
            Some(value) => value.to_string().replace(',', " "),
            None => String::new(),
        };
        let text_default = |value: Option<&Value>, fallback: &str| {
            let value = text(value);
            if value.is_empty() {
                fallback.to_string()
            } else {
                value
            }
        };
        let numeric = |value: Option<&Value>| {
            value
                .and_then(Value::as_f64)
                .or_else(|| value.and_then(Value::as_i64).map(|value| value as f64))
                .or_else(|| value.and_then(Value::as_u64).map(|value| value as f64))
                .map(|value| value.to_string())
                .unwrap_or_default()
        };
        let row = [
            text_default(field(&["domain"]), "attachment"),
            text_default(field(&["origin"]), "accepted"),
            text(field(&["stage"])),
            field(&["frame"])
                .and_then(Value::as_u64)
                .map(|value| value.to_string())
                .unwrap_or_default(),
            field(&["glycanIndex", "glycan_index"])
                .and_then(Value::as_u64)
                .map(|value| value.to_string())
                .unwrap_or_default(),
            text(field(&["glycan", "glycanId", "glycan_id"])),
            text(field(&["site", "residue"])),
            text(field(&["branchPath", "branch_path"])),
            text(field(&["linkage"])),
            numeric(field(&["phiDegrees", "phi_degrees"])),
            numeric(field(&["psiDegrees", "psi_degrees"])),
            numeric(field(&["omegaDegrees", "omega_degrees"])),
            numeric(field(&["populationPercentile", "population_percentile"])),
            text(field(&["assessment"])),
            numeric(field(&["selectedPopulation", "selected_population"])),
            numeric(field(&["selectedPhiComponent", "selected_phi_component"])),
            numeric(field(&["selectedPsiComponent", "selected_psi_component"])),
            text(field(&["involvedAtoms", "involved_atoms"])),
            text(field(&["policyVersion", "policy_version", "policy"])),
        ];
        output.push_str(&row.join(","));
        output.push('\n');
    }
    output
}

/// Flatten the versioned energy explanation into a small, append-friendly CSV
/// artifact.  The rows deliberately retain their diagnostic kind so global
/// physical components, per-glycan cross terms, and glycosidic torsions are
/// never mistaken for one additive partition of the same energy.
pub fn energy_analysis_csv(analysis: &Value) -> String {
    let mut output = String::from(
        "kind,site,glycan,linkage,atoms,component,value,units,backend,drives_selection\n",
    );
    let Some(record) = analysis
        .get("energyAnalysis")
        .or_else(|| analysis.get("energy_analysis"))
        .and_then(Value::as_object)
    else {
        return output;
    };
    let units = record
        .get("units")
        .and_then(Value::as_str)
        .unwrap_or("kcal/mol");
    let backend = record
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let drives_selection = record
        .get("drivesSelection")
        .or_else(|| record.get("drives_selection"))
        .and_then(Value::as_bool)
        .map(|value| value.to_string())
        .unwrap_or_default();
    let cell = |value: &str| {
        if value.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", value.replace('"', "\"\""))
        } else {
            value.to_string()
        }
    };
    let number = |value: Option<&Value>| {
        value
            .and_then(Value::as_f64)
            .map(|value| format!("{value:.9}"))
            .unwrap_or_default()
    };
    let components = record.get("components").and_then(Value::as_object);
    for (name, key) in [
        ("bonds", "bonds"),
        ("angles", "angles"),
        ("proper_torsions", "properTorsions"),
        ("improper_torsions", "improperTorsions"),
        ("van_der_waals", "vanDerWaals"),
        ("electrostatics", "electrostatics"),
        ("generalized_born", "generalizedBorn"),
        ("surface_area", "surfaceArea"),
        ("restraints", "restraints"),
        ("dispersion_correction", "dispersionCorrection"),
    ] {
        let value = number(components.and_then(|values| values.get(key)));
        output.push_str(&format!(
            "global,,,,,{},{},{},{},{}\n",
            cell(name),
            value,
            cell(units),
            cell(backend),
            drives_selection
        ));
    }
    if let Some(score) = record
        .get("selectedScore")
        .or_else(|| record.get("selected_score"))
    {
        output.push_str(&format!(
            "objective,,,,,selected_score,{},{},{},{}\n",
            number(Some(score)),
            cell(units),
            cell(backend),
            drives_selection
        ));
    }
    if let Some(remainder) = record
        .get("diagnosticRemainder")
        .or_else(|| record.get("diagnostic_remainder"))
    {
        output.push_str(&format!(
            "diagnostic,,,,,remainder,{},{},{},false\n",
            number(Some(remainder)),
            cell(units),
            cell(backend)
        ));
    }
    if let Some(entries) = record
        .get("perGlycanInteractions")
        .or_else(|| record.get("per_glycan_interactions"))
        .and_then(Value::as_array)
    {
        for entry in entries.iter().filter_map(Value::as_object) {
            let site = entry.get("site").and_then(Value::as_str).unwrap_or("");
            let glycan = entry
                .get("glycanId")
                .or_else(|| entry.get("glycan_id"))
                .and_then(Value::as_str)
                .unwrap_or("");
            for (component, key) in [
                ("van_der_waals", "vanDerWaals"),
                ("electrostatics", "electrostatics"),
                ("total", "total"),
            ] {
                output.push_str(&format!(
                    "protein_glycan,{},{},,,{},{},{},{},{}\n",
                    cell(site),
                    cell(glycan),
                    cell(component),
                    number(entry.get(key)),
                    cell(units),
                    cell(backend),
                    drives_selection
                ));
            }
        }
    }
    if let Some(entries) = record
        .get("glycosidicTorsions")
        .or_else(|| record.get("glycosidic_torsions"))
        .and_then(Value::as_array)
    {
        for entry in entries.iter().filter_map(Value::as_object) {
            let site = entry.get("site").and_then(Value::as_str).unwrap_or("");
            let linkage = entry.get("linkage").and_then(Value::as_str).unwrap_or("");
            let atoms = entry
                .get("atoms")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(Value::to_string)
                        .collect::<Vec<_>>()
                        .join("-")
                })
                .unwrap_or_default();
            output.push_str(&format!(
                "glycosidic_torsion,{},{},{},{},{},{},{},{},{}\n",
                cell(site),
                "",
                cell(linkage),
                cell(&atoms),
                "energy",
                number(entry.get("energy")),
                cell(units),
                cell(backend),
                drives_selection
            ));
        }
    }
    output
}

fn glycosidic_torsion_observations(
    structure: &Structure,
    stage: &str,
    frame: Option<usize>,
    references: &[Value],
) -> Vec<Value> {
    let glycan_residues = structure
        .metadata()
        .glycan_trees
        .iter()
        .flat_map(|tree| tree.residue_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    if glycan_residues.is_empty() {
        return Vec::new();
    }
    // Many coordinate exports omit CONECT records for glycosidic junctions.
    // Recover only short heavy-atom contacts inside a declared tree; this is
    // enough to measure linkages without inventing bonds between unrelated
    // glycans that merely happen to be nearby.
    let mut bond_pairs = structure.bonds().into_iter().collect::<BTreeSet<_>>();
    let residue_names = structure
        .residues()
        .into_iter()
        .map(|residue| (residue.id, residue.name))
        .collect::<BTreeMap<_, _>>();
    let atoms = structure.atoms();
    for tree in &structure.metadata().glycan_trees {
        let tree_ids = tree.residue_ids.iter().collect::<BTreeSet<_>>();
        let tree_atoms = atoms
            .iter()
            .filter(|atom| {
                tree_ids.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        for (index, left) in tree_atoms.iter().enumerate() {
            for right in tree_atoms.iter().skip(index + 1) {
                if left.residue == right.residue {
                    continue;
                }
                let distance = ((left.position.x - right.position.x).powi(2)
                    + (left.position.y - right.position.y).powi(2)
                    + (left.position.z - right.position.z).powi(2))
                .sqrt();
                if likely_glycosidic_atom_pair(left, right) && (1.1..=1.85).contains(&distance) {
                    bond_pairs.insert(if left.id <= right.id {
                        (left.id, right.id)
                    } else {
                        (right.id, left.id)
                    });
                }
            }
        }
    }
    let mut observations = Vec::new();
    for (first, second) in bond_pairs {
        let Some(first) = structure.atom(first) else {
            continue;
        };
        let Some(second) = structure.atom(second) else {
            continue;
        };
        if first.residue == second.residue
            || !glycan_residues.contains(&first.residue)
            || !glycan_residues.contains(&second.residue)
        {
            continue;
        }
        let (donor, acceptor) = if is_anomeric_atom(&first.name)
            && is_acceptor_atom(
                &second.name,
                residue_names
                    .get(&second.residue)
                    .map(String::as_str)
                    .unwrap_or_default(),
            ) {
            (first, second)
        } else if is_anomeric_atom(&second.name)
            && is_acceptor_atom(
                &first.name,
                residue_names
                    .get(&first.residue)
                    .map(String::as_str)
                    .unwrap_or_default(),
            )
        {
            (second, first)
        } else {
            continue;
        };
        let acceptor_name = acceptor.name.trim().to_ascii_uppercase();
        let acceptor_position = acceptor_name
            .strip_prefix('O')
            .and_then(|value| value.parse::<u8>().ok());
        let Some(acceptor_position) = acceptor_position else {
            continue;
        };
        let donor_name = residue_names.get(&donor.residue).cloned();
        let acceptor_name = residue_names.get(&acceptor.residue).cloned();
        let pdb_branch_path = vec![
            donor_name
                .clone()
                .unwrap_or_default()
                .trim()
                .to_ascii_uppercase(),
            donor
                .name
                .trim()
                .to_ascii_uppercase()
                .strip_prefix('C')
                .unwrap_or_default()
                .to_string(),
            acceptor_name
                .clone()
                .unwrap_or_default()
                .trim()
                .to_ascii_uppercase(),
            acceptor_position.to_string(),
        ];
        let torsion_reference = torsion_reference_for_pdb_occurrence(
            references,
            &pdb_branch_path,
            &donor.residue,
            &acceptor.residue,
        );
        let donor_name_text = donor_name.as_deref().unwrap_or_default();
        let donor_ring_atom = if is_furanose_residue(donor_name_text) {
            structure
                .find_atom(&donor.residue, "O4")
                .or_else(|| structure.find_atom(&donor.residue, "O5"))
                .or_else(|| structure.find_atom(&donor.residue, "O6"))
        } else if is_sialic_residue(donor_name_text) {
            structure
                .find_atom(&donor.residue, "O6")
                .or_else(|| structure.find_atom(&donor.residue, "O5"))
        } else {
            structure
                .find_atom(&donor.residue, "O5")
                .or_else(|| structure.find_atom(&donor.residue, "O6"))
                .or_else(|| structure.find_atom(&donor.residue, "O4"))
        };
        let donor_ring = donor_ring_atom
            .and_then(|id| structure.atom(id))
            .map(|atom| atom.position);
        let acceptor_carbon = structure
            .find_atom(&acceptor.residue, &format!("C{acceptor_position}"))
            .and_then(|id| structure.atom(id))
            .map(|atom| atom.position);
        let previous = acceptor_position
            .checked_sub(1)
            .and_then(|position| structure.find_atom(&acceptor.residue, &format!("C{position}")))
            .and_then(|id| structure.atom(id))
            .map(|atom| atom.position);
        let phi = torsion_reference
            .and_then(|reference| {
                reference_defined_torsion(
                    structure,
                    reference,
                    "phi",
                    &donor.residue,
                    &acceptor.residue,
                )
            })
            .or_else(|| {
                donor_ring.zip(acceptor_carbon).map(|(ring, carbon)| {
                    glycoshape_dihedral_degrees(ring, donor.position, acceptor.position, carbon)
                })
            });
        let psi = torsion_reference
            .and_then(|reference| {
                reference_defined_torsion(
                    structure,
                    reference,
                    "psi",
                    &donor.residue,
                    &acceptor.residue,
                )
            })
            .or_else(|| {
                acceptor_carbon.zip(previous).map(|(carbon, previous)| {
                    glycoshape_dihedral_degrees(donor.position, acceptor.position, carbon, previous)
                })
            });
        // 1→6 linkages have a third periodic torsion around the acceptor
        // sidechain. Keep it optional for other acceptor positions so older
        // references and consumers remain schema-compatible.
        let omega = torsion_reference
            .and_then(|reference| {
                reference_defined_torsion(
                    structure,
                    reference,
                    "omega",
                    &donor.residue,
                    &acceptor.residue,
                )
            })
            .or_else(|| {
                (acceptor_position == 6)
                    .then(|| {
                        // GlycoShape's standard pyranose omega convention is
                        // C4-C5-C6-O6.  The former O5-C5-C6-O6 fallback
                        // measured a different torsion and could not be
                        // compared with the source population.
                        let first = structure
                            .find_atom(&acceptor.residue, "C4")
                            .and_then(|id| structure.atom(id))
                            .map(|atom| atom.position)?;
                        let c5 = structure
                            .find_atom(&acceptor.residue, "C5")
                            .and_then(|id| structure.atom(id))
                            .map(|atom| atom.position)?;
                        let c6 = structure
                            .find_atom(&acceptor.residue, "C6")
                            .and_then(|id| structure.atom(id))
                            .map(|atom| atom.position)?;
                        Some(glycoshape_dihedral_degrees(
                            first,
                            c5,
                            c6,
                            acceptor.position,
                        ))
                    })
                    .flatten()
            });
        let canonical = format!(
            "{}{}-{}{}",
            donor_name.clone().unwrap_or_default().to_ascii_uppercase(),
            donor
                .name
                .trim()
                .to_ascii_uppercase()
                .strip_prefix('C')
                .unwrap_or_default(),
            acceptor_name
                .clone()
                .unwrap_or_default()
                .to_ascii_uppercase(),
            acceptor_position
        );
        let glycan_index = structure
            .metadata()
            .glycan_trees
            .iter()
            .position(|tree| tree.residue_ids.contains(&donor.residue))
            .map(|index| index + 1)
            .unwrap_or(0);
        let site = structure
            .metadata()
            .glycan_trees
            .iter()
            .find(|tree| tree.residue_ids.contains(&donor.residue))
            .and_then(|tree| tree.attachment_site.as_ref())
            .map(ToString::to_string);
        observations.push(json!({
            "domain": "internal_glycosidic",
            "origin": "accepted",
            "stage": stage,
            "frame": frame,
            "glycanIndex": glycan_index,
            "site": site,
            "branchPath": pdb_branch_path,
            "linkage": canonical,
            "involvedAtoms": [format!("{}:{}", donor.residue, donor.name), format!("{}:{}", acceptor.residue, acceptor.name)],
            "phiDegrees": phi,
            "psiDegrees": psi,
            "omegaDegrees": omega,
            "assessment": "no_reference",
            "policyVersion": "glycoshape-torsion-v1",
            "angleConvention": "glycoshape-v1",
        }));
    }
    observations
}

fn torsion_reference_for_pdb_branch<'a>(
    references: &'a [Value],
    branch_path: &[String],
) -> Option<&'a Value> {
    references
        .iter()
        .find(|reference| reference_matches_pdb_branch(reference, branch_path))
}

fn reference_matches_pdb_branch(reference: &Value, branch_path: &[String]) -> bool {
    reference
        .get("pdb_branch_paths")
        .or_else(|| reference.get("pdbBranchPaths"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_array)
        .any(|candidate| {
            candidate.len() == branch_path.len()
                && candidate.iter().zip(branch_path).all(|(left, right)| {
                    left.as_str()
                        .is_some_and(|left| left.trim().eq_ignore_ascii_case(right.trim()))
                })
        })
}

fn residue_occurrence_token(value: &str) -> Option<String> {
    value.split(':').rev().find_map(|part| {
        let part = part.trim();
        let digits = part
            .chars()
            .take_while(|character| character.is_ascii_digit() || *character == '-')
            .count();
        (digits > 0).then(|| part.to_ascii_uppercase())
    })
}

fn residue_id_occurrence_token(value: &ResidueId) -> String {
    format!(
        "{}{}",
        value.number,
        value
            .insertion_code
            .map(|code| code.to_string())
            .unwrap_or_default()
    )
    .to_ascii_uppercase()
}

fn reference_matches_residue_occurrence(
    reference: &Value,
    donor_token: &str,
    acceptor_token: &str,
) -> bool {
    reference
        .get("pdb_output_residue_paths")
        .or_else(|| reference.get("pdbOutputResiduePaths"))
        .or_else(|| reference.get("pdb_residue_paths"))
        .or_else(|| reference.get("pdbResiduePaths"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_array)
        .any(|path| {
            path.len() >= 2
                && path[0]
                    .as_str()
                    .and_then(residue_occurrence_token)
                    .is_some_and(|token| token == donor_token)
                && path[1]
                    .as_str()
                    .and_then(residue_occurrence_token)
                    .is_some_and(|token| token == acceptor_token)
        })
}

fn torsion_reference_for_pdb_occurrence<'a>(
    references: &'a [Value],
    branch_path: &[String],
    donor: &ResidueId,
    acceptor: &ResidueId,
) -> Option<&'a Value> {
    let donor_token = residue_id_occurrence_token(donor);
    let acceptor_token = residue_id_occurrence_token(acceptor);
    references
        .iter()
        .find(|reference| {
            reference_matches_pdb_branch(reference, branch_path)
                && reference_matches_residue_occurrence(reference, &donor_token, &acceptor_token)
        })
        .or_else(|| torsion_reference_for_pdb_branch(references, branch_path))
}

fn torsion_reference_for_observation<'a>(
    observation: &Value,
    references: &'a [Value],
) -> Option<&'a Value> {
    let branch_path = observation
        .get("branchPath")
        .or_else(|| observation.get("branch_path"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if !branch_path.is_empty() {
        let residue_tokens = observation
            .get("involvedAtoms")
            .or_else(|| observation.get("involved_atoms"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter_map(residue_occurrence_token)
            .take(2)
            .collect::<Vec<_>>();
        if residue_tokens.len() == 2 {
            if let Some(reference) = references.iter().find(|reference| {
                reference_matches_pdb_branch(reference, &branch_path)
                    && reference_matches_residue_occurrence(
                        reference,
                        &residue_tokens[0],
                        &residue_tokens[1],
                    )
            }) {
                return Some(reference);
            }
        }
        if let Some(reference) = torsion_reference_for_pdb_branch(references, &branch_path) {
            return Some(reference);
        }
    }
    let linkage = observation.get("linkage").and_then(Value::as_str)?;
    references.iter().find(|reference| {
        reference
            .get("canonicalLinkage")
            .or_else(|| reference.get("canonical_linkage"))
            .or_else(|| reference.get("name"))
            .and_then(Value::as_str)
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(linkage))
            || reference
                .get("pdbLinkages")
                .or_else(|| reference.get("pdb_linkages"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .any(|candidate| candidate.eq_ignore_ascii_case(linkage))
    })
}

fn reference_defined_torsion(
    structure: &Structure,
    reference: &Value,
    axis: &str,
    donor: &ResidueId,
    acceptor: &ResidueId,
) -> Option<f64> {
    let definitions = reference
        .get("angle_definitions")
        .or_else(|| reference.get("angleDefinitions"))?;
    let atoms = definitions.get(axis)?.as_array()?;
    if atoms.len() != 4 {
        return None;
    }
    let positions = atoms
        .iter()
        .map(|atom| {
            let role = atom
                .get("residue_role")
                .or_else(|| atom.get("residueRole"))
                .and_then(Value::as_str)?;
            let residue = match role {
                "donor" => donor,
                "acceptor" => acceptor,
                _ => return None,
            };
            let atom_name = atom
                .get("atom_name")
                .or_else(|| atom.get("atomName"))
                .and_then(Value::as_str)?;
            structure
                .find_atom(residue, atom_name)
                .and_then(|id| structure.atom(id))
                .map(|atom| atom.position)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(glycoshape_dihedral_degrees(
        positions[0],
        positions[1],
        positions[2],
        positions[3],
    ))
}

fn likely_glycosidic_atom_pair(
    left: &glysys::StructureAtom,
    right: &glysys::StructureAtom,
) -> bool {
    let left_name = left.name.trim().to_ascii_uppercase();
    let right_name = right.name.trim().to_ascii_uppercase();
    let anomeric = |name: &str| matches!(name, "C1" | "C2" | "C1A" | "C2A");
    let oxygen = |name: &str| name.starts_with('O');
    (anomeric(&left_name) && oxygen(&right_name)) || (anomeric(&right_name) && oxygen(&left_name))
}

fn is_anomeric_atom(name: &str) -> bool {
    let name = name.trim().to_ascii_uppercase();
    matches!(name.as_str(), "C1" | "C2" | "C1A" | "C2A")
}

fn is_furanose_residue(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "ARA" | "ARB" | "AFL" | "RIB"
    )
}

fn is_sialic_residue(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
    )
}

fn is_acceptor_atom(name: &str, residue_name: &str) -> bool {
    let name = name.trim().to_ascii_uppercase();
    // Exclude the ring oxygen for the acceptor component only. O4 is a valid
    // 1→4 acceptor on pyranoses, while O5 is exocyclic on furanoses and
    // sialic acids. This avoids dropping real linkages while still avoiding
    // ring-oxygen pseudo-bonds inferred from omitted CONECT records.
    let ring_oxygen = if is_furanose_residue(residue_name) {
        "O4"
    } else if is_sialic_residue(residue_name) {
        "O6"
    } else {
        "O5"
    };
    name.starts_with('O') && name != ring_oxygen
}

fn dihedral_degrees(
    first: glysys::Vec3,
    second: glysys::Vec3,
    third: glysys::Vec3,
    fourth: glysys::Vec3,
) -> f64 {
    let sub = |a: glysys::Vec3, b: glysys::Vec3| [a.x - b.x, a.y - b.y, a.z - b.z];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    let b1 = sub(second, first);
    let b2 = sub(third, second);
    let b3 = sub(fourth, third);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let b2_norm = norm(b2);
    let n1_norm = norm(n1);
    let n2_norm = norm(n2);
    if n1_norm <= 1.0e-12 || n2_norm <= 1.0e-12 || b2_norm <= 1.0e-12 {
        return 0.0;
    }
    let b2_unit = [b2[0] / b2_norm, b2[1] / b2_norm, b2[2] / b2_norm];
    let n1_unit = [n1[0] / n1_norm, n1[1] / n1_norm, n1[2] / n1_norm];
    let n2_unit = [n2[0] / n2_norm, n2[1] / n2_norm, n2[2] / n2_norm];
    let cosine = dot(n1_unit, n2_unit);
    let sine = dot(cross(n1_unit, n2_unit), b2_unit);
    sine.atan2(cosine).to_degrees()
}

/// GlycoShapeAnalysisPipeline and RDKit use the ordered-atom plane-normal
/// convention implemented by `dihedral_degrees`. Keep this report-only
/// wrapper explicit so search/build attachment geometry remains independent.
fn glycoshape_dihedral_degrees(
    first: glysys::Vec3,
    second: glysys::Vec3,
    third: glysys::Vec3,
    fourth: glysys::Vec3,
) -> f64 {
    wrap_degrees(dihedral_degrees(first, second, third, fourth))
}

fn assignment_report_sites(request: &ReGlycoRunRequestV1, status: &str) -> Vec<ReportSite> {
    request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded)
        .map(|assignment| ReportSite {
            site: assignment.site.clone(),
            glycan_id: Some(assignment.glycan_id.clone()),
            status: status.into(),
            score: None,
            details: BTreeMap::new(),
        })
        .collect()
}

#[cfg(feature = "full")]
fn execute_full(
    request: &ReGlycoRunRequestV1,
    assets: &InputAssets,
    protein: Structure,
    builder: &SystemBuilder,
    replacements: &[ReplacementRecord],
    control: &mut impl WorkflowControl,
) -> Result<WorkflowBundle> {
    let input = input_text(request, assets)?;
    let mut warnings = vec![
        "This workflow is experimental; inspect validation and provenance before downstream use."
            .into(),
    ];
    let mut artifacts = exportable_input_support_artifacts(assets, &request.input.asset);
    let mut sites = request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded)
        .map(|assignment| ReportSite {
            site: assignment.site.clone(),
            glycan_id: Some(assignment.glycan_id.clone()),
            status: "refined".into(),
            score: None,
            details: BTreeMap::new(),
        })
        .collect::<Vec<_>>();

    let (final_structure, analysis) = match request.workflow {
        WorkflowId::Refine => {
            emit(
                control,
                "refine",
                "Optimizing the selected protein–glycan objective…",
                None,
                None,
            )?;
            let search_sites = search_sites(request, assets, &protein)?;
            if search_sites.is_empty() {
                let system = builder.prepare_structure(&protein)?;
                let relaxed =
                    relax_with_progress(&protein, &system, &relax_options(request), |event| {
                        control.progress(progress_relax(event))
                    })?;
                let analysis = json!({
                    "objective": request.options.scoring_mode,
                    "mode": "existing-glycoprotein",
                    "relaxation": relaxed.diagnostics,
                });
                return finish_workflow(
                    request,
                    input,
                    relaxed.structure,
                    WorkflowStatus::Succeeded,
                    warnings,
                    sites,
                    add_replacement_analysis(analysis, replacements),
                    artifacts,
                    control,
                );
            }
            let refine_input = protein.clone();
            let result = match refine_with_progress(
                RefineRequest {
                    protein,
                    nuisance_protein: None,
                    sites: search_sites,
                    search: search_config(request),
                    relaxation: relax_options(request),
                    objective: RefineObjective::StericEnergy,
                },
                builder,
                Some(builder),
                |event| control.progress(progress_refine(event)),
            ) {
                Ok(result) => result,
                Err(reglyco_refine::RefineError::Ensemble(EnsembleError::StrictVmmFailure {
                    diagnostics,
                })) => {
                    return finish_strict_failure(
                        request,
                        input,
                        refine_input,
                        warnings,
                        assignment_report_sites(request, "strict-search-failed"),
                        *diagnostics,
                        replacements,
                        artifacts,
                        control,
                    );
                }
                Err(error) => return Err(error.into()),
            };
            let analysis = json!({
                "objective": request.options.scoring_mode,
                "preStericScore": result.pre_steric_score,
                "postStericScore": result.post_steric_score,
                "search": result.search,
                "relaxation": result.relaxation,
                "nativeReport": result.report,
            });
            (result.relaxed_structure, analysis)
        }
        WorkflowId::Density => {
            emit(
                control,
                "density",
                "Decoding the density map and calibrating its kernel…",
                None,
                None,
            )?;
            let map_asset = assets
                .get("density.map")
                .ok_or_else(|| WorkflowError::MissingAsset("density.map".into()))?;
            let map = DensityMap::from_bytes("density.map", map_asset.bytes())?;
            let search_sites = search_sites(request, assets, &protein)?;
            let targets = request
                .assignments
                .iter()
                .filter(|assignment| !assignment.excluded)
                .map(|assignment| {
                    DensityTarget::for_site(&protein, &assignment.site.residue_id()).unwrap_or(
                        DensityTarget {
                            site: assignment.site.residue_id(),
                            glycan_residues: Vec::new(),
                        },
                    )
                })
                .collect::<Vec<_>>();
            if targets.is_empty() {
                return Err(WorkflowError::Invalid(
                    "density refinement requires at least one selected glycan site".into(),
                ));
            }
            let calibration_sites = targets
                .iter()
                .map(|target| target.site.clone())
                .collect::<Vec<_>>();
            let mut score_options = DensityScoreOptions {
                periodic: map.is_full_unit_cell(),
                support_threshold: request.options.density_support_threshold,
                ..DensityScoreOptions::default()
            };
            let calibration = DensityScorer::calibrate_sigma_from_protein(
                &map,
                &protein,
                &calibration_sites,
                score_options,
                &[0.65, 0.80, 1.00, 1.20, 1.40, 1.70],
            )?;
            score_options.sigma_angstrom = Some(calibration.selected_sigma_angstrom);
            score_options.glycan_b_factor = Some(calibration.estimated_glycan_b_factor);
            let scorer = DensityScorer::new(map, score_options)?;
            if search_sites.is_empty() {
                let score = scorer.score(&protein, &targets)?;
                warnings.push("The deposited glycans were density-scored in place; choose a replacement GlycoShape glycan to search alternate conformers.".into());
                let analysis = json!({ "density": { "existingModel": score, "sigmaCalibration": calibration } });
                return finish_workflow(
                    request,
                    input,
                    protein,
                    WorkflowStatus::Succeeded,
                    warnings,
                    sites,
                    add_replacement_analysis(analysis, replacements),
                    artifacts,
                    control,
                );
            }
            let mut density = DensityRefinementConfig::new(scorer, targets);
            density.effort = match request.options.density_effort {
                DensityEffort::Fast => NativeDensityEffort::Fast,
                DensityEffort::Adaptive => NativeDensityEffort::Adaptive,
                DensityEffort::Deep => NativeDensityEffort::Deep,
            };
            density.support_threshold = request.options.density_support_threshold;
            density.credible_mass = request.options.density_credible_mass;
            density.max_alternates = request.options.density_max_alternates;
            density.post_relax_energy = request.options.density_post_relax;
            emit(
                control,
                "density",
                "Searching density-supported glycan poses…",
                None,
                None,
            )?;
            let refine_input = protein.clone();
            let result = match refine_with_progress(
                RefineRequest {
                    protein,
                    nuisance_protein: None,
                    sites: search_sites,
                    search: search_config(request),
                    relaxation: relax_options(request),
                    objective: RefineObjective::Density(density),
                },
                builder,
                Some(builder),
                |event| control.progress(progress_refine(event)),
            ) {
                Ok(result) => result,
                Err(reglyco_refine::RefineError::Ensemble(EnsembleError::StrictVmmFailure {
                    diagnostics,
                })) => {
                    return finish_strict_failure(
                        request,
                        input,
                        refine_input,
                        warnings,
                        assignment_report_sites(request, "strict-search-failed"),
                        *diagnostics,
                        replacements,
                        artifacts,
                        control,
                    );
                }
                Err(error) => return Err(error.into()),
            };
            let density_summary = result
                .density
                .as_ref()
                .map(|fit| {
                    json!({
                        "map": fit.map,
                        "preRelax": fit.pre_relax,
                        "postRelax": fit.post_relax,
                        "candidates": fit.candidates,
                        "armEvidence": fit.arm_evidence,
                        "evaluations": fit.evaluations,
                        "stoppingReason": fit.stopping_reason,
                        "timings": fit.timings,
                        "warnings": fit.warnings,
                        "sigmaCalibration": fit.sigma_calibration,
                    })
                })
                .unwrap_or_else(|| json!({}));
            if let Some(fit) = &result.density {
                warnings.extend(fit.warnings.clone());
            }
            let analysis = json!({ "density": density_summary, "nativeReport": result.report });
            (result.relaxed_structure, analysis)
        }
        WorkflowId::Saxs => {
            emit(
                control,
                "saxs",
                "Reading the experimental SAXS curve…",
                None,
                None,
            )?;
            let curve = assets
                .get("saxs.dat")
                .ok_or_else(|| WorkflowError::MissingAsset("saxs.dat".into()))?
                .text("saxs.dat")?;
            let experimental = experimental_curve_from_str(curve)?;
            let parsed_models = parse_pdb_models(input, &build_options(request))?;
            let models = parsed_models
                .iter()
                .enumerate()
                .map(|(index, model)| adapt_structure(format!("model-{}", index + 1), model))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            emit(
                control,
                "saxs",
                "Calculating and fitting scattering curves…",
                None,
                None,
            )?;
            let fit_options = FitOptions::default();
            let pr_options = PrOptions::default();
            let analysis = match request.options.saxs_mode {
                SaxsMode::Model => serde_json::to_value(fit_single_models(
                    &models,
                    &experimental,
                    fit_options,
                    pr_options,
                )?)?,
                SaxsMode::Ensemble => serde_json::to_value(fit_unbiased_ensemble(
                    &models,
                    &experimental,
                    fit_options,
                    pr_options,
                )?)?,
                SaxsMode::Reweight => serde_json::to_value(reweight_ensemble(
                    &models,
                    &experimental,
                    fit_options,
                    MaximumEntropyOptions::default(),
                    pr_options,
                )?)?,
                SaxsMode::Occupancy => {
                    warnings.push("Occupancy mode currently reports per-model/site support; discrete occupancy refinement remains experimental.".into());
                    serde_json::to_value(fit_single_models(
                        &models,
                        &experimental,
                        fit_options,
                        pr_options,
                    )?)?
                }
            };
            artifacts.push(text_artifact(
                "experimental-saxs.dat",
                "text/plain",
                ArtifactRole::Input,
                curve,
            ));
            sites.clear();
            (
                protein,
                json!({
                    "mode": request.options.saxs_mode,
                    "experimental": { "q": experimental.q, "intensity": experimental.intensity, "sigma": experimental.sigma },
                    "fit": analysis
                }),
            )
        }
        _ => return Err(WorkflowError::Capability(request.workflow)),
    };

    finish_workflow(
        request,
        input,
        final_structure,
        WorkflowStatus::Succeeded,
        warnings,
        sites,
        add_replacement_analysis(analysis, replacements),
        artifacts,
        control,
    )
}

/// Public objective-driven refinement.  Density and SAXS are deliberately
/// kept out of this function so the public WASM entry point can compile the
/// safe search/relax path without exposing experimental workflow dispatch.
#[cfg(feature = "refine")]
fn execute_refine(
    request: &ReGlycoRunRequestV1,
    assets: &InputAssets,
    protein: Structure,
    builder: &SystemBuilder,
    replacements: &[ReplacementRecord],
    control: &mut impl WorkflowControl,
) -> Result<WorkflowBundle> {
    let input = input_text(request, assets)?;
    let mut warnings = Vec::new();
    let mut artifacts = exportable_input_support_artifacts(assets, &request.input.asset);
    let sites = request
        .assignments
        .iter()
        .filter(|assignment| !assignment.excluded)
        .map(|assignment| ReportSite {
            site: assignment.site.clone(),
            glycan_id: Some(assignment.glycan_id.clone()),
            status: "refined".into(),
            score: None,
            details: BTreeMap::new(),
        })
        .collect::<Vec<_>>();
    emit(
        control,
        "refine",
        "Optimizing the selected protein–glycan objective…",
        None,
        None,
    )?;
    let search_sites = search_sites(request, assets, &protein)?;
    if search_sites.is_empty() {
        let system = builder.prepare_structure(&protein)?;
        let relaxed = relax_with_progress(&protein, &system, &relax_options(request), |event| {
            control.progress(progress_relax(event))
        })?;
        return finish_workflow(
            request,
            input,
            relaxed.structure,
            WorkflowStatus::Succeeded,
            warnings,
            sites,
            add_replacement_analysis(
                json!({
                    "objective": request.options.scoring_mode,
                    "mode": "existing-glycoprotein",
                    "relaxation": relaxed.diagnostics,
                }),
                replacements,
            ),
            artifacts,
            control,
        );
    }
    // The public optimizer intentionally reuses the deterministic search and
    // staged relax primitives already included in the public worker.  The
    // full profile swaps in the richer density-aware refine crate below.
    emit(
        control,
        "search",
        "Searching objective-ranked glycan conformers…",
        None,
        None,
    )?;
    let cancelled = std::cell::Cell::new(control.cancelled());
    let outcome = match search_with_progress_cancelled(
        &protein,
        &search_sites,
        &search_config(request),
        builder,
        |event| {
            control.progress(progress_search(event));
            cancelled.set(control.cancelled());
        },
        || cancelled.get(),
    ) {
        Ok(outcome) => outcome,
        Err(EnsembleError::StrictVmmFailure { diagnostics }) => {
            return finish_strict_failure(
                request,
                input,
                protein.clone(),
                warnings,
                assignment_report_sites(request, "strict-search-failed"),
                *diagnostics,
                replacements,
                artifacts,
                control,
            );
        }
        Err(EnsembleError::Cancelled) => return Err(WorkflowError::Cancelled),
        Err(error) => return Err(error.into()),
    };
    let built = build_from_outcome(&protein, &search_sites, &outcome, builder, false)?;
    let mut analysis = json!({
        "objective": request.options.scoring_mode,
        "search": outcome.clone(),
        "relaxation": Value::Null,
    });
    append_attachment_reference_data(&mut analysis, &protein, &search_sites, &outcome.sites);
    append_attachment_observations_from_results(&mut analysis, &outcome.sites, "Optimize", None);
    let mut workflow_status = WorkflowStatus::Succeeded;
    let (final_structure, relaxation) = if request.options.post_relax {
        if let Some(system) = built.system {
            match relax_with_progress(
                &built.structure,
                &system,
                &relax_options(request),
                |event| control.progress(progress_relax(event)),
            ) {
                Ok(relaxed) => {
                    artifacts.push(text_artifact(
                        "pre-relax-optimize.pdb",
                        "chemical/x-pdb",
                        ArtifactRole::Structure,
                        built.structure.to_pdb_string(),
                    ));
                    append_attachment_analysis(&mut analysis, "Minimize");
                    (relaxed.structure, Some(relaxed.diagnostics))
                }
                Err(error) => {
                    workflow_status = WorkflowStatus::Partial;
                    warnings.push(format!(
                        "Optimize succeeded but relaxation failed; retaining the optimized parent: {error}"
                    ));
                    (built.structure.clone(), None)
                }
            }
        } else {
            (built.structure, None)
        }
    } else {
        (built.structure, None)
    };
    finish_workflow(
        request,
        input,
        final_structure,
        workflow_status,
        warnings,
        sites,
        {
            analysis["relaxation"] = serde_json::to_value(relaxation)?;
            add_replacement_analysis(analysis, replacements)
        },
        artifacts,
        control,
    )
}

fn parse_pdb_models(contents: &str, options: &BuildOptions) -> Result<Vec<Structure>> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let has_models = contents.lines().any(|line| line.starts_with("MODEL"));
    if !has_models {
        return Ok(vec![read_pdb_str(contents, options)?]);
    }
    for line in contents.lines() {
        if line.starts_with("MODEL") {
            current.clear();
        } else if line.starts_with("ENDMDL") {
            if !current.trim().is_empty() {
                blocks.push(read_pdb_str(&current, options)?);
            }
            current.clear();
        } else if !current.is_empty()
            || line.starts_with("ATOM")
            || line.starts_with("HETATM")
            || line.starts_with("LINK")
        {
            current.push_str(line);
            current.push('\n');
        }
    }
    if blocks.is_empty() {
        return Err(WorkflowError::Invalid(
            "multi-model PDB contains no complete MODEL/ENDMDL blocks".into(),
        ));
    }
    Ok(blocks)
}

#[cfg(feature = "full")]
fn progress_refine(event: RefineProgress) -> ProgressEvent {
    match event {
        RefineProgress::Phase { name } => ProgressEvent {
            stage: "refine".into(),
            message: name.into(),
            current: None,
            total: None,
            fraction: None,
        },
        RefineProgress::Search(event) => progress_search(event),
        RefineProgress::DensityBatchProgress {
            stage,
            evaluated,
            planned,
            ..
        } => ProgressEvent {
            stage: "density".into(),
            message: stage.into(),
            current: Some(evaluated),
            total: Some(planned),
            fraction: (planned > 0).then_some(evaluated as f64 / planned as f64),
        },
        RefineProgress::DensityEvaluation {
            evaluation,
            max_evaluations,
            parameter,
            ..
        } => ProgressEvent {
            stage: "density".into(),
            message: parameter,
            current: Some(evaluation),
            total: max_evaluations,
            fraction: max_evaluations
                .filter(|total| *total > 0)
                .map(|total| evaluation as f64 / total as f64),
        },
        RefineProgress::Relax(event) => progress_relax(event),
        _ => ProgressEvent {
            stage: "density".into(),
            message: "Evaluating refinement candidates…".into(),
            current: None,
            total: None,
            fraction: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, GlycanTree, read_pdb_str};

    const PROTEIN: &str = "\
ATOM      1  N   PRO A   7       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  PRO A   7       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  CB  PRO A   7       1.900   1.400   0.000  1.00 20.00           C
ATOM      4  CG  PRO A   7       0.800   2.200   0.500  1.00 20.00           C
ATOM      5  CD  PRO A   7      -0.200   1.100   0.200  1.00 20.00           C
END
";

    fn pro_assignment(residue_name: &str) -> SiteAssignment {
        SiteAssignment {
            id: "ara-pro-7".into(),
            site: SiteKey {
                model: 1,
                chain: "A".into(),
                residue_number: 7,
                insertion_code: None,
            },
            residue_name: residue_name.into(),
            glycan_id: "GS00558".into(),
            anomer: Anomer::Alpha,
            level: 2,
            provenance: AssignmentProvenance::Manual,
            source_description: Some("O-linked (Ara...) hydroxyproline".into()),
            evidence: None,
            glycan_asset: Some("glycans/GS00558-alpha-L2.pdb".into()),
            excluded: false,
            unresolved: false,
            replace_existing: false,
            existing_glycan_id: None,
            existing_glycan_residues: Vec::new(),
        }
    }

    #[test]
    fn options_default_to_pdb_and_record_format_and_seed_metadata() {
        let mut request = replacement_request(WorkflowId::SiteBuild, pro_assignment("PRO"));
        assert_eq!(request.options.output_format, ResidueNameFormat::Pdb);
        request.options.output_format = ResidueNameFormat::Glycam;
        request.options.seed = 1234;
        let metadata = method_metadata(&request);
        assert_eq!(metadata["outputFormat"], "GLYCAM");
        assert_eq!(metadata["seed"], 1234);
        assert_eq!(search_config(&request).seed, 1234);
        assert_eq!(attachment_search_config(&request).seed, 1234);
        assert_eq!(ensemble_search_config(&request).seed, 1234);
    }

    #[test]
    fn scan_uses_a_fixed_small_budget_independent_of_build_settings() {
        let mut request = replacement_request(WorkflowId::NScan, pro_assignment("PRO"));
        request.options.population_size = 256;
        request.options.generations = 500;
        request.options.search_budget_mode = Some(SearchBudgetMode::Auto);
        request.options.seed = 99;
        let scan = scan_search_config(&request);
        assert_eq!(scan.population_size, 32);
        assert_eq!(scan.generations, 25);
        assert_eq!(scan.seed, 99);
        assert_eq!(
            scan.selection_policy,
            SearchSelectionPolicy::CookbookFirstFeasible
        );

        assert!(!scan.polish_attachment_vmm);
        assert!(!scan.scan_rotamers);
    }

    #[test]
    fn missing_output_format_is_backward_compatible_and_invalid_values_fail() {
        let mut encoded = serde_json::to_value(ReGlycoOptions::default()).unwrap();
        encoded.as_object_mut().unwrap().remove("outputFormat");
        let decoded: ReGlycoOptions = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.output_format, ResidueNameFormat::Pdb);

        let invalid = serde_json::json!({
            "outputFormat": "XYZ"
        });
        assert!(serde_json::from_value::<ReGlycoOptions>(invalid).is_err());
    }

    #[test]
    fn missing_or_null_seed_uses_historical_zero_default() {
        let missing = serde_json::json!({});
        assert_eq!(
            serde_json::from_value::<ReGlycoOptions>(missing)
                .unwrap()
                .seed,
            0
        );
        let null = serde_json::json!({ "seed": null });
        assert_eq!(
            serde_json::from_value::<ReGlycoOptions>(null).unwrap().seed,
            0
        );
    }

    const OCCUPIED_PROTEIN: &str = "\
ATOM      1  N   ASN A   7       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ASN A   7       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  CB  ASN A   7       1.900   1.400   0.000  1.00 20.00           C
ATOM      4  CG  ASN A   7       3.300   1.400   0.000  1.00 20.00           C
ATOM      5  ND2 ASN A   7       3.900   2.500   0.000  1.00 20.00           N
HETATM    6  C1  NAG B   1       5.000   2.500   0.000  1.00 20.00           C
HETATM    7  O5  NAG B   1       5.800   2.500   0.000  1.00 20.00           O
HETATM    8  C1  MAN C   1       7.000   2.500   0.000  1.00 20.00           C
HETATM    9  O5  MAN C   1       7.800   2.500   0.000  1.00 20.00           O
END
";

    fn occupied_site() -> SiteKey {
        SiteKey {
            model: 1,
            chain: "A".into(),
            residue_number: 7,
            insertion_code: None,
        }
    }

    fn replacement_assignment(site: SiteKey, residues: Vec<SiteKey>) -> SiteAssignment {
        SiteAssignment {
            id: "replace-7".into(),
            site,
            residue_name: "ASN".into(),
            glycan_id: "GS00178".into(),
            anomer: Anomer::Beta,
            level: 2,
            provenance: AssignmentProvenance::Manual,
            source_description: None,
            evidence: None,
            glycan_asset: Some("glycans/GS00178-beta-L2.pdb".into()),
            excluded: false,
            unresolved: false,
            replace_existing: true,
            existing_glycan_id: Some("G00028MO".into()),
            existing_glycan_residues: residues,
        }
    }

    fn replacement_request(
        workflow: WorkflowId,
        assignment: SiteAssignment,
    ) -> ReGlycoRunRequestV1 {
        ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "occupied.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: vec![assignment],
            options: ReGlycoOptions::default(),
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    fn occupied_structure() -> Structure {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        };
        let mut structure = read_pdb_str(OCCUPIED_PROTEIN, &options).unwrap();
        structure.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "B".into(),
            residue_ids: vec![ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            }],
            attachment_site: Some(occupied_site().residue_id()),
        });
        structure.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "C".into(),
            residue_ids: vec![ResidueId {
                chain: "C".into(),
                number: 1,
                insertion_code: None,
            }],
            attachment_site: Some(ResidueId {
                chain: "A".into(),
                number: 8,
                insertion_code: None,
            }),
        });
        structure
    }

    #[test]
    fn contract_round_trips_camel_case_request() {
        let value = json!({
            "schemaVersion": 1,
            "workflow": "validate",
            "profile": "public",
            "input": { "kind": "upload", "label": "x.pdb", "asset": "protein.pdb", "sha256": "x" },
            "assignments": [],
            "options": {},
            "createdAt": "2026-01-01T00:00:00Z"
        });
        let request: ReGlycoRunRequestV1 = serde_json::from_value(value).unwrap();
        assert_eq!(request.workflow, WorkflowId::Validate);
        assert_eq!(request.options.ensemble_frames, 50);
        assert!(!request.options.scan_rotamers);
        assert_eq!(request.options.ensemble_temperature_k, 300.0);
        assert_eq!(request.options.ensemble_mh_chains, 8);
        assert_eq!(request.options.ensemble_burn_in_sweeps, 250);
        assert_eq!(request.options.ensemble_thinning_accepted, 50);
        assert_eq!(serde_json::to_value(request).unwrap()["schemaVersion"], 1);
    }

    #[test]
    fn replacement_strips_only_the_selected_deposited_tree() {
        let structure = occupied_structure();
        let assignment = replacement_assignment(
            occupied_site(),
            vec![SiteKey {
                model: 1,
                chain: "B".into(),
                residue_number: 1,
                insertion_code: None,
            }],
        );
        let request = replacement_request(WorkflowId::SiteBuild, assignment);
        let (stripped, records) = prepare_replacements(structure, &request).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].existing_glycan_id.as_deref(), Some("G00028MO"));
        assert_eq!(records[0].removed_residues.len(), 1);
        assert!(
            stripped
                .residues()
                .iter()
                .all(|residue| residue.id.chain != "B")
        );
        assert!(
            stripped
                .residues()
                .iter()
                .any(|residue| residue.id.chain == "C")
        );
        assert!(
            stripped
                .metadata()
                .glycan_trees
                .iter()
                .any(|tree| tree.chain == "C")
        );
    }

    #[test]
    fn replacement_uses_explicit_residue_fallback_and_preserves_insertion_codes() {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        };
        // Keep the PDB column width intact while adding an insertion code to
        // the protein attachment residue.
        let insertion_pdb = OCCUPIED_PROTEIN.replace("ASN A   7       ", "ASN A   7A      ");
        let mut structure = read_pdb_str(&insertion_pdb, &options).unwrap();
        structure.metadata_mut().glycan_trees.clear();
        let site = SiteKey {
            model: 1,
            chain: "A".into(),
            residue_number: 7,
            insertion_code: Some('A'),
        };
        let assignment = replacement_assignment(
            site.clone(),
            vec![SiteKey {
                model: 1,
                chain: "B".into(),
                residue_number: 1,
                insertion_code: None,
            }],
        );
        // The parser-independent fallback still identifies the exact sugar
        // residue even when GlySys did not provide tree metadata. The protein
        // attachment insertion code is kept in the replacement provenance.
        let request = replacement_request(WorkflowId::SiteBuild, assignment);
        let (stripped, records) = prepare_replacements(structure, &request).unwrap();
        assert!(
            stripped
                .residues()
                .iter()
                .all(|residue| residue.id.chain != "B")
        );
        assert_eq!(records[0].site.insertion_code, Some('A'));
    }

    #[test]
    fn replacement_rejects_double_attachment_and_ambiguous_targets() {
        let mut structure = occupied_structure();
        let site = occupied_site();
        let non_replacing = SiteAssignment {
            replace_existing: false,
            existing_glycan_id: None,
            existing_glycan_residues: Vec::new(),
            ..replacement_assignment(site.clone(), Vec::new())
        };
        let error = prepare_replacements(
            structure.clone(),
            &replacement_request(WorkflowId::SiteBuild, non_replacing),
        )
        .unwrap_err();
        assert!(error.to_string().contains("already has a deposited glycan"));

        structure.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "B".into(),
            residue_ids: vec![ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            }],
            attachment_site: Some(site.residue_id()),
        });
        let error = prepare_replacements(
            structure,
            &replacement_request(
                WorkflowId::SiteBuild,
                replacement_assignment(
                    site,
                    vec![SiteKey {
                        model: 1,
                        chain: "B".into(),
                        residue_number: 1,
                        insertion_code: None,
                    }],
                ),
            ),
        )
        .unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
    }

    #[test]
    fn non_replacing_workflows_leave_assignments_and_coordinates_untouched() {
        let structure = occupied_structure();
        let assignment = replacement_assignment(
            occupied_site(),
            vec![SiteKey {
                model: 1,
                chain: "B".into(),
                residue_number: 1,
                insertion_code: None,
            }],
        );
        let request = replacement_request(
            WorkflowId::Validate,
            SiteAssignment {
                replace_existing: false,
                ..assignment
            },
        );
        let (unchanged, records) = prepare_replacements(structure, &request).unwrap();
        assert!(records.is_empty());
        assert_eq!(unchanged.residues().len(), 3);
        assert_eq!(unchanged.metadata().glycan_trees.len(), 2);
    }

    #[test]
    fn attachment_and_ensemble_configs_forward_only_their_controls() {
        let mut request = ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow: WorkflowId::Ensemble,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "protein.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: Vec::new(),
            options: ReGlycoOptions {
                scan_rotamers: true,
                ensemble_temperature_k: 315.0,
                ensemble_mh_chains: 7,
                ensemble_burn_in_sweeps: 31,
                ensemble_thinning_accepted: 13,
                ..ReGlycoOptions::default()
            },
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let build = attachment_search_config(&request);
        assert!(build.scan_rotamers);
        assert!(build.polish_attachment_vmm);
        assert_eq!(build.mh_chains, 4);
        assert_eq!(build.temperature_k, 300.0);

        let ensemble = ensemble_search_config(&request);
        assert!(ensemble.scan_rotamers);
        assert!(!ensemble.polish_attachment_vmm);
        assert_eq!(ensemble.temperature_k, 315.0);
        assert_eq!(ensemble.mh_chains, 7);
        assert_eq!(ensemble.mh_burn_in_sweeps, 31);
        assert_eq!(ensemble.mh_thinning_accepted, 13);

        request.options.ensemble_frames = 0;
        assert!(validate_options(&request).is_err());
        request.options.ensemble_frames = 50;
        request.options.ensemble_temperature_k = f64::NAN;
        assert!(validate_options(&request).is_err());
    }

    #[test]
    fn search_budget_mode_defaults_to_auto_for_new_options_and_manual_for_legacy_json() {
        assert_eq!(
            ReGlycoOptions::default().search_budget_mode,
            Some(SearchBudgetMode::Auto)
        );
        let mut value = serde_json::to_value(ReGlycoOptions::default()).unwrap();
        value
            .as_object_mut()
            .expect("options object")
            .remove("searchBudgetMode");
        let decoded: ReGlycoOptions = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.search_budget_mode, None);
        assert_eq!(decoded.population_size, 128);
        assert_eq!(decoded.generations, 100);
    }

    #[test]
    fn reference_defined_omega_uses_glycoshape_atoms_and_convention() {
        const LINKAGE: &str = "\
HETATM    1  C1  MAN G   1       0.000   0.000   0.000  1.00 20.00           C
HETATM    2  C4  BMA G   2       1.000   0.000   0.000  1.00 20.00           C
HETATM    3  C5  BMA G   2       1.000   1.000   0.000  1.00 20.00           C
HETATM    4  C6  BMA G   2       1.000   1.000   1.000  1.00 20.00           C
HETATM    5  O6  BMA G   2       2.000   1.000   1.000  1.00 20.00           O
HETATM    6  O5  BMA G   2       0.000   2.000   1.000  1.00 20.00           O
END
";
        let structure = read_pdb_str(
            LINKAGE,
            &BuildOptions {
                add_water: false,
                add_ions: false,
                ..BuildOptions::default()
            },
        )
        .unwrap();
        let donor = ResidueId {
            chain: "G".into(),
            number: 1,
            insertion_code: None,
        };
        let acceptor = ResidueId {
            chain: "G".into(),
            number: 2,
            insertion_code: None,
        };
        let reference = json!({
            "angle_definitions": {
                "omega": [
                    { "residue_role": "acceptor", "atom_name": "C4" },
                    { "residue_role": "acceptor", "atom_name": "C5" },
                    { "residue_role": "acceptor", "atom_name": "C6" },
                    { "residue_role": "acceptor", "atom_name": "O6" }
                ]
            }
        });
        let measured =
            reference_defined_torsion(&structure, &reference, "omega", &donor, &acceptor).unwrap();
        let positions = ["C4", "C5", "C6", "O6"].map(|name| {
            structure
                .atom(structure.find_atom(&acceptor, name).unwrap())
                .unwrap()
                .position
        });
        assert!(
            (measured
                - glycoshape_dihedral_degrees(
                    positions[0],
                    positions[1],
                    positions[2],
                    positions[3],
                ))
            .abs()
                < 1.0e-12
        );
        assert!(
            (measured - dihedral_degrees(positions[0], positions[1], positions[2], positions[3],))
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn glycoshape_dihedral_matches_pipeline_representative() {
        let point = |x, y, z| glysys::Vec3 { x, y, z };
        // GS00178 beta L2 model 1. Expected values were independently
        // calculated with GlycanAnalysisPipeline's vectorized RDKit formula.
        let c3_2 = point(26.012, 20.976, 33.193);
        let c4_2 = point(26.986, 20.815, 31.961);
        let o4_2 = point(26.370, 20.682, 30.575);
        let c1_3 = point(26.461, 21.872, 29.766);
        let o5_3 = point(25.506, 22.915, 30.209);
        assert!(
            (glycoshape_dihedral_degrees(o5_3, c1_3, o4_2, c4_2) - -74.435_648_167_389_47).abs()
                < 1.0e-9
        );
        assert!(
            (glycoshape_dihedral_degrees(c1_3, o4_2, c4_2, c3_2) - 104.300_338_349_920_9).abs()
                < 1.0e-9
        );

        let c4_4 = point(22.377, 26.217, 24.065);
        let c5_4 = point(22.420, 24.747, 24.498);
        let c6_4 = point(22.597, 23.823, 23.296);
        let o6_4 = point(23.640, 24.249, 22.321);
        assert!(
            (glycoshape_dihedral_degrees(c4_4, c5_4, c6_4, o6_4) - 45.478_420_696_643_326).abs()
                < 1.0e-9
        );
    }

    #[test]
    fn repeated_pdb_linkages_resolve_by_residue_occurrence() {
        let references = vec![
            json!({
                "canonicalLinkage": "first",
                "pdb_branch_paths": [["MAN", "1", "MAN", "6"]],
                "pdb_output_residue_paths": [["X:4", "X:2"]],
            }),
            json!({
                "canonicalLinkage": "second",
                "pdb_branch_paths": [["MAN", "1", "MAN", "6"]],
                "pdb_output_residue_paths": [["X:8", "X:6"]],
            }),
        ];
        let selected = torsion_reference_for_pdb_occurrence(
            &references,
            &["MAN".into(), "1".into(), "MAN".into(), "6".into()],
            &ResidueId {
                chain: "G".into(),
                number: 8,
                insertion_code: None,
            },
            &ResidueId {
                chain: "G".into(),
                number: 6,
                insertion_code: None,
            },
        )
        .unwrap();
        assert_eq!(selected["canonicalLinkage"], "second");
    }

    #[test]
    fn public_capabilities_exclude_unpublished_workflows() {
        let value = capabilities(ReGlycoProfile::Public, false);
        assert!(value.workflows.contains(&WorkflowId::Validate));
        assert!(value.workflows.contains(&WorkflowId::Refine));
        assert!(!value.workflows.contains(&WorkflowId::Density));
        assert!(!value.workflows.contains(&WorkflowId::Saxs));
    }

    #[cfg(feature = "full")]
    #[test]
    fn full_capabilities_include_experimental_workflows() {
        let value = capabilities(ReGlycoProfile::Full, false);
        assert!(value.workflows.contains(&WorkflowId::Refine));
        assert!(value.workflows.contains(&WorkflowId::Density));
        assert!(value.workflows.contains(&WorkflowId::Saxs));
    }

    #[test]
    fn clash_free_required_is_a_blocked_scan_site() {
        let error = EnsembleError::ReGlyco(ReGlycoError::ClashFreeRequired);
        assert!(blocked_scan_outcome(&error));
    }

    #[test]
    fn browser_search_retains_best_complete_clashing_results() {
        let request = ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow: WorkflowId::SiteBuild,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "protein.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: Vec::new(),
            options: ReGlycoOptions::default(),
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        assert!(!search_config(&request).require_clash_free);
    }

    #[test]
    fn prepares_proline_from_the_structure_not_the_request_residue_name() {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        };
        let parsed = read_pdb_str(PROTEIN, &options).unwrap();
        let prepared = prepare_proline_sites(parsed, &[pro_assignment("")], &options).unwrap();
        let site = ResidueId {
            chain: "A".into(),
            number: 7,
            insertion_code: None,
        };
        assert_eq!(
            prepared
                .residues()
                .into_iter()
                .find(|residue| residue.id == site)
                .unwrap()
                .name,
            "HYP"
        );
        assert!(prepared.find_atom(&site, "OD1").is_some());
    }

    #[test]
    fn reports_proline_conversion_provenance() {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        };
        let original = read_pdb_str(PROTEIN, &options).unwrap();
        let converted = hydroxylate_proline(
            &original,
            &ResidueId {
                chain: "A".into(),
                number: 7,
                insertion_code: None,
            },
            &options,
        )
        .unwrap();
        let request = ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow: WorkflowId::SiteBuild,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "pro.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: vec![pro_assignment("PRO")],
            options: ReGlycoOptions::default(),
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let mut control = NoopControl;
        let bundle = finish_workflow(
            &request,
            PROTEIN,
            converted,
            WorkflowStatus::Succeeded,
            Vec::new(),
            Vec::new(),
            json!({}),
            Vec::new(),
            &mut control,
        )
        .unwrap();
        assert!(
            bundle
                .warnings
                .iter()
                .any(|warning| warning.contains("Converted 1 PRO attachment site"))
        );
        assert_eq!(
            bundle.report.analysis["prolineConversions"][0]["residueNumber"],
            7
        );
        assert_eq!(
            bundle.report.provenance["prolineConversions"][0]["chain"],
            "A"
        );
    }

    #[test]
    fn validate_keeps_an_unconverted_proline_input_intact() {
        let request = ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow: WorkflowId::Validate,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "pro.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: vec![pro_assignment("PRO")],
            options: ReGlycoOptions::default(),
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let assets =
            InputAssets::from([(String::from("protein.pdb"), AssetData::Text(PROTEIN.into()))]);
        let bundle = execute(&request, &assets).unwrap();
        assert!(
            bundle
                .primary_structure
                .as_deref()
                .unwrap()
                .contains(" PRO ")
        );
        assert!(
            bundle.report.provenance["prolineConversions"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(bundle.report.validation.findings.is_empty());
        assert!(bundle.report.validation.errors.is_empty());
        assert!(matches!(bundle.status, WorkflowStatus::Succeeded));
    }

    #[test]
    fn background_torsion_analysis_splits_every_accepted_model() {
        let request = ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow: WorkflowId::Ensemble,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "pro.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: Vec::new(),
            options: ReGlycoOptions::default(),
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let assets =
            InputAssets::from([(String::from("protein.pdb"), AssetData::Text(PROTEIN.into()))]);
        let ensemble =
            format!("MODEL        1\n{PROTEIN}ENDMDL\nMODEL        2\n{PROTEIN}ENDMDL\n");
        let result = analyze_torsions(&request, &assets, &ensemble, json!({})).unwrap();
        assert!(result.get("torsionObservations").is_some());
        assert_eq!(split_pdb_models(&ensemble).len(), 2);
    }

    #[test]
    fn level_one_assignment_prefers_periodic_cluster_grid() {
        let reference = json!({
            "clusters": [
                { "id": 1, "phi_psi": { "bins": 2, "counts": [1, 0, 0, 0] } },
                { "id": 2, "phi_psi": { "bins": 2, "counts": [7, 0, 0, 0] } }
            ]
        });
        assert_eq!(level_one_cluster_at(&reference, -90.0, -90.0), Some(2));
        assert_eq!(level_one_cluster_at(&reference, 90.0, 90.0), None);
        assert_eq!(periodic_bin(180.0, 36), periodic_bin(-180.0, 36));
    }

    #[test]
    fn extracts_terminal_sugars_from_glycoshape_metadata() {
        assert_eq!(
            glycan_terminal_sugar(&serde_json::json!({
                "archetype": { "iupac": "Fuc(a1-2)Gal(b1-3)GalNAc" }
            })),
            Some("GalNAc".into())
        );
        assert_eq!(
            glycan_terminal_sugar(&serde_json::json!({
                "archetype": { "iupac": "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc" }
            })),
            Some("GlcNAc".into())
        );
    }

    #[test]
    fn strict_build_fallback_promotes_complete_candidate_to_partial_result() {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        };
        let protein = read_pdb_str(PROTEIN, &options).unwrap();
        let request = ReGlycoRunRequestV1 {
            schema_version: SCHEMA_VERSION,
            workflow: WorkflowId::SiteBuild,
            profile: ReGlycoProfile::Public,
            input: ProteinInput {
                kind: ProteinInputKind::Upload,
                label: "pro.pdb".into(),
                source_id: None,
                asset: "protein.pdb".into(),
                sha256: "test".into(),
                source_url: None,
            },
            assignments: vec![pro_assignment("PRO")],
            options: ReGlycoOptions::default(),
            parent_job_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let diagnostics = StrictSearchDiagnostics {
            generation: 100,
            frozen_sites: 0,
            evaluations: 128,
            repaired: true,
            sites: Vec::new(),
            outlier_sites: Vec::new(),
            vdw_outlier_sites: Vec::new(),
            vdw_hard_contacts: 0,
            vdw_advisory_contacts: 0,
            vdw_max_overlap_angstrom: 0.0,
            vdw_total_overlap_angstrom: 0.0,
            vdw_contacts: Vec::new(),
            best_candidate_pdb: PROTEIN.into(),
        };
        let mut control = NoopControl;
        let bundle = finish_strict_failure(
            &request,
            PROTEIN,
            protein,
            Vec::new(),
            assignment_report_sites(&request, "strict-search-failed"),
            diagnostics,
            &[],
            Vec::new(),
            &mut control,
        )
        .unwrap();
        assert!(matches!(bundle.status, WorkflowStatus::Partial));
        assert!(bundle.primary_structure.is_some());
        assert!(
            bundle
                .artifacts
                .iter()
                .any(|artifact| artifact.name == "result.pdb")
        );
        assert!(bundle.error.is_none());
    }

    #[test]
    fn private_glycoshape_sources_are_not_exportable_artifacts() {
        let assets = InputAssets::from([
            ("protein.pdb".into(), AssetData::Text(PROTEIN.into())),
            (
                "glycans/GS00178-beta-L2.pdb".into(),
                AssetData::Text("MODEL\n".into()),
            ),
            (
                "glycans/GS00178-beta-L2.json".into(),
                AssetData::Text("{}".into()),
            ),
            (
                "glycans/GS00178-beta-L2-reference.json".into(),
                AssetData::Text("{}".into()),
            ),
            ("scan-glycan.pdb".into(), AssetData::Text("MODEL\n".into())),
            ("density.map".into(), AssetData::Bytes(vec![1, 2, 3])),
            ("saxs.dat".into(), AssetData::Text("0.1 1.0\n".into())),
        ]);
        let artifacts = exportable_input_support_artifacts(&assets, "protein.pdb");
        let names = artifacts
            .iter()
            .map(|artifact| artifact.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["density.map", "saxs.dat"]);
    }
}
