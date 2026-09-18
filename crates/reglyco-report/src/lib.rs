//! Stable machine-readable and publication-ready result reporting for ReGlyco.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use contour::ContourBuilder;
use crabwurcs_core::ResidueGraph;
use crabwurcs_core::write_wurcs_canonical;
use crabwurcs_iupac::write_iupac_condensed_canonical;
use crabwurcs_pdb::{PdbResidueReference, extract_glycans_with_provenance_from_str};
use crabwurcs_snfg::{HighlightSelection, RenderOptions, render_svg, render_svg_with_selection};
use glysys::{ResidueId, Structure, Vec3};
use petgraph::visit::EdgeRef;
use plotters::coord::Shift;
use plotters::prelude::*;
use reglyco_core::{ClashStatus, SearchOutcome, SearchSite, SearchSiteResult};
use reglyco_relax::RelaxationDiagnostics;
use thiserror::Error;
use typst_as_lib::{TypstEngine, typst_kit_options::TypstKitFontOptions};
use typst_layout::PagedDocument;

const TEMPLATE_HEADER: &str = include_str!("../templates/report.typ");

#[derive(Debug, Error)]
pub enum ReportError {
    #[error("report I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("report serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("could not render report figure: {0}")]
    Plot(String),
    #[error("Typst failed to compile the report: {0}")]
    Typst(String),
}

pub type Result<T> = std::result::Result<T, ReportError>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    pub reglyco_version: String,
    pub glysys_revision: String,
    pub command: String,
    pub seed: Option<u64>,
    /// Requested glycan residue-name convention for Build/Ensemble output.
    /// Older reports omit this field and remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    pub ensemble_sources: Vec<String>,
    /// Wall-clock time recorded by workflows that measure their complete run.
    #[serde(default)]
    pub total_seconds: Option<f64>,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            reglyco_version: env!("CARGO_PKG_VERSION").into(),
            glysys_revision: "fcfe86dfee8ef588ab75f9bb9ac38d73a14bb749".into(),
            command: String::new(),
            seed: None,
            output_format: None,
            ensemble_sources: Vec::new(),
            total_seconds: None,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GlycanSummary {
    pub index: usize,
    pub attachment_site: Option<String>,
    pub residue_count: usize,
    pub wurcs: Option<String>,
    pub iupac: Option<String>,
    pub snfg_asset: Option<String>,
    #[serde(skip)]
    snfg_svg: Option<String>,
    #[serde(skip)]
    #[allow(dead_code)]
    snfg_graph: Option<ResidueGraph>,
    #[serde(skip)]
    snfg_residues: Option<Vec<PdbResidueReference>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinkageTorsion {
    pub glycan_index: usize,
    pub linkage: String,
    pub phi_degrees: Option<f64>,
    pub psi_degrees: Option<f64>,
    pub omega_degrees: Option<f64>,
    pub frame: Option<usize>,
    #[serde(default)]
    pub donor_name: Option<String>,
    #[serde(default)]
    pub acceptor_name: Option<String>,
    #[serde(default)]
    pub donor_position: Option<u8>,
    #[serde(default)]
    pub acceptor_position: Option<u8>,
    #[serde(default)]
    pub donor_residue: Option<ResidueId>,
    #[serde(default)]
    pub acceptor_residue: Option<ResidueId>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProteinLinkageTorsion {
    pub site: String,
    pub phi_degrees: f64,
    pub psi_degrees: f64,
    pub frame: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterObservation {
    pub site: String,
    pub cluster_index: usize,
    pub main_cluster: Option<usize>,
    pub expected_weight: f64,
    pub observed_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StericObservation {
    pub site: String,
    pub score: f64,
    pub clash_free: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EnsembleAnalysis {
    pub requested_frames: usize,
    pub returned_frames: usize,
    pub attempts: usize,
    pub acceptance_rate: f64,
    #[serde(default)]
    pub native_frames: usize,
    #[serde(default)]
    pub fallback_frames: usize,
    #[serde(default)]
    pub fallback_used: bool,
    #[serde(default)]
    pub ga_restarts: usize,
    #[serde(default)]
    pub chains: usize,
    #[serde(default)]
    pub native_proposals: usize,
    #[serde(default)]
    pub native_accepts: usize,
    #[serde(default)]
    pub mh_proposals: usize,
    #[serde(default)]
    pub mh_accepts: usize,
    #[serde(default)]
    pub burn_in_sweeps: usize,
    #[serde(default)]
    pub thinning_accepted: usize,
    /// Effective numerical target used by energy/interaction sampling.
    #[serde(default)]
    pub sampling_target: Option<reglyco_core::SamplingTarget>,
    /// Explicit backend segments keep a GPU-to-CPU recovery visible in the
    /// native report rather than pooling unlike numerical targets.
    #[serde(default)]
    pub sampling_segments: Vec<SamplingSegmentAnalysis>,
    pub native_log_probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SamplingSegmentAnalysis {
    pub id: usize,
    pub target: reglyco_core::SamplingTarget,
    pub backend: String,
    pub frames: usize,
    pub attempts: usize,
    pub accepts: usize,
    #[serde(default)]
    pub burn_in_steps: usize,
    #[serde(default)]
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScanSequon {
    /// Four-residue context when a preceding residue is available, e.g. LNTT.
    pub context: String,
    /// The canonical three-residue N-X-S/T motif, e.g. NTT.
    pub motif: String,
    pub asparagine: ResidueId,
    pub independently_accessible: bool,
    pub jointly_selected: bool,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScanAnalysis {
    pub interpretation: String,
    pub independent_accessible_count: usize,
    pub jointly_compatible_count: usize,
    #[serde(default)]
    pub sequons: Vec<ScanSequon>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ValidationAnalysis {
    pub valid: bool,
    pub atom_count: usize,
    pub residue_count: usize,
    pub attachment_count: usize,
    #[serde(default)]
    pub error_count: usize,
    #[serde(default)]
    pub warning_count: usize,
}

/// Classification used by the conformational report.  Internal glycosidic
/// tails are advisory; only attachment VMM gates are hard workflow criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TorsionAssessment {
    Core,
    Allowed,
    Tail,
    Outlier,
    NoReference,
}

impl Default for TorsionAssessment {
    fn default() -> Self {
        Self::NoReference
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TorsionPopulation {
    pub index: usize,
    #[serde(default)]
    pub level: u8,
    #[serde(default)]
    pub parent_index: Option<usize>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub weight: f64,
    #[serde(default)]
    pub phi_mean: Option<f64>,
    #[serde(default)]
    pub psi_mean: Option<f64>,
    #[serde(default)]
    pub omega_mean: Option<f64>,
    #[serde(default)]
    pub phi_kappa: Option<f64>,
    #[serde(default)]
    pub psi_kappa: Option<f64>,
    #[serde(default)]
    pub omega_kappa: Option<f64>,
    #[serde(default)]
    pub medoid_phi: Option<f64>,
    #[serde(default)]
    pub medoid_psi: Option<f64>,
    #[serde(default)]
    pub medoid_omega: Option<f64>,
    /// Source Level-2/3 cluster ID.  It is intentionally separate from the
    /// parent color so finer populations do not create a second palette.
    #[serde(default)]
    pub source_cluster: Option<usize>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TorsionContour {
    #[serde(default)]
    pub mass: f64,
    #[serde(default)]
    pub bins: usize,
    /// Histogram/KDE values.  The browser torsion-reference endpoint calls
    /// this field `counts`; accepting that alias keeps the persisted report
    /// contract compatible with both native and web-produced references.
    #[serde(default, alias = "counts")]
    pub grid: Vec<f64>,
    #[serde(default)]
    pub threshold: Option<f64>,
}

/// Versioned GlycoShape reference data retained with a job.  The grid and
/// contours are compact enough to keep reports reproducible offline while
/// allowing the browser to render the same populations as the API page.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TorsionReference {
    pub version: String,
    #[serde(default)]
    pub source_database: String,
    #[serde(default)]
    pub content_hash: String,
    pub canonical_linkage: String,
    #[serde(default)]
    pub branch_path: Vec<String>,
    #[serde(default)]
    pub phi_psi: Option<TorsionContour>,
    #[serde(default)]
    pub phi_omega: Option<TorsionContour>,
    #[serde(default)]
    pub psi_omega: Option<TorsionContour>,
    #[serde(default)]
    pub populations: Vec<TorsionPopulation>,
    #[serde(default)]
    pub contours: Vec<TorsionContour>,
    /// Highest-density thresholds for the optional φ/ω projection.
    #[serde(default)]
    pub phi_omega_contours: Vec<TorsionContour>,
    /// Highest-density thresholds for the optional ψ/ω projection.
    #[serde(default)]
    pub psi_omega_contours: Vec<TorsionContour>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TorsionObservation {
    pub domain: String,
    pub origin: String,
    pub stage: String,
    #[serde(default)]
    pub frame: Option<usize>,
    /// Stable attachment-site identity for the glycan branch, when known.
    #[serde(default)]
    pub site: Option<String>,
    /// Donor/acceptor identities and positions; independent of PDB residue
    /// numbering so references remain reusable across coordinate exports.
    #[serde(default)]
    pub branch_path: Vec<String>,
    pub glycan_index: usize,
    pub linkage: String,
    #[serde(default)]
    pub involved_atoms: Vec<String>,
    #[serde(default)]
    pub phi_degrees: Option<f64>,
    #[serde(default)]
    pub psi_degrees: Option<f64>,
    #[serde(default)]
    pub omega_degrees: Option<f64>,
    #[serde(default)]
    pub population_percentile: Option<f64>,
    pub assessment: TorsionAssessment,
    #[serde(default)]
    pub nearest_population: Option<usize>,
    #[serde(default)]
    pub selected_population: Option<usize>,
    #[serde(default)]
    pub policy_version: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AttachmentTorsionObservation {
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub origin: String,
    pub site: String,
    pub stage: String,
    #[serde(default)]
    pub frame: Option<usize>,
    pub phi_degrees: f64,
    pub psi_degrees: f64,
    #[serde(default)]
    pub glycan_index: Option<usize>,
    #[serde(default)]
    pub linkage: Option<String>,
    #[serde(default)]
    pub involved_atoms: Vec<String>,
    #[serde(default)]
    pub population_percentile: Option<f64>,
    pub assessment: TorsionAssessment,
    #[serde(default)]
    pub selected_phi_component: Option<usize>,
    #[serde(default)]
    pub selected_psi_component: Option<usize>,
    #[serde(default)]
    pub policy_version: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EnsembleTorsionSummary {
    pub frames: usize,
    #[serde(default)]
    pub observed_frames: Vec<usize>,
    #[serde(default)]
    pub outlier_frames: Vec<usize>,
    #[serde(default)]
    pub linkage_outlier_rates: BTreeMap<String, f64>,
    #[serde(default)]
    pub circular_means: BTreeMap<String, (f64, f64)>,
    #[serde(default)]
    /// Circular dispersion for (phi, psi), represented as one-minus-resultant
    /// length for each periodic axis.
    pub circular_dispersion: BTreeMap<String, (f64, f64)>,
    #[serde(default)]
    pub cluster_coverage: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensitySiteAnalysis {
    pub site: String,
    pub correlation: f64,
    pub voxel_count: usize,
    pub supported_atom_fraction: f64,
    pub ring_support: f64,
    pub connectivity_support: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityRecoveryAnalysis {
    pub site: String,
    #[serde(default)]
    pub root_c1_distance_angstrom: Option<f64>,
    #[serde(default)]
    pub three_residue_heavy_atom_rmsd_angstrom: Option<f64>,
    #[serde(default)]
    pub supported_heavy_atom_rmsd_angstrom: Option<f64>,
    #[serde(default)]
    pub full_tree_heavy_atom_rmsd_angstrom: Option<f64>,
    #[serde(default)]
    pub per_residue_heavy_atom_rmsd_angstrom: BTreeMap<String, f64>,
    #[serde(default)]
    pub arm_heavy_atom_rmsd_angstrom: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityBaselineAnalysis {
    pub kind: String,
    #[serde(default)]
    pub conformer_ids: Vec<String>,
    pub correlation: f64,
    pub likelihood_gain: f64,
    pub rmsd_to_fitted_angstrom: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityArmAnalysis {
    pub label: String,
    pub residues: Vec<String>,
    pub classification: String,
    pub selected_mode: String,
    pub source_conformer_ids: Vec<String>,
    pub fixed_roi_likelihood_gain: f64,
    #[serde(default)]
    pub normalized_density_gain: f64,
    #[serde(default)]
    pub prior_log_probability: f64,
    #[serde(default)]
    pub clash_score: f64,
    #[serde(default)]
    pub selected_mode_posterior: f64,
    #[serde(default = "default_density_credible_mass")]
    pub credible_mass: f64,
    #[serde(default)]
    pub credible_set_size: usize,
    #[serde(default)]
    pub bic_penalty: f64,
    pub ring_support: f64,
    pub linkage_path_support: f64,
    pub evaluations: usize,
    pub escalation_reason: String,
    /// Heterogeneous native-mode alternatives for an ambiguous arm.  These
    /// make the uncertainty human-readable in the report; they never affect
    /// ranking or `fitted.pdb`.
    #[serde(default)]
    pub alternatives: Vec<DensityArmAlternativeAnalysis>,
}

/// Report-local mirror of the refiner's per-arm alternative so an ambiguous
/// arm's distinct conformations are surfaced without coupling the report
/// crate to the refiner.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityArmAlternativeAnalysis {
    pub mode_id: String,
    pub source_conformer_ids: Vec<String>,
    pub population_prior: f64,
    pub objective: f64,
    pub posterior_weight: f64,
    #[serde(default)]
    pub cumulative_posterior_weight: f64,
    #[serde(default)]
    pub in_credible_set: bool,
}

fn default_density_credible_mass() -> f64 {
    0.95
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityBasinAnalysis {
    pub basin_id: String,
    pub pilot_score: f64,
    pub improvement_bound: f64,
    pub pilot_posterior: f64,
    #[serde(default)]
    pub selected: bool,
    #[serde(default)]
    pub fully_polished: bool,
    #[serde(default)]
    pub final_score: Option<f64>,
    #[serde(default)]
    pub evaluations: usize,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub status: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityKernelAnalysis {
    pub residue: String,
    pub b_factor: f64,
    pub effective_sigma_angstrom: f64,
    pub heldout_gain: f64,
    pub bic_gain: f64,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityAnalysis {
    pub pre_relax_correlation: f64,
    #[serde(default)]
    pub sigma_angstrom: Option<f64>,
    #[serde(default)]
    pub effective_sigma_angstrom: Option<f64>,
    #[serde(default)]
    pub capture_sigma_angstrom: Option<f64>,
    #[serde(default)]
    pub anti_alias_floor_angstrom: Option<f64>,
    #[serde(default)]
    pub voxel_spacing_angstrom: [f64; 3],
    #[serde(default)]
    pub training_likelihood_gain: Option<f64>,
    #[serde(default)]
    pub heldout_likelihood_gain: Option<f64>,
    #[serde(default)]
    pub difference_score: Option<f64>,
    #[serde(default)]
    pub post_relax_correlation: Option<f64>,
    pub evaluations: usize,
    /// Number of exact graph-assignment proposals evaluated in deep mode.
    #[serde(default)]
    pub graph_evaluations: usize,
    /// Number of map-derived ring hypotheses retained for the density-first
    /// graph.  Zero is expected for fast/adaptive and legacy reports.
    #[serde(default)]
    pub ring_hypothesis_count: usize,
    #[serde(default)]
    pub estimated_glycan_b_factor: Option<f64>,
    #[serde(default)]
    pub candidate_correlations: Vec<f64>,
    #[serde(default)]
    pub candidate_posterior_weights: Vec<f64>,
    #[serde(default)]
    pub density_determined_residues: Vec<String>,
    #[serde(default)]
    pub ensemble_prior_residues: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub optimization_seconds: Option<f64>,
    #[serde(default)]
    pub relaxation_seconds: Option<f64>,
    #[serde(default)]
    pub total_seconds: Option<f64>,
    /// Coarse production-stage timings copied from the fitter.  Debug traces
    /// remain in density.json only; this compact list is intended for the
    /// human report and stable downstream consumers.
    #[serde(default)]
    pub stage_timings: Vec<DensityStageAnalysis>,
    #[serde(default)]
    pub supported_atom_fraction: Option<f64>,
    #[serde(default)]
    pub ring_support: Option<f64>,
    #[serde(default)]
    pub connectivity_support: Option<f64>,
    #[serde(default)]
    pub site_diagnostics: Vec<DensitySiteAnalysis>,
    #[serde(default)]
    pub recovery: Vec<DensityRecoveryAnalysis>,
    #[serde(default)]
    pub ensemble_baselines: Vec<DensityBaselineAnalysis>,
    #[serde(default)]
    pub arm_evidence: Vec<DensityArmAnalysis>,
    #[serde(default)]
    pub basin_diagnostics: Vec<DensityBasinAnalysis>,
    #[serde(default)]
    pub kernel_decisions: Vec<DensityKernelAnalysis>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityStageAnalysis {
    pub stage: String,
    pub seconds: f64,
    pub evaluations: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SaxsCombinationAssignment {
    pub site: String,
    pub candidate: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SaxsCombinationAnalysis {
    pub assignments: Vec<SaxsCombinationAssignment>,
    pub prior: f64,
    #[serde(default)]
    pub log_likelihood: Option<f64>,
    pub likelihood: f64,
    pub posterior: f64,
    pub robust_score: f64,
    pub features: crabsaxs::SaxsFeatures,
}

/// A glycan candidate used by a SAXS workflow, with its SNFG rendering kept
/// alongside the machine-readable candidate identifier for report figures.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SaxsCandidateVisual {
    pub candidate: String,
    #[serde(default)]
    pub snfg_asset: Option<String>,
    #[serde(skip, default)]
    pub snfg_svg: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SaxsAnalysis {
    pub mode: String,
    pub data_path: Option<String>,
    pub experimental_rg: Option<f64>,
    pub experimental_dmax: Option<f64>,
    pub model_fits: Vec<crabsaxs::SaxsFeatures>,
    #[serde(default)]
    pub fit_features: Option<crabsaxs::SaxsFeatures>,
    pub model_names: Vec<String>,
    pub weights: Vec<f64>,
    pub prior_weights: Vec<f64>,
    #[serde(default)]
    pub native_log_probabilities: Vec<f64>,
    pub kl_divergence: Option<f64>,
    pub effective_sample_size: Option<f64>,
    pub combinations: Vec<SaxsCombinationAnalysis>,
    pub site_occupancy: BTreeMap<String, f64>,
    pub glycoform_distribution: BTreeMap<String, BTreeMap<String, f64>>,
    #[serde(default)]
    pub candidate_visuals: Vec<SaxsCandidateVisual>,
    #[serde(default)]
    pub diagnostic_asset: Option<String>,
    #[serde(default)]
    pub occupancy_asset: Option<String>,
    #[serde(default)]
    pub candidate_asset: Option<String>,
    #[serde(skip)]
    pub diagnostic_svg: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ReportAnalysis {
    pub glycans: Vec<GlycanSummary>,
    pub protein_linkage_torsions: Vec<ProteinLinkageTorsion>,
    pub glycosidic_torsions: Vec<LinkageTorsion>,
    pub clusters: Vec<ClusterObservation>,
    pub sterics: Vec<StericObservation>,
    pub ensemble: Option<EnsembleAnalysis>,
    pub scan: Option<ScanAnalysis>,
    pub validation: Option<ValidationAnalysis>,
    #[serde(default)]
    pub density: Option<DensityAnalysis>,
    #[serde(default)]
    pub saxs: Option<SaxsAnalysis>,
    /// Versioned GlycoShape distributions used for the torsion plots.
    #[serde(default)]
    pub torsion_references: Vec<TorsionReference>,
    /// Accepted-stage internal linkage observations (GA populations are never
    /// persisted here).
    #[serde(default)]
    pub torsion_observations: Vec<TorsionObservation>,
    #[serde(default)]
    pub attachment_observations: Vec<AttachmentTorsionObservation>,
    #[serde(default)]
    pub attachment_references: Vec<TorsionReference>,
    #[serde(default)]
    pub ensemble_torsion: Option<EnsembleTorsionSummary>,
    /// Physical energy decomposition supplied by energy/interaction search
    /// workflows.  Diagnostic subsets remain separate from the additive
    /// component total.
    #[serde(default)]
    pub energy_analysis: Option<reglyco_core::EnergyAnalysis>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowReport {
    pub status: String,
    pub clash_status: Option<ClashStatus>,
    pub search: Option<SearchOutcome>,
    pub relaxation: Option<RelaxationDiagnostics>,
    pub provenance: Provenance,
    pub warnings: Vec<String>,
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub analysis: ReportAnalysis,
}

impl WorkflowReport {
    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let contents = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, contents + "\n")
    }

    /// Attach reusable structural analysis to this report. A failed carbohydrate
    /// extraction is retained as a warning so computational results are never lost.
    pub fn analyze_structure(&mut self, structure: &Structure) {
        self.analyze_structure_at_stage(structure, "result");
    }

    /// Analyze an accepted stage (input/parent, Build, Minimize, Optimize,
    /// or an ensemble frame) without retaining transient search populations.
    pub fn analyze_structure_at_stage(&mut self, structure: &Structure, stage: &str) {
        let (glycans, warnings) = glycan_summaries(structure);
        self.analysis.glycans = glycans;
        let torsions = glycosidic_torsions(structure, None);
        self.analysis.glycosidic_torsions = torsions.clone();
        self.analysis.torsion_observations = torsions
            .iter()
            .map(|torsion| self.torsion_observation(structure, torsion, stage, "accepted", None))
            .collect();
        self.warnings.extend(warnings);
        if let Some(search) = &self.search {
            self.analysis.protein_linkage_torsions = search
                .sites
                .iter()
                .map(|site| ProteinLinkageTorsion {
                    site: site.site.residue.to_string(),
                    phi_degrees: site.phi_degrees,
                    psi_degrees: site.psi_degrees,
                    frame: None,
                })
                .collect();
            self.analysis.sterics = search
                .sites
                .iter()
                .map(|site| StericObservation {
                    site: site.site.residue.to_string(),
                    score: site.steric_score,
                    clash_free: site.steric_score <= 1.1,
                })
                .collect();
            self.analysis.clusters = search
                .sites
                .iter()
                .map(|site| ClusterObservation {
                    site: site.site.residue.to_string(),
                    cluster_index: site.cluster_index,
                    main_cluster: site.main_cluster,
                    expected_weight: site.cluster_weight,
                    observed_count: 1,
                })
                .collect();
            self.analysis.attachment_observations = search
                .sites
                .iter()
                .map(|site| AttachmentTorsionObservation {
                    domain: "attachment".into(),
                    origin: "accepted".into(),
                    site: site.site.residue.to_string(),
                    stage: stage.into(),
                    frame: None,
                    phi_degrees: site.phi_degrees,
                    psi_degrees: site.psi_degrees,
                    glycan_index: None,
                    linkage: None,
                    involved_atoms: Vec::new(),
                    population_percentile: None,
                    assessment: attachment_assessment(site.phi_within_vmm95, site.psi_within_vmm95),
                    selected_phi_component: site.phi_component,
                    selected_psi_component: site.psi_component,
                    policy_version: "glycoshape-attachment-vmm-v1".into(),
                })
                .collect();
        }
    }

    /// Add a compact reference object fetched from the GlycoShape API.  The
    /// same reference is used for internal torsion observations and can be
    /// serialized into an offline report bundle.
    pub fn add_torsion_reference(&mut self, reference: TorsionReference) {
        let canonical_linkage = reference.canonical_linkage.clone();
        let identity = torsion_reference_identity(&reference);
        if let Some(existing) = self
            .analysis
            .torsion_references
            .iter_mut()
            .find(|existing| torsion_reference_identity(existing) == identity)
        {
            *existing = reference;
        } else {
            self.analysis.torsion_references.push(reference);
        }
        // A reference can arrive after the accepted observations (for
        // example when a browser session resolves assets lazily). Reclassify
        // the already retained points in-place so offline and live reports
        // expose identical percentiles.
        if let Some(reference) = self
            .analysis
            .torsion_references
            .iter()
            .find(|item| item.canonical_linkage == canonical_linkage)
            .cloned()
        {
            for observation in &mut self.analysis.torsion_observations {
                if observation.linkage != canonical_linkage {
                    continue;
                }
                let percentile = torsion_percentile(
                    &reference,
                    observation.phi_degrees,
                    observation.psi_degrees,
                );
                observation.population_percentile = Some(percentile);
                observation.assessment = classify_torsion_percentile(percentile);
                observation.nearest_population = nearest_population(
                    &reference,
                    observation.phi_degrees,
                    observation.psi_degrees,
                );
            }
        }
    }

    fn torsion_observation(
        &self,
        structure: &Structure,
        torsion: &LinkageTorsion,
        stage: &str,
        origin: &str,
        frame: Option<usize>,
    ) -> TorsionObservation {
        let canonical = canonical_linkage_id(torsion);
        let reference = self
            .analysis
            .torsion_references
            .iter()
            .find(|reference| reference.canonical_linkage == canonical);
        let (percentile, assessment, nearest) = reference
            .map(|reference| {
                let percentile =
                    torsion_percentile(reference, torsion.phi_degrees, torsion.psi_degrees);
                let assessment = classify_torsion_percentile(percentile);
                (
                    Some(percentile),
                    assessment,
                    nearest_population(reference, torsion.phi_degrees, torsion.psi_degrees),
                )
            })
            .unwrap_or((None, TorsionAssessment::NoReference, None));
        let site = structure
            .metadata()
            .glycan_trees
            .iter()
            .find(|tree| {
                torsion
                    .donor_residue
                    .as_ref()
                    .is_some_and(|residue| tree.residue_ids.contains(residue))
                    || torsion
                        .acceptor_residue
                        .as_ref()
                        .is_some_and(|residue| tree.residue_ids.contains(residue))
            })
            .and_then(|tree| tree.attachment_site.as_ref())
            .map(ToString::to_string);
        let branch_path = [
            torsion.donor_name.clone(),
            torsion.donor_position.map(|value| value.to_string()),
            torsion.acceptor_name.clone(),
            torsion.acceptor_position.map(|value| value.to_string()),
        ]
        .into_iter()
        .flatten()
        .collect();
        TorsionObservation {
            domain: "internal_glycosidic".into(),
            origin: origin.into(),
            stage: stage.into(),
            frame,
            site,
            branch_path,
            glycan_index: torsion.glycan_index,
            linkage: canonical,
            involved_atoms: [
                torsion
                    .donor_residue
                    .as_ref()
                    .zip(torsion.donor_name.as_ref())
                    .map(|(residue, name)| format!("{residue}:{name}")),
                torsion
                    .acceptor_residue
                    .as_ref()
                    .zip(torsion.acceptor_name.as_ref())
                    .map(|(residue, name)| format!("{residue}:{name}")),
            ]
            .into_iter()
            .flatten()
            .collect(),
            phi_degrees: torsion.phi_degrees,
            psi_degrees: torsion.psi_degrees,
            omega_degrees: torsion.omega_degrees,
            population_percentile: percentile,
            assessment,
            nearest_population: nearest,
            selected_population: None,
            policy_version: "glycoshape-torsion-v1".into(),
        }
    }

    /// Add one accepted ensemble frame while retaining the complete torsion,
    /// cluster, steric, and native-probability observations in `report.json`.
    pub fn record_ensemble_frame(
        &mut self,
        frame: usize,
        structure: &Structure,
        sites: &[SearchSiteResult],
        log_native_probability: f64,
    ) {
        let frame_torsions = glycosidic_torsions(structure, Some(frame));
        self.analysis
            .glycosidic_torsions
            .extend(frame_torsions.clone());
        let frame_observations = frame_torsions
            .iter()
            .map(|torsion| {
                self.torsion_observation(structure, torsion, "ensemble", "accepted", Some(frame))
            })
            .collect::<Vec<_>>();
        self.analysis
            .torsion_observations
            .extend(frame_observations);
        self.analysis
            .protein_linkage_torsions
            .extend(sites.iter().map(|site| ProteinLinkageTorsion {
                site: site.site.residue.to_string(),
                phi_degrees: site.phi_degrees,
                psi_degrees: site.psi_degrees,
                frame: Some(frame),
            }));
        let attachment_observations = sites
            .iter()
            .map(|site| AttachmentTorsionObservation {
                domain: "attachment".into(),
                origin: "accepted".into(),
                site: site.site.residue.to_string(),
                stage: "ensemble".into(),
                frame: Some(frame),
                phi_degrees: site.phi_degrees,
                psi_degrees: site.psi_degrees,
                glycan_index: None,
                linkage: None,
                involved_atoms: Vec::new(),
                population_percentile: None,
                assessment: attachment_assessment(site.phi_within_vmm95, site.psi_within_vmm95),
                selected_phi_component: site.phi_component,
                selected_psi_component: site.psi_component,
                policy_version: "glycoshape-attachment-vmm-v1".into(),
            })
            .collect::<Vec<_>>();
        self.analysis
            .attachment_observations
            .extend(attachment_observations);
        self.analysis
            .sterics
            .extend(sites.iter().map(|site| StericObservation {
                site: site.site.residue.to_string(),
                score: site.steric_score,
                clash_free: site.steric_score <= 1.1,
            }));
        for site in sites {
            if let Some(entry) = self.analysis.clusters.iter_mut().find(|entry| {
                entry.site == site.site.residue.to_string()
                    && entry.cluster_index == site.cluster_index
                    && entry.main_cluster == site.main_cluster
            }) {
                entry.observed_count += 1;
            } else {
                self.analysis.clusters.push(ClusterObservation {
                    site: site.site.residue.to_string(),
                    cluster_index: site.cluster_index,
                    main_cluster: site.main_cluster,
                    expected_weight: site.cluster_weight,
                    observed_count: 1,
                });
            }
        }
        self.analysis
            .ensemble
            .get_or_insert_with(EnsembleAnalysis::default)
            .native_log_probabilities
            .push(log_native_probability);
        self.update_ensemble_torsion_summary();
    }

    fn update_ensemble_torsion_summary(&mut self) {
        let mut frames = BTreeSet::new();
        let mut by_linkage = BTreeMap::<String, Vec<&TorsionObservation>>::new();
        for observation in &self.analysis.torsion_observations {
            let Some(frame) = observation.frame else {
                continue;
            };
            frames.insert(frame);
            by_linkage
                .entry(observation.linkage.clone())
                .or_default()
                .push(observation);
        }
        if frames.is_empty() {
            return;
        }
        let mut outlier_frames = BTreeSet::new();
        let mut linkage_outlier_rates = BTreeMap::new();
        let mut circular_means = BTreeMap::new();
        let mut circular_dispersion = BTreeMap::new();
        let mut cluster_coverage = BTreeMap::new();
        for (linkage, observations) in by_linkage {
            let phi = observations
                .iter()
                .filter_map(|observation| observation.phi_degrees)
                .collect::<Vec<_>>();
            let psi = observations
                .iter()
                .filter_map(|observation| observation.psi_degrees)
                .collect::<Vec<_>>();
            if let (Some(phi_mean), Some(psi_mean)) =
                (circular_mean_periodic(&phi), circular_mean_periodic(&psi))
            {
                circular_means.insert(linkage.clone(), (phi_mean, psi_mean));
            }
            let resultant = |values: &[f64]| {
                if values.is_empty() {
                    return 0.0;
                }
                let (sin_sum, cos_sum) = values.iter().fold((0.0, 0.0), |(sin, cos), value| {
                    let radians = value.to_radians();
                    (sin + radians.sin(), cos + radians.cos())
                });
                (sin_sum.hypot(cos_sum) / values.len() as f64).clamp(0.0, 1.0)
            };
            circular_dispersion.insert(
                linkage.clone(),
                (1.0 - resultant(&phi), 1.0 - resultant(&psi)),
            );
            let outliers = observations
                .iter()
                .filter(|observation| observation.assessment == TorsionAssessment::Outlier)
                .filter_map(|observation| observation.frame)
                .collect::<BTreeSet<_>>();
            outlier_frames.extend(outliers.iter().copied());
            linkage_outlier_rates.insert(
                linkage.clone(),
                outliers.len() as f64 / observations.len().max(1) as f64,
            );
            let clusters = observations
                .iter()
                .filter_map(|observation| observation.nearest_population)
                .collect::<BTreeSet<_>>();
            let expected_clusters = self
                .analysis
                .torsion_references
                .iter()
                .find(|reference| reference.canonical_linkage == linkage)
                .map(|reference| {
                    reference
                        .populations
                        .iter()
                        .filter(|population| population.level <= 1)
                        .count()
                })
                .unwrap_or(0);
            let denominator = expected_clusters.max(1);
            cluster_coverage.insert(
                linkage,
                (clusters.len() as f64 / denominator as f64).clamp(0.0, 1.0),
            );
        }
        self.analysis.ensemble_torsion = Some(EnsembleTorsionSummary {
            frames: frames.len(),
            observed_frames: frames.into_iter().collect(),
            outlier_frames: outlier_frames.into_iter().collect(),
            linkage_outlier_rates,
            circular_means,
            circular_dispersion,
            cluster_coverage,
        });
    }

    /// Register every native cluster before frame sampling so a report can
    /// distinguish an absent cluster from a cluster with zero native weight.
    pub fn set_expected_clusters(&mut self, sites: &[SearchSite]) {
        for site in sites {
            for conformer in &site.ensemble.conformers {
                if self.analysis.clusters.iter().any(|entry| {
                    entry.site == site.site.residue.to_string()
                        && entry.cluster_index == conformer.cluster_index
                        && entry.main_cluster == conformer.main_cluster
                }) {
                    continue;
                }
                self.analysis.clusters.push(ClusterObservation {
                    site: site.site.residue.to_string(),
                    cluster_index: conformer.cluster_index,
                    main_cluster: conformer.main_cluster,
                    expected_weight: conformer.cluster_weight,
                    observed_count: 0,
                });
            }
        }
    }

    /// Record the glycan identities needed by reports without adding
    /// structure-specific torsion or search diagnostics. This is useful for
    /// workflows whose primary result is a SAXS fit but whose report should
    /// still open with the actual SNFG identity and attachment information.
    pub fn analyze_glycan_identity(&mut self, structure: &Structure) {
        let (glycans, warnings) = glycan_summaries(structure);
        self.analysis.glycans = glycans;
        self.warnings.extend(warnings);
    }

    /// Write the complete reproducible PDF bundle. `report.json` is written
    /// before rendering so a Typst failure cannot erase scientific results.
    pub fn write_bundle(&self, output_dir: impl AsRef<Path>) -> Result<()> {
        self.write_bundle_with_prefix(output_dir, "report")
    }

    /// Write a report bundle with a caller-selected filename prefix. This
    /// preserves the legacy `validate --output FILE` interface by allowing
    /// `FILE-report.pdf` artifacts to sit beside the validation JSON.
    pub fn write_bundle_with_prefix(
        &self,
        output_dir: impl AsRef<Path>,
        prefix: &str,
    ) -> Result<()> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)?;
        let assets_name = format!("{prefix}-assets");
        let mut rendered = self.clone();
        for glycan in &mut rendered.analysis.glycans {
            if glycan.snfg_svg.is_some() {
                // Keep the source in the canonical placeholder namespace so
                // the single replacement below also works for prefixes that
                // themselves contain the text "report".
                glycan.snfg_asset = Some(format!("report-assets/glycan-{}.svg", glycan.index));
            }
        }
        if let Some(saxs) = rendered.analysis.saxs.as_mut() {
            if saxs.diagnostic_svg.is_some() {
                saxs.diagnostic_asset = Some("report-assets/saxs-diagnostic.svg".into());
            }
            if !saxs.site_occupancy.is_empty() {
                saxs.occupancy_asset = Some("report-assets/saxs-occupancy.svg".into());
            }
            if !saxs.candidate_visuals.is_empty() {
                if saxs.site_occupancy.is_empty() {
                    saxs.candidate_asset = Some("report-assets/saxs-candidates.svg".into());
                }
                for (index, candidate) in saxs.candidate_visuals.iter_mut().enumerate() {
                    if candidate.snfg_svg.is_some() {
                        candidate.snfg_asset =
                            Some(format!("report-assets/saxs-candidate-{}.svg", index + 1));
                    }
                }
            }
        }
        rendered.write_json(output_dir.join(format!("{prefix}.json")))?;
        fs::write(
            output_dir.join(format!("{prefix}-validation.json")),
            serde_json::to_string_pretty(&rendered.analysis.validation)? + "\n",
        )?;
        fs::write(
            output_dir.join(format!("{prefix}-torsions.csv")),
            torsions_csv(
                &rendered.analysis.torsion_observations,
                &rendered.analysis.attachment_observations,
            ),
        )?;
        if let Some(energy_analysis) = rendered.analysis.energy_analysis.as_ref() {
            fs::write(
                output_dir.join(format!("{prefix}-energy.csv")),
                energy_analysis_csv(energy_analysis),
            )?;
        }
        // Keep the reference artifact self-contained: attachment VMM mixtures
        // are a separate validation domain from internal glycosidic contours,
        // but both are needed to reproduce the report offline.  The envelope
        // is versioned so older consumers can still read the `references`
        // array while newer consumers can render the attachment gate.
        let torsion_reference_artifact = serde_json::json!({
            "schemaVersion": "glycoshape-torsion-v1",
            "sourceDatabase": "GlycoShape",
            "references": rendered.analysis.torsion_references,
            "attachmentReferences": rendered.analysis.attachment_references,
        });
        fs::write(
            output_dir.join(format!("{prefix}-torsion-reference.json")),
            serde_json::to_string_pretty(&torsion_reference_artifact)? + "\n",
        )?;
        let assets = output_dir.join(&assets_name);
        // The report owns this generated asset directory.  Remove files from
        // an earlier report before writing so obsolete heatmaps/colourbars
        // cannot remain beside the current contour or point plot and confuse
        // downstream consumers inspecting the bundle.
        if assets.is_dir() {
            for entry in fs::read_dir(&assets)? {
                let path = entry?.path();
                if path.is_file() {
                    fs::remove_file(path)?;
                }
            }
        } else {
            fs::create_dir_all(&assets)?;
        }
        write_snfg_assets(&rendered.analysis.glycans, &assets)?;
        write_plot_assets(&rendered, &assets)?;
        let source = typst_source(&rendered).replace("report-assets/", &format!("{assets_name}/"));
        fs::write(output_dir.join(format!("{prefix}.typ")), &source)?;
        let pdf = compile_typst(&source, output_dir)?;
        fs::write(output_dir.join(format!("{prefix}.pdf")), pdf)?;
        Ok(())
    }
}

fn glycan_summaries(structure: &Structure) -> (Vec<GlycanSummary>, Vec<String>) {
    let mut warnings = Vec::new();
    let pdb = structure.to_pdb_string();
    let extracted = match extract_glycans_with_provenance_from_str(&pdb, false) {
        Ok(glycans) => glycans,
        Err(error) => {
            warnings.push(format!("crabWURCS could not extract glycans: {error}"));
            return (Vec::new(), warnings);
        }
    };
    let expected_attachments = structure
        .metadata()
        .glycosylation_sites
        .iter()
        .filter_map(|site| {
            structure
                .residues()
                .into_iter()
                .find(|residue| residue.id == site.protein_residue)
                .map(|residue| {
                    format!(
                        "{}/{}/{}",
                        residue.id.chain, residue.name, residue.id.number
                    )
                })
        })
        .collect::<BTreeSet<_>>();
    let extracted = if expected_attachments.is_empty() {
        extracted
    } else {
        extracted
            .into_iter()
            .filter(|glycan| {
                glycan
                    .attachment_site
                    .as_ref()
                    .is_some_and(|site| expected_attachments.contains(site))
            })
            .collect()
    };
    let result = extracted
        .into_iter()
        .enumerate()
        .map(|(index, glycan)| {
            let wurcs = write_wurcs_canonical(&glycan.graph).ok();
            let iupac = write_iupac_condensed_canonical(&glycan.graph).ok();
            let snfg_svg = render_svg(&glycan.graph).ok();
            GlycanSummary {
                index: index + 1,
                attachment_site: glycan.attachment_site,
                residue_count: glycan.graph.inner().node_count(),
                wurcs,
                iupac,
                snfg_asset: snfg_svg
                    .as_ref()
                    .map(|_| format!("report-assets/glycan-{}.svg", index + 1)),
                snfg_svg,
                snfg_graph: Some(glycan.graph),
                snfg_residues: Some(glycan.residues),
            }
        })
        .collect();
    (result, warnings)
}

/// Render the first glycan found in a GlySys structure as an SNFG SVG.
///
/// SAXS candidate reports use this for the first conformer in each candidate
/// ensemble so the report can show the actual glycan identity instead of a
/// text-only identifier. A missing/unsupported graph is intentionally a
/// recoverable `None`; the SAXS fit itself remains valid.
pub fn snfg_svg_for_structure(structure: &Structure) -> Option<String> {
    glycan_summaries(structure)
        .0
        .into_iter()
        .find_map(|glycan| glycan.snfg_svg)
}

/// Smallest periodic distance between two angles in degrees.
pub fn circular_angle_distance(first: f64, second: f64) -> f64 {
    (first - second + 180.0).rem_euclid(360.0) - 180.0
}

/// Stable linkage identity independent of PDB residue numbering.  Branch
/// paths can be prepended by callers when two identical donor/acceptor pairs
/// occur in one glycan.
pub fn canonical_linkage_id(torsion: &LinkageTorsion) -> String {
    match (
        torsion.donor_name.as_deref(),
        torsion.donor_position,
        torsion.acceptor_name.as_deref(),
        torsion.acceptor_position,
    ) {
        (Some(donor), Some(donor_position), Some(acceptor), Some(acceptor_position)) => format!(
            "{}{}-{}{}",
            donor.to_ascii_uppercase(),
            donor_position,
            acceptor.to_ascii_uppercase(),
            acceptor_position
        ),
        _ => torsion.linkage.clone(),
    }
}

/// Stable identity for a persisted torsion reference.  A canonical linkage
/// can occur on more than one branch of a branched glycan, and a source
/// database may publish a revised mixture without changing that linkage
/// label. Keep those records distinct so offline reports never silently
/// replace one branch or source version with another.
fn torsion_reference_identity(reference: &TorsionReference) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}",
        reference.canonical_linkage,
        reference.branch_path.join("/"),
        reference.version,
        reference.content_hash,
    )
}

/// Estimate a population percentile from a reference HDR grid.  The grid is
/// periodic and stores either counts or normalized density.  The returned
/// value is deliberately advisory for internal glycosidic torsions.
pub fn torsion_percentile(reference: &TorsionReference, phi: Option<f64>, psi: Option<f64>) -> f64 {
    let Some((phi, psi, contour)) = phi
        .zip(psi)
        .zip(reference.phi_psi.as_ref())
        .map(|((phi, psi), contour)| (phi, psi, contour))
    else {
        return 0.0;
    };
    if contour.grid.is_empty() || contour.bins == 0 {
        return 0.0;
    }
    let bin = |angle: f64| {
        ((angle + 180.0).rem_euclid(360.0) / 360.0 * contour.bins as f64)
            .floor()
            .clamp(0.0, contour.bins.saturating_sub(1) as f64) as usize
    };
    let index = bin(phi) + bin(psi) * contour.bins;
    let density = contour.grid.get(index).copied().unwrap_or(0.0).max(0.0);
    let total = contour
        .grid
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .sum::<f64>();
    if total <= 0.0 {
        return 0.0;
    }
    // Approximate the highest-density-region percentile by the fraction of
    // density in cells at least as populated as the observed cell.
    let mass = contour
        .grid
        .iter()
        .filter(|value| **value + f64::EPSILON >= density)
        .copied()
        .sum::<f64>();
    (mass / total * 100.0).clamp(0.0, 100.0)
}

pub fn classify_torsion_percentile(percentile: f64) -> TorsionAssessment {
    if !percentile.is_finite() || percentile <= 0.0 {
        TorsionAssessment::NoReference
    } else if percentile <= 50.0 {
        TorsionAssessment::Core
    } else if percentile <= 80.0 {
        TorsionAssessment::Allowed
    } else if percentile <= 95.0 {
        TorsionAssessment::Tail
    } else {
        TorsionAssessment::Outlier
    }
}

/// Circular mean for a periodic sample.  Returns `None` for an empty sample.
pub fn circular_mean_periodic(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let (sin_sum, cos_sum) = values.iter().fold((0.0, 0.0), |(sin_sum, cos_sum), value| {
        let radians = value.to_radians();
        (sin_sum + radians.sin(), cos_sum + radians.cos())
    });
    Some(sin_sum.atan2(cos_sum).to_degrees())
}

fn attachment_assessment(phi_within: Option<bool>, psi_within: Option<bool>) -> TorsionAssessment {
    match (phi_within, psi_within) {
        (Some(true), Some(true)) => TorsionAssessment::Allowed,
        (Some(false), _) | (_, Some(false)) => TorsionAssessment::Outlier,
        _ => TorsionAssessment::NoReference,
    }
}

fn nearest_population(
    reference: &TorsionReference,
    phi: Option<f64>,
    psi: Option<f64>,
) -> Option<usize> {
    let (phi, psi) = phi.zip(psi)?;
    reference
        .populations
        .iter()
        .filter_map(|population| {
            let p = population.phi_mean?;
            let q = population.psi_mean?;
            let distance =
                circular_angle_distance(phi, p).powi(2) + circular_angle_distance(psi, q).powi(2);
            Some((distance, population.index))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, index)| index)
}

fn write_snfg_assets(glycans: &[GlycanSummary], assets: &Path) -> Result<()> {
    for glycan in glycans {
        if let Some(svg) = &glycan.snfg_svg {
            fs::write(assets.join(format!("glycan-{}.svg", glycan.index)), svg)?;
        }
    }
    Ok(())
}

fn torsions_csv(
    observations: &[TorsionObservation],
    attachments: &[AttachmentTorsionObservation],
) -> String {
    let mut output = String::from(
        "domain,origin,stage,frame,glycan_index,site,branch_path,linkage,phi_degrees,psi_degrees,omega_degrees,population_percentile,assessment,selected_population,selected_phi_component,selected_psi_component,involved_atoms,policy_version\n",
    );
    for observation in observations {
        let row = [
            csv_field(&observation.domain),
            csv_field(&observation.origin),
            csv_field(&observation.stage),
            observation
                .frame
                .map(|frame| frame.to_string())
                .unwrap_or_default(),
            observation.glycan_index.to_string(),
            csv_field(observation.site.as_deref().unwrap_or_default()),
            csv_field(&observation.branch_path.join("/")),
            csv_field(&observation.linkage),
            observation
                .phi_degrees
                .map(|value| value.to_string())
                .unwrap_or_default(),
            observation
                .psi_degrees
                .map(|value| value.to_string())
                .unwrap_or_default(),
            observation
                .omega_degrees
                .map(|value| value.to_string())
                .unwrap_or_default(),
            observation
                .population_percentile
                .map(|value| value.to_string())
                .unwrap_or_default(),
            serde_json::to_string(&observation.assessment)
                .unwrap_or_else(|_| "\"no_reference\"".into())
                .trim_matches('"')
                .to_string(),
            observation
                .selected_population
                .map(|value| value.to_string())
                .unwrap_or_default(),
            String::new(),
            String::new(),
            csv_field(&observation.involved_atoms.join(";")),
            csv_field(&observation.policy_version),
        ];
        output.push_str(&row.join(","));
        output.push('\n');
    }
    for observation in attachments {
        let row = [
            "attachment".to_string(),
            "accepted".to_string(),
            csv_field(&observation.stage),
            observation
                .frame
                .map(|frame| frame.to_string())
                .unwrap_or_default(),
            String::new(),
            String::new(),
            csv_field(&observation.site),
            String::new(),
            csv_field(observation.linkage.as_deref().unwrap_or_default()),
            observation.phi_degrees.to_string(),
            observation.psi_degrees.to_string(),
            String::new(),
            observation
                .population_percentile
                .map(|value| value.to_string())
                .unwrap_or_default(),
            serde_json::to_string(&observation.assessment)
                .unwrap_or_else(|_| "\"no_reference\"".into())
                .trim_matches('"')
                .to_string(),
            String::new(),
            observation
                .selected_phi_component
                .map(|value| value.to_string())
                .unwrap_or_default(),
            observation
                .selected_psi_component
                .map(|value| value.to_string())
                .unwrap_or_default(),
            csv_field(&observation.involved_atoms.join(";")),
            csv_field(&observation.policy_version),
        ];
        output.push_str(&row.join(","));
        output.push('\n');
    }
    output
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Flatten the energy explanation into a compact CSV while retaining the
/// distinction between additive global components and diagnostic subsets.
/// This artifact is intentionally typed rather than assembled through JSON so
/// native reports keep the same schema and precision as the interactive report.
fn energy_analysis_csv(analysis: &reglyco_core::EnergyAnalysis) -> String {
    let mut output = String::from(
        "kind,site,glycan,linkage,atoms,component,value,units,backend,drives_selection\n",
    );
    let units = csv_field(&analysis.units);
    let backend = csv_field(&analysis.backend);
    let drives_selection = analysis.drives_selection.to_string();
    let mut global = |name: &str, value: f64| {
        output.push_str(&format!(
            "global,,,,,{},{:.9},{},{},{}\n",
            csv_field(name),
            value,
            units,
            backend,
            drives_selection
        ));
    };
    global("bonds", analysis.components.bonds);
    global("angles", analysis.components.angles);
    global("proper_torsions", analysis.components.proper_torsions);
    global("improper_torsions", analysis.components.improper_torsions);
    global("van_der_waals", analysis.components.van_der_waals);
    global("electrostatics", analysis.components.electrostatics);
    global("generalized_born", analysis.components.generalized_born);
    global("surface_area", analysis.components.surface_area);
    global("restraints", analysis.components.restraints);
    global(
        "dispersion_correction",
        analysis.components.dispersion_correction,
    );
    if let Some(value) = analysis.selected_score {
        output.push_str(&format!(
            "objective,,,,,selected_score,{:.9},{},{},{}\n",
            value, units, backend, drives_selection
        ));
    }
    if let Some(value) = analysis.diagnostic_remainder {
        output.push_str(&format!(
            "diagnostic,,,,,remainder,{:.9},{},{},false\n",
            value, units, backend
        ));
    }
    for interaction in &analysis.per_glycan_interactions {
        for (component, value) in [
            ("van_der_waals", interaction.van_der_waals),
            ("electrostatics", interaction.electrostatics),
            ("total", interaction.total),
        ] {
            output.push_str(&format!(
                "protein_glycan,{},{},,,{},{:.9},{},{},{}\n",
                csv_field(&interaction.site),
                csv_field(&interaction.glycan_id),
                csv_field(component),
                value,
                units,
                backend,
                drives_selection
            ));
        }
    }
    for torsion in &analysis.glycosidic_torsions {
        let atoms = torsion
            .atoms
            .iter()
            .map(|atom| atom.to_string())
            .collect::<Vec<_>>()
            .join("-");
        output.push_str(&format!(
            "glycosidic_torsion,{},{},{},{},{},{:.9},{},{},false\n",
            csv_field(&torsion.site),
            "",
            csv_field(&torsion.linkage),
            csv_field(&atoms),
            "energy",
            torsion.energy,
            units,
            backend
        ));
    }
    output
}

fn glycosidic_torsions(structure: &Structure, frame: Option<usize>) -> Vec<LinkageTorsion> {
    let mut torsions = Vec::new();
    let atom_map = structure
        .atoms()
        .into_iter()
        .map(|atom| (atom.id, atom))
        .collect::<BTreeMap<_, _>>();
    let glycan_residues = structure
        .metadata()
        .glycan_trees
        .iter()
        .flat_map(|tree| tree.residue_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let residue_names = structure
        .residues()
        .into_iter()
        .map(|residue| (residue.id, residue.name))
        .collect::<BTreeMap<_, _>>();
    let mut bond_pairs = structure.bonds().into_iter().collect::<BTreeSet<_>>();
    // Deposited carbohydrate PDBs frequently omit CONECT/LINK records.  Use
    // the declared tree and a chemically specific anomeric-C/O pattern to
    // recover only plausible glycosidic junctions; proximity alone would
    // turn crowded but unrelated sugars into fake linkages.
    let atoms = structure.atoms();
    for tree in &structure.metadata().glycan_trees {
        let ids = tree.residue_ids.iter().collect::<BTreeSet<_>>();
        let tree_atoms = atoms
            .iter()
            .filter(|atom| ids.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H"))
            .collect::<Vec<_>>();
        for (index, left) in tree_atoms.iter().enumerate() {
            for right in tree_atoms.iter().skip(index + 1) {
                if left.residue == right.residue
                    || !likely_glycosidic_atom_pair(left, right)
                    || distance(left.position, right.position) < 1.1
                    || distance(left.position, right.position) > 1.85
                {
                    continue;
                }
                bond_pairs.insert(if left.id <= right.id {
                    (left.id, right.id)
                } else {
                    (right.id, left.id)
                });
            }
        }
    }
    for (first, second) in bond_pairs {
        let Some(a) = atom_map.get(&first) else {
            continue;
        };
        let Some(b) = atom_map.get(&second) else {
            continue;
        };
        if a.residue == b.residue
            || !glycan_residues.contains(&a.residue)
            || !glycan_residues.contains(&b.residue)
        {
            continue;
        }
        let (donor, acceptor) = if is_anomeric(&a.name)
            && is_acceptor_oxygen(
                &b.name,
                residue_names
                    .get(&b.residue)
                    .map(String::as_str)
                    .unwrap_or_default(),
            ) {
            (a, b)
        } else if is_anomeric(&b.name)
            && is_acceptor_oxygen(
                &a.name,
                residue_names
                    .get(&a.residue)
                    .map(String::as_str)
                    .unwrap_or_default(),
            )
        {
            (b, a)
        } else {
            continue;
        };
        // Pyranoses use O5 as the ring oxygen (with O6 retained as a
        // deposited-asset fallback); furanoses such as Ara use O4.  Picking
        // the ring anchor from the donor component avoids reporting an
        // attachment torsion against an exocyclic hydroxyl for furanose
        // glycans.
        let donor_name = residue_names
            .get(&donor.residue)
            .map(|name| name.trim().to_ascii_uppercase())
            .unwrap_or_default();
        let donor_ring = if is_furanose_residue(&donor_name) {
            atom_position(structure, &donor.residue, "O4")
                .or_else(|| atom_position(structure, &donor.residue, "O5"))
                .or_else(|| atom_position(structure, &donor.residue, "O6"))
        } else if is_sialic_residue(&donor_name) {
            atom_position(structure, &donor.residue, "O6")
                .or_else(|| atom_position(structure, &donor.residue, "O5"))
        } else {
            atom_position(structure, &donor.residue, "O5")
                .or_else(|| atom_position(structure, &donor.residue, "O6"))
                .or_else(|| atom_position(structure, &donor.residue, "O4"))
        };
        let acceptor_name = acceptor.name.trim().to_ascii_uppercase();
        let position = acceptor_name
            .strip_prefix('O')
            .and_then(|value| value.parse::<u8>().ok());
        let Some(position) = position else { continue };
        let acceptor_carbon = atom_position(structure, &acceptor.residue, &format!("C{position}"));
        let previous = position
            .checked_sub(1)
            .and_then(|value| atom_position(structure, &acceptor.residue, &format!("C{value}")));
        let phi = donor_ring
            .zip(acceptor_carbon)
            .map(|(ring, carbon)| dihedral(ring, donor.position, acceptor.position, carbon));
        let psi = acceptor_carbon.zip(previous).map(|(carbon, previous)| {
            dihedral(donor.position, acceptor.position, carbon, previous)
        });
        let omega = (position == 6)
            .then(|| {
                let carbon = acceptor_carbon?;
                let c5 = atom_position(structure, &acceptor.residue, "C5")?;
                let acceptor_residue_name = residue_names
                    .get(&acceptor.residue)
                    .map(String::as_str)
                    .unwrap_or_default();
                let ring = if is_furanose_residue(acceptor_residue_name) {
                    atom_position(structure, &acceptor.residue, "O4")
                        .or_else(|| atom_position(structure, &acceptor.residue, "O5"))
                } else if is_sialic_residue(acceptor_residue_name) {
                    atom_position(structure, &acceptor.residue, "O6")
                        .or_else(|| atom_position(structure, &acceptor.residue, "O5"))
                } else {
                    atom_position(structure, &acceptor.residue, "O5")
                        .or_else(|| atom_position(structure, &acceptor.residue, "O6"))
                }?;
                Some(dihedral(acceptor.position, carbon, c5, ring))
            })
            .flatten();
        let glycan_index = structure
            .metadata()
            .glycan_trees
            .iter()
            .position(|tree| tree.residue_ids.contains(&donor.residue))
            .map_or(0, |index| index + 1);
        torsions.push(LinkageTorsion {
            glycan_index,
            linkage: format!(
                "{} {}–{} {}",
                donor.residue, donor.name, acceptor.residue, acceptor.name
            ),
            phi_degrees: phi,
            psi_degrees: psi,
            omega_degrees: omega,
            frame,
            donor_name: residue_names.get(&donor.residue).cloned(),
            acceptor_name: residue_names.get(&acceptor.residue).cloned(),
            donor_position: donor
                .name
                .trim()
                .to_ascii_uppercase()
                .strip_prefix('C')
                .and_then(|value| value.parse::<u8>().ok()),
            acceptor_position: Some(position),
            donor_residue: Some(donor.residue.clone()),
            acceptor_residue: Some(acceptor.residue.clone()),
        });
    }
    torsions
}

fn is_anomeric(name: &str) -> bool {
    let name = name.trim().to_ascii_uppercase();
    matches!(name.as_str(), "C1" | "C2" | "C1A" | "C2A")
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

fn is_acceptor_oxygen(name: &str, residue_name: &str) -> bool {
    let name = name.trim().to_ascii_uppercase();
    // Ring oxygen numbering is component-specific: O5 is the ring atom for
    // ordinary pyranoses, O4 for the supported furanoses, and O6 for sialic
    // acids/KDO.  Exclude only that ring oxygen; O4 on a pyranose (1→4) and
    // O5 on a furanose/sialic acid remain valid acceptors.
    let ring_oxygen = if is_furanose_residue(residue_name) {
        "O4"
    } else if is_sialic_residue(residue_name) {
        "O6"
    } else {
        "O5"
    };
    name.starts_with('O') && name != ring_oxygen
}

fn atom_position(structure: &Structure, residue: &ResidueId, name: &str) -> Option<Vec3> {
    structure
        .find_atom(residue, name)
        .and_then(|atom| structure.atom(atom))
        .map(|atom| atom.position)
}

fn distance(first: Vec3, second: Vec3) -> f64 {
    ((first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2))
        .sqrt()
}

fn dihedral(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let b1 = normalize(sub(c, b));
    let v = sub(sub(a, b), scale(b1, dot(sub(a, b), b1)));
    let w = sub(sub(d, c), scale(b1, dot(sub(d, c), b1)));
    dot(cross(b1, v), w).atan2(dot(v, w)).to_degrees()
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}
fn scale(a: Vec3, f: f64) -> Vec3 {
    Vec3 {
        x: a.x * f,
        y: a.y * f,
        z: a.z * f,
    }
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}
fn normalize(a: Vec3) -> Vec3 {
    let n = dot(a, a).sqrt();
    if n > 1.0e-12 {
        scale(a, 1.0 / n)
    } else {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

// ============================================================================
// Plot styling
// ============================================================================

/// ReGlyco accent color used across all report figures.
const ACCENT: RGBColor = RGBColor(15, 107, 107);
/// Muted dark gray used for plot text.
const TEXT_GRAY: RGBColor = RGBColor(70, 76, 76);
/// Faint gray used for grid lines.
const GRID_GRAY: RGBColor = RGBColor(216, 224, 224);
/// Mid gray used for axes and ticks.
const AXIS_GRAY: RGBColor = RGBColor(150, 158, 158);
/// Green used for favorable extrema.
const GOOD_COLOR: RGBColor = RGBColor(5, 150, 105);
/// Red used for unfavorable extrema.
const BAD_COLOR: RGBColor = RGBColor(203, 76, 76);

/// Colorblind-aware categorical palette, accent first.
const SERIES: [RGBColor; 8] = [
    ACCENT,
    RGBColor(217, 119, 6),
    RGBColor(79, 70, 229),
    RGBColor(5, 150, 105),
    RGBColor(192, 38, 211),
    RGBColor(220, 38, 38),
    RGBColor(37, 99, 235),
    RGBColor(154, 52, 18),
];

fn series_color(index: usize) -> RGBColor {
    SERIES[index % SERIES.len()]
}

/// Rough sans-serif glyph width used to center or right-align plotters labels.
fn text_width(text: &str, size: f64) -> i32 {
    (text.chars().count() as f64 * size * 0.58).round() as i32
}

fn draw_text<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    text: &str,
    x: i32,
    y: i32,
    size: f64,
    color: RGBColor,
) -> Result<()> {
    root.draw(&Text::new(
        text,
        (x, y),
        ("sans-serif", size).into_font().color(&color),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))
}

fn draw_title<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    text: &str,
    center_x: i32,
    y: i32,
    size: f64,
    color: RGBColor,
) -> Result<()> {
    draw_text(
        root,
        text,
        center_x - text_width(text, size) / 2,
        y,
        size,
        color,
    )
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    if label.chars().count() <= max_chars {
        label.to_string()
    } else {
        format!("{}…", label.chars().take(max_chars - 1).collect::<String>())
    }
}

// ============================================================================
// Kernel density estimation
// ============================================================================

/// Gaussian kernel function for KDE
fn gaussian_kernel(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Silverman's rule of thumb for bandwidth selection, using the smaller of the
/// standard deviation and the normalized interquartile range. Floored so that
/// degenerate samples still produce a usable angular spread.
fn silverman_bandwidth(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 1.0;
    }
    let n = data.len() as f64;
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite torsion angle"));
    let mean = sorted.iter().sum::<f64>() / n;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();
    let q1 = sorted[(sorted.len() - 1) / 4];
    let q3 = sorted[(3 * (sorted.len() - 1)) / 4];
    let iqr = q3 - q1;
    let scale = if iqr > 0.0 { std.min(iqr / 1.34) } else { std };
    (0.9 * scale * n.powf(-0.2)).max(2.0)
}

/// 1D KDE estimation at a point
#[allow(dead_code)]
fn kde_1d(x: f64, data: &[f64], bandwidth: f64) -> f64 {
    let n = data.len() as f64;
    let mut sum = 0.0;
    for &xi in data {
        let u = (x - xi) / bandwidth;
        sum += gaussian_kernel(u);
    }
    sum / (n * bandwidth)
}

/// 2D KDE estimation at a point using a product kernel with per-axis bandwidths
fn kde_2d(
    x: f64,
    y: f64,
    data_x: &[f64],
    data_y: &[f64],
    bandwidth_x: f64,
    bandwidth_y: f64,
) -> f64 {
    let n = data_x.len() as f64;
    let mut sum = 0.0;
    for i in 0..data_x.len() {
        let u = circular_delta(x, data_x[i]) / bandwidth_x;
        let v = circular_delta(y, data_y[i]) / bandwidth_y;
        sum += gaussian_kernel(u) * gaussian_kernel(v);
    }
    sum / (n * bandwidth_x * bandwidth_y)
}

/// Signed shortest angular difference in degrees.
fn circular_delta(angle: f64, reference: f64) -> f64 {
    (angle - reference + 180.0).rem_euclid(360.0) - 180.0
}

fn circular_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let (sin_sum, cos_sum) = data.iter().fold((0.0, 0.0), |(sin_sum, cos_sum), angle| {
        let radians = angle.to_radians();
        (sin_sum + radians.sin(), cos_sum + radians.cos())
    });
    sin_sum.atan2(cos_sum).to_degrees()
}

/// Select a Silverman bandwidth after unwrapping a circular sample around its
/// circular mean. Angular plots remain smooth without splitting a mode at the
/// ±180° boundary.
fn periodic_bandwidth(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 5.0;
    }
    let center = circular_mean(data);
    let unwrapped = data
        .iter()
        .map(|angle| center + circular_delta(*angle, center))
        .collect::<Vec<_>>();
    silverman_bandwidth(&unwrapped).clamp(5.0, 45.0)
}

// Retained only for compatibility with the older private plot helpers below;
// production ensemble reports use `filled_contour_kde_plot`.
#[allow(dead_code)]
fn density_color(frac: f64) -> RGBColor {
    let t = frac.clamp(0.0, 1.0);
    let channel = |lo: f64, hi: f64| (lo + (hi - lo) * t).round() as u8;
    RGBColor(
        channel(224.0, 15.0),
        channel(242.0, 107.0),
        channel(242.0, 107.0),
    )
}

#[allow(dead_code)]
fn density_heatmap(
    data_x: &[f64],
    data_y: &[f64],
    bandwidth_x: f64,
    bandwidth_y: f64,
    cols: usize,
    rows: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let xs = (0..cols)
        .map(|i| -180.0 + 360.0 * (i as f64 + 0.5) / cols as f64)
        .collect::<Vec<_>>();
    let ys = (0..rows)
        .map(|j| -180.0 + 360.0 * (j as f64 + 0.5) / rows as f64)
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(cols * rows);
    for &y in &ys {
        for &x in &xs {
            values.push(kde_2d(x, y, data_x, data_y, bandwidth_x, bandwidth_y));
        }
    }
    (values, xs, ys)
}

#[allow(dead_code, clippy::too_many_arguments)]
fn draw_heatmap<DB: DrawingBackend>(
    _root: &DrawingArea<DB, Shift>,
    _values: &[f64],
    _xs: &[f64],
    _ys: &[f64],
    _peak: f64,
    _half_size: i32,
    _min_frac: f64,
    _to_x: impl Fn(f64) -> i32,
    _to_y: impl Fn(f64) -> i32,
) -> Result<()> {
    Ok(())
}

fn contour_band_thresholds(values: &[f64], peak: f64) -> Vec<f64> {
    if values.is_empty() || peak <= 0.0 {
        return vec![0.0, 1.0];
    }
    let mut ranked = values.to_vec();
    ranked.sort_by(|a, b| b.total_cmp(a));
    let total = ranked.iter().sum::<f64>().max(f64::EPSILON);
    // Six mass-ranked levels plus the peak boundary produce six filled
    // isobands.  The lowest level keeps sparse tails from becoming noisy.
    let mut thresholds = [0.98, 0.85, 0.70, 0.50, 0.30, 0.10]
        .into_iter()
        .map(|target| {
            let mut cumulative = 0.0;
            ranked
                .iter()
                .find(|value| {
                    cumulative += **value;
                    cumulative / total >= target
                })
                .copied()
                .unwrap_or(peak)
        })
        .collect::<Vec<_>>();
    thresholds.push(peak * (1.0 + 1.0e-6));
    thresholds.sort_by(|a, b| a.total_cmp(b));
    thresholds.dedup_by(|a, b| (*a - *b).abs() <= peak * 1.0e-7);
    if thresholds.len() < 2 {
        vec![peak * 0.05, peak * 1.000001]
    } else {
        thresholds
    }
}

fn draw_contour_bands<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    values: &[f64],
    grid: usize,
    plot_x: i32,
    plot_y: i32,
    plot_w: i32,
    plot_h: i32,
) -> Result<()> {
    let peak = values.iter().copied().fold(0.0, f64::max);
    if peak <= 0.0 {
        return Ok(());
    }
    let step = 360.0 / (grid.saturating_sub(1) as f64);
    let thresholds = contour_band_thresholds(values, peak);
    let bands = ContourBuilder::new(grid, grid, true)
        .x_origin(-180.0)
        .y_origin(-180.0)
        .x_step(step)
        .y_step(step)
        .isobands(values, &thresholds)
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    let colors = [
        RGBColor(224, 242, 242),
        RGBColor(193, 227, 227),
        RGBColor(154, 207, 207),
        RGBColor(112, 184, 184),
        RGBColor(67, 153, 153),
        ACCENT,
    ];
    for (band_index, band) in bands.iter().enumerate() {
        let color = colors[band_index.min(colors.len() - 1)].mix(0.92).filled();
        for polygon in &band.geometry().0 {
            let points = polygon
                .exterior()
                .0
                .iter()
                .map(|point| {
                    (
                        plot_x + ((point.x + 180.0) / 360.0 * plot_w as f64).round() as i32,
                        plot_y
                            + (plot_h as f64 - (point.y + 180.0) / 360.0 * plot_h as f64).round()
                                as i32,
                    )
                })
                .collect::<Vec<_>>();
            if points.len() >= 3 {
                root.draw(&Polygon::new(points, color.clone()))
                    .map_err(|error| ReportError::Plot(error.to_string()))?;
            }
        }
    }
    Ok(())
}

fn filled_contour_kde_plot(path: PathBuf, torsions: &[ProteinLinkageTorsion]) -> Result<()> {
    if torsions.is_empty() {
        return Ok(());
    }
    let mut site_data: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for torsion in torsions {
        site_data
            .entry(torsion.site.clone())
            .or_default()
            .push((torsion.phi_degrees, torsion.psi_degrees));
    }

    const PANEL: i32 = 330;
    const PLOT_LEFT: i32 = 48;
    const PLOT_TOP: i32 = 38;
    const PLOT_RIGHT: i32 = 14;
    const PLOT_BOTTOM: i32 = 40;
    const OUTER_LEFT: i32 = 58;
    const OUTER_TOP: i32 = 42;
    const OUTER_BOTTOM: i32 = 54;
    const GRID: usize = 101;
    let columns = site_data.len().min(2).max(1);
    let rows = site_data.len().div_ceil(columns);
    let width = (OUTER_LEFT + columns as i32 * PANEL + 18) as u32;
    let height = (OUTER_TOP + rows as i32 * PANEL + OUTER_BOTTOM) as u32;
    let root = SVGBackend::new(&path, (width, height)).into_drawing_area();

    for (index, (site, data)) in site_data.iter().enumerate() {
        let panel_x = OUTER_LEFT + (index % columns) as i32 * PANEL;
        let panel_y = OUTER_TOP + (index / columns) as i32 * PANEL;
        let plot_x = panel_x + PLOT_LEFT;
        let plot_y = panel_y + PLOT_TOP;
        let plot_w = PANEL - PLOT_LEFT - PLOT_RIGHT;
        let plot_h = PANEL - PLOT_TOP - PLOT_BOTTOM;
        let (phi, psi): (Vec<_>, Vec<_>) = data.iter().copied().unzip();
        let bandwidth_phi = periodic_bandwidth(&phi);
        let bandwidth_psi = periodic_bandwidth(&psi);
        let axis = |angle: f64, origin: i32, length: i32| {
            origin + ((angle + 180.0) / 360.0 * length as f64).round() as i32
        };
        let to_x = |angle: f64| axis(angle, plot_x, plot_w);
        let to_y = |angle: f64| plot_y + plot_h - (axis(angle, 0, plot_h));
        let mut values = Vec::with_capacity(GRID * GRID);
        for row in 0..GRID {
            let psi_angle = -180.0 + 360.0 * row as f64 / (GRID - 1) as f64;
            for column in 0..GRID {
                let phi_angle = -180.0 + 360.0 * column as f64 / (GRID - 1) as f64;
                values.push(kde_2d(
                    phi_angle,
                    psi_angle,
                    &phi,
                    &psi,
                    bandwidth_phi,
                    bandwidth_psi,
                ));
            }
        }
        draw_contour_bands(&root, &values, GRID, plot_x, plot_y, plot_w, plot_h)?;

        for tick in [-180i32, -90, 0, 90, 180] {
            let x = to_x(tick as f64);
            let y = to_y(tick as f64);
            root.draw(&PathElement::new(
                vec![(x, plot_y), (x, plot_y + plot_h)],
                GRID_GRAY.stroke_width(1),
            ))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
            root.draw(&PathElement::new(
                vec![(plot_x, y), (plot_x + plot_w, y)],
                GRID_GRAY.stroke_width(1),
            ))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
            let label = tick.to_string();
            draw_text(
                &root,
                &label,
                x - text_width(&label, 8.0) / 2,
                plot_y + plot_h + 7,
                8.0,
                TEXT_GRAY,
            )?;
            draw_text(
                &root,
                &label,
                plot_x - text_width(&label, 8.0) - 6,
                y - 4,
                8.0,
                TEXT_GRAY,
            )?;
        }
        root.draw(&PathElement::new(
            vec![
                (plot_x, plot_y),
                (plot_x, plot_y + plot_h),
                (plot_x + plot_w, plot_y + plot_h),
            ],
            AXIS_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        for &(phi_angle, psi_angle) in data {
            let center = (to_x(phi_angle), to_y(psi_angle));
            root.draw(&Circle::new(center, 4, WHITE.filled()))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
            root.draw(&Circle::new(center, 3, ACCENT.stroke_width(1)))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
        }
        draw_title(
            &root,
            site,
            panel_x + PANEL / 2,
            panel_y + 12,
            12.0,
            TEXT_GRAY,
        )?;
    }
    draw_title(
        &root,
        "φ (degrees)",
        OUTER_LEFT + columns as i32 * PANEL / 2,
        height as i32 - 18,
        10.0,
        TEXT_GRAY,
    )?;
    root.draw(&Text::new(
        "ψ (degrees)",
        (16, (OUTER_TOP + rows as i32 * PANEL / 2)),
        ("sans-serif", 10)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

/// Create a periodic filled-contour KDE plot for protein-glycan linkage torsions.
fn protein_linkage_kde_plot(path: PathBuf, torsions: &[ProteinLinkageTorsion]) -> Result<()> {
    filled_contour_kde_plot(path, torsions)
}

/// Single 2D KDE plot for one site: smooth filled contours over the
/// ReGlyco density gradient, a relative-density colorbar, and the raw
/// torsion points emphasized on top.
#[allow(dead_code)]
fn single_kde_plot(path: PathBuf, site: &str, data: &[(f64, f64)]) -> Result<()> {
    let (phi, psi): (Vec<f64>, Vec<f64>) = data.iter().copied().unzip();
    let bandwidth_phi = silverman_bandwidth(&phi);
    let bandwidth_psi = silverman_bandwidth(&psi);

    const WIDTH: u32 = 820;
    const HEIGHT: u32 = 620;
    const MARGIN_LEFT: i32 = 70;
    const MARGIN_TOP: i32 = 64;
    const MARGIN_BOTTOM: i32 = 62;
    const BAR_LEFT: i32 = WIDTH as i32 - 78;
    const BAR_WIDTH: i32 = 14;
    const CELL: i32 = 5;

    let plot_w = (BAR_LEFT - MARGIN_LEFT) as f64;
    let plot_h = (HEIGHT as i32 - MARGIN_TOP - MARGIN_BOTTOM) as f64;
    let to_x = |x: f64| MARGIN_LEFT + ((x + 180.0) / 360.0 * plot_w) as i32;
    let to_y = |y: f64| MARGIN_TOP + (plot_h - (y + 180.0) / 360.0 * plot_h) as i32;

    let root = SVGBackend::new(&path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    // Smooth density surface: one finely shaded cell per grid point. The
    // gradient varies continuously, so the surface reads as a smooth heatmap
    // rather than discrete contour bands.
    let cols = (plot_w as i32 / CELL) as usize;
    let rows = (plot_h as i32 / CELL) as usize;
    let (values, xs, ys) = density_heatmap(&phi, &psi, bandwidth_phi, bandwidth_psi, cols, rows);
    let peak = values.iter().copied().fold(0.0f64, f64::max);
    draw_heatmap(
        &root,
        &values,
        &xs,
        &ys,
        peak,
        CELL / 2 + 1,
        0.004,
        to_x,
        to_y,
    )?;

    // Light grid lines.
    for tick in [-180i32, -90, 0, 90, 180] {
        let x = to_x(tick as f64);
        root.draw(&PathElement::new(
            vec![(x, MARGIN_TOP), (x, MARGIN_TOP + plot_h as i32)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        let y = to_y(tick as f64);
        root.draw(&PathElement::new(
            vec![(MARGIN_LEFT, y), (BAR_LEFT, y)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    }

    // Axes.
    root.draw(&PathElement::new(
        vec![
            (MARGIN_LEFT, MARGIN_TOP),
            (MARGIN_LEFT, MARGIN_TOP + plot_h as i32),
        ],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    root.draw(&PathElement::new(
        vec![
            (MARGIN_LEFT, MARGIN_TOP + plot_h as i32),
            (BAR_LEFT, MARGIN_TOP + plot_h as i32),
        ],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    // Emphasized raw observations.
    for &(x, y) in data {
        let center = (to_x(x), to_y(y));
        root.draw(&Circle::new(center, 4, WHITE.filled()))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        root.draw(&Circle::new(center, 3, ACCENT.mix(0.35).filled()))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
    }

    // Title and axis labels.
    draw_title(
        &root,
        &format!("Linkage torsion density — {site}"),
        (MARGIN_LEFT + BAR_LEFT) / 2,
        26,
        15.0,
        TEXT_GRAY,
    )?;
    draw_title(
        &root,
        "φ (degrees)",
        (MARGIN_LEFT + BAR_LEFT) / 2,
        HEIGHT as i32 - 26,
        11.0,
        TEXT_GRAY,
    )?;
    root.draw(&Text::new(
        "ψ (degrees)",
        (16, MARGIN_TOP + plot_h as i32 / 2),
        ("sans-serif", 11)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    // Tick labels.
    for tick in [-180i32, -90, 0, 90, 180] {
        let label = format!("{tick}");
        draw_text(
            &root,
            &label,
            to_x(tick as f64) - text_width(&label, 9.0) / 2,
            MARGIN_TOP + plot_h as i32 + 8,
            9.0,
            TEXT_GRAY,
        )?;
        draw_text(
            &root,
            &label,
            MARGIN_LEFT - text_width(&label, 9.0) - 8,
            to_y(tick as f64) - 5,
            9.0,
            TEXT_GRAY,
        )?;
    }

    // Relative-density colorbar.
    let bar_top = MARGIN_TOP;
    let bar_bottom = MARGIN_TOP + plot_h as i32;
    for slice in 0..24 {
        let frac = slice as f64 / 23.0;
        let y = bar_bottom - ((slice as f64 + 0.5) / 24.0 * (bar_bottom - bar_top) as f64) as i32;
        root.draw(&Rectangle::new(
            [(BAR_LEFT, y), (BAR_LEFT + BAR_WIDTH, y + 2)],
            density_color(frac).filled(),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    }
    root.draw(&PathElement::new(
        vec![
            (BAR_LEFT, bar_top),
            (BAR_LEFT + BAR_WIDTH, bar_top),
            (BAR_LEFT + BAR_WIDTH, bar_bottom),
            (BAR_LEFT, bar_bottom),
            (BAR_LEFT, bar_top),
        ],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    for (frac, label) in [(1.0, "peak"), (0.5, "half"), (0.0, "zero")] {
        draw_text(
            &root,
            label,
            BAR_LEFT + BAR_WIDTH + 8,
            bar_bottom - (frac * (bar_bottom - bar_top) as f64) as i32 - 5,
            9.0,
            TEXT_GRAY,
        )?;
    }
    root.draw(&Text::new(
        "relative density",
        (BAR_LEFT - 8, (bar_top + bar_bottom) / 2),
        ("sans-serif", 10)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

/// Multi-site 2D KDE plot: one panel per attachment site with per-site
/// bandwidths, a shared relative-density colorbar, and per-site point colors.
#[allow(dead_code)]
fn multi_site_kde_plot(path: PathBuf, site_data: &BTreeMap<String, Vec<(f64, f64)>>) -> Result<()> {
    let sites: Vec<_> = site_data.keys().cloned().collect();
    let n = sites.len();
    let cols = n.min(2);
    let rows = n.div_ceil(cols);

    const PANEL: i32 = 300;
    const MARGIN_LEFT: i32 = 66;
    const MARGIN_TOP: i32 = 46;
    const MARGIN_BOTTOM: i32 = 56;
    const INNER_LEFT: i32 = 6;
    const INNER_TOP: i32 = 34;
    const INNER_RIGHT: i32 = 6;
    const INNER_BOTTOM: i32 = 8;
    const BAR_WIDTH: i32 = 14;
    const CELL: i32 = 5;

    let inner_w = PANEL - INNER_LEFT - INNER_RIGHT;
    let inner_h = PANEL - INNER_TOP - INNER_BOTTOM;
    let bar_x = MARGIN_LEFT + cols as i32 * PANEL + 26;
    let width = (bar_x + BAR_WIDTH + 74) as u32;
    let height = (MARGIN_TOP + rows as i32 * PANEL + MARGIN_BOTTOM) as u32;
    let bar_top = MARGIN_TOP;
    let bar_bottom = MARGIN_TOP + rows as i32 * PANEL;

    let root = SVGBackend::new(&path, (width, height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    for (index, site) in sites.iter().enumerate() {
        let data = &site_data[site];
        if data.is_empty() {
            continue;
        }
        let (phi, psi): (Vec<f64>, Vec<f64>) = data.iter().copied().unzip();
        let bandwidth_phi = silverman_bandwidth(&phi);
        let bandwidth_psi = silverman_bandwidth(&psi);

        let panel_x = MARGIN_LEFT + (index % cols) as i32 * PANEL;
        let panel_y = MARGIN_TOP + (index / cols) as i32 * PANEL;
        let to_x = |x: f64| panel_x + INNER_LEFT + ((x + 180.0) / 360.0 * inner_w as f64) as i32;
        let to_y = |y: f64| {
            panel_y + INNER_TOP + (inner_h as f64 - (y + 180.0) / 360.0 * inner_h as f64) as i32
        };

        draw_title(
            &root,
            site,
            panel_x + PANEL / 2,
            panel_y + 14,
            12.0,
            TEXT_GRAY,
        )?;

        let (values, xs, ys) = density_heatmap(
            &phi,
            &psi,
            bandwidth_phi,
            bandwidth_psi,
            (inner_w / CELL) as usize,
            (inner_h / CELL) as usize,
        );
        let peak = values.iter().copied().fold(0.0f64, f64::max);
        draw_heatmap(
            &root,
            &values,
            &xs,
            &ys,
            peak,
            CELL / 2 + 1,
            0.004,
            to_x,
            to_y,
        )?;

        // Light grid lines.
        for tick in [-180i32, -90, 0, 90, 180] {
            let x = to_x(tick as f64);
            root.draw(&PathElement::new(
                vec![(x, panel_y + INNER_TOP), (x, panel_y + INNER_TOP + inner_h)],
                GRID_GRAY.stroke_width(1),
            ))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
            let y = to_y(tick as f64);
            root.draw(&PathElement::new(
                vec![
                    (panel_x + INNER_LEFT, y),
                    (panel_x + INNER_LEFT + inner_w, y),
                ],
                GRID_GRAY.stroke_width(1),
            ))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        }

        // Panel border.
        root.draw(&PathElement::new(
            vec![
                (panel_x + INNER_LEFT, panel_y + INNER_TOP),
                (panel_x + INNER_LEFT + inner_w, panel_y + INNER_TOP),
                (
                    panel_x + INNER_LEFT + inner_w,
                    panel_y + INNER_TOP + inner_h,
                ),
                (panel_x + INNER_LEFT, panel_y + INNER_TOP + inner_h),
                (panel_x + INNER_LEFT, panel_y + INNER_TOP),
            ],
            AXIS_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;

        // Per-site observations in the series color.
        let point_color = series_color(index);
        for &(x, y) in data {
            let center = (to_x(x), to_y(y));
            root.draw(&Circle::new(center, 4, WHITE.filled()))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
            root.draw(&Circle::new(center, 3, point_color.mix(0.35).filled()))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
        }

        // Ticks on the left column and bottom row only.
        if index % cols == 0 {
            for tick in [-180i32, -90, 0, 90, 180] {
                let label = format!("{tick}");
                draw_text(
                    &root,
                    &label,
                    panel_x + INNER_LEFT - text_width(&label, 8.0) - 5,
                    to_y(tick as f64) - 4,
                    8.0,
                    TEXT_GRAY,
                )?;
            }
        }
        if index / cols == rows - 1 {
            for tick in [-180i32, -90, 0, 90, 180] {
                let label = format!("{tick}");
                draw_text(
                    &root,
                    &label,
                    to_x(tick as f64) - text_width(&label, 8.0) / 2,
                    panel_y + INNER_TOP + inner_h + 7,
                    8.0,
                    TEXT_GRAY,
                )?;
            }
        }
    }

    // Shared axis captions.
    draw_title(
        &root,
        "φ (degrees)",
        (MARGIN_LEFT + cols as i32 * PANEL) / 2,
        height as i32 - 30,
        11.0,
        TEXT_GRAY,
    )?;
    root.draw(&Text::new(
        "ψ (degrees)",
        (14, (MARGIN_TOP + rows as i32 * PANEL) / 2),
        ("sans-serif", 11)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    // Shared relative-density colorbar.
    for slice in 0..24 {
        let frac = slice as f64 / 23.0;
        let y = bar_bottom - ((slice as f64 + 0.5) / 24.0 * (bar_bottom - bar_top) as f64) as i32;
        root.draw(&Rectangle::new(
            [(bar_x, y), (bar_x + BAR_WIDTH, y + 2)],
            density_color(frac).filled(),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    }
    root.draw(&PathElement::new(
        vec![
            (bar_x, bar_top),
            (bar_x + BAR_WIDTH, bar_top),
            (bar_x + BAR_WIDTH, bar_bottom),
            (bar_x, bar_bottom),
            (bar_x, bar_top),
        ],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    for (frac, label) in [(1.0, "peak"), (0.5, "half"), (0.0, "zero")] {
        draw_text(
            &root,
            label,
            bar_x + BAR_WIDTH + 8,
            bar_bottom - (frac * (bar_bottom - bar_top) as f64) as i32 - 5,
            9.0,
            TEXT_GRAY,
        )?;
    }
    root.draw(&Text::new(
        "relative density",
        (bar_x - 8, (bar_top + bar_bottom) / 2),
        ("sans-serif", 10)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

/// Clean periodic scatter plot for protein-glycan attachment torsions in a
/// single structure. Each attachment site gets a restrained color and a
/// legend entry rather than a label placed on top of a data point.
fn protein_linkage_point_plot(path: PathBuf, torsions: &[ProteinLinkageTorsion]) -> Result<()> {
    if torsions.is_empty() {
        return Ok(());
    }

    const WIDTH: i32 = 720;
    const HEIGHT: i32 = 500;
    const PLOT_LEFT: i32 = 66;
    const PLOT_TOP: i32 = 46;
    const PLOT_RIGHT: i32 = 170;
    const PLOT_BOTTOM: i32 = 58;
    let plot_width = WIDTH - PLOT_LEFT - PLOT_RIGHT;
    let plot_height = HEIGHT - PLOT_TOP - PLOT_BOTTOM;

    let root = SVGBackend::new(&path, (WIDTH as u32, HEIGHT as u32)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    let to_x = |phi: f64| {
        PLOT_LEFT + ((phi.clamp(-180.0, 180.0) + 180.0) / 360.0 * plot_width as f64).round() as i32
    };
    let to_y = |psi: f64| {
        PLOT_TOP
            + (plot_height as f64
                - ((psi.clamp(-180.0, 180.0) + 180.0) / 360.0 * plot_height as f64))
                .round() as i32
    };

    let mut site_groups: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for t in torsions {
        site_groups
            .entry(t.site.clone())
            .or_default()
            .push((t.phi_degrees, t.psi_degrees));
    }

    for tick in [-180i32, -90, 0, 90, 180] {
        let x = to_x(tick as f64);
        let y = to_y(tick as f64);
        root.draw(&PathElement::new(
            vec![(x, PLOT_TOP), (x, PLOT_TOP + plot_height)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        root.draw(&PathElement::new(
            vec![(PLOT_LEFT, y), (PLOT_LEFT + plot_width, y)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        let label = tick.to_string();
        draw_text(
            &root,
            &label,
            x - text_width(&label, 8.0) / 2,
            PLOT_TOP + plot_height + 7,
            8.0,
            TEXT_GRAY,
        )?;
        draw_text(
            &root,
            &label,
            PLOT_LEFT - text_width(&label, 8.0) - 8,
            y - 5,
            8.0,
            TEXT_GRAY,
        )?;
    }
    root.draw(&PathElement::new(
        vec![
            (PLOT_LEFT, PLOT_TOP + plot_height),
            (PLOT_LEFT + plot_width, PLOT_TOP + plot_height),
        ],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    root.draw(&PathElement::new(
        vec![(PLOT_LEFT, PLOT_TOP), (PLOT_LEFT, PLOT_TOP + plot_height)],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    draw_title(
        &root,
        "Protein–glycan attachment torsions",
        PLOT_LEFT + plot_width / 2,
        13,
        13.0,
        TEXT_GRAY,
    )?;
    draw_text(
        &root,
        "φ (degrees)",
        PLOT_LEFT + plot_width / 2 - text_width("φ (degrees)", 9.0) / 2,
        HEIGHT - 17,
        9.0,
        TEXT_GRAY,
    )?;
    root.draw(&Text::new(
        "ψ (degrees)",
        (18, PLOT_TOP + plot_height / 2),
        ("sans-serif", 9)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    for (idx, (site, points)) in site_groups.iter().enumerate() {
        let color = series_color(idx);
        for &(phi, psi) in points {
            let center = (to_x(phi), to_y(psi));
            root.draw(&Circle::new(center, 6, color.mix(0.2).filled()))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
            root.draw(&Circle::new(center, 4, WHITE.filled()))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
            root.draw(&Circle::new(center, 3, color.stroke_width(1)))
                .map_err(|error| ReportError::Plot(error.to_string()))?;
        }
        let legend_y = PLOT_TOP + idx as i32 * 20;
        let legend_x = PLOT_LEFT + plot_width + 20;
        root.draw(&Circle::new((legend_x + 4, legend_y), 4, color.filled()))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        draw_text(
            &root,
            &truncate_label(site, 20),
            legend_x + 14,
            legend_y - 5,
            9.0,
            TEXT_GRAY,
        )?;
    }

    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

/// Render the glycan's SNFG diagram with exactly the donor and acceptor from
/// this torsion record highlighted. Source residue provenance is required so
/// repeated structural motifs cannot select a different occurrence.
fn snfg_bond_svg(glycan: &GlycanSummary, torsion: &LinkageTorsion) -> Option<String> {
    let graph = glycan.snfg_graph.as_ref()?;
    let donor_position = torsion.donor_position?;
    let acceptor_position = torsion.acceptor_position?;
    let donor_residue = torsion.donor_residue.as_ref()?;
    let acceptor_residue = torsion.acceptor_residue.as_ref()?;
    let provenance = glycan.snfg_residues.as_ref()?;
    let node_for = |residue: &ResidueId| {
        let insertion_code = residue.insertion_code.map(|code| code.to_string());
        provenance.iter().find(|source| {
            source.chain == residue.chain
                && source.sequence_number == residue.number as isize
                && source.insertion_code.as_deref() == insertion_code.as_deref()
        })
    };
    let donor_node = node_for(donor_residue)?.node_index;
    let acceptor_node = node_for(acceptor_residue)?.node_index;
    let edge = graph.inner().edge_references().find(|edge| {
        edge.source().index() == acceptor_node
            && edge.target().index() == donor_node
            && edge.weight().parent_position.0 == acceptor_position
            && edge.weight().child_position.0 == donor_position
    })?;
    let options = RenderOptions {
        colour: true,
        show_labels: false,
        show_linkages: false,
        font_family: "Arial, Helvetica, sans-serif".into(),
        scale: 0.7,
        source_notation: None,
    };
    let mut selection = HighlightSelection::default();
    selection.node_indices.extend([donor_node, acceptor_node]);
    selection.edge_indices.insert(edge.id().index());
    render_svg_with_selection(graph, &selection, &options).ok()
}

/// Estimate a one-dimensional angular density on a periodic domain.
fn kde_1d_periodic(angle: f64, data: &[f64], bandwidth: f64) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    data.iter()
        .map(|&sample| gaussian_kernel(circular_delta(angle, sample) / bandwidth))
        .sum::<f64>()
        / (n * bandwidth)
}

/// Plot all available glycosidic angles on one shared angular axis.  The
/// translucent fills make overlapping φ/ψ/ω distributions readable without
/// creating a separate mini-axis for every angle.
fn linkage_distribution_plot(
    path: PathBuf,
    label: &str,
    phi: &[f64],
    psi: &[f64],
    omega: &[f64],
) -> Result<()> {
    const WIDTH: u32 = 700;
    const HEIGHT: u32 = 245;
    const MARGIN_LEFT: i32 = 58;
    const MARGIN_RIGHT: i32 = 20;
    const MARGIN_TOP: i32 = 48;
    const MARGIN_BOTTOM: i32 = 42;
    const RESOLUTION: usize = 181;

    let root = SVGBackend::new(&path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    let plot_left = MARGIN_LEFT;
    let plot_top = MARGIN_TOP;
    let plot_width = WIDTH as i32 - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = HEIGHT as i32 - MARGIN_TOP - MARGIN_BOTTOM;
    let baseline = plot_top + plot_height;
    let to_x = |angle: f64| {
        plot_left
            + ((angle.clamp(-180.0, 180.0) + 180.0) / 360.0 * plot_width as f64).round() as i32
    };

    draw_text(
        &root,
        &truncate_label(label, 52),
        plot_left,
        10,
        11.0,
        TEXT_GRAY,
    )?;

    let series = [
        ("φ", phi, ACCENT),
        ("ψ", psi, SERIES[1]),
        ("ω", omega, SERIES[2]),
    ];
    let mut curves = Vec::new();
    let mut peak = 0.0f64;
    for (name, data, color) in series {
        if data.is_empty() {
            continue;
        }
        let bandwidth = periodic_bandwidth(data);
        let curve = (0..RESOLUTION)
            .map(|index| {
                let angle = -180.0 + 360.0 * index as f64 / (RESOLUTION - 1) as f64;
                let density = kde_1d_periodic(angle, data, bandwidth);
                peak = peak.max(density);
                (angle, density)
            })
            .collect::<Vec<_>>();
        curves.push((name, data, color, curve));
    }

    let scale = (plot_height as f64 - 8.0) / peak.max(1.0e-12);
    for (name, data, color, curve) in &curves {
        let points = curve
            .iter()
            .map(|(angle, density)| (to_x(*angle), baseline - (*density * scale).round() as i32))
            .collect::<Vec<_>>();
        let mut fill = points.clone();
        fill.push((to_x(180.0), baseline));
        fill.push((to_x(-180.0), baseline));
        root.draw(&Polygon::new(fill, color.mix(0.22).filled()))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        root.draw(&PathElement::new(points, color.mix(0.86).stroke_width(2)))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        for &value in *data {
            root.draw(&Circle::new(
                (to_x(value), baseline - 3),
                2,
                color.mix(0.5).filled(),
            ))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        }
        let legend_x = WIDTH as i32 - 90;
        let legend_y = 10
            + curves
                .iter()
                .position(|entry| entry.0 == *name)
                .unwrap_or(0) as i32
                * 15;
        root.draw(&PathElement::new(
            vec![(legend_x, legend_y + 1), (legend_x + 14, legend_y + 1)],
            color.stroke_width(3),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        draw_text(&root, name, legend_x + 19, legend_y - 4, 10.0, *color)?;
    }

    for tick in [-180i32, -90, 0, 90, 180] {
        let x = to_x(tick as f64);
        root.draw(&PathElement::new(
            vec![(x, plot_top), (x, baseline)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        let tick_label = tick.to_string();
        draw_text(
            &root,
            &tick_label,
            x - text_width(&tick_label, 8.0) / 2,
            baseline + 7,
            8.0,
            TEXT_GRAY,
        )?;
    }
    root.draw(&PathElement::new(
        vec![(plot_left, baseline), (plot_left + plot_width, baseline)],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    draw_text(
        &root,
        "angle (degrees)",
        plot_left + plot_width / 2 - text_width("angle (degrees)", 9.0) / 2,
        HEIGHT as i32 - 14,
        9.0,
        TEXT_GRAY,
    )?;
    draw_text(
        &root,
        "density",
        7,
        plot_top + plot_height / 2,
        8.0,
        TEXT_GRAY,
    )?;

    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

/// Ordered (glycan index, linkage) groups across all frames, used to keep the
/// per-linkage report rows and their assets in a stable order.
fn linkage_groups(
    torsions: &[LinkageTorsion],
) -> Vec<(usize, String, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut groups: BTreeMap<(usize, String), (Vec<f64>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for torsion in torsions {
        let entry = groups
            .entry((torsion.glycan_index, torsion.linkage.clone()))
            .or_default();
        if let Some(phi) = torsion.phi_degrees {
            entry.0.push(phi);
        }
        if let Some(psi) = torsion.psi_degrees {
            entry.1.push(psi);
        }
        if let Some(omega) = torsion.omega_degrees {
            entry.2.push(omega);
        }
    }
    groups
        .into_iter()
        .map(|((glycan_index, linkage), (phi, psi, omega))| {
            (glycan_index, linkage, phi, psi, omega)
        })
        .collect()
}

/// Write one highlighted SNFG diagram and one combined φ/ψ/ω distribution
/// plot per glycosidic bond observed in the ensemble. When a bond cannot be
/// matched for highlighting, the plain glycan diagram is written instead so
/// the report row always has an SNFG figure.
fn write_linkage_assets(
    glycans: &[GlycanSummary],
    torsions: &[LinkageTorsion],
    assets: &Path,
) -> Result<()> {
    for (index, (glycan_index, linkage, phi, psi, omega)) in
        linkage_groups(torsions).iter().enumerate()
    {
        let snfg_name = format!("snfg-bond-{}.svg", index + 1);
        let ridge_name = format!("glycosidic-{}.svg", index + 1);
        if let Some(glycan) = glycans.get(glycan_index.saturating_sub(1)) {
            let svg = torsions
                .iter()
                .find(|t| t.glycan_index == *glycan_index && t.linkage == *linkage)
                .and_then(|torsion| snfg_bond_svg(glycan, torsion))
                .or_else(|| glycan.snfg_svg.clone());
            if let Some(svg) = svg {
                fs::write(assets.join(&snfg_name), svg)?;
            }
        }
        linkage_distribution_plot(assets.join(&ridge_name), linkage, phi, psi, omega)?;
    }
    Ok(())
}

/// Line plot for energy/score histories: accent area fill under the curve,
/// markers on every step, and annotated min/max extrema. The human-readable
/// description is rendered as the figure caption in the Typst report.
fn energy_plot_with_annotations(
    path: PathBuf,
    title: &str,
    values: &[(String, f64)],
    unit: &str,
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    let min_value = values.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let max_value = values
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_value - min_value).abs().max(1.0e-9);
    let pad = range * 0.12;

    const WIDTH: u32 = 820;
    const HEIGHT: u32 = 420;

    let root = SVGBackend::new(&path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 15).into_font().color(&TEXT_GRAY))
        .margin_top(46)
        .margin_right(30)
        .margin_bottom(26)
        .x_label_area_size(36)
        .y_label_area_size(58)
        .build_cartesian_2d(0..values.len(), (min_value - pad)..(max_value + pad))
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .light_line_style(GRID_GRAY.stroke_width(1))
        .axis_style(AXIS_GRAY.stroke_width(1))
        .x_labels(values.len().min(10))
        .x_label_formatter(&|x| format!("{x}"))
        .y_label_formatter(&|y| format!("{y:.1}"))
        .draw()
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    let points: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .map(|(i, (_, v))| (i, *v))
        .collect();

    // Soft area fill under the curve.
    let mut area = points.clone();
    area.push((values.len() - 1, min_value - pad));
    area.push((0, min_value - pad));
    chart
        .draw_series(std::iter::once(Polygon::new(
            area,
            ACCENT.mix(0.84).filled(),
        )))
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    chart
        .draw_series(LineSeries::new(points.clone(), ACCENT.stroke_width(2)))
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    // White-backed markers on each step.
    chart
        .draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, WHITE.filled())),
        )
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    chart
        .draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 2, ACCENT.filled())),
        )
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    // Annotated extrema.
    let (min_index, min_val) = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
        .map(|(i, (_, v))| (i, *v))
        .unwrap();
    let (max_index, max_val) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
        .map(|(i, (_, v))| (i, *v))
        .unwrap();

    for (index, value, color) in [
        (min_index, min_val, GOOD_COLOR),
        (max_index, max_val, BAD_COLOR),
    ] {
        chart
            .draw_series(std::iter::once(Circle::new(
                (index, value),
                4,
                WHITE.filled(),
            )))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
        chart
            .draw_series(std::iter::once(Circle::new(
                (index, value),
                3,
                color.filled(),
            )))
            .map_err(|error| ReportError::Plot(error.to_string()))?;
    }

    let chart_floor = min_value - pad;
    let chart_ceiling = max_value + pad;
    let min_label = format!("min {min_val:.1} {unit}");
    let min_label_x = (min_index + 1).min(values.len() - 1);
    let min_label_y =
        (min_val + pad * 0.4).clamp(chart_floor + pad * 0.2, chart_ceiling - pad * 0.2);
    let max_label = format!("max {max_val:.1} {unit}");
    let max_label_x = (max_index + 1).min(values.len() - 1);
    let max_label_y =
        (max_val - pad * 0.4).clamp(chart_floor + pad * 0.2, chart_ceiling - pad * 0.2);

    chart
        .draw_series(std::iter::once(Text::new(
            min_label,
            (min_label_x, min_label_y),
            ("sans-serif", 10).into_font().color(&GOOD_COLOR),
        )))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    chart
        .draw_series(std::iter::once(Text::new(
            max_label,
            (max_label_x, max_label_y),
            ("sans-serif", 10).into_font().color(&BAD_COLOR),
        )))
        .map_err(|error| ReportError::Plot(error.to_string()))?;

    // Rotated unit label.
    root.draw(&Text::new(
        format!("Energy ({unit})"),
        (14, HEIGHT as i32 / 2),
        ("sans-serif", 11)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;

    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

fn write_plot_assets(report: &WorkflowReport, assets: &Path) -> Result<()> {
    if let Some(saxs) = &report.analysis.saxs {
        if let Some(svg) = &saxs.diagnostic_svg {
            fs::write(assets.join("saxs-diagnostic.svg"), svg)?;
        }
        for (index, candidate) in saxs.candidate_visuals.iter().enumerate() {
            if let Some(svg) = &candidate.snfg_svg {
                fs::write(
                    assets.join(format!("saxs-candidate-{}.svg", index + 1)),
                    svg,
                )?;
            }
        }
        if !saxs.site_occupancy.is_empty() {
            write_saxs_occupancy_plot(saxs, &assets.join("saxs-occupancy.svg"))?;
        } else if !saxs.candidate_visuals.is_empty() {
            write_saxs_candidate_plot(saxs, &assets.join("saxs-candidates.svg"))?;
        }
    }
    // Steric scores are conveyed by the per-site table; a bar chart of
    // near-identical values would add no information.

    // GA history and energy plots
    if let Some(search) = &report.search {
        let values = search
            .history
            .iter()
            .map(|entry| (entry.generation.to_string(), entry.best_score))
            .collect::<Vec<_>>();
        energy_plot_with_annotations(
            assets.join("ga-history.svg"),
            "Genetic Algorithm: Best Score per Generation",
            &values,
            "score",
        )?;
        let energies = search
            .history
            .iter()
            .filter_map(|entry| {
                entry
                    .best_energy_kcal_per_mol
                    .map(|value| (entry.generation.to_string(), value))
            })
            .collect::<Vec<_>>();
        if !energies.is_empty() {
            energy_plot_with_annotations(
                assets.join("ga-energy-history.svg"),
                "GA: Selected Energy (Amber/GLYCAM)",
                &energies,
                "kcal/mol",
            )?;
        }
    }
    if let Some(relaxation) = &report.relaxation {
        energy_plot_with_annotations(
            assets.join("energy-history.svg"),
            "Staged Energy Minimization",
            &relaxation
                .energy_history
                .iter()
                .enumerate()
                .map(|(i, value)| (i.to_string(), *value))
                .collect::<Vec<_>>(),
            "kcal/mol",
        )?;
    }

    // Protein-glycan linkage plots
    if !report.analysis.protein_linkage_torsions.is_empty() {
        // Check if we have ensemble data (multiple frames)
        let has_ensemble = report
            .analysis
            .protein_linkage_torsions
            .iter()
            .any(|t| t.frame.is_some());

        if has_ensemble {
            // Use 2D KDE plot for ensemble data
            protein_linkage_kde_plot(
                assets.join("protein-linkage-kde.svg"),
                &report.analysis.protein_linkage_torsions,
            )?;
        } else {
            // Use simple point plot for single structure
            protein_linkage_point_plot(
                assets.join("protein-linkage-points.svg"),
                &report.analysis.protein_linkage_torsions,
            )?;
        }
    }

    // Per-linkage glycosidic torsion plots with highlighted SNFG bonds.
    if !report.analysis.glycosidic_torsions.is_empty() {
        write_linkage_assets(
            &report.analysis.glycans,
            &report.analysis.glycosidic_torsions,
            assets,
        )?;
    }
    // Keep a compact representative of every persisted GlycoShape reference
    // in the native bundle. The browser can render the same grids
    // interactively, while PDF/ZIP consumers still receive a standalone,
    // reproducible contour figure when opened offline.
    for (index, reference) in report.analysis.torsion_references.iter().enumerate() {
        if reference.phi_psi.as_ref().is_some_and(|contour| {
            contour.bins >= 2 && contour.grid.len() >= contour.bins * contour.bins
        }) {
            torsion_reference_plot(
                assets.join(format!("torsion-reference-{}.svg", index + 1)),
                reference,
            )?;
        }
    }
    Ok(())
}

/// Render one periodic φ/ψ reference grid with Level-1 population medoids.
/// Finer Level-2/3 populations are shown as small outlined markers carrying
/// their parent colour, matching the interactive report's palette policy.
fn torsion_reference_plot(path: PathBuf, reference: &TorsionReference) -> Result<()> {
    const WIDTH: u32 = 760;
    const HEIGHT: u32 = 500;
    const PLOT_LEFT: i32 = 74;
    const PLOT_TOP: i32 = 62;
    const PLOT_RIGHT: i32 = 32;
    const PLOT_BOTTOM: i32 = 62;
    let contour = reference.phi_psi.as_ref().expect("checked by caller");
    let bins = contour.bins;
    let values = &contour.grid[..bins * bins];
    let root = SVGBackend::new(&path, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    draw_title(
        &root,
        &format!(
            "{} · GlycoShape reference",
            truncate_label(&reference.canonical_linkage, 54)
        ),
        WIDTH as i32 / 2,
        24,
        15.0,
        TEXT_GRAY,
    )?;
    let plot_w = WIDTH as i32 - PLOT_LEFT - PLOT_RIGHT;
    let plot_h = HEIGHT as i32 - PLOT_TOP - PLOT_BOTTOM;
    draw_contour_bands(&root, values, bins, PLOT_LEFT, PLOT_TOP, plot_w, plot_h)?;
    root.draw(&Rectangle::new(
        [
            (PLOT_LEFT, PLOT_TOP),
            (PLOT_LEFT + plot_w, PLOT_TOP + plot_h),
        ],
        AXIS_GRAY.stroke_width(1),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    let to_x = |angle: f64| {
        PLOT_LEFT + ((angle.clamp(-180.0, 180.0) + 180.0) / 360.0 * plot_w as f64).round() as i32
    };
    let to_y = |angle: f64| {
        PLOT_TOP
            + (plot_h as f64 - (angle.clamp(-180.0, 180.0) + 180.0) / 360.0 * plot_h as f64).round()
                as i32
    };
    for tick in [-180i32, -90, 0, 90, 180] {
        let x = to_x(tick as f64);
        let y = to_y(tick as f64);
        root.draw(&PathElement::new(
            vec![(x, PLOT_TOP), (x, PLOT_TOP + plot_h)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        root.draw(&PathElement::new(
            vec![(PLOT_LEFT, y), (PLOT_LEFT + plot_w, y)],
            GRID_GRAY.stroke_width(1),
        ))
        .map_err(|error| ReportError::Plot(error.to_string()))?;
        let label = tick.to_string();
        draw_text(
            &root,
            &label,
            x - text_width(&label, 8.0) / 2,
            PLOT_TOP + plot_h + 18,
            8.0,
            TEXT_GRAY,
        )?;
        draw_text(
            &root,
            &label,
            PLOT_LEFT - text_width(&label, 8.0) - 8,
            y + 3,
            8.0,
            TEXT_GRAY,
        )?;
    }
    draw_text(
        &root,
        "φ (degrees)",
        PLOT_LEFT + plot_w / 2 - text_width("φ (degrees)", 10.0) / 2,
        HEIGHT as i32 - 16,
        10.0,
        TEXT_GRAY,
    )?;
    root.draw(&Text::new(
        "ψ (degrees)",
        (17, PLOT_TOP + plot_h / 2),
        ("sans-serif", 10)
            .into_font()
            .color(&TEXT_GRAY)
            .transform(FontTransform::Rotate270),
    ))
    .map_err(|error| ReportError::Plot(error.to_string()))?;
    let populations = &reference.populations;
    let by_index = populations
        .iter()
        .map(|population| (population.index, population))
        .collect::<BTreeMap<_, _>>();
    let parent_color = |population: &TorsionPopulation| -> String {
        let mut current = population;
        let mut seen = BTreeSet::new();
        while current.level > 1 {
            if !seen.insert(current.index) {
                break;
            }
            let Some(parent) = current
                .parent_index
                .and_then(|index| by_index.get(&index).copied())
            else {
                break;
            };
            current = parent;
        }
        current.color.clone().unwrap_or_else(|| "#0f6b6b".into())
    };
    let parse_color = |value: &str| {
        let value = value.trim_start_matches('#');
        if value.len() == 6 {
            let parse = |range: std::ops::Range<usize>| u8::from_str_radix(&value[range], 16).ok();
            if let (Some(r), Some(g), Some(b)) = (parse(0..2), parse(2..4), parse(4..6)) {
                return RGBColor(r, g, b);
            }
        }
        ACCENT
    };
    for population in populations {
        let (Some(phi), Some(psi)) = (population.medoid_phi, population.medoid_psi) else {
            continue;
        };
        let color = parse_color(&parent_color(population));
        let center = (to_x(phi), to_y(psi));
        if population.level <= 1 {
            root.draw(&Circle::new(center, 6, color.filled()))
        } else {
            root.draw(&Circle::new(center, 4, WHITE.filled()))
                .and_then(|_| root.draw(&Circle::new(center, 4, color.stroke_width(1))))
        }
        .map_err(|error| ReportError::Plot(error.to_string()))?;
    }
    root.present()
        .map_err(|error| ReportError::Plot(error.to_string()))
}

fn write_saxs_occupancy_plot(saxs: &SaxsAnalysis, path: &Path) -> Result<()> {
    fs::write(path, saxs_candidate_matrix_svg(saxs, true))?;
    Ok(())
}

fn write_saxs_candidate_plot(saxs: &SaxsAnalysis, path: &Path) -> Result<()> {
    fs::write(path, saxs_candidate_matrix_svg(saxs, false))?;
    Ok(())
}

/// Render candidate support as a compact report-style table. SNFGs are
/// embedded as data URLs so the standalone SVG remains self-contained.
fn saxs_candidate_matrix_svg(saxs: &SaxsAnalysis, with_sites: bool) -> String {
    let sites = if with_sites {
        saxs.site_occupancy.keys().cloned().collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut candidates = saxs
        .candidate_visuals
        .iter()
        .map(|candidate| candidate.candidate.clone())
        .collect::<Vec<_>>();
    for values in saxs.glycoform_distribution.values() {
        for candidate in values.keys() {
            if !candidates.iter().any(|value| value == candidate) {
                candidates.push(candidate.clone());
            }
        }
    }
    candidates.sort();
    candidates.dedup();
    if candidates.is_empty() {
        candidates.push("none".into());
    }

    const LEFT: u32 = 220;
    const COLUMN_WIDTH: u32 = 220;
    const HEADER_HEIGHT: u32 = 168;
    const ROW_HEIGHT: u32 = 72;
    const RIGHT: u32 = 36;
    const BOTTOM: u32 = 28;
    let rows = sites.len() as u32;
    let width = LEFT + COLUMN_WIDTH * candidates.len() as u32 + RIGHT;
    let height = HEADER_HEIGHT + ROW_HEIGHT * rows.max(1) + BOTTOM;
    let title = if with_sites {
        "SAXS-supported glycoform map"
    } else {
        "Glycan candidates (SNFG)"
    };
    let subtitle = if with_sites {
        "χ²-derived relative support; robust multi-feature rank is reported separately."
    } else {
        "Candidate structures used in the SAXS fit."
    };
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    );
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
    svg.push_str(&format!(
        "<text x=\"28\" y=\"34\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"24\" font-weight=\"600\" fill=\"#172033\">{}</text>",
        svg_escape(title)
    ));
    svg.push_str(&format!(
        "<text x=\"28\" y=\"62\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"14\" fill=\"#5b6472\">{}</text>",
        svg_escape(subtitle)
    ));

    for (column, candidate) in candidates.iter().enumerate() {
        let x = LEFT + column as u32 * COLUMN_WIDTH;
        let center = x + COLUMN_WIDTH / 2;
        svg.push_str(&format!(
            "<line x1=\"{x}\" y1=\"78\" x2=\"{x}\" y2=\"{}\" stroke=\"#D8E5E5\" stroke-width=\"1\"/>",
            height - BOTTOM
        ));
        svg.push_str(&format!(
            "<text x=\"{center}\" y=\"94\" text-anchor=\"middle\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"15\" font-weight=\"600\" fill=\"#0F6B6B\">{}</text>",
            svg_escape(candidate)
        ));
        append_saxs_candidate_header(&mut svg, saxs, candidate, x + 20, 106, COLUMN_WIDTH - 40);
    }

    if with_sites {
        for (row, site) in sites.iter().enumerate() {
            let y = HEADER_HEIGHT + row as u32 * ROW_HEIGHT;
            let occupancy = saxs.site_occupancy.get(site).copied().unwrap_or(0.0);
            svg.push_str(&format!(
                "<rect x=\"0\" y=\"{y}\" width=\"{}\" height=\"{ROW_HEIGHT}\" fill=\"{}\"/><line x1=\"0\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"#D8E5E5\" stroke-width=\"1\"/><text x=\"28\" y=\"{}\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"15\" font-weight=\"600\" fill=\"#172033\">{}</text><text x=\"28\" y=\"{}\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"12\" fill=\"#667477\">occupancy {:.1}%</text>",
                width,
                if row % 2 == 0 { "#F5FAFA" } else { "#FFFFFF" },
                width,
                y + 31,
                svg_escape(site),
                y + 50,
                100.0 * occupancy.clamp(0.0, 1.0)
            ));
            for (column, candidate) in candidates.iter().enumerate() {
                let support = saxs
                    .glycoform_distribution
                    .get(site)
                    .and_then(|values| values.get(candidate))
                    .copied()
                    .unwrap_or(0.0);
                let center = LEFT + column as u32 * COLUMN_WIDTH + COLUMN_WIDTH / 2;
                svg.push_str(&format!(
                    "<text x=\"{center}\" y=\"{}\" text-anchor=\"middle\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"14\" fill=\"#172033\">{:.1}%</text>",
                    y + 31,
                    100.0 * support.clamp(0.0, 1.0)
                ));
            }
        }
    }
    svg.push_str("</svg>");
    svg
}

fn append_saxs_candidate_header(
    svg: &mut String,
    saxs: &SaxsAnalysis,
    candidate: &str,
    x: u32,
    y: u32,
    width: u32,
) {
    let visual = saxs
        .candidate_visuals
        .iter()
        .find(|visual| visual.candidate == candidate && visual.snfg_svg.is_some());
    if let Some(visual) = visual {
        if let Some(snfg_svg) = &visual.snfg_svg {
            let encoded = BASE64.encode(snfg_svg.as_bytes());
            svg.push_str(&format!(
                "<image href=\"data:image/svg+xml;base64,{encoded}\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" preserveAspectRatio=\"xMidYMid meet\"/>",
                x,
                y,
                width,
                48
            ));
        }
    } else if is_none_candidate_label(candidate) {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"12\" fill=\"#667477\">no glycan</text>",
            x + width / 2,
            y + 28
        ));
    } else {
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"DejaVu Sans, sans-serif\" font-size=\"12\" fill=\"#667477\">SNFG unavailable</text>",
            x + width / 2,
            y + 28
        ));
    }
}

fn is_none_candidate_label(candidate: &str) -> bool {
    matches!(
        candidate.trim().to_ascii_lowercase().as_str(),
        "none" | "no_glycan" | "noglycan" | "absent"
    )
}

fn svg_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[derive(Debug, Clone)]
struct StericSiteSummary {
    site: String,
    frames: usize,
    median: f64,
    minimum: f64,
    maximum: f64,
    clash_free: usize,
}

fn steric_site_summaries(observations: &[StericObservation]) -> Vec<StericSiteSummary> {
    let mut grouped: BTreeMap<String, Vec<&StericObservation>> = BTreeMap::new();
    for observation in observations {
        grouped
            .entry(observation.site.clone())
            .or_default()
            .push(observation);
    }
    grouped
        .into_iter()
        .map(|(site, values)| {
            let mut scores = values.iter().map(|value| value.score).collect::<Vec<_>>();
            scores.sort_by(|a, b| a.total_cmp(b));
            let median = if scores.len() % 2 == 0 {
                let upper = scores.len() / 2;
                (scores[upper - 1] + scores[upper]) / 2.0
            } else {
                scores[scores.len() / 2]
            };
            StericSiteSummary {
                site,
                frames: scores.len(),
                median,
                minimum: scores[0],
                maximum: *scores.last().unwrap_or(&scores[0]),
                clash_free: values.iter().filter(|value| value.clash_free).count(),
            }
        })
        .collect()
}

fn amino_acid_name(code: &str) -> Option<&'static str> {
    match code.to_ascii_uppercase().as_str() {
        "ALA" => Some("alanine"),
        "ARG" => Some("arginine"),
        "ASN" => Some("asparagine"),
        "ASP" => Some("aspartate"),
        "CYS" => Some("cysteine"),
        "GLN" => Some("glutamine"),
        "GLU" => Some("glutamate"),
        "GLY" => Some("glycine"),
        "HIS" => Some("histidine"),
        "ILE" => Some("isoleucine"),
        "LEU" => Some("leucine"),
        "LYS" => Some("lysine"),
        "MET" => Some("methionine"),
        "PHE" => Some("phenylalanine"),
        "PRO" => Some("proline"),
        "SER" => Some("serine"),
        "THR" => Some("threonine"),
        "TRP" => Some("tryptophan"),
        "TYR" => Some("tyrosine"),
        "VAL" => Some("valine"),
        _ => None,
    }
}

fn attachment_description(site: Option<&str>) -> String {
    let Some(site) = site else {
        return "Attachment: unresolved protein residue".into();
    };
    let parts = site.split('/').collect::<Vec<_>>();
    if parts.len() >= 3 {
        let chain = parts[0];
        let residue = parts[2];
        let amino_acid = parts[1].to_ascii_uppercase();
        if let Some(full_name) = amino_acid_name(&amino_acid) {
            return format!(
                "Attached to chain {chain}, residue {residue}: {amino_acid} ({full_name})"
            );
        }
        return format!("Attached to chain {chain}, residue {residue}: {amino_acid}");
    }
    format!("Attachment: {site}")
}

fn report_site_label(site: &str, glycans: &[GlycanSummary]) -> String {
    let Some((chain, number)) = site.split_once(':') else {
        return site.to_string();
    };
    glycans
        .iter()
        .filter_map(|glycan| glycan.attachment_site.as_deref())
        .find_map(|attachment| {
            let parts = attachment.split('/').collect::<Vec<_>>();
            (parts.len() >= 3 && parts[0] == chain && parts[2] == number)
                .then(|| attachment_description(Some(attachment)))
        })
        .unwrap_or_else(|| format!("chain {chain}, residue {number}"))
}

fn typst_colored_sequon(context: &str) -> String {
    let characters = context.chars().collect::<Vec<_>>();
    let asparagine_index = characters.len().saturating_sub(3);
    let mut result = String::from("[");
    for (index, character) in characters.iter().enumerate() {
        if index == asparagine_index && *character == 'N' {
            result.push_str("#text(fill: rgb(\"#2166F3\"), weight: \"bold\")[N]");
        } else {
            result.push_str(&typst_escape(&character.to_string()));
        }
    }
    result.push(']');
    result
}

fn append_saxs_candidate_report(source: &mut String, saxs: &SaxsAnalysis) {
    if saxs.candidate_visuals.is_empty() {
        if let Some(asset) = &saxs.occupancy_asset {
            source.push_str(&format!(
                "#figure(image(\"{}\", width: 92%), caption: [SAXS-supported glycoform map])\n",
                typst_escape(asset)
            ));
        } else if let Some(asset) = &saxs.candidate_asset {
            source.push_str(&format!(
                "#figure(image(\"{}\", width: 92%), caption: [SAXS glycan candidates])\n",
                typst_escape(asset)
            ));
        }
        return;
    }

    source.push_str("\n=== SAXS glycan candidates\n");
    source.push_str(
        "Candidate structures are shown in the same compact SNFG-and-description layout used for the ensemble glycan sections.\n",
    );
    source.push_str("#grid(columns: (34%, 66%), row-gutter: 8pt, column-gutter: 12pt,\n");
    for candidate in &saxs.candidate_visuals {
        if let Some(asset) = &candidate.snfg_asset {
            source.push_str(&format!(
                "  image(\"{}\", width: 100%),\n",
                typst_escape(asset)
            ));
        } else {
            source.push_str("  [#text(size: 9pt, fill: muted)[No glycan candidate]],\n");
        }
        let (label, description) = if is_none_candidate_label(&candidate.candidate) {
            (
                "No glycan".to_string(),
                "Explicit empty-site alternative; no SNFG structure is expected.".to_string(),
            )
        } else {
            (
                format!("Glycoform {}", candidate.candidate),
                "SNFG rendering of the glycan candidate used in the SAXS search.".to_string(),
            )
        };
        let support = candidate_mean_support(saxs, &candidate.candidate)
            .map(|value| format!("Mean site-wise posterior support: {:.1}%", value * 100.0));
        let support_line = support
            .map(|value| format!("#text(size: 8pt, fill: muted)[{}]\\\n", value))
            .unwrap_or_default();
        source.push_str(&format!(
            "  [#text(size: 10pt, fill: accent)[{}]\\\n#text(size: 8pt)[{}]\\\n{}],\n",
            typst_escape(&label),
            typst_escape(&description),
            support_line,
        ));
    }
    source.push_str(")\n");

    if !saxs.glycoform_distribution.is_empty() {
        source.push_str("=== Site-wise glycoform support\n");
        source.push_str(
            "Relative support is derived from χ² likelihoods. Robust multi-feature rank is reported separately and is not a probability.\n",
        );
        source.push_str("#report-table(columns: (auto, auto, auto, auto), align: (left, left, right, right), [*Site*], [*Candidate*], [*Relative support*], [*Site occupancy*],");
        for (site, distribution) in &saxs.glycoform_distribution {
            let occupancy = saxs.site_occupancy.get(site).copied();
            for (candidate, support) in distribution {
                source.push_str(&format!(
                    "[{}], [{}], [{:.1}%], [{}],",
                    typst_escape(site),
                    typst_escape(candidate),
                    support * 100.0,
                    occupancy
                        .map_or_else(|| "n/a".into(), |value| format!("{:.1}%", value * 100.0)),
                ));
            }
        }
        source.push_str(")\n");
    }
}

fn candidate_mean_support(saxs: &SaxsAnalysis, candidate: &str) -> Option<f64> {
    let values = saxs
        .glycoform_distribution
        .values()
        .filter_map(|distribution| distribution.get(candidate).copied())
        .filter(|value| value.is_finite());
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), value| {
        (sum + value, count + 1)
    });
    (count > 0).then_some(sum / count as f64)
}

fn density_classification_label(classification: &str) -> &'static str {
    match classification {
        "density_determined" => "Density-supported",
        "ambiguous" => "Ambiguous",
        "ensemble_prior_determined" => "Prior-determined",
        "glycoshape_fallback" => "GlycoShape fallback",
        _ => "Unclassified",
    }
}

fn density_classification_explanation(classification: &str) -> &'static str {
    match classification {
        "density_determined" => {
            "The fixed-ROI density and connected linkage path select one arm mode."
        }
        "ambiguous" => {
            "Density supports the region, but multiple modes remain in the credible set."
        }
        "ensemble_prior_determined" | "glycoshape_fallback" => {
            "The map does not uniquely determine this arm; native GlycoShape geometry is retained as a chemically valid representative."
        }
        _ => "Evidence classification was not recorded by the fitter.",
    }
}

fn typst_source(report: &WorkflowReport) -> String {
    let mut source = String::from(TEMPLATE_HEADER);
    source.push_str(&format!(
        "\n= ReGlyco scientific report\n#text(size: 9pt, fill: rgb(\"#666666\"))[Command: {}  •  Status: {}]\n",
        typst_escape(&report.provenance.command),
        typst_escape(&report.status)
    ));
    source.push_str(&format!(
        "#text(size: 8pt, fill: muted)[ReGlyco {} • GlySys {} • Seed: {}]\n",
        typst_escape(&report.provenance.reglyco_version),
        typst_escape(&report.provenance.glysys_revision),
        report
            .provenance
            .seed
            .map_or("not set".into(), |value| value.to_string())
    ));
    source.push_str("#line(length: 100%, stroke: (thickness: 0.6pt, paint: accent))\n");
    let torsion_groups = linkage_groups(&report.analysis.glycosidic_torsions);
    if !report.analysis.glycans.is_empty() || !torsion_groups.is_empty() {
        if torsion_groups.is_empty() {
            source.push_str("\n== Glycan identities\n");
            source.push_str(
                "SNFG identities, glycan descriptions, and protein attachment information used for this workflow.\n",
            );
        } else {
            source.push_str("\n== Glycans and glycosidic torsions\n");
            source.push_str(
                "Each glycan starts with its SNFG identity and attachment information, followed by the distributions for its observed glycosidic linkages. Each distribution overlays φ, ψ, and ω when an ω angle is defined.\n",
            );
        }
        for glycan in &report.analysis.glycans {
            source.push_str(&format!("\n=== Glycan {}\n", glycan.index));
            source.push_str("#grid(columns: (34%, 66%), column-gutter: 12pt, row-gutter: 6pt,\n");
            if let Some(asset) = &glycan.snfg_asset {
                source.push_str(&format!(
                    "  image(\"{}\", width: 100%),\n",
                    typst_escape(asset)
                ));
            } else {
                source.push_str("  [],\n");
            }
            source.push_str(&format!(
                "  [#text(size: 8pt)[{}]\\\n#text(font: \"Deja Vu Sans Mono\", size: 7pt)[Residues: {}]\\\n#text(font: \"Deja Vu Sans Mono\", size: 7pt)[WURCS: {}]\\\n#text(font: \"Deja Vu Sans Mono\", size: 7pt)[IUPAC: {}]],\n",
                typst_escape(&attachment_description(glycan.attachment_site.as_deref())),
                glycan.residue_count,
                typst_escape(glycan.wurcs.as_deref().unwrap_or("unavailable")),
                typst_escape(glycan.iupac.as_deref().unwrap_or("unavailable"))
            ));
            source.push_str(")\n");

            let glycan_groups = torsion_groups
                .iter()
                .enumerate()
                .filter(|(_, (glycan_index, _, _, _, _))| *glycan_index == glycan.index)
                .collect::<Vec<_>>();
            if !glycan_groups.is_empty() {
                source
                    .push_str("#text(size: 9pt, fill: accent)[Glycosidic torsion distributions]\n");
                source.push_str(
                    "#grid(columns: (34%, 66%), row-gutter: 10pt, column-gutter: 12pt,\n",
                );
                for (index, _) in glycan_groups {
                    source.push_str(&format!(
                        "  image(\"report-assets/snfg-bond-{}.svg\", width: 100%),\n  image(\"report-assets/glycosidic-{}.svg\", width: 100%),\n",
                        index + 1,
                        index + 1
                    ));
                }
                source.push_str(")\n");
            } else if !torsion_groups.is_empty() {
                source.push_str("No glycosidic torsions were resolved for this glycan.\n");
            }
        }
    }
    if !report.analysis.torsion_references.is_empty() {
        source.push_str("\n== GlycoShape conformational references\n");
        source.push_str(
            "Reference contours are versioned GlycoShape populations. Level-1 colours define the broad mixture; Level-2/3 representatives are shown as medoid markers. Internal torsion tails are advisory.\\\n",
        );
        for (index, reference) in report.analysis.torsion_references.iter().enumerate() {
            if reference.phi_psi.as_ref().is_some_and(|contour| {
                contour.bins >= 2 && contour.grid.len() >= contour.bins * contour.bins
            }) {
                source.push_str(&format!(
                    "#figure(image(\"report-assets/torsion-reference-{}.svg\", width: 92%), caption: [{} · {}])\\\n",
                    index + 1,
                    typst_escape(&reference.canonical_linkage),
                    typst_escape(&format!("source {}", reference.source_database)),
                ));
            }
        }
    }
    if !report.analysis.sterics.is_empty() {
        source.push_str("\n== Steric optimization\n");
        if report.analysis.ensemble.is_some() {
            source.push_str("Per-site summary across accepted ensemble frames.\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right), [*Site*], [*Frames*], [*Median*], [*Range*], [*Clash-free*], [*Rate*],");
            for summary in steric_site_summaries(&report.analysis.sterics) {
                let rate = summary.clash_free as f64 / summary.frames.max(1) as f64 * 100.0;
                source.push_str(&format!(
                    "[{}], [{}], [{:.4}], [{:.4}–{:.4}], [{}/{}], [{:.1}%],",
                    typst_escape(&summary.site),
                    summary.frames,
                    summary.median,
                    summary.minimum,
                    summary.maximum,
                    summary.clash_free,
                    summary.frames,
                    rate,
                ));
            }
        } else {
            source.push_str("#report-table(columns: (auto, auto, auto), align: (left, right, center), [*Site*], [*Score*], [*Clash-free*],");
            for item in &report.analysis.sterics {
                source.push_str(&format!(
                    "[{}], [{:.4}], [{}],",
                    typst_escape(&item.site),
                    item.score,
                    item.clash_free
                ));
            }
        }
        source.push_str(")\n");
    }
    if let Some(saxs) = &report.analysis.saxs {
        source.push_str("\n== SAXS modeling\n");
        source.push_str("SAXS results describe best-fitting models and relative support; they are not a claim of structural truth.\n");
        if let Some(asset) = &saxs.diagnostic_asset {
            source.push_str(&format!(
                "#figure(image(\"{}\", width: 100%), caption: [SAXS intensity, Kratky, and P(r) diagnostics])\n",
                typst_escape(asset)
            ));
        }
        if !saxs.model_names.is_empty() {
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right), [*Model*], [*Reduced χ²*], [*Rg error (Å)*], [*Dmax error (Å)*], [*Kratky*], [*P(r) RMS*],");
            for (index, features) in saxs.model_fits.iter().enumerate() {
                source.push_str(&format!(
                    "[{}], [{:.4}], [{}], [{}], [{}], [{}],",
                    typst_escape(
                        saxs.model_names
                            .get(index)
                            .map(String::as_str)
                            .unwrap_or("model")
                    ),
                    features.reduced_chi2,
                    typst_optional(features.rg_abs_error),
                    typst_optional(features.dmax_abs_error),
                    typst_optional(features.kratky_deviation),
                    typst_optional(features.pr_rms),
                ));
            }
            source.push_str(")\n");
        }
        if saxs.model_fits.is_empty() {
            if let Some(features) = &saxs.fit_features {
                source.push_str("#report-table(columns: (auto, auto, auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right, right, right), [*Fit*], [*χ²*], [*Reduced χ²*], [*Rg (Å)*], [*Dmax (Å)*], [*Kratky*], [*P(r) RMS*],");
                source.push_str(&format!(
                    "[Selected fit], [{:.4}], [{:.4}], [{}], [{}], [{}], [{}],",
                    features.chi2,
                    features.reduced_chi2,
                    typst_optional(features.rg_model),
                    typst_optional(features.dmax_model),
                    typst_optional(features.kratky_deviation),
                    typst_optional(features.pr_rms),
                ));
                source.push_str(")\n");
            }
        }
        if !saxs.weights.is_empty() {
            source.push_str("#report-table(columns: (auto, auto, auto), align: (left, right, right), [*Conformer*], [*Prior*], [*Posterior weight*],");
            for (index, weight) in saxs.weights.iter().enumerate() {
                source.push_str(&format!(
                    "[{}], [{}], [{:.6}],",
                    typst_escape(
                        saxs.model_names
                            .get(index)
                            .map(String::as_str)
                            .unwrap_or("conformer")
                    ),
                    saxs.prior_weights
                        .get(index)
                        .map_or_else(|| "n/a".into(), |value| format!("{value:.6}")),
                    weight,
                ));
            }
            source.push_str(")\n");
        }
        if !saxs.native_log_probabilities.is_empty() {
            source.push_str(
                "Re-Glyco native log probabilities are retained as provenance diagnostics; they are not used as additional SAXS reweighting factors.\\\n",
            );
        }
        if let Some(ess) = saxs.effective_sample_size {
            source.push_str(&format!(
                "Effective sample size: {:.3}; KL divergence from the unbiased prior: {}\\\n",
                ess,
                saxs.kl_divergence
                    .map_or_else(|| "n/a".into(), |value| format!("{value:.6}")),
            ));
        }
        if !saxs.combinations.is_empty() {
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto, auto), align: (left, right, right, right, right, right), [*Combination*], [*Likelihood*], [*log likelihood*], [*Posterior*], [*Robust rank*], [*Reduced χ²*],");
            for combination in saxs.combinations.iter().take(20) {
                let label = combination
                    .assignments
                    .iter()
                    .map(|assignment| format!("{}={}", assignment.site, assignment.candidate))
                    .collect::<Vec<_>>()
                    .join(", ");
                source.push_str(&format!(
                    "[{}], [{:.6}], [{}], [{:.6}], [{:.4}], [{:.4}],",
                    typst_escape(&label),
                    combination.likelihood,
                    typst_optional(combination.log_likelihood),
                    combination.posterior,
                    combination.robust_score,
                    combination.features.reduced_chi2,
                ));
            }
            source.push_str(")\n");
        }
        append_saxs_candidate_report(&mut source, saxs);
    }
    if let Some(search) = &report.search
        && let Some(energy) = search.selected_energy_kcal_per_mol
    {
        source.push_str(&format!(
            "\n== Selected energy score\nMode: {}\\\nSelected score: {:.4} kcal/mol\\\nLennard-Jones: {} kcal/mol\\\nCoulomb: {} kcal/mol\\\nCutoff: {:.2} Å; minimization radius: {:.2} Å\\\nTopology parameterizations: {}\\\nEnergy evaluations: {}; local minimizations: {}; cache hits: {}; steric rejections: {}; failures: {}\\\nWinning active region: {} atoms in {} protein residues\\\nTopology setup: {:.3}s; candidate evaluation: {:.3}s\n",
            match search.scoring_mode {
                reglyco_core::SearchScoringMode::StericPrior => "steric/native prior",
                reglyco_core::SearchScoringMode::FullEnergy => "full Amber/GLYCAM potential",
                reglyco_core::SearchScoringMode::ProteinGlycanInteraction => "protein-glycan LJ + Coulomb interaction",
            },
            energy,
            search.interaction_vdw_kcal_per_mol.map_or_else(|| "n/a".into(), |value| format!("{value:.4}")),
            search.interaction_coulomb_kcal_per_mol.map_or_else(|| "n/a".into(), |value| format!("{value:.4}")),
            search.energy_cutoff_angstrom,
            search.minimization_radius_angstrom,
            search.energy_diagnostics.topology_parameterizations,
            search.energy_diagnostics.energy_evaluations,
            search.energy_diagnostics.minimizations,
            search.energy_diagnostics.cache_hits,
            search.energy_diagnostics.steric_rejections,
            search.energy_diagnostics.failed_evaluations,
            search.energy_diagnostics.active_atoms,
            search.energy_diagnostics.active_residues.len(),
            search.energy_diagnostics.topology_seconds,
            search.energy_diagnostics.evaluation_seconds,
        ));
        if search
            .history
            .iter()
            .any(|entry| entry.best_energy_kcal_per_mol.is_some())
        {
            source.push_str("\n== Search optimization\n");
            source.push_str("#figure(image(\"report-assets/ga-energy-history.svg\", width: 92%), caption: [Selected interaction energy per GA generation])\n");
        }
    }
    if let Some(energy_analysis) = &report.analysis.energy_analysis {
        source.push_str("\n== Energy decomposition\n");
        source.push_str(&format!(
            "Model: {}\\\nUnits: {}\\\nSolvent: {}\\\nBackend: {}\\\nThe selected score {} this physical decomposition. Per-glycan interactions and glycosidic torsions are diagnostic subsets and are not added again.\\\n",
            typst_escape(&energy_analysis.model),
            typst_escape(&energy_analysis.units),
            typst_escape(&energy_analysis.solvent),
            typst_escape(&energy_analysis.backend),
            if energy_analysis.drives_selection { "used" } else { "did not drive" },
        ));
        if let Some(score) = energy_analysis.selected_score {
            source.push_str(&format!(
                "Selected score: {:.6} {}\\\n",
                score,
                typst_escape(&energy_analysis.units)
            ));
        }
        source.push_str("#report-table(columns: (auto, auto), align: (left, right), [*Physical component*], [*Energy*],");
        for (label, value) in [
            ("Bonds", energy_analysis.components.bonds),
            ("Angles", energy_analysis.components.angles),
            (
                "Proper torsions",
                energy_analysis.components.proper_torsions,
            ),
            (
                "Improper torsions",
                energy_analysis.components.improper_torsions,
            ),
            ("Lennard-Jones", energy_analysis.components.van_der_waals),
            ("Electrostatics", energy_analysis.components.electrostatics),
            (
                "Generalized Born",
                energy_analysis.components.generalized_born,
            ),
            ("Surface area", energy_analysis.components.surface_area),
            ("Restraints", energy_analysis.components.restraints),
            (
                "Dispersion correction",
                energy_analysis.components.dispersion_correction,
            ),
        ] {
            source.push_str(&format!("[{}], [{:.6}],", typst_escape(label), value));
        }
        source.push_str(")\n");
        if !energy_analysis.per_glycan_interactions.is_empty() {
            source.push_str("\n=== Protein–glycan interactions\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, left, right, right, right), [*Site*], [*Glycan*], [*Lennard-Jones*], [*Coulomb*], [*Total*],");
            for entry in &energy_analysis.per_glycan_interactions {
                source.push_str(&format!(
                    "[{}], [{}], [{:.6}], [{:.6}], [{:.6}],",
                    typst_escape(&entry.site),
                    typst_escape(&entry.glycan_id),
                    entry.van_der_waals,
                    entry.electrostatics,
                    entry.total,
                ));
            }
            source.push_str(")\n");
        }
        if !energy_analysis.glycosidic_torsions.is_empty() {
            source.push_str("\n=== Glycosidic torsion contributions\n");
            source.push_str("#report-table(columns: (auto, auto, auto), align: (left, left, right), [*Site*], [*Linkage atoms*], [*Proper torsion energy*],");
            for entry in &energy_analysis.glycosidic_torsions {
                let atoms = entry
                    .atoms
                    .iter()
                    .map(|atom| atom.to_string())
                    .collect::<Vec<_>>()
                    .join("-");
                source.push_str(&format!(
                    "[{}], [{}], [{:.6}],",
                    typst_escape(&entry.site),
                    typst_escape(&atoms),
                    entry.energy,
                ));
            }
            source.push_str(")\n");
        }
        if let Some(remainder) = energy_analysis.diagnostic_remainder {
            source.push_str(&format!(
                "Diagnostic remainder after cross-interaction subsets: {:.6} {}.\\\n",
                remainder,
                typst_escape(&energy_analysis.units)
            ));
        }
    }
    if report
        .search
        .as_ref()
        .is_some_and(|search| !search.history.is_empty())
    {
        if !report
            .search
            .as_ref()
            .is_some_and(|search| search.selected_energy_kcal_per_mol.is_some())
        {
            source.push_str("\n== Search optimization\n");
        }
        source.push_str("#figure(image(\"report-assets/ga-history.svg\", width: 92%), caption: [Best steric score per GA generation. Lower is better; ≤ 1.1 is clash-free.])\n");
    }
    if let Some(ensemble) = &report.analysis.ensemble {
        source.push_str(&format!("\n== Ensemble sampling\nRequested frames: {}\\\nReturned frames: {}\\\nNative sampler frames: {}\\\nGA-seeded MH frames: {}\\\nFallback used: {}\\\nProposals: {} (native {}, MH {})\\\nAccepted: {} (native {}, MH {})\\\nAcceptance: {:.2}%\\\nChains: {}; burn-in sweeps: {}; thinning: {}\\\nGA restarts: {}\n", ensemble.requested_frames, ensemble.returned_frames, ensemble.native_frames, ensemble.fallback_frames, ensemble.fallback_used, ensemble.attempts, ensemble.native_proposals, ensemble.mh_proposals, ensemble.native_accepts + ensemble.mh_accepts, ensemble.native_accepts, ensemble.mh_accepts, ensemble.acceptance_rate * 100.0, ensemble.chains, ensemble.burn_in_sweeps, ensemble.thinning_accepted, ensemble.ga_restarts));
        if let Some(target) = ensemble.sampling_target {
            let target = match target {
                reglyco_core::SamplingTarget::CpuReferenceV1 => "cpu_reference_v1",
                reglyco_core::SamplingTarget::WebgpuF32V1 => "webgpu_f32_v1",
            };
            source.push_str(&format!("Numerical target: {}.\\\n", typst_escape(target)));
        }
        if !ensemble.sampling_segments.is_empty() {
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, left, right, right, right), [*Segment*], [*Backend*], [*Frames*], [*Attempts*], [*Accepts*],");
            for segment in &ensemble.sampling_segments {
                source.push_str(&format!(
                    "[{}], [{}], [{}], [{}], [{}],",
                    segment.id,
                    typst_escape(&segment.backend),
                    segment.frames,
                    segment.attempts,
                    segment.accepts,
                ));
            }
            source.push_str(")\n");
            for segment in ensemble
                .sampling_segments
                .iter()
                .filter(|segment| segment.fallback_reason.is_some())
            {
                if let Some(reason) = &segment.fallback_reason {
                    source.push_str(&format!(
                        "Segment {} fallback: {}\\\n",
                        segment.id,
                        typst_escape(reason)
                    ));
                }
            }
        }
    }

    // Protein-glycan linkage torsion visualization
    if !report.analysis.protein_linkage_torsions.is_empty() {
        source.push_str("\n== Protein–Glycan Linkage Torsions\n");

        // Check if we have ensemble data (multiple frames)
        let has_ensemble = report
            .analysis
            .protein_linkage_torsions
            .iter()
            .any(|t| t.frame.is_some());

        if has_ensemble {
            source.push_str("Density of (φ, ψ) across accepted ensemble frames; circles mark individual frames.\n");
            source.push_str("#figure(image(\"report-assets/protein-linkage-kde.svg\", width: 92%), caption: [Protein–glycan linkage torsion density across accepted frames])\n");

            // Summary table with ranges
            source.push_str("#report-table(columns: (auto, auto, auto, auto), align: (left, right, right, right), [*Site*], [*Frames*], [*φ range*], [*ψ range*],");
            let mut site_torsions: std::collections::BTreeMap<String, Vec<&ProteinLinkageTorsion>> =
                std::collections::BTreeMap::new();
            for t in &report.analysis.protein_linkage_torsions {
                site_torsions.entry(t.site.clone()).or_default().push(t);
            }
            for (site, torsions) in site_torsions {
                let phi_vals: Vec<_> = torsions.iter().map(|t| t.phi_degrees).collect();
                let psi_vals: Vec<_> = torsions.iter().map(|t| t.psi_degrees).collect();
                let phi_min = phi_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let phi_max = phi_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let psi_min = psi_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let psi_max = psi_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let display_site = report_site_label(&site, &report.analysis.glycans);
                source.push_str(&format!(
                    "[{}], [{}], [{:.1}° to {:.1}°], [{:.1}° to {:.1}°],",
                    typst_escape(&display_site),
                    torsions.len(),
                    phi_min,
                    phi_max,
                    psi_min,
                    psi_max
                ));
            }
            source.push_str(")\n");
        } else {
            source.push_str("Protein–glycan attachment torsion angles for the final structure.\n");
            source.push_str("#figure(image(\"report-assets/protein-linkage-points.svg\", width: 92%), caption: [Protein–glycan linkage torsion angles for the final structure])\n");

            // Table with individual phi/psi values
            source.push_str("#report-table(columns: (auto, auto, auto), align: (left, right, right), [*Site*], [*φ*], [*ψ*],");
            for t in &report.analysis.protein_linkage_torsions {
                let display_site = report_site_label(&t.site, &report.analysis.glycans);
                source.push_str(&format!(
                    "[{}], [{:.1}°], [{:.1}°],",
                    typst_escape(&display_site),
                    t.phi_degrees,
                    t.psi_degrees
                ));
            }
            source.push_str(")\n");
        }
    }

    if let Some(density) = &report.analysis.density {
        source.push_str("\n== Density agreement\n");
        let density_arms = density.arm_evidence.len();
        let density_determined_arms = density
            .arm_evidence
            .iter()
            .filter(|arm| arm.classification == "density_determined")
            .count();
        let ambiguous_arms = density
            .arm_evidence
            .iter()
            .filter(|arm| arm.classification == "ambiguous")
            .count();
        let prior_arms = density_arms.saturating_sub(density_determined_arms + ambiguous_arms);
        source.push_str("=== Executive summary\n");
        source.push_str(
            "The density fit is ranked using a fixed site-local map objective and chemistry gates. Deposited coordinates are recovery diagnostics only. A density-supported label means the map selects a connected arm mode; ambiguous means alternatives remain plausible; prior-determined means the map does not resolve that arm and the native ensemble supplies the representative.\\\n",
        );
        source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, right, right, right, right), [*Metric*], [*Value*], [*Density-supported arms*], [*Ambiguous arms*], [*Prior-determined arms*],");
        source.push_str(&format!(
            "[Masked map correlation], [{:.4}], [{}], [{}], [{}],",
            density.pre_relax_correlation, density_determined_arms, ambiguous_arms, prior_arms,
        ));
        source.push_str(&format!(
            "[Evaluations], [{}], [--], [--], [--],",
            density.evaluations
        ));
        if let Some(seconds) = density.total_seconds {
            source.push_str(&format!(
                "[Total fitting time], [{seconds:.1} s], [--], [--], [--],"
            ));
        }
        source.push_str(")\n");
        if !density.stage_timings.is_empty() {
            source.push_str("\n*Production-stage profile*\\\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto), align: (left, right, right, right), [*Stage*], [*Seconds*], [*Evaluations*], [*Share of optimization*],");
            let optimization_seconds = density.optimization_seconds.unwrap_or(0.0);
            for stage in &density.stage_timings {
                let share = if optimization_seconds > 0.0 {
                    100.0 * stage.seconds / optimization_seconds
                } else {
                    0.0
                };
                source.push_str(&format!(
                    "[{}], [{:.2}], [{}], [{:.1}%],",
                    typst_escape(&stage.stage),
                    stage.seconds,
                    stage.evaluations,
                    share,
                ));
            }
            source.push_str(")\n");
        }
        source.push_str(&format!(
            "Masked map correlation: {:.4}\\\nEvaluations: {}\\\nAtom support: {}\\\nRing support: {}\\\nConnectivity support: {}\\\n",
            density.pre_relax_correlation,
            density.evaluations,
            density
                .supported_atom_fraction
                .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
            density
                .ring_support
                .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
            density
                .connectivity_support
                .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
        ));
        if let Some(sigma) = density.sigma_angstrom {
            source.push_str(&format!("Calibrated density sigma: {sigma:.3} Å\\\n"));
        }
        if let Some(sigma) = density.effective_sigma_angstrom {
            source.push_str(&format!("Effective B-broadened sigma: {sigma:.3} Å\\\\\\n"));
        }
        if let Some(sigma) = density.capture_sigma_angstrom {
            source.push_str(&format!("Automatic capture sigma: {sigma:.3} Å\\\\\\n"));
        }
        if let Some(floor) = density.anti_alias_floor_angstrom {
            source.push_str(&format!("Voxel anti-alias floor: {floor:.3} Å\\\\\\n"));
        }
        if !density.kernel_decisions.is_empty() {
            let accepted = density
                .kernel_decisions
                .iter()
                .filter(|kernel| kernel.accepted)
                .count();
            source.push_str(&format!(
                "Residue-local blur trials: {} ({} accepted by held-out/BIC gate)\\\\\\n",
                density.kernel_decisions.len(),
                accepted
            ));
        }
        if let (Some(training), Some(heldout)) = (
            density.training_likelihood_gain,
            density.heldout_likelihood_gain,
        ) {
            source.push_str(&format!(
                "Fixed-ROI likelihood: training {training:.3}, held-out {heldout:.3}\\\n"
            ));
        }
        if let Some(difference) = density.difference_score {
            source.push_str(&format!(
                "Signed Fo-Fc consistency score: {difference:.4}\\\n"
            ));
        }
        if let Some(post) = density.post_relax_correlation {
            source.push_str(&format!("Post-relaxation correlation: {post:.4}\\\n"));
        }
        if !density.site_diagnostics.is_empty() {
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, right, right, right, right), [*Site*], [*Correlation*], [*Atom support*], [*Ring support*], [*Connectivity*],");
            for site in &density.site_diagnostics {
                source.push_str(&format!(
                    "[{}], [{:.4}], [{:.3}], [{:.3}], [{:.3}],",
                    typst_escape(&site.site),
                    site.correlation,
                    site.supported_atom_fraction,
                    site.ring_support,
                    site.connectivity_support,
                ));
            }
            source.push_str(")\n");
            let has_detailed_recovery = density.recovery.iter().any(|recovery| {
                !recovery.per_residue_heavy_atom_rmsd_angstrom.is_empty()
                    || !recovery.arm_heavy_atom_rmsd_angstrom.is_empty()
            });
            if has_detailed_recovery {
                source.push_str("#report-table(columns: (auto, auto, auto), align: (left, left, right), [*Site*], [*Residue / connected arm*], [*Heavy-atom RMSD (Å)*],");
                for recovery in &density.recovery {
                    for (residue, rmsd) in &recovery.per_residue_heavy_atom_rmsd_angstrom {
                        source.push_str(&format!(
                            "[{}], [{}], [{rmsd:.3}],",
                            typst_escape(&recovery.site),
                            typst_escape(residue),
                        ));
                    }
                    for (arm, rmsd) in &recovery.arm_heavy_atom_rmsd_angstrom {
                        source.push_str(&format!(
                            "[{}], [arm {}], [{rmsd:.3}],",
                            typst_escape(&recovery.site),
                            typst_escape(arm),
                        ));
                    }
                }
                source.push_str(")\n");
            }
        }
        if !density.recovery.is_empty() {
            source.push_str(
                "Recovery comparison is informational only and is not used for pass/fail.\\\n",
            );
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, right, right, right, right), [*Site*], [*Root C1 distance (Å)*], [*Three-residue RMSD (Å)*], [*Supported RMSD (Å)*], [*Full-tree RMSD (Å)*],");
            for recovery in &density.recovery {
                source.push_str(&format!(
                    "[{}], [{}], [{}], [{}], [{}],",
                    typst_escape(&recovery.site),
                    recovery
                        .root_c1_distance_angstrom
                        .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
                    recovery
                        .three_residue_heavy_atom_rmsd_angstrom
                        .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
                    recovery
                        .supported_heavy_atom_rmsd_angstrom
                        .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
                    recovery
                        .full_tree_heavy_atom_rmsd_angstrom
                        .map_or_else(|| "n/a".into(), |value| format!("{value:.3}")),
                ));
            }
            source.push_str(")\n");
        }
        if !density.ensemble_baselines.is_empty() {
            source.push_str("\n=== GlycoShape conformer baselines\n");
            source.push_str("These untouched attached conformers are diagnostics and did not influence final model ranking.\\\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, left, right, right, right), [*Baseline*], [*Conformer*], [*CC*], [*Likelihood gain*], [*RMSD to fit (Å)*],");
            for baseline in &density.ensemble_baselines {
                source.push_str(&format!(
                    "[{}], [{}], [{:.4}], [{:.3}], [{:.3}],",
                    typst_escape(&baseline.kind.replace('_', " ")),
                    typst_escape(&baseline.conformer_ids.join(", ")),
                    baseline.correlation,
                    baseline.likelihood_gain,
                    baseline.rmsd_to_fitted_angstrom,
                ));
            }
            source.push_str(")\n");
        }
        if !density.arm_evidence.is_empty() {
            source.push_str("\n=== Connected-arm evidence\n");
            source.push_str("Each arm is inferred independently conditional on the fitted core. Negative gain is prior-determined; ambiguous arms retain a 95% marginal credible set.\\\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto, auto, auto), align: (left, left, left, right, right, right, right), [*Arm*], [*Classification*], [*Selected mode*], [*Density gain*], [*Gain / heavy atom*], [*Top posterior*], [*95% modes*],");
            for arm in &density.arm_evidence {
                source.push_str(&format!(
                    "[{}], [{}], [{}], [{:.3}], [{:.3}], [{:.3}], [{}],",
                    typst_escape(&arm.label),
                    density_classification_label(&arm.classification),
                    typst_escape(&arm.selected_mode),
                    arm.fixed_roi_likelihood_gain,
                    arm.normalized_density_gain,
                    arm.selected_mode_posterior,
                    arm.credible_set_size,
                ));
            }
            source.push_str(")\n");
            source.push_str("\n*Per-residue evidence status*\\\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto), align: (left, left, left, right, right), [*Residue*], [*Arm*], [*Status*], [*Ring support*], [*Linkage path*],");
            for arm in &density.arm_evidence {
                for residue in &arm.residues {
                    source.push_str(&format!(
                        "[{}], [{}], [{}], [{:.3}], [{:.3}],",
                        typst_escape(residue),
                        typst_escape(&arm.label),
                        density_classification_label(&arm.classification),
                        arm.ring_support,
                        arm.linkage_path_support,
                    ));
                }
            }
            source.push_str(")\n");
            for arm in &density.arm_evidence {
                source.push_str(&format!(
                    "*{}:* {}\\\n",
                    typst_escape(&arm.label),
                    density_classification_explanation(&arm.classification),
                ));
            }
            for arm in &density.arm_evidence {
                if arm.alternatives.is_empty() {
                    continue;
                }
                source.push_str(&format!(
                    "\\\n*{} alternatives:* ",
                    typst_escape(&arm.label)
                ));
                for alternative in arm.alternatives.iter().filter(|mode| mode.in_credible_set) {
                    source.push_str(&format!(
                        "{} (p={:.3}, cumulative={:.3}, sources={}); ",
                        typst_escape(&alternative.mode_id),
                        alternative.posterior_weight,
                        alternative.cumulative_posterior_weight,
                        typst_escape(&alternative.source_conformer_ids.join(", ")),
                    ));
                }
                source.push_str("\\\n");
            }
        }
        if !density.basin_diagnostics.is_empty() {
            source.push_str("\\n=== Automatic basin competition\\n");
            source.push_str("Basins are clustered by attachment/core geometry; pilot posterior and residual bounds determine polishing.\\\\\\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto, auto, auto, auto), align: (left, right, right, right, left, right, right), [*Basin*], [*Pilot*], [*Bound*], [*Posterior*], [*State*], [*Evaluations*], [*Seconds*],");
            for basin in &density.basin_diagnostics {
                source.push_str(&format!(
                    "[{}], [{:.3}], [{:.3}], [{:.3}], [{}], [{}], [{:.1}],",
                    typst_escape(&basin.basin_id),
                    basin.pilot_score,
                    basin.improvement_bound,
                    basin.pilot_posterior,
                    typst_escape(&basin.status.replace('_', " ")),
                    basin.evaluations,
                    basin.seconds,
                ));
            }
            source.push_str(")\\n");
        }
        if !density.candidate_correlations.is_empty() {
            source.push_str("Candidate correlations and posterior weights are recorded in report.json and density.json; local observed, calculated, and mask CCP4 maps are included in the output bundle.\n");
        }
        if !density.density_determined_residues.is_empty() {
            source.push_str(&format!(
                "Density-determined residues: {}.\\\n",
                typst_escape(&density.density_determined_residues.join(", "))
            ));
        }
        if !density.ensemble_prior_residues.is_empty() {
            source.push_str(&format!(
                "Ensemble-prior-determined residues: {}.\\\n",
                typst_escape(&density.ensemble_prior_residues.join(", "))
            ));
        }
    }

    if let Some(relaxation) = &report.relaxation {
        source.push_str("\n== Staged relaxation\n#figure(image(\"report-assets/energy-history.svg\", width: 92%), caption: [Total potential energy after each relaxation stage])\n#table(columns: (auto, auto, auto, auto), align: (left, right, center, right), [*Stage*], [*Iterations*], [*Converged*], [*Final energy (kcal/mol)*],");
        for stage in &relaxation.stages {
            source.push_str(&format!(
                "[{}], [{}], [{}], [{:.4}],",
                typst_escape(&stage.name),
                stage.iterations,
                stage.converged,
                stage.final_energy.total()
            ));
        }
        source.push_str(")\n");
        source.push_str("=== Energy components (kcal/mol)\n#table(columns: (auto, auto, auto), align: (left, right, right), [*Component*], [*Initial*], [*Final*],");
        for (name, initial, final_value) in
            energy_components(&relaxation.initial_energy, &relaxation.final_energy)
        {
            source.push_str(&format!(
                "[{}], [{:.4}], [{:.4}],",
                name, initial, final_value
            ));
        }
        source.push_str(")\n");
    }
    if let Some(scan) = &report.analysis.scan {
        source.push_str(&format!("\n== Scan interpretation\n{}\\\nIndependent accessibility: {}\\\nJointly compatible: {}\n", typst_escape(&scan.interpretation), scan.independent_accessible_count, scan.jointly_compatible_count));
        if !scan.sequons.is_empty() {
            source.push_str("\n=== Unoccupied N-X-S/T sequons\n");
            source.push_str("The asparagine in each context is shown in blue; status distinguishes independently accessible sites from the jointly retained subset. Rows not marked attached in the joint subset are the non-glycosylated candidates in this scan.\n");
            source.push_str("#report-table(columns: (auto, auto, auto, auto), align: (left, left, center, left), [*Sequon*], [*Asparagine site*], [*Accessible*], [*Joint status*],");
            for sequon in &scan.sequons {
                let status = if sequon.jointly_selected {
                    "attached in joint subset"
                } else if sequon.independently_accessible {
                    "not retained jointly"
                } else {
                    "sterically blocked"
                };
                source.push_str(&format!(
                    "{}, [chain {}, ASN {}], [{}], [{}],",
                    typst_colored_sequon(&sequon.context),
                    typst_escape(&sequon.asparagine.chain),
                    sequon.asparagine.number,
                    sequon.independently_accessible,
                    status,
                ));
            }
            source.push_str(")\n");
        }
    }
    if !report.warnings.is_empty() {
        source.push_str("\n== Warnings\n");
        for warning in &report.warnings {
            source.push_str(&format!("- {}\n", typst_escape(warning)));
        }
    }
    if let Some(seconds) = report
        .provenance
        .total_seconds
        .filter(|value| value.is_finite())
    {
        source.push_str("\n== Run summary\n");
        source.push_str(&format!("Total workflow time: {seconds:.2} s\n"));
    }
    source
}

fn energy_components(
    initial: &glysys_energy::EnergyComponents,
    final_value: &glysys_energy::EnergyComponents,
) -> [(&'static str, f64, f64); 9] {
    [
        ("Bonds", initial.bonds, final_value.bonds),
        ("Angles", initial.angles, final_value.angles),
        (
            "Proper torsions",
            initial.proper_torsions,
            final_value.proper_torsions,
        ),
        (
            "Improper torsions",
            initial.improper_torsions,
            final_value.improper_torsions,
        ),
        (
            "Lennard-Jones",
            initial.van_der_waals,
            final_value.van_der_waals,
        ),
        (
            "Coulomb",
            initial.electrostatics,
            final_value.electrostatics,
        ),
        (
            "Generalized Born",
            initial.generalized_born,
            final_value.generalized_born,
        ),
        (
            "Surface area",
            initial.surface_area,
            final_value.surface_area,
        ),
        ("Restraints", initial.restraints, final_value.restraints),
    ]
}

fn compile_typst(source: &str, root: &Path) -> Result<Vec<u8>> {
    let font_options = TypstKitFontOptions::new()
        .include_system_fonts(false)
        .include_embedded_fonts(true);
    let engine = TypstEngine::builder()
        .main_file(source.to_owned())
        .search_fonts_with(font_options)
        .with_file_system_resolver(root)
        .build();
    let document: PagedDocument = engine
        .compile()
        .output
        .map_err(|errors| ReportError::Typst(format!("{errors:?}")))?;
    typst_pdf::pdf(&document, &Default::default())
        .map_err(|error| ReportError::Typst(format!("{error:?}")))
}

fn typst_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('#', "\\#")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('@', "\\@")
}

fn typst_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| format!("{value:.4}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typst_escaping_is_safe() {
        assert_eq!(typst_escape("#[a]"), "\\#\\[a\\]");
    }

    #[test]
    fn empty_report_writes_a_pdf_bundle() {
        let directory = tempfile::tempdir().unwrap();
        let report = WorkflowReport {
            status: "complete".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance::default(),
            warnings: Vec::new(),
            diagnostics: Vec::new(),
            analysis: ReportAnalysis::default(),
        };
        report.write_bundle(directory.path()).unwrap();
        assert!(
            std::fs::read(directory.path().join("report.pdf"))
                .unwrap()
                .starts_with(b"%PDF")
        );
    }

    #[test]
    fn density_basin_diagnostics_render_in_report() {
        let directory = tempfile::tempdir().unwrap();
        let mut density = DensityAnalysis::default();
        density.basin_diagnostics.push(DensityBasinAnalysis {
            basin_id: "model-1@-150,-170".into(),
            pilot_score: 2.0,
            improvement_bound: 2.5,
            pilot_posterior: 0.8,
            selected: true,
            fully_polished: true,
            final_score: Some(3.0),
            evaluations: 4,
            seconds: 0.1,
            status: "polished".into(),
        });
        density.optimization_seconds = Some(1.0);
        density.total_seconds = Some(1.0);
        density.evaluations = 12;
        density.stage_timings.push(DensityStageAnalysis {
            stage: "adaptive local".into(),
            seconds: 1.0,
            evaluations: 12,
        });
        density.arm_evidence.push(DensityArmAnalysis {
            label: "B:3->B:6".into(),
            residues: vec!["B:6".into()],
            classification: "density_determined".into(),
            selected_mode: "mode-1".into(),
            fixed_roi_likelihood_gain: 1.0,
            normalized_density_gain: 0.5,
            selected_mode_posterior: 0.9,
            credible_set_size: 1,
            ring_support: 0.8,
            linkage_path_support: 0.7,
            ..DensityArmAnalysis::default()
        });
        let report = WorkflowReport {
            status: "complete".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance::default(),
            warnings: Vec::new(),
            diagnostics: Vec::new(),
            analysis: ReportAnalysis {
                density: Some(density),
                ..ReportAnalysis::default()
            },
        };
        report.write_bundle(directory.path()).unwrap();
        assert!(directory.path().join("report.pdf").exists());
    }

    #[test]
    fn kde_functions_work() {
        let data = vec![0.0, 10.0, 20.0, 30.0];
        let bw = silverman_bandwidth(&data);
        assert!(bw > 0.0);

        let density = kde_1d(15.0, &data, bw);
        assert!(density > 0.0);

        let data2d_x = vec![0.0, 10.0, 20.0];
        let data2d_y = vec![0.0, 10.0, 20.0];
        let density2d = kde_2d(10.0, 10.0, &data2d_x, &data2d_y, bw, bw);
        assert!(density2d > 0.0);
    }

    #[test]
    fn periodic_kde_keeps_the_angle_boundary_contiguous() {
        let x = [179.0, -179.0, 178.0, -178.0];
        let y = [20.0, 20.0, 21.0, 19.0];
        let bandwidth = periodic_bandwidth(&x);
        assert!((5.0..=45.0).contains(&bandwidth));
        let seam = kde_2d(180.0, 20.0, &x, &y, bandwidth, periodic_bandwidth(&y));
        let opposite = kde_2d(0.0, 20.0, &x, &y, bandwidth, periodic_bandwidth(&y));
        assert!(seam > opposite * 10.0, "seam={seam}, opposite={opposite}");
    }

    #[test]
    fn ensemble_steric_summary_is_one_row_per_site() {
        let observations = vec![
            StericObservation {
                site: "A:1".into(),
                score: 1.0,
                clash_free: true,
            },
            StericObservation {
                site: "A:1".into(),
                score: 1.4,
                clash_free: false,
            },
            StericObservation {
                site: "A:1".into(),
                score: 1.2,
                clash_free: false,
            },
            StericObservation {
                site: "A:2".into(),
                score: 0.9,
                clash_free: true,
            },
        ];
        let summaries = steric_site_summaries(&observations);
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].site, "A:1");
        assert_eq!(summaries[0].frames, 3);
        assert!((summaries[0].median - 1.2).abs() < 1.0e-12);
        assert_eq!(summaries[0].clash_free, 1);
        assert_eq!(summaries[1].site, "A:2");
    }

    #[test]
    fn ensemble_typst_uses_compact_steric_rows_but_keeps_raw_analysis() {
        let observations = vec![
            StericObservation {
                site: "A:1".into(),
                score: 1.0,
                clash_free: true,
            },
            StericObservation {
                site: "A:1".into(),
                score: 1.4,
                clash_free: false,
            },
            StericObservation {
                site: "A:1".into(),
                score: 1.2,
                clash_free: false,
            },
            StericObservation {
                site: "A:2".into(),
                score: 0.9,
                clash_free: true,
            },
        ];
        let report = WorkflowReport {
            status: "complete".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance::default(),
            warnings: Vec::new(),
            diagnostics: Vec::new(),
            analysis: ReportAnalysis {
                sterics: observations.clone(),
                ensemble: Some(EnsembleAnalysis::default()),
                ..ReportAnalysis::default()
            },
        };
        let source = typst_source(&report);
        assert_eq!(source.matches("[A:1]").count(), 1);
        assert_eq!(source.matches("[A:2]").count(), 1);
        assert!(source.contains("[1.2000]"));
        assert!(source.contains("[1/3]"));
        assert_eq!(report.analysis.sterics.len(), observations.len());
    }

    #[test]
    fn glycosidic_plot_overlays_phi_psi_and_omega_on_one_axis() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("linkage.svg");
        linkage_distribution_plot(
            path.clone(),
            "A:151 C1–O6",
            &[10.0, 20.0, 30.0],
            &[-30.0, -20.0, -10.0],
            &[170.0, -175.0, 178.0],
        )
        .unwrap();
        let svg = std::fs::read_to_string(path).unwrap();
        assert_eq!(svg.matches("angle (degrees)").count(), 1);
        assert!(svg.contains("φ"));
        assert!(svg.contains("ψ"));
        assert!(svg.contains("ω"));
        assert!(svg.contains("#0F6B6B"));
    }

    #[test]
    fn combined_glycan_section_precedes_its_torsion_assets() {
        let report = WorkflowReport {
            status: "complete".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance::default(),
            warnings: Vec::new(),
            diagnostics: Vec::new(),
            analysis: ReportAnalysis {
                glycans: vec![GlycanSummary {
                    index: 1,
                    attachment_site: Some("A/ASN/151".into()),
                    residue_count: 3,
                    wurcs: Some("WURCS=2.0/3,3,2/...".into()),
                    iupac: Some("Man(a1-6)GlcNAc".into()),
                    snfg_asset: Some("report-assets/glycan-1.svg".into()),
                    ..GlycanSummary::default()
                }],
                glycosidic_torsions: vec![LinkageTorsion {
                    glycan_index: 1,
                    linkage: "A:151 C1–O6".into(),
                    phi_degrees: Some(10.0),
                    psi_degrees: Some(20.0),
                    omega_degrees: Some(-60.0),
                    frame: Some(1),
                    donor_name: None,
                    acceptor_name: None,
                    donor_position: None,
                    acceptor_position: None,
                    donor_residue: None,
                    acceptor_residue: None,
                }],
                ..ReportAnalysis::default()
            },
        };
        let source = typst_source(&report);
        let section = source.find("== Glycans and glycosidic torsions").unwrap();
        let identity = source.find("WURCS=2.0").unwrap();
        let distributions = source.find("Glycosidic torsion distributions").unwrap();
        assert!(section < identity && identity < distributions);
        assert!(source.contains("Attached to chain A, residue 151: ASN (asparagine)"));
        assert!(source.contains("glycosidic-1.svg"));
        assert!(!source.contains("== Glycans\n"));
        assert!(!source.contains("== Glycosidic Linkage Torsions"));
    }

    #[test]
    fn glycan_identity_opens_report_and_timing_is_last() {
        let report = WorkflowReport {
            status: "complete".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance {
                total_seconds: Some(2.5),
                ..Provenance::default()
            },
            warnings: vec!["informational warning".into()],
            diagnostics: Vec::new(),
            analysis: ReportAnalysis {
                glycans: vec![GlycanSummary {
                    index: 1,
                    attachment_site: Some("A/ASN/139".into()),
                    residue_count: 11,
                    ..GlycanSummary::default()
                }],
                ..ReportAnalysis::default()
            },
        };
        let source = typst_source(&report);
        let identity = source.find("== Glycan identities").unwrap();
        let warning = source.find("== Warnings").unwrap();
        let timing = source.find("== Run summary").unwrap();
        assert!(identity < warning && warning < timing);
        assert!(source.ends_with("Total workflow time: 2.50 s\n"));
    }

    #[test]
    fn scan_report_colors_the_asparagine_in_each_sequon_context() {
        let report = WorkflowReport {
            status: "complete".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance::default(),
            warnings: Vec::new(),
            diagnostics: Vec::new(),
            analysis: ReportAnalysis {
                scan: Some(ScanAnalysis {
                    interpretation: "structural accessibility only".into(),
                    independent_accessible_count: 0,
                    jointly_compatible_count: 0,
                    sequons: vec![ScanSequon {
                        context: "LNTT".into(),
                        motif: "NTT".into(),
                        asparagine: ResidueId {
                            chain: "A".into(),
                            number: 151,
                            insertion_code: None,
                        },
                        independently_accessible: false,
                        jointly_selected: false,
                    }],
                }),
                ..ReportAnalysis::default()
            },
        };
        let source = typst_source(&report);
        assert!(source.contains("L#text(fill: rgb(\"#2166F3\"), weight: \"bold\")[N]TT"));
        assert!(source.contains("[chain A, ASN 151]"));
    }

    #[test]
    fn exact_snfg_selection_does_not_expand_to_repeated_motif() {
        let graph =
            crabwurcs_iupac::parse_iupac_condensed("Fuc(a1-3)GlcNAc(b1-4)Fuc(a1-3)GlcNAc").unwrap();
        let provenance = graph
            .inner()
            .node_indices()
            .map(|node| PdbResidueReference {
                node_index: node.index(),
                chain: "A".into(),
                sequence_number: 100 + node.index() as isize,
                insertion_code: None,
            })
            .collect::<Vec<_>>();
        let edge = graph.inner().edge_references().next().unwrap();
        let torsion = LinkageTorsion {
            glycan_index: 1,
            linkage: "A:100–A:101".into(),
            phi_degrees: Some(10.0),
            psi_degrees: Some(20.0),
            omega_degrees: None,
            frame: Some(1),
            donor_name: Some("FUC".into()),
            acceptor_name: Some("NAG".into()),
            donor_position: Some(edge.weight().child_position.0),
            acceptor_position: Some(edge.weight().parent_position.0),
            donor_residue: Some(ResidueId {
                chain: "A".into(),
                number: edge.target().index() as i32 + 100,
                insertion_code: None,
            }),
            acceptor_residue: Some(ResidueId {
                chain: "A".into(),
                number: edge.source().index() as i32 + 100,
                insertion_code: None,
            }),
        };
        let glycan = GlycanSummary {
            index: 1,
            attachment_site: None,
            residue_count: graph.node_count(),
            wurcs: None,
            iupac: None,
            snfg_asset: None,
            snfg_svg: None,
            snfg_graph: Some(graph),
            snfg_residues: Some(provenance),
        };
        let svg = snfg_bond_svg(&glycan, &torsion).expect("exact pair should resolve");
        assert_eq!(svg.matches("class=\"motif-match\"").count(), 3);
        assert_eq!(svg.matches("class=\"motif-dimmed\"").count(), 4);
    }

    #[test]
    fn unresolved_snfg_provenance_falls_back_to_plain_rendering() {
        let graph = crabwurcs_iupac::parse_iupac_condensed("Fuc(a1-3)GlcNAc").unwrap();
        let glycan = GlycanSummary {
            index: 1,
            attachment_site: None,
            residue_count: graph.node_count(),
            wurcs: None,
            iupac: None,
            snfg_asset: None,
            snfg_svg: None,
            snfg_graph: Some(graph),
            snfg_residues: Some(Vec::new()),
        };
        let torsion = LinkageTorsion {
            glycan_index: 1,
            linkage: "unknown".into(),
            phi_degrees: None,
            psi_degrees: None,
            omega_degrees: None,
            frame: None,
            donor_name: None,
            acceptor_name: None,
            donor_position: Some(1),
            acceptor_position: Some(3),
            donor_residue: Some(ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            }),
            acceptor_residue: Some(ResidueId {
                chain: "A".into(),
                number: 2,
                insertion_code: None,
            }),
        };
        assert!(snfg_bond_svg(&glycan, &torsion).is_none());
    }

    #[test]
    fn filled_contour_kde_writes_smooth_polygon_bands_for_seam_data() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("torsion-kde.svg");
        let torsions = [-179.0, -178.0, 178.0, 179.0, 180.0]
            .into_iter()
            .enumerate()
            .map(|(index, phi)| ProteinLinkageTorsion {
                site: "A:151".into(),
                phi_degrees: phi,
                psi_degrees: 20.0 + index as f64,
                frame: Some(index + 1),
            })
            .collect::<Vec<_>>();
        filled_contour_kde_plot(path.clone(), &torsions).unwrap();
        let svg = std::fs::read_to_string(path).unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.matches("<polygon").count() > 0, "{svg}");
        assert!(!svg.contains("<rect"));
        assert!(!svg.contains("colorbar"));
        assert!(!svg.contains("relative density"));
    }

    #[test]
    fn torsion_dihedral_is_computed() {
        let a = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let b = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let c = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let d = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 1.0,
        };
        let angle = dihedral(a, b, c, d);
        assert!((angle).is_finite());
    }
}
