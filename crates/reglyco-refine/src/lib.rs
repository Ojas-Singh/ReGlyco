//! Search, build, and relaxation orchestration for the steric/energy
//! refinement workflow.
//!
//! Density fitting is not part of this crate: `reglyco refine --objective
//! density` runs the GlycoFlow fitter (crate `reglyco-glycoflow`).

use glysys::{ParameterizedSystem, Structure, SystemBuilder};
use reglyco_build::BuildProduct;
use reglyco_core::{SearchConfig, SearchOutcome, SearchSite};
use reglyco_ensemble::{SearchProgress, build_from_outcome, search_with_progress, steric_score};
use reglyco_relax::{RelaxOptions, RelaxProgress, RelaxationDiagnostics, relax_with_progress};
use reglyco_report::{Provenance, ReportAnalysis, WorkflowReport};

pub type Result<T> = std::result::Result<T, RefineError>;

#[derive(Debug, thiserror::Error)]
pub enum RefineError {
    #[error(transparent)]
    Ensemble(#[from] reglyco_ensemble::EnsembleError),
    #[error(transparent)]
    Build(#[from] reglyco_core::ReGlycoError),
    #[error(transparent)]
    Relax(#[from] reglyco_relax::RelaxError),
    #[error(transparent)]
    GlySys(#[from] glysys::BuildError),
}

/// Live events emitted by [`refine_with_progress`].  The library remains
/// silent when callers use [`refine`], while command-line clients can render
/// these events without duplicating the search or relaxation loops.
#[derive(Debug, Clone)]
pub enum RefineProgress {
    Phase { name: &'static str },
    Search(SearchProgress),
    Relax(RelaxProgress),
}

#[derive(Debug, Clone)]
pub struct RefineRequest {
    pub protein: Structure,
    pub sites: Vec<SearchSite>,
    pub search: SearchConfig,
    pub relaxation: RelaxOptions,
}

#[derive(Debug, Clone)]
pub struct RefineResult {
    pub search: SearchOutcome,
    pub initial: BuildProduct,
    pub relaxed_structure: Structure,
    pub relaxed_system: Option<ParameterizedSystem>,
    pub final_system: Option<ParameterizedSystem>,
    pub pre_steric_score: f64,
    pub post_steric_score: f64,
    pub relaxation: Option<RelaxationDiagnostics>,
    pub report: WorkflowReport,
}

pub fn refine(
    request: RefineRequest,
    dry_builder: &SystemBuilder,
    final_builder: Option<&SystemBuilder>,
) -> Result<RefineResult> {
    refine_with_progress(request, dry_builder, final_builder, |_| {})
}

/// Search, build, parameterize, and relax a structure while reporting
/// progress.
pub fn refine_with_progress<F>(
    request: RefineRequest,
    dry_builder: &SystemBuilder,
    final_builder: Option<&SystemBuilder>,
    mut progress: F,
) -> Result<RefineResult>
where
    F: FnMut(RefineProgress),
{
    progress(RefineProgress::Phase {
        name: "ensemble search",
    });
    let search_outcome = search_with_progress(
        &request.protein,
        &request.sites,
        &request.search,
        dry_builder,
        |event| progress(RefineProgress::Search(event)),
    )?;
    progress(RefineProgress::Phase {
        name: "building initial structure",
    });
    let initial = build_from_outcome(
        &request.protein,
        &request.sites,
        &search_outcome,
        dry_builder,
        true,
    )?;
    let pre_steric_score = steric_score(&initial.structure, request.search.clash_distance);
    progress(RefineProgress::Phase {
        name: "parameterizing fitted structure",
    });
    let system = match initial.system.as_ref() {
        Some(system) => system.clone(),
        None => dry_builder.prepare_structure(&initial.structure)?,
    };
    progress(RefineProgress::Phase {
        name: "energy relaxation",
    });
    let relaxed = relax_with_progress(&initial.structure, &system, &request.relaxation, |event| {
        progress(RefineProgress::Relax(event))
    })?;
    let relaxed_structure = relaxed.structure;
    let relaxation_diagnostics = relaxed.diagnostics;
    let post_steric_score = steric_score(&relaxed_structure, request.search.clash_distance);
    progress(RefineProgress::Phase {
        name: "finalizing reports",
    });
    let final_system = final_builder
        .map(|builder| builder.prepare_structure(&relaxed_structure))
        .transpose()?;
    let report = WorkflowReport {
        status: if search_outcome.warnings.is_empty() {
            "complete".into()
        } else {
            "complete_with_warnings".into()
        },
        clash_status: Some(search_outcome.clash_status),
        search: Some(search_outcome.clone()),
        relaxation: Some(relaxation_diagnostics.clone()),
        provenance: Provenance {
            command: "refine".into(),
            seed: Some(request.search.seed),
            ensemble_sources: request
                .sites
                .iter()
                .map(|site| site.ensemble.provenance.clone())
                .collect(),
            ..Provenance::default()
        },
        warnings: search_outcome.warnings.clone(),
        diagnostics: vec![
            format!("pre_steric_score={pre_steric_score}"),
            format!("post_steric_score={post_steric_score}"),
        ],
        analysis: ReportAnalysis::default(),
    };
    Ok(RefineResult {
        search: search_outcome,
        initial,
        relaxed_structure,
        relaxed_system: Some(relaxed.system),
        final_system,
        pre_steric_score,
        post_steric_score,
        relaxation: Some(relaxation_diagnostics),
        report,
    })
}
