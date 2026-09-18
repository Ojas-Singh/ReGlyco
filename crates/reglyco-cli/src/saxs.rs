use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use crabsaxs::RankWeights;
use crabsaxs::{
    BackgroundMode, ExperimentalCurve, FitOptions, MaximumEntropyOptions, PrOptions, ScaleMode,
};
use glysys::SystemBuilder;
use reglyco_ensemble::sample_attached_ensemble;
use reglyco_report::{
    Provenance, ReportAnalysis, SaxsAnalysis, SaxsCandidateVisual, SaxsCombinationAnalysis,
    SaxsCombinationAssignment, WorkflowReport, snfg_svg_for_structure,
};
use reglyco_saxs::{
    NamedStructure, adapt_structure, assignment, candidate_combination, fit_single_models,
    fit_unbiased_ensemble, plot_for_ensemble, plot_for_single, rank_candidates, read_models,
    reweight_ensemble,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::{
    EnergyArgs, ProteinArgs, ProviderArgs, dry_options, energy_search_config, fetch_protein,
    load_ensemble, parse_site, query_for_source, write_json,
};

#[derive(Debug, Args)]
pub(crate) struct SaxsArgs {
    #[command(subcommand)]
    command: SaxsCommand,
}

#[derive(Debug, Subcommand)]
enum SaxsCommand {
    /// Select the single structure with the lowest fitted reduced χ².
    Model(SaxsModelArgs),
    /// Fit the equal-weight Re-Glyco ensemble without SAXS reweighting.
    Ensemble(SaxsEnsembleArgs),
    /// Maximum-entropy reweight an ensemble against the SAXS curve.
    Reweight(SaxsReweightArgs),
    /// Enumerate per-site glycoforms and occupancy alternatives.
    Occupancy(SaxsOccupancyArgs),
}

#[derive(Debug, Args)]
struct SaxsModelArgs {
    #[command(flatten)]
    common: SaxsCommonArgs,
}

#[derive(Debug, Args)]
struct SaxsEnsembleArgs {
    #[command(flatten)]
    common: SaxsCommonArgs,
}

#[derive(Debug, Args)]
struct SaxsReweightArgs {
    #[command(flatten)]
    common: SaxsCommonArgs,
    #[arg(long, default_value_t = 1.0)]
    kl_strength: f64,
    #[arg(long, default_value_t = 500)]
    max_iterations: usize,
    #[arg(long, default_value_t = 1.0e-8)]
    tolerance: f64,
    #[arg(long, default_value_t = 1.0e-10)]
    min_weight: f64,
}

#[derive(Debug, Args)]
struct SaxsOccupancyArgs {
    #[command(flatten)]
    common: SaxsCommonArgs,
    /// Repeat as `--candidate A:139=G47816DI,none`.
    #[arg(long = "candidate")]
    candidates: Vec<String>,
    /// JSON or TOML candidate manifest.
    #[arg(long)]
    candidate_manifest: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    max_combinations: usize,
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long, default_value_t = 1.0)]
    rank_chi2: f64,
    #[arg(long, default_value_t = 1.0)]
    rank_kratky: f64,
    #[arg(long, default_value_t = 1.0)]
    rank_rg: f64,
    #[arg(long, default_value_t = 1.0)]
    rank_dmax: f64,
    #[arg(long, default_value_t = 1.0)]
    rank_pr: f64,
}

#[derive(Debug, Args)]
struct SaxsCommonArgs {
    /// Experimental SAXS `.dat` curve.
    #[arg(long)]
    data: PathBuf,
    /// Existing single- or multi-model PDB/mmCIF input.
    #[arg(long)]
    models: Option<PathBuf>,
    #[command(flatten)]
    protein: ProteinArgs,
    /// Repeat as `--attach A:139=G47816DI` when generating Re-Glyco frames.
    #[arg(long = "attach")]
    attachments: Vec<String>,
    #[arg(long, default_value_t = 50)]
    frames: usize,
    #[arg(long, default_value_t = 4)]
    chains: usize,
    #[arg(long, default_value_t = 250)]
    burn_in_sweeps: usize,
    #[arg(long, default_value_t = 50)]
    thinning_accepted: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long = "move-sidechains")]
    move_sidechains: bool,
    #[command(flatten)]
    energy: EnergyArgs,
    #[command(flatten)]
    provider: ProviderArgs,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    report: bool,
    #[arg(long)]
    quiet: bool,
    #[arg(long, value_enum, default_value_t = SaxsMethod::Auto)]
    method: SaxsMethod,
    #[arg(long, default_value_t = 20)]
    q_sampling_stride: usize,
    #[arg(long)]
    fixed_scale: Option<f64>,
    #[arg(long)]
    zero_background: bool,
    #[arg(long, default_value_t = 160)]
    pr_bins: usize,
    #[arg(long)]
    pr_max_r: Option<f64>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SaxsMethod {
    Auto,
    Debye,
    Multipole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateSite {
    site: String,
    candidates: Vec<CandidateOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateOption {
    id: String,
    #[serde(default = "default_prior")]
    prior: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CandidateManifest {
    sites: Vec<CandidateSite>,
}

#[derive(Debug, Clone)]
struct CombinationRun {
    assignments: Vec<reglyco_saxs::Assignment>,
    models: Vec<NamedStructure>,
    ensemble: reglyco_saxs::UnbiasedEnsembleResult,
    prior: f64,
}

fn default_prior() -> f64 {
    1.0
}

pub(crate) fn run(arguments: SaxsArgs) -> Result<()> {
    match arguments.command {
        SaxsCommand::Model(arguments) => run_model(arguments),
        SaxsCommand::Ensemble(arguments) => run_ensemble(arguments),
        SaxsCommand::Reweight(arguments) => run_reweight(arguments),
        SaxsCommand::Occupancy(arguments) => run_occupancy(arguments),
    }
}

fn run_model(arguments: SaxsModelArgs) -> Result<()> {
    let started = Instant::now();
    let common = arguments.common;
    let experimental = read_experimental(&common)?;
    let candidate_visuals = candidate_visuals_for_attachments(&common)?;
    let models = load_models(&common, None)?;
    super::prepare_output(&common.output)?;
    let fit = fit_single_models(
        &models,
        &experimental,
        fit_options(&common)?,
        pr_options(&common),
    )?;
    let best = models
        .get(fit.best_index)
        .ok_or_else(|| anyhow::anyhow!("single-model fit returned an invalid best index"))?;
    reglyco_saxs::write_pdb(&best.structure, common.output.join("best-model.pdb"))?;
    write_json(common.output.join("saxs-model.json"), &fit)?;
    let plot = plot_for_single(&experimental, &fit, true);
    let svg = save_plot(&plot, &common.output)?;
    let mut report = base_report("saxs model", &common, &[]);
    annotate_report_glycan_identity(&mut report, std::slice::from_ref(best), &common);
    let mut saxs = single_analysis(&common.data, &fit);
    saxs.candidate_visuals = candidate_visuals;
    report.analysis.saxs = Some(saxs);
    set_report_plot(&mut report, svg);
    write_saxs_workflow_report(&mut report, &common.output, common.report, started)?;
    if !common.quiet {
        println!(
            "Selected {} as the best single SAXS model in {}",
            best.name,
            common.output.display()
        );
    }
    Ok(())
}

fn run_ensemble(arguments: SaxsEnsembleArgs) -> Result<()> {
    let started = Instant::now();
    let common = arguments.common;
    let experimental = read_experimental(&common)?;
    let candidate_visuals = candidate_visuals_for_attachments(&common)?;
    let models = load_models(&common, None)?;
    super::prepare_output(&common.output)?;
    let fit = fit_unbiased_ensemble(
        &models,
        &experimental,
        fit_options(&common)?,
        pr_options(&common),
    )?;
    crabsaxs::io::write_curve_dat(
        common.output.join("unbiased-fit.dat"),
        &experimental.q,
        &fit.fitted_curve,
    )?;
    fs::write(
        common.output.join("ensemble.pdb"),
        reglyco_saxs::multi_model_pdb(&models),
    )?;
    write_json(common.output.join("saxs-ensemble.json"), &fit)?;
    let mut plot = plot_for_ensemble(
        &experimental,
        &fit.fitted_curve,
        None,
        &fit.experimental,
        &fit.features,
        Some(fit.model_pr.clone()),
    );
    plot.effective_sample_size = Some(
        1.0 / fit
            .weights
            .iter()
            .map(|weight| weight * weight)
            .sum::<f64>(),
    );
    let svg = save_plot(&plot, &common.output)?;
    let mut report = base_report("saxs ensemble", &common, &[]);
    annotate_report_glycan_identity(&mut report, &models, &common);
    let mut saxs = ensemble_analysis(&common.data, &fit);
    saxs.candidate_visuals = candidate_visuals;
    report.analysis.saxs = Some(saxs);
    set_report_plot(&mut report, svg);
    write_saxs_workflow_report(&mut report, &common.output, common.report, started)?;
    if !common.quiet {
        println!(
            "Fitted the unbiased ensemble in {}",
            common.output.display()
        );
    }
    Ok(())
}

fn run_reweight(arguments: SaxsReweightArgs) -> Result<()> {
    let started = Instant::now();
    let common = arguments.common;
    let experimental = read_experimental(&common)?;
    let candidate_visuals = candidate_visuals_for_attachments(&common)?;
    let models = load_models(&common, None)?;
    super::prepare_output(&common.output)?;
    let mut options = MaximumEntropyOptions::default();
    options.kl_strength = arguments.kl_strength;
    options.max_iterations = arguments.max_iterations;
    options.tolerance = arguments.tolerance;
    options.min_weight = arguments.min_weight;
    let fit = reweight_ensemble(
        &models,
        &experimental,
        fit_options(&common)?,
        options,
        pr_options(&common),
    )?;
    crabsaxs::io::write_curve_dat(
        common.output.join("reweighted-fit.dat"),
        &experimental.q,
        &fit.fitted_curve,
    )?;
    fs::write(
        common.output.join("ensemble.pdb"),
        reglyco_saxs::multi_model_pdb(&models),
    )?;
    write_json(common.output.join("saxs-reweight.json"), &fit)?;
    write_weights_csv(
        &common.output.join("weights.csv"),
        &fit.names,
        &fit.weights,
        &fit.prior_weights,
    )?;
    let unbiased = uniform_curve(&fit.conformer_curves);
    let mut plot = plot_for_ensemble(
        &experimental,
        &unbiased,
        Some(&fit.fitted_curve),
        &fit.experimental,
        &fit.features,
        Some(fit.model_pr.clone()),
    );
    plot.effective_sample_size = Some(fit.effective_sample_size);
    let svg = save_plot(&plot, &common.output)?;
    let mut report = base_report("saxs reweight", &common, &[]);
    annotate_report_glycan_identity(&mut report, &models, &common);
    let mut saxs = reweight_analysis(&common.data, &fit);
    saxs.candidate_visuals = candidate_visuals;
    report.analysis.saxs = Some(saxs);
    set_report_plot(&mut report, svg);
    write_saxs_workflow_report(&mut report, &common.output, common.report, started)?;
    if !common.quiet {
        println!(
            "Maximum-entropy reweighted ensemble written to {} (ESS {:.3})",
            common.output.display(),
            fit.effective_sample_size
        );
    }
    Ok(())
}

fn run_occupancy(arguments: SaxsOccupancyArgs) -> Result<()> {
    let started = Instant::now();
    let sites = read_candidate_sites(
        &arguments.candidates,
        arguments.candidate_manifest.as_deref(),
    )?;
    let common = arguments.common;
    if common.models.is_some() {
        anyhow::bail!(
            "saxs occupancy requires --protein/--uniprot/--pdb-id so each candidate combination can be built"
        );
    }
    let experimental = read_experimental(&common)?;
    let candidate_visuals = candidate_visuals_for_sources(
        sites
            .iter()
            .flat_map(|site| site.candidates.iter().map(|candidate| candidate.id.clone())),
        &common.provider,
    )?;
    let combinations = enumerate_combinations(&sites, arguments.max_combinations)?;
    super::prepare_output(&common.output)?;
    if !common.quiet {
        eprintln!(
            "saxs occupancy: evaluating {} glycoform combination(s)...",
            combinations.len()
        );
    }
    let mut runs = Vec::with_capacity(combinations.len());
    let fit_options = fit_options(&common)?;
    let pr_options = pr_options(&common);
    for (assignments, prior) in combinations {
        let models = generate_models(&common, &assignments)?;
        let ensemble = fit_unbiased_ensemble(&models, &experimental, fit_options, pr_options)?;
        runs.push(CombinationRun {
            assignments: assignments
                .iter()
                .map(|(site, candidate)| assignment(site.clone(), candidate.clone()))
                .collect(),
            models,
            ensemble,
            prior,
        });
    }
    let candidate_combinations = runs
        .iter()
        .map(|run| {
            candidate_combination(
                run.assignments.clone(),
                run.prior,
                run.ensemble.features.clone(),
            )
        })
        .collect::<Vec<_>>();
    let analysis = rank_candidates(
        &candidate_combinations,
        RankWeights {
            chi2: arguments.rank_chi2,
            kratky: arguments.rank_kratky,
            rg: arguments.rank_rg,
            dmax: arguments.rank_dmax,
            pr: arguments.rank_pr,
        },
    )?;
    write_json(common.output.join("saxs-occupancy.json"), &analysis)?;
    write_combinations_csv(&common.output.join("ranked-combinations.csv"), &analysis)?;
    let best = analysis
        .combinations
        .first()
        .ok_or_else(|| anyhow::anyhow!("candidate analysis returned no combinations"))?;
    let best_run = runs
        .iter()
        .find(|run| run.assignments == best.assignments)
        .ok_or_else(|| anyhow::anyhow!("could not locate the robust winning combination"))?;
    fs::write(
        common.output.join("best-combination.pdb"),
        reglyco_saxs::multi_model_pdb(&best_run.models),
    )?;
    let mut plot = plot_for_ensemble(
        &experimental,
        &best_run.ensemble.fitted_curve,
        None,
        &best_run.ensemble.experimental,
        &best_run.ensemble.features,
        Some(best_run.ensemble.model_pr.clone()),
    );
    plot.effective_sample_size = Some(
        1.0 / best_run
            .ensemble
            .weights
            .iter()
            .map(|weight| weight * weight)
            .sum::<f64>(),
    );
    let svg = save_plot(&plot, &common.output)?;
    let mut report = base_report("saxs occupancy", &common, &[]);
    annotate_report_glycan_identity(&mut report, &best_run.models, &common);
    let mut report_saxs = occupancy_analysis(&common.data, &analysis);
    report_saxs.combinations.truncate(arguments.top);
    report_saxs.candidate_visuals = candidate_visuals;
    report.analysis.saxs = Some(report_saxs);
    set_report_plot(&mut report, svg);
    write_saxs_workflow_report(&mut report, &common.output, common.report, started)?;
    if !common.quiet {
        println!(
            "Robust glycoform combination written to {}",
            common.output.display()
        );
    }
    Ok(())
}

fn read_experimental(common: &SaxsCommonArgs) -> Result<ExperimentalCurve> {
    Ok(ExperimentalCurve::from_dat_file(&common.data)
        .with_context(|| format!("could not read SAXS data {}", common.data.display()))?)
}

fn fit_options(common: &SaxsCommonArgs) -> Result<FitOptions> {
    if common.q_sampling_stride == 0 {
        anyhow::bail!("--q-sampling-stride must be positive");
    }
    let mut options = FitOptions::default();
    options.method = match common.method {
        SaxsMethod::Auto => crabsaxs::ScatteringMethod::Auto,
        SaxsMethod::Debye => crabsaxs::ScatteringMethod::Debye,
        SaxsMethod::Multipole => crabsaxs::ScatteringMethod::Multipole,
    };
    options.q_sampling_stride = common.q_sampling_stride;
    if let Some(scale) = common.fixed_scale {
        if !scale.is_finite() || scale < 0.0 {
            anyhow::bail!("--fixed-scale must be finite and non-negative");
        }
        options.scale_mode = ScaleMode::Fixed(scale);
    }
    if common.zero_background {
        options.background_mode = BackgroundMode::FixedZero;
    }
    Ok(options)
}

fn pr_options(common: &SaxsCommonArgs) -> PrOptions {
    PrOptions {
        bins: common.pr_bins,
        max_r: common.pr_max_r,
    }
}

fn candidate_visuals_for_sources<I>(
    sources: I,
    provider: &ProviderArgs,
) -> Result<Vec<SaxsCandidateVisual>>
where
    I: IntoIterator<Item = String>,
{
    let candidates = sources.into_iter().collect::<BTreeSet<_>>();
    candidates
        .into_iter()
        .map(|candidate| {
            let snfg_svg = if is_none_candidate(&candidate) {
                None
            } else {
                let query = query_for_source(&candidate, provider);
                let ensemble = load_ensemble(&query, provider)?;
                ensemble
                    .conformers
                    .first()
                    .and_then(|conformer| snfg_svg_for_structure(&conformer.structure))
            };
            Ok(SaxsCandidateVisual {
                candidate,
                snfg_asset: None,
                snfg_svg,
            })
        })
        .collect()
}

fn candidate_visuals_for_attachments(common: &SaxsCommonArgs) -> Result<Vec<SaxsCandidateVisual>> {
    if common.models.is_some() {
        return Ok(Vec::new());
    }
    let attachments = parse_attachments(&common.attachments)?;
    candidate_visuals_for_sources(
        attachments.into_iter().map(|(_, candidate)| candidate),
        &common.provider,
    )
}

fn load_models(
    common: &SaxsCommonArgs,
    assignments: Option<&[(String, String)]>,
) -> Result<Vec<NamedStructure>> {
    if let Some(path) = &common.models {
        if assignments.is_some() {
            anyhow::bail!(
                "model-file input cannot be combined with generated candidate assignments"
            );
        }
        return Ok(read_models(path)?);
    }
    let assignments = if let Some(assignments) = assignments {
        assignments.to_owned()
    } else {
        parse_attachments(&common.attachments)?
    };
    generate_models(common, &assignments)
}

fn generate_models(
    common: &SaxsCommonArgs,
    assignments: &[(String, String)],
) -> Result<Vec<NamedStructure>> {
    let has_protein = common.protein.protein.is_some()
        || common.protein.uniprot.is_some()
        || common.protein.pdb_id.is_some();
    if !has_protein {
        anyhow::bail!("generated SAXS input requires one of --protein, --uniprot, or --pdb-id");
    }
    if common.frames == 0 {
        anyhow::bail!("--frames must be positive");
    }
    let protein = fetch_protein(&common.protein, &dry_options(false))?.structure;
    let attached_specifications = assignments
        .iter()
        .filter(|(_, candidate)| !is_none_candidate(candidate))
        .map(|(site, source)| format!("{site}={source}"))
        .collect::<Vec<_>>();
    let attached = super::load_search_sites(&attached_specifications, &common.provider)?;
    if attached.is_empty() {
        return Ok(vec![adapt_structure("protein", &protein)?]);
    }
    let builder = SystemBuilder::new(dry_options(false))?;
    let mut config = energy_search_config(&common.energy)?;
    config.seed = common.seed;
    config.ensemble_size = common.frames;
    config.scan_rotamers = common.move_sidechains;
    config.mh_chains = common.chains;
    config.mh_burn_in_sweeps = common.burn_in_sweeps;
    config.mh_thinning_accepted = common.thinning_accepted;
    let (frames, _diagnostics) =
        sample_attached_ensemble(&protein, &attached, common.frames, &config, &builder)?;
    if frames.len() != common.frames {
        anyhow::bail!(
            "Re-Glyco returned {} frame(s), expected {}",
            frames.len(),
            common.frames
        );
    }
    frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let mut model = adapt_structure(format!("frame-{}", index + 1), &frame.structure)
                .map_err(anyhow::Error::from)?;
            model.log_native_probability = Some(frame.log_native_probability);
            Ok(model)
        })
        .collect()
}

fn parse_attachments(values: &[String]) -> Result<Vec<(String, String)>> {
    let mut seen = std::collections::BTreeSet::new();
    values
        .iter()
        .map(|value| {
            let (site, candidate) = value
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--attach must use SITE=GLYCAN syntax"))?;
            let site = site.trim();
            let candidate = candidate.trim();
            if site.is_empty() || candidate.is_empty() {
                anyhow::bail!("--attach must contain a site and glycan ID");
            }
            parse_site(site)?;
            if !seen.insert(site.to_owned()) {
                anyhow::bail!("--attach contains duplicate site {site}");
            }
            Ok((site.to_owned(), candidate.to_owned()))
        })
        .collect()
}

fn read_candidate_sites(
    shorthand_values: &[String],
    manifest_path: Option<&Path>,
) -> Result<Vec<CandidateSite>> {
    let mut sites = if let Some(path) = manifest_path {
        let text = fs::read_to_string(path)?;
        if path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("toml"))
        {
            toml::from_str::<CandidateManifest>(&text)?
        } else {
            serde_json::from_str::<CandidateManifest>(&text)?
        }
        .sites
    } else {
        Vec::new()
    };
    for shorthand in shorthand_values {
        let (site, values) = shorthand
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--candidate must use SITE=ID[,ID...] syntax"))?;
        let candidates = values
            .split(',')
            .filter(|value| !value.trim().is_empty())
            .map(|id| CandidateOption {
                id: id.trim().into(),
                prior: 1.0,
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            anyhow::bail!("--candidate {site} has no candidate IDs");
        }
        sites.push(CandidateSite {
            site: site.into(),
            candidates,
        });
    }
    if sites.is_empty() {
        anyhow::bail!("provide --candidate values or --candidate-manifest");
    }
    let mut seen = std::collections::BTreeSet::new();
    for site in &sites {
        parse_site(&site.site)?;
        if site.candidates.is_empty() || !seen.insert(site.site.clone()) {
            anyhow::bail!("candidate sites must be unique and non-empty");
        }
        if site
            .candidates
            .iter()
            .any(|candidate| !candidate.prior.is_finite() || candidate.prior < 0.0)
        {
            anyhow::bail!("candidate priors must be finite and non-negative");
        }
    }
    Ok(sites)
}

fn enumerate_combinations(
    sites: &[CandidateSite],
    maximum: usize,
) -> Result<Vec<(Vec<(String, String)>, f64)>> {
    if maximum == 0 {
        anyhow::bail!("--max-combinations must be positive");
    }
    let mut count = 1usize;
    for site in sites {
        count = count
            .checked_mul(site.candidates.len())
            .ok_or_else(|| anyhow::anyhow!("candidate combination count overflowed"))?;
        if count > maximum {
            anyhow::bail!(
                "candidate search has at least {count} combinations; increase --max-combinations"
            );
        }
    }
    let mut output = Vec::with_capacity(count);
    enumerate_recursive(sites, 0, &mut Vec::new(), 1.0, &mut output);
    Ok(output)
}

fn enumerate_recursive(
    sites: &[CandidateSite],
    index: usize,
    current: &mut Vec<(String, String)>,
    prior: f64,
    output: &mut Vec<(Vec<(String, String)>, f64)>,
) {
    if index == sites.len() {
        output.push((current.clone(), prior));
        return;
    }
    let site = &sites[index];
    for candidate in &site.candidates {
        current.push((site.site.clone(), candidate.id.clone()));
        enumerate_recursive(sites, index + 1, current, prior * candidate.prior, output);
        current.pop();
    }
}

fn is_none_candidate(candidate: &str) -> bool {
    matches!(
        candidate.trim().to_ascii_lowercase().as_str(),
        "none" | "no_glycan" | "noglycan" | "absent"
    )
}

fn uniform_curve(curves: &[Vec<f64>]) -> Vec<f64> {
    let count = curves.len().max(1) as f64;
    (0..curves.first().map_or(0, Vec::len))
        .map(|index| curves.iter().map(|curve| curve[index]).sum::<f64>() / count)
        .collect()
}

fn save_plot(plot: &crabsaxs::DiagnosticPlot, output: &Path) -> Result<String> {
    let path = output.join("saxs-diagnostic.svg");
    plot.save_svg(&path)?;
    Ok(fs::read_to_string(path)?)
}

fn base_report(mode: &str, common: &SaxsCommonArgs, sources: &[String]) -> WorkflowReport {
    WorkflowReport {
        status: "complete".into(),
        clash_status: Some(reglyco_core::ClashStatus::ClashFree),
        search: None,
        relaxation: None,
        provenance: Provenance {
            command: mode.into(),
            seed: Some(common.seed),
            ensemble_sources: sources.to_vec(),
            ..Provenance::default()
        },
        warnings: Vec::new(),
        diagnostics: Vec::new(),
        analysis: ReportAnalysis::default(),
    }
}

fn annotate_report_glycan_identity(
    report: &mut WorkflowReport,
    models: &[NamedStructure],
    common: &SaxsCommonArgs,
) {
    if let Some(structure) = models
        .first()
        .and_then(|model| model.source_structure.as_ref())
    {
        report.analyze_glycan_identity(structure);
        return;
    }

    // Supplied model files are parsed by crabSAXS for fitting. Reuse the
    // first PDB model through GlySys for report metadata when possible; a
    // report-only parsing failure must not invalidate an otherwise valid SAXS
    // fit.
    if let Some(path) = common.models.as_deref()
        && let Ok(structure) = glysys::read_pdb(path, &dry_options(false))
    {
        report.analyze_glycan_identity(&structure);
    }
}

fn write_saxs_workflow_report(
    report: &mut WorkflowReport,
    output: &Path,
    render_pdf: bool,
    started: Instant,
) -> Result<()> {
    report.provenance.total_seconds = Some(started.elapsed().as_secs_f64());
    super::write_workflow_report(report, output, render_pdf)
}

fn set_report_plot(report: &mut WorkflowReport, svg: String) {
    if let Some(saxs) = report.analysis.saxs.as_mut() {
        saxs.diagnostic_svg = Some(svg);
    }
}

fn single_analysis(path: &Path, fit: &reglyco_saxs::SingleModelResult) -> SaxsAnalysis {
    SaxsAnalysis {
        mode: "single_model".into(),
        data_path: Some(path.display().to_string()),
        experimental_rg: fit.experimental.rg,
        experimental_dmax: fit.experimental.p_r.as_ref().and_then(|value| value.dmax),
        model_fits: fit
            .models
            .iter()
            .map(|model| model.features.clone())
            .collect(),
        model_names: fit.models.iter().map(|model| model.name.clone()).collect(),
        fit_features: fit
            .models
            .get(fit.best_index)
            .map(|model| model.features.clone()),
        ..SaxsAnalysis::default()
    }
}

fn ensemble_analysis(path: &Path, fit: &reglyco_saxs::UnbiasedEnsembleResult) -> SaxsAnalysis {
    SaxsAnalysis {
        mode: "unbiased_ensemble".into(),
        data_path: Some(path.display().to_string()),
        experimental_rg: fit.experimental.rg,
        experimental_dmax: fit.experimental.p_r.as_ref().and_then(|value| value.dmax),
        model_fits: vec![fit.features.clone()],
        model_names: vec!["unbiased ensemble".into()],
        fit_features: Some(fit.features.clone()),
        weights: fit.weights.clone(),
        prior_weights: fit.weights.clone(),
        native_log_probabilities: fit.native_log_probabilities.clone(),
        effective_sample_size: Some(
            1.0 / fit
                .weights
                .iter()
                .map(|weight| weight * weight)
                .sum::<f64>(),
        ),
        ..SaxsAnalysis::default()
    }
}

fn reweight_analysis(path: &Path, fit: &reglyco_saxs::ReweightFitResult) -> SaxsAnalysis {
    SaxsAnalysis {
        mode: "maximum_entropy_reweight".into(),
        data_path: Some(path.display().to_string()),
        experimental_rg: fit.experimental.rg,
        experimental_dmax: fit.experimental.p_r.as_ref().and_then(|value| value.dmax),
        model_names: fit.names.clone(),
        fit_features: Some(fit.features.clone()),
        weights: fit.weights.clone(),
        prior_weights: fit.prior_weights.clone(),
        native_log_probabilities: fit.native_log_probabilities.clone(),
        kl_divergence: Some(fit.kl_divergence),
        effective_sample_size: Some(fit.effective_sample_size),
        ..SaxsAnalysis::default()
    }
}

fn occupancy_analysis(path: &Path, analysis: &crabsaxs::CombinationAnalysis) -> SaxsAnalysis {
    SaxsAnalysis {
        mode: "occupancy_glycoform_search".into(),
        data_path: Some(path.display().to_string()),
        combinations: analysis
            .combinations
            .iter()
            .map(|combination| SaxsCombinationAnalysis {
                assignments: combination
                    .assignments
                    .iter()
                    .map(|assignment| SaxsCombinationAssignment {
                        site: assignment.site.clone(),
                        candidate: assignment.candidate.clone(),
                    })
                    .collect(),
                prior: combination.prior,
                log_likelihood: combination.log_likelihood,
                likelihood: combination.likelihood,
                posterior: combination.posterior,
                robust_score: combination.robust_score,
                features: combination.features.clone(),
            })
            .collect(),
        site_occupancy: analysis.site_occupancy.clone(),
        glycoform_distribution: analysis.glycoform_distribution.clone(),
        fit_features: analysis
            .combinations
            .first()
            .map(|combination| combination.features.clone()),
        ..SaxsAnalysis::default()
    }
}

fn write_weights_csv(path: &Path, names: &[String], weights: &[f64], priors: &[f64]) -> Result<()> {
    let mut text = String::from("name,prior_weight,posterior_weight\n");
    for index in 0..weights.len() {
        text.push_str(&format!(
            "{},{:.12},{:.12}\n",
            names.get(index).map(String::as_str).unwrap_or("conformer"),
            priors.get(index).copied().unwrap_or(0.0),
            weights[index]
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_combinations_csv(path: &Path, analysis: &crabsaxs::CombinationAnalysis) -> Result<()> {
    let mut text = String::from(
        "rank,combination,log_likelihood,likelihood,posterior,robust_score,reduced_chi2\n",
    );
    for (index, combination) in analysis.combinations.iter().enumerate() {
        let label = combination
            .assignments
            .iter()
            .map(|assignment| format!("{}={}", assignment.site, assignment.candidate))
            .collect::<Vec<_>>()
            .join(";");
        let log_likelihood = combination
            .log_likelihood
            .map_or_else(String::new, |value| format!("{value:.12}"));
        text.push_str(&format!(
            "{},{},{},{:.12},{:.12},{:.12},{:.12}\n",
            index + 1,
            label,
            log_likelihood,
            combination.likelihood,
            combination.posterior,
            combination.robust_score,
            combination.features.reduced_chi2
        ));
    }
    fs::write(path, text)?;
    Ok(())
}
