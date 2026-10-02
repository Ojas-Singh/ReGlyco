use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use crabwurcs_core::{AnomericSymbol, parse_wurcs, write_wurcs_canonical};
use crabwurcs_iupac::write_iupac_condensed_canonical;
use crabwurcs_pdb::extract_glycans_with_provenance_from_str;
use glysys::{BuildOptions, ResidueId, SystemBuilder, read_pdb, write_pdb};
use reglyco_build::{
    FetchedProtein, LinkageDefinition, ProteinFetchOptions, ProteinProvider, ProteinSource,
    remove_glycan_at_site, scan_n_linked_sequons,
};
use reglyco_core::{
    Anomer, GlycanQuery, GlycanSource, GlycosylationSite, ResidueNameFormat, SearchBudgetMode,
    SearchBudgetResolution, SearchConfig, SearchScoringMode, SearchSelectionPolicy, SearchSite,
};
use reglyco_density::rcsb::{RcsbMapAcquisition, fetch_2fo_fc_map};
use reglyco_density::{DensityMap, DensityScoreOptions, DensityScorer, DensityTarget};
use reglyco_ensemble::{
    CachingProvider, EnsembleProvider, GlycoShapeProvider, LocalBundleProvider, SearchPhase,
    SearchProgress, build_from_outcome, calculate_sasa, configure_threads, resolve_search_budget,
    sample_attached_ensemble, search, search_with_progress, steric_site_scores,
};
use reglyco_refine::{RefineProgress, RefineRequest, refine_with_progress};
use reglyco_relax::{MovableSelection, RelaxOptions, RelaxProgress, relax_with_progress};
use reglyco_report::{
    ClusterObservation, DensityAnalysis, DensitySiteAnalysis, EnsembleAnalysis, Provenance,
    ReportAnalysis, SamplingSegmentAnalysis, ScanAnalysis, ScanSequon, ValidationAnalysis,
    WorkflowReport,
};
use sha2::{Digest, Sha256};

mod glycoflow;
mod saxs;

#[derive(Debug, Parser)]
#[command(
    name = "reglyco",
    version,
    about = "Build, search, and refine glycoproteins in native Rust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Attach selected glycan conformers to explicit protein sites.
    Build(BuildArgs),
    /// Fetch or inspect a local/remote glycan ensemble.
    Ensemble(EnsembleArgs),
    /// Select a compatible multi-site conformer set.
    Search(SearchArgs),
    /// Minimize an already assembled structure in vacuum or with opt-in OBC2 GBSA.
    Relax(RelaxArgs),
    /// Run search, build, parameterization, and relaxation (`--objective steric`), or fit the
    /// glycans at `--replace-glycan` sites into a density map with GlycoFlow
    /// (`--objective density`).
    Refine(RefineArgs),
    /// Attach GlcNAc to every unoccupied N-X-S/T sequon.
    Scan(ScanArgs),
    /// Validate structure and attachment metadata.
    Validate(ValidateArgs),
    /// Score one or more carbohydrate trees against a CCP4/MRC density map.
    Density(DensityArgs),
    /// Fit glycoprotein models and ensembles against an experimental SAXS curve.
    Saxs(saxs::SaxsArgs),
}

/// Residue-name convention used for remote glycan structure assets.  The
/// output remains a PDB container in either mode; this only selects the
/// provider representation that is carried into the generated structure.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormatArg {
    Pdb,
    Glycam,
}

impl OutputFormatArg {
    fn residue_name_format(self) -> ResidueNameFormat {
        match self {
            Self::Pdb => ResidueNameFormat::Pdb,
            Self::Glycam => ResidueNameFormat::Glycam,
        }
    }

    fn api_segment(self) -> &'static str {
        self.residue_name_format().api_segment()
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SearchBudgetArg {
    Auto,
    Manual,
}

impl SearchBudgetArg {
    fn mode(self) -> SearchBudgetMode {
        match self {
            Self::Auto => SearchBudgetMode::Auto,
            Self::Manual => SearchBudgetMode::Manual,
        }
    }
}

impl std::fmt::Display for OutputFormatArg {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Pdb => "pdb",
            Self::Glycam => "glycam",
        })
    }
}

#[derive(Debug, Args)]
struct BuildArgs {
    #[command(flatten)]
    protein: ProteinArgs,
    #[arg(long = "site")]
    sites: Vec<String>,
    #[arg(long = "glycan")]
    glycans: Vec<String>,
    /// Repeat as `--attach A:42=G00028MO` or `--attach A:42=path/to/bundle`.
    #[arg(long = "attach")]
    attachments: Vec<String>,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long)]
    no_system: bool,
    #[arg(long)]
    no_water: bool,
    #[arg(long)]
    no_ions: bool,
    #[arg(long)]
    overwrite: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long = "output-format", value_enum, default_value_t = OutputFormatArg::Pdb)]
    output_format: OutputFormatArg,
    /// Select a search budget automatically from the loaded site/conformer
    /// conflict graph, or provide explicit population/generation values.
    #[arg(long = "search-budget", value_enum)]
    search_budget: Option<SearchBudgetArg>,
    #[arg(long)]
    population: Option<usize>,
    #[arg(long)]
    generations: Option<usize>,
    /// Number of worker threads for prepared scoring and pose preparation.
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    rotamers: bool,
    #[arg(long)]
    allow_clashes: bool,
    /// Return a nonzero status after writing artifacts unless the selected
    /// Build is clash-free.  The diagnostic PDB/report are always preserved.
    #[arg(long, conflicts_with = "allow_clashes")]
    require_clash_free: bool,
    /// Suppress progress messages on stderr.
    #[arg(long)]
    quiet: bool,
    #[command(flatten)]
    energy: EnergyArgs,
    /// Render report.pdf, report.typ, and SVG figures alongside report.json.
    #[arg(long)]
    report: bool,
    #[command(flatten)]
    provider: ProviderArgs,
}

#[derive(Debug, Clone, Args)]
struct ProteinArgs {
    /// Local protein PDB.
    #[arg(long, conflicts_with_all = ["uniprot", "pdb_id"])]
    protein: Option<PathBuf>,
    /// UniProt accession resolved through AlphaFold DB.
    #[arg(long, conflicts_with_all = ["protein", "pdb_id"])]
    uniprot: Option<String>,
    /// Experimental structure identifier downloaded from RCSB PDB.
    #[arg(long = "pdb-id", conflicts_with_all = ["protein", "uniprot"])]
    pdb_id: Option<String>,
    /// Biological assembly number for an RCSB PDB source.
    #[arg(long, requires = "pdb_id")]
    assembly: Option<u32>,
    /// Use the asymmetric unit instead of the default biological assembly 1.
    #[arg(long, conflicts_with = "assembly", requires = "pdb_id")]
    asymmetric_unit: bool,
    #[arg(long, default_value = ".reglyco-cache/proteins")]
    protein_cache: PathBuf,
    #[arg(long)]
    protein_offline: bool,
}

#[derive(Debug, Args)]
struct EnsembleArgs {
    #[command(flatten)]
    protein: ProteinArgs,
    /// GlyTouCan identifier or local bundle/PDB path.
    #[arg(long)]
    glycan: Option<String>,
    /// Attached mode: repeat as `--attach A:42=G00028MO`.
    #[arg(long = "attach")]
    attachments: Vec<String>,
    #[arg(long, default_value_t = 50)]
    frames: usize,
    /// Number of independent constrained-MH chains used if native sampling
    /// cannot fill the requested frame count.
    #[arg(long, default_value_t = 4)]
    chains: usize,
    /// Burn-in sweeps per fallback chain (one sweep visits every site).
    #[arg(long, default_value_t = 250)]
    burn_in_sweeps: usize,
    /// Accepted native proposals between fallback frames.
    #[arg(long, default_value_t = 50)]
    thinning_accepted: usize,
    /// Write cookbook-compatible sasa.pdb and real_sasa.pdb files for an
    /// attached ensemble.
    #[arg(long = "calculate-sasa", visible_alias = "sasa")]
    calculate_sasa: bool,
    /// Also write the cookbook-compatible hotspots.pdb binder-design mask.
    #[arg(
        long = "calculate-hotspots",
        visible_alias = "hotspots",
        requires = "calculate_sasa"
    )]
    calculate_hotspots: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Select a search budget automatically from the loaded site/conformer
    /// conflict graph, or provide explicit population/generation values.
    #[arg(long = "search-budget", value_enum)]
    search_budget: Option<SearchBudgetArg>,
    #[arg(long)]
    population: Option<usize>,
    #[arg(long)]
    generations: Option<usize>,
    #[arg(long = "output-format", value_enum, default_value_t = OutputFormatArg::Pdb)]
    output_format: OutputFormatArg,
    /// Number of worker threads for prepared scoring and pose preparation.
    #[arg(long)]
    threads: Option<usize>,
    /// Allow local protein-sidechain rotamer changes while sampling. By
    /// default ensemble sampling leaves every protein atom fixed.
    #[arg(long = "move-sidechains", visible_alias = "rotamers")]
    move_sidechains: bool,
    #[command(flatten)]
    energy: EnergyArgs,
    #[arg(short, long)]
    output: PathBuf,
    /// Render report.pdf, report.typ, and SVG figures alongside report.json.
    #[arg(long)]
    report: bool,
    #[arg(long)]
    quiet: bool,
    #[command(flatten)]
    provider: ProviderArgs,
}

#[derive(Debug, Clone, Args)]
struct ProviderArgs {
    /// GlycoShape API to fetch glycan ensembles from.
    #[arg(long, default_value = "https://glycoshape.org")]
    api_base: String,
    #[arg(long, default_value = ".reglyco-cache")]
    cache: PathBuf,
    #[arg(long)]
    offline: bool,
    #[arg(long, default_value = "beta")]
    anomer: String,
    /// GlycoShape dataset level. glycoshape.org serves Level 1 to
    /// command-line clients; Levels 2 and 3 are not available from it.
    #[arg(long, default_value = "1")]
    level: String,
}

#[derive(Debug, Clone, Args)]
struct SearchCommon {
    #[command(flatten)]
    protein: ProteinArgs,
    /// Repeat as `--attach A:42=G00028MO` or `--attach A:42=path/to/bundle`.
    #[arg(long = "attach")]
    attachments: Vec<String>,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 128)]
    population: usize,
    #[arg(long, default_value_t = 100)]
    generations: usize,
    /// Write the best complete result even when the GA cannot remove all clashes.
    #[arg(long)]
    allow_clashes: bool,
    #[arg(long)]
    rotamers: bool,
    #[arg(long)]
    quiet: bool,
    #[command(flatten)]
    energy: EnergyArgs,
    /// Render report.pdf, report.typ, and SVG figures alongside report.json.
    #[arg(long)]
    report: bool,
    #[command(flatten)]
    provider: ProviderArgs,
}

#[derive(Debug, Clone, Args)]
struct EnergyArgs {
    /// Rank compatible structures by full Amber/GLYCAM potential energy.
    #[arg(long)]
    energy: bool,
    /// Rank by protein-glycan Lennard-Jones plus Coulomb interaction energy.
    #[arg(long)]
    interact: bool,
    /// Include OBC2 GBSA in full-energy scoring.
    #[arg(id = "energy_obc2", long = "energy-obc2")]
    obc2: bool,
    /// Brief all-atom L-BFGS minimization before each energy evaluation.
    #[arg(long)]
    min: bool,
    #[arg(long, default_value_t = 5)]
    min_iterations: usize,
    /// Complete-residue radius around glycans for the short local minimization.
    #[arg(long, default_value_t = 5.0)]
    min_radius: f64,
    /// Nonperiodic cutoff used by high-throughput energy scoring.
    #[arg(long, default_value_t = 10.0)]
    energy_cutoff: f64,
    #[arg(long, default_value_t = 300.0)]
    temperature_k: f64,
    #[arg(long, default_value_t = 100)]
    burn_in: usize,
    #[arg(long, default_value_t = 10)]
    thinning: usize,
}

#[derive(Debug, Args)]
struct ScanArgs {
    #[command(flatten)]
    protein: ProteinArgs,
    /// GlcNAc ensemble source; defaults to the compute scan identifier.
    #[arg(long, default_value = "G14843DJ")]
    glycan: String,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// GlcNAc scan GA population size (valid range: 32..=512).
    #[arg(long, default_value_t = 32)]
    population: usize,
    /// GlcNAc scan maximum GA generations (valid range: 1..=20000).
    #[arg(long, default_value_t = 25)]
    generations: usize,
    #[arg(long)]
    rotamers: bool,
    /// Parameterize the scanned structure and write an Amber/GROMACS bundle.
    #[arg(long)]
    system: bool,
    #[arg(long)]
    no_water: bool,
    #[arg(long)]
    no_ions: bool,
    #[arg(long)]
    overwrite: bool,
    /// Render report.pdf, report.typ, and SVG figures alongside report.json.
    #[arg(long)]
    report: bool,
    #[command(flatten)]
    provider: ProviderArgs,
}

#[derive(Debug, Args)]
struct SearchArgs {
    #[command(flatten)]
    common: SearchCommon,
}

#[derive(Debug, Clone, Args)]
struct RelaxCommon {
    #[arg(long, value_enum, default_value_t = MovableArg::Glycans)]
    movable: MovableArg,
    #[arg(
        long = "no-local-sidechains",
        action = clap::ArgAction::SetFalse,
        default_value_t = true
    )]
    local_sidechains: bool,
    #[arg(long, default_value_t = 5.0)]
    local_radius: f64,
    /// Optional nonbonded cutoff for minimization. Density refinement uses
    /// 10 Å automatically when this is not supplied.
    #[arg(long = "relax-cutoff")]
    nonbonded_cutoff: Option<f64>,
    /// Maximum L-BFGS iterations per relaxation stage.  Density fitting uses
    /// a local objective, so 50 is a responsive default; increase it for a
    /// more exhaustive minimization.
    #[arg(long, default_value_t = 50)]
    max_iterations: usize,
    /// Include OBC2 GBSA implicit solvent. Disabled by default for responsive
    /// vacuum minimization; enable deliberately for a solvent-aware run.
    #[arg(long)]
    obc2: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MovableArg {
    Glycans,
    All,
}

#[derive(Debug, Args)]
struct RelaxArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    relaxation: RelaxCommon,
    #[arg(long)]
    overwrite: bool,
    /// Render report.pdf, report.typ, and SVG figures alongside report.json.
    #[arg(long)]
    report: bool,
}

#[derive(Debug, Args)]
struct RefineArgs {
    #[command(flatten)]
    common: SearchCommon,
    #[command(flatten)]
    relaxation: RelaxCommon,
    #[arg(long)]
    solvate: bool,
    /// Refinement objective: `steric` (ensemble search + energy relaxation) or `density`
    /// (GlycoFlow fit of the glycan at each `--replace-glycan` site; needs
    /// `--glycoflow-model` or $GLYCOFLOW_MODEL).
    #[arg(long, value_enum, default_value_t = ObjectiveArg::Steric)]
    objective: ObjectiveArg,
    /// CCP4/MRC map path, or `auto` for a nearby sidecar map or (with --pdb-id) a PDBe EDS /
    /// RCSB 2Fo-Fc download.
    #[arg(long)]
    density_map: Option<String>,
    /// Atom width of the density model (default: calibrated on the protein around each site).
    #[arg(long)]
    density_sigma: Option<f64>,
    /// Map resolution (default: the model's REMARK 2 RESOLUTION record).
    #[arg(long)]
    density_resolution: Option<f64>,
    /// Map acquisition source for `--density-map auto`.
    #[arg(long = "density-map-source", value_enum, default_value_t = DensityMapSourceArg::Auto)]
    density_map_source: DensityMapSourceArg,
    /// RCSB VolumeServer detail level (0 is native; larger values are more
    /// aggressively downsampled).
    #[arg(long = "density-map-detail", default_value_t = 4)]
    density_map_detail: u8,
    /// Deprecated: density refinement always uses GlycoFlow. Accepted only as `glycoflow`.
    #[arg(long = "density-search", hide = true)]
    density_search: Option<String>,
    /// Remove the exact glycan tree at a site before fitting a replacement.
    /// Repeat as `SITE` or `SITE=SOURCE`. With `--objective steric`, SOURCE supplies the
    /// replacement ensemble (so --attach is optional); with `--objective density`, SOURCE may
    /// be a GLYCAM sequence (default: the deposited glycan's sequence).
    #[arg(long = "replace-glycan")]
    replace_glycans: Vec<String>,
    #[arg(long)]
    overwrite: bool,
    #[command(flatten)]
    glycoflow: glycoflow::GlycoflowArgs,
}

/// A carbohydrate component discovered from the deposited PDB connectivity.
/// This is an input/preflight record only: its coordinates are never passed to
/// the search.
#[derive(Debug, Clone, serde::Serialize)]
struct DiscoveredGlycanSite {
    site: ResidueId,
    topology_hash: String,
    canonical_wurcs: String,
    iupac: String,
    glytoucan: String,
    glycoshape_id: Option<String>,
    root_anomer: Anomer,
    source_residues: Vec<String>,
    residue_count: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ObjectiveArg {
    Steric,
    Density,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DensityMapSourceArg {
    Auto,
    Rcsb,
    Pdbe,
}

#[derive(Debug, Args)]
struct DensityArgs {
    /// Input PDB model to score.
    #[arg(long)]
    input: PathBuf,
    /// CCP4/MRC map path. `auto` searches sidecars and known EDS locations.
    #[arg(long = "density-map")]
    density_map: String,
    /// Restrict scoring to attachment sites; defaults to every glycan tree.
    #[arg(long = "site")]
    sites: Vec<String>,
    #[arg(long)]
    sigma: Option<f64>,
    #[arg(long)]
    resolution: Option<f64>,
    #[arg(long)]
    periodic: bool,
    #[arg(long, default_value_t = 2.0)]
    mask_radius: f64,
    #[arg(long, default_value_t = 1.0)]
    mask_falloff: f64,
    #[arg(short, long, default_value = "density-output")]
    output: PathBuf,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    report: bool,
}

#[derive(Debug, Args)]
struct ValidateArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Write a PDF bundle next to --output, or validation-report.* in the current directory.
    #[arg(long)]
    report: bool,
    /// Exit nonzero at the requested severity; default is informational and
    /// writes the complete report while exiting successfully.
    #[arg(long, value_enum, default_value_t = FailOnArg::Never)]
    fail_on: FailOnArg,
    #[arg(long)]
    min_density_cc: Option<f64>,
    #[arg(long = "density-map")]
    density_map: Option<PathBuf>,
    #[arg(long)]
    density_sigma: Option<f64>,
    #[arg(long)]
    density_resolution: Option<f64>,
    #[arg(long = "site")]
    density_sites: Vec<String>,
    /// Repeat as `--reference A:79=path/to/glycan-bundle`.
    #[arg(long)]
    references: Vec<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FailOnArg {
    Never,
    Errors,
    Warnings,
}

pub fn run() -> anyhow::Result<()> {
    run_with_version(env!("CARGO_PKG_VERSION"))
}

pub fn run_with_version(version: &'static str) -> anyhow::Result<()> {
    let matches = Cli::command().version(version).get_matches();
    let cli = Cli::from_arg_matches(&matches)?;
    match cli.command {
        Command::Build(arguments) => run_build(arguments),
        Command::Ensemble(arguments) => run_ensemble(arguments),
        Command::Search(arguments) => run_search(arguments),
        Command::Relax(arguments) => run_relax(arguments),
        Command::Refine(arguments) => run_refine(arguments),
        Command::Scan(arguments) => run_scan(arguments),
        Command::Validate(arguments) => run_validate(arguments),
        Command::Density(arguments) => run_density(arguments),
        Command::Saxs(arguments) => saxs::run(arguments),
    }
}

fn resolve_cli_attachment_budget(
    mode: Option<SearchBudgetArg>,
    population: Option<usize>,
    generations: Option<usize>,
    protein: &glysys::Structure,
    sites: &[SearchSite],
    scan_rotamers: bool,
) -> anyhow::Result<SearchBudgetResolution> {
    let explicit_pair = population.is_some() || generations.is_some();
    if population.is_some() != generations.is_some() {
        anyhow::bail!("--population and --generations must be supplied together");
    }
    if matches!(mode, Some(SearchBudgetArg::Auto)) && explicit_pair {
        anyhow::bail!("--search-budget auto cannot be combined with --population or --generations");
    }
    if matches!(mode, Some(SearchBudgetArg::Manual)) && !explicit_pair {
        anyhow::bail!("--search-budget manual requires --population and --generations");
    }
    let resolved_mode = match (mode, explicit_pair) {
        (Some(value), false) => value.mode(),
        (Some(SearchBudgetArg::Manual), true) => SearchBudgetMode::Manual,
        (Some(SearchBudgetArg::Auto), true) => unreachable!(),
        (None, true) => SearchBudgetMode::Manual,
        (None, false) => SearchBudgetMode::Auto,
    };
    let resolution = resolve_search_budget(
        protein,
        sites,
        scan_rotamers,
        Some(resolved_mode),
        population.unwrap_or(128),
        generations.unwrap_or(100),
    )?;
    Ok(resolution)
}

fn print_resolved_budget(resolution: &SearchBudgetResolution, quiet: bool) {
    if !quiet {
        eprintln!(
            "search budget: {} → {} candidates × {} generations{}",
            match resolution.requested_mode {
                SearchBudgetMode::Auto => "auto",
                SearchBudgetMode::Manual => "manual",
            },
            resolution.population_size,
            resolution.generations,
            if resolution.capped {
                " (Auto ceiling reached)"
            } else {
                ""
            }
        );
    }
}

fn run_build(arguments: BuildArgs) -> anyhow::Result<()> {
    configure_threads(arguments.threads).map_err(anyhow::Error::msg)?;
    if !arguments.quiet {
        eprintln!("build: loading protein and glycan ensembles...");
    }
    if arguments.sites.len() != arguments.glycans.len() {
        anyhow::bail!(
            "received {} --site values but {} --glycan values",
            arguments.sites.len(),
            arguments.glycans.len()
        );
    }
    if arguments.sites.is_empty() && arguments.attachments.is_empty() {
        anyhow::bail!("provide paired --site/--glycan values or at least one --attach");
    }
    let search_options = dry_options(arguments.overwrite);
    let protein = fetch_protein(&arguments.protein, &search_options)?.structure;
    let mut specifications = arguments
        .sites
        .iter()
        .zip(&arguments.glycans)
        .map(|(site, glycan)| format!("{site}={glycan}"))
        .collect::<Vec<_>>();
    specifications.extend(arguments.attachments);
    let sites = load_search_sites_with_format(
        &specifications,
        &arguments.provider,
        arguments.output_format.residue_name_format(),
    )?;
    let budget = resolve_cli_attachment_budget(
        arguments.search_budget,
        arguments.population,
        arguments.generations,
        &protein,
        &sites,
        arguments.rotamers,
    )?;
    print_resolved_budget(&budget, arguments.quiet);
    let search_builder = SystemBuilder::new(search_options)?;
    let mut config = SearchConfig {
        seed: arguments.seed,
        population_size: budget.population_size,
        generations: budget.generations,
        // Build always materialises the best complete candidate.  Strict
        // automation is enforced after artifacts are written below.
        require_clash_free: false,
        scan_rotamers: arguments.rotamers,
        polish_attachment_vmm: true,
        ..energy_search_config(&arguments.energy)?
    };
    if config.scoring_mode == SearchScoringMode::StericPrior {
        config.selection_policy = SearchSelectionPolicy::JointPriorV1;
    }
    let outcome = search_with_progress(&protein, &sites, &config, &search_builder, |event| {
        print_search_progress(event, arguments.quiet)
    })?;
    let result = if arguments.no_system {
        build_from_outcome(&protein, &sites, &outcome, &search_builder, false)?
    } else {
        let final_builder = SystemBuilder::new(build_options(
            arguments.overwrite,
            arguments.no_water,
            arguments.no_ions,
        ))?;
        build_from_outcome(&protein, &sites, &outcome, &final_builder, true)?
    };
    prepare_output(&arguments.output)?;
    let pdb_path = arguments.output.join("glycoprotein.pdb");
    check_output(&pdb_path, arguments.overwrite)?;
    write_pdb(&result.structure, &pdb_path)?;
    if let Some(system) = result.system {
        system.write_bundle(&arguments.output)?;
    }
    write_json(arguments.output.join("search.json"), &outcome)?;
    write_json(arguments.output.join("search-budget.json"), &budget)?;
    let mut warnings = outcome.warnings.clone();
    if sites
        .iter()
        .any(|site| matches!(site.ensemble.query.source, GlycanSource::LocalBundle(_)))
    {
        warnings.push(
            "Local glycan bundles preserve their input residue names; --output-format selects the representation for remote GlycoShape assets only.".into(),
        );
    }
    let partial =
        outcome.clash_status != reglyco_core::ClashStatus::ClashFree || !outcome.vmm_gate_satisfied;
    let mut report = WorkflowReport {
        status: if partial { "partial" } else { "complete" }.into(),
        clash_status: Some(outcome.clash_status),
        search: Some(outcome.clone()),
        relaxation: None,
        provenance: Provenance {
            command: "build".into(),
            seed: Some(config.seed),
            output_format: Some(arguments.output_format.api_segment().into()),
            search_budget: Some(budget.clone()),
            ensemble_sources: sites
                .iter()
                .map(|site| site.ensemble.provenance.clone())
                .collect(),
            ..Provenance::default()
        },
        warnings,
        diagnostics: Vec::new(),
        analysis: ReportAnalysis {
            energy_analysis: outcome.energy_analysis.clone(),
            search_budget: Some(budget.clone()),
            ..ReportAnalysis::default()
        },
    };
    report.analyze_structure(&result.structure);
    write_workflow_report(&report, &arguments.output, arguments.report)?;
    println!(
        "Built {} attachment(s) in {}",
        result.structure.metadata().glycosylation_sites.len(),
        arguments.output.display()
    );
    print_clash_status(outcome.clash_status);
    let unresolved_sites = outcome
        .sites
        .iter()
        .filter(|site| {
            site.steric_score > 1.1
                || !site.phi_within_vmm95.unwrap_or(true)
                || !site.psi_within_vmm95.unwrap_or(true)
        })
        .map(|site| site.site.residue.to_string())
        .collect::<Vec<_>>();
    if !unresolved_sites.is_empty() {
        println!(
            "WARNING: unresolved attachment sites: {}",
            unresolved_sites.join(", ")
        );
    }
    if !outcome.vmm_gate_satisfied {
        println!("WARNING: selected complete result is outside the VMM acceptance gate");
    }
    if arguments.require_clash_free && partial {
        anyhow::bail!(
            "Build wrote the best complete result to {}, but it is not clash-free",
            arguments.output.display()
        );
    }
    Ok(())
}

fn run_ensemble(arguments: EnsembleArgs) -> anyhow::Result<()> {
    configure_threads(arguments.threads).map_err(anyhow::Error::msg)?;
    let has_protein = arguments.protein.protein.is_some()
        || arguments.protein.uniprot.is_some()
        || arguments.protein.pdb_id.is_some();
    prepare_output(&arguments.output)?;
    if has_protein {
        if !arguments.quiet {
            eprintln!("ensemble: loading inputs and preparing reusable energy topology...");
        }
        if arguments.attachments.is_empty() {
            anyhow::bail!("attached ensemble mode requires at least one --attach SITE=GLYCAN");
        }
        let options = dry_options(false);
        let protein = fetch_protein(&arguments.protein, &options)?.structure;
        let sites = load_search_sites_with_format(
            &arguments.attachments,
            &arguments.provider,
            arguments.output_format.residue_name_format(),
        )?;
        let budget = resolve_cli_attachment_budget(
            arguments.search_budget,
            arguments.population,
            arguments.generations,
            &protein,
            &sites,
            arguments.move_sidechains,
        )?;
        print_resolved_budget(&budget, arguments.quiet);
        let builder = SystemBuilder::new(options)?;
        let config = SearchConfig {
            seed: arguments.seed,
            ensemble_size: arguments.frames,
            population_size: budget.population_size,
            generations: budget.generations,
            scan_rotamers: arguments.move_sidechains,
            mh_chains: arguments.chains,
            mh_burn_in_sweeps: arguments.burn_in_sweeps,
            mh_thinning_accepted: arguments.thinning_accepted,
            ..energy_search_config(&arguments.energy)?
        };
        let (frames, diagnostics) =
            sample_attached_ensemble(&protein, &sites, arguments.frames, &config, &builder)?;
        if !arguments.quiet {
            if diagnostics.fallback_used {
                eprintln!(
                    "ensemble: native sampling produced {} frame(s); GA-seeded constrained MH filled the remaining {}",
                    diagnostics.native_frames, diagnostics.fallback_frames
                );
            }
            eprintln!(
                "ensemble: accepted {} frame(s) from {} proposals",
                frames.len(),
                diagnostics.attempts
            );
        }
        if frames.len() != arguments.frames {
            anyhow::bail!(
                "ensemble could not produce the requested {} frame(s); returned {} after native sampling and GA-seeded constrained MH",
                arguments.frames,
                frames.len()
            );
        }
        let mut multi_model = String::new();
        for (index, frame) in frames.iter().enumerate() {
            multi_model.push_str(&format!("MODEL     {:>4}\n", index + 1));
            for line in frame.structure.to_pdb_string().lines() {
                if line != "END" {
                    multi_model.push_str(line);
                    multi_model.push('\n');
                }
            }
            multi_model.push_str("ENDMDL\n");
        }
        multi_model.push_str("END\n");
        std::fs::write(arguments.output.join("ensemble.pdb"), multi_model)?;
        if arguments.calculate_sasa {
            let analysis = calculate_sasa(frames.iter().map(|frame| &frame.structure))?;
            std::fs::write(
                arguments.output.join("sasa.pdb"),
                analysis.sasa_pdb_string(&frames[0].structure)?,
            )?;
            std::fs::write(
                arguments.output.join("real_sasa.pdb"),
                analysis.real_sasa_pdb_string(&frames[0].structure)?,
            )?;
            if arguments.calculate_hotspots {
                std::fs::write(
                    arguments.output.join("hotspots.pdb"),
                    analysis.hotspots_pdb_string(&frames[0].structure)?,
                )?;
            }
            if !arguments.quiet {
                eprintln!(
                    "ensemble: wrote sasa.pdb, real_sasa.pdb{}",
                    arguments
                        .calculate_hotspots
                        .then_some(" and hotspots.pdb")
                        .unwrap_or_default()
                );
            }
        }
        write_json(
            arguments.output.join("ensemble.json"),
            &serde_json::json!({
                "outputFormat": arguments.output_format.api_segment(),
                "seed": arguments.seed,
                "searchBudget": budget,
                "diagnostics": diagnostics,
                "frames": frames.iter().enumerate().map(|(index, frame)| serde_json::json!({
                    "model": index + 1,
                    "proposal_index": frame.proposal_index,
                    "source": frame.source,
                    "log_native_probability": frame.log_native_probability,
                    "selected_energy_kcal_per_mol": frame.selected_energy_kcal_per_mol,
                    "sites": frame.sites,
                })).collect::<Vec<_>>(),
            }),
        )?;
        write_json(arguments.output.join("search-budget.json"), &budget)?;
        let mut warnings = Vec::new();
        if sites
            .iter()
            .any(|site| matches!(site.ensemble.query.source, GlycanSource::LocalBundle(_)))
        {
            warnings.push(
                "Local glycan bundles preserve their input residue names; --output-format selects the representation for remote GlycoShape assets only.".into(),
            );
        }
        let mut report = WorkflowReport {
            status: "complete".into(),
            clash_status: Some(reglyco_core::ClashStatus::ClashFree),
            search: None,
            relaxation: None,
            provenance: Provenance {
                command: "ensemble".into(),
                seed: Some(arguments.seed),
                output_format: Some(arguments.output_format.api_segment().into()),
                search_budget: Some(budget.clone()),
                ensemble_sources: sites
                    .iter()
                    .map(|site| site.ensemble.provenance.clone())
                    .collect(),
                ..Provenance::default()
            },
            warnings,
            diagnostics: vec![format!(
                "sampled {} sterically compatible frame(s)",
                frames.len()
            )],
            analysis: ReportAnalysis {
                search_budget: Some(budget.clone()),
                ensemble: Some(EnsembleAnalysis {
                    requested_frames: diagnostics.requested_frames,
                    returned_frames: diagnostics.returned_frames,
                    attempts: diagnostics.attempts,
                    acceptance_rate: diagnostics.acceptance_rate,
                    native_frames: diagnostics.native_frames,
                    fallback_frames: diagnostics.fallback_frames,
                    fallback_used: diagnostics.fallback_used,
                    ga_restarts: diagnostics.ga_restarts,
                    chains: diagnostics.chains,
                    native_proposals: diagnostics.native_proposals,
                    native_accepts: diagnostics.native_accepts,
                    mh_proposals: diagnostics.mh_proposals,
                    mh_accepts: diagnostics.mh_accepts,
                    burn_in_sweeps: diagnostics.burn_in_sweeps,
                    thinning_accepted: diagnostics.thinning_accepted,
                    sampling_target: diagnostics.sampling_target,
                    sampling_segments: diagnostics
                        .segments
                        .iter()
                        .map(|segment| SamplingSegmentAnalysis {
                            id: segment.id,
                            target: segment.target,
                            backend: segment.backend.clone(),
                            frames: segment.frames,
                            attempts: segment.attempts,
                            accepts: segment.accepts,
                            burn_in_steps: segment.burn_in_steps,
                            fallback_reason: segment.fallback_reason.clone(),
                        })
                        .collect(),
                    native_log_probabilities: Vec::new(),
                }),
                ..ReportAnalysis::default()
            },
        };
        if let Some(first) = frames.first() {
            report.analyze_structure(&first.structure);
            report.analysis.glycosidic_torsions.clear();
            report.analysis.protein_linkage_torsions.clear();
            report.analysis.sterics.clear();
        }
        if config.scoring_mode != SearchScoringMode::StericPrior {
            if let Some(first) = frames.first() {
                let backend = match diagnostics.segments.as_slice() {
                    [] => "CPU",
                    [segment] => segment.backend.as_str(),
                    _ => "Mixed",
                };
                match reglyco_ensemble::analyze_energy_structure(
                    &sites,
                    &first.sites,
                    &first.structure,
                    &builder,
                    config.scoring_mode,
                    config.use_obc2,
                    config.energy_cutoff,
                    backend,
                ) {
                    Ok(energy_analysis) => {
                        report.analysis.energy_analysis = Some(energy_analysis);
                    }
                    Err(error) => report.warnings.push(format!(
                        "Energy contribution analysis was unavailable for the ensemble output: {error}"
                    )),
                }
            }
        }
        report.set_expected_clusters(&sites);
        for (index, frame) in frames.iter().enumerate() {
            report.record_ensemble_frame(
                index + 1,
                &frame.structure,
                &frame.sites,
                frame.log_native_probability,
            );
        }
        write_workflow_report(&report, &arguments.output, arguments.report)?;
        println!(
            "Sampled {} compatible glycoprotein frame(s) in {} attempts",
            frames.len(),
            diagnostics.attempts
        );
        return Ok(());
    }
    if arguments.calculate_sasa || arguments.calculate_hotspots {
        anyhow::bail!("SASA and hotspot outputs require attached ensemble mode");
    }
    let glycan = arguments
        .glycan
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("free-glycan mode requires --glycan"))?;
    let query = query_for_source(glycan, &arguments.provider);
    let ensemble = load_ensemble(&query, &arguments.provider)?;
    let summary = serde_json::json!({
        "conformers": ensemble.conformers.len(),
        "provenance": ensemble.provenance,
        "clusters": ensemble.conformers.iter().map(|c| serde_json::json!({
            "id": c.id,
            "cluster_index": c.cluster_index,
            "weight": c.cluster_weight,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        arguments.output.join("ensemble.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    write_pdb(
        &ensemble.conformers[0].structure,
        arguments.output.join("conformer-1.pdb"),
    )?;
    let mut multi_model = String::new();
    for (index, conformer) in ensemble.conformers.iter().enumerate() {
        multi_model.push_str(&format!("MODEL     {:>4}\n", index + 1));
        for line in conformer.structure.to_pdb_string().lines() {
            if line != "END" {
                multi_model.push_str(line);
                multi_model.push('\n');
            }
        }
        multi_model.push_str("ENDMDL\n");
    }
    multi_model.push_str("END\n");
    std::fs::write(arguments.output.join("ensemble.pdb"), multi_model)?;
    let mut report = WorkflowReport {
        status: "complete".into(),
        clash_status: None,
        search: None,
        relaxation: None,
        provenance: Provenance {
            command: "ensemble".into(),
            ensemble_sources: vec![ensemble.provenance.clone()],
            ..Provenance::default()
        },
        warnings: Vec::new(),
        diagnostics: vec![format!(
            "loaded {} free-glycan conformer(s)",
            ensemble.conformers.len()
        )],
        analysis: ReportAnalysis {
            ensemble: Some(EnsembleAnalysis {
                requested_frames: ensemble.conformers.len(),
                returned_frames: ensemble.conformers.len(),
                attempts: ensemble.conformers.len(),
                acceptance_rate: 1.0,
                native_log_probabilities: Vec::new(),
                ..EnsembleAnalysis::default()
            }),
            clusters: ensemble
                .conformers
                .iter()
                .map(|conformer| ClusterObservation {
                    site: "free glycan".into(),
                    cluster_index: conformer.cluster_index,
                    main_cluster: conformer.main_cluster,
                    expected_weight: conformer.cluster_weight,
                    observed_count: 1,
                })
                .collect(),
            ..ReportAnalysis::default()
        },
    };
    report.analyze_structure(&ensemble.conformers[0].structure);
    for (index, conformer) in ensemble.conformers.iter().enumerate() {
        report.record_ensemble_frame(index + 1, &conformer.structure, &[], 0.0);
    }
    write_workflow_report(&report, &arguments.output, arguments.report)?;
    println!("Loaded {} conformers", ensemble.conformers.len());
    Ok(())
}

fn run_search(arguments: SearchArgs) -> anyhow::Result<()> {
    let (protein, sites, builder, config) = load_search(&arguments.common)?;
    let outcome = search_with_progress(&protein, &sites, &config, &builder, |event| {
        print_search_progress(event, arguments.common.quiet)
    })?;
    let product = build_from_outcome(&protein, &sites, &outcome, &builder, false)?;
    prepare_output(&arguments.common.output)?;
    write_pdb(
        &product.structure,
        arguments.common.output.join("structure.pdb"),
    )?;
    write_json(arguments.common.output.join("search.json"), &outcome)?;
    let report = WorkflowReport {
        status: "complete".into(),
        clash_status: Some(outcome.clash_status),
        search: Some(outcome.clone()),
        relaxation: None,
        provenance: Provenance {
            command: "search".into(),
            seed: Some(config.seed),
            ensemble_sources: sites
                .iter()
                .map(|site| site.ensemble.provenance.clone())
                .collect(),
            ..Provenance::default()
        },
        warnings: outcome.warnings.clone(),
        diagnostics: Vec::new(),
        analysis: ReportAnalysis::default(),
    };
    let mut report = report;
    report.analyze_structure(&product.structure);
    write_workflow_report(&report, &arguments.common.output, arguments.common.report)?;
    print_clash_status(outcome.clash_status);
    Ok(())
}

fn run_relax(arguments: RelaxArgs) -> anyhow::Result<()> {
    let started = Instant::now();
    eprintln!("relax: reading {}", arguments.input.display());
    let options = dry_options(arguments.overwrite);
    let structure = read_pdb(&arguments.input, &options)?;
    eprintln!(
        "relax: parameterizing {} atoms for {} minimization",
        structure.atoms().len(),
        if arguments.relaxation.obc2 {
            "OBC2 GBSA"
        } else {
            "vacuum"
        }
    );
    let builder = SystemBuilder::new(options)?;
    let system = builder.prepare_structure(&structure)?;
    let relax_options = relax_options(&arguments.relaxation);
    eprintln!(
        "relax: parameterized {} atoms in {:.1}s; starting staged minimization",
        system.atom_count(),
        started.elapsed().as_secs_f64()
    );
    let result = relax_with_progress(&structure, &system, &relax_options, |event| match event {
        RelaxProgress::InitialEnergyStarted => {
            eprintln!("relax: evaluating initial energy...");
        }
        RelaxProgress::InitialEnergy { energy } => {
            eprintln!("relax: initial energy {:.3} kcal/mol", energy.total());
        }
        RelaxProgress::StageStarted {
            name,
            movable_atoms,
        } => {
            eprintln!("relax: stage {name}: {movable_atoms} movable atoms; evaluating...");
        }
        RelaxProgress::Iteration {
            stage,
            iteration,
            energy,
            rms_gradient,
            max_gradient,
            accepted_steps,
        } => {
            eprintln!(
                "relax: {stage} iteration {iteration} (accepted {accepted_steps}): \
                 E={energy:.3} kcal/mol, rms|g|={rms_gradient:.4}, max|g|={max_gradient:.4}"
            );
        }
        RelaxProgress::StageFinished { name, diagnostics } => {
            eprintln!(
                "relax: stage {name} finished after {} iterations ({}, E={:.3} kcal/mol)",
                diagnostics.iterations,
                diagnostics.convergence_reason,
                diagnostics.final_energy.total()
            );
        }
    })?;
    prepare_output(&arguments.output)?;
    let structure_path = arguments.output.join("structure.pdb");
    write_pdb(&result.structure, &structure_path)?;
    let written = read_pdb(&structure_path, &dry_options(false))?;
    verify_relaxed_glycans(&result.structure, &written)?;
    write_json(
        arguments.output.join("relaxation.json"),
        &result.diagnostics,
    )?;
    let mut report = WorkflowReport {
        status: "complete".into(),
        clash_status: None,
        search: None,
        relaxation: Some(result.diagnostics),
        provenance: Provenance {
            command: "relax".into(),
            ..Provenance::default()
        },
        warnings: Vec::new(),
        diagnostics: Vec::new(),
        analysis: ReportAnalysis::default(),
    };
    report.analyze_structure(&result.structure);
    write_workflow_report(&report, &arguments.output, arguments.report)?;
    eprintln!(
        "relax: wrote {} glycan atoms across {} attachment(s) to {}",
        glycan_atom_count(&written),
        written.metadata().glycosylation_sites.len(),
        structure_path.display()
    );
    eprintln!("relax: complete in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_refine(arguments: RefineArgs) -> anyhow::Result<()> {
    let started = Instant::now();
    if let Some(search) = arguments.density_search.as_deref() {
        if !search.eq_ignore_ascii_case("glycoflow") {
            anyhow::bail!(
                "--density-search {search} was removed together with the legacy density fitter; \
                 `--objective density` always fits with GlycoFlow (drop --density-search)"
            );
        }
        if !matches!(arguments.objective, ObjectiveArg::Density) {
            anyhow::bail!("--density-search glycoflow requires --objective density");
        }
    }
    if matches!(arguments.objective, ObjectiveArg::Density) {
        return glycoflow::run_refine_glycoflow(arguments, started);
    }
    let quiet = arguments.common.quiet;
    if !quiet {
        eprintln!("refine: loading protein, replacement ensemble, and search configuration...");
    }
    let mut search_arguments = arguments.common.clone();
    let mut discovered_glycans = Vec::new();
    let mut discovered_anomers = BTreeMap::<ResidueId, Anomer>::new();
    let mut replacement_requests = arguments.replace_glycans.clone();
    let has_bare_replacement = arguments
        .replace_glycans
        .iter()
        .any(|replacement| !replacement.contains('='));
    let needs_discovery = search_arguments.attachments.is_empty()
        && ((arguments.common.protein.pdb_id.is_some() && arguments.replace_glycans.is_empty())
            || has_bare_replacement);
    if needs_discovery {
        let fetched = fetch_protein(&search_arguments.protein, &dry_options(false))?;
        discovered_glycans = discover_glycan_sites(&fetched.structure, &search_arguments.provider)?;
        if discovered_glycans.is_empty() {
            anyhow::bail!(
                "automatic glycan discovery found no protein-linked carbohydrate components"
            );
        }
        let requested_sites = arguments
            .replace_glycans
            .iter()
            .map(|replacement| {
                replacement
                    .split_once('=')
                    .map_or(replacement.as_str(), |(site, _)| site)
                    .to_string()
            })
            .collect::<BTreeSet<_>>();
        let selected = if requested_sites.is_empty() {
            discovered_glycans.iter().collect::<Vec<_>>()
        } else {
            discovered_glycans
                .iter()
                .filter(|site| requested_sites.contains(&site.site.to_string()))
                .collect::<Vec<_>>()
        };
        if !requested_sites.is_empty() && selected.len() != requested_sites.len() {
            let found = selected
                .iter()
                .map(|site| site.site.to_string())
                .collect::<BTreeSet<_>>();
            let missing = requested_sites
                .difference(&found)
                .cloned()
                .collect::<Vec<_>>();
            anyhow::bail!(
                "requested glycan site(s) were not discovered: {}",
                missing.join(", ")
            );
        }
        search_arguments.attachments = selected
            .into_iter()
            .map(|site| {
                discovered_anomers.insert(site.site.clone(), site.root_anomer.clone());
                format!("{}={}", site.site, site.glytoucan)
            })
            .collect();
        if replacement_requests.is_empty() {
            replacement_requests = discovered_glycans
                .iter()
                .map(|site| site.site.to_string())
                .collect();
        }
        if !quiet {
            eprintln!(
                "refine: discovered {} protein-linked glycan site(s) from crabWURCS 0.3.1",
                search_arguments.attachments.len()
            );
        }
    } else if search_arguments.attachments.is_empty() {
        search_arguments.attachments = arguments
            .replace_glycans
            .iter()
            .filter(|replacement| replacement.contains('='))
            .cloned()
            .collect();
    }
    // A bare --replace-glycan SITE is resolved from deposited connectivity;
    // explicit SITE=SOURCE remains authoritative and does not need discovery.
    if search_arguments.attachments.is_empty() {
        anyhow::bail!("provide --attach/--replace-glycan or use automatic PDB glycan discovery");
    }
    let (mut protein, sites, dry_builder, search_config) =
        load_search_with_anomers(&search_arguments, Some(&discovered_anomers))?;
    if !quiet {
        let conformers = sites
            .iter()
            .map(|site| site.ensemble.conformers.len())
            .sum::<usize>();
        eprintln!(
            "refine: loaded {} protein atoms, {} attachment site(s), {} ensemble conformer(s) in {:.1}s",
            protein.atoms().len(),
            sites.len(),
            conformers,
            started.elapsed().as_secs_f64()
        );
    }
    for replacement in &replacement_requests {
        let site_text = replacement
            .split_once('=')
            .map_or(replacement.as_str(), |(site, _)| site);
        let site = parse_site(site_text)?;
        let (stripped, removed) = remove_glycan_at_site(&protein, &site)?;
        protein = stripped;
        if !quiet {
            eprintln!(
                "refine: removed {} residue(s) from existing glycan at {}",
                removed.len(),
                site
            );
        }
    }
    let relaxation = relax_options(&arguments.relaxation);
    let final_builder = arguments
        .solvate
        .then(|| {
            SystemBuilder::new(BuildOptions {
                overwrite: arguments.overwrite,
                ..BuildOptions::default()
            })
        })
        .transpose()?;
    let result = refine_with_progress(
        RefineRequest {
            protein,
            sites,
            search: search_config,
            relaxation,
        },
        &dry_builder,
        final_builder.as_ref(),
        |event| print_refine_progress(event, quiet, started),
    )?;
    if !quiet {
        eprintln!(
            "refine: optimization complete in {:.1}s; writing output bundle...",
            started.elapsed().as_secs_f64()
        );
    }
    prepare_output(&arguments.common.output)?;
    if !discovered_glycans.is_empty() {
        write_json(
            arguments.common.output.join("discovered-glycans.json"),
            &discovered_glycans,
        )?;
    }
    write_pdb(
        &result.initial.structure,
        arguments.common.output.join("fitted.pdb"),
    )?;
    write_pdb(
        &result.relaxed_structure,
        arguments.common.output.join("structure.pdb"),
    )?;
    write_json(arguments.common.output.join("search.json"), &result.search)?;
    if let Some(relaxation) = &result.relaxation {
        write_json(arguments.common.output.join("relaxation.json"), relaxation)?;
    }
    if !quiet {
        eprintln!("refine: validating fitted structure...");
    }
    let validation = reglyco_validate::validate_with_options(
        &result.relaxed_structure,
        &reglyco_validate::ValidationOptions::default(),
    );
    write_json(arguments.common.output.join("validation.json"), &validation)?;
    let mut report = result.report;
    report.analyze_structure(&result.relaxed_structure);
    if !quiet {
        eprintln!("refine: generating report bundle...");
    }
    write_workflow_report(&report, &arguments.common.output, arguments.common.report)?;
    if let Some(system) = result.final_system {
        system.write_bundle(&arguments.common.output)?;
    }
    print_clash_status(result.search.clash_status);
    if !quiet {
        eprintln!(
            "refine: complete in {:.1}s ({})",
            started.elapsed().as_secs_f64(),
            arguments.common.output.display()
        );
    }
    Ok(())
}

fn run_density(arguments: DensityArgs) -> anyhow::Result<()> {
    let structure = read_pdb(&arguments.input, &dry_options(false))?;
    let map_path = resolve_density_map(&arguments.density_map, Some(&arguments.input), None)?;
    let map = DensityMap::open(&map_path)?;
    let mut options = DensityScoreOptions {
        sigma_angstrom: arguments.sigma,
        glycan_b_factor: None,
        mask_radius_angstrom: arguments.mask_radius,
        mask_falloff_angstrom: arguments.mask_falloff,
        periodic: arguments.periodic || map.is_full_unit_cell(),
        support_threshold: DensityScoreOptions::default().support_threshold,
    };
    if let Some(resolution) = arguments.resolution {
        options = options.with_resolution(resolution)?;
    }
    let scorer = DensityScorer::new(map, options)?;
    let targets = if arguments.sites.is_empty() {
        DensityTarget::all(&structure)
    } else {
        arguments
            .sites
            .iter()
            .map(|site| {
                Ok(DensityTarget {
                    site: parse_site(site)?,
                    glycan_residues: Vec::new(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    scorer.attach_protein_background(&structure, &targets)?;
    let score = scorer.score(&structure, &targets)?;
    let validation = reglyco_validate::validate_with_density(
        &structure,
        &reglyco_validate::ValidationOptions::default(),
        &scorer,
        &targets,
    )?;
    prepare_output(&arguments.output)?;
    check_output(&arguments.output.join("density.json"), arguments.overwrite)?;
    write_json(
        arguments.output.join("density.json"),
        &serde_json::json!({
            "map": scorer.map.metadata(),
            "score": score,
            "targets": targets,
        }),
    )?;
    write_json(arguments.output.join("validation.json"), &validation)?;
    write_pdb(&structure, arguments.output.join("structure.pdb"))?;
    let mut report = WorkflowReport {
        status: if validation.valid {
            "complete".into()
        } else {
            "invalid".into()
        },
        clash_status: None,
        search: None,
        relaxation: None,
        provenance: Provenance {
            command: "density".into(),
            ..Provenance::default()
        },
        warnings: validation.warnings.clone(),
        diagnostics: vec![format!(
            "union_density_correlation={:.6}",
            score.correlation
        )],
        analysis: ReportAnalysis {
            validation: Some(ValidationAnalysis {
                valid: validation.valid,
                atom_count: validation.atom_count,
                residue_count: validation.residue_count,
                attachment_count: validation.attachment_count,
                error_count: validation.errors.len(),
                warning_count: validation.warnings.len(),
            }),
            density: Some(DensityAnalysis {
                correlation: score.correlation,
                sigma_angstrom: Some(score.sigma_angstrom),
                training_likelihood_gain: Some(score.training_likelihood_gain),
                heldout_likelihood_gain: Some(score.heldout_likelihood_gain),
                supported_residues: score
                    .residue_support
                    .iter()
                    .filter(|support| support.supported)
                    .map(|support| support.residue.to_string())
                    .collect(),
                unsupported_residues: score
                    .residue_support
                    .iter()
                    .filter(|support| !support.supported)
                    .map(|support| support.residue.to_string())
                    .collect(),
                supported_atom_fraction: Some(score.supported_atom_fraction),
                ring_support: Some(score.ring_support),
                connectivity_support: Some(score.connectivity_support),
                site_diagnostics: score
                    .sites
                    .iter()
                    .map(|site| DensitySiteAnalysis {
                        site: site.site.to_string(),
                        correlation: site.correlation,
                        voxel_count: site.voxel_count,
                        supported_atom_fraction: site.supported_atom_fraction,
                        ring_support: site.ring_support,
                        connectivity_support: site.connectivity_support,
                    })
                    .collect(),
            }),
            ..ReportAnalysis::default()
        },
    };
    report.analyze_structure(&structure);
    write_workflow_report(&report, &arguments.output, arguments.report)?;
    println!(
        "Density correlation {:.4} for {} target site(s)",
        score.correlation,
        score.sites.len()
    );
    Ok(())
}

fn run_scan(arguments: ScanArgs) -> anyhow::Result<()> {
    validate_cookbook_scan_config(arguments.population, arguments.generations)?;
    let dry = dry_options(arguments.overwrite);
    let fetched = fetch_protein(&arguments.protein, &dry)?;
    let sequons = scan_n_linked_sequons(&fetched.structure);
    prepare_output(&arguments.output)?;
    if sequons.is_empty() {
        let scan = serde_json::json!({
            "protein": fetched.provenance,
            "glycan": arguments.glycan,
            "sequons": [],
            "status": "no_sequons",
        });
        write_json(arguments.output.join("scan.json"), &scan)?;
        write_pdb(&fetched.structure, arguments.output.join("structure.pdb"))?;
        let mut report = WorkflowReport {
            status: "no_sequons".into(),
            clash_status: None,
            search: None,
            relaxation: None,
            provenance: Provenance {
                command: "scan".into(),
                seed: Some(arguments.seed),
                ..Provenance::default()
            },
            warnings: Vec::new(),
            diagnostics: vec!["identified 0 N-X-S/T sequons".into()],
            analysis: ReportAnalysis {
                scan: Some(ScanAnalysis {
                    interpretation:
                        "structural accessibility only; not proof of biological glycosylation"
                            .into(),
                    independent_accessible_count: 0,
                    jointly_compatible_count: 0,
                    sequons: Vec::new(),
                }),
                ..ReportAnalysis::default()
            },
        };
        report.analyze_structure(&fetched.structure);
        write_workflow_report(&report, &arguments.output, arguments.report)?;
        println!("No unoccupied N-X-S/T sequons found");
        return Ok(());
    }
    let query = query_for_source(&arguments.glycan, &arguments.provider);
    let ensemble = load_ensemble(&query, &arguments.provider)?;
    let sites = sequons
        .iter()
        .map(|sequon| SearchSite {
            site: GlycosylationSite {
                residue: sequon.asparagine.clone(),
            },
            ensemble: ensemble.clone(),
        })
        .collect::<Vec<_>>();
    let builder = SystemBuilder::new(dry)?;
    let config = SearchConfig {
        seed: arguments.seed,
        population_size: arguments.population,
        generations: arguments.generations,
        require_clash_free: false,
        scan_rotamers: arguments.rotamers,
        ..SearchConfig::default()
    };
    let mut independent = Vec::with_capacity(sites.len());
    for (index, site) in sites.iter().enumerate() {
        let mut site_config = config.clone();
        site_config.seed = config.seed.wrapping_add(index as u64);
        let outcome = search(
            &fetched.structure,
            std::slice::from_ref(site),
            &site_config,
            &builder,
        )?;
        independent.push(outcome);
    }

    // Greedily construct a deterministic maximal compatible subset from
    // independently accessible sites. Blocked sites are never materialized.
    let mut accepted_sites = Vec::new();
    let mut accepted_results = Vec::new();
    for (site, outcome) in sites.iter().zip(&independent) {
        if outcome.clash_status != reglyco_core::ClashStatus::ClashFree {
            continue;
        }
        let mut trial_sites = accepted_sites.clone();
        trial_sites.push(site.clone());
        let mut trial_results = accepted_results.clone();
        trial_results.push(outcome.sites[0].clone());
        let trial = reglyco_core::SearchOutcome {
            sites: trial_results.clone(),
            total_score: trial_results.iter().map(|result| result.steric_score).sum(),
            seed: config.seed,
            generations: outcome.generations,
            clash_status: reglyco_core::ClashStatus::ClashFree,
            complete_output: true,
            vmm_gate_satisfied: true,
            termination_reason: "completed".into(),
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
            energy_diagnostics: reglyco_core::EnergySearchDiagnostics::default(),
            minimized_coordinates: Vec::new(),
            timings: Default::default(),
            vmm_polish: Default::default(),
            energy_analysis: None,
        };
        let product =
            build_from_outcome(&fetched.structure, &trial_sites, &trial, &builder, false)?;
        if steric_site_scores(&product.structure, config.clash_distance)
            .iter()
            .all(|score| *score <= 1.1)
        {
            accepted_sites = trial_sites;
            accepted_results = trial_results;
        }
    }
    let outcome = reglyco_core::SearchOutcome {
        total_score: accepted_results
            .iter()
            .map(|result| result.steric_score)
            .sum(),
        sites: accepted_results,
        seed: config.seed,
        generations: independent.iter().map(|outcome| outcome.generations).sum(),
        clash_status: reglyco_core::ClashStatus::ClashFree,
        complete_output: true,
        vmm_gate_satisfied: true,
        termination_reason: "completed".into(),
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
        energy_diagnostics: reglyco_core::EnergySearchDiagnostics::default(),
        minimized_coordinates: Vec::new(),
        timings: Default::default(),
        vmm_polish: Default::default(),
        energy_analysis: None,
    };
    let product = if accepted_sites.is_empty() {
        reglyco_core::BuildProduct {
            structure: fetched.structure.clone(),
            system: None,
        }
    } else {
        build_from_outcome(
            &fetched.structure,
            &accepted_sites,
            &outcome,
            &builder,
            false,
        )?
    };
    write_pdb(&product.structure, arguments.output.join("structure.pdb"))?;
    write_json(arguments.output.join("search.json"), &outcome)?;
    let scan_report = serde_json::json!({
        "status": "complete",
        "interpretation": "structural accessibility only; not proof of biological glycosylation",
        "protein": fetched.provenance,
        "glycan": arguments.glycan,
        "sequons": sequons,
        "sites": sequons.iter().zip(&independent).map(|(sequon, result)| serde_json::json!({
            "sequon": sequon,
            "accessible": result.clash_status == reglyco_core::ClashStatus::ClashFree,
            "joint_subset": outcome.sites.iter().any(|selected| selected.site.residue == sequon.asparagine),
            "steric_score": result.sites.first().map(|site| site.steric_score),
            "cluster": result.sites.first().map(|site| site.cluster_index),
            "phi": result.sites.first().map(|site| site.phi_degrees),
            "psi": result.sites.first().map(|site| site.psi_degrees),
            "rotamer": result.sites.first().and_then(|site| site.rotamer_index),
            "generations": result.generations,
        })).collect::<Vec<_>>(),
        "independent_accessible_count": independent.iter().filter(|result| result.clash_status == reglyco_core::ClashStatus::ClashFree).count(),
        "joint_attachment_count": outcome.sites.len(),
        "attachment_count": outcome.sites.len(),
    });
    write_json(arguments.output.join("scan.json"), &scan_report)?;
    let scan_sequons = sequons
        .iter()
        .zip(&independent)
        .map(|(sequon, result)| ScanSequon {
            context: sequon.context.clone(),
            motif: sequon.motif.clone(),
            asparagine: sequon.asparagine.clone(),
            independently_accessible: result.clash_status == reglyco_core::ClashStatus::ClashFree,
            jointly_selected: outcome
                .sites
                .iter()
                .any(|selected| selected.site.residue == sequon.asparagine),
        })
        .collect::<Vec<_>>();
    let mut report = WorkflowReport {
        status: "complete".into(),
        clash_status: Some(outcome.clash_status),
        search: Some(outcome.clone()),
        relaxation: None,
        provenance: Provenance {
            command: "scan".into(),
            seed: Some(config.seed),
            ensemble_sources: vec![ensemble.provenance],
            ..Provenance::default()
        },
        warnings: Vec::new(),
        diagnostics: vec![
            format!("identified {} N-X-S/T sequons", sequons.len()),
            format!(
                "{} independently accessible",
                independent
                    .iter()
                    .filter(|result| result.clash_status == reglyco_core::ClashStatus::ClashFree)
                    .count()
            ),
            format!(
                "{} retained in the jointly compatible subset",
                outcome.sites.len()
            ),
        ],
        analysis: ReportAnalysis {
            scan: Some(ScanAnalysis {
                interpretation:
                    "structural accessibility only; not proof of biological glycosylation".into(),
                independent_accessible_count: independent
                    .iter()
                    .filter(|result| result.clash_status == reglyco_core::ClashStatus::ClashFree)
                    .count(),
                jointly_compatible_count: outcome.sites.len(),
                sequons: scan_sequons,
            }),
            ..ReportAnalysis::default()
        },
    };
    report.analyze_structure(&product.structure);
    write_workflow_report(&report, &arguments.output, arguments.report)?;
    if arguments.system {
        let final_builder = SystemBuilder::new(build_options(
            arguments.overwrite,
            arguments.no_water,
            arguments.no_ions,
        ))?;
        final_builder
            .prepare_structure(&product.structure)?
            .write_bundle(&arguments.output)?;
    }
    println!(
        "Scanned {} sequon(s); {} independently accessible, {} jointly compatible",
        sequons.len(),
        independent
            .iter()
            .filter(|result| result.clash_status == reglyco_core::ClashStatus::ClashFree)
            .count(),
        outcome.sites.len(),
    );
    Ok(())
}

fn run_validate(arguments: ValidateArgs) -> anyhow::Result<()> {
    let structure = read_pdb(&arguments.input, &dry_options(false))?;
    let references = arguments
        .references
        .iter()
        .map(|reference| {
            let (site, source) = reference
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--reference must use SITE=SOURCE syntax"))?;
            Ok(reglyco_validate::ValidationReference {
                site: parse_site(site)?,
                source: source.into(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    // --site focuses the structural checks on those sites (as refine does for its targets),
    // so other deposited glycans and unrelated protein contacts do not decide validity.
    let focus_sites = arguments
        .density_sites
        .iter()
        .map(|site| parse_site(site))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let validation_options = reglyco_validate::ValidationOptions {
        min_density_cc: arguments.min_density_cc,
        references,
        focus_sites,
        ..reglyco_validate::ValidationOptions::default()
    };
    let density_scorer = if let Some(path) = &arguments.density_map {
        let map = DensityMap::open(path)?;
        let mut options = DensityScoreOptions {
            sigma_angstrom: arguments.density_sigma,
            periodic: map.is_full_unit_cell(),
            ..DensityScoreOptions::default()
        };
        if let Some(resolution) = arguments.density_resolution {
            options = options.with_resolution(resolution)?;
        }
        Some(DensityScorer::new(map, options)?)
    } else {
        if arguments.min_density_cc.is_some() {
            anyhow::bail!("--min-density-cc requires --density-map");
        }
        None
    };
    let density_targets = arguments
        .density_sites
        .iter()
        .map(|site| {
            Ok(DensityTarget {
                site: parse_site(site)?,
                glycan_residues: Vec::new(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let validation = if let Some(scorer) = &density_scorer {
        let targets = if density_targets.is_empty() {
            DensityTarget::all(&structure)
        } else {
            density_targets.as_slice().to_vec()
        };
        reglyco_validate::validate_with_density(&structure, &validation_options, scorer, &targets)?
    } else {
        reglyco_validate::validate_with_options(&structure, &validation_options)
    };
    let json = serde_json::to_string_pretty(&validation)? + "\n";
    let output = arguments.output;
    if let Some(output) = &output {
        std::fs::write(output, json)?;
    } else {
        print!("{json}");
    }
    let root = output
        .as_deref()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."));
    let prefix = output
        .as_deref()
        .and_then(Path::file_stem)
        .and_then(|stem| stem.to_str())
        .map_or_else(
            || "validation-report".to_string(),
            |stem| format!("{stem}-report"),
        );
    let mut workflow = WorkflowReport {
        status: if validation.valid {
            "complete".into()
        } else {
            "invalid".into()
        },
        clash_status: None,
        search: None,
        relaxation: None,
        provenance: Provenance {
            command: "validate".into(),
            ..Provenance::default()
        },
        warnings: validation.warnings.clone(),
        diagnostics: Vec::new(),
        analysis: ReportAnalysis {
            validation: Some(ValidationAnalysis {
                valid: validation.valid,
                atom_count: validation.atom_count,
                residue_count: validation.residue_count,
                attachment_count: validation.attachment_count,
                error_count: validation.errors.len(),
                warning_count: validation.warnings.len(),
            }),
            density: validation.density.as_ref().map(|score| DensityAnalysis {
                correlation: score.correlation,
                sigma_angstrom: Some(score.sigma_angstrom),
                training_likelihood_gain: Some(score.training_likelihood_gain),
                heldout_likelihood_gain: Some(score.heldout_likelihood_gain),
                supported_residues: score
                    .residue_support
                    .iter()
                    .filter(|support| support.supported)
                    .map(|support| support.residue.to_string())
                    .collect(),
                unsupported_residues: score
                    .residue_support
                    .iter()
                    .filter(|support| !support.supported)
                    .map(|support| support.residue.to_string())
                    .collect(),
                supported_atom_fraction: Some(score.supported_atom_fraction),
                ring_support: Some(score.ring_support),
                connectivity_support: Some(score.connectivity_support),
                site_diagnostics: score
                    .sites
                    .iter()
                    .map(|site| DensitySiteAnalysis {
                        site: site.site.to_string(),
                        correlation: site.correlation,
                        voxel_count: site.voxel_count,
                        supported_atom_fraction: site.supported_atom_fraction,
                        ring_support: site.ring_support,
                        connectivity_support: site.connectivity_support,
                    })
                    .collect(),
            }),
            ..ReportAnalysis::default()
        },
    };
    workflow.analyze_structure(&structure);
    if arguments.report {
        workflow.write_bundle_with_prefix(root, &prefix)?;
    } else {
        workflow.write_json(root.join(format!("{prefix}.json")))?;
    }
    let should_fail = match arguments.fail_on {
        FailOnArg::Never => false,
        FailOnArg::Errors => !validation.errors.is_empty(),
        FailOnArg::Warnings => !validation.errors.is_empty() || !validation.warnings.is_empty(),
    };
    if should_fail {
        anyhow::bail!("structure validation reached the requested fail-on severity");
    }
    Ok(())
}

fn resolve_density_map(
    value: &str,
    input: Option<&Path>,
    protein: Option<&ProteinArgs>,
) -> anyhow::Result<PathBuf> {
    if value != "auto" {
        let path = PathBuf::from(value);
        if !path.is_file() {
            anyhow::bail!("density map does not exist: {}", path.display());
        }
        return Ok(path);
    }
    if let Some(arguments) = protein {
        if let Some(identifier) = &arguments.pdb_id {
            let provider = ProteinProvider::new(ProteinFetchOptions {
                cache_dir: arguments.protein_cache.join("maps"),
                offline: arguments.protein_offline,
                ..ProteinFetchOptions::default()
            })?;
            return Ok(provider.fetch_eds_map(identifier)?);
        }
    }
    let input = input.ok_or_else(|| {
        anyhow::anyhow!(
            "--density-map auto requires a local input PDB or --pdb-id; provide a map path for other sources"
        )
    })?;
    let mut candidates = Vec::new();
    for extension in ["mrc", "map", "ccp4", "mrc.gz", "map.gz", "ccp4.gz"] {
        let mut path = input.to_path_buf();
        path.set_extension(extension);
        candidates.push(path);
    }
    candidates.push(input.with_file_name("density.mrc"));
    candidates.push(input.with_file_name("density.map"));
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "could not resolve --density-map auto; place a .mrc/.map/.ccp4 sidecar next to {} or provide an explicit path",
                input.display()
            )
        })
}

fn resolve_density_map_for_refine(
    value: &str,
    input: Option<&Path>,
    protein: Option<&ProteinArgs>,
    source: DensityMapSourceArg,
    detail: u8,
) -> anyhow::Result<(PathBuf, Option<RcsbMapAcquisition>)> {
    if value != "auto" {
        let path = PathBuf::from(value);
        if !path.is_file() {
            anyhow::bail!("density map does not exist: {}", path.display());
        }
        return Ok((path, None));
    }
    if let Some(arguments) = protein {
        if let Some(identifier) = &arguments.pdb_id {
            // PDBe EDS is the stable primary-map source used by the cached
            // scientific regressions.  Automatic acquisition must not switch
            // to the RCSB converted 2Fo-Fc representation merely because it
            // happens to be available in the cache: those maps have different
            // axes, normalization, and detail settings and therefore define a
            // different optimization problem.  RCSB remains an explicit
            // expert choice; it is only a fallback for `auto` when EDS cannot
            // be acquired.
            if matches!(source, DensityMapSourceArg::Rcsb) {
                let cache_dir = arguments.protein_cache.join("maps");
                match fetch_2fo_fc_map(identifier, &cache_dir, detail) {
                    Ok(acquisition) => {
                        return Ok((acquisition.converted_map_path.clone(), Some(acquisition)));
                    }
                    Err(error) => return Err(anyhow::anyhow!(error.to_string())),
                }
            }
            let provider = ProteinProvider::new(ProteinFetchOptions {
                cache_dir: arguments.protein_cache.join("maps"),
                offline: arguments.protein_offline,
                ..ProteinFetchOptions::default()
            })?;
            match provider.fetch_eds_map(identifier) {
                Ok(path) => return Ok((path, None)),
                Err(error) if matches!(source, DensityMapSourceArg::Auto) => {
                    // Automatic mode is deterministic when EDS is present,
                    // but still recovers gracefully for entries without an
                    // EDS map by trying the explicitly documented RCSB
                    // primary coefficients.
                    eprintln!(
                        "refine: warning: PDBe EDS acquisition failed ({error}); trying RCSB 2Fo-Fc"
                    );
                    let cache_dir = arguments.protein_cache.join("maps");
                    let acquisition = fetch_2fo_fc_map(identifier, &cache_dir, detail)
                        .map_err(|rcsb_error| {
                            anyhow::anyhow!(
                                "PDBe EDS failed ({error}); RCSB 2Fo-Fc fallback failed ({rcsb_error})"
                            )
                        })?;
                    return Ok((acquisition.converted_map_path.clone(), Some(acquisition)));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    let path = resolve_density_map(value, input, protein)?;
    Ok((path, None))
}

fn load_search(
    arguments: &SearchCommon,
) -> anyhow::Result<(
    glysys::Structure,
    Vec<SearchSite>,
    SystemBuilder,
    SearchConfig,
)> {
    load_search_with_anomers(arguments, None)
}

fn load_search_with_anomers(
    arguments: &SearchCommon,
    anomer_overrides: Option<&BTreeMap<ResidueId, Anomer>>,
) -> anyhow::Result<(
    glysys::Structure,
    Vec<SearchSite>,
    SystemBuilder,
    SearchConfig,
)> {
    if arguments.attachments.is_empty() {
        anyhow::bail!("provide at least one --attach SITE=GLYCAN value");
    }
    let options = dry_options(false);
    let protein = fetch_protein(&arguments.protein, &options)?.structure;
    let mut ensemble_cache = BTreeMap::<String, reglyco_core::GlycanEnsemble>::new();
    let sites = arguments
        .attachments
        .iter()
        .map(|attachment| {
            let (site, source) = attachment
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--attach must use SITE=GLYCAN syntax"))?;
            let site = parse_site(site)?;
            let query = query_for_source_with_anomer(
                source,
                &arguments.provider,
                anomer_overrides.and_then(|overrides| overrides.get(&site)),
            );
            let cache_key = match &query.source {
                GlycanSource::LocalBundle(path) => format!("local:{}", path.display()),
                GlycanSource::GlyTouCan(identifier) => format!(
                    "glytoucan:{}:{}:{}",
                    identifier.to_ascii_uppercase(),
                    match query.anomer {
                        Anomer::Alpha => "alpha",
                        Anomer::Beta => "beta",
                        Anomer::Unknown => "unknown",
                    },
                    query.level
                ),
            };
            let ensemble = if let Some(ensemble) = ensemble_cache.get(&cache_key) {
                ensemble.clone()
            } else {
                let ensemble = load_ensemble(&query, &arguments.provider)?;
                ensemble_cache.insert(cache_key, ensemble.clone());
                ensemble
            };
            Ok(SearchSite {
                site: GlycosylationSite { residue: site },
                ensemble,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let builder = SystemBuilder::new(options)?;
    let config = SearchConfig {
        seed: arguments.seed,
        population_size: arguments.population,
        generations: arguments.generations,
        require_clash_free: !arguments.allow_clashes,
        scan_rotamers: arguments.rotamers,
        ..energy_search_config(&arguments.energy)?
    };
    Ok((protein, sites, builder, config))
}

fn discover_glycan_sites(
    structure: &glysys::Structure,
    provider: &ProviderArgs,
) -> anyhow::Result<Vec<DiscoveredGlycanSite>> {
    let pdb = structure.to_pdb_string();
    let extracted = extract_glycans_with_provenance_from_str(&pdb, false)
        .map_err(|error| anyhow::anyhow!("crabWURCS glycan extraction failed: {error}"))?;
    let attachment_links = pdb_attachment_links(&pdb, structure);
    let client = reqwest::blocking::Client::builder()
        .user_agent("reglyco-rs/0.1")
        .build()?;
    let mut cache = BTreeMap::<String, (String, String, Option<String>)>::new();
    let mut discovered = Vec::new();
    for glycan in extracted {
        let Some(attachment) = glycan.attachment_site.as_deref() else {
            continue;
        };
        let mut fields = attachment.split('/');
        let (Some(chain), Some(_sugar_name), Some(number)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let sugar_root = parse_site(&format!("{chain}:{number}"))?;
        let Some(link) = attachment_links.get(&sugar_root).cloned() else {
            // A disconnected sugar (for example the FRU ligand in 6S2G) is
            // intentionally not a remodeling site.
            continue;
        };
        let site = link.protein_residue.clone();
        let Some(protein_residue) = structure
            .residues()
            .into_iter()
            .find(|residue| residue.id == site)
        else {
            continue;
        };
        // CrabWURCS can report an external carbohydrate/ligand contact as an
        // attachment when the source PDB contains unusual LINK records. It is
        // not a remodeling site unless the external residue is a protein
        // residue with a supported glycosylation atom.
        if !matches!(
            protein_residue.name.as_str(),
            "ASN" | "SER" | "THR" | "TRP" | "HYP" | "PRO"
        ) {
            continue;
        }
        // CrabWURCS only exposes an external attachment for a sugar. The
        // linkage definition is the final chemistry gate (and excludes
        // sugar–ligand or unsupported protein contacts).
        let _ = LinkageDefinition::for_residue(&site, &protein_residue.name)
            .map_err(|error| anyhow::anyhow!("unsupported glycan attachment at {site}: {error}"))?;
        if structure.find_atom(&site, &link.protein_atom).is_none() {
            anyhow::bail!(
                "protein-linked glycan at {site} references missing atom {}",
                link.protein_atom
            );
        }
        if glycan.graph.root().is_none() || glycan.residues.is_empty() {
            continue;
        }
        let canonical = write_wurcs_canonical(&glycan.graph)
            .map_err(|error| anyhow::anyhow!("cannot canonicalize glycan at {site}: {error}"))?;
        let topology_hash = format!("{:x}", Sha256::digest(canonical.as_bytes()));
        let (glytoucan, iupac, glycoshape_id) = if let Some(value) = cache.get(&canonical) {
            value.clone()
        } else {
            let resolver_url = format!("{}/api/resolve", provider.api_base.trim_end_matches('/'));
            let cache_dir = provider.cache.join("topology-resolutions");
            let cache_path = cache_dir.join(format!("{topology_hash}.json"));
            let from_cache = cache_path.is_file();
            let mut value: serde_json::Value = if from_cache {
                serde_json::from_slice(&std::fs::read(&cache_path)?)?
            } else {
                if provider.offline {
                    anyhow::bail!(
                        "GlycoShape topology cache miss for {canonical} in offline mode ({})",
                        cache_path.display()
                    );
                }
                let response = client
                    .post(&resolver_url)
                    .json(&serde_json::json!({ "identifier": canonical }))
                    .send()?
                    .error_for_status()?;
                response.json()?
            };
            if let Some(error) = value.get("error").and_then(serde_json::Value::as_str)
                && !error.trim().is_empty()
            {
                anyhow::bail!("GlycoShape resolver rejected {canonical}: {error}");
            }
            // Some GlycoShape database records use a reducing-end unknown
            // marker (`u2122`) where crabWURCS preserves the declared root
            // anomer. Retry through the resolver's canonical IUPAC endpoint;
            // this keeps exact graph chemistry while avoiding a notation-only
            // WURCS mismatch.
            if value
                .get("glytoucan")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|value| value.trim().is_empty())
            {
                let iupac = write_iupac_condensed_canonical(&glycan.graph)?;
                if provider.offline && from_cache {
                    anyhow::bail!(
                        "GlycoShape topology cache contains no resolvable identifier for {canonical}"
                    );
                }
                value = client
                    .post(&resolver_url)
                    .json(&serde_json::json!({ "identifier": iupac }))
                    .send()?
                    .error_for_status()?
                    .json()?;
            }
            let resolved_wurcs = value
                .get("wurcs")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("resolver returned no WURCS for {canonical}"))?;
            let reparsed = parse_wurcs(resolved_wurcs)
                .map_err(|error| anyhow::anyhow!("resolver returned invalid WURCS: {error}"))?;
            let deposited_iupac = write_iupac_condensed_canonical(&glycan.graph)?;
            let resolved_iupac = write_iupac_condensed_canonical(&reparsed)?;
            // Run this check for both network responses and cached resolver
            // records.  A stale or manually edited cache must never bypass
            // the atomic preflight and cause a topology-incompatible
            // ensemble to be attached to a discovered site.
            if resolved_iupac != deposited_iupac {
                anyhow::bail!(
                    "GlycoShape topology mismatch for {site}: deposited {deposited_iupac}, resolver {resolved_iupac}"
                );
            }
            let glytoucan = value
                .get("glytoucan")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| anyhow::anyhow!("no GlyTouCan identifier for {canonical}"))?
                .to_ascii_uppercase();
            let iupac = value
                .get("iupac")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    write_iupac_condensed_canonical(&glycan.graph)
                        .unwrap_or_else(|_| canonical.clone())
                });
            let glycoshape_id = value
                .get("glycoshape_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let resolved = (glytoucan, iupac, glycoshape_id);
            std::fs::create_dir_all(&cache_dir)?;
            std::fs::write(&cache_path, serde_json::to_vec_pretty(&value)?)?;
            cache.insert(canonical.clone(), resolved.clone());
            resolved
        };
        let root_anomer = glycan
            .graph
            .root()
            .and_then(|root| glycan.graph.residue(root))
            .map(|residue| match residue.anomeric_symbol {
                AnomericSymbol::Alpha => Anomer::Alpha,
                AnomericSymbol::Beta => Anomer::Beta,
                _ => Anomer::Unknown,
            })
            .unwrap_or(Anomer::Unknown);
        let source_residues = glycan
            .residues
            .iter()
            .map(|residue| format!("{}:{}", residue.chain, residue.sequence_number))
            .collect::<Vec<_>>();
        discovered.push(DiscoveredGlycanSite {
            site,
            topology_hash,
            canonical_wurcs: canonical,
            iupac,
            glytoucan,
            glycoshape_id,
            root_anomer,
            source_residues,
            residue_count: glycan.residues.len(),
        });
    }
    discovered.sort_by(|left, right| left.site.cmp(&right.site));
    discovered.dedup_by(|left, right| left.site == right.site);
    if discovered.is_empty() {
        anyhow::bail!("no supported protein-linked carbohydrate attachment was found");
    }
    Ok(discovered)
}

fn pdb_attachment_links(
    pdb: &str,
    structure: &glysys::Structure,
) -> BTreeMap<ResidueId, glysys::GlycosylationSite> {
    let residues = structure.residues();
    let mut links = BTreeMap::new();
    for line in pdb.lines().filter(|line| line.starts_with("LINK  ")) {
        let Some(first_chain) = line
            .get(21..22)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let Some(first_number) = line.get(22..26).and_then(|value| value.trim().parse().ok())
        else {
            continue;
        };
        let first_insertion_code = line
            .get(26..27)
            .and_then(|value| value.chars().next())
            .filter(|value| !value.is_whitespace());
        let Some(second_chain) = line
            .get(51..52)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let Some(second_number) = line.get(52..56).and_then(|value| value.trim().parse().ok())
        else {
            continue;
        };
        let second_insertion_code = line
            .get(56..57)
            .and_then(|value| value.chars().next())
            .filter(|value| !value.is_whitespace());
        let first = ResidueId {
            chain: first_chain.to_string(),
            number: first_number,
            insertion_code: first_insertion_code,
        };
        let second = ResidueId {
            chain: second_chain.to_string(),
            number: second_number,
            insertion_code: second_insertion_code,
        };
        let first_name = residues
            .iter()
            .find(|residue| residue.id == first)
            .map(|residue| residue.name.as_str());
        let second_name = residues
            .iter()
            .find(|residue| residue.id == second)
            .map(|residue| residue.name.as_str());
        let first_protein = first_name
            .is_some_and(|name| matches!(name, "ASN" | "SER" | "THR" | "TRP" | "HYP" | "PRO"));
        let second_protein = second_name
            .is_some_and(|name| matches!(name, "ASN" | "SER" | "THR" | "TRP" | "HYP" | "PRO"));
        let (protein_residue, glycan_residue, protein_atom, glycan_atom) =
            match (first_protein, second_protein) {
                (true, false) => (
                    first,
                    second,
                    line.get(12..16).unwrap_or("ND2").trim().to_string(),
                    line.get(42..46).unwrap_or("C1").trim().to_string(),
                ),
                (false, true) => (
                    second,
                    first,
                    line.get(42..46).unwrap_or("ND2").trim().to_string(),
                    line.get(12..16).unwrap_or("C1").trim().to_string(),
                ),
                _ => continue,
            };
        links.insert(
            glycan_residue.clone(),
            glysys::GlycosylationSite {
                protein_residue,
                protein_atom,
                glycan_residue,
                glycan_atom,
            },
        );
    }
    links
}

fn load_ensemble(
    query: &GlycanQuery,
    provider: &ProviderArgs,
) -> anyhow::Result<reglyco_core::GlycanEnsemble> {
    match &query.source {
        GlycanSource::LocalBundle(_) => Ok(LocalBundleProvider.load(query)?),
        GlycanSource::GlyTouCan(_) => {
            let remote = GlycoShapeProvider::new(&provider.api_base)?;
            Ok(CachingProvider::new(remote, &provider.cache, provider.offline).load(query)?)
        }
    }
}

/// Load identical glycan sources once per command. Cloning a parsed ensemble
/// is substantially cheaper than reparsing its multi-model PDB and preserves
/// the existing owned `SearchSite` API.
fn load_search_sites(
    specifications: &[String],
    provider: &ProviderArgs,
) -> anyhow::Result<Vec<SearchSite>> {
    load_search_sites_with_format(specifications, provider, ResidueNameFormat::Pdb)
}

fn load_search_sites_with_format(
    specifications: &[String],
    provider: &ProviderArgs,
    format: ResidueNameFormat,
) -> anyhow::Result<Vec<SearchSite>> {
    let mut cache = HashMap::<(String, ResidueNameFormat), reglyco_core::GlycanEnsemble>::new();
    specifications
        .iter()
        .map(|attachment| {
            let (site, source) = attachment
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--attach must use SITE=GLYCAN syntax"))?;
            let cache_key = (source.to_owned(), format);
            let ensemble = if let Some(ensemble) = cache.get(&cache_key) {
                ensemble.clone()
            } else {
                let query = query_for_source_with_format(source, provider, format);
                let ensemble = load_ensemble(&query, provider)?;
                cache.insert(cache_key, ensemble.clone());
                ensemble
            };
            Ok(SearchSite {
                site: GlycosylationSite {
                    residue: parse_site(site)?,
                },
                ensemble,
            })
        })
        .collect()
}

fn fetch_protein(
    arguments: &ProteinArgs,
    options: &BuildOptions,
) -> anyhow::Result<FetchedProtein> {
    let source = match (
        arguments.protein.as_ref(),
        arguments.uniprot.as_ref(),
        arguments.pdb_id.as_ref(),
    ) {
        (Some(path), None, None) => ProteinSource::Local(path.clone()),
        (None, Some(accession), None) => ProteinSource::AlphaFold(accession.clone()),
        (None, None, Some(identifier)) => {
            if arguments.asymmetric_unit {
                ProteinSource::Pdb(identifier.clone())
            } else {
                Some(arguments.assembly.unwrap_or(1)).map_or_else(
                    || ProteinSource::Pdb(identifier.clone()),
                    |assembly| ProteinSource::PdbAssembly {
                        identifier: identifier.clone(),
                        assembly,
                    },
                )
            }
        }
        _ => anyhow::bail!("provide exactly one of --protein, --uniprot, or --pdb-id"),
    };
    let provider = ProteinProvider::new(ProteinFetchOptions {
        cache_dir: arguments.protein_cache.clone(),
        offline: arguments.protein_offline,
        ..ProteinFetchOptions::default()
    })?;
    Ok(provider.fetch(&source, options)?)
}

fn query_for_source(source: &str, provider: &ProviderArgs) -> GlycanQuery {
    query_for_source_with_format(source, provider, ResidueNameFormat::Pdb)
}

fn query_for_source_with_format(
    source: &str,
    provider: &ProviderArgs,
    format: ResidueNameFormat,
) -> GlycanQuery {
    query_for_source_with_anomer_and_format(source, provider, None, format)
}

fn query_for_source_with_anomer(
    source: &str,
    provider: &ProviderArgs,
    anomer_override: Option<&Anomer>,
) -> GlycanQuery {
    query_for_source_with_anomer_and_format(
        source,
        provider,
        anomer_override,
        ResidueNameFormat::Pdb,
    )
}

fn query_for_source_with_anomer_and_format(
    source: &str,
    provider: &ProviderArgs,
    anomer_override: Option<&Anomer>,
    format: ResidueNameFormat,
) -> GlycanQuery {
    let path = PathBuf::from(source);
    GlycanQuery {
        source: if path.exists() {
            GlycanSource::LocalBundle(path)
        } else {
            GlycanSource::GlyTouCan(source.into())
        },
        anomer: anomer_override.cloned().unwrap_or_else(|| {
            if provider.anomer.eq_ignore_ascii_case("alpha") {
                Anomer::Alpha
            } else {
                Anomer::Beta
            }
        }),
        format: format.api_segment().into(),
        level: provider.level.clone(),
    }
}

fn relax_options(arguments: &RelaxCommon) -> RelaxOptions {
    let mut options = RelaxOptions {
        movable: match arguments.movable {
            MovableArg::Glycans => MovableSelection::Glycans,
            MovableArg::All => MovableSelection::All,
        },
        include_local_sidechains: arguments.local_sidechains,
        local_radius: arguments.local_radius,
        nonbonded_cutoff: arguments.nonbonded_cutoff,
        ..RelaxOptions::default()
    };
    options.lbfgs.max_iterations = arguments.max_iterations;
    options.obc2 = arguments.obc2.then(Default::default);
    options
}

fn energy_search_config(arguments: &EnergyArgs) -> anyhow::Result<SearchConfig> {
    let scoring_mode = if arguments.interact {
        SearchScoringMode::ProteinGlycanInteraction
    } else if arguments.energy {
        SearchScoringMode::FullEnergy
    } else {
        SearchScoringMode::StericPrior
    };
    if arguments.obc2 && scoring_mode == SearchScoringMode::ProteinGlycanInteraction {
        anyhow::bail!("--obc2 cannot be used with --interact");
    }
    if (arguments.obc2 || arguments.min) && scoring_mode == SearchScoringMode::StericPrior {
        anyhow::bail!("--energy-obc2 and --min require --energy or --interact");
    }
    if arguments.temperature_k <= 0.0
        || arguments.thinning == 0
        || arguments.min_radius <= 0.0
        || arguments.energy_cutoff <= 0.0
    {
        anyhow::bail!("--temperature-k and --thinning must be positive");
    }
    Ok(SearchConfig {
        scoring_mode,
        use_obc2: arguments.obc2,
        pre_minimization: arguments.min,
        pre_minimization_iterations: arguments.min_iterations,
        minimization_radius: arguments.min_radius,
        energy_cutoff: arguments.energy_cutoff,
        temperature_k: arguments.temperature_k,
        burn_in: arguments.burn_in,
        thinning: arguments.thinning,
        ..SearchConfig::default()
    })
}

fn print_search_progress(event: SearchProgress, quiet: bool) {
    print_search_progress_label(event, quiet, "build");
}

fn print_search_progress_label(event: SearchProgress, quiet: bool, label: &str) {
    if quiet {
        return;
    }
    match event {
        SearchProgress::PreparingTopology => {
            eprintln!("{label}: preparing reusable force-field topology...")
        }
        SearchProgress::TopologyReady { seconds, atoms } => {
            eprintln!(
                "{label}: topology ready: {atoms} atoms in {seconds:.2}s; starting parallel GA"
            )
        }
        SearchProgress::Generation {
            phase,
            generation,
            best_score,
            mean_score,
            evaluations,
            cache_hits,
            steric_rejections,
            elapsed_seconds,
            ..
        } => eprintln!(
            "{label}: {} generation {generation}: best={best_score:.4}, mean={mean_score:.4}, energy-evals={evaluations}, cache-hits={cache_hits}, steric-rejected={steric_rejections}, elapsed={elapsed_seconds:.1}s",
            match phase {
                SearchPhase::Feasibility => "feasibility",
                SearchPhase::Compatibility => "compatibility",
                SearchPhase::ProbabilityImprovement => "probability",
            }
        ),
    }
}

fn print_refine_progress(event: RefineProgress, quiet: bool, started: Instant) {
    if quiet {
        return;
    }
    match event {
        RefineProgress::Phase { name } => eprintln!(
            "refine: {name} (elapsed {:.1}s)",
            started.elapsed().as_secs_f64()
        ),
        RefineProgress::Search(event) => {
            print_search_progress_label(event, false, "refine: search")
        }
        RefineProgress::Relax(event) => match event {
            RelaxProgress::InitialEnergyStarted => {
                eprintln!("refine: relaxation evaluating initial energy...")
            }
            RelaxProgress::InitialEnergy { energy } => {
                eprintln!(
                    "refine: relaxation initial energy {:.3} kcal/mol",
                    energy.total()
                )
            }
            RelaxProgress::StageStarted {
                name,
                movable_atoms,
            } => eprintln!(
                "refine: relaxation stage {name}: {movable_atoms} movable atoms; evaluating..."
            ),
            RelaxProgress::Iteration {
                stage,
                iteration,
                energy,
                rms_gradient,
                max_gradient,
                accepted_steps,
            } if iteration <= 3 || iteration.is_multiple_of(10) => eprintln!(
                "refine: relaxation {stage} iteration {iteration} (accepted {accepted_steps}): objective E={energy:.3} kcal/mol, rms|g|={rms_gradient:.4}, max|g|={max_gradient:.4}"
            ),
            RelaxProgress::Iteration { .. } => {}
            RelaxProgress::StageFinished { name, diagnostics } => eprintln!(
                "refine: relaxation stage {name} finished after {} iterations ({}, full E={:.3} kcal/mol)",
                diagnostics.iterations,
                diagnostics.convergence_reason,
                diagnostics.final_energy.total()
            ),
        },
    }
}

fn validate_cookbook_scan_config(population: usize, generations: usize) -> anyhow::Result<()> {
    if !(32..=512).contains(&population) {
        anyhow::bail!(
            "--population must be in 32..=512 for Cookbook-compatible scan jobs (received {population})"
        );
    }
    if !(1..=20_000).contains(&generations) {
        anyhow::bail!(
            "--generations must be in 1..=20000 for Cookbook-compatible scan jobs (received {generations})"
        );
    }
    Ok(())
}

fn glycan_atom_count(structure: &glysys::Structure) -> usize {
    structure
        .atoms()
        .iter()
        .filter(|atom| {
            structure
                .metadata()
                .glycan_trees
                .iter()
                .any(|tree| tree.residue_ids.contains(&atom.residue))
        })
        .count()
}

fn verify_relaxed_glycans(
    expected: &glysys::Structure,
    written: &glysys::Structure,
) -> anyhow::Result<()> {
    let expected_atoms = glycan_atom_count(expected);
    let written_atoms = glycan_atom_count(written);
    let expected_sites = expected.metadata().glycosylation_sites.len();
    let written_sites = written.metadata().glycosylation_sites.len();
    if expected_atoms != written_atoms || expected_sites != written_sites {
        anyhow::bail!(
            "relaxation output failed glycan-integrity verification: expected {expected_atoms} glycan atoms and {expected_sites} attachment(s), wrote {written_atoms} and {written_sites}"
        );
    }
    Ok(())
}

fn build_options(overwrite: bool, no_water: bool, no_ions: bool) -> BuildOptions {
    let mut options = BuildOptions {
        overwrite,
        ..BuildOptions::default()
    };
    if no_water {
        options.add_water = false;
        options.add_ions = false;
    } else if no_ions {
        options.add_ions = false;
    }
    options
}

fn dry_options(overwrite: bool) -> BuildOptions {
    BuildOptions {
        overwrite,
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    }
}

fn prepare_output(path: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    Ok(())
}

fn check_output(path: &Path, overwrite: bool) -> anyhow::Result<()> {
    if path.exists() && !overwrite {
        anyhow::bail!(
            "{} already exists; pass --overwrite to replace it",
            path.display()
        );
    }
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl serde::Serialize) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

fn write_workflow_report(
    report: &WorkflowReport,
    output: &Path,
    render_pdf: bool,
) -> anyhow::Result<()> {
    if render_pdf {
        report.write_bundle(output)?;
    } else {
        report.write_json(output.join("report.json"))?;
    }
    Ok(())
}

fn print_clash_status(status: reglyco_core::ClashStatus) {
    match status {
        reglyco_core::ClashStatus::ClashFree => println!("Selected a clash-free complete result"),
        reglyco_core::ClashStatus::BestCompleteClashing => {
            println!("WARNING: selected the best complete result, but clashes remain")
        }
    }
}

fn parse_site(value: &str) -> anyhow::Result<ResidueId> {
    let (chain, residue) = value
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("site must use CHAIN:NUMBER syntax"))?;
    if chain.chars().count() != 1 {
        anyhow::bail!("site chain must be one character");
    }
    let digit_count = residue.chars().take_while(char::is_ascii_digit).count();
    if digit_count == 0 {
        anyhow::bail!("site residue number is missing");
    }
    let number = residue[..digit_count].parse()?;
    let insertion = &residue[digit_count..];
    let insertion_code = match insertion.chars().collect::<Vec<_>>().as_slice() {
        [] => None,
        [code] => Some(*code),
        _ => anyhow::bail!("site insertion code must be at most one character"),
    };
    Ok(ResidueId {
        chain: chain.into(),
        number,
        insertion_code,
    })
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use glysys::{AtomId, BuildOptions, GlycanTree, GlycosylationSite, ResidueId, read_pdb_str};

    use super::{
        Cli, Command, ObjectiveArg, OutputFormatArg, glycan_atom_count, parse_site,
        validate_cookbook_scan_config, verify_relaxed_glycans,
    };

    #[test]
    fn parses_site_selectors() {
        let site = parse_site("A:42B").unwrap();
        assert_eq!(site.chain, "A");
        assert_eq!(site.number, 42);
        assert_eq!(site.insertion_code, Some('B'));
        assert!(parse_site("AA:42").is_err());
        assert!(parse_site("A:x").is_err());
    }

    #[test]
    fn cookbook_scan_parameter_contract_is_enforced() {
        assert!(validate_cookbook_scan_config(128, 100).is_ok());
        assert!(validate_cookbook_scan_config(31, 100).is_err());
        assert!(validate_cookbook_scan_config(513, 100).is_err());
        assert!(validate_cookbook_scan_config(128, 0).is_err());
        assert!(validate_cookbook_scan_config(128, 20_001).is_err());
    }

    #[test]
    fn scan_uses_fixed_fast_defaults() {
        let cli = Cli::try_parse_from(["reglyco", "scan", "--output", "result"]).unwrap();
        let Command::Scan(arguments) = cli.command else {
            panic!("expected scan command");
        };
        assert_eq!(arguments.population, 32);
        assert_eq!(arguments.generations, 25);
    }

    #[test]
    fn build_and_ensemble_expose_shared_output_format_and_seed_defaults() {
        let cli = Cli::try_parse_from([
            "reglyco",
            "build",
            "--protein",
            "protein.pdb",
            "--site",
            "A:42",
            "--glycan",
            "G00028MO",
            "--output",
            "result",
            "--output-format",
            "glycam",
        ])
        .unwrap();
        let Command::Build(arguments) = cli.command else {
            panic!("expected build command");
        };
        assert_eq!(arguments.seed, 0);
        assert!(matches!(arguments.output_format, OutputFormatArg::Glycam));
        assert!(!arguments.require_clash_free);

        let strict = Cli::try_parse_from([
            "reglyco",
            "build",
            "--protein",
            "protein.pdb",
            "--site",
            "A:42",
            "--glycan",
            "G00028MO",
            "--output",
            "result",
            "--require-clash-free",
        ])
        .unwrap();
        let Command::Build(arguments) = strict.command else {
            panic!("expected build command");
        };
        assert!(arguments.require_clash_free);
        assert!(
            Cli::try_parse_from([
                "reglyco",
                "build",
                "--protein",
                "protein.pdb",
                "--site",
                "A:42",
                "--glycan",
                "G00028MO",
                "--output",
                "result",
                "--allow-clashes",
                "--require-clash-free",
            ])
            .is_err()
        );

        let cli = Cli::try_parse_from([
            "reglyco",
            "ensemble",
            "--protein",
            "protein.pdb",
            "--attach",
            "A:42=G00028MO",
            "--output",
            "result",
        ])
        .unwrap();
        let Command::Ensemble(arguments) = cli.command else {
            panic!("expected ensemble command");
        };
        assert_eq!(arguments.seed, 0);
        assert!(matches!(arguments.output_format, OutputFormatArg::Pdb));
    }

    #[test]
    fn search_budget_cli_requires_a_complete_manual_pair_and_preserves_compatibility() {
        let auto = Cli::try_parse_from([
            "reglyco",
            "build",
            "--protein",
            "protein.pdb",
            "--site",
            "A:42",
            "--glycan",
            "G00028MO",
            "--output",
            "result",
        ])
        .unwrap();
        let Command::Build(auto) = auto.command else {
            panic!("expected build command");
        };
        assert!(auto.search_budget.is_none());
        assert!(auto.population.is_none());
        assert!(auto.generations.is_none());

        let manual = Cli::try_parse_from([
            "reglyco",
            "build",
            "--protein",
            "protein.pdb",
            "--site",
            "A:42",
            "--glycan",
            "G00028MO",
            "--output",
            "result",
            "--population",
            "37",
            "--generations",
            "19",
        ])
        .unwrap();
        let Command::Build(manual) = manual.command else {
            panic!("expected build command");
        };
        assert_eq!(manual.population, Some(37));
        assert_eq!(manual.generations, Some(19));

        let explicit_manual = Cli::try_parse_from([
            "reglyco",
            "build",
            "--protein",
            "protein.pdb",
            "--site",
            "A:42",
            "--glycan",
            "G00028MO",
            "--output",
            "result",
            "--search-budget",
            "manual",
        ])
        .unwrap();
        let Command::Build(explicit_manual) = explicit_manual.command else {
            panic!("expected build command");
        };
        assert!(matches!(
            explicit_manual.search_budget,
            Some(super::SearchBudgetArg::Manual)
        ));
        assert!(explicit_manual.population.is_none());
        assert!(explicit_manual.generations.is_none());
    }

    fn density_refine(extra: &[&str]) -> super::RefineArgs {
        let mut args = vec![
            "reglyco",
            "refine",
            "--protein",
            "protein.pdb",
            "--objective",
            "density",
            "--density-map",
            "density.map",
            "--replace-glycan",
            "A:79",
            "--output",
            "result",
        ];
        args.extend_from_slice(extra);
        let cli = Cli::try_parse_from(args).unwrap();
        let Command::Refine(arguments) = cli.command else {
            panic!("expected refine command");
        };
        arguments
    }

    #[test]
    fn density_refine_uses_glycoflow_options() {
        let arguments =
            density_refine(&["--glycoflow-model", "model", "--glycoflow-samples", "64"]);
        assert!(matches!(arguments.objective, ObjectiveArg::Density));
        assert!(arguments.density_search.is_none());
        assert_eq!(
            arguments.glycoflow.model.as_deref(),
            Some(std::path::Path::new("model"))
        );
        assert_eq!(arguments.glycoflow.samples, 64);
    }

    #[test]
    fn legacy_density_search_values_are_rejected() {
        let error = super::run_refine(density_refine(&["--density-search", "adaptive"]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("legacy density fitter"), "{error}");
    }

    #[test]
    fn density_refine_requires_a_glycoflow_model() {
        if std::env::var_os("GLYCOFLOW_MODEL").is_some() {
            return;
        }
        let error = super::run_refine(density_refine(&[]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("--glycoflow-model"), "{error}");
    }

    #[test]
    fn pdb_id_defaults_to_assembly_one_and_supports_asymmetric_opt_out() {
        let cli = Cli::try_parse_from([
            "reglyco",
            "refine",
            "--pdb-id",
            "5GSQ",
            "--objective",
            "density",
            "--density-map",
            "map.ccp4",
            "--output",
            "result",
        ])
        .unwrap();
        let Command::Refine(arguments) = cli.command else {
            panic!("expected refine command");
        };
        assert_eq!(arguments.common.provider.level, "1");
        assert_eq!(arguments.common.provider.api_base, "https://glycoshape.org");
        assert!(!arguments.common.protein.asymmetric_unit);
        let cli = Cli::try_parse_from([
            "reglyco",
            "refine",
            "--pdb-id",
            "5GSQ",
            "--asymmetric-unit",
            "--objective",
            "density",
            "--density-map",
            "map.ccp4",
            "--output",
            "result",
        ])
        .unwrap();
        let Command::Refine(arguments) = cli.command else {
            panic!("expected refine command");
        };
        assert!(arguments.common.protein.asymmetric_unit);
    }

    #[test]
    fn attached_ensemble_keeps_protein_fixed_unless_requested() {
        let base = [
            "reglyco",
            "ensemble",
            "--protein",
            "protein.pdb",
            "--attach",
            "A:42=G00028MO",
            "--output",
            "result",
        ];
        let cli = Cli::try_parse_from(base).unwrap();
        let Command::Ensemble(arguments) = cli.command else {
            panic!("expected ensemble command");
        };
        assert!(!arguments.move_sidechains);

        let cli = Cli::try_parse_from([
            "reglyco",
            "ensemble",
            "--protein",
            "protein.pdb",
            "--attach",
            "A:42=G00028MO",
            "--output",
            "result",
            "--rotamers",
        ])
        .unwrap();
        let Command::Ensemble(arguments) = cli.command else {
            panic!("expected ensemble command");
        };
        assert!(arguments.move_sidechains);
    }

    #[test]
    fn attached_ensemble_accepts_cookbook_sasa_outputs() {
        let cli = Cli::try_parse_from([
            "reglyco",
            "ensemble",
            "--protein",
            "protein.pdb",
            "--attach",
            "A:42=G00028MO",
            "--output",
            "result",
            "--calculate-sasa",
            "--calculate-hotspots",
        ])
        .unwrap();
        let Command::Ensemble(arguments) = cli.command else {
            panic!("expected ensemble command");
        };
        assert!(arguments.calculate_sasa);
        assert!(arguments.calculate_hotspots);

        assert!(
            Cli::try_parse_from([
                "reglyco",
                "ensemble",
                "--glycan",
                "G00028MO",
                "--output",
                "result",
                "--calculate-hotspots",
            ])
            .is_err()
        );
    }

    #[test]
    fn attached_ensemble_uses_constrained_fallback_controls_not_attempt_limit() {
        let cli = Cli::try_parse_from([
            "reglyco",
            "ensemble",
            "--protein",
            "protein.pdb",
            "--attach",
            "A:42=G00028MO",
            "--frames",
            "12",
            "--chains",
            "3",
            "--burn-in-sweeps",
            "7",
            "--thinning-accepted",
            "11",
            "--output",
            "result",
        ])
        .unwrap();
        let Command::Ensemble(arguments) = cli.command else {
            panic!("expected ensemble command");
        };
        assert_eq!(arguments.frames, 12);
        assert_eq!(arguments.chains, 3);
        assert_eq!(arguments.burn_in_sweeps, 7);
        assert_eq!(arguments.thinning_accepted, 11);
        assert!(
            Cli::try_parse_from([
                "reglyco",
                "ensemble",
                "--protein",
                "protein.pdb",
                "--attach",
                "A:42=G00028MO",
                "--max-attempts",
                "100",
                "--output",
                "result",
            ])
            .is_err()
        );
    }

    #[test]
    fn glycan_integrity_check_rejects_a_missing_attachment() {
        let options = BuildOptions::default();
        let mut expected = read_pdb_str(
            "ATOM      1  ND2 ASN A   1       0.000   0.000   0.000  1.00  0.00           N\n\
             HETATM    2  C1  NAG B   1       1.450   0.000   0.000  1.00  0.00           C\nEND\n",
            &options,
        )
        .unwrap();
        expected.add_bond(AtomId(1), AtomId(2)).unwrap();
        expected.add_glycosylation_site(GlycosylationSite {
            protein_residue: ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            },
            protein_atom: "ND2".into(),
            glycan_residue: ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            },
            glycan_atom: "C1".into(),
        });
        expected.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "B".into(),
            residue_ids: vec![ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            }],
            attachment_site: Some(ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            }),
        });
        let missing = read_pdb_str(
            "ATOM      1  ND2 ASN A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n",
            &options,
        )
        .unwrap();
        assert_eq!(glycan_atom_count(&expected), 1);
        assert!(verify_relaxed_glycans(&expected, &missing).is_err());
    }
}
