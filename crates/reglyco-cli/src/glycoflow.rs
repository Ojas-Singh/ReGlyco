//! `reglyco refine --objective density`: fit the glycan at each `--replace-glycan` site with the
//! frozen GlycoFlow model (crate `reglyco-glycoflow`).

use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};
use reglyco_density::DensityMap;
use reglyco_glycoflow::glycoflow_core::model::Precision;
use reglyco_glycoflow::workflow::{
    SiteRequest, WorkflowInput, WorkflowOptions, run, write_outputs,
};
use reglyco_glycoflow::{ComputeDevice, GlycoflowModel, SymmetryMode};

use super::{RefineArgs, dry_options, fetch_protein, parse_site, resolve_density_map_for_refine};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum GlycoflowDeviceArg {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum GlycoflowPrecisionArg {
    F32,
    Bf16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum GlycoflowSymmetryArg {
    Auto,
    Off,
}

/// GlycoFlow options of `refine --objective density`.
#[derive(Debug, Clone, Args)]
pub(crate) struct GlycoflowArgs {
    /// GlycoFlow model directory (glycoflow.safetensors, glycoflow.json, residue_library.json);
    /// default: $GLYCOFLOW_MODEL.
    #[arg(long = "glycoflow-model")]
    pub(crate) model: Option<PathBuf>,
    /// Device of the GlycoFlow network (cuda needs a build with reglyco-cli/glycoflow-cuda).
    #[arg(long = "glycoflow-device", value_enum, default_value_t = GlycoflowDeviceArg::Cpu)]
    pub(crate) device: GlycoflowDeviceArg,
    #[arg(long = "glycoflow-precision", value_enum, default_value_t = GlycoflowPrecisionArg::F32)]
    pub(crate) precision: GlycoflowPrecisionArg,
    /// Observation-guided GlycoFlow samples per site.
    #[arg(long = "glycoflow-samples", default_value_t = 384)]
    pub(crate) samples: usize,
    /// Heun steps of the guided GlycoFlow sampler.
    #[arg(long = "glycoflow-steps", default_value_t = 8)]
    pub(crate) steps: usize,
    /// Distinct basins refined per site.
    #[arg(long = "glycoflow-basins", default_value_t = 24)]
    pub(crate) basins: usize,
    /// Weight of the GlycoFlow marginal prior in the objective.
    #[arg(long = "glycoflow-prior-weight", default_value_t = 0.25)]
    pub(crate) prior_weight: f64,
    /// Weight of the protein/environment and intra-glycan contact energies in the final
    /// objective (the search uses 10 and continues to this weight; 0 disables the stage).
    #[arg(long = "glycoflow-clash-weight", default_value_t = 100.0)]
    pub(crate) clash_weight: f64,
    /// Fit without the GlycoFlow marginal prior.
    #[arg(long = "glycoflow-no-prior")]
    pub(crate) no_prior: bool,
    /// Crystal-symmetry expansion of the site environment (auto: full-cell crystallographic maps).
    #[arg(long = "glycoflow-symmetry", value_enum, default_value_t = GlycoflowSymmetryArg::Auto)]
    pub(crate) symmetry: GlycoflowSymmetryArg,
}

/// A GLYCAM condensed sequence (as opposed to a GlyTouCan accession or a bundle path).
fn is_glycam(source: &str) -> bool {
    source.ends_with("-OH") || source.ends_with("-OME")
}

pub(crate) fn run_refine_glycoflow(arguments: RefineArgs, started: Instant) -> anyhow::Result<()> {
    let quiet = arguments.common.quiet;
    let g = &arguments.glycoflow;
    if arguments.replace_glycans.is_empty() {
        anyhow::bail!(
            "--objective density fits the glycan at --replace-glycan SITE (or SITE=<GLYCAM sequence>)"
        );
    }
    // Fail before any download or parsing when no model is configured.
    let dir = GlycoflowModel::resolve_dir(g.model.as_deref()).map_err(|_| {
        anyhow::anyhow!(
            "--objective density needs a GlycoFlow model: pass --glycoflow-model <dir> or set \
             $GLYCOFLOW_MODEL to a directory with glycoflow.safetensors, glycoflow.json and \
             residue_library.json"
        )
    })?;
    if !dir.is_dir() {
        anyhow::bail!(
            "GlycoFlow model directory {} does not exist (--glycoflow-model / $GLYCOFLOW_MODEL)",
            dir.display()
        );
    }
    let sites = arguments
        .replace_glycans
        .iter()
        .map(|value| {
            let (site, sequence) = match value.split_once('=') {
                Some((site, source)) if is_glycam(source) => (site, Some(source.to_string())),
                Some((_, source)) => anyhow::bail!(
                    "--objective density takes --replace-glycan SITE or SITE=<GLYCAM sequence ending in -OH>, not {source:?}"
                ),
                None => (value.as_str(), None),
            };
            Ok(SiteRequest {
                residue: parse_site(site)?,
                sequence,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let output = &arguments.common.output;
    if output.join("fitted.pdb").exists() && !arguments.overwrite {
        anyhow::bail!(
            "{} already contains fitted.pdb; pass --overwrite to replace it",
            output.display()
        );
    }
    if !quiet {
        eprintln!("refine: loading protein...");
    }
    let fetched = fetch_protein(&arguments.common.protein, &dry_options(false))?;
    let source_path = arguments
        .common
        .protein
        .protein
        .clone()
        .or(fetched.cache_path.clone());
    let text = source_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok());
    let map_value = arguments.density_map.as_deref().unwrap_or("auto");
    let (map_path, _) = resolve_density_map_for_refine(
        map_value,
        arguments.common.protein.protein.as_deref(),
        Some(&arguments.common.protein),
        arguments.density_map_source,
        arguments.density_map_detail,
    )?;
    let map = DensityMap::open(&map_path)?;
    let device = match g.device {
        GlycoflowDeviceArg::Cpu => ComputeDevice::Cpu,
        GlycoflowDeviceArg::Cuda => ComputeDevice::Cuda,
    };
    let precision = match g.precision {
        GlycoflowPrecisionArg::F32 => Precision::F32,
        GlycoflowPrecisionArg::Bf16 => Precision::Bf16,
    };
    if !quiet {
        eprintln!(
            "refine: loading GlycoFlow model from {} ({device:?})...",
            dir.display()
        );
    }
    let model = GlycoflowModel::load(&dir, device, precision)?;
    let mut options = WorkflowOptions::default();
    options.fit.seed = arguments.common.seed;
    options.fit.n_samples = g.samples;
    options.fit.flow_steps = g.steps;
    options.fit.n_basins = g.basins;
    options.prior.enabled = !g.no_prior;
    options.problem.w_prior = g.prior_weight;
    options.fit.final_clash_weight = (g.clash_weight > 0.0).then_some(g.clash_weight);
    options.site.symmetry = match g.symmetry {
        GlycoflowSymmetryArg::Auto => SymmetryMode::Auto,
        GlycoflowSymmetryArg::Off => SymmetryMode::Off,
    };
    options.sigma = arguments.density_sigma;
    options.resolution = arguments.density_resolution;
    let input = WorkflowInput {
        structure: &fetched.structure,
        structure_text: text.as_deref(),
        map: &map,
        sites,
        model: &model,
        options,
    };
    if !quiet {
        eprintln!(
            "refine: GlycoFlow fitting of {} site(s) against {}...",
            input.sites.len(),
            map_path.display()
        );
    }
    let result = run(&input)?;
    write_outputs(&result, output)?;
    if !quiet {
        for fit in &result.fits {
            let best = &fit.outcome.basins[fit.outcome.best];
            let supported = fit.outcome.support.iter().filter(|s| s.supported).count() + 1;
            eprintln!(
                "refine: {} {}: objective {:.2} (loglik {:.2}, partial CC {:.3}), {}/{} residues supported, {} alternative(s), {} prior completion(s), {:.1}s",
                fit.site.label(),
                fit.site.sequence.sequence,
                best.terms.total,
                best.terms.loglik,
                best.terms.partial_cc,
                supported,
                fit.outcome.support.len() + 1,
                fit.outcome.alternatives.len().saturating_sub(1),
                fit.outcome.completions.len(),
                fit.outcome.wall_seconds
            );
            if let Some(r) = &fit.recovery {
                eprintln!(
                    "refine: {} vs deposited (evaluation only): full {:.2} A, core {:.2} A",
                    fit.site.label(),
                    r.full_rmsd,
                    r.core_rmsd
                );
            }
        }
        if let Some(v) = &result.validation {
            eprintln!(
                "refine: validation {} ({} errors, {} warnings)",
                if v.valid { "valid" } else { "invalid" },
                v.errors.len(),
                v.warnings.len()
            );
        }
        eprintln!(
            "refine: wrote fitted.pdb, candidates.pdb, glycoflow-fit.json, validation.json to {} in {:.1}s",
            output.display(),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn glycam_sources() {
        assert!(super::is_glycam("DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH"));
        assert!(!super::is_glycam("G00028MO"));
    }
}
