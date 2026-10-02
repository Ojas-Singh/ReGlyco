//! End-to-end workflow: sites of a model + a density map -> fitted structure, candidate models,
//! JSON report and validation. Used by `reglyco refine --objective density --density-search
//! glycoflow`.

use std::path::Path;
use std::time::Instant;

use glycoflow_core::sampler::Sampler;
use glysys::{ResidueId, Structure};
use reglyco_density::{DensityMap, DensityScoreOptions, DensityScorer, DensityTarget};
use reglyco_validate::{ValidationOptions, ValidationReport, validate_with_density};
use serde_json::{Value, json};

use crate::error::{Result, invalid, write_file};
use crate::evaluation::{Recovery, recovery};
use crate::model::GlycoflowModel;
use crate::output::{
    PlacedGlycan, ResidueNaming, candidates_pdb, fitted_structure, residue_naming,
};
use crate::pipeline::{
    FitConfig, FitOutcome, PriorConfig, SigmaCalibration, build_prior, calibrate_sigma,
    density_problem, fit_site,
};
use crate::problem::{ProblemOptions, SiteProblem};
use crate::site::{CrystalInput, Site, SiteOptions, deposited_glycan, load_site};
use crate::symmetry::{UnitCell, parse_cryst1, parse_resolution};

/// One site to fit.
#[derive(Debug, Clone)]
pub struct SiteRequest {
    pub residue: ResidueId,
    /// GLYCAM sequence to fit; default: the deposited glycan's
    pub sequence: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct WorkflowOptions {
    pub fit: FitConfig,
    pub prior: PriorConfig,
    pub problem: ProblemOptions,
    pub site: SiteOptions,
    /// atom width; default: calibrated on the protein around each site
    pub sigma: Option<f64>,
    /// map resolution; default: the model's REMARK 2 record
    pub resolution: Option<f64>,
}

pub struct WorkflowInput<'a> {
    /// the input model, deposited glycans included
    pub structure: &'a Structure,
    /// raw PDB text of the input (CRYST1 and REMARK 2 records)
    pub structure_text: Option<&'a str>,
    pub map: &'a DensityMap,
    pub sites: Vec<SiteRequest>,
    pub model: &'a GlycoflowModel,
    pub options: WorkflowOptions,
}

/// Everything known about one fitted site.
pub struct SiteFit {
    pub site: Site,
    pub problem: SiteProblem,
    pub outcome: FitOutcome,
    pub naming: Vec<ResidueNaming>,
    pub sigma: SigmaCalibration,
    pub sigma_source: String,
    pub resolution: f64,
    pub recovery: Option<Recovery>,
    pub preparation_seconds: f64,
    pub prior_seconds: f64,
    pub prior_network_evaluations: usize,
}

pub struct WorkflowResult {
    pub fits: Vec<SiteFit>,
    /// protein with every target glycan removed
    pub protein: Structure,
    pub fitted: Structure,
    pub candidates_pdb: String,
    pub report: Value,
    pub validation: Option<ValidationReport>,
    pub validation_error: Option<String>,
}

/// Peak resident memory of this process (MB), Linux only.
pub fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn crystal_input(text: Option<&str>, map: &DensityMap) -> CrystalInput {
    let m = map.metadata();
    CrystalInput {
        cryst1: text.and_then(parse_cryst1),
        map_cell: UnitCell::new(
            m.cell_lengths_angstrom[0],
            m.cell_lengths_angstrom[1],
            m.cell_lengths_angstrom[2],
            m.cell_angles_degrees[0],
            m.cell_angles_degrees[1],
            m.cell_angles_degrees[2],
        ),
        map_space_group: Some(m.space_group),
        map_full_cell: map.is_full_unit_cell(),
    }
}

/// Fit one site.
pub fn fit_one(
    input: &WorkflowInput,
    request: &SiteRequest,
    protein: &Structure,
) -> Result<SiteFit> {
    let options = &input.options;
    let t0 = Instant::now();
    let crystal = crystal_input(input.structure_text, input.map);
    let site_options = SiteOptions {
        sequence: request.sequence.clone(),
        ..options.site.clone()
    };
    let site = load_site(input.structure, &request.residue, &crystal, &site_options)?;
    let resolution = options
        .resolution
        .or_else(|| input.structure_text.and_then(parse_resolution))
        .ok_or_else(|| invalid("map resolution unknown: the model has no REMARK 2 RESOLUTION record; pass --density-resolution"))?;
    let (sigma, sigma_source) = match options.sigma {
        Some(s) => (
            SigmaCalibration {
                selected: s,
                curve: Vec::new(),
            },
            "given".to_string(),
        ),
        None => (
            calibrate_sigma(&site, input.map, resolution)?,
            "calibrated on the protein shell 4-12 A from the link atom".to_string(),
        ),
    };
    let mut problem = density_problem(
        &site,
        input.model,
        input.map,
        sigma.selected,
        resolution,
        &options.problem,
    )?;
    let sampler = Sampler::for_glycan(&input.model.net, &problem.glycan, &input.model.meta.vocab)?;
    let preparation_seconds = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut prior_network_evaluations = 0;
    if options.prior.enabled && problem.n_torsions > 0 {
        problem.prior = Some(build_prior(&problem, &sampler, &options.prior)?);
        prior_network_evaluations = options.prior.n_samples * options.prior.steps;
    }
    let prior_seconds = t1.elapsed().as_secs_f64();
    let outcome = fit_site(&problem, &sampler, &options.fit)?;
    let naming = residue_naming(&problem, &site, protein)?;
    let recovery = site.deposited.as_ref().map(|dep| {
        let claimed: Vec<String> = std::iter::once("r".to_string())
            .chain(
                outcome
                    .support
                    .iter()
                    .filter(|s| s.supported)
                    .map(|s| s.residue.clone()),
            )
            .collect();
        recovery(&problem, dep, &outcome.basins[outcome.best].x, &claimed)
    });
    Ok(SiteFit {
        site,
        problem,
        outcome,
        naming,
        sigma,
        sigma_source,
        resolution,
        recovery,
        preparation_seconds,
        prior_seconds,
        prior_network_evaluations,
    })
}

fn deg(v: f64) -> f64 {
    v.to_degrees()
}

fn site_report(f: &SiteFit, input: &WorkflowInput) -> Value {
    let o = &f.outcome;
    let p = &f.problem;
    let best = &o.basins[o.best];
    let lik = p.observation.as_ref();
    let basins: Vec<Value> = o
        .order
        .iter()
        .enumerate()
        .map(|(rank, &i)| {
            let b = &o.basins[i];
            json!({
                "rank": rank, "basin": i, "objective": b.terms.total, "unrefined_objective": b.unrefined,
                "terms": b.terms, "psi_N_deg": deg(b.pose.psi), "phi_N_deg": deg(b.pose.phi), "template": b.pose.template,
                "torsions_rad": b.pose.tau,
            })
        })
        .collect();
    let alternatives: Vec<Value> = o
        .alternatives
        .iter()
        .filter(|&&i| i != o.best)
        .map(|&i| {
            let b = &o.basins[i];
            json!({
                "basin": i,
                "label": "density-ambiguous alternative (distinct basin within the objective margin; not a calibrated probability)",
                "delta_objective": b.terms.total - best.terms.total,
                "rmsd_to_best": crate::search::rmsd(p, &b.x, &best.x),
                "terms": b.terms,
                "evaluation": f.site.deposited.as_ref().map(|d| recovery(p, d, &b.x, &[])),
            })
        })
        .collect();
    let mut completion_order: Vec<usize> = (0..o.completions.len()).collect();
    completion_order.sort_by(|&a, &b| {
        o.completions[a]
            .terms
            .total
            .total_cmp(&o.completions[b].terms.total)
    });
    let completions: Vec<Value> = completion_order
        .iter()
        .map(|&k| {
            let c = &o.completions[k];
            json!({
                "label": "prior-driven completion of unsupported residues (GlycoFlow inpainting with clash guidance; not fitted to density)",
                "objective": c.terms.total, "e_env": c.terms.e_env, "e_self": c.terms.e_self, "terms": c.terms,
            })
        })
        .collect();
    let density = json!({
        "kind": lik.kind(),
        "constants": crate::pipeline::density_constants(p),
        "radius": p.radius,
        "sigma": f.sigma.selected,
        "sigma_source": f.sigma_source,
        "sigma_curve": f.sigma.curve,
        "resolution": f.resolution,
        "grid_dims": p.grids.dims,
    });
    json!({
        "site": f.site.label(),
        "residue": f.site.residue_name,
        "sequence": f.site.sequence,
        "deposited_residues": f.site.deposited.as_ref().map(|d| d.residues.clone()),
        "naming": f.naming,
        "symmetry": f.site.symmetry,
        "environment_atoms": f.site.environment.len(),
        "site_pairs": p.site_pairs.len(),
        "atoms": p.n_atoms,
        "torsions": p.n_torsions,
        "templates": p.n_templates(),
        "density": density,
        "weights": {"w_env": p.w_env, "w_self": p.w_self, "w_prior": p.w_prior, "amide_kappa": p.amide_kappa},
        "prior": {
            "enabled": p.prior.is_some(),
            "config": input.options.prior,
            "approximation": "product of GlycoFlow per-torsion marginals (von Mises KDE over GlycoFlow samples); torsion couplings ignored",
        },
        "config": input.options.fit,
        "best": {
            "label": "best fit (density objective + GlycoFlow marginal prior)",
            "basin": o.best,
            "objective": best.terms.total,
            "terms": best.terms,
            "psi_N_deg": deg(best.pose.psi),
            "phi_N_deg": deg(best.pose.phi),
            "template": best.pose.template,
            "torsions_rad": best.pose.tau,
            "env_contacts_below_2.2A": crate::evaluation::contacts_below(p, &f.site, &best.x, 2.2),
        },
        "basins": basins,
        "support": o.support,
        "supported_residues": std::iter::once("r".to_string()).chain(o.support.iter().filter(|s| s.supported).map(|s| s.residue.clone())).collect::<Vec<_>>(),
        "unsupported_residues": o.support.iter().filter(|s| !s.supported).map(|s| s.residue.clone()).collect::<Vec<_>>(),
        "free_torsions": o.free_torsions.iter().filter(|f| **f).count(),
        "alternatives": alternatives,
        "completions": completions,
        "costs": {
            "network_evaluations": o.counters.network_evaluations,
            "prior_network_evaluations": f.prior_network_evaluations,
            "objective_evaluations": o.counters.objective_evaluations,
            "objective_gradient_evaluations": o.counters.objective_gradient_evaluations,
            "stages_seconds": o.counters.stages,
            "fit_seconds": o.wall_seconds,
            "preparation_seconds": f.preparation_seconds,
            "prior_seconds": f.prior_seconds,
            "device": input.model.device,
        },
        "evaluation": f.recovery.as_ref().map(|r| json!({
            "note": "in-place RMSD to the deposited glycan (evaluation only; never used by the fit)",
            "recovery": r,
        })),
    })
}

/// Run the workflow on every requested site.
pub fn run(input: &WorkflowInput) -> Result<WorkflowResult> {
    let started = Instant::now();
    if input.sites.is_empty() {
        return Err(invalid("no site to fit"));
    }
    // protein without any target glycan (other glycans and ligands stay)
    let mut protein = input.structure.clone();
    let mut remove = std::collections::BTreeSet::new();
    for request in &input.sites {
        if let Some(d) = deposited_glycan(input.structure, &request.residue)? {
            remove.extend(d.residue_ids());
        }
    }
    protein.remove_residues(&remove);
    let mut fits = Vec::new();
    for request in &input.sites {
        fits.push(fit_one(input, request, &protein)?);
    }
    let best_glycans: Vec<PlacedGlycan> = fits
        .iter()
        .map(|f| PlacedGlycan {
            problem: &f.problem,
            site: &f.site,
            naming: &f.naming,
            x: &f.outcome.basins[f.outcome.best].x,
        })
        .collect();
    let fitted = fitted_structure(&protein, &best_glycans)?;
    // candidate models: per site, best, alternatives and prior completions (others at their best)
    let mut models: Vec<(String, Vec<PlacedGlycan>)> = Vec::new();
    fn with<'a>(fits: &'a [SiteFit], k: usize, x: &'a [[f64; 3]]) -> Vec<PlacedGlycan<'a>> {
        fits.iter()
            .enumerate()
            .map(|(j, f)| PlacedGlycan {
                problem: &f.problem,
                site: &f.site,
                naming: &f.naming,
                x: if j == k {
                    x
                } else {
                    &f.outcome.basins[f.outcome.best].x
                },
            })
            .collect()
    }
    for (k, f) in fits.iter().enumerate() {
        let o = &f.outcome;
        let best = &o.basins[o.best];
        let site = f.site.label();
        models.push((
            format!("{site}: best fit (density objective + GlycoFlow marginal prior)"),
            with(&fits, k, &best.x),
        ));
        for &i in o.alternatives.iter().filter(|&&i| i != o.best).take(5) {
            let b = &o.basins[i];
            models.push((
                format!(
                    "{site}: density-ambiguous alternative, objective +{:.1}",
                    b.terms.total - best.terms.total
                ),
                with(&fits, k, &b.x),
            ));
        }
        let mut order: Vec<usize> = (0..o.completions.len()).collect();
        order.sort_by(|&a, &b| {
            o.completions[a]
                .terms
                .total
                .total_cmp(&o.completions[b].terms.total)
        });
        for &j in order.iter().take(5) {
            models.push((format!("{site}: prior-driven completion of unsupported residues (not fitted to density)"), with(&fits, k, &o.completions[j].x)));
        }
    }
    let candidates = candidates_pdb(&protein, &models)?;
    // validation of the fitted sites (structure checks + ReGlyco density score)
    let sigma = fits[0].sigma.selected;
    let site_ids: Vec<ResidueId> = fits.iter().map(|f| f.site.residue.clone()).collect();
    let validation_options = ValidationOptions {
        focus_sites: site_ids.clone(),
        ..ValidationOptions::default()
    };
    let score_options = DensityScoreOptions {
        sigma_angstrom: Some(sigma),
        periodic: input.map.is_full_unit_cell(),
        ..DensityScoreOptions::default()
    };
    let targets: Vec<DensityTarget> = site_ids
        .iter()
        .map(|s| DensityTarget {
            site: s.clone(),
            glycan_residues: Vec::new(),
        })
        .collect();
    let (validation, validation_error) = match DensityScorer::new(input.map.clone(), score_options)
        .and_then(|scorer| validate_with_density(&fitted, &validation_options, &scorer, &targets))
    {
        Ok(v) => (Some(v), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let report = json!({
        "method": "GlycoFlow observation-guided generation + Adam refinement (Rust port of glycoflow/fitting/pipeline.py)",
        "model": input.model.dir,
        "device": input.model.device,
        "map": input.map.metadata().path,
        "map_sha256": input.map.metadata().sha256,
        "sites": fits.iter().map(|f| site_report(f, input)).collect::<Vec<_>>(),
        "validation": validation.as_ref().map(|v| json!({"valid": v.valid, "errors": v.errors.len(), "warnings": v.warnings.len()})),
        "validation_error": validation_error,
        "wall_seconds": started.elapsed().as_secs_f64(),
        "peak_rss_mb": peak_rss_mb(),
    });
    Ok(WorkflowResult {
        fits,
        protein,
        fitted,
        candidates_pdb: candidates,
        report,
        validation,
        validation_error,
    })
}

/// Write `fitted.pdb`, `candidates.pdb`, `glycoflow-fit.json` and `validation.json`.
pub fn write_outputs(result: &WorkflowResult, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|source| crate::error::FitError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    write_file(&dir.join("fitted.pdb"), result.fitted.to_pdb_string())?;
    write_file(&dir.join("candidates.pdb"), &result.candidates_pdb)?;
    write_file(
        &dir.join("glycoflow-fit.json"),
        serde_json::to_string_pretty(&result.report)? + "\n",
    )?;
    let validation = match &result.validation {
        Some(v) => serde_json::to_value(v)?,
        None => json!({"error": result.validation_error}),
    };
    write_file(
        &dir.join("validation.json"),
        serde_json::to_string_pretty(&validation)? + "\n",
    )?;
    Ok(())
}
