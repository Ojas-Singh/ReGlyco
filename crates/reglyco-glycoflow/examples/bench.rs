//! The fitting benchmark in one process: every site of a list, the model loaded once (sweep.rs per
//! site pays the model load and GPU initialisation every time).
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example bench -- \
//!   <model dir> <sites.tsv: name set pdb map CHAIN:NUM> <cpu|cuda> <seeds> <samples:steps>
//!
//! Same output and switches (RINGS, MAXENT, PRE_REFINE, RENOISE, GUIDANCE) as sweep.rs, one JSON line per site.
//! `RINGS=generated|library` forces the pucker templates' ring source (default: generated when the
//! model has a ring head, `RingSource::Auto`); `MAXENT=1` also weights the retained basins into a max-ent ensemble.

use std::time::Instant;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::workflow::run;
use reglyco_glycoflow::{
    ComputeDevice, GlycoflowModel, SiteRequest, WorkflowInput, WorkflowOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        return Err("usage: bench <model> <sites.tsv> <cpu|cuda> <seeds> <samples:steps>".into());
    }
    let device = if a[3] == "cuda" {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let model = GlycoflowModel::load(std::path::Path::new(&a[1]), device, Precision::F32)?;
    let seeds: Vec<u64> = a[4]
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    for line in std::fs::read_to_string(&a[2])?.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        if let Err(e) = run_site(&model, f[0], f[1], f[2], f[3], f[4], &seeds, &a[5]) {
            println!(
                "{}",
                serde_json::json!({"name": f[0], "set": f[1], "error": e.to_string()})
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_site(
    model: &GlycoflowModel,
    name: &str,
    set: &str,
    pdb: &str,
    map: &str,
    site: &str,
    seeds: &[u64],
    configs: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let structure = read_pdb(pdb, &options)?;
    let text = std::fs::read_to_string(pdb).ok();
    let map = DensityMap::open(map)?;
    let (chain, number) = site.split_once(':').ok_or("site must be CHAIN:NUM")?;
    let residue = ResidueId {
        chain: chain.into(),
        number: number.parse()?,
        insertion_code: None,
    };
    for config in configs.split(',') {
        let (samples, steps) = config
            .split_once(':')
            .ok_or("config must be samples:steps")?;
        for &seed in seeds {
            let mut opts = WorkflowOptions::default();
            opts.fit.seed = seed;
            opts.fit.n_samples = samples.parse()?;
            opts.fit.flow_steps = steps.parse()?;
            opts.problem.ring_source = match std::env::var("RINGS").as_deref() {
                Ok("generated") => reglyco_glycoflow::RingSource::Generated,
                Ok("library") => reglyco_glycoflow::RingSource::Library,
                _ => reglyco_glycoflow::RingSource::Auto,
            };
            opts.maxent = std::env::var("MAXENT").as_deref() == Ok("1");
            // evaluation: SEED_DEPOSIT=1 also starts a basin at the deposit (the reachable best)
            opts.seed_deposit = std::env::var("SEED_DEPOSIT").as_deref() == Ok("1");
            // BATCH=<n>: conformers per network batch (the browser uses 32)
            if let Ok(v) = std::env::var("BATCH") {
                opts.problem.batch = Some(v.parse()?);
            }
            // search variants: PRE_REFINE=<multiple of n_basins>,
            // RENOISE=<basins>:<per basin>:<t>:<rounds>, GUIDANCE=<scale>:<start>
            if let Ok(v) = std::env::var("PRE_REFINE") {
                opts.fit.pre_refine = v.parse()?;
            }
            if let Ok(v) = std::env::var("RENOISE") {
                let p: Vec<&str> = v.split(':').collect();
                opts.fit.renoise_basins = p[0].parse()?;
                opts.fit.renoise_per_basin = p.get(1).map_or(Ok(8), |x| x.parse())?;
                opts.fit.renoise_t = p.get(2).map_or(Ok(0.5), |x| x.parse())?;
                opts.fit.renoise_rounds = p.get(3).map_or(Ok(3), |x| x.parse())?;
            }
            // CORE_FIRST=<fraction of the samples>:<cores>:<per core>, BRANCH=<basins>:<per branch>
            if let Ok(v) = std::env::var("CORE_FIRST") {
                let p: Vec<&str> = v.split(':').collect();
                opts.fit.core_first_fraction = p[0].parse()?;
                opts.fit.core_first_cores = p.get(1).map_or(Ok(8), |x| x.parse())?;
                opts.fit.core_first_per_core = p.get(2).map_or(Ok(4), |x| x.parse())?;
            }
            if let Ok(v) = std::env::var("BRANCH") {
                let p: Vec<&str> = v.split(':').collect();
                opts.fit.branch_basins = p[0].parse()?;
                opts.fit.branch_per_branch = p.get(1).map_or(Ok(2), |x| x.parse())?;
            }
            if let Ok(v) = std::env::var("GUIDANCE") {
                let p: Vec<&str> = v.split(':').collect();
                opts.fit.guidance_scale = p[0].parse()?;
                if let Some(x) = p.get(1) {
                    opts.fit.guidance_start = x.parse()?;
                }
            }
            let input = WorkflowInput {
                structure: &structure,
                structure_text: text.as_deref(),
                map: &map,
                sites: vec![SiteRequest {
                    residue: residue.clone(),
                    sequence: None,
                }],
                model: &model,
                options: opts,
            };
            let opts_echo = (
                input.options.fit.pre_refine,
                input.options.fit.renoise_basins,
            );
            let t0 = Instant::now();
            let result = run(&input)?;
            let wall = t0.elapsed().as_secs_f64();
            let fit = &result.fits[0];
            let best = &fit.outcome.basins[fit.outcome.best];
            let rec = fit.recovery.as_ref();
            let dep =
                fit.site.deposited.as_ref().map(|d| {
                    reglyco_glycoflow::evaluation::deposited_score(&fit.problem, d, &best.x)
                });
            println!(
                "{}",
                serde_json::json!({
                    "name": name, "set": set, "site": site, "samples": samples, "steps": steps, "seed": seed,
                    "wall_s": wall, "fit_s": fit.outcome.wall_seconds, "prep_s": fit.preparation_seconds,
                    "nfe": fit.outcome.counters.network_evaluations,
                    "objective": best.terms.total, "loglik": best.terms.loglik,
                    "full_rmsd": rec.map(|r| r.full_rmsd), "core_rmsd": rec.map(|r| r.core_rmsd),
                    "claimed_rmsd": rec.map(|r| r.claimed_supported_rmsd),
                    "claimed": rec.map(|r| r.claimed_supported.len()),
                    "valid": result.validation.as_ref().map(|v| v.valid),
                    "validation_errors": result.validation.as_ref().map(|v| v.errors.clone()),
                    "final_contact_weight": fit.outcome.final_contact_weight,
                    "e_env": best.terms.e_env, "e_self": best.terms.e_self, "e_restraint": best.e_restraint,
                    "prior_deviation": fit.outcome.prior_deviation,
                    "rings": std::env::var("RINGS").unwrap_or_else(|_| "auto".into()),
                    "pre_refine": opts_echo.0, "renoise": opts_echo.1,
                    "maxent": fit.maxent.as_ref().map(|m| serde_json::json!({
                        "ess": m.at_half.ess, "loglik": m.at_half.mixture.loglik, "best_pose": m.best_pose_loglik,
                        "flat": m.flat.loglik, "ratio": m.gain_ratio, "members": m.members.len(),
                        "top_weights": top_weights(&m.at_half.weights) })),
                    "deposited": dep.as_ref().map(|d| serde_json::json!({
                        "loglik": d.terms.loglik, "total": d.terms.total, "e_env": d.terms.e_env,
                        "e_self": d.terms.e_self, "e_att": d.terms.e_att, "e_prior": d.terms.e_prior,
                        "matched_atoms": d.matched_atoms, "scored_atoms": d.scored_atoms})),
                })
            );
        }
    }
    Ok(())
}
/// The five largest weights, descending.
fn top_weights(w: &[f64]) -> Vec<f64> {
    let mut w = w.to_vec();
    w.sort_by(|a, b| b.total_cmp(a));
    w.truncate(5);
    w
}
