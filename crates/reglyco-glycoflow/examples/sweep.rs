//! Speed/accuracy sweep of the fitting configuration on one site.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example sweep -- \
//!   <model dir> <protein.pdb> <map> <CHAIN:NUM> <cpu|cuda> <seeds, e.g. 0,1,2> <samples:steps,...>
//!
//! Prints one JSON line per configuration and seed: fit seconds, network evaluations, objective,
//! and in-place recovery against the deposited glycan.
//!
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
    if a.len() < 8 {
        return Err(
            "usage: sweep <model> <pdb> <map> <CHAIN:NUM> <cpu|cuda> <seeds> <samples:steps,...>"
                .into(),
        );
    }
    let device = if a[5] == "cuda" {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let model = GlycoflowModel::load(std::path::Path::new(&a[1]), device, Precision::F32)?;
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let structure = read_pdb(&a[2], &options)?;
    let text = std::fs::read_to_string(&a[2]).ok();
    let map = DensityMap::open(&a[3])?;
    let (chain, number) = a[4].split_once(':').ok_or("site must be CHAIN:NUM")?;
    let residue = ResidueId {
        chain: chain.into(),
        number: number.parse()?,
        insertion_code: None,
    };
    let seeds: Vec<u64> = a[6]
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    for config in a[7].split(',') {
        let (samples, steps) = config
            .split_once(':')
            .ok_or("config must be samples:steps")?;
        for &seed in &seeds {
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
                    "site": a[4], "samples": samples, "steps": steps, "seed": seed,
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
