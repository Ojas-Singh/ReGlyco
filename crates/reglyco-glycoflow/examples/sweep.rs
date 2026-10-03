//! Speed/accuracy sweep of the fitting configuration on one site.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example sweep -- \
//!   <model dir> <protein.pdb> <map> <CHAIN:NUM> <cpu|cuda> <seeds, e.g. 0,1,2> <samples:steps,...>
//!
//! Prints one JSON line per configuration and seed: fit seconds, network evaluations, objective,
//! and in-place recovery against the deposited glycan.

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
                    "final_contact_weight": fit.outcome.final_contact_weight,
                    "e_env": best.terms.e_env, "e_self": best.terms.e_self, "e_restraint": best.e_restraint,
                    "prior_deviation": fit.outcome.prior_deviation,
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
