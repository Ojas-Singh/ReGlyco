//! Glycan out of density with the sequence hidden: fit the default N-glycan candidates at one
//! site, prune to the density-supported tree, choose, and compare with the deposited glycan.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example infer -- \
//!   <model dir> <protein.pdb> <map> <CHAIN:NUM> <cpu|cuda> <seed> [samples:steps]
//!
//! Prints one JSON line.

use std::collections::BTreeSet;
use std::time::Instant;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::infer::{N_GLYCAN_CANDIDATES, composition, glycan_from_density, rmsd_over};
use reglyco_glycoflow::site::deposited_glycan;
use reglyco_glycoflow::{ComputeDevice, GlycoflowModel, WorkflowInput, WorkflowOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        return Err(
            "usage: infer <model> <pdb> <map> <CHAIN:NUM> <cpu|cuda> <seed> [samples:steps]".into(),
        );
    }
    let device = if a[5] == "cuda" {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let model = GlycoflowModel::load(std::path::Path::new(&a[1]), device, Precision::F32)?;
    let build = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let structure = read_pdb(&a[2], &build)?;
    let text = std::fs::read_to_string(&a[2]).ok();
    let map = DensityMap::open(&a[3])?;
    let (chain, number) = a[4].split_once(':').ok_or("site must be CHAIN:NUM")?;
    let residue = ResidueId {
        chain: chain.into(),
        number: number.parse()?,
        insertion_code: None,
    };
    let mut options = WorkflowOptions::default();
    options.fit.seed = a[6].parse()?;
    if let Some((samples, steps)) = a.get(7).and_then(|c| c.split_once(':')) {
        options.fit.n_samples = samples.parse()?;
        options.fit.flow_steps = steps.parse()?;
    }
    let deposited = deposited_glycan(&structure, &residue)?;
    let mut protein = structure.clone();
    if let Some(d) = &deposited {
        protein.remove_residues(&d.residue_ids());
    }
    let input = WorkflowInput {
        structure: &structure,
        structure_text: text.as_deref(),
        map: &map,
        sites: Vec::new(),
        model: &model,
        options,
    };
    let candidates: Vec<(String, String)> = N_GLYCAN_CANDIDATES
        .iter()
        .map(|(n, s)| (n.to_string(), s.to_string()))
        .collect();
    let t0 = Instant::now();
    let inf = glycan_from_density(&input, &residue, &protein, &candidates, true)?;
    let wall = t0.elapsed().as_secs_f64();
    let best = inf.chosen();
    let x = &best.fit.outcome.basins[best.fit.outcome.best].x;
    let keep: BTreeSet<String> = best.supported.iter().cloned().collect();
    let (comp, kept_rmsd) = match &deposited {
        Some(d) if !keep.is_empty() => (
            Some(composition(&best.tokens, d)),
            rmsd_over(&best.fit.problem, x, d, &keep),
        ),
        _ => (None, None),
    };
    let nfe: u64 = inf
        .candidates
        .iter()
        .map(|c| {
            c.fit.outcome.counters.network_evaluations + c.fit.prior_network_evaluations as u64
        })
        .sum();
    let cands: Vec<_> = inf
        .candidates
        .iter()
        .map(|c| {
            serde_json::json!({"name": c.name, "score": c.score, "root_gain": c.root_gain,
                "supported": c.supported, "pruned": c.pruned})
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "site": a[4], "seed": a[6], "wall_s": wall, "nfe": nfe, "sigma": inf.sigma,
            "region_radius": inf.region_radius, "glycosylated": inf.glycosylated(),
            "chosen": best.name, "pruned": best.pruned, "supported": best.supported,
            "deposited_sequence": deposited.as_ref().and_then(|d| d.glycam_sequence().ok()),
            "composition": comp, "kept_rmsd": kept_rmsd, "candidates": cands,
        })
    );
    Ok(())
}
