//! Glycan out of density with the sequence hidden: fit the default N-glycan candidates at one
//! site, prune to the density-supported tree, choose, and compare with the deposited glycan.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example infer -- \
//!   <model dir> <protein.pdb> <map> <CHAIN:NUM> <cpu|cuda> <seed> [samples:steps] [output dir]
//!
//! Prints one JSON line, with the map value (standard deviations of the map) at every kept
//! residue next to the protein and bulk-solvent levels around the site (is a residue the
//! deposit lacks sitting in real density?). With an output directory, also writes the inferred
//! model (`inferred.pdb`: protein + the chosen glycan, prior-placed residues included).

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
            "usage: infer <model> <pdb> <map> <CHAIN:NUM> <cpu|cuda> <seed> [samples:steps] [output dir]".into(),
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
    let evidence = map_evidence(&map, best, &keep);
    if let Some(dir) = a.get(8) {
        let dir = std::path::Path::new(dir);
        std::fs::create_dir_all(dir)?;
        let placed = reglyco_glycoflow::output::PlacedGlycan {
            problem: &best.fit.problem,
            site: &best.fit.site,
            naming: &best.fit.naming,
            x,
        };
        let model = reglyco_glycoflow::output::fitted_structure(&protein, &[placed])?;
        std::fs::write(dir.join("inferred.pdb"), model.to_pdb_string())?;
    }
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
                "supported": c.supported, "weak": c.weak, "pruned": c.pruned})
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
            "map_evidence": evidence, "weak": best.weak, "density_fraction": best.density_fraction,
            "density_levels": inf.levels, "density_gate": reglyco_glycoflow::infer::DENSITY_GATE,
        })
    );
    Ok(())
}

/// Map value (in standard deviations of the map) averaged over each kept residue's atoms, and the
/// protein (environment atoms within 12 A of the link atom) and bulk-solvent (decoy centres more
/// than 3.5 A from every environment atom) reference levels.
fn map_evidence(
    map: &DensityMap,
    best: &reglyco_glycoflow::infer::CandidateFit,
    keep: &BTreeSet<String>,
) -> serde_json::Value {
    let v = map.values();
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let sd = (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let periodic = map.is_full_unit_cell();
    let z = |p: [f64; 3]| map.value_at_cartesian(p, periodic).map(|x| (x - mean) / sd);
    let mean_of = |pts: &[[f64; 3]]| {
        let vals: Vec<f64> = pts.iter().filter_map(|p| z(*p)).collect();
        (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
    };
    let problem = &best.fit.problem;
    let x = &best.fit.outcome.basins[best.fit.outcome.best].x;
    let mut residues = serde_json::Map::new();
    for path in keep {
        let pts: Vec<[f64; 3]> = (0..problem.n_atoms)
            .filter(|&i| problem.keep[i] && &problem.glycan.res_paths[i] == path)
            .map(|i| x[i])
            .collect();
        residues.insert(path.clone(), serde_json::json!(mean_of(&pts)));
    }
    let link = problem.anchor[2];
    let env: Vec<[f64; 3]> = best
        .fit
        .site
        .environment
        .iter()
        .map(|a| a.position)
        .collect();
    let d = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
    let protein: Vec<[f64; 3]> = env.iter().copied().filter(|p| d(*p, link) < 12.0).collect();
    let solvent: Vec<[f64; 3]> = reglyco_glycoflow::problem::decoy_poses(2048)
        .into_iter()
        .map(|(dir, u, _)| [0, 1, 2].map(|k| link[k] + dir[k] * (problem.radius - 3.0) * u))
        .filter(|p| env.iter().all(|e| d(*p, *e) > 3.5))
        .collect();
    serde_json::json!({"residues": residues, "protein": mean_of(&protein), "solvent": mean_of(&solvent),
                       "solvent_points": solvent.len()})
}
