//! Diagnosis of the PDB-wide recipe at prepared sites: the deposited glycan refitted from its own
//! pose and from scratch, against the deposited coordinates on the same map and the same measures.
//!
//!   pdbwide_diag <model dir> <manifest.jsonl> <out.jsonl> <cpu|cuda> [shard i/n]

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::evaluation::deposited_score;
use reglyco_glycoflow::infer::{DensityLevels, density_gate, statistical_support};
use reglyco_glycoflow::site::deposited_glycan;
use reglyco_glycoflow::workflow::{SiteFit, SiteRequest, fit_one};
use reglyco_glycoflow::{
    ComputeDevice, GlycoflowModel, PriorCache, WorkflowInput, WorkflowOptions,
};
use serde_json::{Value, json};

type Error = Box<dyn std::error::Error>;

fn residue_id(site: &str) -> Result<ResidueId, Error> {
    let (chain, rest) = site.split_once(':').ok_or("site must be CHAIN:NUM")?;
    let digits = rest
        .rfind(|c: char| c.is_ascii_digit())
        .ok_or("no residue number")?;
    let (number, insertion) = rest.split_at(digits + 1);
    Ok(ResidueId {
        chain: chain.into(),
        number: number.parse()?,
        insertion_code: insertion.chars().next(),
    })
}

fn describe(fit: &SiteFit, map: &DensityMap, options: &WorkflowOptions) -> Value {
    let p = &fit.problem;
    let env: Vec<[f64; 3]> = fit.site.environment.iter().map(|a| a.position).collect();
    let levels = DensityLevels::new(map, p, &env);
    let (statistical, root_gain) = statistical_support(
        p,
        &fit.outcome,
        options.fit.support_base,
        options.fit.support_per_torsion,
    );
    let best = &fit.outcome.basins[fit.outcome.best];
    let gate = density_gate(p, &best.x, map, levels.as_ref(), &statistical, root_gain);
    // every residue's density at the fitted pose, and at the deposited coordinates
    let mut at_fit = BTreeMap::new();
    let mut at_deposit = BTreeMap::new();
    if let Some(l) = &levels {
        let mut paths: Vec<String> = p.glycan.res_paths.clone();
        paths.sort();
        paths.dedup();
        for path in paths {
            let pts: Vec<[f64; 3]> = (0..p.n_atoms)
                .filter(|&i| p.keep[i] && p.glycan.res_paths[i] == path)
                .map(|i| best.x[i])
                .collect();
            if let Some(f) = l.fraction(map, &pts) {
                at_fit.insert(path.clone(), f);
            }
            if let Some(d) = &fit.site.deposited {
                let pts: Vec<[f64; 3]> = d
                    .atoms
                    .iter()
                    .filter(|((r, _), _)| r == &path)
                    .map(|(_, x)| *x)
                    .collect();
                if let Some(f) = l.fraction(map, &pts) {
                    at_deposit.insert(path, f);
                }
            }
        }
    }
    let dep = fit
        .site
        .deposited
        .as_ref()
        .map(|d| deposited_score(p, d, &best.x));
    json!({
        "terms": best.terms, "deposit_terms": dep.as_ref().map(|d| &d.terms), "deposit_matched": dep.as_ref().map(|d| d.matched_atoms),
        "scored_atoms": dep.as_ref().map(|d| d.scored_atoms),
        "core_rmsd": fit.recovery.as_ref().map(|r| r.core_rmsd), "full_rmsd": fit.recovery.as_ref().map(|r| r.full_rmsd),
        "per_residue_rmsd": fit.recovery.as_ref().map(|r| &r.per_residue),
        "built": gate.built, "weak": gate.weak, "statistical": statistical, "root_gain": root_gain,
        "fraction_fit": at_fit, "fraction_deposit": at_deposit, "levels": levels,
        "sigma": fit.sigma.selected, "seconds": fit.outcome.wall_seconds, "best_basin": fit.outcome.best,
    })
}

fn site(model: &GlycoflowModel, site: &Value, priors: &PriorCache) -> Result<Value, Error> {
    let field = |k: &str| site[k].as_str().ok_or(format!("site has no {k}"));
    let pdb = field("pdb")?;
    let build = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let structure = read_pdb(pdb, &build)?;
    let text = std::fs::read_to_string(pdb).ok();
    let map = DensityMap::open(field("map")?)?;
    let residue = residue_id(field("site")?)?;
    let deposited = deposited_glycan(&structure, &residue)?.ok_or("no deposited glycan")?;
    let sequence = deposited.glycam_sequence()?;
    let mut protein = structure.clone();
    protein.remove_residues(&deposited.residue_ids());
    let mut out = json!({"sequence": sequence, "residues": deposited.residues.len()});
    for (label, seed_deposit) in [("seeded", true), ("scratch", false)] {
        let mut options = WorkflowOptions::default();
        options.prior.cache = Some(priors.clone());
        options.seed_deposit = seed_deposit;
        let input = WorkflowInput {
            structure: &structure,
            structure_text: text.as_deref(),
            map: &map,
            sites: Vec::new(),
            model,
            options,
        };
        let fit = fit_one(
            &input,
            &SiteRequest {
                residue: residue.clone(),
                sequence: Some(sequence.clone()),
            },
            &protein,
        )?;
        out[label] = describe(&fit, &map, &input.options);
    }
    Ok(out)
}

fn main() -> Result<(), Error> {
    let a: Vec<String> = std::env::args().collect();
    let device = if a[4] == "cuda" {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let (shard, shards): (usize, usize) = match a.get(5).and_then(|s| s.split_once('/')) {
        Some((i, n)) => (i.parse()?, n.parse()?),
        None => (0, 1),
    };
    let model = GlycoflowModel::load(Path::new(&a[1]), device, Precision::F32)?;
    let priors = PriorCache::default();
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&a[3])?;
    let mut k = 0usize;
    for line in std::io::BufReader::new(std::fs::File::open(&a[2])?).lines() {
        let entry: Value = serde_json::from_str(&line?)?;
        for s in entry["sites"].as_array().into_iter().flatten() {
            if s["pdb"].is_null() || s["map"].is_null() {
                continue;
            }
            k += 1;
            if k % shards != shard {
                continue;
            }
            let mut row = json!({"name": s["name"], "entry": entry["entry"], "method": entry["method"], "resolution": entry["resolution"]});
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                site(&model, s, &priors).map_err(|e| e.to_string())
            })) {
                Ok(Ok(v)) => row["result"] = v,
                Ok(Err(e)) => row["error"] = e.into(),
                Err(_) => row["error"] = "panic".into(),
            }
            writeln!(log, "{row}")?;
            eprintln!("{}", s["name"]);
        }
    }
    Ok(())
}
