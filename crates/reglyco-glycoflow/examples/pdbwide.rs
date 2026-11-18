//! Batch fitting over prepared PDB entries: the model is loaded once, every N-glycosylation site of
//! a manifest is refitted to its map and extended past the deposit, and each result is written as it
//! finishes, so a stopped run resumes where it was.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example pdbwide -- \
//!   <model dir> <manifest.jsonl> <output dir> <cpu|cuda> [seed] [members] [shard, e.g. 0/4]
//!
//! The manifest has one JSON line per entry (GlycoFlow `scripts/pdbwide/prepare.py`):
//! `{"entry", "method", "sites": [{"name", "site": "CHAIN:NUM", "pdb", "map", "host_class", "status"}]}`.
//! A site is fitted with the first glycan suggested for its expression system that holds the
//! deposited one (`infer::suggestions_for`), or with the deposited glycan when none does.
//!
//! Output: `results.jsonl` (one line per site: what was fitted, what the map supports, timings, or
//! the error), and per site `<entry>/<name>.json` (the full `site_report` and `extension_report`) and
//! `<entry>/<name>.pdb` (the glycan alone: the fit, then the medoid of every ensemble cluster).

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::time::Instant;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::ensemble::EnsembleOptions;
use reglyco_glycoflow::infer::{contains_tree, suggestions_for};
use reglyco_glycoflow::site::deposited_glycan;
use reglyco_glycoflow::workflow::{
    SiteRequest, extend_one, extension_models, extension_report, fit_one, site_report,
};
use reglyco_glycoflow::{
    ComputeDevice, GlycoflowModel, PriorCache, WorkflowInput, WorkflowOptions,
};
use serde_json::{Value, json};

type Error = Box<dyn std::error::Error>;

/// `CHAIN:NUM`, with an insertion code after the number when the residue has one.
fn residue_id(site: &str) -> Result<ResidueId, Error> {
    let (chain, rest) = site.split_once(':').ok_or("site must be CHAIN:NUM")?;
    let digits = rest
        .rfind(|c: char| c.is_ascii_digit())
        .ok_or("site has no residue number")?;
    let (number, insertion) = rest.split_at(digits + 1);
    Ok(ResidueId {
        chain: chain.into(),
        number: number.parse()?,
        insertion_code: insertion.chars().next(),
    })
}

/// Fit and extend one site; the summary line, and the files written under `dir`.
fn fit_site(
    model: &GlycoflowModel,
    site: &Value,
    dir: &Path,
    seed: u64,
    members: usize,
    priors: &PriorCache,
) -> Result<Value, Error> {
    let field = |k: &str| site[k].as_str().ok_or(format!("site has no {k}"));
    let (name, pdb) = (field("name")?, field("pdb")?);
    let build = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let structure = read_pdb(pdb, &build)?;
    let text = std::fs::read_to_string(pdb).ok();
    let map = DensityMap::open(field("map")?)?;
    let residue = residue_id(field("site")?)?;
    let mut options = WorkflowOptions::default();
    options.fit.seed = seed;
    options.prior.cache = Some(priors.clone());
    let deposited = deposited_glycan(&structure, &residue)?;
    let deposited_sequence = deposited
        .as_ref()
        .map(|d| d.glycam_sequence())
        .transpose()?;
    let site_residue = structure
        .residues()
        .into_iter()
        .find(|r| r.id == residue)
        .map(|r| r.name)
        .ok_or("site residue not found")?;
    let host = site["host_class"].as_str().unwrap_or("");
    let holds = |seq: &str| {
        deposited_sequence
            .as_deref()
            .is_none_or(|d| contains_tree(seq, d).unwrap_or(false))
    };
    // the glycan to extend to; the deposit itself when no suggestion holds it
    let suggestion = suggestions_for(&site_residue, host)
        .iter()
        .find(|(_, s)| holds(s))
        .map(|(n, s)| (n.to_string(), s.to_string()));
    let (glycan, sequence) = match (suggestion, &deposited_sequence) {
        (Some(s), _) => s,
        (None, Some(d)) => ("deposited".to_string(), d.clone()),
        (None, None) => return Err("no glycan to fit".into()),
    };
    let mut protein = structure.clone();
    if let Some(d) = &deposited {
        protein.remove_residues(&d.residue_ids());
    }
    let input = WorkflowInput {
        structure: &structure,
        structure_text: text.as_deref(),
        map: &map,
        sites: Vec::new(),
        model,
        options,
    };
    let t0 = Instant::now();
    let fit = fit_one(
        &input,
        &SiteRequest {
            residue,
            sequence: Some(sequence.clone()),
        },
        &protein,
    )?;
    let fit_seconds = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let extension = extend_one(
        &input,
        &fit,
        &EnsembleOptions {
            members,
            seed,
            ..EnsembleOptions::default()
        },
    )?;
    let ensemble_seconds = t1.elapsed().as_secs_f64();
    let mut models = String::new();
    for (k, (label, pdb)) in extension_models(&fit, &extension)?.iter().enumerate() {
        models.push_str(&format!("REMARK 250 MODEL {}: {label}\n", k + 1));
        models.push_str(&format!("MODEL     {:4}\n", k + 1));
        models.extend(
            pdb.lines()
                .filter(|l| l.starts_with("HETATM"))
                .map(|l| format!("{l}\n")),
        );
        models.push_str("ENDMDL\n");
    }
    models.push_str("END\n");
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{name}.pdb")), models)?;
    let best = &fit.outcome.basins[fit.outcome.best];
    let recovery = fit.recovery.as_ref();
    let summary = json!({
        "deposited": deposited_sequence,
        "deposited_residues": deposited.as_ref().map(|d| d.residue_ids().len()),
        "glycan": glycan, "sequence": sequence,
        "built": extension.gate.built, "weak": extension.gate.weak,
        "density_fraction": extension.gate.density_fraction,
        "root_gain": extension.gate.root_gain,
        "clusters": extension.ensemble.as_ref().map(|e| e.clusters.len()),
        "objective": best.terms.total, "loglik": best.terms.loglik,
        "core_rmsd": recovery.map(|r| r.core_rmsd), "full_rmsd": recovery.map(|r| r.full_rmsd),
        "resolution": fit.resolution, "sigma": fit.sigma.selected,
        "fit_seconds": fit_seconds, "ensemble_seconds": ensemble_seconds,
    });
    let report = json!({
        "name": name, "summary": summary,
        "fit": site_report(&fit, &input), "extend": extension_report(&extension),
    });
    std::fs::write(dir.join(format!("{name}.json")), report.to_string())?;
    Ok(summary)
}

fn main() -> Result<(), Error> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        return Err("usage: pdbwide <model> <manifest.jsonl> <output dir> <cpu|cuda> [seed] [members] [shard i/n]".into());
    }
    let device = if a[4] == "cuda" {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let seed: u64 = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let members: usize = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(128);
    let (shard, shards): (usize, usize) = match a.get(7).and_then(|s| s.split_once('/')) {
        Some((i, n)) => (i.parse()?, n.parse()?),
        None => (0, 1),
    };
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    // one results file per shard, so shards can run side by side
    let results = out.join(if shards > 1 {
        format!("results.{shard}.jsonl")
    } else {
        "results.jsonl".to_string()
    });
    let mut done = HashSet::new();
    if let Ok(f) = std::fs::File::open(&results) {
        for line in std::io::BufReader::new(f).lines() {
            if let Ok(v) = serde_json::from_str::<Value>(&line?) {
                done.extend(v["name"].as_str().map(str::to_string));
            }
        }
    }
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&results)?;
    let model = GlycoflowModel::load(Path::new(&a[1]), device, Precision::F32)?;
    // sites repeat a few glycans; the samples of the free glycan are drawn once for each
    let priors = PriorCache::default();
    let manifest = std::io::BufReader::new(std::fs::File::open(&a[2])?);
    let (mut fitted, mut failed) = (0usize, 0usize);
    let start = Instant::now();
    for (k, line) in manifest.lines().enumerate() {
        let entry: Value = serde_json::from_str(&line?)?;
        if k % shards != shard {
            continue;
        }
        let id = entry["entry"].as_str().unwrap_or("unknown");
        for site in entry["sites"].as_array().into_iter().flatten() {
            let Some(name) = site["name"].as_str() else {
                continue;
            };
            // sites the preparation could not make inputs for carry their reason in `status`
            if done.contains(name) || site["pdb"].is_null() || site["map"].is_null() {
                continue;
            }
            let t = Instant::now();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                fit_site(&model, site, &out.join(id), seed, members, &priors)
                    .map_err(|e| e.to_string())
            }))
            .unwrap_or_else(|_| Err("panic".to_string()));
            let mut row = json!({
                "name": name, "entry": id, "method": entry["method"], "site": site["site"],
                "host_class": site["host_class"], "seconds": t.elapsed().as_secs_f64(),
            });
            match outcome {
                Ok(summary) => {
                    fitted += 1;
                    row["status"] = "ok".into();
                    row["result"] = summary;
                }
                Err(e) => {
                    failed += 1;
                    row["status"] = "failed".into();
                    row["error"] = e.into();
                }
            }
            writeln!(log, "{row}")?;
            log.flush()?;
            eprintln!(
                "{name}: {} in {:.0} s ({fitted} fitted, {failed} failed, {:.0} s)",
                row["status"].as_str().unwrap_or(""),
                t.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
