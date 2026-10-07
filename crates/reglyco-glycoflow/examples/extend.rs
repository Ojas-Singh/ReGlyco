//! The extend mode at one site: fit a glycan larger than the deposit, build what the map supports
//! and return the ensemble beyond it (`reglyco_glycoflow::ensemble`).
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example extend -- \
//!   <model dir> <protein.pdb> <map> <CHAIN:NUM> <cpu|cuda> <seed> <glycan> [members] [output dir]
//!
//! <glycan>: an expression system ("mammalian", "insect", "yeast or fungus", "plant", "" for
//! unknown: the first suggestion of `infer::suggestions_for` that holds the deposited glycan), the
//! name of a suggestion, or a GLYCAM sequence. Prints one JSON line (what is built, the ensemble's
//! clusters, flexibility and reading per residue). With an output directory, also writes
//! `extend.pdb`: the glycan alone, the fit and then the medoid of every cluster as models.

use std::time::Instant;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::ensemble::EnsembleOptions;
use reglyco_glycoflow::infer::{N_GLYCAN_SUGGESTIONS, contains_tree, suggestions_for};
use reglyco_glycoflow::site::deposited_glycan;
use reglyco_glycoflow::workflow::{
    SiteRequest, extend_one, extension_models, extension_report, fit_one,
};
use reglyco_glycoflow::{ComputeDevice, GlycoflowModel, WorkflowInput, WorkflowOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 8 {
        return Err("usage: extend <model> <pdb> <map> <CHAIN:NUM> <cpu|cuda> <seed> <glycan> [members] [output dir]".into());
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
    // the glycan to extend to
    let holds = |seq: &str| {
        deposited_sequence
            .as_deref()
            .is_none_or(|d| contains_tree(seq, d).unwrap_or(false))
    };
    let is_host = N_GLYCAN_SUGGESTIONS.iter().any(|(host, _)| *host == a[7]);
    let (name, sequence) = if is_host {
        suggestions_for(&site_residue, &a[7])
            .iter()
            .find(|(_, s)| holds(s))
            .map(|(n, s)| (n.to_string(), s.to_string()))
            .ok_or("no suggestion holds the deposited glycan")?
    } else if let Some((n, s)) = N_GLYCAN_SUGGESTIONS
        .iter()
        .flat_map(|(_, o)| o.iter())
        .find(|(n, _)| *n == a[7])
    {
        (n.to_string(), s.to_string())
    } else {
        (a[7].clone(), a[7].clone())
    };
    if !holds(&sequence) {
        return Err(format!(
            "{name} does not hold the deposited glycan ({})",
            deposited_sequence.unwrap_or_default()
        )
        .into());
    }
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
    let t0 = Instant::now();
    let fit = fit_one(
        &input,
        &SiteRequest {
            residue: residue.clone(),
            sequence: Some(sequence.clone()),
        },
        &protein,
    )?;
    let fit_seconds = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let ensemble_options = EnsembleOptions {
        members: a.get(8).map(|m| m.parse()).transpose()?.unwrap_or(128),
        seed: input.options.fit.seed,
        ..EnsembleOptions::default()
    };
    let extension = extend_one(&input, &fit, &ensemble_options)?;
    let ensemble_seconds = t1.elapsed().as_secs_f64();
    let mut report = extension_report(&extension);
    report["site"] = a[4].clone().into();
    report["deposited"] = deposited_sequence.into();
    report["glycan"] = name.into();
    report["sequence"] = sequence.into();
    report["fit_seconds"] = fit_seconds.into();
    report["ensemble_seconds"] = ensemble_seconds.into();
    println!("{report}");
    if let Some(dir) = a.get(9) {
        std::fs::create_dir_all(dir)?;
        let mut out = String::new();
        for (k, (label, pdb)) in extension_models(&fit, &extension)?.iter().enumerate() {
            out.push_str(&format!("REMARK 250 MODEL {}: {label}\n", k + 1));
            out.push_str(&format!("MODEL     {:4}\n", k + 1));
            out.extend(
                pdb.lines()
                    .filter(|l| l.starts_with("HETATM"))
                    .map(|l| format!("{l}\n")),
            );
            out.push_str("ENDMDL\n");
        }
        out.push_str("END\n");
        std::fs::write(std::path::Path::new(dir).join("extend.pdb"), out)?;
    }
    Ok(())
}
