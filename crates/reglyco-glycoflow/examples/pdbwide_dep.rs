//! Does the fitting objective hold a deposited glycan? Restrained Cartesian refinement started from
//! the deposited coordinates, beside the usual fit of the same glycan.
//!
//!   pdbwide_dep <model dir> <manifest.jsonl> <out.jsonl> <cpu|cuda> <site names, comma separated>

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::cartesian::{RestraintOptions, cartesian_refine};
use reglyco_glycoflow::ensemble::superpose;
use reglyco_glycoflow::evaluation::{deposited_pose, deposited_score};
use reglyco_glycoflow::infer::DensityLevels;
use reglyco_glycoflow::problem::{Pose, V3};
use reglyco_glycoflow::site::deposited_glycan;
use reglyco_glycoflow::workflow::{SiteRequest, fit_one};
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
    let mut options = WorkflowOptions::default();
    options.prior.cache = Some(priors.clone());
    options.seed_deposit = true;
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
            residue,
            sequence: Some(sequence.clone()),
        },
        &protein,
    )?;
    let p = &fit.problem;
    let best = &fit.outcome.basins[fit.outcome.best];
    let dep = deposited_score(p, &deposited, &best.x);
    // the deposit as a start: the template of its own torsions, every residue moved onto its
    // deposited atoms, then the deposited heavy atoms themselves
    let fallback = Pose {
        tau: vec![std::f64::consts::PI; p.n_torsions],
        psi: p.link_torsion.start(),
        phi: 0.0,
        template: 0,
    };
    let pose = deposited_pose(p, &deposited, &fallback);
    let mut x0 = p.place(&pose);
    let at = |i: usize| {
        deposited
            .atoms
            .get(&(
                p.glycan.res_paths[i].clone(),
                p.glycan.atom_names[i].clone(),
            ))
            .copied()
    };
    let mut paths: Vec<String> = p.glycan.res_paths[..p.n_atoms].to_vec();
    paths.sort();
    paths.dedup();
    for path in &paths {
        let idx: Vec<usize> = (0..p.n_atoms)
            .filter(|&i| &p.glycan.res_paths[i] == path)
            .collect();
        let pairs: Vec<(usize, V3)> = idx.iter().filter_map(|&i| at(i).map(|d| (i, d))).collect();
        if pairs.len() < 4 {
            continue;
        }
        let from: Vec<V3> = pairs.iter().map(|(i, _)| x0[*i]).collect();
        let to: Vec<V3> = pairs.iter().map(|(_, d)| *d).collect();
        let (r, t) = superpose(&from, &to);
        for &i in &idx {
            if p.movable[i] {
                let v = x0[i];
                x0[i] = [0, 1, 2].map(|k| r[k][0] * v[0] + r[k][1] * v[1] + r[k][2] * v[2] + t[k]);
            }
        }
        for (i, d) in pairs {
            if p.movable[i] {
                x0[i] = d;
            }
        }
    }
    let env: Vec<V3> = fit.site.environment.iter().map(|a| a.position).collect();
    let levels = DensityLevels::new(&map, p, &env);
    let rmsd = |x: &[V3]| {
        let (mut s, mut n) = (0.0, 0usize);
        for i in 0..p.n_atoms {
            if let (true, Some(d)) = (p.keep[i], at(i)) {
                s += (0..3).map(|k| (x[i][k] - d[k]).powi(2)).sum::<f64>();
                n += 1;
            }
        }
        (s / n.max(1) as f64).sqrt()
    };
    let fractions = |x: &[V3]| {
        let mut out = BTreeMap::new();
        if let Some(l) = &levels {
            for path in &paths {
                let pts: Vec<V3> = (0..p.n_atoms)
                    .filter(|&i| p.keep[i] && &p.glycan.res_paths[i] == path)
                    .map(|i| x[i])
                    .collect();
                if let Some(f) = l.fraction(&map, &pts) {
                    out.insert(path.clone(), (f * 100.0).round() / 100.0);
                }
            }
        }
        out
    };
    let mut out = json!({
        "sequence": sequence,
        "deposit": {"terms": dep.terms, "fractions": fractions(&x0)},
        "fit": {"terms": best.terms, "e_restraint": best.e_restraint, "rmsd": rmsd(&best.x), "fractions": fractions(&best.x), "basin": fit.outcome.best, "basins": fit.outcome.basins.len()},
    });
    for (label, steps, sd) in [
        ("start", 0usize, 1.0),
        ("refined", 300, 1.0),
        ("refined_long", 1000, 1.0),
        ("loose", 300, 2.5),
    ] {
        let d = RestraintOptions::default();
        let o = RestraintOptions {
            sd_bond: d.sd_bond * sd,
            sd_angle: d.sd_angle * sd,
            sd_volume: d.sd_volume * sd,
            sd_ring: d.sd_ring * sd,
            sd_amide: d.sd_amide * sd,
        };
        let c = cartesian_refine(
            p,
            &x0,
            pose.template,
            steps,
            input.options.fit.cartesian_lr,
            &o,
        );
        out[label] = json!({"terms": c.terms, "e_restraint": c.e_restraint, "total": c.total, "bond_rmsd": c.bond_rmsd, "one_three_rmsd": c.one_three_rmsd,
            "max_shift": c.max_shift, "rmsd": rmsd(&c.x), "fractions": fractions(&c.x)});
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
    let names: Vec<&str> = a[5].split(',').collect();
    let model = GlycoflowModel::load(Path::new(&a[1]), device, Precision::F32)?;
    let priors = PriorCache::default();
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&a[3])?;
    for line in std::io::BufReader::new(std::fs::File::open(&a[2])?).lines() {
        let entry: Value = serde_json::from_str(&line?)?;
        for s in entry["sites"].as_array().into_iter().flatten() {
            if !names.contains(&s["name"].as_str().unwrap_or("")) {
                continue;
            }
            let mut row = json!({"name": s["name"], "resolution": entry["resolution"]});
            match site(&model, s, &priors) {
                Ok(v) => row["result"] = v,
                Err(e) => row["error"] = e.to_string().into(),
            }
            writeln!(log, "{row}")?;
            eprintln!("{}", s["name"]);
        }
    }
    Ok(())
}
