//! Batch refit of prepared PDB entries, deposit first: every N-glycosylation site keeps the glycan
//! its depositor built, with the anomers corrected, refitted from the deposited coordinates; the
//! map then says which of its residues it supports, and whether it holds more of the glycan.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example pdbwide2 -- \
//!   <model dir> <manifest.jsonl> <output dir> <cpu|cuda> [seed] [members] [shard, e.g. 0/4]
//!
//! The manifest has one JSON line per entry (GlycoFlow `scripts/pdbwide/prepare.py`):
//! `{"entry", "method", "sites": [{"name", "site": "CHAIN:NUM", "pdb", "map", "host_class", "status"}]}`.
//!
//! One site:
//! 1. the deposited tree with the anomers of the known N-glycans (`infer::corrected_anomers`);
//! 2. a fit of it that also starts at the deposited atoms (`known`), so the model is never worse
//!    by the objective than the deposit with ideal geometry;
//! 3. growth: while the map supports the outermost residues, the next residues of the glycans of
//!    the expression system are fitted on top of the model, and those that pass the support test
//!    and the density gate are added.
//!
//! Deposited residues stay in the model whatever the map says, each with its verdict (supported:
//! it passes the support test and the density gate; weak: one of the two; unsupported: neither);
//! added residues are in the model only when the map supports them.
//!
//! Output: `results.jsonl` (one line per site), and per site `<entry>/<name>.json` (the summary,
//! the `site_report` of the final fit, every stage) and `<entry>/<name>.pdb` (the glycan alone).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::time::Instant;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, Structure, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::cartesian::{Restraints, cartesian_refine};
use reglyco_glycoflow::ensemble::EnsembleOptions;
use reglyco_glycoflow::infer::{
    DENSITY_GATE, DensityLevels, Gate, N_GLYCAN_SUGGESTIONS, contains_tree, corrected_anomers,
    density_gate, pruned_sequence, statistical_support, suggestions_for,
};
use reglyco_glycoflow::known::{
    CHAIR_TOLERANCE, GeometryError, KnownAtoms, hybrid, pose_from_atoms,
};
use reglyco_glycoflow::output::{PlacedGlycan, glycan_pdb};
use reglyco_glycoflow::problem::V3;
use reglyco_glycoflow::site::{deposited_glycan, sequence_paths};
use reglyco_glycoflow::workflow::{
    SiteFit, SiteRequest, extend_one, extension_report, fit_one, site_report,
};
use reglyco_glycoflow::{
    ComputeDevice, GlycoflowModel, PriorCache, WorkflowInput, WorkflowOptions,
};
use serde_json::{Value, json};

type Error = Box<dyn std::error::Error>;

/// Fits of one site at most (the deposit's, then growth).
const MAX_FITS: usize = 6;

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

fn parent(path: &str) -> Option<&str> {
    path.rsplit_once('/').map(|(p, _)| p)
}

/// What the map says about a fit: the support test, the density gate, and the density of every
/// residue (fraction of the way from the solvent to the protein level).
struct Assessed {
    statistical: Vec<String>,
    gate: Gate,
    levels: Option<DensityLevels>,
    density: BTreeMap<String, f64>,
}

impl Assessed {
    /// Residues the map supports on their own: they pass the support test and their density is
    /// at least [`DENSITY_GATE`] of the way from the solvent to the protein level.
    fn supported(&self) -> BTreeSet<String> {
        self.statistical
            .iter()
            .filter(|p| self.density.get(*p).is_none_or(|f| *f >= DENSITY_GATE))
            .cloned()
            .collect()
    }
}

fn residue_paths(fit: &SiteFit) -> BTreeSet<String> {
    let p = &fit.problem;
    (0..p.n_atoms)
        .filter(|&i| p.keep[i])
        .map(|i| p.glycan.res_paths[i].clone())
        .collect()
}

fn assess(fit: &SiteFit, map: &DensityMap, options: &WorkflowOptions) -> Assessed {
    let p = &fit.problem;
    let env: Vec<V3> = fit.site.environment.iter().map(|a| a.position).collect();
    let levels = DensityLevels::new(map, p, &env);
    let (statistical, root_gain) = statistical_support(
        p,
        &fit.outcome,
        options.fit.support_base,
        options.fit.support_per_torsion,
    );
    let x = &fit.outcome.basins[fit.outcome.best].x;
    let gate = density_gate(p, x, map, levels.as_ref(), &statistical, root_gain);
    let mut density = BTreeMap::new();
    if let Some(l) = &levels {
        for path in residue_paths(fit) {
            let pts: Vec<V3> = (0..p.n_atoms)
                .filter(|&i| p.keep[i] && p.glycan.res_paths[i] == path)
                .map(|i| x[i])
                .collect();
            if let Some(f) = l.fraction(map, &pts) {
                density.insert(path, f);
            }
        }
    }
    Assessed {
        statistical,
        gate,
        levels,
        density,
    }
}

/// One fit of the site: the glycan `sequence`, also started at the `known` atoms.
struct Fitter<'a> {
    structure: &'a Structure,
    text: Option<&'a str>,
    map: &'a DensityMap,
    model: &'a GlycoflowModel,
    protein: &'a Structure,
    residue: ResidueId,
    options: WorkflowOptions,
}

impl Fitter<'_> {
    fn input(&self, known: Option<KnownAtoms>) -> WorkflowInput<'_> {
        let mut options = self.options.clone();
        options.known = known;
        WorkflowInput {
            structure: self.structure,
            structure_text: self.text,
            map: self.map,
            sites: Vec::new(),
            model: self.model,
            options,
        }
    }

    fn fit(&self, sequence: &str, known: Option<KnownAtoms>) -> Result<(SiteFit, Assessed), Error> {
        let input = self.input(known);
        let request = SiteRequest {
            residue: self.residue.clone(),
            sequence: Some(sequence.to_string()),
        };
        let fit = fit_one(&input, &request, self.protein)?;
        let assessed = assess(&fit, self.map, &self.options);
        Ok((fit, assessed))
    }
}

fn stage_record(label: &str, sequence: &str, fit: &SiteFit, a: &Assessed, seconds: f64) -> Value {
    let best = &fit.outcome.basins[fit.outcome.best];
    json!({
        "stage": label, "sequence": sequence, "seconds": seconds,
        "objective": best.terms.total, "loglik": best.terms.loglik, "restraint": best.e_restraint,
        "from_known": best.from_known, "basins": fit.outcome.basins.len(),
        "statistical": a.statistical, "supported": a.supported(),
        "density": a.density, "root_gain": a.gate.root_gain,
    })
}

/// Refit one site; the summary line, and the files written under `dir`.
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
    // a chair set where the deposit has a boat is held between its neighbours: with the wider default
    // the ring settles half-way, twisted, so the rings are held to their chair more tightly here
    options.fit.restraints.sd_ring = 5.0;
    // experiments: restraint width, Cartesian steps and the margin of the known positions
    let number = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    if let Some(s) = number("PDBWIDE_SD") {
        options.fit.restraints.sd_bond *= s;
        options.fit.restraints.sd_angle *= s;
        options.fit.restraints.sd_volume *= s;
    }
    if let Some(s) = number("PDBWIDE_RING") {
        options.fit.restraints.sd_ring = s;
    }
    if let Some(s) = number("PDBWIDE_CSTEPS") {
        options.fit.cartesian_steps = s as usize;
    }
    if let Some(s) = number("PDBWIDE_CLR") {
        options.fit.cartesian_lr = s;
    }
    if let Some(s) = number("PDBWIDE_MARGIN") {
        options.fit.alt_margin = s;
    }
    if let Some(s) = number("PDBWIDE_GUARD") {
        options.fit.known_guard = s;
    }
    if let Some(s) = number("PDBWIDE_SHARE") {
        options.fit.known_share = s;
    }
    let deposited = deposited_glycan(&structure, &residue)?.ok_or("no deposited glycan")?;
    let deposited_sequence = deposited.glycam_sequence()?;
    let site_residue = structure
        .residues()
        .into_iter()
        .find(|r| r.id == residue)
        .map(|r| r.name)
        .ok_or("site residue not found")?;
    let host = site["host_class"].as_str().unwrap_or("");
    let n_linked = site_residue == "ASN";
    // 1. the deposited tree with corrected anomers
    let corrected = if n_linked {
        corrected_anomers(&deposited_sequence)?
    } else {
        reglyco_glycoflow::infer::CorrectedTree {
            sequence: deposited_sequence.clone(),
            changed: Vec::new(),
        }
    };
    let suggestions = suggestions_for(&site_residue, host);
    // the glycan of a list that is the tree (told by `true`), or the first one that holds it
    let holding = |list: &[(&str, &str)], sequence: &str| {
        let holds = |big: &str, small: &str| contains_tree(big, small).unwrap_or(false);
        list.iter()
            .find(|(_, s)| holds(s, sequence) && holds(sequence, s))
            .map(|(n, _)| (n.to_string(), true))
            .or_else(|| {
                list.iter()
                    .find(|(_, s)| holds(s, sequence))
                    .map(|(n, _)| (n.to_string(), false))
            })
    };
    let mut protein = structure.clone();
    protein.remove_residues(&deposited.residue_ids());
    let fitter = Fitter {
        structure: &structure,
        text: text.as_deref(),
        map: &map,
        model,
        protein: &protein,
        residue: residue.clone(),
        options,
    };
    // 2. the deposit's own glycan, also from its atoms; a residue with a corrected anomer only
    // takes its place from them, and so does one with a distorted ring; a cis N-acetyl group is
    // left out
    let deposit_tokens: BTreeMap<String, String> = sequence_paths(&corrected.sequence)?
        .into_iter()
        .map(|(path, (token, _))| (path, token))
        .collect();
    let deposit_atoms = KnownAtoms {
        atoms: deposited.atoms.clone(),
        rigid: BTreeSet::new(),
    };
    // C-mannose on Trp is 1C4 in crystals: no error
    let both_chairs =
        reglyco_glycoflow::anchor::anchor(&site_residue).is_some_and(|a| a.both_chairs);
    let geometry_errors = |atoms: &KnownAtoms, residues: &BTreeMap<String, String>| {
        let mut errors = atoms.geometry_errors(residues);
        errors.retain(|(path, error)| {
            !(both_chairs
                && path == "r"
                && matches!(error, GeometryError::Ring(t) if *t > 180.0 - CHAIR_TOLERANCE))
        });
        errors
    };
    let errors = geometry_errors(&deposit_atoms, &deposit_tokens);
    let known = KnownAtoms {
        atoms: deposited.atoms.clone(),
        rigid: corrected.changed.iter().map(|c| c.0.clone()).collect(),
    }
    .without(&errors);
    let t0 = Instant::now();
    let (mut fit, mut assessed) = fitter.fit(&corrected.sequence, Some(known))?;
    let mut sequence = corrected.sequence.clone();
    let deposit_paths = residue_paths(&fit);
    let mut stages = vec![stage_record(
        "deposit",
        &sequence,
        &fit,
        &assessed,
        t0.elapsed().as_secs_f64(),
    )];
    // the deposited coordinates on the same map and objective (hydrogens and missing atoms on
    // ideal residues moved onto them), and the density of every deposited residue
    let comparison = {
        let p = &fit.problem;
        let start = KnownAtoms {
            atoms: deposited.atoms.clone(),
            rigid: BTreeSet::new(),
        }
        .on(p);
        let best = &fit.outcome.basins[fit.outcome.best];
        let pose = pose_from_atoms(p, |i| start.x[i], &best.pose);
        let x0 = hybrid(p, &p.place(&pose), &start);
        let (lr, restraints) = (
            fitter.options.fit.cartesian_lr,
            &fitter.options.fit.restraints,
        );
        let d = cartesian_refine(p, &x0, pose.template, 0, lr, restraints);
        let f = cartesian_refine(p, &best.x, best.pose.template, 0, lr, restraints);
        if std::env::var("PDBWIDE_BASINS").is_ok() {
            let rmsd = |x: &[V3]| {
                let (mut s, mut n) = (0.0, 0usize);
                for (i, k) in start.x.iter().enumerate() {
                    if let Some(k) = k {
                        s += (0..3).map(|c| (x[i][c] - k[c]).powi(2)).sum::<f64>();
                        n += 1;
                    }
                }
                (s / n.max(1) as f64).sqrt()
            };
            eprintln!(
                "{name}: deposit loglik {:.2} objective {:.2} prior {:.2} att {:.2} env {:.4} self {:.4} restraint {:.1}",
                d.terms.loglik,
                d.terms.total,
                d.terms.e_prior,
                d.terms.e_att,
                d.terms.e_env,
                d.terms.e_self,
                d.e_restraint
            );
            // what the restraints still hold against in the lowest basin from the deposit
            if let Some(&k) = fit
                .outcome
                .order
                .iter()
                .find(|&&i| fit.outcome.basins[i].from_known)
            {
                let b = &fit.outcome.basins[k];
                let r = Restraints::new(p, b.pose.template, restraints);
                let g = &p.glycan;
                let label = |i: usize| format!("{}:{}", g.res_paths[i], g.atom_names[i]);
                let dist = |x: &[V3], i: usize, j: usize| {
                    (0..3)
                        .map(|c| (x[i][c] - x[j][c]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                let mut terms: Vec<(f64, String)> = Vec::new();
                let (mut eb, mut ea, mut ev) = (0.0, 0.0, 0.0);
                for &(i, j, d0, w, bond) in &r.pairs {
                    let d = dist(&b.x, i, j);
                    let e = 0.5 * w * (d - d0).powi(2);
                    if bond {
                        eb += e
                    } else {
                        ea += e
                    }
                    terms.push((
                        e,
                        format!(
                            "{} {}-{} {:.2} (ideal {:.2}, deposit {:.2})",
                            if bond { "bond" } else { "1-3" },
                            label(i),
                            label(j),
                            d,
                            d0,
                            dist(&x0, i, j)
                        ),
                    ));
                }
                let vol = |x: &[V3], c: usize, n: [usize; 3]| {
                    let s = |a: V3, b: V3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                    let (u, v, w) = (s(x[n[0]], x[c]), s(x[n[1]], x[c]), s(x[n[2]], x[c]));
                    u[0] * (v[1] * w[2] - v[2] * w[1])
                        + u[1] * (v[2] * w[0] - v[0] * w[2])
                        + u[2] * (v[0] * w[1] - v[1] * w[0])
                };
                for &(c, n, v0) in &r.chiral {
                    let v = vol(&b.x, c, n);
                    let e = 0.5 * r.w_volume * (v - v0).powi(2);
                    ev += e;
                    terms.push((
                        e,
                        format!(
                            "chiral {} {:.2} (ideal {:.2}, deposit {:.2})",
                            label(c),
                            v,
                            v0,
                            vol(&x0, c, n)
                        ),
                    ));
                }
                terms.sort_by(|a, b| b.0.total_cmp(&a.0));
                eprintln!(
                    "  basin #{k}: bonds {eb:.1} 1-3 {ea:.1} chiral {ev:.1} of {:.1}",
                    b.e_restraint
                );
                for (e, t) in terms.iter().take(8) {
                    eprintln!("     {e:6.2} {t}");
                }
            }
            for (rank, &i) in fit.outcome.order.iter().enumerate() {
                let b = &fit.outcome.basins[i];
                if rank < 6 || b.from_known || i == fit.outcome.best {
                    eprintln!(
                        "  {rank:3} #{i:3} {} {} total {:9.2} loglik {:8.2} prior {:6.2} att {:5.2} env {:.4} self {:.4} restraint {:6.2} rmsd {:.2}",
                        if b.from_known { "known" } else { "     " },
                        if i == fit.outcome.best { "*" } else { " " },
                        b.terms.total + b.e_restraint,
                        b.terms.loglik,
                        b.terms.e_prior,
                        b.terms.e_att,
                        b.terms.e_env,
                        b.terms.e_self,
                        b.e_restraint,
                        rmsd(&b.x)
                    );
                }
            }
        }
        let mut density = BTreeMap::new();
        if let Some(l) = &assessed.levels {
            for path in &deposit_paths {
                let pts: Vec<V3> = deposited
                    .atoms
                    .iter()
                    .filter(|((r, _), _)| r == path)
                    .map(|(_, x)| *x)
                    .collect();
                if let Some(v) = l.fraction(&map, &pts) {
                    density.insert(path.clone(), v);
                }
            }
        }
        // bond and 1-3 distances between the deposited atoms, against the ideal ones
        let ideal = Restraints::new(p, pose.template, restraints);
        let geometry = |x: &[V3]| {
            let (mut sums, mut counts) = ([0.0; 2], [0usize; 2]);
            for &(i, j, d0, _, bond) in &ideal.pairs {
                if start.x[i].is_some() && start.x[j].is_some() {
                    let d = (0..3)
                        .map(|k| (x[i][k] - x[j][k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        - d0;
                    sums[bond as usize] += d * d;
                    counts[bond as usize] += 1;
                }
            }
            [1, 0].map(|k| (sums[k] / counts[k].max(1) as f64).sqrt())
        };
        let side = |c: &reglyco_glycoflow::cartesian::CartesianFit| {
            let [bond_rmsd, one_three_rmsd] = geometry(&c.x);
            json!({
                "loglik": c.terms.loglik, "objective": c.terms.total,
                "bond_rmsd": bond_rmsd, "one_three_rmsd": one_three_rmsd,
                "clash": c.terms.e_env + c.terms.e_self,
            })
        };
        json!({
            "deposit": side(&d), "fit": side(&f), "density_deposit": density,
            "density_fit": assessed.density,
            "from_deposit": best.from_known,
            "psi_n_deposit": pose.psi.to_degrees(),
            "matched_atoms": start.x.iter().filter(|x| x.is_some()).count(),
            "rmsd": fit.recovery.as_ref().map(|r| r.full_rmsd),
            "core_rmsd": fit.recovery.as_ref().map(|r| r.core_rmsd),
            "per_residue_rmsd": fit.recovery.as_ref().map(|r| &r.per_residue),
        })
    };
    // 3. growth along the glycans of the expression system, one layer of residues at a time
    let mut tree = deposit_paths.clone();
    let mut tried: BTreeSet<(String, String)> = BTreeSet::new();
    let mut added: Vec<String> = Vec::new();
    let mut fits = 1;
    while fits < MAX_FITS {
        let built = assessed.supported();
        let mut next = None;
        for (glycan, s) in suggestions {
            if !contains_tree(s, &sequence)? {
                continue;
            }
            let frontier: Vec<(String, String)> = sequence_paths(s)?
                .into_iter()
                .map(|(path, (token, anomer))| (path, format!("{token}{anomer}")))
                .filter(|(path, _)| {
                    !tree.contains(path) && parent(path).is_some_and(|q| built.contains(q))
                })
                .filter(|key| !tried.contains(key))
                .collect();
            if !frontier.is_empty() {
                next = Some((*glycan, *s, frontier));
                break;
            }
        }
        let Some((glycan, s, frontier)) = next else {
            break;
        };
        let mut keep = tree.clone();
        keep.extend(frontier.iter().map(|(p, _)| p.clone()));
        let (grown, _) = pruned_sequence(s, &keep)?;
        let x = &fit.outcome.basins[fit.outcome.best].x;
        let known = KnownAtoms::of_fit(&fit.problem, x, None);
        let t = Instant::now();
        let (g_fit, g_assessed) = fitter.fit(&grown, Some(known))?;
        fits += 1;
        // an added residue needs its own support and that of the residue it is linked to
        let g_supported = g_assessed.supported();
        let accepted: Vec<String> = frontier
            .iter()
            .map(|(p, _)| p.clone())
            .filter(|p| {
                g_supported.contains(p) && parent(p).is_some_and(|q| g_supported.contains(q))
            })
            .collect();
        let mut record = stage_record(
            &format!("grow ({glycan})"),
            &grown,
            &g_fit,
            &g_assessed,
            t.elapsed().as_secs_f64(),
        );
        record["tried"] = json!(frontier);
        record["accepted"] = json!(accepted);
        stages.push(record);
        tried.extend(frontier.iter().cloned());
        if accepted.is_empty() {
            continue;
        }
        tree.extend(accepted.iter().cloned());
        added.extend(accepted.iter().cloned());
        if accepted.len() == frontier.len() {
            (fit, assessed, sequence) = (g_fit, g_assessed, grown);
            continue;
        }
        // some of the layer was refused: the model is the fit of the accepted tree alone
        let (kept, _) = pruned_sequence(s, &tree)?;
        let gx = &g_fit.outcome.basins[g_fit.outcome.best].x;
        let known = KnownAtoms::of_fit(&g_fit.problem, gx, Some(&tree));
        let t = Instant::now();
        let (k_fit, k_assessed) = fitter.fit(&kept, Some(known))?;
        fits += 1;
        stages.push(stage_record(
            "refit of the accepted residues",
            &kept,
            &k_fit,
            &k_assessed,
            t.elapsed().as_secs_f64(),
        ));
        (fit, assessed, sequence) = (k_fit, k_assessed, kept);
    }
    let fit_seconds = t0.elapsed().as_secs_f64();
    // the model: every deposited residue, and the added ones
    let best = &fit.outcome.basins[fit.outcome.best];
    let tokens: BTreeMap<String, String> = sequence_paths(&sequence)?
        .into_iter()
        .map(|(path, (token, anomer))| (path, format!("{token}{anomer}")))
        .collect();
    let changed: BTreeSet<&str> = corrected.changed.iter().map(|c| c.0.as_str()).collect();
    let per_residue = fit.recovery.as_ref().map(|r| &r.per_residue);
    let supported = assessed.supported();
    // errors of the deposited geometry, with the same measure on the model
    let model_atoms = KnownAtoms::of_fit(&fit.problem, &best.x, None);
    let mut deposit_errors: Vec<Value> = errors
        .iter()
        .map(|(path, error)| match error {
            GeometryError::Acetyl(t) => json!({
                "path": path, "kind": "acetyl", "deposit": t,
                "fit": model_atoms.acetyl_torsion(path),
            }),
            GeometryError::Ring(t) => json!({
                "path": path, "kind": "ring", "deposit": t,
                "fit": model_atoms.chair_deviation(path, &deposit_tokens[path]),
            }),
        })
        .collect();
    // the same errors in the model itself: none is the aim
    let mut fit_errors: Vec<Value> = geometry_errors(&model_atoms, &tokens)
        .iter()
        .map(|(path, error)| match error {
            GeometryError::Acetyl(t) => json!({"path": path, "kind": "acetyl", "fit": t}),
            GeometryError::Ring(t) => json!({"path": path, "kind": "ring", "fit": t}),
        })
        .collect();
    // a cis amide between the site and the glycan (CB-CG-ND2-C1 is 180 deg in N-glycans)
    if let Some(psi) = comparison["psi_n_deposit"]
        .as_f64()
        .filter(|p| n_linked && p.abs() < 90.0)
    {
        deposit_errors.push(json!({
            "path": "r", "kind": "link", "deposit": psi, "fit": best.pose.psi.to_degrees(),
        }));
    }
    // which the fit cannot mend when the side chain of the asparagine itself is misplaced
    let psi = best.pose.psi.to_degrees();
    if n_linked && psi.abs() < 90.0 {
        fit_errors.push(json!({"path": "r", "kind": "link", "fit": psi}));
    }
    let residues: Vec<Value> = residue_paths(&fit)
        .into_iter()
        .map(|path| {
            let origin = if !deposit_paths.contains(&path) {
                "added"
            } else if changed.contains(path.as_str()) {
                "corrected"
            } else {
                "deposited"
            };
            // supported: the support test and the density gate; weak: one of the two (density
            // the map cannot tell from noise at its resolution, or a gain without density of
            // its own); unsupported: neither
            let statistical = assessed.statistical.contains(&path);
            let dense = assessed
                .density
                .get(&path)
                .is_some_and(|f| *f >= DENSITY_GATE);
            let verdict = if supported.contains(&path) {
                "supported"
            } else if statistical || dense {
                "weak"
            } else {
                "unsupported"
            };
            json!({
                "path": path, "residue": tokens.get(&path), "origin": origin, "verdict": verdict,
                "statistical": statistical,
                "density": assessed.density.get(&path),
                "density_deposit": comparison["density_deposit"].get(&path),
                "rmsd": per_residue.and_then(|r| r.get(&path)),
            })
        })
        .collect();
    let count = |origin: &[&str], verdict: &str| {
        residues
            .iter()
            .filter(|r| {
                origin.contains(&r["origin"].as_str().unwrap_or("")) && r["verdict"] == verdict
            })
            .count()
    };
    let input = fitter.input(None);
    let extension = if members > 0 {
        Some(extend_one(
            &input,
            &fit,
            &EnsembleOptions {
                members,
                seed,
                ..EnsembleOptions::default()
            },
        )?)
    } else {
        None
    };
    let model_pdb = glycan_pdb(&[PlacedGlycan {
        problem: &fit.problem,
        site: &fit.site,
        naming: &fit.naming,
        x: &best.x,
    }])?;
    let mut models = format!(
        "REMARK 250 MODEL 1: fit; built: {}\nMODEL        1\n",
        supported.iter().cloned().collect::<Vec<_>>().join(" ")
    );
    models.extend(
        model_pdb
            .lines()
            .filter(|l| l.starts_with("HETATM"))
            .map(|l| format!("{l}\n")),
    );
    models.push_str("ENDMDL\nEND\n");
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{name}.pdb")), models)?;
    // the glycan of this expression system that holds the model, or the system whose glycans do
    let named = holding(suggestions, &sequence);
    let elsewhere = if named.is_some() || !n_linked {
        None
    } else {
        N_GLYCAN_SUGGESTIONS
            .iter()
            .filter(|(h, _)| !h.is_empty() && *h != host)
            .find_map(|(h, list)| {
                holding(list, &sequence).map(|(n, whole)| (h.to_string(), n, whole))
            })
    };
    let whole_glycan = named
        .as_ref()
        .map(|n| n.1)
        .or(elsewhere.as_ref().map(|e| e.2));
    let typical = named.map(|n| n.0);
    let elsewhere = elsewhere.map(|(h, n, _)| (h, n));
    let glycan = typical.clone().unwrap_or_else(|| "deposited".to_string());
    let weak: Vec<&str> = residues
        .iter()
        .filter(|r| r["verdict"] == "weak")
        .filter_map(|r| r["path"].as_str())
        .collect();
    let summary = json!({
        "recipe": "v2",
        "deposited": deposited_sequence,
        "deposited_residues": deposit_paths.len(),
        "glycan": glycan, "sequence": sequence,
        "corrections": corrected.changed,
        "deposit_errors": deposit_errors,
        "fit_errors": fit_errors,
        "typical": typical, "elsewhere": elsewhere, "whole_glycan": whole_glycan,
        "added": added,
        "residues": residues,
        "deposited_supported": count(&["deposited", "corrected"], "supported"),
        "deposited_weak": count(&["deposited", "corrected"], "weak"),
        "deposited_unsupported": count(&["deposited", "corrected"], "unsupported"),
        "added_supported": count(&["added"], "supported"),
        "comparison": comparison,
        "built": supported, "weak": weak,
        "density_fraction": assessed.density,
        "root_gain": assessed.gate.root_gain,
        "clusters": extension.as_ref().and_then(|e| e.ensemble.as_ref()).map(|e| e.clusters.len()),
        "objective": best.terms.total, "loglik": best.terms.loglik,
        "core_rmsd": fit.recovery.as_ref().map(|r| r.core_rmsd),
        "full_rmsd": fit.recovery.as_ref().map(|r| r.full_rmsd),
        "resolution": fit.resolution, "sigma": fit.sigma.selected,
        "fits": fits, "fit_seconds": fit_seconds,
    });
    let extend = match &extension {
        Some(e) => extension_report(e),
        None => json!({
            "built": supported, "weak": weak,
            "density_fraction": assessed.density,
            "root_gain": assessed.gate.root_gain, "density_gate": DENSITY_GATE,
            "ensemble": null,
        }),
    };
    let report = json!({
        "name": name, "summary": summary, "stages": stages,
        "fit": site_report(&fit, &input), "extend": extend,
    });
    std::fs::write(dir.join(format!("{name}.json")), report.to_string())?;
    Ok(summary)
}

fn main() -> Result<(), Error> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        return Err("usage: pdbwide2 <model> <manifest.jsonl> <output dir> <cpu|cuda> [seed] [members] [shard i/n]".into());
    }
    let device = if a[4] == "cuda" {
        ComputeDevice::Cuda
    } else {
        ComputeDevice::Cpu
    };
    let seed: u64 = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let members: usize = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let (shard, shards): (usize, usize) = match a.get(7).and_then(|s| s.split_once('/')) {
        Some((i, n)) => (i.parse()?, n.parse()?),
        None => (0, 1),
    };
    let only: Option<HashSet<String>> = std::env::var("PDBWIDE_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
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
            if only.as_ref().is_some_and(|o| !o.contains(name)) {
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
