//! Batch refit of prepared PDB entries, deposit first: every N-glycosylation site keeps the glycan
//! its depositor built, with the core linkages and the anomers corrected, refitted from the
//! deposited coordinates; the map then says which of its residues it supports, and whether it
//! holds more of the glycan.
//!
//! cargo run --release -p reglyco-glycoflow --features cuda --example pdbwide2 -- \
//!   <model dir> <manifest.jsonl> <output dir> <cpu|cuda> [seed] [members] [shard, e.g. 0/4]
//!
//! The manifest has one JSON line per entry (GlycoFlow `scripts/pdbwide/prepare.py`):
//! `{"entry", "method", "sites": [{"name", "site": "CHAIN:NUM", "pdb", "map", "host_class", "status"}]}`.
//!
//! One site:
//! 1. the deposited tree with the core linkages and the anomers of the known N-glycans
//!    (`infer::corrected_tree`);
//! 2. a fit of it that also starts at the deposited atoms (`known`), so the model is never worse
//!    by the objective than the deposit with ideal geometry; it leaves the deposited pose only for
//!    one that explains the map at least as well;
//! 3. a cis amide between the asparagine and the glycan that the fit could not turn: the side
//!    chain of the asparagine is rebuilt under the glycan (`sidechain`), and kept when the link
//!    is then trans, the glycan stays in its pose and the map holds the glycan and the side
//!    chain as well as before;
//! 4. the residues the map does not hold (no residue from them outwards is in density) are placed
//!    again by GlycoFlow on the residues in density, and that model is taken when its conformation
//!    is clearly the more usual one and the residues in density keep their density;
//! 5. growth: while the map supports the outermost residues, the next residues of the glycans of
//!    the expression system are fitted on top of the model, and those that pass the support test
//!    and the density gate are added, unless a residue the map supported loses its support to
//!    them, or a residue in density more than a fifth of its density.
//!
//! Deposited residues stay in the model whatever the map says, each with its verdict (supported:
//! it passes the support test and the density gate; weak: one of the two; unsupported: neither);
//! added residues are in the model only when the map supports them.
//!
//! Output: `results.jsonl` (one line per site), and per site `<entry>/<name>.json` (the summary,
//! the `site_report` of the final fit, every stage) and `<entry>/<name>.pdb` (the glycan alone).
//!
//! `PDBWIDE_FINAL=1` (same arguments) scores final models in place of fitting: a site with a row of
//! the first pass and a `<entry>/<name>.final.json` beside its files ([`Final`]: its glycan as a
//! later stage left it, such as the refinement against the diffraction data) gets a line in
//! `final.jsonl`, `<entry>/<name>.final.pdb` and `<entry>/<name>.final.report.json`, in the format
//! of the first pass: its decisions, and the numbers of the final coordinates. No search, no
//! refinement; the files of the first pass are left as they are.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::time::Instant;

use glycoflow_core::model::Precision;
use glysys::{BuildOptions, ResidueId, Structure, Vec3, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::cartesian::{
    DataWeight, Restraints, cartesian_refine, cartesian_refine_weighted,
};
use reglyco_glycoflow::ensemble::EnsembleOptions;
use reglyco_glycoflow::evaluation::recovery;
use reglyco_glycoflow::infer::{
    CoreFault, CorrectedTree, DENSITY_GATE, DensityLevels, Gate, N_GLYCAN_SUGGESTIONS,
    contains_tree, core_faults, corrected_tree, density_gate, pruned_sequence, statistical_support,
    suggestions_for,
};
use reglyco_glycoflow::known::{
    CHAIR_TOLERANCE, GeometryError, KnownAtoms, hybrid, pose_from_atoms,
};
use reglyco_glycoflow::output::{PlacedGlycan, ResidueNaming, glycan_pdb};
use reglyco_glycoflow::problem::{SiteProblem, V3, dihedral};
use reglyco_glycoflow::sidechain::{AsnSideChain, asn_side_chains};
use reglyco_glycoflow::site::{DepositedGlycan, atom_rename, deposited_glycan, sequence_paths};
use reglyco_glycoflow::workflow::{
    SiteFit, SiteRequest, extend_one, extension_report, fit_one, place_one, site_report,
};
use reglyco_glycoflow::pipeline::density_problem;
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

/// Share of its density a residue may lose to the growth of the glycan before the growth is refused.
const DENSITY_LOSS: f64 = 0.2;

/// Whether a residue that is in density in `before` has lost more than [`DENSITY_LOSS`] of it in `after`.
fn loses_density(before: &BTreeMap<String, f64>, after: &BTreeMap<String, f64>) -> bool {
    before.iter().any(|(path, &level)| {
        level >= DENSITY_GATE
            && after
                .get(path)
                .is_some_and(|&now| now < (1.0 - DENSITY_LOSS) * level)
    })
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

impl<'a> Fitter<'a> {
    /// The same fits with atoms of the site residue moved: `protein` has them there.
    fn moved<'b>(&self, protein: &'b Structure, atoms: &[(&str, V3)]) -> Fitter<'b>
    where
        'a: 'b,
    {
        let mut options = self.options.clone();
        options.site.moved = atoms.iter().map(|(n, x)| (n.to_string(), *x)).collect();
        Fitter {
            structure: self.structure,
            text: self.text,
            map: self.map,
            model: self.model,
            protein,
            residue: self.residue.clone(),
            options,
        }
    }

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

    /// The glycan `sequence` at the atoms of a final model: no search and no refinement.
    fn place(&self, sequence: &str, model: &Final) -> Result<(SiteFit, Assessed), Error> {
        let input = self.input(None);
        let request = SiteRequest {
            residue: self.residue.clone(),
            sequence: Some(sequence.to_string()),
        };
        let atoms = KnownAtoms {
            atoms: model.atoms.clone(),
            rigid: BTreeSet::new(),
        };
        let torsions = model.torsions.as_deref();
        let fit = place_one(&input, &request, self.protein, &atoms, torsions)?;
        let assessed = assess(&fit, self.map, &self.options);
        Ok((fit, assessed))
    }
}

/// Chain, number and insertion code of the residue of a HETATM record, and its atom name.
fn record_atom(line: &str) -> Option<((String, i32, Option<char>), &str)> {
    let insertion = line.get(26..27)?.chars().next().filter(|c| *c != ' ');
    let residue = (
        line.get(21..22)?.to_string(),
        line.get(22..26)?.trim().parse().ok()?,
        insertion,
    );
    Some((residue, line.get(12..16)?.trim()))
}

/// The final model of a site, made of its fit by a later stage (the refinement against the
/// diffraction data, which also chooses between the fit and the deposit): `<name>.final.json`
/// beside the files of the first pass,
/// `{"choice", "residues": [{"path", "ccd", "atoms": [[name, x, y, z, B], ...]}], "asn", "xray"}`,
/// with the heavy atoms of every residue of the fitted model under their PDB names and, in `asn`,
/// CG, OD1 and ND2 of the asparagine where they are not the deposited ones. With `PDBWIDE_FINAL`
/// the runner scores this model in place of fitting one.
struct Final {
    /// the stage's own record
    sidecar: Value,
    /// the result of the first pass, and its report
    first: Value,
    report: Value,
    /// the atoms by tree path and GLYCAM name, and the model of the first pass likewise
    atoms: BTreeMap<(String, String), V3>,
    engine: BTreeMap<(String, String), V3>,
    /// B-factors by tree path and PDB atom name
    b: BTreeMap<(String, String), f64>,
    /// residue names by tree path
    ccd: BTreeMap<String, String>,
    /// the side chain of the asparagine, when it is not the deposited one
    asn: Option<[(&'static str, V3); 3]>,
    /// the torsions no heavy atom determines, as the first pass had them
    torsions: Option<Vec<f64>>,
}

impl Final {
    fn read(dir: &Path, name: &str, first: &Value) -> Result<Self, Error> {
        let read = |file: String| -> Result<String, Error> {
            std::fs::read_to_string(dir.join(&file)).map_err(|e| format!("{file}: {e}").into())
        };
        let sidecar: Value = serde_json::from_str(&read(format!("{name}.final.json"))?)?;
        let report: Value = serde_json::from_str(&read(format!("{name}.json"))?)?;
        let glycam = |ccd: &str, pdb: &str| {
            let renamed = atom_rename(ccd).iter().find(|(p, _)| *p == pdb);
            renamed.map_or(pdb, |(_, g)| *g).to_string()
        };
        let point = |v: &Value| -> Option<V3> {
            Some([
                v.get(0)?.as_f64()?,
                v.get(1)?.as_f64()?,
                v.get(2)?.as_f64()?,
            ])
        };
        let (mut atoms, mut b, mut ccd) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
        let residues = sidecar["residues"].as_array();
        for r in residues.ok_or("final model without residues")? {
            let (Some(path), Some(name)) = (r["path"].as_str(), r["ccd"].as_str()) else {
                return Err("final model: residue without path or ccd".into());
            };
            ccd.insert(path.to_string(), name.to_string());
            for a in r["atoms"].as_array().into_iter().flatten() {
                let (Some(atom), Some(x), Some(y), Some(z)) =
                    (a[0].as_str(), a[1].as_f64(), a[2].as_f64(), a[3].as_f64())
                else {
                    return Err(format!("final model: atom of {path} unreadable").into());
                };
                atoms.insert((path.to_string(), glycam(name, atom)), [x, y, z]);
                if let Some(v) = a[4].as_f64() {
                    b.insert((path.to_string(), atom.to_string()), v);
                }
            }
        }
        // the model of the first pass, from its file and the residue names of its report
        let mut naming = BTreeMap::new();
        for n in report["fit"]["naming"].as_array().into_iter().flatten() {
            let insertion = n["insertion_code"].as_str().and_then(|s| s.chars().next());
            let (Some(chain), Some(number), Some(path), Some(name)) = (
                n["chain"].as_str(),
                n["number"].as_i64(),
                n["path"].as_str(),
                n["ccd"].as_str(),
            ) else {
                return Err("report of the first pass: residue names unreadable".into());
            };
            naming.insert((chain.to_string(), number as i32, insertion), (path, name));
        }
        let mut engine = BTreeMap::new();
        for line in read(format!("{name}.pdb"))?.lines() {
            if line.starts_with("ENDMDL") {
                break;
            }
            if !line.starts_with("HETATM") {
                continue;
            }
            let coordinate = |k: usize| line.get(k..k + 8)?.trim().parse::<f64>().ok();
            let (Some((residue, atom)), Some(x), Some(y), Some(z)) = (
                record_atom(line),
                coordinate(30),
                coordinate(38),
                coordinate(46),
            ) else {
                return Err("model of the first pass: record unreadable".into());
            };
            let (path, name) = naming
                .get(&residue)
                .ok_or("model of the first pass: residue not in its report")?;
            engine.insert((path.to_string(), glycam(name, atom)), [x, y, z]);
        }
        // the side chain of the asparagine: the final model's; without one, the one its glycan
        // was fitted on (the rebuilt one under the fit, the deposited one under the deposit)
        let rebuilt = &first["asn"]["atoms"];
        let side = match (&sidecar["asn"], sidecar["choice"].as_str()) {
            (Value::Null, Some("deposit")) => &Value::Null,
            (Value::Null, _) => rebuilt,
            (given, _) => given,
        };
        let asn = match side {
            Value::Null => None,
            s => Some(["CG", "OD1", "ND2"].map(|k| (k, point(&s[k])))),
        };
        let asn = match asn {
            Some(a) if a.iter().any(|(_, x)| x.is_none()) => {
                return Err("final model: asparagine without CG, OD1 and ND2".into());
            }
            Some(a) => Some(a.map(|(k, x)| (k, x.unwrap_or_default()))),
            None => None,
        };
        let torsions = report["fit"]["best"]["torsions_rad"]
            .as_array()
            .map(|t| t.iter().filter_map(Value::as_f64).collect());
        Ok(Self {
            sidecar,
            first: first.clone(),
            report,
            atoms,
            engine,
            b,
            ccd,
            asn,
            torsions,
        })
    }

    /// The strings of a list of the first pass's result.
    fn list(&self, key: &str) -> Vec<String> {
        let items = self.first[key].as_array().into_iter().flatten();
        items
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()
    }

    /// The model of the first pass on the atoms of `p`.
    fn engine_on(&self, p: &SiteProblem) -> Vec<Option<V3>> {
        KnownAtoms {
            atoms: self.engine.clone(),
            rigid: BTreeSet::new(),
        }
        .on(p)
        .x
    }

    /// The result of a rescored site from that of the final model (`summary`, made as a fit's):
    /// the decisions of the first pass, the numbers of the final coordinates. `asn` is the first
    /// pass's record with the final side chain, `scored` what only this pass knows.
    fn summary(&self, mut summary: Value, asn: Value, scored: Value) -> Value {
        let first = &self.first;
        summary["recipe"] = json!("v2.1");
        for k in [
            "added",
            "replaced",
            "built",
            "weak",
            "deposited_supported",
            "deposited_weak",
            "deposited_unsupported",
            "added_supported",
            "clusters",
            "fits",
            "fit_seconds",
        ] {
            summary[k] = first[k].clone();
        }
        summary["asn"] = asn;
        let same = |a: &Value, b: &Value, keys: &[&str]| keys.iter().all(|k| a[*k] == b[*k]);
        let of_first = |list: &str, item: &Value, keys: &[&str]| {
            let items = first[list].as_array().into_iter().flatten();
            items.into_iter().find(|f| same(f, item, keys)).cloned()
        };
        for r in summary["residues"].as_array_mut().into_iter().flatten() {
            if let Some(f) = of_first("residues", r, &["path"]) {
                for k in ["origin", "verdict", "statistical", "density_deposit"] {
                    r[k] = f[k].clone();
                }
            }
        }
        for t in summary["torsions"].as_array_mut().into_iter().flatten() {
            if let Some(f) = of_first("torsions", t, &["kind", "path", "atoms"]) {
                for k in ["deposit", "deposit_percentile", "samples"] {
                    t[k] = f[k].clone();
                }
            }
        }
        // the errors of the deposit are the first pass's, with the measure on the final model
        let mut errors = first["deposit_errors"].clone();
        for e in errors.as_array_mut().into_iter().flatten() {
            if matches!(e["kind"].as_str(), Some("acetyl" | "ring" | "link")) {
                let now = summary["deposit_errors"].as_array().into_iter().flatten();
                e["fit"] = now
                    .into_iter()
                    .find(|n| same(n, e, &["path", "kind"]))
                    .map_or(Value::Null, |n| n["fit"].clone());
            }
        }
        summary["deposit_errors"] = errors;
        summary["xray"] = self.sidecar["xray"].clone();
        summary["final"] = scored;
        summary
    }

    /// The records of the model with the B-factors of the final model (an atom without one: the
    /// mean of its residue).
    fn with_b(&self, records: &str, naming: &[ResidueNaming]) -> String {
        let path_of: BTreeMap<(String, i32, Option<char>), &str> = naming
            .iter()
            .map(|n| {
                (
                    (n.chain.clone(), n.number, n.insertion_code),
                    n.path.as_str(),
                )
            })
            .collect();
        let mean = |path: &str| {
            let b: Vec<f64> = self
                .b
                .iter()
                .filter(|(k, _)| k.0 == path)
                .map(|(_, b)| *b)
                .collect();
            (!b.is_empty()).then(|| b.iter().sum::<f64>() / b.len() as f64)
        };
        let mut out = String::new();
        for line in records.lines() {
            let b = record_atom(line)
                .filter(|_| line.starts_with("HETATM") && line.len() >= 66)
                .and_then(|(residue, atom)| {
                    let path = *path_of.get(&residue)?;
                    let own = self.b.get(&(path.to_string(), atom.to_string())).copied();
                    own.or_else(|| mean(path))
                });
            match b {
                Some(b) => out.push_str(&format!("{}{b:>6.2}{}\n", &line[..60], &line[66..])),
                None => out.push_str(&format!("{line}\n")),
            }
        }
        out
    }

    /// The report of a rescored site from that of the final model: the stages and the search of
    /// the first pass around it.
    fn report(&self, report: &mut Value, stage: Value) {
        let first = &self.report;
        let mut stages = first["stages"].clone();
        if let Some(s) = stages.as_array_mut() {
            s.push(stage);
        }
        report["stages"] = stages;
        if first["extend"]["ensemble"].is_null() {
            report["extend"]["weak"] = first["extend"]["weak"].clone();
        } else {
            report["extend"] = first["extend"].clone();
        }
        for k in [
            "basins",
            "support",
            "supported_residues",
            "unsupported_residues",
            "free_torsions",
            "alternatives",
            "completions",
            "maxent",
        ] {
            report["fit"][k] = first["fit"][k].clone();
        }
        let deviation = report["fit"]["costs"]["prior_deviation"].take();
        report["fit"]["costs"] = first["fit"]["costs"].clone();
        report["fit"]["costs"]["prior_deviation"] = deviation;
        report["fit"]["best"]["basin"] = Value::Null;
    }
}

fn stage_record(label: &str, sequence: &str, fit: &SiteFit, a: &Assessed, seconds: f64) -> Value {
    let best = &fit.outcome.basins[fit.outcome.best];
    json!({
        "stage": label, "sequence": sequence, "seconds": seconds,
        "objective": best.terms.total, "loglik": best.terms.loglik, "restraint": best.e_restraint,
        "from_known": best.from_known, "basins": fit.outcome.basins.len(),
        "data_weight": best.data_weight,
        "statistical": a.statistical, "supported": a.supported(),
        "density": a.density, "root_gain": a.gate.root_gain,
    })
}

fn distance(a: V3, b: V3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

fn atom_at(structure: &Structure, residue: &ResidueId, name: &str) -> Option<V3> {
    let p = structure.atom_position(structure.find_atom(residue, name)?)?;
    Some([p.x, p.y, p.z])
}

/// Width in degrees of the bins GlycoFlow's samples of a torsion are counted in.
const TORSION_BIN: f64 = 10.0;

/// The torsions of the glycosidic linkages (phi, psi and the omega of a 1-6 linkage) in the model
/// and in the deposit, against what GlycoFlow generates for the glycan: the angle, its percentile
/// (the share of GlycoFlow's samples that are less likely than it) and the samples themselves,
/// counted in bins of `TORSION_BIN` degrees from -180.
fn linkage_torsions(p: &SiteProblem, deposit: &[Option<V3>], x: &[V3]) -> Vec<Value> {
    let Some(prior) = p.prior.as_ref() else {
        return Vec::new();
    };
    let paths = &p.glycan.res_paths;
    let quads = &p.glycan.topology.quads;
    let mut neighbours = vec![Vec::new(); p.n_atoms];
    for &[a, b] in &p.glycan.bonds {
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    let parent_of: BTreeMap<&str, &str> = quads
        .iter()
        .filter(|q| paths[q[1]] != paths[q[2]])
        .map(|q| (paths[q[2]].as_str(), paths[q[1]].as_str()))
        .collect();
    let bridging = |i: usize| {
        p.glycan.elements[i] == "O" && neighbours[i].iter().any(|&j| paths[j] != paths[i])
    };
    // the residue a torsion carries, when it is one of its linkage to the parent
    let linkage = |t: usize| -> Option<(&'static str, &str)> {
        let [_, b, c, _] = quads[t];
        if paths[b] != paths[c] {
            return Some(("phi", paths[c].as_str()));
        }
        let child = p.glycan.topology.distal[t]
            .iter()
            .map(|&i| paths[i].as_str())
            .find(|r| parent_of.get(r) == Some(&paths[c].as_str()))?;
        Some((if bridging(c) { "psi" } else { "omega" }, child))
    };
    let fit: Vec<f64> = quads
        .iter()
        .map(|q| dihedral(x[q[0]], x[q[1]], x[q[2]], x[q[3]]))
        .collect();
    let dep: Vec<Option<f64>> = quads
        .iter()
        .map(|q| Some(dihedral(deposit[q[0]]?, deposit[q[1]]?, deposit[q[2]]?, deposit[q[3]]?)))
        .collect();
    let fit_pct = prior.percentiles(&fit);
    let dep_pct = prior.percentiles(&dep.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<_>>());
    let heavy = p.heavy_torsions();
    let bins = (360.0 / TORSION_BIN) as usize;
    (0..quads.len())
        .filter(|&t| heavy[t])
        .filter_map(|t| {
            let (kind, child) = linkage(t)?;
            if child == "agl" || paths[quads[t][1]] == "agl" {
                return None;
            }
            let mut counts = vec![0u32; bins];
            for s in 0..prior.n_samples {
                let deg = prior.samples[s * prior.n_torsions + t].to_degrees();
                let bin = ((deg + 180.0).rem_euclid(360.0) / TORSION_BIN) as usize;
                counts[bin.min(bins - 1)] += 1;
            }
            Some(json!({
                "kind": kind, "path": child,
                "atoms": quads[t].iter().map(|&i| p.glycan.atom_names[i].as_str()).collect::<Vec<_>>(),
                "fit": fit[t].to_degrees(), "fit_percentile": fit_pct[t],
                "deposit": dep[t].map(f64::to_degrees),
                "deposit_percentile": dep[t].map(|_| dep_pct[t]),
                "samples": counts,
            }))
        })
        .collect()
}

/// Grid of the side-chain torsions of a rebuilt asparagine (degrees).
const ASN_STEP: f64 = 10.0;
/// Rebuilt side chains tried at one site.
const ASN_TRIALS: usize = 2;
/// Two trials are at least this far apart in chi1 (degrees).
const ASN_APART: f64 = 60.0;
/// Closest approach of a rebuilt carbon, and of a rebuilt N or O, to the atoms around (A).
const ASN_CONTACT: [f64; 2] = [3.0, 2.6];

/// Side chains of the asparagine of a fitted site that would put a trans link within `reach` of
/// C1 of the fitted glycan, the nearest first, each with that distance; none of them in contact
/// with the rest of the structure.
fn side_chains_under(
    fit: &SiteFit,
    structure: &Structure,
    residue: &ResidueId,
    reach: f64,
) -> Vec<(f64, AsnSideChain)> {
    let at = |name: &str| atom_at(structure, residue, name);
    let (Some(n), Some(ca), Some(cb), Some(c), Some(o)) =
        (at("N"), at("CA"), at("CB"), at("C"), at("O"))
    else {
        return Vec::new();
    };
    let p = &fit.problem;
    let x = &fit.outcome.basins[fit.outcome.best].x;
    let Some(c1) = (0..p.n_atoms)
        .find(|&i| p.glycan.res_paths[i] == "r" && p.glycan.atom_names[i] == "C1")
        .map(|i| x[i])
    else {
        return Vec::new();
    };
    // the atoms around, without the residue itself
    let around: Vec<V3> = fit
        .site
        .environment
        .iter()
        .filter(|a| {
            !(a.is_identity() && a.chain == residue.chain && a.residue_number == residue.number)
        })
        .map(|a| a.position)
        .filter(|x| distance(*x, cb) < 8.0)
        .collect();
    let free = |s: &AsnSideChain| {
        let clear = |x: V3, d: f64| around.iter().all(|a| distance(x, *a) >= d);
        clear(s.cg, ASN_CONTACT[0])
            && clear(s.od1, ASN_CONTACT[1])
            && clear(s.nd2, ASN_CONTACT[1])
            // its own backbone
            && [n, c].iter().all(|b| distance(s.cg, *b) >= 2.75)
            && [n, c, o]
                .iter()
                .all(|b| distance(s.od1, *b) >= 2.6 && distance(s.nd2, *b) >= 2.6)
    };
    let mut all: Vec<(f64, AsnSideChain)> = asn_side_chains(n, ca, cb, ASN_STEP)
        .into_iter()
        .map(|s| (distance(s.c1, c1), s))
        .filter(|(d, s)| *d <= reach && free(s))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all
}

/// The first of `all`, and the next ones that are another rotamer than those taken.
fn trials(all: &[(f64, AsnSideChain)]) -> Vec<(f64, AsnSideChain)> {
    let apart = |a: f64, b: f64| ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs() >= ASN_APART;
    let mut taken: Vec<(f64, AsnSideChain)> = Vec::new();
    for (d, s) in all {
        if taken.len() < ASN_TRIALS && taken.iter().all(|(_, t)| apart(t.chi1, s.chi1)) {
            taken.push((*d, *s));
        }
    }
    taken
}

/// The protein with atoms of the side chain of the asparagine at `atoms`.
fn with_side_chain(
    protein: &Structure,
    residue: &ResidueId,
    atoms: &[(&str, V3)],
) -> Result<Structure, Error> {
    let mut rebuilt = protein.clone();
    for &(name, x) in atoms {
        let id = rebuilt
            .find_atom(residue, name)
            .ok_or("asparagine without its side chain")?;
        let position = Vec3 {
            x: x[0],
            y: x[1],
            z: x[2],
        };
        rebuilt.set_atom_position(id, position)?;
    }
    Ok(rebuilt)
}

/// The largest weight on the map in the final refinement of a cryo-EM site.
const CRYO_EM_DATA_WEIGHT: f64 = 32.0;

/// Refit one site; the summary line, and the files written under `dir`. With `rescore`, that
/// final model takes the place of the fits (steps 2 to 5), and the files are its own.
#[allow(clippy::too_many_arguments)]
fn fit_site(
    model: &GlycoflowModel,
    method: &str,
    site: &Value,
    dir: &Path,
    seed: u64,
    members: usize,
    priors: &PriorCache,
    rescore: Option<&Final>,
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
    // a map the fit does not see (the other half map, say), for the weight probe to score in
    let map2 = site["map2"].as_str().map(DensityMap::open).transpose()?;
    let residue = residue_id(field("site")?)?;
    let mut options = WorkflowOptions::default();
    options.fit.seed = seed;
    options.prior.cache = Some(priors.clone());
    // a chair set where the deposit has a boat is held between its neighbours: with the wider default
    // the ring settles half-way, twisted, so the rings are held to their chair more tightly here
    options.fit.restraints.sd_ring = 5.0;
    // a model that leaves the known positions has to explain the map at least as well as they do:
    // a pose the map likes less is not taken for its conformation alone
    options.fit.known_guard = 0.0;
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
    // the selected model is refined once more with the map weighted up, on the residues the map shows.
    // For a cryo-EM map this holds up in the half map the fit did not see; against a crystal's map, whose
    // phases come from the model, it fits noise (R-free rises), so those sites keep weight 1 here and get
    // their final refinement against the diffraction data. PDBWIDE_DW overrides (1 switches it off).
    let default_weight = (method == "cryo-EM").then_some(CRYO_EM_DATA_WEIGHT);
    if let Some(max) = number("PDBWIDE_DW").or(default_weight).filter(|&w| w > 1.0) {
        let mut weight = DataWeight::up_to(max);
        if let Some(s) = number("PDBWIDE_DW_STEPS") {
            weight.steps = s as usize;
        }
        if let Some(s) = number("PDBWIDE_DW_BOND") {
            weight.bond_rmsd = s;
        }
        if let Some(s) = number("PDBWIDE_DW_ANGLE") {
            weight.one_three_rmsd = s;
        }
        if let Some(s) = number("PDBWIDE_DW_RING") {
            weight.ring_rmsd = s;
        }
        options.fit.data_weight = Some(weight);
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
    // 1. the deposited tree with the core linkages and the anomers corrected
    let corrected = if n_linked {
        corrected_tree(&deposited_sequence, host)?
    } else {
        CorrectedTree::unchanged(&deposited_sequence)
    };
    // the deposited atoms by the paths of the corrected tree
    let deposited_atoms: BTreeMap<(String, String), V3> = deposited
        .atoms
        .iter()
        .map(|((path, name), x)| {
            let path = corrected.moved.get(path).unwrap_or(path).clone();
            ((path, name.clone()), *x)
        })
        .collect();
    // how far a placed glycan is from the deposit, residue by residue of the corrected tree
    let moved_deposit = DepositedGlycan {
        residues: deposited.residues.clone(),
        atoms: deposited_atoms.clone(),
    };
    let from_deposit = |fit: &SiteFit, x: &[V3]| recovery(&fit.problem, &moved_deposit, x, &[]);
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
    // the same with the side chain of the asparagine rebuilt (step 3)
    let rebuilt_protein: Structure;
    // or with the side chain of a final model (rescore)
    let final_protein: Structure;
    let refine = (options.fit.cartesian_lr, options.fit.restraints.clone());
    let problem_options = options.problem.clone();
    let mut fitter = Fitter {
        structure: &structure,
        text: text.as_deref(),
        map: &map,
        model,
        protein: &protein,
        residue: residue.clone(),
        options,
    };
    if let Some(side) = rescore.and_then(|r| r.asn.as_ref()) {
        final_protein = with_side_chain(&protein, &residue, side)?;
        fitter = fitter.moved(&final_protein, side);
    }
    // 2. the deposit's own glycan, also from its atoms; a residue with a corrected anomer only
    // takes its place from them, and so does one with a distorted ring; a cis N-acetyl group is
    // left out
    let deposit_tokens: BTreeMap<String, String> = sequence_paths(&corrected.sequence)?
        .into_iter()
        .map(|(path, (token, _))| (path, token))
        .collect();
    let deposit_atoms = KnownAtoms {
        atoms: deposited_atoms.clone(),
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
        atoms: deposited_atoms.clone(),
        rigid: corrected
            .changed
            .iter()
            .map(|c| c.0.clone())
            .chain(corrected.relinked.iter().map(|c| c.1.clone()))
            .collect(),
    }
    .without(&errors);
    let t0 = Instant::now();
    let (mut fit, mut assessed) = match rescore {
        None => fitter.fit(&corrected.sequence, Some(known))?,
        Some(r) => fitter.place(&corrected.sequence, r)?,
    };
    let mut fits = 1;
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
    let compare = |fit: &SiteFit, assessed: &Assessed| -> Value {
        let p = &fit.problem;
        let start = KnownAtoms {
            atoms: deposited_atoms.clone(),
            rigid: BTreeSet::new(),
        }
        .on(p);
        let best = &fit.outcome.basins[fit.outcome.best];
        let pose = pose_from_atoms(p, |i| start.x[i], &best.pose);
        let x0 = hybrid(p, &p.place(&pose), &start);
        let (lr, restraints) = (refine.0, &refine.1);
        let d = cartesian_refine(p, &x0, pose.template, 0, lr, restraints);
        let f = cartesian_refine(p, &best.x, best.template, 0, lr, restraints);
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
            // the selected basin and the lowest from the deposit under a rising weight of the map
            if std::env::var("PDBWIDE_PROBE").is_ok() {
                // the log-likelihood of a model in the unseen map
                let unseen = map2.as_ref().and_then(|m| {
                    density_problem(&fit.site, model, m, fit.sigma.selected, fit.resolution, &problem_options).ok()
                });
                let held_out = |x: &[V3]| match &unseen {
                    Some(q) => format!(" unseen {:.2}", q.placed_terms(x, 0.0, None).0.loglik),
                    None => String::new(),
                };
                if unseen.is_some() {
                    eprintln!("  deposit{}", held_out(&d.x));
                }
                let known = fit
                    .outcome
                    .order
                    .iter()
                    .copied()
                    .find(|&i| fit.outcome.basins[i].from_known);
                for k in [Some(fit.outcome.best), known].into_iter().flatten() {
                    let b = &fit.outcome.basins[k];
                    let r = Restraints::new(p, b.template, restraints);
                    eprintln!(
                        "  probe basin #{k} (weight {}): loglik {:8.2} prior {:6.2} restraint {:7.2} rmsd {:.2}{}",
                        b.data_weight, b.terms.loglik, b.terms.e_prior, b.e_restraint, rmsd(&b.x), held_out(&b.x)
                    );
                    // PDBWIDE_PROBE_OUT=<dir>: the model of every weight, for cross-validation
                    let dump = |w: f64, x: &[V3]| -> Result<(), Error> {
                        let Ok(dir) = std::env::var("PDBWIDE_PROBE_OUT") else {
                            return Ok(());
                        };
                        let text = glycan_pdb(&[PlacedGlycan {
                            problem: p,
                            site: &fit.site,
                            naming: &fit.naming,
                            x,
                        }])?;
                        std::fs::create_dir_all(&dir)?;
                        let tag = if k == fit.outcome.best { "selected" } else { "known" };
                        std::fs::write(format!("{dir}/{name}.{tag}.w{w}.pdb"), text)?;
                        Ok(())
                    };
                    dump(0.0, &b.x).expect("probe model");
                    let mut x = b.x.clone();
                    for w in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] {
                        let c = cartesian_refine_weighted(p, &x, b.template, 600, lr, restraints, w, None, None);
                        dump(w, &c.x).expect("probe model");
                        eprintln!(
                            "    x{w:<3}: loglik {:8.2} prior {:6.2} att {:5.2} contact {:.4} restraint {:7.1} bonds {:.4} 1-3 {:.4} ring {:4.1} rmsd {:.2}{}",
                            c.terms.loglik,
                            c.terms.e_prior,
                            c.terms.e_att,
                            p.worst_contact(&c.x),
                            c.e_restraint,
                            c.bond_rmsd,
                            c.one_three_rmsd,
                            r.ring_rmsd(&c.x),
                            rmsd(&c.x),
                            held_out(&c.x)
                        );
                        if w > 1.0 {
                            x = c.x;
                        }
                    }
                    if known == Some(fit.outcome.best) {
                        break;
                    }
                }
            }
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
                let pts: Vec<V3> = deposited_atoms
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
                "prior": c.terms.e_prior,
                "bond_rmsd": bond_rmsd, "one_three_rmsd": one_three_rmsd,
                "clash": c.terms.e_env + c.terms.e_self,
            })
        };
        let away = from_deposit(fit, &best.x);
        json!({
            "deposit": side(&d), "fit": side(&f), "density_deposit": density,
            "density_fit": assessed.density,
            "from_deposit": best.from_known,
            "data_weight": best.data_weight,
            "ring_rmsd": Restraints::new(p, best.template, restraints).ring_rmsd(&best.x),
            "psi_n_deposit": pose.psi.to_degrees(),
            "matched_atoms": start.x.iter().filter(|x| x.is_some()).count(),
            "rmsd": away.full_rmsd,
            "core_rmsd": away.core_rmsd,
            "per_residue_rmsd": away.per_residue,
        })
    };
    let mut comparison = compare(&fit, &assessed);
    // rescore: the deposit's side of the comparison is the first pass's; how far the deposit's
    // log-likelihood is from it here tells whether this is the same map and the same objective
    let mut deposit_drift = None;
    if let Some(r) = rescore {
        let was = &r.first["comparison"];
        let loglik = |c: &Value| c["deposit"]["loglik"].as_f64();
        deposit_drift = loglik(&comparison)
            .zip(loglik(was))
            .map(|(now, was)| now - was);
        for k in [
            "deposit",
            "density_deposit",
            "from_deposit",
            "data_weight",
            "psi_n_deposit",
            "matched_atoms",
        ] {
            comparison[k] = was[k].clone();
        }
    }
    // 3. a cis amide the fit could not turn on the deposited side chain: the side chain rebuilt
    // under the glycan, kept when the link is then trans, the glycan stays in its pose and the
    // map holds the glycan and the side chain as well as before
    let psi_of = |fit: &SiteFit| fit.outcome.basins[fit.outcome.best].pose.psi.to_degrees();
    let mut asn = Value::Null;
    if rescore.is_none() && n_linked && psi_of(&fit).abs() < 90.0 {
        let reach = number("PDBWIDE_ASN_REACH").unwrap_or(2.5);
        let at = |name: &str| atom_at(&structure, &residue, name);
        let old = [at("CG"), at("OD1"), at("ND2")];
        let level = |x: &[V3]| assessed.levels.as_ref().and_then(|l| l.fraction(&map, x));
        let old_level = match old {
            [Some(a), Some(b), Some(c)] => level(&[a, b, c]),
            _ => None,
        };
        let (old_psi, old_loglik, old_known) = {
            let b = &fit.outcome.basins[fit.outcome.best];
            (psi_of(&fit), b.terms.loglik, b.from_known)
        };
        let all = side_chains_under(&fit, &structure, &residue, reach);
        let mut tried = Vec::new();
        let mut taken = None;
        for (away, side) in trials(&all) {
            let Some(old_level) = old_level else {
                break;
            };
            let t = Instant::now();
            let rebuilt = with_side_chain(&protein, &residue, &side.atoms())?;
            let x = &fit.outcome.basins[fit.outcome.best].x;
            let known = KnownAtoms::of_fit(&fit.problem, x, None);
            let (r_fit, r_assessed) = fitter
                .moved(&rebuilt, &side.atoms())
                .fit(&sequence, Some(known))?;
            fits += 1;
            let b = &r_fit.outcome.basins[r_fit.outcome.best];
            let (psi, loglik) = (psi_of(&r_fit), b.terms.loglik);
            let new_level = level(&[side.cg, side.od1, side.nd2]);
            let trans = psi.abs() >= 90.0;
            let holds = loglik >= old_loglik - fitter.options.fit.alt_margin
                || (old_loglik > 0.0 && loglik >= fitter.options.fit.known_share * old_loglik);
            let dense = new_level.is_some_and(|l| l >= DENSITY_GATE.min(old_level));
            // the glycan stays where the fit had it: a refit that leaves for another pose is
            // not this glycan with its side chain turned
            let stays = b.from_known;
            let accepted = trans && holds && dense && stays;
            tried.push(json!({
                "chi1": side.chi1, "chi2": side.chi2, "away": away, "psi": psi,
                "loglik": loglik, "level": new_level, "stays": stays, "accepted": accepted,
                "seconds": t.elapsed().as_secs_f64(),
            }));
            if accepted {
                stages.push(stage_record(
                    "asparagine side chain rebuilt",
                    &sequence,
                    &r_fit,
                    &r_assessed,
                    t.elapsed().as_secs_f64(),
                ));
                taken = Some((side, rebuilt, r_fit, r_assessed, new_level));
                break;
            }
        }
        let chi = match (at("N"), at("CA"), at("CB"), old) {
            (Some(n), Some(ca), Some(cb), [Some(cg), Some(od1), _]) => Some([
                dihedral(n, ca, cb, cg).to_degrees(),
                dihedral(ca, cb, cg, od1).to_degrees(),
            ]),
            _ => None,
        };
        asn = json!({
            "rebuilt": taken.is_some(), "candidates": all.len(), "tried": tried,
            "chi_deposit": chi, "psi_deposit_side_chain": old_psi,
            "loglik_deposit_side_chain": old_loglik, "level_deposit_side_chain": old_level,
        });
        if let Some((side, rebuilt, r_fit, r_assessed, new_level)) = taken {
            rebuilt_protein = rebuilt;
            fitter = fitter.moved(&rebuilt_protein, &side.atoms());
            (fit, assessed) = (r_fit, r_assessed);
            let c = compare(&fit, &assessed);
            for k in [
                "fit",
                "density_fit",
                "rmsd",
                "core_rmsd",
                "per_residue_rmsd",
            ] {
                comparison[k] = c[k].clone();
            }
            comparison["from_deposit"] = json!(old_known && c["from_deposit"] == true);
            asn["chi"] = json!([side.chi1, side.chi2]);
            asn["psi"] = json!(psi_of(&fit));
            asn["loglik"] = json!(fit.outcome.basins[fit.outcome.best].terms.loglik);
            asn["level"] = json!(new_level);
            asn["shift"] = json!(old[2].map(|x| distance(x, side.nd2)));
            asn["atoms"] = json!({"CG": side.cg, "OD1": side.od1, "ND2": side.nd2});
        }
    }
    // 4. the residues the map does not hold, placed by GlycoFlow on the ones it does: where the
    // map is silent the deposited pose is no evidence, so the model takes the conformation the
    // glycan has there, when that is clearly the more usual one and costs the residues in density
    // nothing
    let mut replaced: Vec<String> = Vec::new();
    let free: BTreeSet<String> = assessed
        .density
        .keys()
        .filter(|path| {
            let below = format!("{path}/");
            !assessed
                .density
                .iter()
                .any(|(q, &level)| level >= DENSITY_GATE && (q == *path || q.starts_with(&below)))
        })
        .cloned()
        .collect();
    let held: BTreeSet<String> = deposit_paths.difference(&free).cloned().collect();
    if rescore.is_none() && !free.is_empty() && held.contains("r") {
        let x = &fit.outcome.basins[fit.outcome.best].x;
        let known = KnownAtoms::of_fit(&fit.problem, x, Some(&held));
        let t = Instant::now();
        let (f_fit, f_assessed) = fitter.fit(&sequence, Some(known))?;
        fits += 1;
        // clearly the better conformation, and no worse a model on the whole
        let terms = |fit: &SiteFit| fit.outcome.basins[fit.outcome.best].terms.clone();
        let (old, new) = (terms(&fit), terms(&f_fit));
        let taken = new.e_prior < old.e_prior - fitter.options.fit.alt_margin
            && new.total < old.total
            && assessed.supported().is_subset(&f_assessed.supported())
            && !loses_density(&assessed.density, &f_assessed.density);
        let mut record = stage_record(
            "residues without density placed by GlycoFlow",
            &sequence,
            &f_fit,
            &f_assessed,
            t.elapsed().as_secs_f64(),
        );
        record["free"] = json!(free);
        record["prior"] = json!([old.e_prior, new.e_prior]);
        record["taken"] = json!(taken);
        stages.push(record);
        if taken {
            (fit, assessed) = (f_fit, f_assessed);
            let c = compare(&fit, &assessed);
            for k in [
                "fit",
                "density_fit",
                "rmsd",
                "core_rmsd",
                "per_residue_rmsd",
            ] {
                comparison[k] = c[k].clone();
            }
            // the residues in density are where the first fit had them
            comparison["from_deposit"] =
                json!(comparison["from_deposit"] == true && c["from_deposit"] == true);
            replaced = free.into_iter().collect();
        }
    }
    // 5. growth along the glycans of the expression system, one layer of residues at a time
    let mut tree = deposit_paths.clone();
    let mut tried: BTreeSet<(String, String)> = BTreeSet::new();
    let mut added: Vec<String> = Vec::new();
    let limit = fits + MAX_FITS - 1;
    while rescore.is_none() && fits < limit {
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
        // an added residue needs its own support and that of the residue it is linked to, and
        // the layer is refused when a residue the map supported loses its support to it, or a
        // residue in density a good part of its density
        let g_supported = g_assessed.supported();
        let costs = !built.is_subset(&g_supported)
            || loses_density(&assessed.density, &g_assessed.density);
        let accepted: Vec<String> = frontier
            .iter()
            .map(|(p, _)| p.clone())
            .filter(|p| {
                !costs
                    && g_supported.contains(p)
                    && parent(p).is_some_and(|q| g_supported.contains(q))
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
        record["costs"] = json!(costs);
        stages.push(record);
        tried.extend(frontier.iter().cloned());
        if accepted.is_empty() {
            continue;
        }
        if accepted.len() == frontier.len() {
            tree.extend(accepted.iter().cloned());
            added.extend(accepted.iter().cloned());
            (fit, assessed, sequence) = (g_fit, g_assessed, grown);
            continue;
        }
        // some of the layer was refused: the model is the fit of the accepted tree alone, when
        // that fit keeps the support of the accepted residues and of those before them
        let mut wider = tree.clone();
        wider.extend(accepted.iter().cloned());
        let (kept, _) = pruned_sequence(s, &wider)?;
        let gx = &g_fit.outcome.basins[g_fit.outcome.best].x;
        let known = KnownAtoms::of_fit(&g_fit.problem, gx, Some(&wider));
        let t = Instant::now();
        let (k_fit, k_assessed) = fitter.fit(&kept, Some(known))?;
        fits += 1;
        let k_supported = k_assessed.supported();
        let holds = built.is_subset(&k_supported)
            && accepted.iter().all(|p| k_supported.contains(p))
            && !loses_density(&assessed.density, &k_assessed.density);
        let mut record = stage_record(
            "refit of the accepted residues",
            &kept,
            &k_fit,
            &k_assessed,
            t.elapsed().as_secs_f64(),
        );
        record["kept"] = json!(holds);
        stages.push(record);
        if holds {
            tree = wider;
            added.extend(accepted.iter().cloned());
            (fit, assessed, sequence) = (k_fit, k_assessed, kept);
        }
    }
    // rescore: the final model on the tree of the first pass, with its residue names
    let mut missing_atoms = 0;
    if let Some(r) = rescore {
        let grown = r.first["sequence"]
            .as_str()
            .ok_or("first pass without a sequence")?;
        if grown != sequence {
            (fit, assessed) = fitter.place(grown, r)?;
            sequence = grown.to_string();
        }
        if json!(fit.naming) != r.report["fit"]["naming"] {
            return Err("final model: not the residues of the first pass".into());
        }
        let p = &fit.problem;
        let given = KnownAtoms {
            atoms: r.atoms.clone(),
            rigid: BTreeSet::new(),
        }
        .on(p)
        .x;
        for n in &fit.naming {
            if r.ccd.get(&n.path) != Some(&n.ccd) {
                return Err(format!("final model: {} is not {}", n.path, n.ccd).into());
            }
            let atoms = |known: bool| {
                (0..p.n_atoms)
                    .filter(|&i| p.keep[i] && p.glycan.res_paths[i] == n.path)
                    .filter(|&i| given[i].is_some() == known)
                    .count()
            };
            // fewer atoms do not place a residue (`known::hybrid`)
            if atoms(true) < 4 {
                return Err(format!("final model: {} has {} atoms", n.path, atoms(true)).into());
            }
            missing_atoms += atoms(false);
        }
    }
    let fit_seconds = t0.elapsed().as_secs_f64();
    // the model: every deposited residue, and the added ones
    let best = &fit.outcome.basins[fit.outcome.best];
    let tokens: BTreeMap<String, String> = sequence_paths(&sequence)?
        .into_iter()
        .map(|(path, (token, anomer))| (path, format!("{token}{anomer}")))
        .collect();
    let changed: BTreeSet<&str> = corrected
        .changed
        .iter()
        .map(|c| c.0.as_str())
        .chain(corrected.relinked.iter().map(|c| c.1.as_str()))
        .collect();
    let away = from_deposit(&fit, &best.x);
    // rescore: the residues the first pass built
    let supported: BTreeSet<String> = match rescore {
        Some(r) => r.list("built").into_iter().collect(),
        None => assessed.supported(),
    };
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
    // a core residue on another position of its parent than in any N-glycan, now where they
    // have it; and what of the core no known N-glycan has and the correction had to leave
    let position = |path: &str| path.rsplit('/').next().and_then(|p| p.parse::<u32>().ok());
    for (from, to, residue) in &corrected.relinked {
        deposit_errors.push(json!({
            "path": to, "kind": "linkage", "residue": residue,
            "deposit": position(from), "fit": position(to),
        }));
    }
    if n_linked {
        for (path, fault) in core_faults(&corrected.sequence, host)? {
            let kind = match fault {
                CoreFault::Linkage => "linkage",
                CoreFault::Residue => "residue",
            };
            let residue = tokens.get(&path);
            deposit_errors.push(json!({
                "path": path, "kind": kind, "residue": residue,
                "deposit": position(&path), "fit": position(&path),
            }));
            fit_errors.push(json!({
                "path": path, "kind": kind, "residue": residue, "fit": position(&path),
            }));
        }
    }
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
                "rmsd": away.per_residue.get(&path),
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
    let extension = if members > 0 && rescore.is_none() {
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
    match rescore {
        Some(r) => {
            let models = r.with_b(&models, &fit.naming);
            std::fs::write(dir.join(format!("{name}.final.pdb")), models)?
        }
        None => std::fs::write(dir.join(format!("{name}.pdb")), models)?,
    }
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
    // the linkage torsions of the final model and of the deposit against GlycoFlow's own
    let torsions = linkage_torsions(
        &fit.problem,
        &KnownAtoms {
            atoms: deposited_atoms.clone(),
            rigid: BTreeSet::new(),
        }
        .on(&fit.problem)
        .x,
        &best.x,
    );
    let mut summary = json!({
        "recipe": "v2.1",
        "deposited": deposited_sequence,
        "deposited_residues": deposit_paths.len(),
        "glycan": glycan, "sequence": sequence,
        "corrections": corrected.changed,
        "relinked": corrected.relinked,
        "asn": asn,
        "deposit_errors": deposit_errors,
        "fit_errors": fit_errors,
        "typical": typical, "elsewhere": elsewhere, "whole_glycan": whole_glycan,
        "added": added,
        "replaced": replaced,
        "residues": residues,
        "deposited_supported": count(&["deposited", "corrected"], "supported"),
        "deposited_weak": count(&["deposited", "corrected"], "weak"),
        "deposited_unsupported": count(&["deposited", "corrected"], "unsupported"),
        "added_supported": count(&["added"], "supported"),
        "comparison": comparison,
        "torsions": torsions,
        "built": supported, "weak": weak,
        "density_fraction": assessed.density,
        "root_gain": assessed.gate.root_gain,
        "clusters": extension.as_ref().and_then(|e| e.ensemble.as_ref()).map(|e| e.clusters.len()),
        "objective": best.terms.total, "loglik": best.terms.loglik,
        "core_rmsd": away.core_rmsd,
        "full_rmsd": away.full_rmsd,
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
    // rescore: the decisions of the first pass around the numbers of the final model
    if let Some(r) = rescore {
        let mut asn = r.first["asn"].clone();
        if let Some(side) = &r.asn {
            let [cg, od1, nd2] = side.map(|(_, x)| x);
            let at = |name: &str| atom_at(&structure, &residue, name);
            let chi = match (at("N"), at("CA"), at("CB")) {
                (Some(n), Some(ca), Some(cb)) => Some([
                    dihedral(n, ca, cb, cg).to_degrees(),
                    dihedral(ca, cb, cg, od1).to_degrees(),
                ]),
                _ => None,
            };
            let levels = assessed.levels.as_ref();
            asn["chi"] = json!(chi);
            asn["psi"] = json!(psi);
            asn["loglik"] = json!(best.terms.loglik);
            asn["level"] = json!(levels.and_then(|l| l.fraction(&map, &[cg, od1, nd2])));
            asn["shift"] = json!(at("ND2").map(|x| distance(x, nd2)));
            asn["atoms"] = json!({"CG": cg, "OD1": od1, "ND2": nd2});
            asn["rebuilt"] = true.into();
        } else if asn["rebuilt"] == true {
            // the side chain is the deposited one again: as a site whose rebuild was not accepted
            let tried = asn["tried"].as_array().cloned().unwrap_or_default();
            let mut back = json!({"rebuilt": false, "reverted": true, "tried": tried});
            for k in [
                "candidates",
                "chi_deposit",
                "level_deposit_side_chain",
                "loglik_deposit_side_chain",
                "psi_deposit_side_chain",
            ] {
                back[k] = asn[k].clone();
            }
            back["first_pass"] = asn;
            asn = back;
        }
        // how far the final model is from the first pass's, residue by residue
        let p = &fit.problem;
        let mut apart: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        for (i, e) in r.engine_on(p).iter().enumerate() {
            if let Some(e) = e {
                let sum = apart.entry(p.glycan.res_paths[i].as_str()).or_default();
                sum.0 += distance(*e, best.x[i]).powi(2);
                sum.1 += 1;
            }
        }
        let rms = |(sum, n): (f64, usize)| (sum / n.max(1) as f64).sqrt();
        let all = apart.values().fold((0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let per_residue: BTreeMap<&str, f64> = apart.iter().map(|(k, v)| (*k, rms(*v))).collect();
        let scored = json!({
            "choice": r.sidecar["choice"],
            "engine": {
                "objective": r.first["objective"], "loglik": r.first["loglik"],
                "rmsd": rms(all), "per_residue_rmsd": per_residue,
            },
            "built": assessed.supported(), "statistical": assessed.statistical,
            "asn_moved": r.asn.is_some(),
            "missing_atoms": missing_atoms,
            "deposit_loglik_drift": deposit_drift,
        });
        summary = r.summary(summary, asn, scored);
    }
    let mut report = json!({
        "name": name, "summary": summary, "stages": stages,
        "fit": site_report(&fit, &input), "extend": extend,
    });
    let file = match rescore {
        Some(r) => {
            let seconds = t0.elapsed().as_secs_f64();
            let stage = stage_record("final model", &sequence, &fit, &assessed, seconds);
            r.report(&mut report, stage);
            format!("{name}.final.report.json")
        }
        None => format!("{name}.json"),
    };
    std::fs::write(dir.join(file), report.to_string())?;
    Ok(summary)
}

/// The lines of the files of `dir` named `<stem>*.jsonl`.
fn lines_of(dir: &Path, stem: &str) -> Result<Vec<Value>, Error> {
    let mut rows = Vec::new();
    for file in std::fs::read_dir(dir)? {
        let file = file?;
        let name = file.file_name().to_string_lossy().into_owned();
        if !(name.starts_with(stem) && name.ends_with(".jsonl")) {
            continue;
        }
        for line in std::fs::read_to_string(file.path())?.lines() {
            rows.extend(serde_json::from_str::<Value>(line).ok());
        }
    }
    Ok(rows)
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
    // PDBWIDE_FINAL: score the final models of the sites that have one ([`Final`]), into final.jsonl
    let rescore = std::env::var("PDBWIDE_FINAL").is_ok_and(|v| v != "0" && !v.is_empty());
    let stem = if rescore { "final" } else { "results" };
    let results = out.join(if shards > 1 {
        format!("{stem}.{shard}.jsonl")
    } else {
        format!("{stem}.jsonl")
    });
    let mut done = HashSet::new();
    // the rows of the first pass by site; a final model is scored once, whatever the shard
    let mut first: BTreeMap<String, Value> = BTreeMap::new();
    if rescore {
        for row in lines_of(out, "results")? {
            if let (Some(name), "ok") = (row["name"].as_str(), row["status"].as_str().unwrap_or(""))
            {
                first.insert(name.to_string(), row.clone());
            }
        }
        for row in lines_of(out, "final")? {
            done.extend(row["name"].as_str().map(str::to_string));
        }
    }
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
        let method = entry["method"].as_str().unwrap_or("");
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
            let dir = out.join(id);
            let was = first.get(name);
            if rescore && (was.is_none() || !dir.join(format!("{name}.final.json")).exists()) {
                continue;
            }
            let t = Instant::now();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let model_of = |row: &Value| Final::read(&dir, name, &row["result"]);
                let rescore = was.map(model_of).transpose()?;
                fit_site(
                    &model,
                    method,
                    site,
                    &dir,
                    seed,
                    members,
                    &priors,
                    rescore.as_ref(),
                )
            }))
            .unwrap_or_else(|_| Err("panic".into()))
            .map_err(|e| e.to_string());
            let mut row = json!({
                "name": name, "entry": id, "method": entry["method"], "site": site["site"],
                "host_class": site["host_class"], "seconds": t.elapsed().as_secs_f64(),
            });
            match outcome {
                Ok(mut summary) => {
                    fitted += 1;
                    // a rescored site keeps the time of its fit
                    if let Some(was) = was {
                        summary["final"]["seconds"] = row["seconds"].clone();
                        row["seconds"] = was["seconds"].clone();
                    }
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
