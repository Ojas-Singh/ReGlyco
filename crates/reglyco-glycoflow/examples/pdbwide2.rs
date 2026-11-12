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
//!    by the objective than the deposit with ideal geometry;
//! 3. a cis amide between the asparagine and the glycan that the fit could not turn: the side
//!    chain of the asparagine is rebuilt under the glycan (`sidechain`), and kept when the link
//!    is then trans, the glycan stays in its pose and the map holds the glycan and the side
//!    chain as well as before;
//! 4. growth: while the map supports the outermost residues, the next residues of the glycans of
//!    the expression system are fitted on top of the model, and those that pass the support test
//!    and the density gate are added, unless a residue the map supported loses its support to
//!    them.
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
use glysys::{BuildOptions, ResidueId, Structure, Vec3, read_pdb};
use reglyco_density::DensityMap;
use reglyco_glycoflow::cartesian::{Restraints, cartesian_refine};
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
use reglyco_glycoflow::output::{PlacedGlycan, glycan_pdb};
use reglyco_glycoflow::problem::{SiteProblem, V3, dihedral};
use reglyco_glycoflow::sidechain::{AsnSideChain, asn_side_chains};
use reglyco_glycoflow::site::{DepositedGlycan, deposited_glycan, sequence_paths};
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

/// The protein with the side chain of the asparagine at `s`.
fn with_side_chain(
    protein: &Structure,
    residue: &ResidueId,
    s: &AsnSideChain,
) -> Result<Structure, Error> {
    let mut rebuilt = protein.clone();
    for (name, x) in s.atoms() {
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
    let refine = (options.fit.cartesian_lr, options.fit.restraints.clone());
    let mut fitter = Fitter {
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
    let (mut fit, mut assessed) = fitter.fit(&corrected.sequence, Some(known))?;
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
            "psi_n_deposit": pose.psi.to_degrees(),
            "matched_atoms": start.x.iter().filter(|x| x.is_some()).count(),
            "rmsd": away.full_rmsd,
            "core_rmsd": away.core_rmsd,
            "per_residue_rmsd": away.per_residue,
        })
    };
    let mut comparison = compare(&fit, &assessed);
    // 3. a cis amide the fit could not turn on the deposited side chain: the side chain rebuilt
    // under the glycan, kept when the link is then trans, the glycan stays in its pose and the
    // map holds the glycan and the side chain as well as before
    let psi_of = |fit: &SiteFit| fit.outcome.basins[fit.outcome.best].pose.psi.to_degrees();
    let mut asn = Value::Null;
    if n_linked && psi_of(&fit).abs() < 90.0 {
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
            let rebuilt = with_side_chain(&protein, &residue, &side)?;
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
    // 4. growth along the glycans of the expression system, one layer of residues at a time
    let mut tree = deposit_paths.clone();
    let mut tried: BTreeSet<(String, String)> = BTreeSet::new();
    let mut added: Vec<String> = Vec::new();
    let limit = fits + MAX_FITS - 1;
    while fits < limit {
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
    let summary = json!({
        "recipe": "v2",
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
