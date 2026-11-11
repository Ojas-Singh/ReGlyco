//! Glycan out of density: no sequence given (`glycoflow/fitting/infer.py`).
//!
//! Generous candidate glycans (high-mannose, complex, hybrid, ...) are fitted on one shared
//! scoring region. Each fit is pruned to the subtrees that pass the calibrated support test
//! (5 + 0.5 per torsion), and the root residue itself must pass the same test against "no
//! glycan". Every distinct pruned tree is refitted on its own (unsupported branches distort a
//! fit), and the tree with the best penalised log-likelihood gain is chosen, so a candidate wins
//! only with the density of the residues it adds.
//!
//! A residue is *built* only when it also sits in density at least `DENSITY_GATE` of the way from
//! the local bulk-solvent level to the local protein level (and its parent is built). Depositors
//! rarely model weaker density (94-100% of deposited residues in the PDB benchmark lie above 0.3);
//! residues that pass the statistical test below the gate are reported as weak, not built. When the map supports only the core, all
//! candidates prune to the same core and that core is the answer.

use std::collections::{BTreeMap, BTreeSet};

use glycoflow_core::sequence::{Sequence, parse_glycam};
use glysys::{ResidueId, Structure};

use crate::error::Result;
use reglyco_density::DensityMap;

use crate::observation::{Observation, ObservationValue, ObservedAtoms};
use crate::pipeline::FitOutcome;
use crate::problem::{SiteProblem, V3, build_glycan, decoy_poses, max_span};
use crate::site::{DepositedGlycan, glycam_of, sequence_paths};
use crate::workflow::{SiteFit, SiteRequest, WorkflowInput, fit_one};

/// Default N-glycan candidates (the largest common forms; pruning finds the rest).
pub const N_GLYCAN_CANDIDATES: [(&str, &str); 4] = [
    (
        "high-mannose (Man9)",
        "DManpa1-2DManpa1-6[DManpa1-2DManpa1-3]DManpa1-6[DManpa1-2DManpa1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
    ),
    (
        "complex biantennary, core Fuc",
        "DGalpb1-4DGlcpNAcb1-2DManpa1-6[DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH",
    ),
    (
        "complex biantennary, sialylated, core Fuc",
        "DNeup5Aca2-6DGalpb1-4DGlcpNAcb1-2DManpa1-6[DNeup5Aca2-6DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH",
    ),
    (
        "hybrid",
        "DManpa1-6[DManpa1-3]DManpa1-6[DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
    ),
];

/// O-glycan candidates for Ser/Thr: the mucin GalNAc cores (sialylated core 2 holds Tn, T,
/// sialyl-T and core 2; 2,6-sialyl-T), O-mannose (mammalian core M1 and the fungal
/// alpha1-2 chain), O-fucose (Notch EGF), O-glucose (EGF, xylosylated) and O-GlcNAc, and each
/// family's root sugar alone: a large candidate whose root does not settle in the density is never
/// pruned to that root (5T5L a:102, a lone GalNAc, went to O-fucose without them).
pub const O_GLYCAN_CANDIDATES: [(&str, &str); 11] = [
    ("Tn (GalNAc)", "DGalpNAca1-OH"),
    ("O-mannose", "DManpa1-OH"),
    ("O-fucose", "LFucpa1-OH"),
    ("O-glucose", "DGlcpb1-OH"),
    (
        "mucin core 2, sialylated",
        "DNeup5Aca2-3DGalpb1-4DGlcpNAcb1-6[DNeup5Aca2-3DGalpb1-3]DGalpNAca1-OH",
    ),
    (
        "mucin 2,6-sialyl T",
        "DNeup5Aca2-3DGalpb1-3[DNeup5Aca2-6]DGalpNAca1-OH",
    ),
    (
        "O-mannose core M1",
        "DNeup5Aca2-3DGalpb1-4DGlcpNAcb1-2DManpa1-OH",
    ),
    ("O-mannose, fungal", "DManpa1-2DManpa1-2DManpa1-OH"),
    (
        "O-fucose, extended",
        "DNeup5Aca2-3DGalpb1-4DGlcpNAcb1-3LFucpa1-OH",
    ),
    ("O-glucose, xylosylated", "DXylpa1-3DXylpa1-3DGlcpb1-OH"),
    ("O-GlcNAc", "DGlcpNAcb1-OH"),
];

/// Candidates for the other glycosylated residues.
pub const C_MANNOSE_CANDIDATES: [(&str, &str); 1] = [("C-mannose", "DManpa1-OH")];
pub const TYROSINE_CANDIDATES: [(&str, &str); 1] =
    [("glycogenin glucan", "DGlcpa1-4DGlcpa1-4DGlcpa1-OH")];
pub const HYDROXYPROLINE_CANDIDATES: [(&str, &str); 1] = [(
    "extensin arabinoside",
    "LArafa1-3LArafb1-2LArafb1-2LArafb1-OH",
)];
pub const HYDROXYLYSINE_CANDIDATES: [(&str, &str); 1] =
    [("collagen glucosylgalactose", "DGlcpa1-2DGalpb1-OH")];
pub const CYSTEINE_CANDIDATES: [(&str, &str); 2] =
    [("S-glucose", "DGlcpb1-OH"), ("S-GlcNAc", "DGlcpNAcb1-OH")];

/// The inference candidates of a site residue (CCD name): the largest common forms of the
/// glycans it carries; pruning by density finds the rest. Empty for residues without glycans.
pub fn candidates_for(residue: &str) -> &'static [(&'static str, &'static str)] {
    match residue {
        "ASN" => &N_GLYCAN_CANDIDATES,
        "SER" | "THR" => &O_GLYCAN_CANDIDATES,
        "TRP" => &C_MANNOSE_CANDIDATES,
        "TYR" => &TYROSINE_CANDIDATES,
        "HYP" => &HYDROXYPROLINE_CANDIDATES,
        "LYZ" => &HYDROXYLYSINE_CANDIDATES,
        "CYS" => &CYSTEINE_CANDIDATES,
        _ => &[],
    }
}

const CORE: &str = "DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH";
const CORE_FUC: &str = "DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH";
const MAN5: &str = "DManpa1-6[DManpa1-3]DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH";
const MAN9: &str = "DManpa1-2DManpa1-6[DManpa1-2DManpa1-3]DManpa1-6[DManpa1-2DManpa1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH";
const COMPLEX_FUC: &str = "DGalpb1-4DGlcpNAcb1-2DManpa1-6[DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH";
const COMPLEX_SIA_FUC: &str = "DNeup5Aca2-6DGalpb1-4DGlcpNAcb1-2DManpa1-6[DNeup5Aca2-6DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH";
const HYBRID: &str = "DManpa1-6[DManpa1-3]DManpa1-6[DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH";

/// What to extend an N-glycan to, by the expression system of the protein
/// (`infer.N_GLYCAN_SUGGESTIONS`): the first is the usual product; a fit page offers the list and
/// the user picks.
pub const N_GLYCAN_SUGGESTIONS: [(&str, &[(&str, &str)]); 5] = [
    (
        "mammalian",
        &[
            ("complex biantennary, core Fuc", COMPLEX_FUC),
            ("complex biantennary, sialylated, core Fuc", COMPLEX_SIA_FUC),
            ("Man5 (GnTI-deficient cells)", MAN5),
            ("high-mannose (Man9)", MAN9),
            ("hybrid", HYBRID),
            ("core (Man3GlcNAc2), core Fuc", CORE_FUC),
        ],
    ),
    (
        "insect",
        &[
            ("paucimannose, core Fuc a1-6", CORE_FUC),
            (
                "paucimannose, core Fuc a1-3 and a1-6",
                "DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-3][LFucpa1-6]DGlcpNAcb1-OH",
            ),
            ("paucimannose (Man3GlcNAc2)", CORE),
            ("Man5", MAN5),
            ("high-mannose (Man9)", MAN9),
        ],
    ),
    (
        "yeast or fungus",
        &[
            ("high-mannose (Man9)", MAN9),
            ("Man5", MAN5),
            ("core (Man3GlcNAc2)", CORE),
        ],
    ),
    (
        "plant",
        &[
            (
                "paucimannose with Xyl and Fuc a1-3",
                "DManpa1-6[DXylpb1-2][DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-3]DGlcpNAcb1-OH",
            ),
            (
                "complex with Xyl and Fuc a1-3",
                "DGlcpNAcb1-2DManpa1-6[DXylpb1-2][DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-3]DGlcpNAcb1-OH",
            ),
            ("core (Man3GlcNAc2)", CORE),
        ],
    ),
    (
        // expression system not known
        "",
        &[
            ("core (Man3GlcNAc2)", CORE),
            ("core (Man3GlcNAc2), core Fuc", CORE_FUC),
            ("high-mannose (Man9)", MAN9),
            ("complex biantennary, core Fuc", COMPLEX_FUC),
        ],
    ),
];

/// Expression hosts by what they do to an N-glycan (lower-case prefixes of the organism name).
const HOSTS: [(&str, &[&str]); 5] = [
    (
        "mammalian",
        &[
            "homo sapiens",
            "cricetulus",
            "mus musculus",
            "chlorocebus",
            "mesocricetus",
            "rattus",
            "oryctolagus",
            "bos taurus",
            "sus scrofa",
            "macaca",
            "canis",
        ],
    ),
    (
        "insect",
        &[
            "spodoptera",
            "trichoplusia",
            "drosophila",
            "bombyx",
            "mamestra",
            "aedes",
            "unidentified baculovirus",
            "autographa",
            "baculovirus",
        ],
    ),
    (
        "yeast or fungus",
        &[
            "komagataella",
            "pichia",
            "saccharomyces",
            "aspergillus",
            "trichoderma",
            "ogataea",
            "kluyveromyces",
            "schizosaccharomyces",
            "yarrowia",
            "neurospora",
            "thermothelomyces",
            "myceliophthora",
        ],
    ),
    (
        "plant",
        &[
            "nicotiana",
            "arabidopsis",
            "oryza",
            "zea mays",
            "hordeum",
            "solanum",
            "glycine max",
        ],
    ),
    (
        "no N-glycosylation (bacteria, cell-free)",
        &[
            "escherichia",
            "bacillus",
            "cell-free",
            "lactococcus",
            "pseudomonas",
            "corynebacterium",
            "synthetic",
            "brevibacillus",
            "vibrio",
        ],
    ),
];

/// The expression system of a protein from the names of its host organisms ("" when none is
/// recognised): "mammalian", "insect", "yeast or fungus", "plant", or
/// "no N-glycosylation (bacteria, cell-free)".
pub fn host_class(names: &[&str]) -> &'static str {
    for name in names {
        let low = name.to_lowercase();
        for (label, keys) in HOSTS {
            if keys.iter().any(|k| low.starts_with(k)) {
                return label;
            }
        }
    }
    ""
}

/// Glycans to offer for extending the glycan of a site residue (CCD name): for Asn by expression
/// system ([`host_class`]), otherwise the residue's inference candidates.
pub fn suggestions_for(residue: &str, host_class: &str) -> &'static [(&'static str, &'static str)] {
    if residue != "ASN" {
        return candidates_for(residue);
    }
    N_GLYCAN_SUGGESTIONS
        .iter()
        .find(|(host, _)| *host == host_class)
        .map_or(N_GLYCAN_SUGGESTIONS[4].1, |(_, options)| options)
}

/// Whether the glycan `sequence` holds every residue of `deposited` at the same position, so that
/// fitting it extends the deposit instead of contradicting it.
pub fn contains_tree(sequence: &str, deposited: &str) -> Result<bool> {
    let (big, small) = (sequence_paths(sequence)?, sequence_paths(deposited)?);
    Ok(small
        .iter()
        .all(|(path, token)| big.get(path) == Some(token)))
}

/// A deposited N-glycan with its core linkages and its anomers corrected ([`corrected_tree`]).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CorrectedTree {
    pub sequence: String,
    /// residues whose anomer changed: (path, deposited residue, corrected residue)
    pub changed: Vec<(String, String, String)>,
    /// residues of the core linked to another position of their parent: (deposited path,
    /// corrected path, residue)
    pub relinked: Vec<(String, String, String)>,
    /// deposited path -> corrected path of every residue whose path changed (the relinked
    /// residues and the residues they carry)
    pub moved: BTreeMap<String, String>,
}

impl CorrectedTree {
    /// The deposit as it is.
    pub fn unchanged(deposited: &str) -> Self {
        Self {
            sequence: deposited.to_string(),
            ..Self::default()
        }
    }
}

/// The residues that carry the core of every N-glycan (GlcNAc b1-4 GlcNAc b1-4 Man, with its
/// a1-3 and a1-6 arms and the core fucose): the linkages of their children are the same in every
/// N-glycan, so another one is an error of the deposit.
const CORE_PARENTS: [&str; 3] = ["r", "r/4", "r/4/4"];

/// The N-glycans that say what is known of a host: its own for a mammalian host, which makes
/// one core fucose (a1-6); all of them otherwise, since the hosts differ only in what they add.
fn known_glycans(host_class: &str) -> Vec<&'static str> {
    N_GLYCAN_SUGGESTIONS
        .iter()
        .filter(|(host, _)| host_class != "mammalian" || *host == host_class)
        .flat_map(|(_, glycans)| glycans.iter().map(|(_, sequence)| *sequence))
        .collect()
}

fn child_at(seq: &Sequence, k: usize, position: u32) -> Option<usize> {
    seq.nodes[k]
        .children
        .iter()
        .copied()
        .find(|&c| seq.nodes[c].ppos == Some(position))
}

fn node_paths(seq: &Sequence) -> Vec<String> {
    let mut paths = vec![String::new(); seq.nodes.len()];
    let mut stack = vec![(seq.root, "r".to_string())];
    while let Some((k, path)) = stack.pop() {
        for &c in &seq.nodes[k].children {
            stack.push((c, format!("{path}/{}", seq.nodes[c].ppos.unwrap_or(0))));
        }
        paths[k] = path;
    }
    paths
}

/// Link the residues of the core to the position the known N-glycans have them at: a residue
/// at a position of its parent where none of them has it (a GlcNAc b1-6 on the first GlcNAc, a
/// core fucose a1-3 in a mammalian host) moves to the one free position where they do. A residue
/// with no such position, or with two, stays as deposited.
fn relink_core(seq: &mut Sequence, host_class: &str) -> Result<()> {
    let mut known: BTreeMap<(String, String), BTreeSet<u32>> = BTreeMap::new();
    for sequence in known_glycans(host_class) {
        let known_seq = parse_glycam(sequence)?;
        let paths = node_paths(&known_seq);
        for (k, node) in known_seq.nodes.iter().enumerate() {
            for &c in &node.children {
                let child = &known_seq.nodes[c];
                known
                    .entry((paths[k].clone(), child.name.clone()))
                    .or_default()
                    .extend(child.ppos);
            }
        }
    }
    let mut parent = Some(seq.root);
    for path in CORE_PARENTS {
        let Some(k) = parent else { break };
        let children = seq.nodes[k].children.clone();
        let positions = |c: usize| known.get(&(path.to_string(), seq.nodes[c].name.clone()));
        let placed = |c: usize| {
            positions(c).is_some_and(|q| seq.nodes[c].ppos.is_some_and(|p| q.contains(&p)))
        };
        let taken: BTreeSet<u32> = children
            .iter()
            .filter(|&&c| placed(c))
            .filter_map(|&c| seq.nodes[c].ppos)
            .collect();
        // a misplaced residue with one free position to go to
        let mut moves: BTreeMap<usize, u32> = children
            .iter()
            .filter(|&&c| !placed(c))
            .filter_map(|&c| {
                let free: Vec<u32> = positions(c)?.difference(&taken).copied().collect();
                (free.len() == 1).then(|| (c, free[0]))
            })
            .collect();
        // which no residue that stays holds, and no other one goes to
        loop {
            let blocked: Vec<usize> = moves
                .iter()
                .filter(|&(&c, &to)| {
                    children.iter().any(|&o| {
                        o != c
                            && match moves.get(&o) {
                                Some(&other) => other == to,
                                None => seq.nodes[o].ppos == Some(to),
                            }
                    })
                })
                .map(|(&c, _)| c)
                .collect();
            if blocked.is_empty() {
                break;
            }
            for c in blocked {
                moves.remove(&c);
            }
        }
        for (c, to) in moves {
            seq.nodes[c].ppos = Some(to);
        }
        parent = child_at(seq, k, 4);
    }
    Ok(())
}

/// What a residue of the core of an N-glycan has that no known N-glycan has ([`core_faults`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreFault {
    /// linked to a position of its parent where the known glycans do not have it
    Linkage,
    /// another residue than the one every N-glycan has at this position
    Residue,
}

/// The residues of the core of an N-glycan (one that starts with GlcNAc) that no known N-glycan
/// has: what [`corrected_tree`] could not correct, e.g. a third GlcNAc in place of the
/// b-mannose, or an arm on O2 of it with both O3 and O6 free.
pub fn core_faults(sequence: &str, host_class: &str) -> Result<Vec<(String, CoreFault)>> {
    let mut positions: BTreeMap<(String, String), BTreeSet<u32>> = BTreeMap::new();
    let mut tokens: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for known in known_glycans(host_class) {
        for (path, (token, _)) in sequence_paths(known)? {
            if let Some((parent, position)) = path.rsplit_once('/') {
                positions
                    .entry((parent.to_string(), token.clone()))
                    .or_default()
                    .extend(position.parse::<u32>());
            }
            tokens.entry(path).or_default().insert(token);
        }
    }
    let residues = sequence_paths(sequence)?;
    if residues.get("r").map(|r| r.0.as_str()) != Some("DGlcpNAc") {
        return Ok(Vec::new());
    }
    let mut faults = Vec::new();
    for (path, (token, _)) in &residues {
        let Some((parent, position)) = path.rsplit_once('/') else {
            continue;
        };
        if !CORE_PARENTS.contains(&parent) {
            continue;
        }
        let here = position.parse::<u32>().ok();
        if let Some(known) = positions.get(&(parent.to_string(), token.clone())) {
            if !here.is_some_and(|p| known.contains(&p)) {
                faults.push((path.clone(), CoreFault::Linkage));
            }
        } else if tokens.get(path).is_some_and(|t| !t.contains(token)) {
            faults.push((path.clone(), CoreFault::Residue));
        }
    }
    Ok(faults)
}

/// The deposited N-glycan as the known N-glycans have it: the residues of the core at their
/// linkage position ([`relink_core`]), and every residue with the anomer the known glycans have at
/// its position (a core Man a1-4, a Fuc b1-6, a GlcNAc a1-4 takes the one they all have there).
/// Such a linkage or anomer is an error of the deposit, not a different glycan: the enzymes that
/// build N-glycans make one of each at these positions. Residues and positions the known glycans
/// do not have are left as deposited.
pub fn corrected_tree(deposited: &str, host_class: &str) -> Result<CorrectedTree> {
    let mut known: BTreeMap<(String, String), BTreeSet<char>> = BTreeMap::new();
    for (_, glycans) in N_GLYCAN_SUGGESTIONS {
        for (_, sequence) in glycans.iter() {
            for (path, (token, anomer)) in sequence_paths(sequence)? {
                known.entry((path, token)).or_default().insert(anomer);
            }
        }
    }
    let mut seq = parse_glycam(deposited)?;
    let before = node_paths(&seq);
    relink_core(&mut seq, host_class)?;
    let after = node_paths(&seq);
    let moved: BTreeMap<String, String> = before
        .iter()
        .zip(&after)
        .filter(|(a, b)| a != b)
        .map(|(a, b)| (a.clone(), b.clone()))
        .collect();
    let mut relinked: Vec<(String, String, String)> = (0..seq.nodes.len())
        .filter(|&k| {
            // its own position changed, not only that of a residue below it
            before[k] != after[k] && before[k].rsplit('/').next() != after[k].rsplit('/').next()
        })
        .map(|k| {
            let node = &seq.nodes[k];
            (
                before[k].clone(),
                after[k].clone(),
                format!("{}{}", node.name, node.anomer),
            )
        })
        .collect();
    relinked.sort();
    let (mut changed, mut keep) = (Vec::new(), BTreeSet::new());
    for (k, path) in after.iter().enumerate() {
        let node = &mut seq.nodes[k];
        if let Some(anomers) = known.get(&(path.clone(), node.name.clone()))
            && anomers.len() == 1
            && !anomers.contains(&node.anomer)
        {
            let anomer = *anomers.first().expect("one anomer");
            changed.push((
                path.clone(),
                format!("{}{}", node.name, node.anomer),
                format!("{}{anomer}", node.name),
            ));
            node.anomer = anomer;
        }
        keep.insert(path.clone());
    }
    changed.sort();
    let sequence = if changed.is_empty() && moved.is_empty() {
        deposited.to_string()
    } else {
        let tail = if seq.aglycone == "ROH" { "-OH" } else { "-OME" };
        write_node(&seq, seq.root, "r", tail, &keep, &mut BTreeMap::new())
    };
    Ok(CorrectedTree {
        sequence,
        changed,
        relinked,
        moved,
    })
}

fn kept_children(
    seq: &Sequence,
    k: usize,
    path: &str,
    keep: &BTreeSet<String>,
) -> Vec<(usize, String)> {
    seq.nodes[k]
        .children
        .iter()
        .map(|&c| (c, format!("{path}/{}", seq.nodes[c].ppos.unwrap_or(0))))
        .filter(|(_, p)| keep.contains(p))
        .collect()
}

fn subtree_size(seq: &Sequence, k: usize, path: &str, keep: &BTreeSet<String>) -> usize {
    1 + kept_children(seq, k, path, keep)
        .iter()
        .map(|(c, p)| subtree_size(seq, *c, p, keep))
        .sum::<usize>()
}

fn write_node(
    seq: &Sequence,
    k: usize,
    path: &str,
    link: &str,
    keep: &BTreeSet<String>,
    tokens: &mut BTreeMap<String, String>,
) -> String {
    let node = &seq.nodes[k];
    tokens.insert(path.to_string(), format!("{}{}", node.name, node.anomer));
    // main chain last: the largest kept subtree, ties to the higher linkage position
    let mut kids = kept_children(seq, k, path, keep);
    kids.sort_by_key(|(c, p)| (subtree_size(seq, *c, p, keep), seq.nodes[*c].ppos));
    let mut prefix = String::new();
    if let Some(((m, mp), rest)) = kids.split_last() {
        let ml = format!("-{}", seq.nodes[*m].ppos.unwrap_or(0));
        prefix = write_node(seq, *m, mp, &ml, keep, tokens);
        for (c, p) in rest {
            let l = format!("-{}", seq.nodes[*c].ppos.unwrap_or(0));
            prefix.push_str(&format!("[{}]", write_node(seq, *c, p, &l, keep, tokens)));
        }
    }
    let mods = if node.mods.is_empty() {
        String::new()
    } else {
        format!("[{}]", node.mods.join(","))
    };
    format!(
        "{prefix}{}{mods}{}{}{link}",
        node.name, node.anomer, node.cpos
    )
}

/// GLYCAM sequence of the subtree of `sequence` whose residue paths are in `keep` (connected to
/// the root), and {path: residue token + anomer}.
pub fn pruned_sequence(
    sequence: &str,
    keep: &BTreeSet<String>,
) -> Result<(String, BTreeMap<String, String>)> {
    let seq = parse_glycam(sequence)?;
    let mut tokens = BTreeMap::new();
    let tail = if seq.aglycone == "ROH" { "-OH" } else { "-OME" };
    let s = write_node(&seq, seq.root, "r", tail, keep, &mut tokens);
    Ok((s, tokens))
}

/// Minimum density of a built residue: fraction of the way from the bulk-solvent to the protein level.
pub const DENSITY_GATE: f64 = 0.3;

/// Map levels around a site (standard deviations of the map): protein = environment atoms within
/// 12 A of the link atom, solvent = decoy centres in the scoring ball > 3.5 A from the environment.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DensityLevels {
    pub protein: f64,
    pub solvent: f64,
    #[serde(skip)]
    mean: f64,
    #[serde(skip)]
    sd: f64,
    #[serde(skip)]
    periodic: bool,
}

impl DensityLevels {
    pub fn new(map: &DensityMap, problem: &SiteProblem, environment: &[V3]) -> Option<Self> {
        let v = map.values();
        let n = v.len() as f64;
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
        let sd = (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
        let mut levels = Self {
            protein: 0.0,
            solvent: 0.0,
            mean,
            sd,
            periodic: map.is_full_unit_cell(),
        };
        let d = |a: V3, b: V3| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
        let link = problem.anchor[2];
        let protein: Vec<V3> = environment
            .iter()
            .copied()
            .filter(|p| d(*p, link) < 12.0)
            .collect();
        let solvent: Vec<V3> = decoy_poses(2048)
            .into_iter()
            .map(|(dir, u, _)| [0, 1, 2].map(|k| link[k] + dir[k] * (problem.radius - 3.0) * u))
            .filter(|p| environment.iter().all(|e| d(*p, *e) > 3.5))
            .collect();
        levels.protein = levels.mean_at(map, &protein)?;
        levels.solvent = levels.mean_at(map, &solvent)?;
        (levels.protein > levels.solvent).then_some(levels)
    }

    fn mean_at(&self, map: &DensityMap, pts: &[V3]) -> Option<f64> {
        let vals: Vec<f64> = pts
            .iter()
            .filter_map(|p| map.value_at_cartesian(*p, self.periodic))
            .map(|x| (x - self.mean) / self.sd)
            .collect();
        (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// Density over `pts` as a fraction of the way from the solvent to the protein level.
    pub fn fraction(&self, map: &DensityMap, pts: &[V3]) -> Option<f64> {
        self.mean_at(map, pts)
            .map(|v| (v - self.solvent) / (self.protein - self.solvent))
    }
}

/// One candidate fit, pruned to its density-supported tree.
pub struct CandidateFit {
    pub name: String,
    pub sequence: String,
    pub fit: SiteFit,
    /// built residue paths (root first): statistically supported and at or above `DENSITY_GATE`,
    /// parent built; empty: no glycan built
    pub supported: Vec<String>,
    /// statistically supported but below the density gate (or below a weak parent): not built
    pub weak: Vec<String>,
    /// density fraction of every statistically supported residue
    pub density_fraction: BTreeMap<String, f64>,
    pub root_gain: f64,
    /// penalised log-likelihood gain of the supported tree (0 when nothing is supported)
    pub score: f64,
    pub pruned: Option<String>,
    pub tokens: BTreeMap<String, String>,
}

fn loglik_of(problem: &SiteProblem, x: &[V3], paths: &BTreeSet<String>) -> f64 {
    let active: Vec<bool> = problem
        .glycan
        .res_paths
        .iter()
        .zip(&problem.keep)
        .map(|(p, k)| *k && paths.contains(p))
        .collect();
    problem
        .observation
        .evaluate(x, Some(&active), None)
        .log_likelihood
}

/// Log-likelihood the residues in `paths` explain over the protein-only model.
pub fn tree_gain(problem: &SiteProblem, x: &[V3], paths: &BTreeSet<String>) -> f64 {
    loglik_of(problem, x, paths) - loglik_of(problem, x, &BTreeSet::new())
}

/// Torsions inside the kept tree (both bond atoms kept) plus the two attachment torsions.
pub fn tree_torsions(problem: &SiteProblem, paths: &BTreeSet<String>) -> usize {
    let rp = &problem.glycan.res_paths;
    problem
        .glycan
        .topology
        .quads
        .iter()
        .filter(|q| paths.contains(&rp[q[1]]) && paths.contains(&rp[q[2]]))
        .count()
        + 2
}

/// A candidate fit and its statistically supported residues (before the density gate).
struct StatFit {
    name: String,
    sequence: String,
    fit: SiteFit,
    root_gain: f64,
    statistical: Vec<String>,
}

fn fit_candidate(
    input: &WorkflowInput,
    residue: &ResidueId,
    protein: &Structure,
    name: &str,
    sequence: &str,
) -> Result<StatFit> {
    let request = SiteRequest {
        residue: residue.clone(),
        sequence: Some(sequence.to_string()),
    };
    let fit = fit_one(input, &request, protein)?;
    let (base, per) = (
        input.options.fit.support_base,
        input.options.fit.support_per_torsion,
    );
    let (statistical, root_gain) = statistical_support(&fit.problem, &fit.outcome, base, per);
    Ok(StatFit {
        name: name.to_string(),
        sequence: sequence.to_string(),
        fit,
        root_gain,
        statistical,
    })
}

/// The residues of a fitted pose that pass the support test (the root against no glycan), root
/// first, and the root's gain.
pub fn statistical_support(
    problem: &SiteProblem,
    outcome: &FitOutcome,
    base: f64,
    per: f64,
) -> (Vec<String>, f64) {
    let x = &outcome.basins[outcome.best].x;
    let root: BTreeSet<String> = ["r".to_string()].into();
    let root_gain = tree_gain(problem, x, &root);
    let root_ok = root_gain > base + per * tree_torsions(problem, &root) as f64;
    let statistical: Vec<String> = if root_ok {
        std::iter::once("r".to_string())
            .chain(
                outcome
                    .support
                    .iter()
                    .filter(|s| s.supported)
                    .map(|s| s.residue.clone()),
            )
            .collect()
    } else {
        Vec::new()
    };
    (statistical, root_gain)
}

/// What a fit builds (`infer.gate`): of the residues that pass the support test, those whose
/// density is at least [`DENSITY_GATE`] of the way from solvent to protein and whose parent is
/// built; the others are `weak`. Without levels every supported residue is built.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Gate {
    pub built: Vec<String>,
    pub weak: Vec<String>,
    /// density fraction of every residue that passes the support test
    pub density_fraction: BTreeMap<String, f64>,
    pub root_gain: f64,
}

/// Apply the density gate to `statistical` (in depth order: parents before children).
pub fn density_gate(
    problem: &SiteProblem,
    x: &[V3],
    map: &DensityMap,
    levels: Option<&DensityLevels>,
    statistical: &[String],
    root_gain: f64,
) -> Gate {
    let mut gate = Gate {
        root_gain,
        ..Gate::default()
    };
    for path in statistical {
        let pts: Vec<V3> = (0..problem.n_atoms)
            .filter(|&i| problem.keep[i] && &problem.glycan.res_paths[i] == path)
            .map(|i| x[i])
            .collect();
        let frac = levels.and_then(|l| l.fraction(map, &pts));
        if let Some(f) = frac {
            gate.density_fraction.insert(path.clone(), f);
        }
        let parent_built = path == "r"
            || path
                .rsplit_once('/')
                .is_some_and(|(parent, _)| gate.built.iter().any(|b| b == parent));
        if parent_built && frac.is_none_or(|f| f >= DENSITY_GATE) {
            gate.built.push(path.clone());
        } else {
            gate.weak.push(path.clone());
        }
    }
    gate
}

/// Apply the density gate (in depth order: parents before children) and score the built tree.
fn gate(
    input: &WorkflowInput,
    levels: Option<&DensityLevels>,
    stat: StatFit,
) -> Result<CandidateFit> {
    let (base, per) = (
        input.options.fit.support_base,
        input.options.fit.support_per_torsion,
    );
    let StatFit {
        name,
        sequence,
        fit,
        root_gain,
        statistical,
    } = stat;
    let problem = &fit.problem;
    let x = &fit.outcome.basins[fit.outcome.best].x;
    let Gate {
        built,
        weak,
        density_fraction,
        ..
    } = density_gate(problem, x, input.map, levels, &statistical, root_gain);
    let keep: BTreeSet<String> = built.iter().cloned().collect();
    let (score, pruned, tokens) = if keep.is_empty() {
        (0.0, None, BTreeMap::new())
    } else {
        let score =
            tree_gain(problem, x, &keep) - base - per * tree_torsions(problem, &keep) as f64;
        let (p, t) = pruned_sequence(&sequence, &keep)?;
        (score, Some(p), t)
    };
    Ok(CandidateFit {
        name,
        sequence,
        fit,
        supported: built,
        weak,
        density_fraction,
        root_gain,
        score,
        pruned,
        tokens,
    })
}

pub struct Inference {
    /// first-round fits, then refits of distinct pruned trees
    pub candidates: Vec<CandidateFit>,
    /// map levels behind the density gate (None: no density map)
    pub levels: Option<DensityLevels>,
    pub best: usize,
    /// shared scoring-ball radius (A)
    pub region_radius: f64,
    pub sigma: f64,
}

impl Inference {
    pub fn chosen(&self) -> &CandidateFit {
        &self.candidates[self.best]
    }
    /// False when no candidate's root residue is supported (no glycan density at the site).
    pub fn glycosylated(&self) -> bool {
        !self.chosen().supported.is_empty()
    }
}

/// Ranking of candidates: penalised density gain, then the smaller supported tree.
fn ranks_below(a: &CandidateFit, b: &CandidateFit) -> bool {
    (a.score, -(a.supported.len() as i64))
        .partial_cmp(&(b.score, -(b.supported.len() as i64)))
        .unwrap_or(std::cmp::Ordering::Equal)
        != std::cmp::Ordering::Greater
}

/// Adds a gated candidate and frees the maps of the one that is no longer the best: every
/// candidate holds a full site problem (density box, clash grids; ~100 MB for an N-glycan's
/// scoring region), and only the chosen one is evaluated again. Ties go to the later candidate.
fn keep_best(fits: &mut Vec<CandidateFit>, best: &mut usize, candidate: CandidateFit) {
    fits.push(candidate);
    let new = fits.len() - 1;
    if new == 0 {
        return;
    }
    let loser = if ranks_below(&fits[*best], &fits[new]) {
        std::mem::replace(best, new)
    } else {
        new
    };
    release_maps(&mut fits[loser].fit.problem);
}

/// The observation of a candidate that lost the inference: its data were freed.
struct Released;

impl Observation for Released {
    fn kind(&self) -> &'static str {
        "released"
    }
    fn prepare(&mut self, _atoms: &ObservedAtoms) -> Result<()> {
        Ok(())
    }
    fn evaluate(
        &self,
        _x: &[[f64; 3]],
        _active: Option<&[bool]>,
        _grad: Option<&mut [[f64; 3]]>,
    ) -> ObservationValue {
        panic!("a candidate that lost the inference was evaluated after its maps were freed")
    }
}

fn release_maps(problem: &mut SiteProblem) {
    problem.observation = Box::new(Released);
    problem.grids.carbon = Vec::new();
    problem.grids.polar = Vec::new();
}

/// Infer the glycan at `residue` from the density. `protein` is the model without any target
/// glycan; `input.sites` is ignored.
pub fn glycan_from_density(
    input: &WorkflowInput,
    residue: &ResidueId,
    protein: &Structure,
    candidates: &[(String, String)],
    refit: bool,
) -> Result<Inference> {
    let mut options = input.options.clone();
    // one scoring region for every candidate: the reach of the largest
    let mut reach: f64 = 0.0;
    for (_, s) in candidates {
        let g = build_glycan(
            &input.model.library,
            s,
            options.problem.n_templates,
            options.problem.template_seed,
        )?;
        reach = reach.max(max_span(&g));
    }
    let region_radius = options.problem.region_radius.unwrap_or(reach + 2.5);
    options.problem.region_radius = Some(region_radius);
    let mut fits: Vec<CandidateFit> = Vec::new();
    let mut best = 0;
    let sub = |options: &crate::workflow::WorkflowOptions| WorkflowInput {
        structure: input.structure,
        structure_text: input.structure_text,
        map: input.map,
        sites: Vec::new(),
        model: input.model,
        options: options.clone(),
    };
    // density levels of the site (candidate independent): from the first fit's site
    let mut levels: Option<DensityLevels> = None;
    // Root check: the candidates share their reducing-end residue. When the map does not support
    // even that residue fitted on its own, the site carries no glycan density and the large fits
    // are skipped (an empty sequon costs one small fit instead of one per candidate). The check
    // leaves the atom width to the first candidate, so a glycosylated site's inference is unchanged.
    let roots = candidates
        .iter()
        .map(|(_, s)| pruned_sequence(s, &BTreeSet::from(["r".to_string()])).map(|(r, _)| r))
        .collect::<Result<BTreeSet<String>>>()?;
    if !input.options.infer_without_root_check && roots.len() == 1 {
        let root = roots.into_iter().next().unwrap_or_default();
        let name = "reducing-end residue alone (root check)";
        if let Some(o) = options.observer.get() {
            o.stage(&format!("candidate {name}"));
        }
        let stat = fit_candidate(&sub(&options), residue, protein, name, &root)?;
        let env: Vec<V3> = stat
            .fit
            .site
            .environment
            .iter()
            .map(|a| a.position)
            .collect();
        levels = DensityLevels::new(input.map, &stat.fit.problem, &env);
        let check = gate(&sub(&options), levels.as_ref(), stat)?;
        if check.supported.is_empty() {
            return Ok(Inference {
                levels,
                sigma: check.fit.sigma.selected,
                candidates: vec![check],
                best: 0,
                region_radius,
            });
        }
    }
    for (name, seq) in candidates {
        if let Some(o) = options.observer.get() {
            o.stage(&format!("candidate {name}"));
        }
        let stat = fit_candidate(&sub(&options), residue, protein, name, seq)?;
        // the atom width is candidate independent: calibrate once
        options.sigma.get_or_insert(stat.fit.sigma.selected);
        if levels.is_none() {
            let env: Vec<V3> = stat
                .fit
                .site
                .environment
                .iter()
                .map(|a| a.position)
                .collect();
            levels = DensityLevels::new(input.map, &stat.fit.problem, &env);
        }
        keep_best(
            &mut fits,
            &mut best,
            gate(&sub(&options), levels.as_ref(), stat)?,
        );
    }
    let run = |options: &crate::workflow::WorkflowOptions, name: &str, seq: &str| {
        if let Some(o) = options.observer.get() {
            o.stage(&format!("candidate {name}"));
        }
        let stat = fit_candidate(&sub(options), residue, protein, name, seq)?;
        gate(&sub(options), levels.as_ref(), stat)
    };
    if refit {
        let mut seen: BTreeSet<String> = fits.iter().map(|f| f.sequence.clone()).collect();
        let n = fits.len();
        for k in 0..n {
            let Some(p) = fits[k].pruned.clone() else {
                continue;
            };
            if !seen.insert(p.clone()) {
                continue;
            }
            let name = format!("{} (pruned, refitted)", fits[k].name);
            let f = run(&options, &name, &p)?;
            keep_best(&mut fits, &mut best, f);
        }
    }
    Ok(Inference {
        levels,
        sigma: options.sigma.unwrap_or(f64::NAN),
        candidates: fits,
        best,
        region_radius,
    })
}

/// Residue-level agreement of a pruned tree with the deposited tree (same path scheme):
/// precision and recall over (path, residue token + anomer).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Composition {
    pub predicted: usize,
    pub deposited: usize,
    pub matched: usize,
    pub precision: f64,
    pub recall: f64,
    pub extra: Vec<String>,
    pub missing: Vec<String>,
}

pub fn composition(tokens: &BTreeMap<String, String>, deposited: &DepositedGlycan) -> Composition {
    let dep: BTreeMap<String, String> = deposited
        .residues
        .iter()
        .filter_map(|r| glycam_of(&r.name).map(|(t, a, _)| (r.path.clone(), format!("{t}{a}"))))
        .collect();
    let matched = tokens
        .iter()
        .filter(|(p, t)| dep.get(*p) == Some(t))
        .count();
    Composition {
        predicted: tokens.len(),
        deposited: deposited.residues.len(),
        matched,
        precision: matched as f64 / tokens.len().max(1) as f64,
        recall: matched as f64 / deposited.residues.len().max(1) as f64,
        extra: tokens
            .iter()
            .filter(|(p, t)| dep.get(*p) != Some(t))
            .map(|(p, _)| p.clone())
            .collect(),
        missing: dep
            .iter()
            .filter(|(p, t)| tokens.get(*p) != Some(t))
            .map(|(p, _)| p.clone())
            .collect(),
    }
}

/// In-place RMSD to the deposited glycan over the residues in `paths` (matched by path and atom
/// name; residues of a different type still match on shared atom names).
pub fn rmsd_over(
    problem: &SiteProblem,
    x: &[V3],
    deposited: &DepositedGlycan,
    paths: &BTreeSet<String>,
) -> Option<f64> {
    let (mut s, mut n) = (0.0, 0usize);
    for (i, p) in problem.glycan.res_paths.iter().enumerate() {
        if !problem.keep[i] || !paths.contains(p) {
            continue;
        }
        if let Some(d) = deposited
            .atoms
            .get(&(p.clone(), problem.glycan.atom_names[i].clone()))
        {
            s += (0..3).map(|k| (x[i][k] - d[k]).powi(2)).sum::<f64>();
            n += 1;
        }
    }
    (n > 0).then(|| (s / n as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruned_sequence_keeps_the_connected_subtree() {
        let hybrid = N_GLYCAN_CANDIDATES[3].1;
        let keep: BTreeSet<String> = ["r", "r/4", "r/4/4", "r/4/4/3", "r/4/4/6"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (s, tokens) = pruned_sequence(hybrid, &keep).unwrap();
        assert_eq!(s, "DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH");
        assert_eq!(tokens.keys().cloned().collect::<BTreeSet<_>>(), keep);
        assert_eq!(tokens["r/4/4"], "DManpb");
    }

    #[test]
    #[ignore = "needs GLYCOFLOW_MODEL (residue_library.json of the licensed model)"]
    fn every_candidate_builds() {
        let path = std::path::Path::new(&std::env::var("GLYCOFLOW_MODEL").unwrap())
            .join("residue_library.json");
        let library =
            glycoflow_core::ResidueLibrary::from_json_slice(&std::fs::read(path).unwrap()).unwrap();
        for a in &crate::anchor::ANCHORS {
            for (name, seq) in candidates_for(a.residue) {
                let built = library
                    .build(seq, None)
                    .unwrap_or_else(|e| panic!("{name} ({seq}): {e}"));
                assert_eq!(built.res_paths[0], "agl", "{name}");
            }
        }
    }

    #[test]
    fn suggestions_extend_the_deposit() {
        for (host, options) in N_GLYCAN_SUGGESTIONS {
            for (name, seq) in options {
                assert!(
                    contains_tree(seq, "DGlcpNAcb1-4DGlcpNAcb1-OH").unwrap(),
                    "{host}: {name}"
                );
            }
        }
        assert!(
            suggestions_for("ASN", "insect")[0]
                .0
                .starts_with("paucimannose")
        );
        assert_eq!(
            suggestions_for("ASN", "something else"),
            N_GLYCAN_SUGGESTIONS[4].1
        );
        assert_eq!(suggestions_for("TRP", "mammalian")[0].0, "C-mannose");
        let fucosylated = "LFucpa1-6[DGlcpNAcb1-4]DGlcpNAcb1-OH";
        assert!(!contains_tree(MAN9, fucosylated).unwrap()); // no core fucose in Man9
        assert!(contains_tree(COMPLEX_FUC, fucosylated).unwrap());
        // an alpha-mannose where the core has beta
        assert!(!contains_tree(MAN9, "DManpa1-4DGlcpNAcb1-4DGlcpNAcb1-OH").unwrap());
        assert_eq!(host_class(&["Spodoptera frugiperda"]), "insect");
        assert_eq!(host_class(&["Homo sapiens"]), "mammalian");
        assert_eq!(host_class(&["Komagataella pastoris"]), "yeast or fungus");
        assert_eq!(
            host_class(&["Escherichia coli BL21(DE3)"]),
            "no N-glycosylation (bacteria, cell-free)"
        );
        assert_eq!(host_class(&[]), "");
        // the candidates of sequence-free inference are among the suggestions
        for (name, seq) in N_GLYCAN_CANDIDATES {
            assert!(
                N_GLYCAN_SUGGESTIONS[0].1.iter().any(|(_, s)| *s == seq),
                "{name}"
            );
        }
    }

    #[test]
    fn every_anchor_has_candidates() {
        for a in &crate::anchor::ANCHORS {
            assert!(!candidates_for(a.residue).is_empty(), "{}", a.residue);
        }
        assert!(candidates_for("ALA").is_empty());
    }

    #[test]
    fn pruned_sequence_round_trips_full_trees() {
        let all = crate::anchor::ANCHORS
            .iter()
            .flat_map(|a| candidates_for(a.residue).iter());
        for (_, seq) in all {
            let parsed = parse_glycam(seq).unwrap();
            let mut keep = BTreeSet::new();
            let mut stack = vec![(parsed.root, "r".to_string())];
            while let Some((k, p)) = stack.pop() {
                for &c in &parsed.nodes[k].children {
                    stack.push((c, format!("{p}/{}", parsed.nodes[c].ppos.unwrap())));
                }
                keep.insert(p);
            }
            assert_eq!(pruned_sequence(seq, &keep).unwrap().0, *seq);
        }
    }

    #[test]
    fn a_core_linkage_no_n_glycan_has_is_corrected() {
        // 8ZJ4 D:909: the two core GlcNAc joined b1-6
        let t = corrected_tree("DManpb1-4DGlcpNAcb1-6DGlcpNAcb1-OH", "mammalian").unwrap();
        assert_eq!(t.sequence, "DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH");
        assert_eq!(
            t.relinked,
            vec![(
                "r/6".to_string(),
                "r/4".to_string(),
                "DGlcpNAcb".to_string()
            )]
        );
        assert_eq!(t.moved["r/6"], "r/4");
        assert_eq!(t.moved["r/6/4"], "r/4/4");
        assert!(t.changed.is_empty());
        // the b-mannose on O6 of the second GlcNAc, and an arm on O2 of it beside the a1-3 arm
        let t =
            corrected_tree("DManpa1-2[DManpa1-3]DManpb1-6DGlcpNAcb1-4DGlcpNAcb1-OH", "").unwrap();
        assert_eq!(t.sequence, CORE);
        assert_eq!(t.relinked.len(), 2);
        assert_eq!(t.moved["r/4/6/2"], "r/4/4/6");
        assert_eq!(t.moved["r/4/6/3"], "r/4/4/3");
        // the linkage and the anomer of one residue
        let t = corrected_tree("DGlcpNAca1-6DGlcpNAcb1-OH", "insect").unwrap();
        assert_eq!(t.sequence, "DGlcpNAcb1-4DGlcpNAcb1-OH");
        assert_eq!(
            t.changed,
            vec![(
                "r/4".to_string(),
                "DGlcpNAca".to_string(),
                "DGlcpNAcb".to_string()
            )]
        );
    }

    #[test]
    fn a_core_fucose_goes_to_o6_in_a_mammalian_host_only() {
        let deposit = "DGlcpNAcb1-4[LFucpa1-3]DGlcpNAcb1-OH";
        let t = corrected_tree(deposit, "mammalian").unwrap();
        assert_eq!(
            sequence_paths(&t.sequence).unwrap(),
            sequence_paths("DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH").unwrap()
        );
        assert_eq!(
            t.relinked,
            vec![("r/3".to_string(), "r/6".to_string(), "LFucpa".to_string())]
        );
        for host in ["insect", "plant", "yeast or fungus", ""] {
            let t = corrected_tree(deposit, host).unwrap();
            assert_eq!(t.sequence, deposit, "{host}");
            assert!(t.relinked.is_empty() && t.moved.is_empty());
        }
        // both fucoses deposited: O6 is taken, the a1-3 one stays
        let both = "DGlcpNAcb1-4[LFucpa1-3][LFucpa1-6]DGlcpNAcb1-OH";
        assert!(
            corrected_tree(both, "mammalian")
                .unwrap()
                .relinked
                .is_empty()
        );
    }

    #[test]
    fn what_the_known_glycans_do_not_say_stays_as_deposited() {
        // every known glycan itself
        for (host, glycans) in N_GLYCAN_SUGGESTIONS {
            for (_, sequence) in glycans.iter() {
                let t = corrected_tree(sequence, host).unwrap();
                assert_eq!(t.sequence, *sequence);
                assert!(t.changed.is_empty() && t.moved.is_empty());
            }
        }
        // a bisecting GlcNAc, a third GlcNAc in place of the mannose, an arm with two free
        // positions, a residue beyond the core, and two residues that would go to one position
        for deposit in [
            "DManpa1-6[DGlcpNAcb1-4][DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
            "DGlcpNAcb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
            "DManpa1-2DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
            "DManpa1-3DManpa1-3DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
            "DGlcpNAcb1-6[DGlcpNAcb1-3]DGlcpNAcb1-OH",
        ] {
            let t = corrected_tree(deposit, "mammalian").unwrap();
            assert!(t.moved.is_empty(), "{deposit} -> {}", t.sequence);
        }
        // a swap: the GlcNAc on O6 and the fucose on O4 exchange in a mammalian host, and stay
        // where the fucose has two positions to go to
        let swap = "DGlcpNAcb1-6[LFucpa1-4]DGlcpNAcb1-OH";
        let t = corrected_tree(swap, "mammalian").unwrap();
        assert_eq!(
            sequence_paths(&t.sequence).unwrap(),
            sequence_paths("DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH").unwrap()
        );
        assert!(corrected_tree(swap, "").unwrap().moved.is_empty());
    }

    #[test]
    fn core_faults_are_what_the_correction_leaves() {
        let faults = |deposit: &str, host: &str| {
            let t = corrected_tree(deposit, host).unwrap();
            core_faults(&t.sequence, host).unwrap()
        };
        for (host, glycans) in N_GLYCAN_SUGGESTIONS {
            for (_, sequence) in glycans.iter() {
                assert!(faults(sequence, host).is_empty(), "{sequence}");
            }
        }
        // corrected: nothing left
        assert!(faults("DManpb1-4DGlcpNAcb1-6DGlcpNAcb1-OH", "mammalian").is_empty());
        // 5Z65 A:82: a third GlcNAc where every N-glycan has the b-mannose
        assert_eq!(
            faults("DGlcpNAcb1-4DGlcpNAcb1-4DGlcpNAcb1-OH", "mammalian"),
            vec![("r/4/4".to_string(), CoreFault::Residue)]
        );
        // an arm on O2 with two positions to go to, and a second fucose in a mammalian host
        assert_eq!(
            faults("DManpa1-2DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH", ""),
            vec![("r/4/4/2".to_string(), CoreFault::Linkage)]
        );
        assert_eq!(
            faults(
                "DGlcpNAcb1-4[LFucpa1-3][LFucpa1-6]DGlcpNAcb1-OH",
                "mammalian"
            ),
            vec![("r/3".to_string(), CoreFault::Linkage)]
        );
        // a bisecting GlcNAc and a plant xylose are no faults, nor is a glycan that is not an
        // N-glycan of this kind
        for deposit in [
            "DManpa1-6[DGlcpNAcb1-4][DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
            "DManpa1-6[DXylpb1-2][DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH",
            "DGalpb1-3DGlcpb1-OH",
        ] {
            assert!(faults(deposit, "mammalian").is_empty(), "{deposit}");
        }
    }
}
