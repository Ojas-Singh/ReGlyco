//! A glycosylation site in a model (`glycoflow/fitting/site.py`): protein anchor, the deposited
//! glycan (kept apart: it names the fitted residues and is used for evaluation only, never to
//! propose or rank), the GLYCAM sequence, and the environment with symmetry mates.

use std::collections::{BTreeMap, BTreeSet};

use glysys::{ResidueId, Structure};

use crate::error::{Result, invalid};
use crate::symmetry::{EnvAtom, SpaceGroup, UnitCell, expand_environment};

/// CCD component -> (GLYCAM token, anomer, anomeric carbon number).
pub const CCD_TO_GLYCAM: &[(&str, &str, char, u32)] = &[
    ("NAG", "DGlcpNAc", 'b', 1),
    ("NDG", "DGlcpNAc", 'a', 1),
    ("BMA", "DManp", 'b', 1),
    ("MAN", "DManp", 'a', 1),
    ("GAL", "DGalp", 'b', 1),
    ("GLA", "DGalp", 'a', 1),
    ("BGC", "DGlcp", 'b', 1),
    ("GLC", "DGlcp", 'a', 1),
    ("NGA", "DGalpNAc", 'b', 1),
    ("A2G", "DGalpNAc", 'a', 1),
    ("FUC", "LFucp", 'a', 1),
    ("FUL", "LFucp", 'b', 1),
    ("XYP", "DXylp", 'b', 1),
    ("XYS", "DXylp", 'a', 1),
    ("SIA", "DNeup5Ac", 'a', 2),
    ("SLB", "DNeup5Ac", 'b', 2),
];

/// PDB (CCD) atom name -> GLYCAM atom name where they differ (N-acetyl groups).
pub fn atom_rename(ccd: &str) -> &'static [(&'static str, &'static str)] {
    match ccd {
        "NAG" | "NDG" | "NGA" | "A2G" => &[("C7", "C2N"), ("C8", "CME"), ("O7", "O2N")],
        "SIA" | "SLB" => &[("C10", "C5N"), ("C11", "CME"), ("O10", "O5N")],
        _ => &[],
    }
}

pub fn glycam_of(ccd: &str) -> Option<(&'static str, char, u32)> {
    CCD_TO_GLYCAM
        .iter()
        .find(|e| e.0 == ccd)
        .map(|e| (e.1, e.2, e.3))
}

/// CCD component of a GLYCAM token and anomer (first match of [`CCD_TO_GLYCAM`]).
pub fn ccd_of(token: &str, anomer: char) -> Option<&'static str> {
    CCD_TO_GLYCAM
        .iter()
        .find(|e| e.1 == token && e.2 == anomer)
        .map(|e| e.0)
}

/// Protein residues that carry glycans: anchor atoms (A, B, link); see [`crate::anchor`].
pub fn site_atoms(residue: &str) -> Option<[&'static str; 3]> {
    crate::anchor::anchor(residue).map(|a| a.atoms)
}

const WATERS: &[&str] = &[
    "HOH", "WAT", "H2O", "DOD", "D2O", "TIP", "TIP3", "TP3", "SOL", "OH2",
];

/// Atomic number of an element symbol (any case); 0 for unknown symbols.
pub fn atomic_number(element: &str) -> f64 {
    const TABLE: &[&str] = &[
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S",
        "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA",
        "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD",
        "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND", "PM",
        "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS",
        "IR", "PT", "AU", "HG", "TL", "PB", "BI",
    ];
    let key = element.trim().to_ascii_uppercase();
    TABLE
        .iter()
        .position(|e| *e == key)
        .map_or(0.0, |i| (i + 1) as f64)
}

fn v3(p: glysys::Vec3) -> [f64; 3] {
    [p.x, p.y, p.z]
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// One residue of the deposited glycan tree.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DepositedResidue {
    pub id: ResidueId,
    /// CCD component name
    pub name: String,
    /// residue path in the builder's convention ("r", "r/4", "r/4/4/6", ...)
    pub path: String,
    /// parent oxygen number (None for the root, bonded to the protein)
    pub parent_position: Option<u32>,
}

/// The deposited glycan at a site.
#[derive(Debug, Clone)]
pub struct DepositedGlycan {
    pub residues: Vec<DepositedResidue>,
    /// (residue path, GLYCAM atom name) -> position (heavy atoms)
    pub atoms: BTreeMap<(String, String), [f64; 3]>,
}

impl DepositedGlycan {
    pub fn residue_ids(&self) -> BTreeSet<ResidueId> {
        self.residues.iter().map(|r| r.id.clone()).collect()
    }

    pub fn by_path(&self, path: &str) -> Option<&DepositedResidue> {
        self.residues.iter().find(|r| r.path == path)
    }

    /// GLYCAM condensed sequence of the tree (`site.glycam_sequence`): the highest-position child
    /// continues the main chain, the others are written as branches.
    pub fn glycam_sequence(&self) -> Result<String> {
        fn write(g: &DepositedGlycan, k: usize, link: &str) -> Result<String> {
            let node = &g.residues[k];
            let (name, anomer, cpos) = glycam_of(&node.name)
                .ok_or_else(|| invalid(format!("no GLYCAM token for {}", node.name)))?;
            let mut kids: Vec<usize> = (0..g.residues.len())
                .filter(|&j| {
                    g.residues[j].path.rsplit_once('/').map(|(p, _)| p) == Some(node.path.as_str())
                })
                .collect();
            kids.sort_by_key(|&j| g.residues[j].parent_position);
            let mut prefix = String::new();
            if let Some((&last, rest)) = kids.split_last() {
                prefix = write(
                    g,
                    last,
                    &format!("-{}", g.residues[last].parent_position.unwrap_or(0)),
                )?;
                for &j in rest {
                    prefix.push('[');
                    prefix.push_str(&write(
                        g,
                        j,
                        &format!("-{}", g.residues[j].parent_position.unwrap_or(0)),
                    )?);
                    prefix.push(']');
                }
            }
            Ok(format!("{prefix}{name}{anomer}{cpos}{link}"))
        }
        let root = self
            .residues
            .iter()
            .position(|r| r.path == "r")
            .ok_or_else(|| invalid("deposited glycan has no root"))?;
        write(self, root, "-OH")
    }
}

/// Glycan residues connected to the protein link atom, by distance (anomeric carbon within
/// `max_bond` of the link atom / a parent ring oxygen), as `site._sugar_tree`.
fn sugar_tree(
    structure: &Structure,
    link: [f64; 3],
    max_bond: f64,
) -> Result<Option<Vec<DepositedResidue>>> {
    let atoms = structure.atoms();
    let mut by_residue: BTreeMap<ResidueId, Vec<&glysys::StructureAtom>> = BTreeMap::new();
    for atom in &atoms {
        by_residue
            .entry(atom.residue.clone())
            .or_default()
            .push(atom);
    }
    let sugars: Vec<(ResidueId, String, [f64; 3])> = structure
        .residues()
        .into_iter()
        .filter_map(|r| {
            let (_, _, cpos) = glycam_of(&r.name)?;
            let anomeric = by_residue
                .get(&r.id)?
                .iter()
                .find(|a| a.name == format!("C{cpos}"))?;
            Some((r.id, r.name, v3(anomeric.position)))
        })
        .collect();
    let attached = |p: [f64; 3], seen: &BTreeSet<ResidueId>| -> Vec<usize> {
        (0..sugars.len())
            .filter(|&i| !seen.contains(&sugars[i].0) && dist(sugars[i].2, p) < max_bond)
            .collect()
    };
    let mut seen = BTreeSet::new();
    let roots = attached(link, &seen);
    if roots.is_empty() {
        return Ok(None);
    }
    if roots.len() != 1 {
        return Err(invalid(format!(
            "expected one glycan residue bonded to the site, found {}",
            roots.len()
        )));
    }
    seen.insert(sugars[roots[0]].0.clone());
    let mut out = vec![DepositedResidue {
        id: sugars[roots[0]].0.clone(),
        name: sugars[roots[0]].1.clone(),
        path: "r".into(),
        parent_position: None,
    }];
    let mut stack = vec![0usize];
    while let Some(k) = stack.pop() {
        let parent = out[k].clone();
        for atom in by_residue.get(&parent.id).map(Vec::as_slice).unwrap_or(&[]) {
            if !atom.element.eq_ignore_ascii_case("O") {
                continue;
            }
            let Ok(position) = atom.name[1..].parse::<u32>() else {
                continue;
            };
            if !atom.name[1..].chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            for i in attached(v3(atom.position), &seen) {
                seen.insert(sugars[i].0.clone());
                out.push(DepositedResidue {
                    id: sugars[i].0.clone(),
                    name: sugars[i].1.clone(),
                    path: format!("{}/{position}", parent.path),
                    parent_position: Some(position),
                });
                stack.push(out.len() - 1);
            }
        }
    }
    Ok(Some(out))
}

/// How the site environment is expanded by crystal symmetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SymmetryMode {
    /// Expand when the map is a full-cell crystallographic map with a Sohncke space group whose
    /// cell agrees with the model's CRYST1 record (cryo-EM maps: no expansion).
    Auto,
    /// Expand from the model's CRYST1 record even when the map is a box: for maps cut from a
    /// crystallographic map that keep its cell (e.g. a volume-server box around the site). The
    /// other checks of `Auto` still apply.
    Model,
    Off,
}

/// Crystal information available for a site.
#[derive(Debug, Clone, Default)]
pub struct CrystalInput {
    /// model CRYST1 cell and space-group symbol
    pub cryst1: Option<(UnitCell, String)>,
    /// map header cell, space-group number, whether the map covers the full cell
    pub map_cell: Option<UnitCell>,
    pub map_space_group: Option<i32>,
    pub map_full_cell: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SymmetryInfo {
    pub applied: bool,
    pub space_group: Option<String>,
    pub space_group_number: Option<u32>,
    pub operators: usize,
    pub cell: Option<UnitCell>,
    pub reason: String,
}

/// Decide whether (and with which group and cell) to expand by symmetry.
pub fn resolve_symmetry(
    input: &CrystalInput,
    mode: SymmetryMode,
) -> (Option<(UnitCell, SpaceGroup)>, SymmetryInfo) {
    let none = |reason: String| {
        (
            None,
            SymmetryInfo {
                applied: false,
                space_group: None,
                space_group_number: None,
                operators: 1,
                cell: None,
                reason,
            },
        )
    };
    if mode == SymmetryMode::Off {
        return none("disabled".into());
    }
    if !input.map_full_cell && mode != SymmetryMode::Model {
        return none("map does not cover a full unit cell (cryo-EM box or cropped map)".into());
    }
    let model_cell = input.cryst1.as_ref().map(|(c, _)| *c);
    if let Some(cell) = model_cell
        && cell.volume() <= 1000.0
    {
        return none("model has no crystal cell (CRYST1 volume <= 1000 A^3, e.g. cryo-EM)".into());
    }
    if let (Some(model), Some(map)) = (model_cell, input.map_cell)
        && !model.matches(&map, 0.01)
    {
        return none("map cell differs from the model CRYST1 cell".into());
    }
    let from_symbol = input
        .cryst1
        .as_ref()
        .and_then(|(_, s)| SpaceGroup::from_hm(s));
    let from_map = input
        .map_space_group
        .filter(|n| *n > 0)
        .and_then(|n| SpaceGroup::from_number(n as u32));
    let group = match (from_symbol, from_map) {
        (Some(g), Some(m)) if g.number != m.number => {
            return none(format!(
                "model space group {} differs from the map header ({})",
                g.hm, m.hm
            ));
        }
        (Some(g), _) => g,
        (None, Some(m)) => {
            if input.cryst1.is_none() && m.number == 1 {
                return none(
                    "P1 map without a model CRYST1 record (treated as a cryo-EM box)".into(),
                );
            }
            m
        }
        (None, None) => return none("no Sohncke space group in the model or map header".into()),
    };
    let Some(cell) = model_cell.or(input.map_cell) else {
        return none("no unit cell".into());
    };
    let info = SymmetryInfo {
        applied: true,
        space_group: Some(group.hm.clone()),
        space_group_number: Some(group.number),
        operators: group.ops.len(),
        cell: Some(cell),
        reason: if input.map_full_cell {
            "full-cell crystallographic map".into()
        } else {
            "model CRYST1 (map box cut from a crystallographic map)".into()
        },
    };
    (Some((cell, group)), info)
}

#[derive(Debug, Clone)]
pub struct SiteOptions {
    /// environment radius around the link atom (A)
    pub env_radius: f64,
    pub symmetry: SymmetryMode,
    /// GLYCAM sequence to fit instead of the deposited glycan's
    pub sequence: Option<String>,
}

impl Default for SiteOptions {
    fn default() -> Self {
        Self {
            env_radius: 45.0,
            symmetry: SymmetryMode::Auto,
            sequence: None,
        }
    }
}

/// Where the fitted sequence came from.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SequenceProvenance {
    pub sequence: String,
    /// "requested", "crabwurcs write_glycam", or "deposited tree (distance-based)"
    pub source: String,
    pub crabwurcs_glycam: Option<String>,
    pub deposited_tree_glycam: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Site {
    pub residue: ResidueId,
    pub residue_name: String,
    pub anchor_names: [String; 3],
    /// anchor atoms (A, B, link), f32-rounded as in the reference
    pub anchor: [[f64; 3]; 3],
    pub deposited: Option<DepositedGlycan>,
    pub sequence: SequenceProvenance,
    pub environment: Vec<EnvAtom>,
    pub symmetry: SymmetryInfo,
}

impl Site {
    pub fn label(&self) -> String {
        format!("{}:{}", self.residue.chain, self.residue.number)
    }

    /// Environment atoms of the site residue itself (identity image, side-chain atoms within
    /// three bonds of C1), handled by explicit pairs instead of the clash grids.
    pub fn is_site_atom(&self, atom: &EnvAtom) -> bool {
        atom.chain == self.residue.chain
            && atom.residue_number == self.residue.number
            && crate::anchor::anchor(&self.residue_name).is_some_and(|a| a.bonds_to(&atom.atom_name).is_some())
            && atom.is_identity()
    }
}

/// Normalise a GLYCAM string from crabWURCS to the builder's reducing-end form ("...b1-OH").
fn reducing_end_glycam(glycam: &str, root_anomer: char) -> String {
    let s = glycam.trim();
    if s.ends_with("-OH") || s.ends_with("-OME") {
        return s.to_string();
    }
    if let Some(stripped) = s.strip_suffix('-') {
        return format!("{stripped}-OH");
    }
    let tail = s.chars().rev().take(3).collect::<Vec<_>>();
    if tail.len() == 3 && tail[0].is_ascii_digit() && (tail[1] == 'a' || tail[1] == 'b') {
        return format!("{s}-OH");
    }
    format!("{s}{root_anomer}1-OH")
}

fn crabwurcs_glycam(
    structure: &Structure,
    tree: &[DepositedResidue],
) -> std::result::Result<String, String> {
    let pdb = structure.to_pdb_string();
    let glycans = crabwurcs_pdb::extract_glycans_with_provenance_from_str(&pdb, false)
        .map_err(|e| e.to_string())?;
    let want: BTreeSet<(String, isize)> = tree
        .iter()
        .map(|r| (r.id.chain.clone(), r.id.number as isize))
        .collect();
    let glycan = glycans
        .iter()
        .find(|g| {
            g.residues
                .iter()
                .map(|r| (r.chain.clone(), r.sequence_number))
                .collect::<BTreeSet<_>>()
                == want
        })
        .ok_or_else(|| "crabWURCS found no glycan with the deposited residues".to_string())?;
    let glycam = crabwurcs_iupac::write_glycam(&glycan.graph).map_err(|e| e.to_string())?;
    let root_anomer = glycam_of(&tree[0].name).map_or('b', |g| g.1);
    Ok(reducing_end_glycam(&glycam, root_anomer))
}

/// Residue paths of a GLYCAM sequence (builder convention) with their (token, anomer).
pub fn sequence_paths(sequence: &str) -> Result<BTreeMap<String, (String, char)>> {
    let seq = glycoflow_core::sequence::parse_glycam(sequence)?;
    let mut out = BTreeMap::new();
    let mut stack = vec![(seq.root, "r".to_string())];
    while let Some((k, path)) = stack.pop() {
        let node = &seq.nodes[k];
        for &c in &node.children {
            let p = seq.nodes[c].ppos.unwrap_or(0);
            stack.push((c, format!("{path}/{p}")));
        }
        out.insert(path, (node.name.clone(), node.anomer));
    }
    Ok(out)
}

fn same_tree(sequence: &str, tree: &[DepositedResidue]) -> bool {
    let Ok(paths) = sequence_paths(sequence) else {
        return false;
    };
    paths.len() == tree.len()
        && tree.iter().all(|r| {
            paths
                .get(&r.path)
                .zip(glycam_of(&r.name))
                .is_some_and(|((token, anomer), (t, a, _))| token == t && *anomer == a)
        })
}

/// (residue name, anchor atom names, anchor positions).
type AnchorInfo = (String, [&'static str; 3], [[f64; 3]; 3]);

/// Anchor atoms (A, B, link) of a site residue, f32-rounded as in the reference.
fn anchor_of(structure: &Structure, residue: &ResidueId) -> Result<AnchorInfo> {
    let residues = structure.residues();
    let site_residue = residues.iter().find(|r| &r.id == residue).ok_or_else(|| {
        invalid(format!(
            "site residue {}:{} not found",
            residue.chain, residue.number
        ))
    })?;
    let names = site_atoms(&site_residue.name).ok_or_else(|| {
        invalid(format!(
            "{}:{} is {}, not a supported glycosylation site",
            residue.chain, residue.number, site_residue.name
        ))
    })?;
    let mut anchor = [[0.0; 3]; 3];
    for (k, name) in names.iter().enumerate() {
        let id = structure.find_atom(residue, name).ok_or_else(|| {
            invalid(format!(
                "site residue {}:{} has no {name} atom",
                residue.chain, residue.number
            ))
        })?;
        let p = structure
            .atom_position(id)
            .ok_or_else(|| invalid("site atom without a position"))?;
        anchor[k] = v3(p).map(|x| x as f32 as f64);
    }
    Ok((site_residue.name.clone(), names, anchor))
}

/// The deposited glycan attached at a site (None when the site carries no glycan).
pub fn deposited_glycan(
    structure: &Structure,
    residue: &ResidueId,
) -> Result<Option<DepositedGlycan>> {
    let (_, _, anchor) = anchor_of(structure, residue)?;
    let tree = sugar_tree(structure, anchor[2], 1.75)?;
    Ok(match tree {
        Some(tree) => {
            let mut atoms = BTreeMap::new();
            let all = structure.atoms();
            for r in &tree {
                let rename = atom_rename(&r.name);
                for a in all.iter().filter(|a| a.residue == r.id) {
                    if a.element.eq_ignore_ascii_case("H") || a.element.eq_ignore_ascii_case("D") {
                        continue;
                    }
                    let name = rename
                        .iter()
                        .find(|(p, _)| *p == a.name)
                        .map_or(a.name.as_str(), |(_, g)| *g);
                    atoms.insert(
                        (r.path.clone(), name.to_string()),
                        v3(a.position).map(|x| x as f32 as f64),
                    );
                }
            }
            Some(DepositedGlycan {
                residues: tree,
                atoms,
            })
        }
        None => None,
    })
}

/// Extract a site from a model. `crystal` describes the model/map crystal records.
pub fn load_site(
    structure: &Structure,
    residue: &ResidueId,
    crystal: &CrystalInput,
    options: &SiteOptions,
) -> Result<Site> {
    let (residue_name, names, anchor) = anchor_of(structure, residue)?;
    let deposited = deposited_glycan(structure, residue)?;
    let tree = deposited.as_ref().map(|d| d.residues.clone());
    // sequence: requested, else crabWURCS from the deposited graph (checked against the
    // distance-based tree), else the tree itself
    let tree_glycam = deposited
        .as_ref()
        .map(DepositedGlycan::glycam_sequence)
        .transpose()?;
    let mut warnings = Vec::new();
    let crab = match &tree {
        Some(t) => match crabwurcs_glycam(structure, t) {
            Ok(s) => Some(s),
            Err(e) => {
                warnings.push(format!("crabWURCS GLYCAM export failed: {e}"));
                None
            }
        },
        None => None,
    };
    let (sequence, source) = if let Some(s) = &options.sequence {
        (s.clone(), "requested".to_string())
    } else {
        let tree = tree.as_ref().ok_or_else(|| {
            invalid(format!(
                "no glycan is attached at {}:{}; give the GLYCAM sequence to fit",
                residue.chain, residue.number
            ))
        })?;
        match &crab {
            Some(s) if same_tree(s, tree) => (s.clone(), "crabwurcs write_glycam".to_string()),
            Some(s) => {
                warnings.push(format!(
                    "crabWURCS GLYCAM {s} does not match the deposited tree; using the tree"
                ));
                (
                    tree_glycam.clone().unwrap_or_default(),
                    "deposited tree (distance-based)".to_string(),
                )
            }
            None => (
                tree_glycam.clone().unwrap_or_default(),
                "deposited tree (distance-based)".to_string(),
            ),
        }
    };
    // environment: everything but waters, hydrogens and the target glycan (all its images)
    let exclude = deposited
        .as_ref()
        .map(DepositedGlycan::residue_ids)
        .unwrap_or_default();
    let base: Vec<EnvAtom> = structure
        .atoms()
        .into_iter()
        .filter(|a| {
            !exclude.contains(&a.residue)
                && !WATERS.contains(&a.residue_name.as_str())
                && !a.element.eq_ignore_ascii_case("H")
                && !a.element.eq_ignore_ascii_case("D")
        })
        .map(|a| EnvAtom {
            position: v3(a.position),
            atomic_number: atomic_number(&a.element),
            chain: a.residue.chain.clone(),
            residue_number: a.residue.number,
            residue_name: a.residue_name.clone(),
            atom_name: a.name.clone(),
            op: 0,
            shift: [0, 0, 0],
        })
        .collect();
    let (crystal_ops, symmetry) = resolve_symmetry(crystal, options.symmetry);
    let environment = expand_environment(
        &base,
        crystal_ops.as_ref().map(|(c, g)| (c, g)),
        anchor[2],
        options.env_radius,
    );
    Ok(Site {
        residue: residue.clone(),
        residue_name,
        anchor_names: names.map(str::to_string),
        anchor,
        deposited,
        sequence: SequenceProvenance {
            sequence,
            source,
            crabwurcs_glycam: crab,
            deposited_tree_glycam: tree_glycam,
            warnings,
        },
        environment,
        symmetry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reducing_end_forms() {
        assert_eq!(
            reducing_end_glycam("DGlcpNAcb1-4DGlcpNAcb1-OH", 'b'),
            "DGlcpNAcb1-4DGlcpNAcb1-OH"
        );
        assert_eq!(
            reducing_end_glycam("DGlcpNAcb1-4DGlcpNAcb1-", 'b'),
            "DGlcpNAcb1-4DGlcpNAcb1-OH"
        );
        assert_eq!(
            reducing_end_glycam("DGlcpNAcb1-4DGlcpNAc", 'b'),
            "DGlcpNAcb1-4DGlcpNAcb1-OH"
        );
    }

    #[test]
    fn sequence_paths_follow_the_builder_convention() {
        let p = sequence_paths("DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH").unwrap();
        let keys = p.keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys, vec!["r", "r/4", "r/4/4", "r/4/4/3", "r/4/4/6"]);
        assert_eq!(p["r/4/4"], ("DManp".to_string(), 'b'));
    }

    #[test]
    fn atomic_numbers() {
        assert_eq!(atomic_number("C"), 6.0);
        assert_eq!(atomic_number("Zn"), 30.0);
        assert_eq!(atomic_number("SE"), 34.0);
    }
}
