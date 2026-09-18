//! Shared, serializable domain types used by the ReGlyco workspace.

use std::path::PathBuf;

use glysys::{ParameterizedSystem, ResidueId, Structure, Vec3};

/// Shared, topology-aware steric primitives used by both the search engine
/// and the validator.  The browser and native workflows must make the same
/// decision about a contact; keeping the radius table and thresholds here
/// prevents the old 1.7 Å centre-distance heuristic from diverging between
/// execution paths.
pub mod steric {
    use std::collections::{BTreeSet, HashMap};

    use glysys::{AtomId, ResidueId, Structure};
    use serde::{Deserialize, Serialize};

    /// Fast spatial prescreen retained for compatibility with Cookbook's
    /// search. It is not a chemically meaningful final clash criterion.
    pub const COOKBOOK_PRESCREEN_DISTANCE_ANGSTROM: f64 = 1.7;
    /// Surface overlap at or below this value is considered clear for the
    /// heavy-atom model used by ReGlyco.
    pub const VDW_CLEAR_OVERLAP_ANGSTROM: f64 = 0.4;
    /// Larger introduced overlaps are hard failures.  The 0.6 Å tolerance is
    /// deliberately less strict than all-atom MolProbity because ReGlyco's
    /// generated PDBs do not add a complete hydrogen model.
    pub const VDW_HARD_OVERLAP_ANGSTROM: f64 = 0.6;
    /// Spatial hashing cell used by the topology-aware contact evaluator.
    /// The largest radius sum in the supported heavy-atom table is below
    /// 4 Å, so querying the surrounding cells is exact for every positive
    /// overlap while avoiding the old O(glycan_atoms × protein_atoms) pass.
    const VDW_GRID_CELL_ANGSTROM: f64 = 3.0;
    const MAX_VDW_RADIUS_ANGSTROM: f64 = 1.98;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ContactClass {
        Clear,
        Advisory,
        Hard,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Contact {
        pub first: AtomId,
        pub second: AtomId,
        pub distance_angstrom: f64,
        pub overlap_angstrom: f64,
        pub class: ContactClass,
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ContactSummary {
        pub contacts: Vec<Contact>,
        pub clear_count: usize,
        pub advisory_count: usize,
        pub hard_count: usize,
        pub maximum_overlap_angstrom: f64,
        pub total_overlap_angstrom: f64,
    }

    /// Element-specific Bondi-like heavy-atom radii.  These values are
    /// versioned in the report policy metadata so future all-atom radii can be
    /// introduced without making historical reports ambiguous.
    pub fn vdw_radius(element: &str) -> f64 {
        match element.trim().to_ascii_uppercase().as_str() {
            "H" => 1.20,
            "C" => 1.70,
            "N" => 1.55,
            "O" => 1.52,
            "F" => 1.47,
            "P" => 1.80,
            "S" => 1.80,
            "CL" => 1.75,
            "BR" => 1.85,
            "I" => 1.98,
            _ => 1.70,
        }
    }

    pub fn overlap(first_element: &str, second_element: &str, distance_angstrom: f64) -> f64 {
        vdw_radius(first_element) + vdw_radius(second_element) - distance_angstrom
    }

    pub fn classify(overlap_angstrom: f64) -> ContactClass {
        classify_with_thresholds(
            overlap_angstrom,
            VDW_CLEAR_OVERLAP_ANGSTROM,
            VDW_HARD_OVERLAP_ANGSTROM,
        )
    }

    pub fn classify_with_thresholds(
        overlap_angstrom: f64,
        clear_overlap_angstrom: f64,
        hard_overlap_angstrom: f64,
    ) -> ContactClass {
        if overlap_angstrom > hard_overlap_angstrom {
            ContactClass::Hard
        } else if overlap_angstrom > clear_overlap_angstrom {
            ContactClass::Advisory
        } else {
            ContactClass::Clear
        }
    }

    pub fn distance(first: glysys::Vec3, second: glysys::Vec3) -> f64 {
        ((first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2))
            .sqrt()
    }

    fn ordered(first: AtomId, second: AtomId) -> (AtomId, AtomId) {
        if first <= second {
            (first, second)
        } else {
            (second, first)
        }
    }

    type Cell = (i32, i32, i32);

    fn grid_cell(position: glysys::Vec3) -> Cell {
        (
            (position.x / VDW_GRID_CELL_ANGSTROM).floor() as i32,
            (position.y / VDW_GRID_CELL_ANGSTROM).floor() as i32,
            (position.z / VDW_GRID_CELL_ANGSTROM).floor() as i32,
        )
    }

    fn append_grid_candidates(
        grid: &HashMap<Cell, Vec<usize>>,
        position: glysys::Vec3,
        radius_angstrom: f64,
        candidates: &mut Vec<usize>,
    ) {
        let center = grid_cell(position);
        let radius = (radius_angstrom / VDW_GRID_CELL_ANGSTROM).ceil().max(1.0) as i32;
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    if let Some(indices) = grid.get(&(center.0 + dx, center.1 + dy, center.2 + dz))
                    {
                        candidates.extend(indices.iter().copied());
                    }
                }
            }
        }
    }

    fn bonded_or_13(first: AtomId, second: AtomId, bonds: &BTreeSet<(AtomId, AtomId)>) -> bool {
        if bonds.contains(&ordered(first, second)) {
            return true;
        }
        let neighbours = |atom: AtomId| {
            bonds.iter().filter_map(move |(left, right)| {
                if *left == atom {
                    Some(*right)
                } else if *right == atom {
                    Some(*left)
                } else {
                    None
                }
            })
        };
        neighbours(first).any(|middle| bonds.contains(&ordered(middle, second)))
    }

    /// Return whether two atoms are a plausible omitted-CONNECT glycosidic
    /// bond.  A proximity-only rule is too permissive: two different sugars
    /// can be closer than 1.85 Å in a malformed pose and must still be
    /// reported as a clash.  Glycosidic junctions, however, have a stable
    /// atom-name pattern (anomeric C1, or C2 for keto-acids, to an oxygen).
    fn likely_glycosidic_bond(
        first: &glysys::StructureAtom,
        second: &glysys::StructureAtom,
    ) -> bool {
        let first_name = first.name.trim().to_ascii_uppercase();
        let second_name = second.name.trim().to_ascii_uppercase();
        let anomeric = |name: &str| matches!(name, "C1" | "C2" | "C1A" | "C2A");
        let oxygen = |name: &str| name.starts_with('O');
        (anomeric(&first_name) && oxygen(&second_name))
            || (anomeric(&second_name) && oxygen(&first_name))
    }

    /// Infer only the small set of covalent edges that are routinely omitted
    /// from carbohydrate PDB exports.  A previous proximity-only inference
    /// treated every pair of atoms in one residue as bonded; that can hide a
    /// genuine same-residue overlap in a malformed or alternate conformer.
    /// Ring edges and common substituent edges are name-stable across the
    /// GlycoShape component library, while all other connectivity should come
    /// from explicit CONECT/LINK records.
    fn likely_same_residue_bond(
        first: &glysys::StructureAtom,
        second: &glysys::StructureAtom,
        residue_name: &str,
    ) -> bool {
        let mut names = [
            first.name.trim().to_ascii_uppercase(),
            second.name.trim().to_ascii_uppercase(),
        ];
        names.sort();
        let [left, right] = names;
        matches!(
            (left.as_str(), right.as_str()),
            // Backbone/attachment-residue edges needed for the 1–3 exclusion
            // around a glycosylation anchor when a PDB omits CONECT records.
            // These are residue-name independent and therefore safe for the
            // supported ASN/SER/THR/TRP/PRO/HYP sidechains.
            ("CA", "CB")
                | ("CB", "CG")
                | ("CB", "OG")
                | ("CB", "OG1")
                | ("CG", "ND2")
                | ("CG", "OD1")
                | ("CD1", "CG")
                | ("C", "CA")
                | ("C1", "C2")
                | ("C2", "C3")
                | ("C3", "C4")
                | ("C4", "C5")
                | ("C1", "O5")
                | ("C5", "O5")
                | ("C4", "O4")
                | ("C5", "C6")
                | ("C6", "O6")
                | ("C2", "O6")
                | ("C2", "N2")
                | ("C2N", "N2")
                | ("C2N", "O2N")
                | ("C2N", "CME")
        ) || {
            // Modified sugars often retain numbered C/O substituent names
            // (for example C3–O3 or C6–O6).  Accept these local edges only
            // when one atom is carbon and the other is a heteroatom with the
            // same numeric position; unrelated O–O/N–O proximity remains a
            // reportable contact.
            let carbon_hetero = |carbon: &str, hetero: &str| {
                carbon.starts_with('C')
                    && matches!(hetero.chars().next(), Some('O' | 'N' | 'S'))
                    && carbon.get(1..).is_some()
                    && hetero.get(1..).is_some()
                    && carbon.get(1..) == hetero.get(1..)
            };
            carbon_hetero(&left, &right) || carbon_hetero(&right, &left)
        } || (matches!(
            residue_name.trim().to_ascii_uppercase().as_str(),
            "ARA" | "ARB" | "AFL" | "RIB"
        ) && (left.as_str(), right.as_str()) == ("C1", "O4"))
            || (matches!(
                residue_name.trim().to_ascii_uppercase().as_str(),
                "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
            ) && (left.as_str(), right.as_str()) == ("C2", "O6"))
    }

    fn is_solvent(name: &str) -> bool {
        matches!(
            name.trim().to_ascii_uppercase().as_str(),
            "HOH" | "WAT" | "DOD" | "H2O" | "SOL" | "ROH"
        )
    }

    fn is_hydrogen(atom: &glysys::StructureAtom) -> bool {
        atom.element.eq_ignore_ascii_case("H")
            || atom.element.eq_ignore_ascii_case("D")
            || atom.name.trim().chars().next().is_some_and(|value| {
                value.eq_ignore_ascii_case(&'H') || value.eq_ignore_ascii_case(&'D')
            })
    }

    /// Evaluate glycan-to-protein and glycan-to-glycan heavy-atom contacts.
    /// Covalent 1–2/1–3 neighbours and solvent are excluded.  The optional
    /// site focus is useful for reports that validate a single refined arm.
    pub fn evaluate_structure(structure: &Structure, focus_sites: &[ResidueId]) -> ContactSummary {
        evaluate_structure_with_thresholds(
            structure,
            focus_sites,
            VDW_CLEAR_OVERLAP_ANGSTROM,
            VDW_HARD_OVERLAP_ANGSTROM,
        )
    }

    /// Evaluate contacts with caller-supplied policy thresholds.  This is
    /// useful for validation options while preserving a single topology and
    /// radius implementation for search, native validation, and WASM.
    pub fn evaluate_structure_with_thresholds(
        structure: &Structure,
        focus_sites: &[ResidueId],
        clear_overlap_angstrom: f64,
        hard_overlap_angstrom: f64,
    ) -> ContactSummary {
        let atoms = structure.atoms();
        let residues = structure.residues();
        let mut glycan_ids = structure
            .metadata()
            .glycan_trees
            .iter()
            .filter(|tree| {
                focus_sites.is_empty()
                    || tree
                        .attachment_site
                        .as_ref()
                        .is_some_and(|site| focus_sites.contains(site))
            })
            .flat_map(|tree| tree.residue_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        if glycan_ids.is_empty() && focus_sites.is_empty() {
            glycan_ids = residues
                .iter()
                .filter(|residue| is_glycan_name(&residue.name))
                .map(|residue| residue.id.clone())
                .collect();
        }
        if glycan_ids.is_empty() {
            return ContactSummary::default();
        }
        let mut bonds = structure
            .bonds()
            .into_iter()
            .map(|(left, right)| ordered(left, right))
            .collect::<BTreeSet<_>>();
        for site in &structure.metadata().glycosylation_sites {
            if let (Some(left), Some(right)) = (
                structure.find_atom(&site.protein_residue, &site.protein_atom),
                structure.find_atom(&site.glycan_residue, &site.glycan_atom),
            ) {
                bonds.insert(ordered(left, right));
            }
        }
        let attachment_residues = structure
            .metadata()
            .glycosylation_sites
            .iter()
            .map(|site| site.protein_residue.clone())
            .collect::<BTreeSet<_>>();
        // Infer only known short same-residue carbohydrate/attachment edges
        // for PDBs that omit CONECT records.  Other protein topology is not
        // needed for glycan-vs-protein contacts and is deliberately left
        // untouched so malformed close pairs cannot disappear as exclusions.
        for residue in &residues {
            if !glycan_ids.contains(&residue.id) && !attachment_residues.contains(&residue.id) {
                continue;
            }
            let ids = residue
                .atoms
                .iter()
                .filter_map(|id| atoms.iter().find(|atom| atom.id == *id));
            let collected = ids.collect::<Vec<_>>();
            for (index, left) in collected.iter().enumerate() {
                for right in collected.iter().skip(index + 1) {
                    let d = distance(left.position, right.position);
                    if (0.85..=1.85).contains(&d)
                        && likely_same_residue_bond(left, right, &residue.name)
                    {
                        bonds.insert(ordered(left.id, right.id));
                    }
                }
            }
        }
        // Glycan PDBs frequently omit CONECT records at glycosidic junctions.
        // Metadata supplies the tree topology, so infer only short bonds
        // between residues in the same tree rather than all nearby sugars.
        for tree in &structure.metadata().glycan_trees {
            let ids = tree.residue_ids.iter().collect::<BTreeSet<_>>();
            let tree_atoms = atoms
                .iter()
                .filter(|atom| ids.contains(&atom.residue) && !is_hydrogen(atom))
                .collect::<Vec<_>>();
            for (index, left) in tree_atoms.iter().enumerate() {
                for right in tree_atoms.iter().skip(index + 1) {
                    let residue_name = residues
                        .iter()
                        .find(|residue| residue.id == left.residue)
                        .map(|residue| residue.name.as_str())
                        .unwrap_or_default();
                    if (left.residue == right.residue
                        && !likely_same_residue_bond(left, right, residue_name))
                        || (left.residue != right.residue && !likely_glycosidic_bond(left, right))
                    {
                        continue;
                    }
                    let d = distance(left.position, right.position);
                    if (0.6..=1.85).contains(&d) {
                        bonds.insert(ordered(left.id, right.id));
                    }
                }
            }
        }
        let glycan_atoms = atoms
            .iter()
            .filter(|atom| {
                glycan_ids.contains(&atom.residue)
                    && !residues
                        .iter()
                        .any(|residue| residue.id == atom.residue && is_solvent(&residue.name))
                    && !is_hydrogen(atom)
                    && atom.occupancy > 0.0
            })
            .collect::<Vec<_>>();
        let other_atoms = atoms
            .iter()
            .filter(|atom| {
                // When a caller focuses one attachment, keep the remaining
                // glycans in the comparison set so glycan–glycan contacts are
                // still reported. Only the focused tree is treated as the
                // moving side of the contact; protein and non-focused glycan
                // atoms are both legitimate neighbours.
                !glycan_ids.contains(&atom.residue)
                    && !residues
                        .iter()
                        .any(|residue| residue.id == atom.residue && is_solvent(&residue.name))
                    && !is_hydrogen(atom)
                    && atom.occupancy > 0.0
            })
            .collect::<Vec<_>>();
        // Query only atoms that can have a positive VDW overlap.  The prior
        // implementation compared every glycan atom with every protein atom
        // for every chromosome in the steric GA; on a 10k-atom protein that
        // made each generation scale quadratically in the population.  The
        // hash is deterministic in its output because contacts are sorted by
        // atom IDs before the counters are derived.
        let mut other_grid = HashMap::<Cell, Vec<usize>>::new();
        for (index, atom) in other_atoms.iter().enumerate() {
            other_grid
                .entry(grid_cell(atom.position))
                .or_default()
                .push(index);
        }
        let mut glycan_grid = HashMap::<Cell, Vec<usize>>::new();
        for (index, atom) in glycan_atoms.iter().enumerate() {
            glycan_grid
                .entry(grid_cell(atom.position))
                .or_default()
                .push(index);
        }

        let mut summary = ContactSummary::default();
        let mut candidates = Vec::with_capacity(64);
        let mut add_contact = |first: &glysys::StructureAtom, second: &glysys::StructureAtom| {
            if bonded_or_13(first.id, second.id, &bonds) {
                return;
            }
            let distance_angstrom = distance(first.position, second.position);
            let overlap_angstrom = overlap(&first.element, &second.element, distance_angstrom);
            // Pairs with no positive surface overlap are not contacts at all.
            if overlap_angstrom <= 0.0 {
                return;
            }
            let class = classify_with_thresholds(
                overlap_angstrom,
                clear_overlap_angstrom,
                hard_overlap_angstrom,
            );
            summary.contacts.push(Contact {
                first: first.id,
                second: second.id,
                distance_angstrom,
                overlap_angstrom,
                class,
            });
        };

        for first in &glycan_atoms {
            candidates.clear();
            append_grid_candidates(
                &other_grid,
                first.position,
                vdw_radius(&first.element) + MAX_VDW_RADIUS_ANGSTROM,
                &mut candidates,
            );
            for index in candidates.iter().copied() {
                if let Some(second) = other_atoms.get(index) {
                    add_contact(first, second);
                }
            }
        }
        for (index, first) in glycan_atoms.iter().enumerate() {
            candidates.clear();
            append_grid_candidates(
                &glycan_grid,
                first.position,
                vdw_radius(&first.element) + MAX_VDW_RADIUS_ANGSTROM,
                &mut candidates,
            );
            for second_index in candidates.iter().copied().filter(|second| *second > index) {
                if let Some(second) = glycan_atoms.get(second_index) {
                    add_contact(first, second);
                }
            }
        }
        summary
            .contacts
            .sort_unstable_by_key(|contact| (contact.first, contact.second));
        for contact in &summary.contacts {
            match contact.class {
                ContactClass::Clear => summary.clear_count += 1,
                ContactClass::Advisory => summary.advisory_count += 1,
                ContactClass::Hard => summary.hard_count += 1,
            }
            summary.maximum_overlap_angstrom = summary
                .maximum_overlap_angstrom
                .max(contact.overlap_angstrom);
            summary.total_overlap_angstrom += contact.overlap_angstrom.max(0.0);
        }
        summary
    }

    fn is_glycan_name(name: &str) -> bool {
        matches!(
            name.trim().to_ascii_uppercase().as_str(),
            "NAG"
                | "BMA"
                | "MAN"
                | "MAG"
                | "GAL"
                | "GLC"
                | "GAM"
                | "FUC"
                | "XYS"
                | "XYP"
                | "SIA"
                | "NEU"
                | "KDO"
                | "RIB"
                | "ARA"
                | "IDO"
                | "GCU"
                | "G6D"
                | "GNA"
                | "NDG"
                | "AMN"
                | "NGA"
                | "GLA"
                | "NAN"
                | "A2G"
                | "3VA"
                | "3LB"
                | "2LA"
                | "0FA"
                | "8GL"
                | "VVA"
                | "0VA"
                | "4GA"
                | "0YB"
                | "4YB"
                | "UYB"
                | "BGC"
                | "4YA"
                | "VMB"
                | "VMA"
                | "VGA"
                | "0SA"
                | "ARB"
                | "AFL"
                | "NGC"
                | "KDN"
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn vdw_policy_has_clear_advisory_and_hard_bands() {
            assert_eq!(classify(0.4), ContactClass::Clear);
            assert_eq!(classify(0.40001), ContactClass::Advisory);
            assert_eq!(classify(0.6), ContactClass::Advisory);
            assert_eq!(classify(0.60001), ContactClass::Hard);
            assert!((overlap("C", "O", 3.22) - 0.0).abs() < 1.0e-9);
        }

        #[test]
        fn periodic_policy_is_independent_of_atom_element_cutoff() {
            // A 1.7 Å centre distance can be clear for small atoms and a hard
            // overlap for carbon/oxygen; the shared evaluator uses radii.
            let carbon_nitrogen = overlap("C", "N", 3.25);
            let oxygen_oxygen = overlap("O", "O", 2.30);
            assert!(carbon_nitrogen < VDW_CLEAR_OVERLAP_ANGSTROM);
            assert!(oxygen_oxygen > VDW_HARD_OVERLAP_ANGSTROM);
        }
    }
}

pub type Result<T> = std::result::Result<T, ReGlycoError>;

#[derive(Debug, thiserror::Error)]
pub enum ReGlycoError {
    #[error(transparent)]
    GlySys(#[from] glysys::BuildError),
    #[error("glycosylation site {0} does not exist")]
    SiteNotFound(ResidueId),
    #[error("residue {site} ({residue_name}) is not a supported ASN/SER/THR/TRP/HYP/PRO site")]
    UnsupportedSite {
        site: ResidueId,
        residue_name: String,
    },
    #[error("glycosylation site {0} is already occupied")]
    OccupiedSite(ResidueId),
    #[error("required attachment atom {atom} is missing from {site}")]
    MissingSiteAtom { site: ResidueId, atom: String },
    #[error(
        "glycan conformer must contain exactly one resolvable root with the required anomeric and ring-anchor atoms"
    )]
    AmbiguousGlycanRoot,
    #[error("glycan conformer is missing orientation atom {0}")]
    MissingGlycanAtom(String),
    #[error("attachment geometry contains coincident or collinear anchors")]
    InvalidGeometry,
    #[error("no unused single-character chain identifier remains")]
    NoAvailableChain,
    #[error("ensemble input is missing: {0}")]
    MissingInput(String),
    #[error("invalid attachment chemistry at {site}: {message}")]
    InvalidChemistry { site: ResidueId, message: String },
    #[error("ensemble search produced a clashing result")]
    ClashFreeRequired,
    #[error("operation is not supported in this milestone: {0}")]
    UnsupportedOperation(String),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GlycosylationSite {
    pub residue: ResidueId,
}

impl GlycosylationSite {
    pub fn new(chain: impl Into<String>, number: i32) -> Self {
        Self {
            residue: ResidueId {
                chain: chain.into(),
                number,
                insertion_code: None,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlycanConformer {
    pub structure: Structure,
}

impl GlycanConformer {
    pub fn new(structure: Structure) -> Self {
        Self { structure }
    }
}

#[derive(Debug, Clone)]
pub struct AttachmentRequest {
    pub site: GlycosylationSite,
    pub conformer: GlycanConformer,
}

#[derive(Debug, Clone)]
pub struct BuildRequest {
    pub protein: Structure,
    pub attachments: Vec<AttachmentRequest>,
    pub parameterize: bool,
}

#[derive(Debug, Clone)]
pub struct BuildProduct {
    pub structure: Structure,
    pub system: Option<ParameterizedSystem>,
}

pub type BuildResult = BuildProduct;

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anomer {
    Alpha,
    Beta,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlycanSource {
    LocalBundle(PathBuf),
    GlyTouCan(String),
}

/// Residue-name convention used when fetching glycan structure assets.
///
/// The coordinate container remains PDB in both cases.  This enum selects
/// the authoritative residue names supplied by the structure provider; it is
/// deliberately kept separate from the force-field/template names used by
/// GlySys so changing the display convention cannot change scoring chemistry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ResidueNameFormat {
    #[serde(rename = "PDB", alias = "pdb")]
    Pdb,
    #[serde(rename = "GLYCAM", alias = "glycam")]
    Glycam,
}

impl Default for ResidueNameFormat {
    fn default() -> Self {
        Self::Pdb
    }
}

impl ResidueNameFormat {
    /// Canonical path segment accepted by the GlycoShape structure endpoint.
    pub const fn api_segment(self) -> &'static str {
        match self {
            Self::Pdb => "PDB",
            Self::Glycam => "GLYCAM",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GlycanQuery {
    pub source: GlycanSource,
    #[serde(default)]
    pub anomer: Anomer,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_level")]
    pub level: String,
}

fn default_format() -> String {
    "pdb".into()
}

fn default_level() -> String {
    "2".into()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VonMisesComponent {
    pub mean_degrees: f64,
    pub concentration: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LinkagePrior {
    pub phi: Vec<VonMisesComponent>,
    pub psi: Vec<VonMisesComponent>,
}

#[derive(Debug, Clone)]
pub struct EnsembleConformer {
    pub id: String,
    pub structure: Structure,
    pub cluster_index: usize,
    pub cluster_weight: f64,
    pub main_cluster: Option<usize>,
    pub anomer: Anomer,
    pub linkage_anchor: Option<(String, String)>,
    pub priors: LinkagePrior,
}

#[derive(Debug, Clone)]
pub struct GlycanEnsemble {
    pub query: GlycanQuery,
    pub conformers: Vec<EnsembleConformer>,
    pub provenance: String,
    /// Whether whole-conformer populations came from the authoritative asset
    /// metadata or the deterministic equal-weight fallback.
    pub population_source: ConformerPopulationSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformerPopulationSource {
    AssetMetadata,
    EqualFallback,
}

#[derive(Debug, Clone)]
pub struct SearchSite {
    pub site: GlycosylationSite,
    pub ensemble: GlycanEnsemble,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub ensemble_mode: Option<String>,
    pub burn_in_steps: Option<usize>,
    pub thinning_steps: Option<usize>,
    pub seed: u64,
    pub ensemble_size: usize,
    pub population_size: usize,
    pub generations: usize,
    pub clash_distance: f64,
    pub require_clash_free: bool,
    pub scan_rotamers: bool,
    /// Apply one bounded attachment-only VMM likelihood polish after the
    /// strict steric search has already found a complete solution.
    pub polish_attachment_vmm: bool,
    /// Candidate selection policy for attachment searches.  The historical
    /// Cookbook policy returns the first feasible chromosome; Build requests
    /// opt into the probability-aware policy explicitly.
    #[serde(default)]
    pub selection_policy: SearchSelectionPolicy,
    pub mh_steps: usize,
    pub scoring_mode: SearchScoringMode,
    pub use_obc2: bool,
    pub pre_minimization: bool,
    pub pre_minimization_iterations: usize,
    pub minimization_radius: f64,
    pub energy_cutoff: f64,
    pub temperature_k: f64,
    pub burn_in: usize,
    pub thinning: usize,
    /// Number of independent chains used by the constrained ensemble fallback.
    #[serde(default = "default_mh_chains")]
    pub mh_chains: usize,
    /// Burn-in sweeps per chain; one sweep proposes every attachment site once.
    #[serde(default = "default_mh_burn_in_sweeps")]
    pub mh_burn_in_sweeps: usize,
    /// Number of accepted native proposals between emitted fallback frames.
    #[serde(default = "default_mh_thinning_accepted")]
    pub mh_thinning_accepted: usize,
    /// Numerical target used by sampled energy/interaction ensembles.  The
    /// field is optional at the request boundary so historical requests keep
    /// the CPU-reference sampler, while development GPU requests can opt in
    /// to the resident f32 target explicitly.
    #[serde(default)]
    pub sampling_target: Option<SamplingTarget>,
}

fn default_mh_chains() -> usize {
    4
}

fn default_mh_burn_in_sweeps() -> usize {
    250
}

fn default_mh_thinning_accepted() -> usize {
    50
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchScoringMode {
    #[default]
    StericPrior,
    FullEnergy,
    ProteinGlycanInteraction,
}

/// Versioned numerical targets for energy-weighted statistical ensembles.
/// These are sampler identities, not backend labels: a GPU execution can
/// still fall back to a separately labelled CPU segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingTarget {
    CpuReferenceV1,
    WebgpuF32V1,
}

impl Default for SamplingTarget {
    fn default() -> Self {
        Self::CpuReferenceV1
    }
}

/// How a completed steric search chooses among candidates that pass the
/// complete prepared steric gate.  This is deliberately separate from the
/// physical scoring mode so Ensemble sampling can retain its own target
/// distribution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSelectionPolicy {
    #[default]
    CookbookFirstFeasible,
    JointPriorV1,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            ensemble_mode: None,
            burn_in_steps: None,
            thinning_steps: None,
            seed: 0,
            ensemble_size: 100,
            population_size: 128,
            generations: 100,
            clash_distance: 1.7,
            require_clash_free: true,
            scan_rotamers: false,
            polish_attachment_vmm: false,
            selection_policy: SearchSelectionPolicy::CookbookFirstFeasible,
            mh_steps: 1000,
            scoring_mode: SearchScoringMode::StericPrior,
            use_obc2: false,
            pre_minimization: false,
            pre_minimization_iterations: 5,
            minimization_radius: 5.0,
            energy_cutoff: 10.0,
            temperature_k: 300.0,
            burn_in: 100,
            thinning: 10,
            mh_chains: default_mh_chains(),
            mh_burn_in_sweeps: default_mh_burn_in_sweeps(),
            mh_thinning_accepted: default_mh_thinning_accepted(),
            sampling_target: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClashStatus {
    ClashFree,
    BestCompleteClashing,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchSiteResult {
    pub site: GlycosylationSite,
    pub conformer_index: usize,
    pub conformer_id: String,
    pub cluster_index: usize,
    #[serde(default)]
    pub main_cluster: Option<usize>,
    pub cluster_weight: f64,
    pub phi_degrees: f64,
    pub psi_degrees: f64,
    /// Selected mixture components used by the strict steric solver.  These
    /// are optional so older/native energy outcomes remain schema-compatible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phi_component: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psi_component: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phi_within_vmm95: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psi_within_vmm95: Option<bool>,
    pub rotamer_index: Option<usize>,
    pub prior_score: f64,
    /// Normalized population of the selected whole-glycan conformer.  This is
    /// persisted separately from `prior_score` so reports can explain why a
    /// pose was preferred without reconstructing the asset metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conformer_probability: Option<f64>,
    /// Log density of the complete attachment φ/ψ mixture, per radian axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachment_log_density: Option<f64>,
    /// Per-site contribution to the joint negative log probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_prior_score: Option<f64>,
    pub steric_score: f64,
    pub coordinates: Vec<Vec3>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchGeneration {
    pub generation: usize,
    pub best_score: f64,
    pub mean_score: f64,
    #[serde(default)]
    pub best_energy_kcal_per_mol: Option<f64>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EnergySearchDiagnostics {
    pub topology_parameterizations: usize,
    pub energy_evaluations: usize,
    pub minimizations: usize,
    pub cache_hits: usize,
    pub steric_rejections: usize,
    pub failed_evaluations: usize,
    pub neighbor_pairs: usize,
    pub active_atoms: usize,
    pub active_residues: Vec<ResidueId>,
    pub topology_seconds: f64,
    pub evaluation_seconds: f64,
}

/// Force-field terms retained for an informative workflow report.  These are
/// physical energy components in kcal/mol; diagnostic subsets must never be
/// added to this total a second time.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnergyComponentBreakdown {
    pub bonds: f64,
    pub angles: f64,
    pub proper_torsions: f64,
    pub improper_torsions: f64,
    pub van_der_waals: f64,
    pub electrostatics: f64,
    pub generalized_born: f64,
    pub surface_area: f64,
    pub restraints: f64,
    #[serde(default)]
    pub dispersion_correction: f64,
}

impl EnergyComponentBreakdown {
    pub fn total(self) -> f64 {
        self.bonds
            + self.angles
            + self.proper_torsions
            + self.improper_torsions
            + self.van_der_waals
            + self.electrostatics
            + self.generalized_born
            + self.surface_area
            + self.restraints
            + self.dispersion_correction
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlycanInteractionBreakdown {
    pub site: String,
    pub glycan_id: String,
    pub van_der_waals: f64,
    pub electrostatics: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlycosidicTorsionContribution {
    pub site: String,
    pub linkage: String,
    pub atoms: [usize; 4],
    pub energy: f64,
}

/// Versioned, explicitly labelled energy decomposition for Build and
/// Ensemble outputs.  Per-glycan interactions and glycosidic torsions are
/// diagnostic subsets of the global physical components.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnergyAnalysis {
    pub version: u32,
    pub units: String,
    pub model: String,
    pub cutoff_angstrom: Option<f64>,
    pub solvent: String,
    pub backend: String,
    pub drives_selection: bool,
    pub selected_score: Option<f64>,
    pub components: EnergyComponentBreakdown,
    pub per_glycan_interactions: Vec<GlycanInteractionBreakdown>,
    pub glycosidic_torsions: Vec<GlycosidicTorsionContribution>,
    pub diagnostic_remainder: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AtomCoordinateRecord {
    pub residue: ResidueId,
    pub atom_name: String,
    pub position: Vec3,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchOutcome {
    pub sites: Vec<SearchSiteResult>,
    pub total_score: f64,
    pub seed: u64,
    pub generations: usize,
    pub clash_status: ClashStatus,
    /// True when the selected chromosome produced coordinates for every
    /// requested attachment.  This is independent from `clash_status`:
    /// diagnostic Builds may be complete while still clashing.
    #[serde(default = "default_true")]
    pub complete_output: bool,
    /// Whether the selected attachment angles satisfy the applicable VMM
    /// gate.  A complete output can intentionally be outside this gate for
    /// diagnostic Build reporting.
    #[serde(default = "default_true")]
    pub vmm_gate_satisfied: bool,
    /// Search termination is reported separately from scientific status so
    /// budget exhaustion does not look like an input or execution failure.
    #[serde(default)]
    pub termination_reason: String,
    /// Residue-level partners for steric contacts in the exported complete
    /// structure.  Entries are prefixed with `protein:` or `glycan:` so a
    /// glycan–glycan contact can be attributed to both attachment sites.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clash_partners: Vec<Vec<String>>,
    pub history: Vec<SearchGeneration>,
    pub warnings: Vec<String>,
    #[serde(default)]
    pub scoring_mode: SearchScoringMode,
    #[serde(default)]
    pub selected_energy_kcal_per_mol: Option<f64>,
    #[serde(default)]
    pub interaction_energy_kcal_per_mol: Option<f64>,
    #[serde(default)]
    pub energy_evaluations: usize,
    #[serde(default = "default_energy_cutoff")]
    pub energy_cutoff_angstrom: f64,
    #[serde(default = "default_minimization_radius")]
    pub minimization_radius_angstrom: f64,
    #[serde(default)]
    pub interaction_vdw_kcal_per_mol: Option<f64>,
    #[serde(default)]
    pub interaction_coulomb_kcal_per_mol: Option<f64>,
    #[serde(default)]
    pub energy_diagnostics: EnergySearchDiagnostics,
    /// Locally minimized winning coordinates, keyed independently of serials.
    #[serde(default)]
    pub minimized_coordinates: Vec<AtomCoordinateRecord>,
    #[serde(default)]
    pub timings: SearchTimingDiagnostics,
    #[serde(default)]
    pub vmm_polish: VmmPolishDiagnostics,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "energyAnalysis",
        alias = "energy_analysis"
    )]
    pub energy_analysis: Option<EnergyAnalysis>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VmmPolishDiagnostics {
    pub applied: bool,
    pub proposals: usize,
    pub accepted_moves: usize,
    pub score_before: f64,
    pub score_after: f64,
    pub sites: Vec<VmmPolishSiteDiagnostics>,
    /// Selection/probability diagnostics for the strict Build search.  These
    /// fields are optional in spirit and default cleanly for older reports.
    #[serde(default)]
    pub selection_policy: String,
    #[serde(default)]
    pub prior_model: String,
    #[serde(default)]
    pub first_feasible_score: Option<f64>,
    #[serde(default)]
    pub final_prior_score: Option<f64>,
    #[serde(default)]
    pub valid_candidates: usize,
    #[serde(default)]
    pub search_budget: usize,
    #[serde(default)]
    pub termination_reason: String,
    #[serde(default)]
    pub evaluations: usize,
    /// Geometry-stage counters are kept separate from energy counters so a
    /// report can explain where a steric search spent its budget.
    #[serde(default)]
    pub geometry_gpu_evaluations: usize,
    #[serde(default)]
    pub geometry_cpu_evaluations: usize,
    #[serde(default)]
    pub geometry_gpu_seconds: f64,
    #[serde(default)]
    pub geometry_cpu_seconds: f64,
    #[serde(default)]
    pub geometry_transform_seconds: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VmmPolishSiteDiagnostics {
    pub site: String,
    pub original_phi_component: usize,
    pub original_psi_component: usize,
    pub final_phi_component: usize,
    pub final_psi_component: usize,
    pub proposals: usize,
    pub component_switched: bool,
    pub score_before: f64,
    pub score_after: f64,
    pub steric_score_before: f64,
    pub steric_score_after: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SearchTimingDiagnostics {
    pub loading_seconds: f64,
    pub preparation_seconds: f64,
    pub proposal_seconds: f64,
    pub scoring_seconds: f64,
    pub ga_seconds: f64,
    pub mh_seconds: f64,
    pub materialization_seconds: f64,
    pub parameterization_seconds: f64,
    pub solvation_seconds: f64,
    pub output_seconds: f64,
    pub states_per_second: f64,
}

fn default_energy_cutoff() -> f64 {
    10.0
}
fn default_minimization_radius() -> f64 {
    5.0
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod residue_name_format_tests {
    use super::ResidueNameFormat;

    #[test]
    fn defaults_to_pdb_and_uses_canonical_api_segments() {
        assert_eq!(ResidueNameFormat::default(), ResidueNameFormat::Pdb);
        assert_eq!(ResidueNameFormat::Pdb.api_segment(), "PDB");
        assert_eq!(ResidueNameFormat::Glycam.api_segment(), "GLYCAM");
    }

    #[test]
    fn accepts_legacy_lowercase_wire_values() {
        assert_eq!(
            serde_json::from_str::<ResidueNameFormat>("\"pdb\"").unwrap(),
            ResidueNameFormat::Pdb
        );
        assert_eq!(
            serde_json::from_str::<ResidueNameFormat>("\"glycam\"").unwrap(),
            ResidueNameFormat::Glycam
        );
        assert_eq!(
            serde_json::to_string(&ResidueNameFormat::Glycam).unwrap(),
            "\"GLYCAM\""
        );
    }
}
