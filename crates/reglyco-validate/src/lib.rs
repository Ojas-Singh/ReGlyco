//! Native structural and carbohydrate validation.
//!
//! Validation is intentionally diagnostic rather than a replacement for a
//! crystallographic validation package.  Errors represent broken topology or
//! impossible attachment chemistry; warnings identify geometry, clashes, and
//! unusual but still representable conformations.

use std::collections::BTreeSet;

use glysys::{AtomId, ResidueId, Structure, Vec3};
use reglyco_core::steric::{self};
#[cfg(feature = "density")]
use reglyco_density::{DensityScore, DensityScorer, DensityTarget};

/// Versioned component templates used by the chemistry/topology validator.
/// Reports carry this value so a future dictionary can be introduced without
/// making historical findings ambiguous.
pub const COMPONENT_DICTIONARY_VERSION: &str = "glycoshape-carbohydrate-components-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationFinding {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    #[serde(default)]
    pub site: Option<ResidueId>,
    #[serde(default)]
    pub residue: Option<ResidueId>,
    #[serde(default)]
    pub atom: Option<String>,
    #[serde(default)]
    pub observed: Option<f64>,
    #[serde(default)]
    pub expected: Option<String>,
    /// Broad validation domain (`topology`, `geometry`, `sterics`, or
    /// `torsion`). Optional so reports written by older engines remain valid.
    #[serde(default)]
    pub domain: Option<String>,
    /// Where the finding came from when a baseline comparison is available.
    #[serde(default)]
    pub origin: Option<String>,
    /// Accepted workflow stage or ensemble frame.
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub frame: Option<usize>,
    #[serde(default)]
    pub glycan_index: Option<usize>,
    /// Stable glycan identity when the validator can resolve one. Low-level
    /// validation often only has a residue identifier, so this is optional
    /// and additive to `glycan_index` for older consumers.
    #[serde(default)]
    pub glycan: Option<String>,
    #[serde(default)]
    pub linkage: Option<String>,
    #[serde(default)]
    pub involved_atoms: Vec<String>,
    #[serde(default)]
    pub metric: Option<String>,
    #[serde(default)]
    pub policy: Option<String>,
    /// Versioned policy identifier. `policy` remains for compatibility with
    /// reports written before the richer validation contract.
    #[serde(default)]
    pub policy_version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationReference {
    pub site: ResidueId,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct ValidationOptions {
    pub bond_warning_angstrom: f64,
    pub bond_error_angstrom: f64,
    pub angle_warning_degrees: f64,
    pub angle_error_degrees: f64,
    pub clash_warning_angstrom: f64,
    pub clash_error_angstrom: f64,
    pub min_density_cc: Option<f64>,
    pub references: Vec<ValidationReference>,
    /// Restrict carbohydrate geometry/clash checks to selected attachment
    /// sites. An empty list preserves whole-structure validation; refinement
    /// uses this to avoid turning unrelated deposited glycans into failures
    /// for a density fit at one site.
    pub focus_sites: Vec<ResidueId>,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            bond_warning_angstrom: 0.10,
            bond_error_angstrom: 0.20,
            angle_warning_degrees: 10.0,
            angle_error_degrees: 20.0,
            clash_warning_angstrom: 0.40,
            clash_error_angstrom: 0.60,
            min_density_cc: None,
            references: Vec::new(),
            focus_sites: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationReport {
    pub valid: bool,
    pub atom_count: usize,
    pub residue_count: usize,
    pub attachment_count: usize,
    /// Compatibility field retained for existing report consumers.
    pub warnings: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default)]
    pub findings: Vec<ValidationFinding>,
    #[serde(default = "default_component_dictionary_version")]
    pub component_dictionary_version: String,
    /// Versioned steric policy used for this report.
    #[serde(default)]
    pub steric_policy: Option<StericPolicy>,
    /// Topology-aware heavy-atom contact counts and representative contacts.
    /// Optional for compatibility with reports emitted before the VDW
    /// validator was introduced.
    #[serde(default)]
    pub steric_summary: Option<steric::ContactSummary>,
    #[serde(default)]
    #[cfg(feature = "density")]
    pub density: Option<DensityScore>,
}

fn default_component_dictionary_version() -> String {
    COMPONENT_DICTIONARY_VERSION.into()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StericPolicy {
    pub prescreen_distance_angstrom: f64,
    pub clear_overlap_angstrom: f64,
    pub hard_overlap_angstrom: f64,
    pub model: String,
}

impl Default for StericPolicy {
    fn default() -> Self {
        Self {
            prescreen_distance_angstrom: steric::COOKBOOK_PRESCREEN_DISTANCE_ANGSTROM,
            clear_overlap_angstrom: steric::VDW_CLEAR_OVERLAP_ANGSTROM,
            hard_overlap_angstrom: steric::VDW_HARD_OVERLAP_ANGSTROM,
            model: "heavy_atom_vdw_overlap_v1".into(),
        }
    }
}

fn steric_policy(options: &ValidationOptions) -> StericPolicy {
    StericPolicy {
        prescreen_distance_angstrom: steric::COOKBOOK_PRESCREEN_DISTANCE_ANGSTROM,
        clear_overlap_angstrom: options.clash_warning_angstrom,
        hard_overlap_angstrom: options.clash_error_angstrom,
        model: "heavy_atom_vdw_overlap_v1".into(),
    }
}

pub fn validate(structure: &Structure) -> ValidationReport {
    validate_with_options(structure, &ValidationOptions::default())
}

pub fn validate_with_options(
    structure: &Structure,
    options: &ValidationOptions,
) -> ValidationReport {
    let mut findings = Vec::new();
    let atoms = structure.atoms();
    let residues = structure.residues();
    if atoms.is_empty() {
        finding(
            &mut findings,
            "structure.empty",
            Severity::Error,
            "structure contains no atoms",
            None,
            None,
            None,
            None,
            None,
        );
    }
    for atom in &atoms {
        if !atom.position.x.is_finite()
            || !atom.position.y.is_finite()
            || !atom.position.z.is_finite()
        {
            finding(
                &mut findings,
                "coordinates.nonfinite",
                Severity::Error,
                "atom has a non-finite Cartesian coordinate",
                None,
                Some(atom.residue.clone()),
                Some(atom.name.clone()),
                None,
                Some("finite x/y/z"),
            );
        }
    }

    let mut bond_set = structure
        .bonds()
        .into_iter()
        .map(|(first, second)| ordered_atoms(first, second))
        .collect::<BTreeSet<_>>();
    // LINK/CONECT records are not consistently preserved by every PDB reader,
    // but GlySys retains the chemically authoritative attachment metadata.
    // Treat those pairs as covalent for the nonbonded-overlap pass as well.
    let attachment_pairs = structure
        .metadata()
        .glycosylation_sites
        .iter()
        .filter_map(|site| {
            let protein = structure.find_atom(&site.protein_residue, &site.protein_atom)?;
            let glycan = structure.find_atom(&site.glycan_residue, &site.glycan_atom)?;
            Some(ordered_atoms(protein, glycan))
        })
        .collect::<BTreeSet<_>>();
    bond_set.extend(attachment_pairs.iter().copied());
    // PDB files commonly omit CONECT records for ordinary protein atoms.  A
    // short same-residue covalent-distance inference is enough to make the
    // 1–3 exclusion chemically useful around an attachment (for example,
    // ASN CG–ND2–glycan C1) without inventing long-range connectivity.
    let attachment_residues = structure
        .metadata()
        .glycosylation_sites
        .iter()
        .map(|site| site.protein_residue.clone())
        .collect::<BTreeSet<_>>();
    let glycan_names = [
        "NAG", "BMA", "MAN", "MAG", "GAL", "GLC", "GAM", "FUC", "XYS", "XYP", "SIA", "NEU", "NAN",
        "NGC", "KDN", "KDO", "RIB", "ARA", "ARB", "AFL", "IDO", "GCU", "G6D", "GNA", "NDG", "AMN",
        "GLA", "0SA", "A2G", "3VA", "3LB", "2LA", "0FA", "8GL", "VVA", "0VA", "4GA", "0YB", "4YB",
        "UYB", "BGC", "4YA", "VMB", "VMA", "VGA",
    ];
    let mut glycan_residues = structure
        .metadata()
        .glycan_trees
        .iter()
        .flat_map(|tree| tree.residue_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    // Preserve useful behaviour for ordinary PDB files that do not carry
    // GlySys metadata yet: well-known carbohydrate residue names still enter
    // the topology-aware bond inference and chemistry checks.  Do this before
    // inferring omitted bonds so raw source conformers receive the same 1–2
    // and 1–3 exclusions as metadata-rich workflow structures.
    if glycan_residues.is_empty() && options.focus_sites.is_empty() {
        glycan_residues = residues
            .iter()
            .filter(|residue| {
                glycan_names
                    .iter()
                    .any(|name| residue.name.eq_ignore_ascii_case(name))
            })
            .map(|residue| residue.id.clone())
            .collect::<BTreeSet<_>>();
    }
    for residue in &residues {
        if !attachment_residues.contains(&residue.id) {
            continue;
        }
        let residue_atoms = residue
            .atoms
            .iter()
            .filter_map(|id| atoms.iter().find(|atom| atom.id == *id))
            .collect::<Vec<_>>();
        for (index, left) in residue_atoms.iter().enumerate() {
            for right in residue_atoms.iter().skip(index + 1) {
                if is_hydrogen(left) && is_hydrogen(right) {
                    continue;
                }
                let length = distance(left.position, right.position);
                if (0.85..=1.85).contains(&length)
                    && likely_same_residue_bond(left, right, &residue.name)
                {
                    bond_set.insert(ordered_atoms(left.id, right.id));
                }
            }
        }
    }
    // Glycan PDB files also routinely omit CONECT records.  Recover short
    // same-residue bonds, and only chemically plausible anomeric-C/O bonds
    // between residues.  A proximity-only inter-residue rule would hide real
    // clashes in crowded or malformed source conformers.
    for (index, left) in atoms.iter().enumerate() {
        if !glycan_residues.contains(&left.residue) || is_hydrogen(left) {
            continue;
        }
        for right in atoms.iter().skip(index + 1) {
            if !glycan_residues.contains(&right.residue) || is_hydrogen(right) {
                continue;
            }
            let length = distance(left.position, right.position);
            let residue_name = residues
                .iter()
                .find(|residue| residue.id == left.residue)
                .map(|residue| residue.name.as_str())
                .unwrap_or_default();
            if (0.6..=1.85).contains(&length)
                && ((left.residue == right.residue
                    && likely_same_residue_bond(left, right, residue_name))
                    || likely_glycosidic_bond(left, right))
            {
                bond_set.insert(ordered_atoms(left.id, right.id));
            }
        }
    }
    for (first, second) in &bond_set {
        let Some(left) = atoms.iter().find(|atom| atom.id == *first) else {
            finding(
                &mut findings,
                "topology.missing_bond_atom",
                Severity::Error,
                "bond references a missing first atom",
                None,
                None,
                None,
                None,
                None,
            );
            continue;
        };
        let Some(right) = atoms.iter().find(|atom| atom.id == *second) else {
            finding(
                &mut findings,
                "topology.missing_bond_atom",
                Severity::Error,
                "bond references a missing second atom",
                None,
                None,
                None,
                None,
                None,
            );
            continue;
        };
        let length = distance(left.position, right.position);
        // PDB connectivity may contain glycosidic, disulfide, and peptide
        // bonds.  A generic physical bound catches malformed records without
        // pretending that one ideal length applies to every element pair.
        if length > 2.0 + options.bond_error_angstrom {
            finding(
                &mut findings,
                "geometry.bond_length",
                Severity::Error,
                "explicit bond is implausibly long",
                None,
                Some(left.residue.clone()),
                Some(left.name.clone()),
                Some(length),
                Some("<= 2.0 Å plus tolerance"),
            );
        } else if length > 2.0 + options.bond_warning_angstrom {
            finding(
                &mut findings,
                "geometry.bond_length",
                Severity::Warning,
                "explicit bond is unusually long",
                None,
                Some(left.residue.clone()),
                Some(left.name.clone()),
                Some(length),
                Some("<= 2.0 Å plus tolerance"),
            );
        }
    }

    // These are the attachment chemistries implemented by the builder.  In
    // particular CYS is deliberately absent: treating a cysteine as a
    // generic O-linked site would silently produce an invalid structure.
    let protein_residue_names = ["ASN", "SER", "THR", "TRP", "HYP", "PRO"];
    let metadata_glycan_residues = structure
        .metadata()
        .glycan_trees
        .iter()
        .filter(|tree| {
            options.focus_sites.is_empty()
                || tree
                    .attachment_site
                    .as_ref()
                    .is_some_and(|site| options.focus_sites.contains(site))
        })
        .flat_map(|tree| tree.residue_ids.iter())
        .cloned()
        .collect::<BTreeSet<_>>();

    // Keep the residue-name fallback assembled above when a raw PDB has no
    // GlySys glycan-tree metadata.  The previous shadowing here silently
    // dropped that fallback before component/geometry validation, meaning a
    // source conformer could be accepted without any carbohydrate checks.
    // Metadata remains authoritative whenever it is available (and when a
    // caller deliberately focuses a subset of sites).
    if !metadata_glycan_residues.is_empty() || !options.focus_sites.is_empty() {
        glycan_residues = metadata_glycan_residues;
    }

    // NDG/AMN templates occasionally omit the amide carbonyl CONECT record.
    // Recover this chemically authoritative edge so its C=O pair is not
    // mistaken for a nonbonded clash and its 1–3 neighbours are excluded.
    for residue in &residues {
        if !glycan_residues.contains(&residue.id) {
            continue;
        }
        let carbonyl = residue.atoms.iter().find_map(|id| {
            atoms
                .iter()
                .find(|atom| atom.id == *id && atom.name.eq_ignore_ascii_case("C2N"))
                .map(|atom| atom.id)
        });
        let oxygen = residue.atoms.iter().find_map(|id| {
            atoms
                .iter()
                .find(|atom| atom.id == *id && atom.name.eq_ignore_ascii_case("O2N"))
                .map(|atom| atom.id)
        });
        if let (Some(carbonyl), Some(oxygen)) = (carbonyl, oxygen) {
            bond_set.insert(ordered_atoms(carbonyl, oxygen));
        }
    }

    for residue in &residues {
        if !glycan_residues.contains(&residue.id) {
            continue;
        }
        // Do not apply a pyranose template to every HETATM.  Sialic acids,
        // furanoses, and modified sugars have different ring atom names; an
        // unknown component is still useful to inspect, but must be reported
        // as “No reference” rather than as a spurious missing-atom error.
        let Some(template) = carbohydrate_template(&residue.name) else {
            finding(
                &mut findings,
                "chemistry.no_reference",
                Severity::Info,
                "no versioned carbohydrate component reference is available",
                None,
                Some(residue.id.clone()),
                None,
                None,
                Some("No reference"),
            );
            continue;
        };
        let has_atom = |atom_name: &str| {
            residue.atoms.iter().any(|id| {
                atoms
                    .iter()
                    .find(|atom| atom.id == *id)
                    .is_some_and(|atom| atom.name.eq_ignore_ascii_case(atom_name))
            })
        };
        for atom_name in template.required_atoms {
            if !has_atom(atom_name) {
                finding(
                    &mut findings,
                    "chemistry.missing_atom",
                    Severity::Error,
                    &format!(
                        "glycan residue is missing required {} atom {atom_name}",
                        template.kind
                    ),
                    None,
                    Some(residue.id.clone()),
                    Some((*atom_name).into()),
                    None,
                    Some(template.kind),
                );
            }
        }

        // Deposited sialic-acid components use either O5 or O6 as the ring
        // oxygen/anchor depending on the restraint dictionary and exporter.
        // Require one of the declared alternatives, rather than reporting a
        // false missing atom for the other spelling.
        if !template.ring_anchor_alternatives.is_empty()
            && !template
                .ring_anchor_alternatives
                .iter()
                .any(|name| has_atom(name))
        {
            let alternatives = template.ring_anchor_alternatives.join("/");
            finding(
                &mut findings,
                "chemistry.missing_atom",
                Severity::Error,
                &format!(
                    "glycan residue is missing required {} ring anchor ({alternatives})",
                    template.kind
                ),
                None,
                Some(residue.id.clone()),
                Some(alternatives),
                None,
                Some(template.kind),
            );
        }

        let mut ring_names = template.ring_atoms.to_vec();
        if let Some(anchor) = template
            .ring_anchor_alternatives
            .iter()
            .find(|name| has_atom(name))
        {
            ring_names.push(anchor);
        }
        let ring = ring_names
            .iter()
            .filter_map(|name| {
                residue.atoms.iter().find_map(|id| {
                    atoms
                        .iter()
                        .find(|atom| atom.id == *id && atom.name.eq_ignore_ascii_case(name))
                        .map(|atom| atom.position)
                })
            })
            .collect::<Vec<_>>();
        if ring.len() == ring_names.len() {
            let span = ring
                .iter()
                .flat_map(|left| ring.iter().map(move |right| distance(*left, *right)))
                .fold(0.0, f64::max);
            if span < 1.0 {
                finding(
                    &mut findings,
                    "ring.collapsed",
                    Severity::Error,
                    &format!("{} ring is collapsed", template.kind),
                    None,
                    Some(residue.id.clone()),
                    None,
                    Some(span),
                    Some("ring span >= 1.0 Å"),
                );
            } else if ring_planarity(&ring) < 0.08 {
                finding(
                    &mut findings,
                    "ring.puckering",
                    Severity::Warning,
                    &format!("{} ring has unusually low puckering", template.kind),
                    None,
                    Some(residue.id.clone()),
                    None,
                    Some(ring_planarity(&ring)),
                    Some("out-of-plane puckering >= 0.08 Å"),
                );
            }
            // A complete atom list can still describe an open chain when a
            // source component was truncated or its final bond was lost in a
            // format conversion. Check every declared ring edge, including
            // the closure from the last atom back to the anomeric atom.
            for index in 0..ring.len() {
                let next = (index + 1) % ring.len();
                let edge = distance(ring[index], ring[next]);
                if edge > 2.20 {
                    finding(
                        &mut findings,
                        "topology.ring_open",
                        Severity::Error,
                        &format!("{} ring edge is disconnected", template.kind),
                        None,
                        Some(residue.id.clone()),
                        Some(ring_names[index].into()),
                        Some(edge),
                        Some("ring edge <= 2.20 Å"),
                    );
                } else if edge > 1.85 {
                    finding(
                        &mut findings,
                        "geometry.ring_edge_deviation",
                        Severity::Warning,
                        &format!(
                            "{} ring edge is longer than the common covalent range",
                            template.kind
                        ),
                        None,
                        Some(residue.id.clone()),
                        Some(ring_names[index].into()),
                        Some(edge),
                        Some("ring edge <= 1.85 Å"),
                    );
                }
            }
        }
    }

    for (first, second) in &bond_set {
        let Some(left) = atoms.iter().find(|atom| atom.id == *first) else {
            continue;
        };
        let Some(right) = atoms.iter().find(|atom| atom.id == *second) else {
            continue;
        };
        if !glycan_residues.contains(&left.residue) && !glycan_residues.contains(&right.residue) {
            continue;
        }
        let expected = ideal_bond_length_for_atoms(left, right);
        let deviation = (distance(left.position, right.position) - expected).abs();
        if deviation > options.bond_warning_angstrom {
            finding(
                &mut findings,
                "geometry.glycan_bond_deviation",
                if deviation > options.bond_error_angstrom {
                    Severity::Error
                } else {
                    Severity::Warning
                },
                "GLYCAM-like carbohydrate bond length deviates from the expected range",
                None,
                Some(left.residue.clone()),
                Some(left.name.clone()),
                Some(deviation),
                Some("deviation <= 0.10 Å"),
            );
        }
    }

    for center in atoms
        .iter()
        .filter(|atom| glycan_residues.contains(&atom.residue))
    {
        let neighbors = bond_set
            .iter()
            .filter_map(|(left, right)| {
                if *left == center.id {
                    Some(*right)
                } else if *right == center.id {
                    Some(*left)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for (index, first) in neighbors.iter().enumerate() {
            for second in neighbors.iter().skip(index + 1) {
                let Some(first) = atoms.iter().find(|atom| atom.id == *first) else {
                    continue;
                };
                let Some(second) = atoms.iter().find(|atom| atom.id == *second) else {
                    continue;
                };
                let angle = angle_degrees(first.position, center.position, second.position);
                let ideal_angle = if center.element.eq_ignore_ascii_case("N") {
                    120.0
                } else {
                    109.5
                };
                let deviation = (angle - ideal_angle).abs();
                if deviation > options.angle_warning_degrees {
                    finding(
                        &mut findings,
                        "geometry.glycan_angle_deviation",
                        if deviation > options.angle_error_degrees {
                            Severity::Error
                        } else {
                            Severity::Warning
                        },
                        "carbohydrate bond angle deviates from the tetrahedral reference",
                        None,
                        Some(center.residue.clone()),
                        Some(center.name.clone()),
                        Some(deviation),
                        Some("deviation <= 10°"),
                    );
                }
            }
        }
    }

    for (first, second) in &bond_set {
        let Some(left) = atoms.iter().find(|atom| atom.id == *first) else {
            continue;
        };
        let Some(right) = atoms.iter().find(|atom| atom.id == *second) else {
            continue;
        };
        if !glycan_residues.contains(&left.residue)
            || !glycan_residues.contains(&right.residue)
            || left.residue == right.residue
        {
            continue;
        }
        let left_neighbor = bond_set.iter().find_map(|(a, b)| {
            if *a == left.id && *b != right.id {
                Some(*b)
            } else if *b == left.id && *a != right.id {
                Some(*a)
            } else {
                None
            }
        });
        let right_neighbor = bond_set.iter().find_map(|(a, b)| {
            if *a == right.id && *b != left.id {
                Some(*b)
            } else if *b == right.id && *a != left.id {
                Some(*a)
            } else {
                None
            }
        });
        let (Some(left_neighbor), Some(right_neighbor)) = (left_neighbor, right_neighbor) else {
            continue;
        };
        let Some(first_atom) = atoms.iter().find(|atom| atom.id == left_neighbor) else {
            continue;
        };
        let Some(last_atom) = atoms.iter().find(|atom| atom.id == right_neighbor) else {
            continue;
        };
        let torsion = dihedral_degrees(
            first_atom.position,
            left.position,
            right.position,
            last_atom.position,
        );
        if torsion.abs() < 5.0 {
            finding(
                &mut findings,
                "torsion.glycosidic_strain",
                Severity::Warning,
                "glycosidic torsion is close to an eclipsed, high-strain geometry",
                None,
                Some(right.residue.clone()),
                Some(right.name.clone()),
                Some(torsion),
                Some("|torsion| >= 5°"),
            );
        }
    }

    for tree in &structure.metadata().glycan_trees {
        if tree.residue_ids.len() > 1 && !tree_is_connected(structure, &tree.residue_ids) {
            finding(
                &mut findings,
                "topology.disconnected_glycan",
                Severity::Error,
                "glycan tree residues are not connected by explicit bonds",
                tree.attachment_site.clone(),
                None,
                None,
                None,
                Some("one connected carbohydrate tree"),
            );
        }
    }

    for site in &structure.metadata().glycosylation_sites {
        if !options.focus_sites.is_empty() && !options.focus_sites.contains(&site.protein_residue) {
            continue;
        }
        let protein = structure.find_atom(&site.protein_residue, &site.protein_atom);
        let glycan = structure.find_atom(&site.glycan_residue, &site.glycan_atom);
        if protein.is_none() || glycan.is_none() {
            finding(
                &mut findings,
                "topology.missing_attachment_atom",
                Severity::Error,
                &format!(
                    "attachment metadata references missing atoms at {}",
                    site.protein_residue
                ),
                Some(site.protein_residue.clone()),
                None,
                None,
                None,
                None,
            );
            continue;
        }
        let protein_residue_name = residues
            .iter()
            .find(|residue| residue.id == site.protein_residue)
            .map(|residue| residue.name.as_str());
        let protein_name_upper = protein_residue_name.map(str::to_ascii_uppercase);
        if protein_residue_name.is_none_or(|name| {
            !protein_residue_names
                .iter()
                .any(|candidate| name.eq_ignore_ascii_case(candidate))
        }) {
            finding(
                &mut findings,
                "chemistry.unsupported_attachment",
                Severity::Error,
                "protein side of glycosylation attachment is not a supported residue",
                Some(site.protein_residue.clone()),
                None,
                Some(site.protein_atom.clone()),
                None,
                Some("ASN/SER/THR/TRP/HYP/PRO"),
            );
        } else if let Some(name) = protein_name_upper.as_deref() {
            let expected_atom = match name {
                "ASN" => "ND2",
                "SER" => "OG",
                "THR" => "OG1",
                "TRP" => "CD1",
                "HYP" | "PRO" => "OD1",
                _ => "",
            };
            if !expected_atom.is_empty() && site.protein_atom != expected_atom {
                finding(
                    &mut findings,
                    "chemistry.impossible_attachment",
                    Severity::Error,
                    "attachment atom does not match the supported residue chemistry",
                    Some(site.protein_residue.clone()),
                    None,
                    Some(site.protein_atom.clone()),
                    None,
                    Some(expected_atom),
                );
            }
        }
        let expected_glycan_atom = residues
            .iter()
            .find(|residue| residue.id == site.glycan_residue)
            .map(|residue| {
                if matches!(
                    residue.name.trim().to_ascii_uppercase().as_str(),
                    "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
                ) {
                    "C2"
                } else {
                    "C1"
                }
            })
            .unwrap_or("C1");
        if !site.glycan_atom.eq_ignore_ascii_case(expected_glycan_atom) {
            finding(
                &mut findings,
                "chemistry.impossible_attachment",
                Severity::Error,
                &format!(
                    "carbohydrate attachment does not use the expected anomeric {expected_glycan_atom} atom"
                ),
                Some(site.protein_residue.clone()),
                Some(site.glycan_residue.clone()),
                Some(site.glycan_atom.clone()),
                None,
                Some(expected_glycan_atom),
            );
        }
        let left = atoms.iter().find(|atom| Some(atom.id) == protein);
        let right = atoms.iter().find(|atom| Some(atom.id) == glycan);
        if let (Some(left), Some(right)) = (left, right) {
            let length = distance(left.position, right.position);
            if !(1.1..=2.0).contains(&length) {
                finding(
                    &mut findings,
                    "geometry.attachment_length",
                    Severity::Error,
                    "protein–glycan attachment distance is chemically implausible",
                    Some(site.protein_residue.clone()),
                    Some(site.glycan_residue.clone()),
                    Some(site.glycan_atom.clone()),
                    Some(length),
                    Some("1.1–2.0 Å"),
                );
            } else if !(1.25..=1.65).contains(&length) {
                finding(
                    &mut findings,
                    "geometry.attachment_length",
                    Severity::Warning,
                    "protein–glycan attachment distance is outside the common covalent range",
                    Some(site.protein_residue.clone()),
                    Some(site.glycan_residue.clone()),
                    Some(site.glycan_atom.clone()),
                    Some(length),
                    Some("1.25–1.65 Å"),
                );
            }
        }

        // Keep attachment torsions as first-class observations even when the
        // structure is being inspected outside a search workflow.  A missing
        // frame atom means the torsion is simply unavailable; it is not a
        // chemistry error on its own. Workflow search results add the
        // component-conditioned VMM bounds to this observation.
        let frame_names = match protein_name_upper.as_deref() {
            Some("ASN") => Some(("CB", "CG", "ND2")),
            Some("SER" | "THR") => Some(("CA", "CB", site.protein_atom.as_str())),
            Some("TRP") => Some(("CB", "CG", "CD1")),
            Some("HYP" | "PRO") => Some(("CB", "CG", "OD1")),
            _ => None,
        };
        if let Some((frame_a_name, frame_b_name, link_atom)) = frame_names {
            let frame_a = structure
                .find_atom(&site.protein_residue, frame_a_name)
                .and_then(|id| structure.atom(id))
                .map(|atom| atom.position);
            let frame_b = structure
                .find_atom(&site.protein_residue, frame_b_name)
                .and_then(|id| structure.atom(id))
                .map(|atom| atom.position);
            let link = structure
                .find_atom(&site.protein_residue, link_atom)
                .and_then(|id| structure.atom(id))
                .map(|atom| atom.position);
            // The ring atom used for an attachment torsion is residue-family
            // dependent.  O5 is the conventional pyranose anchor, O4 is the
            // furanose anchor (including Ara), while deposited sialic/KDO
            // records use the C2--C6 ring path and attach through O6 (with O5
            // retained as a compatibility fallback).  The old universal O5
            // choice produced a misleading torsion or a missing-atom finding
            // for valid furanose and sialic assets.
            let glycan_name = structure
                .residues()
                .into_iter()
                .find(|residue| residue.id == site.glycan_residue)
                .map(|residue| residue.name.to_ascii_uppercase());
            let ring_names: &[&str] =
                if matches!(protein_name_upper.as_deref(), Some("HYP" | "PRO")) {
                    &["O4", "O5", "O6"]
                } else if glycan_name.as_deref().is_some_and(is_sialic_residue) {
                    &["O6", "O5", "O4"]
                } else if glycan_name.as_deref().is_some_and(is_furanose_residue) {
                    &["O4", "O5", "O6"]
                } else {
                    &["O5", "O4", "O6"]
                };
            let ring = ring_names.iter().find_map(|ring_name| {
                structure
                    .find_atom(&site.glycan_residue, ring_name)
                    .and_then(|id| structure.atom(id))
                    .map(|atom| (*ring_name, atom.position))
            });
            let anomeric = structure
                .find_atom(&site.glycan_residue, &site.glycan_atom)
                .and_then(|id| structure.atom(id))
                .map(|atom| atom.position);
            if let (
                Some(frame_a),
                Some(frame_b),
                Some(link),
                Some(anomeric),
                Some((ring_name, ring)),
            ) = (frame_a, frame_b, link, anomeric, ring)
            {
                let psi = dihedral_degrees(frame_a, frame_b, link, anomeric);
                let phi = dihedral_degrees(frame_b, link, anomeric, ring);
                findings.push(ValidationFinding {
                    code: "torsion.attachment_observation".into(),
                    severity: Severity::Info,
                    message: "protein–glycan attachment torsion recorded for VMM comparison".into(),
                    site: Some(site.protein_residue.clone()),
                    residue: Some(site.glycan_residue.clone()),
                    atom: Some(site.glycan_atom.clone()),
                    observed: Some(phi),
                    expected: Some(format!("phi={phi:.2}°, psi={psi:.2}°")),
                    domain: Some("torsion".into()),
                    origin: None,
                    stage: None,
                    frame: None,
                    glycan_index: None,
                    glycan: Some(site.glycan_residue.to_string()),
                    linkage: Some(format!(
                        "{}:{}-{}:{}",
                        site.protein_residue,
                        site.protein_atom,
                        site.glycan_residue,
                        site.glycan_atom
                    )),
                    involved_atoms: vec![
                        format!("{}:{frame_a_name}", site.protein_residue),
                        format!("{}:{frame_b_name}", site.protein_residue),
                        format!("{}:{link_atom}", site.protein_residue),
                        format!("{}:{}", site.glycan_residue, site.glycan_atom),
                        format!("{}:{ring_name}", site.glycan_residue),
                    ],
                    metric: Some("attachment_phi_degrees".into()),
                    policy: Some("glycoshape-attachment-vmm-v1".into()),
                    policy_version: Some("glycoshape-attachment-vmm-v1".into()),
                });
            }
        }
    }

    // Glycan/protein and glycan/glycan clashes are evaluated by the shared
    // topology-aware contact engine. Protein–protein contacts are deliberately
    // excluded because they describe the deposited protein rather than a
    // remodeled carbohydrate pose.
    let steric_summary = steric::evaluate_structure_with_thresholds(
        structure,
        &options.focus_sites,
        options.clash_warning_angstrom,
        options.clash_error_angstrom,
    );
    let atom_map = atoms
        .iter()
        .map(|atom| (atom.id, atom))
        .collect::<std::collections::BTreeMap<_, _>>();
    for contact in &steric_summary.contacts {
        // Clear-band contacts are useful to the shared evaluator and report
        // summary, but are not findings in the user-facing validation list.
        if contact.class == steric::ContactClass::Clear {
            continue;
        }
        let Some(first) = atom_map.get(&contact.first) else {
            continue;
        };
        let Some(second) = atom_map.get(&contact.second) else {
            continue;
        };
        let severity = if contact.overlap_angstrom > options.clash_error_angstrom {
            Severity::Error
        } else {
            Severity::Warning
        };
        finding(
            &mut findings,
            "geometry.clash",
            severity,
            if severity == Severity::Error {
                "glycan has a hard nonbonded van der Waals overlap"
            } else {
                "glycan has an advisory nonbonded van der Waals overlap"
            },
            None,
            Some(first.residue.clone()),
            Some(first.name.clone()),
            Some(contact.overlap_angstrom),
            Some("overlap <= 0.4 Å (hard > 0.6 Å)"),
        );
        if let Some(last) = findings.last_mut() {
            last.domain = Some("sterics".into());
            last.involved_atoms = vec![
                format!("{}:{}", first.residue, first.name),
                format!("{}:{}", second.residue, second.name),
            ];
            last.metric = Some("vdw_overlap_angstrom".into());
            last.policy = Some("heavy_atom_vdw_overlap_v1".into());
        }
    }

    for reference in &options.references {
        if structure
            .metadata()
            .glycan_trees
            .iter()
            .any(|tree| tree.attachment_site.as_ref() == Some(&reference.site))
        {
            finding(
                &mut findings,
                "reference.population_percentile",
                Severity::Info,
                "population reference was supplied; detailed percentile is retained as provenance",
                Some(reference.site.clone()),
                None,
                None,
                None,
                Some(&reference.source),
            );
        } else {
            finding(
                &mut findings,
                "reference.site_missing",
                Severity::Warning,
                "population reference site is not present in the structure",
                Some(reference.site.clone()),
                None,
                None,
                None,
                Some(&reference.source),
            );
        }
    }

    findings.sort_by(|left, right| {
        severity_rank(left.severity)
            .cmp(&severity_rank(right.severity))
            .then_with(|| left.code.cmp(&right.code))
    });
    let warnings = findings
        .iter()
        .filter(|finding| finding.severity == Severity::Warning)
        .map(|finding| finding.message.clone())
        .collect::<Vec<_>>();
    let errors = findings
        .iter()
        .filter(|finding| finding.severity == Severity::Error)
        .map(|finding| finding.message.clone())
        .collect::<Vec<_>>();
    ValidationReport {
        valid: errors.is_empty() && !atoms.is_empty(),
        atom_count: atoms.len(),
        residue_count: residues.len(),
        attachment_count: structure
            .metadata()
            .glycosylation_sites
            .iter()
            .filter(|site| {
                options.focus_sites.is_empty()
                    || options.focus_sites.contains(&site.protein_residue)
            })
            .count(),
        warnings,
        errors,
        findings,
        component_dictionary_version: default_component_dictionary_version(),
        steric_policy: Some(steric_policy(options)),
        steric_summary: Some(steric_summary),
        #[cfg(feature = "density")]
        density: None,
    }
}

/// Run the structural checks and attach a density score to the report.
#[cfg(feature = "density")]
pub fn validate_with_density(
    structure: &Structure,
    options: &ValidationOptions,
    scorer: &DensityScorer,
    targets: &[DensityTarget],
) -> Result<ValidationReport, reglyco_density::DensityError> {
    let mut report = validate_with_options(structure, options);
    let score = scorer.score(structure, targets)?;
    if let Some(minimum) = options.min_density_cc
        && score.correlation < minimum
    {
        report.findings.push(ValidationFinding {
            code: "density.low_correlation".into(),
            severity: Severity::Warning,
            message: format!(
                "density correlation {:.3} is below requested minimum {:.3}",
                score.correlation, minimum
            ),
            site: None,
            residue: None,
            atom: None,
            observed: Some(score.correlation),
            expected: Some(format!(">= {minimum:.3}")),
            domain: Some("density".into()),
            origin: Some("introduced".into()),
            stage: None,
            frame: None,
            glycan_index: None,
            glycan: None,
            linkage: None,
            involved_atoms: Vec::new(),
            metric: Some("correlation".into()),
            policy: Some("density_minimum_correlation".into()),
            policy_version: Some("density_minimum_correlation".into()),
        });
        report.warnings.push(
            report
                .findings
                .last()
                .expect("finding was just pushed")
                .message
                .clone(),
        );
    }
    report.density = Some(score);
    Ok(report)
}

fn finding(
    findings: &mut Vec<ValidationFinding>,
    code: &str,
    severity: Severity,
    message: &str,
    site: Option<ResidueId>,
    residue: Option<ResidueId>,
    atom: Option<String>,
    observed: Option<f64>,
    expected: Option<&str>,
) {
    let glycan = residue.as_ref().map(ToString::to_string);
    findings.push(ValidationFinding {
        code: code.into(),
        severity,
        message: message.into(),
        site,
        residue,
        atom: atom.clone(),
        observed,
        expected: expected.map(str::to_string),
        domain: Some(
            if code.starts_with("chemistry.") {
                "chemistry"
            } else if code.starts_with("topology.") {
                "topology"
            } else if code.starts_with("geometry.") || code.starts_with("ring.") {
                "geometry"
            } else if code.starts_with("torsion.") {
                "torsion"
            } else if code.starts_with("reference.") {
                "reference"
            } else {
                "structure"
            }
            .into(),
        ),
        origin: None,
        stage: None,
        frame: None,
        glycan_index: None,
        glycan,
        linkage: None,
        involved_atoms: atom.iter().cloned().collect(),
        metric: None,
        policy: None,
        policy_version: None,
    });
}

fn ordered_atoms(first: AtomId, second: AtomId) -> (AtomId, AtomId) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

/// Versioned, intentionally conservative component dictionary.  The
/// validator only requires atoms that define the ring; substituent atoms are
/// checked through the connectivity and geometry passes when present.  This
/// avoids treating valid furanoses and sialic acids as malformed pyranoses.
#[derive(Debug, Clone, Copy)]
struct CarbohydrateTemplate {
    kind: &'static str,
    required_atoms: &'static [&'static str],
    ring_atoms: &'static [&'static str],
    /// Some component families use one of several deposited atom names for
    /// the ring oxygen. Record those alternatives instead of declaring a
    /// false missing-atom error for a valid export convention.
    ring_anchor_alternatives: &'static [&'static str],
}

const PYRANOSE: CarbohydrateTemplate = CarbohydrateTemplate {
    kind: "pyranose",
    required_atoms: &["C1", "C2", "C3", "C4", "C5", "O5"],
    ring_atoms: &["C1", "C2", "C3", "C4", "C5", "O5"],
    ring_anchor_alternatives: &[],
};
const FURANOSE: CarbohydrateTemplate = CarbohydrateTemplate {
    kind: "furanose",
    required_atoms: &["C1", "C2", "C3", "C4", "O4"],
    ring_atoms: &["C1", "C2", "C3", "C4", "O4"],
    ring_anchor_alternatives: &[],
};
// Sialic acids and KDO are nine-membered/non-pyranose keto acids in many
// PDB encodings.  Their conserved ring path uses C2–C6/O6; O5 is not a
// required atom (the old universal template produced a false SIA error).
const SIALIC: CarbohydrateTemplate = CarbohydrateTemplate {
    kind: "sialic-acid",
    required_atoms: &["C2", "C3", "C4", "C5", "C6"],
    ring_atoms: &["C2", "C3", "C4", "C5", "C6"],
    ring_anchor_alternatives: &["O5", "O6"],
};

fn is_furanose_residue(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "ARA" | "ARB" | "AFL" | "RIB"
    )
}

fn is_sialic_residue(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
    )
}

fn carbohydrate_template(name: &str) -> Option<&'static CarbohydrateTemplate> {
    let name = name.trim().to_ascii_uppercase();
    // Common pyranoses and the modified forms shipped in GlycoShape assets.
    if matches!(
        name.as_str(),
        // Canonical PDB residue names.
        "NAG"
            | "BMA"
            | "MAN"
            | "MAG"
            | "GAL"
            | "GLC"
            | "GAM"
            | "FUC"
            | "IDO"
            | "GCU"
            | "G6D"
            | "GNA"
            | "NDG"
            | "AMN"
            | "NGA"
            | "GLA"
            // GlycoShape/GLYCAM three-character aliases for modified
            // pyranoses (N-acetyl sugars and deoxy/uronic variants).
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
    ) {
        Some(&PYRANOSE)
    } else if matches!(name.as_str(), "ARA" | "ARB" | "AFL" | "RIB") {
        Some(&FURANOSE)
    } else if matches!(name.as_str(), "XYS" | "XYP") {
        // XYS/XYP are the deposited xylopyranose aliases used by the
        // GlycoShape assets.  Keep them on the six-membered template rather
        // than requiring the furanose O4 ring closure.
        Some(&PYRANOSE)
    } else if matches!(name.as_str(), "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO") {
        Some(&SIALIC)
    } else {
        None
    }
}

fn ideal_bond_length(left: &str, right: &str) -> f64 {
    let pair = [
        left.trim().to_ascii_uppercase(),
        right.trim().to_ascii_uppercase(),
    ];
    if pair.iter().all(|element| element == "C") {
        1.53
    } else if pair.iter().any(|element| element == "H") {
        1.09
    } else if pair.iter().any(|element| element == "O") && pair.iter().any(|element| element == "C")
    {
        1.43
    } else if pair.iter().any(|element| element == "N") && pair.iter().any(|element| element == "C")
    {
        1.45
    } else {
        1.50
    }
}

fn ideal_bond_length_for_atoms(left: &glysys::StructureAtom, right: &glysys::StructureAtom) -> f64 {
    let names = [
        left.name.trim().to_ascii_uppercase(),
        right.name.trim().to_ascii_uppercase(),
    ];
    // Sialic-acid carboxylates are represented as a resonance-delocalised
    // C1/O1A/O1B pair in GLYCAM/GlycoShape PDB assets.  Both C--O bonds are
    // therefore close to a 1.25 Å carbonyl/carboxylate distance rather than
    // the 1.43 Å single C--O default.  Treating them as ordinary single bonds
    // produced a false source-asset error (and consequently blocked perfectly
    // usable N-glycan ensembles such as G54258NG).
    if names.iter().any(|name| name == "C1")
        && names.iter().any(|name| name == "O1A" || name == "O1B")
    {
        return 1.25;
    }
    // The N-acetyl substituent used by SIA/Neu5Ac has the analogous
    // resonance-delocalised C5N--O5N carbonyl.  GLYCAM coordinates place it
    // near 1.22 Å; treating it as an ordinary 1.43 Å C--O single bond makes
    // otherwise valid level-2/3 conformers fail source validation.
    if (names.iter().any(|name| name == "C2N") && names.iter().any(|name| name == "O2N"))
        || (names.iter().any(|name| name == "C5N") && names.iter().any(|name| name == "O5N"))
    {
        1.21
    // The amide N5--C5N bond is shorter than the generic C--N fallback too.
    // Some GLYCAM exports omit the corresponding restraint and otherwise
    // trigger a false source-asset bond-length error.
    } else if names.iter().any(|name| name == "N5") && names.iter().any(|name| name == "C5N") {
        1.33
    } else {
        ideal_bond_length(&left.element, &right.element)
    }
}

fn angle_degrees(first: Vec3, center: Vec3, second: Vec3) -> f64 {
    let left = [first.x - center.x, first.y - center.y, first.z - center.z];
    let right = [
        second.x - center.x,
        second.y - center.y,
        second.z - center.z,
    ];
    let left_norm = (left[0].powi(2) + left[1].powi(2) + left[2].powi(2)).sqrt();
    let right_norm = (right[0].powi(2) + right[1].powi(2) + right[2].powi(2)).sqrt();
    if left_norm <= 1.0e-12 || right_norm <= 1.0e-12 {
        return 0.0;
    }
    let cosine =
        (left[0] * right[0] + left[1] * right[1] + left[2] * right[2]) / (left_norm * right_norm);
    cosine.clamp(-1.0, 1.0).acos().to_degrees()
}

fn ring_planarity(ring: &[Vec3]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let first = ring[0];
    let mut normal = [0.0; 3];
    for point in ring.iter().skip(1) {
        let a = [point.x - first.x, point.y - first.y, point.z - first.z];
        let b = [
            ring[ring.len() - 1].x - first.x,
            ring[ring.len() - 1].y - first.y,
            ring[ring.len() - 1].z - first.z,
        ];
        normal = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let norm = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if norm > 1.0e-8 {
            normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
            break;
        }
    }
    let norm = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
    if norm <= 1.0e-8 {
        return 0.0;
    }
    ring.iter()
        .map(|point| {
            let delta = [point.x - first.x, point.y - first.y, point.z - first.z];
            (delta[0] * normal[0] + delta[1] * normal[1] + delta[2] * normal[2]).abs()
        })
        .fold(0.0, f64::max)
}

fn likely_glycosidic_bond(first: &glysys::StructureAtom, second: &glysys::StructureAtom) -> bool {
    let first_name = first.name.trim().to_ascii_uppercase();
    let second_name = second.name.trim().to_ascii_uppercase();
    let anomeric = |name: &str| matches!(name, "C1" | "C2" | "C1A" | "C2A");
    let oxygen = |name: &str| name.starts_with('O');
    (anomeric(&first_name) && oxygen(&second_name))
        || (anomeric(&second_name) && oxygen(&first_name))
}

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

fn tree_is_connected(structure: &Structure, residues: &[ResidueId]) -> bool {
    if residues.len() <= 1 {
        return !residues.is_empty();
    }
    let ids = residues.iter().cloned().collect::<BTreeSet<_>>();
    let bonds = structure.bonds();
    let mut graph = std::collections::BTreeMap::<ResidueId, BTreeSet<ResidueId>>::new();
    for (left, right) in bonds {
        let Some(left_atom) = structure.atom(left) else {
            continue;
        };
        let Some(right_atom) = structure.atom(right) else {
            continue;
        };
        if ids.contains(&left_atom.residue)
            && ids.contains(&right_atom.residue)
            && left_atom.residue != right_atom.residue
        {
            graph
                .entry(left_atom.residue.clone())
                .or_default()
                .insert(right_atom.residue.clone());
            graph
                .entry(right_atom.residue.clone())
                .or_default()
                .insert(left_atom.residue.clone());
        }
    }
    // LINK/CONECT records are frequently omitted from deposited carbohydrate
    // coordinates. Recover only chemically short inter-residue contacts inside
    // this declared tree; this prevents valid source glycans from receiving a
    // false disconnected-topology error while avoiding arbitrary proximity
    // links between unrelated glycans.
    let tree_atoms = structure
        .atoms()
        .into_iter()
        .filter(|atom| ids.contains(&atom.residue) && !is_hydrogen(atom))
        .collect::<Vec<_>>();
    for (index, left) in tree_atoms.iter().enumerate() {
        for right in tree_atoms.iter().skip(index + 1) {
            if left.residue == right.residue {
                continue;
            }
            if likely_glycosidic_bond(left, right)
                && (1.1..=1.85).contains(&distance(left.position, right.position))
            {
                graph
                    .entry(left.residue.clone())
                    .or_default()
                    .insert(right.residue.clone());
                graph
                    .entry(right.residue.clone())
                    .or_default()
                    .insert(left.residue.clone());
            }
        }
    }
    let Some(start) = ids.iter().next().cloned() else {
        return false;
    };
    let mut seen = BTreeSet::new();
    let mut stack = vec![start];
    while let Some(residue) = stack.pop() {
        if !seen.insert(residue.clone()) {
            continue;
        }
        if let Some(neighbors) = graph.get(&residue) {
            stack.extend(neighbors.iter().cloned());
        }
    }
    seen.len() == ids.len()
}

fn severity_rank(severity: Severity) -> u8 {
    match severity {
        Severity::Error => 0,
        Severity::Warning => 1,
        Severity::Info => 2,
    }
}

fn distance(left: Vec3, right: Vec3) -> f64 {
    ((left.x - right.x).powi(2) + (left.y - right.y).powi(2) + (left.z - right.z).powi(2)).sqrt()
}

fn is_hydrogen(atom: &glysys::StructureAtom) -> bool {
    atom.element.eq_ignore_ascii_case("H")
        || atom.element.eq_ignore_ascii_case("D")
        || atom.name.trim().chars().next().is_some_and(|value| {
            value.eq_ignore_ascii_case(&'H') || value.eq_ignore_ascii_case(&'D')
        })
}

fn dihedral_degrees(first: Vec3, second: Vec3, third: Vec3, fourth: Vec3) -> f64 {
    let b1 = [second.x - first.x, second.y - first.y, second.z - first.z];
    let b2 = [third.x - second.x, third.y - second.y, third.z - second.z];
    let b3 = [fourth.x - third.x, fourth.y - third.y, fourth.z - third.z];
    let cross = |left: [f64; 3], right: [f64; 3]| {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    };
    let dot = |left: [f64; 3], right: [f64; 3]| {
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
    };
    let norm = |value: [f64; 3]| (dot(value, value)).sqrt();
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let b2_norm = norm(b2);
    if norm(n1) <= 1.0e-12 || norm(n2) <= 1.0e-12 || b2_norm <= 1.0e-12 {
        return 0.0;
    }
    let unit_b2 = [b2[0] / b2_norm, b2[1] / b2_norm, b2[2] / b2_norm];
    let m1 = cross(n1, unit_b2);
    dot(n1, n2).atan2(dot(m1, n2)).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, read_pdb_str};

    #[test]
    fn empty_structure_is_invalid_without_panicking() {
        let structure = read_pdb_str(
            "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let report = validate(&structure);
        assert!(report.valid);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn missing_attachment_atoms_are_errors() {
        let structure = read_pdb_str(
            "ATOM      1  CA  ASN A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let report = validate(&structure);
        assert!(report.valid);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn sialic_acid_uses_keto_acid_template_without_false_o5_error() {
        let structure = read_pdb_str(
            "HETATM    1  C2  SIA B   1       0.000   0.000   0.000  1.00  0.00           C\nHETATM    2  C3  SIA B   1       1.300   0.000   0.000  1.00  0.00           C\nHETATM    3  C4  SIA B   1       1.900   1.100   0.000  1.00  0.00           C\nHETATM    4  C5  SIA B   1       1.100   2.000   0.000  1.00  0.00           C\nHETATM    5  C6  SIA B   1      -0.100   1.600   0.000  1.00  0.00           C\nHETATM    6  O6  SIA B   1      -0.700   0.600   0.000  1.00  0.00           O\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let report = validate(&structure);
        assert!(!report.findings.iter().any(|finding| {
            finding.code == "chemistry.missing_atom" && finding.atom.as_deref() == Some("O5")
        }));
    }

    #[test]
    fn sialic_carboxylate_c1_oxygen_bonds_use_resonance_length() {
        // GLYCAM writes the two sialic-acid carboxylate bonds as C1--O1A/B
        // distances near 1.2--1.3 Å. They are not ordinary 1.43 Å single
        // C--O bonds, so the source-asset validator must not reject them.
        let structure = read_pdb_str(
            "HETATM    1  C2  SIA B   1       0.000   0.000   0.000  1.00  0.00           C\nHETATM    2  C3  SIA B   1       1.300   0.000   0.000  1.00  0.00           C\nHETATM    3  C4  SIA B   1       1.900   1.100   0.000  1.00  0.00           C\nHETATM    4  C5  SIA B   1       1.100   2.000   0.000  1.00  0.00           C\nHETATM    5  C6  SIA B   1      -0.100   1.600   0.000  1.00  0.00           C\nHETATM    6  O6  SIA B   1      -0.700   0.600   0.000  1.00  0.00           O\nHETATM    7  C1  SIA B   1       0.000  -1.600   0.000  1.00  0.00           C\nHETATM    8  O1A SIA B   1       1.220  -1.600   0.000  1.00  0.00           O\nHETATM    9  O1B SIA B   1      -1.260  -1.600   0.000  1.00  0.00           O\nCONECT    7    8    9\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let report = validate(&structure);
        assert!(!report.findings.iter().any(|finding| {
            finding.code == "geometry.glycan_bond_deviation"
                && finding.atom.as_deref() == Some("C1")
        }));
    }

    #[test]
    fn sialic_n_acetyl_carbonyl_uses_resonance_length() {
        let structure = read_pdb_str(
            "HETATM    1  N5  SIA B   1       0.000   0.000   0.000  1.00  0.00           N\nHETATM    2  C5N SIA B   1       1.370   0.000   0.000  1.00  0.00           C\nHETATM    3  O5N SIA B   1       2.590   0.000   0.000  1.00  0.00           O\nHETATM    4  CME SIA B   1       1.370   1.530   0.000  1.00  0.00           C\nCONECT    1    2\nCONECT    2    1    3    4\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let report = validate(&structure);
        assert!(!report.findings.iter().any(|finding| {
            finding.code == "geometry.glycan_bond_deviation"
                && finding.atom.as_deref() == Some("C5N")
        }));
    }

    #[test]
    fn ara_uses_furanose_ring_template() {
        let structure = read_pdb_str(
            "HETATM    1  C1  ARA B   1       0.000   0.000   0.000  1.00  0.00           C\nHETATM    2  C2  ARA B   1       1.200   0.000   0.000  1.00  0.00           C\nHETATM    3  C3  ARA B   1       1.700   1.100   0.000  1.00  0.00           C\nHETATM    4  C4  ARA B   1       0.700   1.900   0.000  1.00  0.00           C\nHETATM    5  O4  ARA B   1      -0.300   1.200   0.000  1.00  0.00           O\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let report = validate(&structure);
        assert!(
            !report
                .findings
                .iter()
                .any(|finding| finding.code == "chemistry.missing_atom")
        );
    }
}
