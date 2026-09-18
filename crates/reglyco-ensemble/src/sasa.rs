//! Cookbook-compatible solvent-accessible surface area analysis.
//!
//! The Cookbook uses `gmx sasa` with a 1.4 Å probe and `-ndots 15`.  This
//! module keeps the same Eisenhaber double-cubic-lattice calculation in Rust
//! so attached ensemble jobs do not need GROMACS, trajectories, or temporary
//! files.

use std::collections::HashMap;

use glysys::{AtomId, ResidueId, Structure, StructureAtom, Vec3};

/// Water-probe radius used by the Cookbook calculation, in Ångström.
pub const COOKBOOK_PROBE_RADIUS_ANGSTROM: f64 = 1.4;

/// Surface dot density requested by the Cookbook calculation.
///
/// GROMACS expands this density to the nearest icosahedral/dodecahedral dot
/// set.  For the requested value 15, that set contains 32 dots.
pub const COOKBOOK_NDOTS: usize = 15;

/// Maximum glycan shielding percentage for a hotspot.
pub const COOKBOOK_HOTSPOT_SHIELDING_PERCENT: f64 = 30.0;

/// Errors raised while calculating or serializing ensemble SASA outputs.
#[derive(Debug, thiserror::Error)]
pub enum SasaError {
    #[error("SASA requires at least one ensemble frame")]
    EmptyFrames,
    #[error("ensemble frames do not contain the same atom identifiers")]
    MismatchedAtoms,
    #[error("ensemble frame is missing atom {0:?}")]
    MissingAtom(AtomId),
    #[error("SASA output is missing protein residue {0}")]
    MissingResidue(ResidueId),
}

pub type Result<T> = std::result::Result<T, SasaError>;

/// Per-residue SASA values averaged over the sampled ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct SasaResidue {
    pub residue: ResidueId,
    /// Glycan shielding as a percentage of bare-protein SASA.
    pub shielding_percent: f64,
    /// Absolute glycosylated-protein SASA in nm², matching GROMACS output.
    pub real_sasa_nm2: f64,
    /// Cookbook occupancy mask: 1 for a residue with non-zero averaged bare
    /// SASA, otherwise 0.
    pub occupancy: f64,
    /// Cookbook binder-design hotspot mask.
    pub hotspot: bool,
}

/// Result of the cookbook-compatible calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct SasaAnalysis {
    pub frame_count: usize,
    pub residues: Vec<SasaResidue>,
}

impl SasaAnalysis {
    /// Serialize the relative shielding map as a protein-only PDB.
    pub fn sasa_pdb_string(&self, template: &Structure) -> Result<String> {
        self.pdb_string(template, PdbValue::Shielding)
    }

    /// Serialize absolute glycosylated-protein SASA as a protein-only PDB.
    pub fn real_sasa_pdb_string(&self, template: &Structure) -> Result<String> {
        self.pdb_string(template, PdbValue::RealSasa)
    }

    /// Serialize the binary binder-design hotspot mask as a protein-only PDB.
    pub fn hotspots_pdb_string(&self, template: &Structure) -> Result<String> {
        self.pdb_string(template, PdbValue::Hotspot)
    }

    fn pdb_string(&self, template: &Structure, value: PdbValue) -> Result<String> {
        let values = self
            .residues
            .iter()
            .map(|residue| (residue.residue.clone(), residue))
            .collect::<HashMap<_, _>>();
        let atoms = template
            .atoms()
            .into_iter()
            .map(|atom| (atom.id, atom))
            .collect::<HashMap<_, _>>();
        let residues = template.residues();
        let protein_residues = residues
            .iter()
            .filter(|residue| is_protein_residue(&residue.name));

        let mut output = String::new();
        let mut last_chain: Option<&str> = None;
        for residue in protein_residues {
            let Some(result) = values.get(&residue.id) else {
                return Err(SasaError::MissingResidue(residue.id.clone()));
            };
            if last_chain.is_some_and(|chain| chain != residue.id.chain) {
                output.push_str("TER\n");
            }
            last_chain = Some(residue.id.chain.as_str());
            for atom_id in &residue.atoms {
                let Some(atom) = atoms.get(atom_id) else {
                    return Err(SasaError::MissingAtom(*atom_id));
                };
                let bfactor = match value {
                    PdbValue::Shielding => result.shielding_percent,
                    PdbValue::RealSasa => result.real_sasa_nm2,
                    PdbValue::Hotspot => result.hotspot.then_some(100.0).unwrap_or(0.0),
                };
                output.push_str(&format_protein_atom(atom, result.occupancy, bfactor));
            }
        }
        if last_chain.is_some() {
            output.push_str("TER\n");
        }
        output.push_str("END\n");
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
enum PdbValue {
    Shielding,
    RealSasa,
    Hotspot,
}

/// Calculate the Cookbook SASA, absolute SASA, and hotspot values from sampled
/// attached structures.  The structures must retain the same atom IDs across
/// frames, as Re-Glyco sampled frames do.
pub fn calculate_sasa<'a, I>(frames: I) -> Result<SasaAnalysis>
where
    I: IntoIterator<Item = &'a Structure>,
{
    let frames = frames.into_iter().collect::<Vec<_>>();
    let Some(template) = frames.first() else {
        return Err(SasaError::EmptyFrames);
    };

    let template_atoms = template.atoms();
    let protein_atoms = template_atoms
        .iter()
        .filter(|atom| is_protein_residue(&atom.residue_name))
        .cloned()
        .collect::<Vec<_>>();
    let protein_residues = template
        .residues()
        .into_iter()
        .filter(|residue| is_protein_residue(&residue.name))
        .map(|residue| residue.id)
        .collect::<Vec<_>>();
    if protein_atoms.is_empty() || protein_residues.is_empty() {
        return Err(SasaError::MismatchedAtoms);
    }

    let residue_indices = protein_residues
        .iter()
        .enumerate()
        .map(|(index, residue)| (residue.clone(), index))
        .collect::<HashMap<_, _>>();
    let protein_atom_ids = protein_atoms.iter().map(|atom| atom.id).collect::<Vec<_>>();
    let mut bare_sasa_sum = vec![0.0; protein_residues.len()];
    let mut glyco_sasa_sum = vec![0.0; protein_residues.len()];

    for frame in &frames {
        let frame_atoms = frame.atoms();
        let frame_by_id = frame_atoms
            .iter()
            .map(|atom| (atom.id, atom))
            .collect::<HashMap<_, _>>();
        if frame_by_id.len() != template_atoms.len()
            || template_atoms
                .iter()
                .any(|atom| !frame_by_id.contains_key(&atom.id))
        {
            return Err(SasaError::MismatchedAtoms);
        }

        let bare_atoms = protein_atoms
            .iter()
            .map(|atom| {
                frame_by_id
                    .get(&atom.id)
                    .copied()
                    .ok_or(SasaError::MissingAtom(atom.id))
            })
            .collect::<Result<Vec<_>>>()?;
        let bare_positions = bare_atoms
            .iter()
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let bare_radii = bare_atoms
            .iter()
            .map(|atom| vdw_radius(&atom.element))
            .collect::<Vec<_>>();
        let bare_areas = double_cubic_lattice_sasa(
            &bare_positions,
            &bare_radii,
            &(0..bare_atoms.len()).collect::<Vec<_>>(),
        );

        let glyco_positions = frame_atoms
            .iter()
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let glyco_radii = frame_atoms
            .iter()
            .map(|atom| vdw_radius(&atom.element))
            .collect::<Vec<_>>();
        let frame_indices = frame_atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| (atom.id, index))
            .collect::<HashMap<_, _>>();
        let glyco_indices = protein_atom_ids
            .iter()
            .map(|atom_id| {
                frame_indices
                    .get(atom_id)
                    .copied()
                    .ok_or(SasaError::MissingAtom(*atom_id))
            })
            .collect::<Result<Vec<_>>>()?;
        let glyco_areas = double_cubic_lattice_sasa(&glyco_positions, &glyco_radii, &glyco_indices);

        let mut bare_by_residue = vec![0.0; protein_residues.len()];
        let mut glyco_by_residue = vec![0.0; protein_residues.len()];
        for (index, atom) in protein_atoms.iter().enumerate() {
            let residue_index = residue_indices
                .get(&atom.residue)
                .copied()
                .ok_or_else(|| SasaError::MissingResidue(atom.residue.clone()))?;
            bare_by_residue[residue_index] += bare_areas[index];
            glyco_by_residue[residue_index] += glyco_areas[glyco_indices[index]];
        }

        for index in 0..protein_residues.len() {
            let bare = bare_by_residue[index];
            let glyco = glyco_by_residue[index];
            bare_sasa_sum[index] += bare;
            glyco_sasa_sum[index] += glyco;
        }
    }

    let frame_count = frames.len();
    let frame_count_f64 = frame_count as f64;
    let residues = protein_residues
        .into_iter()
        .enumerate()
        .map(|(index, residue)| {
            let mean_bare_sasa = bare_sasa_sum[index] / frame_count_f64;
            let mean_glyco_sasa = glyco_sasa_sum[index] / frame_count_f64;
            let shielding_percent = (mean_bare_sasa > 0.0)
                .then_some((mean_bare_sasa - mean_glyco_sasa) / mean_bare_sasa * 100.0)
                .unwrap_or(0.0);
            let real_sasa_nm2 = mean_glyco_sasa * 0.01;
            let occupancy = (mean_bare_sasa > 0.0).then_some(1.0).unwrap_or(0.0);
            let hotspot = occupancy > 0.5
                && shielding_percent <= COOKBOOK_HOTSPOT_SHIELDING_PERCENT
                && real_sasa_nm2 > 1.0e-6;
            SasaResidue {
                residue,
                shielding_percent,
                real_sasa_nm2,
                occupancy,
                hotspot,
            }
        })
        .collect();

    Ok(SasaAnalysis {
        frame_count,
        residues,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell(i64, i64, i64);

/// GROMACS's double-cubic-lattice rolling-probe surface area for the
/// requested atom indices. Coordinates and radii are in Ångström; returned
/// areas are in Å².
fn double_cubic_lattice_sasa(positions: &[Vec3], radii: &[f64], targets: &[usize]) -> Vec<f64> {
    if positions.is_empty() || targets.is_empty() {
        return vec![0.0; positions.len()];
    }
    let solvent_radii = radii
        .iter()
        .map(|radius| radius + COOKBOOK_PROBE_RADIUS_ANGSTROM)
        .collect::<Vec<_>>();
    let max_radius = solvent_radii.iter().copied().fold(0.0, f64::max);
    let cell_size = (2.0 * max_radius).max(1.0e-6);
    let mut grid = HashMap::<GridCell, Vec<usize>>::new();
    for (index, position) in positions.iter().enumerate() {
        grid.entry(grid_cell(*position, cell_size))
            .or_default()
            .push(index);
    }

    let unit_points = gromacs_unit_sphere_points(COOKBOOK_NDOTS);
    let dot_area = 4.0 * std::f64::consts::PI / unit_points.len() as f64;
    let mut areas = vec![0.0; positions.len()];
    for &index in targets {
        let atom_radius = solvent_radii[index];
        let center = positions[index];
        let cell = grid_cell(center, cell_size);
        let mut accessible = 0usize;
        for point in &unit_points {
            let buried = (-1..=1).any(|dx| {
                (-1..=1).any(|dy| {
                    (-1..=1).any(|dz| {
                        grid.get(&GridCell(cell.0 + dx, cell.1 + dy, cell.2 + dz))
                            .into_iter()
                            .flatten()
                            .any(|&other| {
                                if other == index {
                                    return false;
                                }
                                let neighbor_radius = solvent_radii[other];
                                let displacement = Vec3 {
                                    x: positions[other].x - center.x,
                                    y: positions[other].y - center.y,
                                    z: positions[other].z - center.z,
                                };
                                let distance_squared = dot(displacement, displacement);
                                if distance_squared
                                    > (atom_radius + neighbor_radius)
                                        * (atom_radius + neighbor_radius)
                                {
                                    return false;
                                }
                                let reference_dot = (distance_squared + atom_radius * atom_radius
                                    - neighbor_radius * neighbor_radius)
                                    / (2.0 * atom_radius);
                                dot(*point, displacement) > reference_dot
                            })
                    })
                })
            });
            if !buried {
                accessible += 1;
            }
        }
        areas[index] = accessible as f64 * dot_area * atom_radius * atom_radius;
    }
    areas
}

/// Build the 32-point icosahedral/dodecahedral dot set selected by GROMACS
/// for the Cookbook's requested `-ndots 15` density.
///
/// The first twelve points are the vertices of an icosahedron. The remaining
/// twenty are normalized sums of the three vertices making each icosahedron
/// face, which produces the dual dodecahedron vertices. This is the
/// `ico_dot_dod` construction used by GROMACS's `gmx sasa` implementation.
fn gromacs_unit_sphere_points(_requested_density: usize) -> Vec<Vec3> {
    let cos_72 = (2.0 * std::f64::consts::PI / 5.0).cos();
    let horizontal_radius = (1.0 - 2.0 * cos_72).sqrt() / (1.0 - cos_72);
    let ring_height = cos_72 / (1.0 - cos_72);

    let mut icosahedron = Vec::with_capacity(12);
    icosahedron.push(Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    });
    for angle_degrees in [72.0_f64, 144.0, 216.0, 288.0, 0.0] {
        let angle = angle_degrees.to_radians();
        icosahedron.push(Vec3 {
            x: horizontal_radius * angle.cos(),
            y: horizontal_radius * angle.sin(),
            z: ring_height,
        });
    }
    for angle_degrees in [36.0_f64, 108.0, 180.0, 252.0, 324.0] {
        let angle = angle_degrees.to_radians();
        icosahedron.push(Vec3 {
            x: horizontal_radius * angle.cos(),
            y: horizontal_radius * angle.sin(),
            z: -ring_height,
        });
    }
    icosahedron.push(Vec3 {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    });

    let edge_squared = 2.0 * horizontal_radius * horizontal_radius * (1.0 - cos_72);
    let mut points = icosahedron.clone();
    for first in 0..icosahedron.len() {
        for second in (first + 1)..icosahedron.len() {
            for third in (second + 1)..icosahedron.len() {
                let first_second = squared_distance(icosahedron[first], icosahedron[second]);
                let first_third = squared_distance(icosahedron[first], icosahedron[third]);
                let second_third = squared_distance(icosahedron[second], icosahedron[third]);
                if (first_second - edge_squared).abs() > 1.0e-3
                    || (first_third - edge_squared).abs() > 1.0e-3
                    || (second_third - edge_squared).abs() > 1.0e-3
                {
                    continue;
                }
                let sum = Vec3 {
                    x: icosahedron[first].x + icosahedron[second].x + icosahedron[third].x,
                    y: icosahedron[first].y + icosahedron[second].y + icosahedron[third].y,
                    z: icosahedron[first].z + icosahedron[second].z + icosahedron[third].z,
                };
                let length = dot(sum, sum).sqrt();
                points.push(Vec3 {
                    x: sum.x / length,
                    y: sum.y / length,
                    z: sum.z / length,
                });
            }
        }
    }
    debug_assert_eq!(points.len(), 32);
    points
}

fn grid_cell(position: Vec3, cell_size: f64) -> GridCell {
    GridCell(
        (position.x / cell_size).floor() as i64,
        (position.y / cell_size).floor() as i64,
        (position.z / cell_size).floor() as i64,
    )
}

fn squared_distance(first: Vec3, second: Vec3) -> f64 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2)
}

fn dot(first: Vec3, second: Vec3) -> f64 {
    first.x * second.x + first.y * second.y + first.z * second.z
}

fn format_protein_atom(atom: &StructureAtom, occupancy: f64, bfactor: f64) -> String {
    let insertion_code = atom.residue.insertion_code.unwrap_or(' ');
    let chain = atom.residue.chain.chars().next().unwrap_or(' ');
    let element = atom.element.trim();
    format!(
        "ATOM  {:>5} {:<4} {:>3} {:1}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
        atom.id.0,
        atom.name,
        atom.residue_name,
        chain,
        atom.residue.number,
        insertion_code,
        atom.position.x,
        atom.position.y,
        atom.position.z,
        occupancy,
        bfactor,
        element,
    )
}

fn is_protein_residue(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "ASH"
            | "CYS"
            | "CYM"
            | "CYX"
            | "GLN"
            | "GLU"
            | "GLH"
            | "GLY"
            | "HIS"
            | "HID"
            | "HIE"
            | "HIP"
            | "HYP"
            | "ILE"
            | "LEU"
            | "LYN"
            | "LYS"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "NLN"
            | "OLS"
            | "OLT"
            | "OLP"
    )
}

fn vdw_radius(element: &str) -> f64 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "F" => 1.47,
        "P" => 1.80,
        "S" => 1.80,
        "CL" => 1.75,
        // GROMACS's vdwradii.dat uses 0.14 nm as its fallback when the
        // residue/atom-name lookup has no match.
        _ => 1.40,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, read_pdb_str};

    fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    #[test]
    fn isolated_atom_has_full_rolling_probe_area() {
        let areas = double_cubic_lattice_sasa(&[vec3(0.0, 0.0, 0.0)], &[1.7], &[0]);
        let radius = 1.7 + COOKBOOK_PROBE_RADIUS_ANGSTROM;
        let expected = 4.0 * std::f64::consts::PI * radius * radius;
        assert!((areas[0] - expected).abs() < 1.0e-10);
    }

    #[test]
    fn a_neighbor_reduces_accessible_area() {
        let positions = [vec3(0.0, 0.0, 0.0), vec3(2.0, 0.0, 0.0)];
        let radii = [1.7, 1.7];
        let one = double_cubic_lattice_sasa(&positions[..1], &radii[..1], &[0])[0];
        let two = double_cubic_lattice_sasa(&positions, &radii, &[0])[0];
        assert!(two < one);
    }

    #[test]
    fn cookbook_density_expands_to_gromacs_dot_set() {
        assert_eq!(gromacs_unit_sphere_points(COOKBOOK_NDOTS).len(), 32);
    }

    #[test]
    fn output_maps_contain_only_protein_atoms() {
        let structure = read_pdb_str(
            include_str!("../../../tests/fixtures/protein.pdb"),
            &BuildOptions::default(),
        )
        .unwrap();
        let analysis = calculate_sasa([&structure]).unwrap();
        assert_eq!(analysis.frame_count, 1);
        assert_eq!(analysis.residues.len(), 2);
        let output = analysis.sasa_pdb_string(&structure).unwrap();
        assert!(output.lines().all(|line| !line.starts_with("HETATM")));
        assert!(output.contains(" ASN A   1"));
        assert!(output.ends_with("END\n"));
    }
}
