//! Compact subset of the 2010 backbone-dependent Dunbrack rotamer library.
//!
//! The bundled table contains the ten highest-probability entries per
//! backbone bin for ASN, SER, THR, and TRP. The source library is described by
//! Shapovalov & Dunbrack, Structure 2011 and distributed under CC BY 4.0.

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

use flate2::read::GzDecoder;
use glysys::{ResidueId, Structure, Vec3};

#[derive(Debug, Clone)]
pub(crate) struct Rotamer {
    pub probability: f64,
    pub chi_degrees: Vec<f64>,
}

type Library = HashMap<(String, i32, i32), Vec<Rotamer>>;

pub(crate) fn rotamers(structure: &Structure, site: &ResidueId) -> Vec<Rotamer> {
    let Some(residue) = structure
        .residues()
        .into_iter()
        .find(|residue| residue.id == *site)
    else {
        return Vec::new();
    };
    let Some((phi, psi)) = backbone_angles(structure, site) else {
        return Vec::new();
    };
    library()
        .get(&(residue.name, snap(phi), snap(psi)))
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn apply(
    structure: &mut Structure,
    site: &ResidueId,
    rotamer_index: usize,
) -> Result<(), reglyco_core::ReGlycoError> {
    let residue = structure
        .residues()
        .into_iter()
        .find(|residue| residue.id == *site)
        .ok_or_else(|| reglyco_core::ReGlycoError::SiteNotFound(site.clone()))?;
    let choices = rotamers(structure, site);
    let selected =
        choices
            .get(rotamer_index)
            .ok_or_else(|| reglyco_core::ReGlycoError::InvalidChemistry {
                site: site.clone(),
                message: format!("Dunbrack rotamer {rotamer_index} is unavailable"),
            })?;
    for (chi_index, definition) in definitions(&residue.name).iter().enumerate() {
        let Some(&target) = selected.chi_degrees.get(chi_index) else {
            break;
        };
        let points = definition
            .dihedral
            .iter()
            .map(|name| atom_position(structure, site, name))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| reglyco_core::ReGlycoError::InvalidChemistry {
                site: site.clone(),
                message: format!("missing atoms for chi {} rotation", chi_index + 1),
            })?;
        let origin = atom_position(structure, site, definition.axis[0]).unwrap();
        let axis_point = atom_position(structure, site, definition.axis[1]).unwrap();
        let delta = (target - dihedral(points[0], points[1], points[2], points[3])).to_radians();
        let axis = normalize(subtract(axis_point, origin))
            .ok_or(reglyco_core::ReGlycoError::InvalidGeometry)?;
        let cosine = delta.cos();
        let sine = delta.sin();
        let updates = definition
            .mobile
            .iter()
            .filter_map(|name| {
                let atom = structure.find_atom(site, name)?;
                let position = structure.atom(atom)?.position;
                let relative = subtract(position, origin);
                let rotated = add(
                    add(scale(relative, cosine), scale(cross(axis, relative), sine)),
                    scale(axis, dot(axis, relative) * (1.0 - cosine)),
                );
                Some((atom, add(origin, rotated)))
            })
            .collect::<Vec<_>>();
        structure.set_atom_positions(updates)?;
    }
    Ok(())
}

pub(crate) fn probability(structure: &Structure, site: &ResidueId, index: usize) -> Option<f64> {
    rotamers(structure, site)
        .get(index)
        .map(|entry| entry.probability)
}

fn library() -> &'static Library {
    static LIBRARY: OnceLock<Library> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let bytes = include_bytes!("../data/dunbrack-supported-top10.tsv.gz");
        let mut decoder = GzDecoder::new(bytes.as_slice());
        let mut contents = String::new();
        decoder
            .read_to_string(&mut contents)
            .expect("bundled Dunbrack table is valid gzip text");
        let mut result = Library::new();
        for line in contents.lines() {
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() < 5 {
                continue;
            }
            let key = (
                fields[0].to_string(),
                fields[1].parse().unwrap_or_default(),
                fields[2].parse().unwrap_or_default(),
            );
            result.entry(key).or_default().push(Rotamer {
                probability: fields[3].parse().unwrap_or_default(),
                chi_degrees: fields[4..]
                    .iter()
                    .filter_map(|value| value.parse().ok())
                    .collect(),
            });
        }
        result
    })
}

struct ChiDefinition {
    dihedral: [&'static str; 4],
    axis: [&'static str; 2],
    mobile: &'static [&'static str],
}

fn definitions(name: &str) -> &'static [ChiDefinition] {
    static ASN: [ChiDefinition; 2] = [
        ChiDefinition {
            dihedral: ["N", "CA", "CB", "CG"],
            axis: ["CA", "CB"],
            mobile: &["CG", "OD1", "ND2"],
        },
        ChiDefinition {
            dihedral: ["CA", "CB", "CG", "OD1"],
            axis: ["CB", "CG"],
            mobile: &["OD1", "ND2"],
        },
    ];
    static SER: [ChiDefinition; 1] = [ChiDefinition {
        dihedral: ["N", "CA", "CB", "OG"],
        axis: ["CA", "CB"],
        mobile: &["OG"],
    }];
    static THR: [ChiDefinition; 1] = [ChiDefinition {
        dihedral: ["N", "CA", "CB", "OG1"],
        axis: ["CA", "CB"],
        mobile: &["OG1", "CG2"],
    }];
    static TRP: [ChiDefinition; 2] = [
        ChiDefinition {
            dihedral: ["N", "CA", "CB", "CG"],
            axis: ["CA", "CB"],
            mobile: &["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        },
        ChiDefinition {
            dihedral: ["CA", "CB", "CG", "CD1"],
            axis: ["CB", "CG"],
            mobile: &["CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        },
    ];
    match name {
        "ASN" => &ASN,
        "SER" => &SER,
        "THR" => &THR,
        "TRP" => &TRP,
        _ => &[],
    }
}

fn backbone_angles(structure: &Structure, site: &ResidueId) -> Option<(f64, f64)> {
    let residues = structure
        .residues()
        .into_iter()
        .filter(|residue| residue.id.chain == site.chain)
        .collect::<Vec<_>>();
    let index = residues.iter().position(|residue| residue.id == *site)?;
    let current = &residues[index].id;
    let n = atom_position(structure, current, "N")?;
    let ca = atom_position(structure, current, "CA")?;
    let c = atom_position(structure, current, "C")?;
    let phi = index
        .checked_sub(1)
        .and_then(|previous| atom_position(structure, &residues[previous].id, "C"))
        .map_or(-60.0, |previous_c| dihedral(previous_c, n, ca, c));
    let psi = residues
        .get(index + 1)
        .and_then(|next| atom_position(structure, &next.id, "N"))
        .map_or(-40.0, |next_n| dihedral(n, ca, c, next_n));
    Some((phi, psi))
}

fn atom_position(structure: &Structure, residue: &ResidueId, name: &str) -> Option<Vec3> {
    structure
        .find_atom(residue, name)
        .and_then(|atom| structure.atom(atom))
        .map(|atom| atom.position)
}

fn snap(angle: f64) -> i32 {
    let mut value = (angle / 10.0).round() as i32 * 10;
    if value > 180 {
        value -= 360;
    } else if value < -180 {
        value += 360;
    }
    value
}

fn dihedral(first: Vec3, second: Vec3, third: Vec3, fourth: Vec3) -> f64 {
    let b1 = normalize(subtract(third, second)).unwrap_or(Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    });
    let b0 = subtract(first, second);
    let b2 = subtract(fourth, third);
    let v = subtract(b0, scale(b1, dot(b0, b1)));
    let w = subtract(b2, scale(b1, dot(b2, b1)));
    dot(cross(b1, v), w).atan2(dot(v, w)).to_degrees()
}

fn normalize(vector: Vec3) -> Option<Vec3> {
    let length = dot(vector, vector).sqrt();
    (length > 1.0e-12).then(|| scale(vector, 1.0 / length))
}

fn add(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.x + second.x,
        y: first.y + second.y,
        z: first.z + second.z,
    }
}

fn subtract(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.x - second.x,
        y: first.y - second.y,
        z: first.z - second.z,
    }
}

fn scale(vector: Vec3, factor: f64) -> Vec3 {
    Vec3 {
        x: vector.x * factor,
        y: vector.y * factor,
        z: vector.z * factor,
    }
}

fn dot(first: Vec3, second: Vec3) -> f64 {
    first.x * second.x + first.y * second.y + first.z * second.z
}

fn cross(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.y * second.z - first.z * second.y,
        y: first.z * second.x - first.x * second.z,
        z: first.x * second.y - first.y * second.x,
    }
}
