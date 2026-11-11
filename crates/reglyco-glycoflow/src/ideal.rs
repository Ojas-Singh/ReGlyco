//! Ideal bond lengths and angles of the fitted glycan, from the CCP4 monomer library
//! (`data/ideal_geometry.json`, written by `data/make_ideal_geometry.py`).
//!
//! The residue templates of the GlycoFlow library are frames of molecular dynamics: their bonds
//! scatter by 0.03 A around the force field's values (up to 0.1 A), and the force field's values
//! are not those a crystallographic model is refined and validated against. A model restrained
//! to them leaves a well-refined deposit and its density. The restraints therefore take their
//! targets from the monomer library wherever it has the residue and the link; anything else
//! (hydrogens, substituents, rare sugars) keeps the template's value.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use glycoflow_core::topology::Glycan;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Table {
    /// GLYCAM residue code ("YB") -> its bonds and angles, in GLYCAM atom names
    residues: BTreeMap<String, Residue>,
    /// "(anomeric carbon number)-(position)" -> the glycosidic link
    links: BTreeMap<String, Link>,
    /// protein residue -> its link to the anomeric carbon
    anchors: BTreeMap<String, Link>,
}

#[derive(Debug, Deserialize)]
struct Residue {
    bonds: Vec<(String, String, f64)>,
    angles: Vec<(String, String, String, f64)>,
}

#[derive(Debug, Deserialize)]
struct Link {
    /// anomeric carbon - linking atom (A)
    bond: f64,
    /// at the linking atom (degrees)
    #[serde(default)]
    angle: Option<f64>,
    /// at the anomeric carbon, by the name of its other neighbour (degrees)
    anomeric: BTreeMap<String, f64>,
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        serde_json::from_str(include_str!("../data/ideal_geometry.json"))
            .expect("data/ideal_geometry.json")
    })
}

/// Ideal geometry on the atoms of one glycan at one site.
pub struct Geometry<'a> {
    glycan: &'a Glycan,
    anchor: Option<&'static Link>,
    link_bond: f64,
}

impl<'a> Geometry<'a> {
    /// `site_residue`: the protein residue that carries the glycan; `link_bond`: its bond to the
    /// anomeric carbon (the site's own, see [`crate::anchor`]).
    pub fn new(glycan: &'a Glycan, site_residue: &str, link_bond: f64) -> Self {
        Self {
            glycan,
            anchor: table().anchors.get(site_residue),
            link_bond,
        }
    }

    fn residue(&self, i: usize) -> Option<&'static Residue> {
        // "4YB": linkage prefix, then the residue code
        let name = &self.glycan.res_names[i];
        table().residues.get(name.get(name.len().checked_sub(2)?..)?)
    }

    fn name(&self, i: usize) -> &str {
        &self.glycan.atom_names[i]
    }

    fn path(&self, i: usize) -> &str {
        &self.glycan.res_paths[i]
    }

    /// Whether the residue of `child` hangs on the residue of `parent`.
    fn hangs_on(&self, child: usize, parent: usize) -> bool {
        let (c, p) = (self.path(child), self.path(parent));
        c.strip_prefix(p)
            .and_then(|rest| rest.strip_prefix('/'))
            .is_some_and(|rest| !rest.contains('/'))
    }

    /// The link of the anomeric carbon `carbon` to `atom` of the parent residue or the protein.
    fn link(&self, carbon: usize, atom: usize) -> Option<&'static Link> {
        if self.path(atom) == "agl" {
            return (self.path(carbon) == "r").then_some(self.anchor).flatten();
        }
        if !self.hangs_on(carbon, atom) {
            return None;
        }
        let number = self.name(carbon).strip_prefix('C')?;
        let position = self.name(atom).strip_prefix('O')?;
        table().links.get(&format!("{number}-{position}"))
    }

    /// Ideal length of the bond i-j.
    pub fn bond(&self, i: usize, j: usize) -> Option<f64> {
        if self.path(i) == self.path(j) {
            let (a, b) = (self.name(i), self.name(j));
            return self
                .residue(i)?
                .bonds
                .iter()
                .find(|(x, y, _)| (x == a && y == b) || (x == b && y == a))
                .map(|e| e.2);
        }
        for (carbon, atom) in [(i, j), (j, i)] {
            if let Some(link) = self.link(carbon, atom) {
                return Some(if self.path(atom) == "agl" {
                    self.link_bond
                } else {
                    link.bond
                });
            }
        }
        None
    }

    /// Ideal angle i-k-j (radians).
    pub fn angle(&self, i: usize, k: usize, j: usize) -> Option<f64> {
        let degrees = if self.path(i) == self.path(k) && self.path(j) == self.path(k) {
            let (a, c, b) = (self.name(i), self.name(k), self.name(j));
            self.residue(k)?
                .angles
                .iter()
                .find(|(x, m, y, _)| m == c && ((x == a && y == b) || (x == b && y == a)))
                .map(|e| e.3)
        } else {
            [(i, j), (j, i)].into_iter().find_map(|(own, other)| {
                if self.path(own) != self.path(k) {
                    return None;
                }
                // at the anomeric carbon, or at the atom it binds
                match self.link(k, other) {
                    Some(link) => link.anomeric.get(self.name(own)).copied(),
                    None => self.link(other, k).and_then(|link| link.angle),
                }
            })
        };
        degrees.map(f64::to_radians)
    }
}

/// The restraint targets of a glycan: which atom pairs and centres are restrained, and the ideal
/// value of each where the monomer library has one. Resolved once per problem.
#[derive(Debug, Clone, Default)]
pub struct Targets {
    /// (i, j, bonded, ideal distance): bonds and 1-3 pairs
    pub pairs: Vec<(usize, usize, bool, Option<f64>)>,
    /// (centre, its first three bonded neighbours, ideal unsigned volume)
    pub chiral: Vec<(usize, [usize; 3], Option<f64>)>,
    /// the torsions around every sugar ring, which hold its shape: bonds and angles are the same
    /// in a chair and a boat. With a `geometry` only.
    pub ring_torsions: Vec<[usize; 4]>,
    /// (the torsion, 0 or pi) around every amide bond over its heavy atoms: nothing else keeps
    /// an N-acetyl group planar and trans. With a `geometry` only.
    pub amide_torsions: Vec<([usize; 4], f64)>,
}

impl Targets {
    /// Without a `geometry`, no pair has an ideal value (all come from a template).
    pub fn new(glycan: &Glycan, geometry: Option<&Geometry>) -> Self {
        let n = glycan.n_atoms();
        let topo = &glycan.topology.topo_dist;
        let nt = glycan.topology.n_atoms;
        let bonded = |i: usize, j: usize| topo[i * nt + j] == 1;
        // The hydrogen of the aglycone placeholder is no atom of the model: it stays where the
        // pose put it, and a restraint to it would hold the anomeric carbon to that pose.
        let placeholder = |i: usize| {
            geometry.is_some() && glycan.res_paths[i] == "agl" && glycan.elements[i] == "H"
        };
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                if placeholder(i) || placeholder(j) {
                    continue;
                }
                let ideal = match (topo[i * nt + j], geometry) {
                    (1, Some(g)) => g.bond(i, j),
                    (2, Some(g)) => (0..n).find(|&k| bonded(i, k) && bonded(k, j)).and_then(|k| {
                        let (a, b) = (g.bond(i, k)?, g.bond(k, j)?);
                        Some((a * a + b * b - 2.0 * a * b * g.angle(i, k, j)?.cos()).sqrt())
                    }),
                    (1 | 2, None) => None,
                    _ => continue,
                };
                pairs.push((i, j, topo[i * nt + j] == 1, ideal));
            }
        }
        let mut chiral = Vec::new();
        for c in 0..n {
            let nb: Vec<usize> = (0..n).filter(|&j| bonded(c, j)).collect();
            if nb.len() >= 3 {
                let k = [nb[0], nb[1], nb[2]];
                let ideal = geometry.and_then(|g| {
                    let [a, b, d] = [g.bond(c, k[0])?, g.bond(c, k[1])?, g.bond(c, k[2])?];
                    let [ca, cb, cd] = [
                        g.angle(k[1], c, k[2])?.cos(),
                        g.angle(k[0], c, k[2])?.cos(),
                        g.angle(k[0], c, k[1])?.cos(),
                    ];
                    let gram = 1.0 + 2.0 * ca * cb * cd - ca * ca - cb * cb - cd * cd;
                    Some(a * b * d * gram.max(0.0).sqrt())
                });
                chiral.push((c, k, ideal));
            }
        }
        let ring_torsions = match geometry {
            Some(_) => glycan
                .topology
                .rings
                .iter()
                .flat_map(|ring| {
                    let (a, n) = (&ring.atoms, ring.len());
                    (0..n).map(move |i| [0, 1, 2, 3].map(|k| a[(i + k) % n]))
                })
                .collect(),
            None => Vec::new(),
        };
        let mut amide_torsions = Vec::new();
        if geometry.is_some() {
            let heavy = |i: usize| glycan.elements[i] != "H";
            let around = |i: usize, not: usize| {
                (0..n).filter(move |&j| j != not && heavy(j) && bonded(i, j))
            };
            let x = &glycan.templates[0];
            for (nitrogen, carbon) in (0..n).flat_map(|i| (0..n).map(move |j| (i, j))) {
                // an amide: N bonded to a carbon that carries a carbonyl oxygen
                let carbonyl = |o: usize| glycan.elements[o] == "O" && around(o, carbon).count() == 0;
                if glycan.elements[nitrogen] != "N"
                    || glycan.elements[carbon] != "C"
                    || !bonded(nitrogen, carbon)
                    || !around(carbon, nitrogen).any(carbonyl)
                {
                    continue;
                }
                for a in around(nitrogen, carbon) {
                    for d in around(carbon, nitrogen) {
                        let p = [a, nitrogen, carbon, d].map(|i| x[i].map(f64::from));
                        let cis = crate::problem::dihedral(p[0], p[1], p[2], p[3]).cos() > 0.0;
                        let target = if cis { 0.0 } else { std::f64::consts::PI };
                        amide_torsions.push(([a, nitrogen, carbon, d], target));
                    }
                }
            }
        }
        Self {
            pairs,
            chiral,
            ring_torsions,
            amide_torsions,
        }
    }

    /// Share of the heavy-atom bonds that have an ideal value.
    pub fn coverage(&self, glycan: &Glycan) -> f64 {
        let heavy = |i: usize| glycan.elements[i] != "H";
        let bonds: Vec<_> = self
            .pairs
            .iter()
            .filter(|p| p.2 && heavy(p.0) && heavy(p.1))
            .collect();
        bonds.iter().filter(|p| p.3.is_some()).count() as f64 / bonds.len().max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_the_common_sugars() {
        let t = table();
        for code in ["YB", "MB", "MA", "LB", "fA", "SA", "XB"] {
            assert!(t.residues.contains_key(code), "{code}");
        }
        let nag = &t.residues["YB"];
        let bond = |a: &str, b: &str| {
            nag.bonds
                .iter()
                .find(|(x, y, _)| (x == a && y == b) || (x == b && y == a))
                .map(|e| e.2)
        };
        assert_eq!(bond("C2", "N2"), Some(1.448));
        assert_eq!(bond("C2N", "O2N"), Some(1.227));
        assert_eq!(bond("C1", "O1"), None);
        assert_eq!(t.links["1-4"].bond, 1.393);
        assert_eq!(t.links["2-6"].anomeric["O6"], 110.76);
        assert!(t.anchors.contains_key("ASN"));
    }
}
