//! Chemistry of a deposited glycan, independent of any fit (port of GlycoFlow's
//! `scripts/fitting/deposition_checks.py`):
//!
//! * anomer: the signed volume at each anomeric carbon (ring oxygen, next ring carbon, linkage
//!   atom) of the deposited coordinates against GlycoFlow's template built from the deposited
//!   labels (CCD codes). Opposite signs: the label contradicts the geometry (e.g. MAN for a
//!   beta-mannose, or every anomer of a tree inverted).
//! * amide: psi_N = CB-CG-ND2-C1 of an N-glycan; the N-glycosidic amide is trans (|psi_N| near
//!   180 deg). Flagged below [`AMIDE_CIS_LIMIT`], which catches cis (near 0) and twisted amides.
//! * ring plane: links that must lie in a ring plane (C-mannose on Trp: CB-CG-CD1-C1 near 0 deg)
//!   are flagged more than [`PLANE_LIMIT`] out of it.

use std::collections::BTreeMap;

use glycoflow_core::ResidueLibrary;
use serde::Serialize;

use crate::anchor::LinkTorsion;
use crate::error::Result;
use crate::site::{DepositedGlycan, Site, glycam_of};

type V3 = [f64; 3];

/// |psi_N| below this (degrees) is flagged.
pub const AMIDE_CIS_LIMIT: f64 = 150.0;

/// A planar link further than this from its plane (degrees) is flagged (Asn: the amide limit).
pub const PLANE_LIMIT: f64 = 180.0 - AMIDE_CIS_LIMIT;

#[derive(Debug, Clone, Serialize)]
pub struct AnomerMismatch {
    /// residue path ("r", "r/4", ...)
    pub path: String,
    pub chain: String,
    pub number: i32,
    /// CCD label of the deposit
    pub ccd: String,
    /// anomer the label stands for ('a' or 'b'); the geometry has the other one
    pub label_anomer: char,
}

#[derive(Debug, Clone, Serialize)]
pub struct DepositionChecks {
    /// residues whose anomeric centre could be compared
    pub checked_residues: usize,
    pub anomer_mismatch: Vec<AnomerMismatch>,
    /// CB-CG-ND2-C1 of the deposit (degrees); N-glycans only
    pub psi_n_deg: Option<f64>,
    pub amide_not_trans: bool,
    /// A-B-link-C1 of the deposit (degrees), any site
    pub link_torsion_deg: Option<f64>,
    /// a link that must be planar (C-mannose on Trp) lies out of its ring plane
    pub link_out_of_plane: bool,
    /// any of the above
    pub flagged: bool,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Signed volume of (a, b, c) about `o`.
pub fn signed_volume(o: V3, a: V3, b: V3, c: V3) -> f64 {
    dot(sub(a, o), cross(sub(b, o), sub(c, o)))
}

/// Dihedral a-b-c-d in degrees (IUPAC sign).
pub fn dihedral_deg(a: V3, b: V3, c: V3, d: V3) -> f64 {
    let (b0, b1, b2) = (sub(a, b), sub(c, b), sub(d, c));
    let n = dot(b1, b1).sqrt();
    let b1 = [b1[0] / n, b1[1] / n, b1[2] / n];
    let v = sub(b0, [b1[0] * dot(b0, b1), b1[1] * dot(b0, b1), b1[2] * dot(b0, b1)]);
    let w = sub(b2, [b1[0] * dot(b2, b1), b1[1] * dot(b2, b1), b1[2] * dot(b2, b1)]);
    dot(cross(b1, v), w).atan2(dot(v, w)).to_degrees()
}

/// The checks on coordinates: `template` holds the reference built from the deposited labels,
/// keyed like the deposit by (residue path, GLYCAM atom name); `template_root_link` is the
/// template's aglycone oxygen; `anchor` the site's anchor atoms (A, B, link).
pub fn check(
    deposited: &DepositedGlycan,
    anchor: &[V3; 3],
    torsion: LinkTorsion,
    template: &BTreeMap<(String, String), V3>,
    template_root_link: V3,
) -> DepositionChecks {
    let dep = &deposited.atoms;
    let mut checked = 0;
    let mut mismatch = Vec::new();
    for residue in &deposited.residues {
        let Some((_, anomer, cpos)) = glycam_of(&residue.name) else {
            continue;
        };
        let path = residue.path.clone();
        let ring_o = if cpos == 2 { "O6" } else { "O5" };
        let need = [format!("C{cpos}"), ring_o.to_string(), format!("C{}", cpos + 1)];
        let (link_dep, link_tpl) = if path == "r" {
            (Some(anchor[2]), Some(template_root_link))
        } else {
            match (path.rsplit_once('/'), residue.parent_position) {
                (Some((parent, _)), Some(position)) => {
                    let key = (parent.to_string(), format!("O{position}"));
                    (dep.get(&key).copied(), template.get(&key).copied())
                }
                _ => (None, None),
            }
        };
        let (Some(link_dep), Some(link_tpl)) = (link_dep, link_tpl) else {
            continue;
        };
        let get = |m: &BTreeMap<(String, String), V3>, name: &str| m.get(&(path.clone(), name.to_string())).copied();
        let (Some(d0), Some(d1), Some(d2), Some(t0), Some(t1), Some(t2)) = (
            get(dep, &need[0]),
            get(dep, &need[1]),
            get(dep, &need[2]),
            get(template, &need[0]),
            get(template, &need[1]),
            get(template, &need[2]),
        ) else {
            continue;
        };
        checked += 1;
        let vd = signed_volume(d0, d1, d2, link_dep);
        let vt = signed_volume(t0, t1, t2, link_tpl);
        if vd.signum() != vt.signum() {
            mismatch.push(AnomerMismatch {
                path,
                chain: residue.id.chain.clone(),
                number: residue.id.number,
                ccd: residue.name.clone(),
                label_anomer: anomer,
            });
        }
    }
    let link_torsion_deg = deposited
        .residues
        .iter()
        .find(|r| r.path == "r")
        .and_then(|r| glycam_of(&r.name))
        .and_then(|(_, _, cpos)| dep.get(&("r".to_string(), format!("C{cpos}"))))
        .map(|c| dihedral_deg(anchor[0], anchor[1], anchor[2], *c));
    let (psi_n_deg, amide_not_trans, link_out_of_plane) = match torsion {
        // the Asn amide
        LinkTorsion::Planar { centre: 180.0 } => (link_torsion_deg, link_torsion_deg.is_some_and(|p| p.abs() < AMIDE_CIS_LIMIT), false),
        LinkTorsion::Planar { centre } => {
            let off = link_torsion_deg.is_some_and(|p| {
                let d = (p - centre).rem_euclid(360.0);
                d.min(360.0 - d) > PLANE_LIMIT
            });
            (None, false, off)
        }
        LinkTorsion::Free => (None, false, false),
    };
    DepositionChecks {
        checked_residues: checked,
        flagged: !mismatch.is_empty() || amide_not_trans || link_out_of_plane,
        anomer_mismatch: mismatch,
        psi_n_deg,
        amide_not_trans,
        link_torsion_deg,
        link_out_of_plane,
    }
}

/// The checks for a site's deposited glycan (None without a deposit). The reference is GlycoFlow's
/// majority-pucker template of the deposited labels' sequence.
pub fn deposition_checks(site: &Site, library: &ResidueLibrary) -> Result<Option<DepositionChecks>> {
    let Some(deposited) = site.deposited.as_ref() else {
        return Ok(None);
    };
    let sequence = site
        .sequence
        .deposited_tree_glycam
        .as_deref()
        .unwrap_or(&site.sequence.sequence);
    let built = library.build(sequence, None)?;
    let mut template = BTreeMap::new();
    let mut root_link = None;
    for ((path, name), x) in built.res_paths.iter().zip(&built.atom_names).zip(&built.coords) {
        if path == "agl" {
            root_link.get_or_insert(*x);
        }
        template.insert((path.clone(), name.clone()), *x);
    }
    let Some(root_link) = root_link else {
        return Ok(None);
    };
    let torsion = crate::anchor::anchor(&site.residue_name).map_or(LinkTorsion::Free, |a| a.torsion);
    Ok(Some(check(deposited, &site.anchor, torsion, &template, root_link)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::site::DepositedResidue;
    use glysys::ResidueId;

    fn glycan(atoms: &[(&str, &str, V3)], residues: &[(&str, &str, Option<u32>)]) -> DepositedGlycan {
        DepositedGlycan {
            residues: residues
                .iter()
                .enumerate()
                .map(|(i, (path, name, parent))| DepositedResidue {
                    id: ResidueId { chain: "B".into(), number: i as i32 + 1, insertion_code: None },
                    name: name.to_string(),
                    path: path.to_string(),
                    parent_position: *parent,
                })
                .collect(),
            atoms: atoms.iter().map(|(p, n, x)| ((p.to_string(), n.to_string()), *x)).collect(),
        }
    }

    const CB: V3 = [0.0, 1.5, 0.0];
    const CG: V3 = [0.0, 0.0, 0.0];
    const ND2: V3 = [1.3, -0.6, 0.0];
    const AMIDE: LinkTorsion = LinkTorsion::Planar { centre: 180.0 };

    #[test]
    fn flags_a_cis_amide_and_an_inverted_anomer() {
        // trans: C1 opposite CB across the CG-ND2 bond (psi_N = 180)
        let trans_c1 = [1.3, -2.0, 0.0];
        // cis: C1 on the CB side (psi_N = 0)
        let cis_c1 = [2.6, 0.0, 0.0];
        let ring = |c1: V3| vec![("r", "C1", c1), ("r", "O5", [c1[0] + 0.5, c1[1] + 1.2, 0.4]), ("r", "C2", [c1[0] + 1.2, c1[1] - 0.6, -0.5])];
        let template: BTreeMap<(String, String), V3> =
            ring(trans_c1).into_iter().map(|(p, n, x)| ((p.to_string(), n.to_string()), x)).collect();
        let anchor = [CB, CG, ND2];

        let good = glycan(&ring(trans_c1), &[("r", "NAG", None)]);
        let c = check(&good, &anchor, AMIDE, &template, ND2);
        assert_eq!(c.checked_residues, 1);
        assert!(c.anomer_mismatch.is_empty());
        assert!((c.psi_n_deg.unwrap().abs() - 180.0).abs() < 1e-6);
        assert!(!c.flagged);

        let cis = glycan(&ring(cis_c1), &[("r", "NAG", None)]);
        let c = check(&cis, &anchor, AMIDE, &template, ND2);
        assert!(c.psi_n_deg.unwrap().abs() < 1e-6 && c.amide_not_trans && c.flagged);

        // the mirror image of the anomeric centre (ring oxygen below the plane instead of above)
        let mirrored: Vec<(&str, &str, V3)> = ring(trans_c1).into_iter().map(|(p, n, x)| (p, n, [x[0], x[1], -x[2]])).collect();
        let inverted = glycan(&mirrored, &[("r", "NAG", None)]);
        let c = check(&inverted, &anchor, AMIDE, &template, ND2);
        assert_eq!(c.anomer_mismatch.len(), 1);
        assert_eq!((c.anomer_mismatch[0].ccd.as_str(), c.anomer_mismatch[0].label_anomer), ("NAG", 'b'));
        assert!(c.flagged);
    }

    #[test]
    fn serine_sites_have_no_amide() {
        let template = BTreeMap::new();
        let dep = glycan(&[("r", "C1", [1.3, -2.0, 0.0])], &[("r", "NGA", None)]);
        let c = check(&dep, &[CB, CG, ND2], LinkTorsion::Free, &template, ND2);
        assert!(c.psi_n_deg.is_none() && !c.flagged && c.checked_residues == 0);
        assert!((c.link_torsion_deg.unwrap().abs() - 180.0).abs() < 1e-6);
    }

    #[test]
    fn c_mannose_must_lie_in_the_indole_plane() {
        // (CB, CG, CD1): C1 cis to CB in the ring plane is right; trans or tilted is not
        let trp = LinkTorsion::Planar { centre: 0.0 };
        let template = BTreeMap::new();
        let in_plane = glycan(&[("r", "C1", [2.6, 0.0, 0.0])], &[("r", "MAN", None)]);
        let c = check(&in_plane, &[CB, CG, ND2], trp, &template, ND2);
        assert!(c.link_torsion_deg.unwrap().abs() < 1e-6 && !c.link_out_of_plane && !c.flagged);
        let tilted = glycan(&[("r", "C1", [2.0, -0.2, 1.4])], &[("r", "MAN", None)]);
        let c = check(&tilted, &[CB, CG, ND2], trp, &template, ND2);
        assert!(c.link_torsion_deg.unwrap().abs() > PLANE_LIMIT && c.link_out_of_plane && c.flagged);
        assert!(c.psi_n_deg.is_none() && !c.amide_not_trans);
    }
}
