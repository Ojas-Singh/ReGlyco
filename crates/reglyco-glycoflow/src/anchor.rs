//! Protein residues that carry glycans and the geometry of each link: N-glycans on Asn, O-glycans
//! on Ser/Thr (mucin GalNAc, O-Man, O-Fuc, O-Glc, O-GlcNAc), C-mannose on Trp, O-Glc on Tyr
//! (glycogenin), O-glycans on hydroxyproline and hydroxylysine, S-glycans on Cys.
//!
//! Bond lengths, angles at the link atom and psi = A-B-link-C1 were measured on the deposits of
//! X-ray entries at <= 2.3 A (RCSB, 2026-10): Ser/Thr O-links 1.42 A and 114 deg with psi spread over
//! about 90-180 deg (both signs), Trp CD1-C1 1.51 A and 129 deg with psi = 0 +- 6 deg (C1 in the
//! indole plane, cis to CB), Tyr OH-C1 about 121 deg with no planar preference. Asn keeps ReGlyco's
//! N-glycosidic amide geometry.

/// Preference of psi = A-B-link-C1, the torsion about the protein side of the link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkTorsion {
    /// planar, near `centre` (degrees): the trans Asn amide (180), C1 in the Trp indole plane (0)
    Planar { centre: f64 },
    /// no preference (sp3 O and S links, aryl ethers)
    Free,
}

#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub residue: &'static str,
    /// (A, B, link)
    pub atoms: [&'static str; 3],
    /// link - C1 (A)
    pub bond: f64,
    /// B - link - C1 (degrees)
    pub angle: f64,
    pub torsion: LinkTorsion,
    /// side-chain atoms within three bonds of C1, with their bond count to C1 (link = 1); they
    /// meet the glycan through explicit pairs (1-4 and beyond) instead of the clash grids
    pub near: &'static [(&'static str, u32)],
    /// "N-linked", "O-linked", "C-linked", "S-linked"
    pub linkage: &'static str,
    /// search the root ring in both chairs (half the templates flipped; see [`crate::ring`])
    pub both_chairs: bool,
}

pub const ANCHORS: [Anchor; 8] = [
    Anchor {
        residue: "ASN",
        atoms: ["CB", "CG", "ND2"],
        bond: 1.45,
        angle: 123.0,
        torsion: LinkTorsion::Planar { centre: 180.0 },
        near: &[("ND2", 1), ("CG", 2), ("OD1", 3), ("CB", 3)],
        linkage: "N-linked",
        both_chairs: false,
    },
    Anchor {
        residue: "SER",
        atoms: ["CA", "CB", "OG"],
        bond: 1.42,
        angle: 114.0,
        torsion: LinkTorsion::Free,
        near: &[("OG", 1), ("CB", 2), ("CA", 3)],
        linkage: "O-linked",
        both_chairs: false,
    },
    Anchor {
        residue: "THR",
        atoms: ["CA", "CB", "OG1"],
        bond: 1.42,
        angle: 114.0,
        torsion: LinkTorsion::Free,
        near: &[("OG1", 1), ("CB", 2), ("CA", 3), ("CG2", 3)],
        linkage: "O-linked",
        both_chairs: false,
    },
    Anchor {
        residue: "TRP",
        atoms: ["CB", "CG", "CD1"],
        bond: 1.51,
        angle: 129.0,
        torsion: LinkTorsion::Planar { centre: 0.0 },
        near: &[
            ("CD1", 1),
            ("CG", 2),
            ("NE1", 2),
            ("CB", 3),
            ("CD2", 3),
            ("CE2", 3),
        ],
        linkage: "C-linked",
        both_chairs: true,
    },
    Anchor {
        residue: "TYR",
        atoms: ["CE1", "CZ", "OH"],
        bond: 1.40,
        angle: 121.0,
        torsion: LinkTorsion::Free,
        near: &[("OH", 1), ("CZ", 2), ("CE1", 3), ("CE2", 3)],
        linkage: "O-linked",
        both_chairs: false,
    },
    Anchor {
        // 4-hydroxyproline
        residue: "HYP",
        atoms: ["CB", "CG", "OD1"],
        bond: 1.42,
        angle: 114.0,
        torsion: LinkTorsion::Free,
        near: &[("OD1", 1), ("CG", 2), ("CB", 3), ("CD", 3)],
        linkage: "O-linked",
        both_chairs: false,
    },
    Anchor {
        // 5-hydroxylysine
        residue: "LYZ",
        atoms: ["CG", "CD", "OH"],
        bond: 1.42,
        angle: 114.0,
        torsion: LinkTorsion::Free,
        near: &[("OH", 1), ("CD", 2), ("CG", 3), ("CE", 3)],
        linkage: "O-linked",
        both_chairs: false,
    },
    Anchor {
        residue: "CYS",
        atoms: ["CA", "CB", "SG"],
        bond: 1.81,
        angle: 101.0,
        torsion: LinkTorsion::Free,
        near: &[("SG", 1), ("CB", 2), ("CA", 3)],
        linkage: "S-linked",
        both_chairs: false,
    },
];

/// The anchor of a residue (CCD name), if it carries glycans.
pub fn anchor(residue: &str) -> Option<&'static Anchor> {
    ANCHORS.iter().find(|a| a.residue == residue)
}

impl Anchor {
    /// Bonds from C1 to a side-chain atom of the site residue, within three.
    pub fn bonds_to(&self, atom: &str) -> Option<u32> {
        self.near
            .iter()
            .find(|(name, _)| *name == atom)
            .map(|(_, b)| *b)
    }
}

impl LinkTorsion {
    /// Energy of psi (radians) at stiffness `kappa` (1/rad^2) and its derivative.
    pub fn energy(&self, psi: f64, kappa: f64) -> (f64, f64) {
        match *self {
            // the Asn amide, written as before
            LinkTorsion::Planar { centre: 180.0 } => {
                (kappa * (1.0 + psi.cos()), -kappa * psi.sin())
            }
            LinkTorsion::Planar { centre } => {
                let d = psi - centre.to_radians();
                (kappa * (1.0 - d.cos()), kappa * d.sin())
            }
            LinkTorsion::Free => (0.0, 0.0),
        }
    }

    /// psi values (radians) of the attachment grid: the planar value and +-15 deg, or every 30 deg.
    pub fn grid(&self) -> Vec<f64> {
        let wrap = |d: f64| {
            if d > 180.0 {
                d - 360.0
            } else if d <= -180.0 {
                d + 360.0
            } else {
                d
            }
        };
        match *self {
            LinkTorsion::Planar { centre } => [centre, centre - 15.0, centre + 15.0]
                .map(|d| wrap(d).to_radians())
                .to_vec(),
            LinkTorsion::Free => (0..12)
                .map(|k| (180.0 - 30.0 * k as f64).to_radians())
                .collect(),
        }
    }

    /// Where guided samples start, before the first attachment search.
    pub fn start(&self) -> f64 {
        match *self {
            LinkTorsion::Planar { centre } => centre.to_radians(),
            LinkTorsion::Free => std::f64::consts::PI,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asparagine_keeps_its_amide_grid_and_energy() {
        let asn = anchor("ASN").unwrap();
        let grid = asn.torsion.grid();
        let old = [180.0f64, 165.0, -165.0].map(f64::to_radians);
        assert_eq!(grid, old.to_vec());
        for psi in [-3.0, -1.0, 0.0, 0.5, 2.0, std::f64::consts::PI] {
            let (e, d) = asn.torsion.energy(psi, 32.8);
            assert_eq!(e, 32.8 * (1.0 + psi.cos()));
            assert_eq!(d, -32.8 * psi.sin());
        }
        assert_eq!((asn.bond, asn.angle), (1.45, 123.0));
    }

    #[test]
    fn tryptophan_wants_c1_in_the_ring_plane_cis_to_cb() {
        let trp = anchor("TRP").unwrap();
        let (e0, d0) = trp.torsion.energy(0.0, 10.0);
        let (e180, _) = trp.torsion.energy(std::f64::consts::PI, 10.0);
        assert!(e0.abs() < 1e-12 && d0.abs() < 1e-12 && e180 > 19.9);
        // the derivative is the slope of the energy
        let h = 1e-6;
        let (_, d) = trp.torsion.energy(0.3, 10.0);
        let fd =
            (trp.torsion.energy(0.3 + h, 10.0).0 - trp.torsion.energy(0.3 - h, 10.0).0) / (2.0 * h);
        assert!((d - fd).abs() < 1e-6);
    }

    #[test]
    fn free_links_search_the_whole_circle() {
        let ser = anchor("SER").unwrap();
        assert_eq!(ser.torsion.grid().len(), 12);
        assert_eq!(ser.torsion.energy(1.0, 50.0), (0.0, 0.0));
        // every anchor lists its link at one bond and A, B within reach
        for a in &ANCHORS {
            assert_eq!(a.bonds_to(a.atoms[2]), Some(1), "{}", a.residue);
            assert_eq!(a.bonds_to(a.atoms[1]), Some(2), "{}", a.residue);
            assert_eq!(a.bonds_to(a.atoms[0]), Some(3), "{}", a.residue);
        }
    }
}
