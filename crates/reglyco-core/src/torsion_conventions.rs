//! The definitions of the glycosidic torsions stored in GlycoShape.
//!
//! Every GlycoShape entry stores φ, ψ and ω of its glycosidic linkages. The
//! entries analysed with the IUPAC definitions (GlycanAnalysisPipeline from
//! PR #9 on) declare them in `torsions.conventions` of their `info.json`,
//! `version: 2`; the entries published before carry no version and use the
//! earlier definitions, version 1. Both are served side by side while the
//! collection is re-analysed, so a reader of a GlycoShape torsion follows the
//! version of the entry it reads.
//!
//! | torsion | version 1 | version 2 (IUPAC-IUB, heavy atoms) |
//! |---|---|---|
//! | φ | O5′–C1′–Ox–Cx (O4′ of an aldofuranose, O6′–C2′ of a sialic acid) | the same |
//! | ψ | C1′–Ox–Cx–C(x−1) or C1′–Ox–Cx–C(x+1), whichever the bond table listed first | C1′–Ox–Cx–C(x−1) |
//! | ω | C4–C5–C6–O6 for 1–6, C6–C7–C8–O8 for 2–8 | O5–C5–C6–O6 for 1–6, O7–C7–C8–O8 for 2–8, O8–C8–C9–O9 for 2–9, O4–C4–C5–O5 for 1–5 to Araf, O5–C2–C1–O1 for 2–1 to Fruf |
//!
//! The ω of version 1 is the IUPAC ω plus about 124° (2–8: 127°), so its
//! rotamers gg, gt and tg lie near +60°, 180° and −60° instead of −60°, +60°
//! and 180°. The ψ of version 1 cannot be told from the rule: the served
//! torsion references carry the four atoms of each stored torsion, and a
//! torsion measured on those atoms is comparable whatever the version. The
//! rules here are for a reference (or a measurement) without them.
//!
//! ReGlyco's own glycosidic torsions are the IUPAC ones (version 2). The
//! protein–glycan attachment torsions (φ = CG–ND2–C1–O5, ψ = CB–CG–ND2–C1 …)
//! are ReGlyco's and are not described here.

use std::fmt;

/// Version of GlycoShape's torsion definitions (`torsions.conventions.version`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum GlycoShapeTorsionConvention {
    /// Entries published before the IUPAC re-analysis (no version declared).
    #[serde(rename = "glycoshape-v1")]
    V1,
    /// `torsions.conventions.version` 2: the IUPAC-IUB definitions.
    #[serde(rename = "glycoshape-v2")]
    V2,
}

impl GlycoShapeTorsionConvention {
    /// The definitions of ReGlyco's own glycosidic torsions.
    pub const IUPAC: Self = Self::V2;

    /// `torsions.conventions.version` of an entry with these definitions.
    pub fn version(self) -> u32 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
        }
    }

    /// The definitions of `torsions.conventions.version`; `None` for a
    /// version this reader does not know.
    pub fn from_version(version: u64) -> Option<Self> {
        match version {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            _ => None,
        }
    }

    /// The label the GlycoShape API writes as `angle_convention`.
    pub fn label(self) -> &'static str {
        match self {
            Self::V1 => "glycoshape-v1",
            Self::V2 => "glycoshape-v2",
        }
    }

    /// The definitions an `angle_convention` label names; `None` for a label
    /// this reader does not know.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "glycoshape-v1" | "v1" | "1" => Some(Self::V1),
            "glycoshape-v2" | "v2" | "2" | "iupac" => Some(Self::V2),
            _ => None,
        }
    }

    /// The fourth atom of ψ = C1′–Ox–Cx–A (C2′ for a 2-ketose donor) on the
    /// acceptor, for the acceptor oxygen at `position`: C(x−1); for x = 1, the
    /// ring oxygen of an aldose and C2 of a 2-ketose.
    ///
    /// Version 1 took C(x−1) or C(x+1) as the bond table listed them, which no
    /// rule recovers; C(x−1) is returned for it too (right for about half of
    /// its ψ columns), and a version 1 ψ is to be measured on the atoms the
    /// reference declares.
    pub fn psi_fourth_atom(self, acceptor: SugarRing, position: u8) -> Option<String> {
        match position {
            0 => None,
            1 if acceptor.anomeric_carbon() == 1 => Some(acceptor.ring_oxygen().into()),
            1 => Some("C2".into()),
            x => Some(format!("C{}", x - 1)),
        }
    }

    /// The four acceptor atoms of ω for the acceptor oxygen at `position`, or
    /// `None` when the linkage has no ω in these definitions.
    ///
    /// Version 2 gives an ω to every acceptor carbon Cx outside the ring:
    /// O–C(x−1)–Cx–Ox, the oxygen being the ring oxygen when C(x−1) is in the
    /// ring and the hydroxyl oxygen of C(x−1) otherwise (C2 and the ring
    /// oxygen for 2–1 to a 2-ketose). Version 1 has ω only for 1–6
    /// (C4–C5–C6–O6) and for 2–8 to a sialic acid (C6–C7–C8–O8).
    pub fn omega_atoms(self, acceptor: SugarRing, position: u8) -> Option<[String; 4]> {
        match self {
            Self::V1 => match (acceptor, position) {
                (SugarRing::Ketopyranose, 8) => Some(["C6", "C7", "C8", "O8"].map(String::from)),
                (SugarRing::Ketopyranose, _) => None,
                (_, 6) => Some(["C4", "C5", "C6", "O6"].map(String::from)),
                _ => None,
            },
            Self::V2 => {
                let ring_oxygen = acceptor.ring_oxygen().to_string();
                if position > acceptor.last_ring_carbon() {
                    let previous = position - 1;
                    let first = if previous == acceptor.last_ring_carbon() {
                        ring_oxygen
                    } else {
                        format!("O{previous}")
                    };
                    Some([
                        first,
                        format!("C{previous}"),
                        format!("C{position}"),
                        format!("O{position}"),
                    ])
                } else if position >= 1 && position < acceptor.anomeric_carbon() {
                    let next = position + 1;
                    Some([
                        ring_oxygen,
                        format!("C{next}"),
                        format!("C{position}"),
                        format!("O{position}"),
                    ])
                } else {
                    None
                }
            }
        }
    }

    /// The nominal angles of the gg, gt and tg rotamers of the ω of a 1–6
    /// linkage to a D-hexopyranose in these definitions (version 1: the IUPAC
    /// angle plus 120°; the measured offset is about 124°).
    pub fn omega_rotamer_centres(self) -> [(&'static str, f64); 3] {
        match self {
            Self::V1 => [("gg", 60.0), ("gt", 180.0), ("tg", -60.0)],
            Self::V2 => [("gg", -60.0), ("gt", 60.0), ("tg", 180.0)],
        }
    }
}

impl fmt::Display for GlycoShapeTorsionConvention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// The ring of a monosaccharide, as far as the atoms of its torsions need it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SugarRing {
    /// C1–C5 and O5 (Glc, Gal, Man, GlcNAc, Fuc, Xyl …).
    Aldopyranose,
    /// C1–C4 and O4 (Araf, Galf, Ribf).
    Aldofuranose,
    /// C2–C6 and O6 (sialic acids, Kdn, Kdo).
    Ketopyranose,
    /// C2–C5 and O5 (Fruf).
    Ketofuranose,
}

impl SugarRing {
    /// The ring oxygen.
    pub fn ring_oxygen(self) -> &'static str {
        match self {
            Self::Aldopyranose | Self::Ketofuranose => "O5",
            Self::Aldofuranose => "O4",
            Self::Ketopyranose => "O6",
        }
    }

    /// The number of the anomeric carbon.
    pub fn anomeric_carbon(self) -> u8 {
        match self {
            Self::Aldopyranose | Self::Aldofuranose => 1,
            Self::Ketopyranose | Self::Ketofuranose => 2,
        }
    }

    /// The number of the ring carbon bonded to the ring oxygen on the side
    /// away from the anomeric carbon.
    pub fn last_ring_carbon(self) -> u8 {
        match self {
            Self::Aldopyranose | Self::Ketofuranose => 5,
            Self::Aldofuranose => 4,
            Self::Ketopyranose => 6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GlycoShapeTorsionConvention as Convention, SugarRing};

    fn atoms(values: [&str; 4]) -> Option<[String; 4]> {
        Some(values.map(String::from))
    }

    #[test]
    fn versions_and_labels_round_trip() {
        for convention in [Convention::V1, Convention::V2] {
            assert_eq!(
                Convention::from_version(u64::from(convention.version())),
                Some(convention)
            );
            assert_eq!(Convention::from_label(convention.label()), Some(convention));
            assert_eq!(
                serde_json::to_value(convention).unwrap(),
                serde_json::json!(convention.label())
            );
        }
        assert_eq!(Convention::from_version(3), None);
        assert_eq!(Convention::from_label("glycoshape-v3"), None);
        assert_eq!(Convention::IUPAC, Convention::V2);
    }

    #[test]
    fn omega_follows_each_version() {
        use SugarRing::*;
        // 1-6 to a hexopyranose: C4-C5-C6-O6 before, the IUPAC O5-C5-C6-O6 now.
        assert_eq!(
            Convention::V1.omega_atoms(Aldopyranose, 6),
            atoms(["C4", "C5", "C6", "O6"])
        );
        assert_eq!(
            Convention::V2.omega_atoms(Aldopyranose, 6),
            atoms(["O5", "C5", "C6", "O6"])
        );
        // 2-8 between sialic acids.
        assert_eq!(
            Convention::V1.omega_atoms(Ketopyranose, 8),
            atoms(["C6", "C7", "C8", "O8"])
        );
        assert_eq!(
            Convention::V2.omega_atoms(Ketopyranose, 8),
            atoms(["O7", "C7", "C8", "O8"])
        );
        // ω that only version 2 has: 2-9, Araf 1-5, Fruf 2-1; Galf 1-6.
        assert_eq!(Convention::V1.omega_atoms(Ketopyranose, 9), None);
        assert_eq!(
            Convention::V2.omega_atoms(Ketopyranose, 9),
            atoms(["O8", "C8", "C9", "O9"])
        );
        assert_eq!(Convention::V1.omega_atoms(Aldofuranose, 5), None);
        assert_eq!(
            Convention::V2.omega_atoms(Aldofuranose, 5),
            atoms(["O4", "C4", "C5", "O5"])
        );
        assert_eq!(
            Convention::V2.omega_atoms(Aldofuranose, 6),
            atoms(["O5", "C5", "C6", "O6"])
        );
        assert_eq!(
            Convention::V2.omega_atoms(Ketofuranose, 1),
            atoms(["O5", "C2", "C1", "O1"])
        );
        // Ring positions have no ω in either version.
        for convention in [Convention::V1, Convention::V2] {
            for position in 1..=4 {
                assert_eq!(convention.omega_atoms(Aldopyranose, position), None);
            }
            assert_eq!(convention.omega_atoms(Ketopyranose, 3), None);
        }
    }

    #[test]
    fn psi_ends_at_the_previous_carbon_or_the_ring() {
        use SugarRing::*;
        for convention in [Convention::V1, Convention::V2] {
            assert_eq!(
                convention.psi_fourth_atom(Aldopyranose, 4).as_deref(),
                Some("C3")
            );
            assert_eq!(
                convention.psi_fourth_atom(Aldopyranose, 6).as_deref(),
                Some("C5")
            );
            assert_eq!(
                convention.psi_fourth_atom(Ketopyranose, 8).as_deref(),
                Some("C7")
            );
            // 1-1 to an aldose: the ring oxygen; 2-1 to a 2-ketose: C2.
            assert_eq!(
                convention.psi_fourth_atom(Aldopyranose, 1).as_deref(),
                Some("O5")
            );
            assert_eq!(
                convention.psi_fourth_atom(Ketofuranose, 1).as_deref(),
                Some("C2")
            );
        }
    }

    #[test]
    fn rotamer_centres_move_with_the_omega_definition() {
        let centre = |convention: Convention, name: &str| {
            convention
                .omega_rotamer_centres()
                .into_iter()
                .find(|(rotamer, _)| *rotamer == name)
                .map(|(_, angle)| angle)
                .unwrap()
        };
        for name in ["gg", "gt", "tg"] {
            // Version 1 is the IUPAC angle plus 120 degrees (wrapped).
            let shifted = (centre(Convention::V2, name) + 120.0 + 180.0).rem_euclid(360.0) - 180.0;
            let v1 = centre(Convention::V1, name);
            assert!(((shifted - v1 + 180.0).rem_euclid(360.0) - 180.0).abs() < 1e-9);
        }
    }
}
