//! The side chain of a glycosylated asparagine, rebuilt under its glycan.
//!
//! A deposit with a cis amide between ND2 and C1 (CB-CG-ND2-C1 near 0) has the glycan where the
//! density is and the side chain turned the wrong way: with CG, OD1 and ND2 where they are, C1 of
//! a trans link is about 2.4 A from the deposited one, and the map holds the glycan back. With
//! the side chain rebuilt from N, CA and CB the same link is often trans within 1 A of it.

use crate::problem::{V3, linkage, nerf};

/// Bonds (A) and angles (degrees) of the asparagine side chain (Engh and Huber).
const CB_CG: f64 = 1.516;
const CG_OD1: f64 = 1.231;
const CG_ND2: f64 = 1.328;
const CA_CB_CG: f64 = 112.6;
const CB_CG_OD1: f64 = 120.8;
const CB_CG_ND2: f64 = 116.4;

/// One conformation of the side chain, and where C1 of a trans link to it sits.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct AsnSideChain {
    /// N-CA-CB-CG and CA-CB-CG-OD1 (degrees)
    pub chi1: f64,
    pub chi2: f64,
    pub cg: V3,
    pub od1: V3,
    pub nd2: V3,
    /// C1 of the glycan with CB-CG-ND2-C1 = 180 degrees
    pub c1: V3,
}

impl AsnSideChain {
    /// The rebuilt atoms by name ([`crate::site::SiteOptions::moved`]).
    pub fn atoms(&self) -> [(&'static str, V3); 3] {
        [("CG", self.cg), ("OD1", self.od1), ("ND2", self.nd2)]
    }
}

/// The side chain at (chi1, chi2) in degrees on the backbone atoms N, CA and CB.
pub fn asn_side_chain(n: V3, ca: V3, cb: V3, chi1: f64, chi2: f64) -> AsnSideChain {
    let cg = nerf(n, ca, cb, CB_CG, CA_CB_CG.to_radians(), chi1.to_radians());
    let od1 = nerf(
        ca,
        cb,
        cg,
        CG_OD1,
        CB_CG_OD1.to_radians(),
        chi2.to_radians(),
    );
    let nd2 = nerf(
        ca,
        cb,
        cg,
        CG_ND2,
        CB_CG_ND2.to_radians(),
        (chi2 + 180.0).to_radians(),
    );
    let (bond, angle) = linkage("ASN");
    let c1 = nerf(cb, cg, nd2, bond, angle.to_radians(), 180f64.to_radians());
    AsnSideChain {
        chi1,
        chi2,
        cg,
        od1,
        nd2,
        c1,
    }
}

/// Every side chain on a grid of `step` degrees in chi1 and chi2.
pub fn asn_side_chains(n: V3, ca: V3, cb: V3, step: f64) -> Vec<AsnSideChain> {
    let count = (360.0 / step).round() as usize;
    let angle = |k: usize| -180.0 + k as f64 * step;
    (0..count)
        .flat_map(|i| (0..count).map(move |j| (i, j)))
        .map(|(i, j)| asn_side_chain(n, ca, cb, angle(i), angle(j)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::dihedral;

    fn distance(a: V3, b: V3) -> f64 {
        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
    }

    #[test]
    fn the_side_chain_has_its_geometry_and_a_trans_link() {
        let (n, ca, cb) = ([1.458, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.53, 1.43, 0.1]);
        let s = asn_side_chain(n, ca, cb, -65.0, 40.0);
        assert!((distance(cb, s.cg) - 1.516).abs() < 1e-9);
        assert!((distance(s.cg, s.od1) - 1.231).abs() < 1e-9);
        assert!((distance(s.cg, s.nd2) - 1.328).abs() < 1e-9);
        assert!((distance(s.nd2, s.c1) - 1.45).abs() < 1e-9);
        assert!((dihedral(n, ca, cb, s.cg).to_degrees() + 65.0).abs() < 1e-6);
        assert!((dihedral(ca, cb, s.cg, s.od1).to_degrees() - 40.0).abs() < 1e-6);
        // the amide is planar and the link trans
        assert!((dihedral(ca, cb, s.cg, s.nd2).to_degrees() + 140.0).abs() < 1e-6);
        assert!((dihedral(cb, s.cg, s.nd2, s.c1).to_degrees().abs() - 180.0).abs() < 1e-6);
        assert!((dihedral(s.od1, s.cg, s.nd2, s.c1).to_degrees()).abs() < 1e-6);
    }

    #[test]
    fn the_grid_covers_both_torsions() {
        let (n, ca, cb) = ([1.458, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.53, 1.43, 0.1]);
        let all = asn_side_chains(n, ca, cb, 30.0);
        assert_eq!(all.len(), 144);
        assert!(all.iter().any(|s| s.chi1 == -180.0 && s.chi2 == 150.0));
    }
}
