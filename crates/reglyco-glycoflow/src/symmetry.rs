//! Crystal symmetry for the site environment (`glycoflow/fitting/site.py::_symmetry_environment`).
//!
//! The operator table (`data/sohncke_spacegroups.json`) holds every setting of the 65 Sohncke
//! space groups (the only ones a protein crystal can have), generated with gemmi
//! (`gemmi.spacegroup_table()`, operators as integer rotation matrices and translations over
//! `den` = 24).

use std::collections::HashSet;
use std::sync::OnceLock;

use serde::Deserialize;

const TABLE_JSON: &str = include_str!("../data/sohncke_spacegroups.json");

#[derive(Deserialize)]
struct Table {
    den: i32,
    groups: Vec<GroupEntry>,
}

#[derive(Deserialize)]
struct GroupEntry {
    number: u32,
    hm: String,
    xhm: String,
    ccp4: i32,
    ops: Vec<[i32; 12]>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| serde_json::from_str(TABLE_JSON).expect("embedded space-group table"))
}

/// One symmetry operator in fractional coordinates: `x' = rot x + tran`.
#[derive(Debug, Clone, Copy)]
pub struct SymOp {
    pub rot: [[f64; 3]; 3],
    pub tran: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct SpaceGroup {
    pub number: u32,
    pub hm: String,
    pub ops: Vec<SymOp>,
}

fn normalize(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_uppercase()
}

impl SpaceGroup {
    fn from_entry(entry: &GroupEntry, den: i32) -> Self {
        let d = den as f64;
        let ops = entry
            .ops
            .iter()
            .map(|o| SymOp {
                rot: [
                    [o[0] as f64 / d, o[1] as f64 / d, o[2] as f64 / d],
                    [o[3] as f64 / d, o[4] as f64 / d, o[5] as f64 / d],
                    [o[6] as f64 / d, o[7] as f64 / d, o[8] as f64 / d],
                ],
                tran: [o[9] as f64 / d, o[10] as f64 / d, o[11] as f64 / d],
            })
            .collect();
        Self {
            number: entry.number,
            hm: entry.hm.clone(),
            ops,
        }
    }

    /// Standard setting of a space-group number (as stored in a CCP4/MRC header), Sohncke
    /// groups only.
    pub fn from_number(number: u32) -> Option<Self> {
        let t = table();
        t.groups
            .iter()
            .find(|g| g.number == number && g.ccp4 == number as i32)
            .or_else(|| t.groups.iter().find(|g| g.number == number))
            .map(|g| Self::from_entry(g, t.den))
    }

    /// Hermann-Mauguin symbol (e.g. a PDB CRYST1 record "P 1 21 1"), any tabulated setting.
    pub fn from_hm(symbol: &str) -> Option<Self> {
        let t = table();
        let key = normalize(symbol);
        t.groups
            .iter()
            .find(|g| normalize(&g.hm) == key || normalize(&g.xhm) == key)
            .map(|g| Self::from_entry(g, t.den))
    }
}

/// Unit cell with gemmi's orthogonalization convention (a along x, b in the xy plane).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct UnitCell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    #[serde(skip)]
    pub orth: [[f64; 3]; 3],
    #[serde(skip)]
    pub frac: [[f64; 3]; 3],
}

fn cos_sin_deg(angle: f64) -> (f64, f64) {
    if angle == 90.0 {
        (0.0, 1.0)
    } else {
        let r = angle.to_radians();
        (r.cos(), r.sin())
    }
}

fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn invert(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |i: usize, j: usize| {
        m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]
            - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = c(j, i) / det;
        }
    }
    Some(out)
}

impl UnitCell {
    pub fn new(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> Option<Self> {
        let (cos_a, _) = cos_sin_deg(alpha);
        let (cos_b, sin_b) = cos_sin_deg(beta);
        let (cos_g, sin_g) = cos_sin_deg(gamma);
        let cos_a_star = (cos_b * cos_g - cos_a) / (sin_b * sin_g);
        let sin_a_star = (1.0 - cos_a_star * cos_a_star).sqrt();
        let orth = [
            [a, b * cos_g, c * cos_b],
            [0.0, b * sin_g, -c * cos_a_star * sin_b],
            [0.0, 0.0, c * sin_b * sin_a_star],
        ];
        let frac = invert(&orth)?;
        Some(Self {
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            orth,
            frac,
        })
    }

    pub fn volume(&self) -> f64 {
        self.orth[0][0] * self.orth[1][1] * self.orth[2][2]
    }

    pub fn fractionalize(&self, p: [f64; 3]) -> [f64; 3] {
        mat_vec(&self.frac, p)
    }

    pub fn orthogonalize(&self, f: [f64; 3]) -> [f64; 3] {
        mat_vec(&self.orth, f)
    }

    /// Same cell within a relative tolerance (lengths) and 0.1 degree (angles).
    pub fn matches(&self, other: &UnitCell, rel: f64) -> bool {
        let close = |x: f64, y: f64| (x - y).abs() <= rel * x.abs().max(y.abs());
        close(self.a, other.a)
            && close(self.b, other.b)
            && close(self.c, other.c)
            && (self.alpha - other.alpha).abs() < 0.1
            && (self.beta - other.beta).abs() < 0.1
            && (self.gamma - other.gamma).abs() < 0.1
    }
}

/// Crystal records of a PDB file: CRYST1 cell and space-group symbol.
pub fn parse_cryst1(pdb_text: &str) -> Option<(UnitCell, String)> {
    let line = pdb_text.lines().find(|l| l.starts_with("CRYST1"))?;
    let field = |a: usize, b: usize| line.get(a..b.min(line.len())).map(str::trim);
    let num = |a: usize, b: usize| field(a, b).and_then(|s| s.parse::<f64>().ok());
    let cell = UnitCell::new(
        num(6, 15)?,
        num(15, 24)?,
        num(24, 33)?,
        num(33, 40)?,
        num(40, 47)?,
        num(47, 54)?,
    )?;
    let symbol = field(55, 66).unwrap_or("").to_string();
    Some((cell, symbol))
}

/// Resolution from a PDB `REMARK   2 RESOLUTION.` record.
pub fn parse_resolution(pdb_text: &str) -> Option<f64> {
    pdb_text
        .lines()
        .find(|l| l.starts_with("REMARK   2 RESOLUTION."))
        .and_then(|l| {
            l["REMARK   2 RESOLUTION.".len()..]
                .split_whitespace()
                .next()
        })
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|r| r.is_finite() && *r > 0.0)
}

/// One environment atom (protein, ligand, other glycan, or a symmetry image of one).
#[derive(Debug, Clone)]
pub struct EnvAtom {
    /// position rounded to f32 precision, as the Python reference stores it
    pub position: [f64; 3],
    pub atomic_number: f64,
    pub chain: String,
    pub residue_number: i32,
    pub residue_name: String,
    pub atom_name: String,
    /// symmetry operator index (0 = identity)
    pub op: usize,
    /// lattice shift
    pub shift: [i32; 3],
}

impl EnvAtom {
    /// The deposited copy (identity operator, no lattice shift).
    pub fn is_identity(&self) -> bool {
        self.op == 0 && self.shift == [0, 0, 0]
    }
}

#[inline]
fn f32_round(v: [f64; 3]) -> [f64; 3] {
    v.map(|x| x as f32 as f64)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Atoms of every symmetry image within `radius` of `center`: each operator image is moved,
/// atom by atom, to the unit cell nearest the centre, then lattice shifts -1..1 are applied;
/// coincident positions (0.01 A) are kept once. Without a crystal (`None`, cryo-EM / NMR) the
/// deposited atoms within the radius are returned.
pub fn expand_environment(
    atoms: &[EnvAtom],
    crystal: Option<(&UnitCell, &SpaceGroup)>,
    center: [f64; 3],
    radius: f64,
) -> Vec<EnvAtom> {
    let Some((cell, group)) = crystal else {
        return atoms
            .iter()
            .filter(|a| dist(a.position, center) < radius)
            .map(|a| EnvAtom {
                position: f32_round(a.position),
                ..a.clone()
            })
            .collect();
    };
    let fc = cell.fractionalize(center);
    let frac: Vec<[f64; 3]> = atoms
        .iter()
        .map(|a| cell.fractionalize(a.position))
        .collect();
    // shift order of numpy.meshgrid([-1,0,1], [-1,0,1], [-1,0,1]).reshape(3, -1).T
    let mut shifts = Vec::with_capacity(27);
    for y in -1..=1 {
        for x in -1..=1 {
            for z in -1..=1 {
                shifts.push([x, y, z]);
            }
        }
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (k, op) in group.ops.iter().enumerate() {
        let moved: Vec<[f64; 3]> = frac
            .iter()
            .map(|f| {
                let g = [0, 1, 2].map(|i| {
                    op.rot[i][0] * f[0] + op.rot[i][1] * f[1] + op.rot[i][2] * f[2] + op.tran[i]
                });
                [0, 1, 2].map(|i| g[i] - (g[i] - fc[i]).round_ties_even())
            })
            .collect();
        for shift in &shifts {
            for (atom, g) in atoms.iter().zip(&moved) {
                let cart = cell.orthogonalize([
                    g[0] + shift[0] as f64,
                    g[1] + shift[1] as f64,
                    g[2] + shift[2] as f64,
                ]);
                if dist(cart, center) >= radius {
                    continue;
                }
                let key = cart.map(|x| (x * 100.0).round_ties_even() as i64);
                if !seen.insert(key) {
                    continue;
                }
                out.push(EnvAtom {
                    position: f32_round(cart),
                    op: k,
                    shift: *shift,
                    ..atom.clone()
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_the_sohncke_groups() {
        let numbers = table()
            .groups
            .iter()
            .map(|g| g.number)
            .collect::<HashSet<_>>();
        assert_eq!(numbers.len(), 65);
        let p21 = SpaceGroup::from_number(4).unwrap();
        assert_eq!(p21.ops.len(), 2);
        assert_eq!(SpaceGroup::from_hm("P 1 21 1").unwrap().number, 4);
        assert_eq!(SpaceGroup::from_hm("P 41 21 2").unwrap().ops.len(), 8);
        assert!(
            SpaceGroup::from_number(2).is_none(),
            "P-1 is not a Sohncke group"
        );
    }

    #[test]
    fn cell_round_trip_and_images() {
        let cell = UnitCell::new(50.183, 158.406, 66.905, 90.0, 109.04, 90.0).unwrap();
        let p = [3.0, -4.0, 20.0];
        let back = cell.orthogonalize(cell.fractionalize(p));
        assert!(dist(p, back) < 1e-9);
        let group = SpaceGroup::from_number(4).unwrap();
        let atom = EnvAtom {
            position: [10.0, 20.0, 5.0],
            atomic_number: 6.0,
            chain: "A".into(),
            residue_number: 1,
            residue_name: "GLY".into(),
            atom_name: "CA".into(),
            op: 0,
            shift: [0, 0, 0],
        };
        let near = expand_environment(&[atom], Some((&cell, &group)), [10.0, 20.0, 5.0], 90.0);
        assert!(near.iter().any(|a| a.is_identity()));
        // the 2_1 screw image (b/2 = 79 A away) is within 90 A
        assert!(near.iter().any(|a| a.op == 1));
        let positions = near
            .iter()
            .map(|a| a.position.map(|x| (x * 100.0).round() as i64))
            .collect::<HashSet<_>>();
        assert_eq!(positions.len(), near.len());
    }

    #[test]
    fn cryst1_and_resolution_records() {
        let text = "REMARK   2 RESOLUTION.    1.85 ANGSTROMS.\nCRYST1   50.183  158.406   66.905  90.00 109.04  90.00 P 1 21 1      8          \n";
        let (cell, symbol) = parse_cryst1(text).unwrap();
        assert_eq!(symbol, "P 1 21 1");
        assert!((cell.b - 158.406).abs() < 1e-9);
        assert_eq!(parse_resolution(text), Some(1.85));
    }
}
