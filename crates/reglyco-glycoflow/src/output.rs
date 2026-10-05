//! Fitted glycans written back into the model (`glycoflow/fitting/output.py`): residues named
//! after the deposited (or requested) CCD components with PDB atom names, CONECT records for
//! every glycan bond and a LINK record for the protein-glycan bond, so tools that read bonds
//! only from CONECT/LINK (GlySys/ReGlyco) see the complete topology.

use std::collections::{BTreeMap, BTreeSet};

use glysys::{BuildOptions, ResidueId, Structure, read_pdb_str};

use crate::error::{Result, invalid};
use crate::problem::{SiteProblem, V3};
use crate::site::{Site, atom_rename, ccd_of, sequence_paths};

/// Output identity of one glycan residue.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResidueNaming {
    pub path: String,
    pub chain: String,
    pub number: i32,
    pub insertion_code: Option<char>,
    pub ccd: String,
    /// "deposited" (same residue as in the input model) or "generated"
    pub source: String,
}

/// Names and numbers of the fitted residues: the deposited residue at the same tree path when
/// there is one, otherwise a new residue number in the site's chain.
pub fn residue_naming(
    problem: &SiteProblem,
    site: &Site,
    protein: &Structure,
) -> Result<Vec<ResidueNaming>> {
    let tokens = sequence_paths(&problem.sequence)?;
    let chain = site.residue.chain.clone();
    let mut used: BTreeSet<i32> = protein
        .residues()
        .iter()
        .filter(|r| r.id.chain == chain)
        .map(|r| r.id.number)
        .collect();
    if let Some(dep) = &site.deposited {
        used.extend(
            dep.residues
                .iter()
                .filter(|r| r.id.chain == chain)
                .map(|r| r.id.number),
        );
    }
    let mut next = used.iter().max().copied().unwrap_or(0) + 1;
    let mut paths: Vec<&String> = tokens.keys().collect();
    paths.sort_by(|a, b| (a.matches('/').count(), *a).cmp(&(b.matches('/').count(), *b)));
    let mut out = Vec::new();
    for path in paths {
        let (token, anomer) = &tokens[path];
        let ccd = ccd_of(token, *anomer)
            .ok_or_else(|| invalid(format!("no CCD component for {token} ({anomer})")))?;
        let deposited = site
            .deposited
            .as_ref()
            .and_then(|d| d.by_path(path))
            .filter(|r| r.name == ccd);
        match deposited {
            Some(r) => out.push(ResidueNaming {
                path: path.clone(),
                chain: r.id.chain.clone(),
                number: r.id.number,
                insertion_code: r.id.insertion_code,
                ccd: ccd.to_string(),
                source: "deposited".into(),
            }),
            None => {
                out.push(ResidueNaming {
                    path: path.clone(),
                    chain: chain.clone(),
                    number: next,
                    insertion_code: None,
                    ccd: ccd.to_string(),
                    source: "generated".into(),
                });
                next += 1;
            }
        }
    }
    Ok(out)
}

/// A glycan pose to write.
pub struct PlacedGlycan<'a> {
    pub problem: &'a SiteProblem,
    pub site: &'a Site,
    pub naming: &'a [ResidueNaming],
    pub x: &'a [V3],
}

fn atom_field(name: &str) -> String {
    if name.len() >= 4 {
        name.to_string()
    } else {
        format!(" {name:<3}")
    }
}

/// PDB records of glycans: (HETATM lines, CONECT lines, LINK lines); serials from `first_serial`.
fn glycan_records(
    glycans: &[PlacedGlycan],
    first_serial: u32,
) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let mut serial = first_serial;
    let (mut atoms, mut conect, mut links) = (Vec::new(), Vec::new(), Vec::new());
    for g in glycans {
        let p = g.problem;
        let naming: BTreeMap<&str, &ResidueNaming> =
            g.naming.iter().map(|n| (n.path.as_str(), n)).collect();
        let mut serial_of = vec![None; p.n_atoms];
        // residues in output order, atoms in template order within each residue
        for n in g.naming {
            let back: Vec<(&str, &str)> = atom_rename(&n.ccd)
                .iter()
                .map(|(pdb, gly)| (*gly, *pdb))
                .collect();
            for i in (0..p.n_atoms).filter(|&i| p.glycan.res_paths[i] == n.path) {
                let gname = p.glycan.atom_names[i].as_str();
                let name = back
                    .iter()
                    .find(|(gly, _)| *gly == gname)
                    .map_or(gname, |(_, pdb)| *pdb);
                let x = g.x[i];
                atoms.push(format!(
                    "HETATM{serial:>5} {:<4} {:>3} {:1}{:>4}{:1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
                    atom_field(name),
                    n.ccd,
                    n.chain,
                    n.number,
                    n.insertion_code.unwrap_or(' '),
                    x[0],
                    x[1],
                    x[2],
                    1.0,
                    30.0,
                    p.glycan.elements[i]
                ));
                serial_of[i] = Some(serial);
                serial += 1;
            }
        }
        for &[a, b] in &p.glycan.bonds {
            if let (Some(sa), Some(sb)) = (serial_of[a], serial_of[b]) {
                conect.push(format!("CONECT{sa:>5}{sb:>5}"));
            }
        }
        let root = naming
            .get("r")
            .ok_or_else(|| invalid("glycan has no root residue"))?;
        let mut record = vec![b' '; 80];
        let mut put = |at: usize, s: &str| record[at..at + s.len()].copy_from_slice(s.as_bytes());
        put(0, "LINK");
        put(12, &format!("{:>4}", g.site.anchor_names[2]));
        put(17, &format!("{:>3}", g.site.residue_name));
        put(21, &g.site.residue.chain);
        put(22, &format!("{:>4}", g.site.residue.number));
        put(42, &format!("{:>4}", "C1"));
        put(47, &format!("{:>3}", root.ccd));
        put(51, &root.chain);
        put(52, &format!("{:>4}", root.number));
        let link = g.site.anchor[2];
        let c1 = g.x[p.c1];
        let d = ((link[0] - c1[0]).powi(2) + (link[1] - c1[1]).powi(2) + (link[2] - c1[2]).powi(2))
            .sqrt();
        put(73, &format!("{d:>5.2}"));
        links.push(
            String::from_utf8(record)
                .expect("ASCII LINK record")
                .trim_end()
                .to_string(),
        );
    }
    Ok((atoms, conect, links))
}

/// Protein records split by type: (LINK/SSBOND, ATOM/HETATM/TER, CONECT), and the largest serial.
/// LINK records of residues that are no longer in the structure (removed target glycans) are
/// dropped.
fn protein_records(protein: &Structure) -> (Vec<String>, Vec<String>, Vec<String>, u32) {
    let text = protein.to_pdb_string();
    let present: BTreeSet<(String, i32)> = protein
        .residues()
        .into_iter()
        .map(|r| (r.id.chain, r.id.number))
        .collect();
    let side = |line: &str, chain: usize| -> Option<(String, i32)> {
        let c = line.get(chain..chain + 1)?.trim().to_string();
        let n = line.get(chain + 1..chain + 5)?.trim().parse().ok()?;
        Some((c, n))
    };
    let (mut head, mut body, mut conect) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines() {
        if line.starts_with("LINK") {
            let keep = [21, 51]
                .iter()
                .all(|&c| side(line, c).is_none_or(|r| present.contains(&r)));
            if keep {
                head.push(line.to_string());
            }
        } else if line.starts_with("SSBOND") {
            head.push(line.to_string());
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") || line.starts_with("TER")
        {
            body.push(line.to_string());
        } else if line.starts_with("CONECT") {
            conect.push(line.to_string());
        }
    }
    let max_serial = protein.atoms().iter().map(|a| a.id.0).max().unwrap_or(0);
    (head, body, conect, max_serial)
}

/// Deposited waters closer than this to a fitted glycan heavy atom are removed from the output:
/// the objective ignores waters on purpose (a model built without the glycan may have placed
/// waters into glycan density), so overlapping waters are artefacts of the replacement.
pub const WATER_OVERLAP_ANGSTROM: f64 = 2.6;

fn is_water_record(line: &str) -> bool {
    (line.starts_with("HETATM") || line.starts_with("ATOM"))
        && matches!(
            line.get(17..20).map(str::trim),
            Some("HOH" | "WAT" | "DOD" | "H2O")
        )
}

fn record_position(line: &str) -> Option<[f64; 3]> {
    Some([
        line.get(30..38)?.trim().parse().ok()?,
        line.get(38..46)?.trim().parse().ok()?,
        line.get(46..54)?.trim().parse().ok()?,
    ])
}

/// Body records without the waters that overlap any fitted glycan atom, and how many were removed.
fn without_overlapping_waters(body: &[String], glycans: &[PlacedGlycan]) -> (Vec<String>, usize) {
    let limit = WATER_OVERLAP_ANGSTROM * WATER_OVERLAP_ANGSTROM;
    let mut removed = 0;
    let kept = body
        .iter()
        .filter(|line| {
            if !is_water_record(line) {
                return true;
            }
            let Some(p) = record_position(line) else {
                return true;
            };
            let overlaps = glycans.iter().any(|g| {
                g.x.iter()
                    .skip(1)
                    .any(|a| (0..3).map(|i| (a[i] - p[i]).powi(2)).sum::<f64>() < limit)
            });
            if overlaps {
                removed += 1;
            }
            !overlaps
        })
        .cloned()
        .collect();
    (kept, removed)
}

/// Number of deposited waters that the output drops because they overlap the fitted glycans.
pub fn overlapping_water_count(protein: &Structure, glycans: &[PlacedGlycan]) -> usize {
    let (_, body, _, _) = protein_records(protein);
    without_overlapping_waters(&body, glycans).1
}

/// The protein (target glycans already removed) with the fitted glycans, as a GlySys structure.
pub fn fitted_structure(protein: &Structure, glycans: &[PlacedGlycan]) -> Result<Structure> {
    let (head, body, conect, max_serial) = protein_records(protein);
    let (body, _) = without_overlapping_waters(&body, glycans);
    let (atoms, g_conect, links) = glycan_records(glycans, max_serial + 1)?;
    let mut text = Vec::new();
    text.extend(head);
    text.extend(links);
    text.extend(body);
    text.extend(atoms);
    text.push("TER".into());
    text.extend(conect);
    text.extend(g_conect);
    text.push("END".into());
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    Ok(read_pdb_str(&(text.join("\n") + "\n"), &options)?)
}

/// Multi-model PDB of candidate poses: one MODEL per entry (protein + glycans), labelled by
/// `REMARK 250` records.
pub fn candidates_pdb(
    protein: &Structure,
    models: &[(String, Vec<PlacedGlycan>)],
) -> Result<String> {
    let (head, body, conect, max_serial) = protein_records(protein);
    let mut out = Vec::new();
    for (k, (label, _)) in models.iter().enumerate() {
        out.push(format!("REMARK 250 MODEL {}: {label}", k + 1));
    }
    out.extend(head);
    let mut glycan_conect = Vec::new();
    for (k, (_, glycans)) in models.iter().enumerate() {
        let (atoms, g_conect, links) = glycan_records(glycans, max_serial + 1)?;
        if k == 0 {
            out.extend(links);
            glycan_conect = g_conect;
        }
        out.push(format!("MODEL     {:>4}", k + 1));
        out.extend(without_overlapping_waters(&body, glycans).0);
        out.extend(atoms);
        out.push("TER".into());
        out.push("ENDMDL".into());
    }
    out.extend(conect);
    out.extend(glycan_conect);
    out.push("END".into());
    Ok(out.join("\n") + "\n")
}

/// PDB records of glycans alone (no protein): HETATM records, then LINK and CONECT records, with
/// serials from 1. For overlays and per-frame trajectories of a pose.
pub fn glycan_pdb(glycans: &[PlacedGlycan]) -> Result<String> {
    let (atoms, conect, links) = glycan_records(glycans, 1)?;
    let mut out = links;
    out.extend(atoms);
    out.push("TER".into());
    out.extend(conect);
    out.push("END".into());
    Ok(out.join("\n") + "\n")
}

/// Residue ids of the fitted glycan (output naming).
pub fn fitted_residue_ids(naming: &[ResidueNaming]) -> Vec<ResidueId> {
    naming
        .iter()
        .map(|n| ResidueId {
            chain: n.chain.clone(),
            number: n.number,
            insertion_code: n.insertion_code,
        })
        .collect()
}
