//! Multi-model PDB output and ensemble superposition (`api.write_pdb`, `api.core_atoms`,
//! `api.align_ensemble`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::geometry::P3;
use crate::linalg::{kabsch, matvec, mean, sub, V3};

/// `{:8.3f}` with Python's round-half-even on exact ties.
fn f83(x: f32) -> String {
    let v = x as f64;
    let scaled = v * 1000.0;
    if (scaled - scaled.trunc()).abs() == 0.5 {
        format!("{:8.3}", scaled.round_ties_even() / 1000.0)
    } else {
        format!("{v:8.3}")
    }
}

/// Atom identity needed to write a PDB.
pub struct PdbAtoms<'a> {
    pub atom_names: &'a [String],
    pub res_names: &'a [String],
    pub res_ids: &'a [i64],
    pub elements: &'a [String],
}

/// Multi-model PDB text with GLYCAM residue and atom names; `models` holds conformers [N] each.
pub fn write_pdb(atoms: &PdbAtoms, models: &[&[P3]]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (m, x) in models.iter().enumerate() {
        lines.push(format!("MODEL     {:4}", m + 1));
        for i in 0..atoms.atom_names.len() {
            let name = &atoms.atom_names[i];
            let pdb_name = if name.chars().count() == 4 {
                name.clone()
            } else {
                format!(" {name:<3}")
            };
            let mut line = String::with_capacity(80);
            write!(
                line,
                "ATOM  {:5} {} {:>3} X{:4}    {}{}{}  1.00  0.00          {:>2}",
                i + 1,
                pdb_name,
                atoms.res_names[i],
                atoms.res_ids[i],
                f83(x[i][0]),
                f83(x[i][1]),
                f83(x[i][2]),
                atoms.elements[i]
            )
            .unwrap();
            lines.push(line);
        }
        lines.push("ENDMDL".to_string());
    }
    lines.join("\n") + "\nEND\n"
}

/// Heavy-atom indices of an `n_residues` core grown from the reducing end (aglycone skipped),
/// always adding the attached residue with the largest subtree, ties towards the deeper residue.
pub fn core_atoms(res_ids: &[i64], bonds: &[[usize; 2]], n_residues: usize) -> Vec<usize> {
    let mut adj: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for &[a, b] in bonds {
        let (ra, rb) = (res_ids[a], res_ids[b]);
        if ra != rb {
            adj.entry(ra).or_default().insert(rb);
            adj.entry(rb).or_default().insert(ra);
        }
    }
    let root = res_ids[0];
    let mut children: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut depth: BTreeMap<i64, usize> = BTreeMap::from([(root, 0)]);
    let mut seen: BTreeSet<i64> = BTreeSet::from([root]);
    let mut stack = vec![root];
    while let Some(r) = stack.pop() {
        let ch: Vec<i64> = adj
            .get(&r)
            .map(|s| s.iter().copied().filter(|x| !seen.contains(x)).collect())
            .unwrap_or_default();
        seen.extend(ch.iter().copied());
        for c in &ch {
            depth.insert(*c, depth[&r] + 1);
        }
        stack.extend(ch.iter().copied());
        children.insert(r, ch);
    }
    fn size(r: i64, children: &BTreeMap<i64, Vec<i64>>) -> usize {
        1 + children
            .get(&r)
            .map(|c| c.iter().map(|&x| size(x, children)).sum())
            .unwrap_or(0)
    }
    let mut chosen: Vec<i64> = Vec::new();
    let mut frontier: Vec<i64> = children.get(&root).cloned().unwrap_or_default();
    while !frontier.is_empty() && chosen.len() < n_residues {
        let key = |r: i64| (size(r, &children), depth[&r], -r);
        let mut bi = 0;
        for k in 1..frontier.len() {
            if key(frontier[k]) > key(frontier[bi]) {
                bi = k;
            }
        }
        let best = frontier.remove(bi);
        chosen.push(best);
        frontier.extend(children.get(&best).cloned().unwrap_or_default());
    }
    if chosen.is_empty() {
        chosen.push(root);
    }
    (0..res_ids.len())
        .filter(|&i| chosen.contains(&res_ids[i]))
        .collect()
}

/// Superimpose every conformer on the mean structure of `atoms` (iterative Kabsch, 3 rounds, f64),
/// then centre on that mean (`api.align_ensemble`).
pub fn align_ensemble(models: &[Vec<P3>], atoms: &[usize]) -> Vec<Vec<P3>> {
    let mut x: Vec<Vec<V3>> = models
        .iter()
        .map(|m| {
            m.iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect()
        })
        .collect();
    if x.is_empty() || atoms.is_empty() {
        return models.to_vec();
    }
    let mut reference: Vec<V3> = atoms.iter().map(|&i| x[0][i]).collect();
    for _ in 0..3 {
        let rc = mean(&reference);
        let out: Vec<Vec<V3>> = x
            .iter()
            .map(|xi| {
                let p: Vec<V3> = atoms.iter().map(|&i| xi[i]).collect();
                let pc = mean(&p);
                let (r, _) = kabsch(&p, &reference);
                xi.iter()
                    .map(|q| {
                        let y = matvec(&r, sub(*q, pc));
                        [y[0] + rc[0], y[1] + rc[1], y[2] + rc[2]]
                    })
                    .collect()
            })
            .collect();
        reference = (0..atoms.len())
            .map(|k| mean(&out.iter().map(|o| o[atoms[k]]).collect::<Vec<_>>()))
            .collect();
        x = out;
    }
    let c = mean(&reference);
    x.iter()
        .map(|xi| {
            xi.iter()
                .map(|q| {
                    [
                        (q[0] - c[0]) as f32,
                        (q[1] - c[1]) as f32,
                        (q[2] - c[2]) as f32,
                    ]
                })
                .collect()
        })
        .collect()
}
