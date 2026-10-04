//! Covalent graph, rotatable torsions, distal masks, topological distances (`data.structure_entry`)
//! and model tokens (`dataset.Vocab.encode`).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::builder::BuiltGlycan;
use crate::error::{Error, Result};
use crate::geometry;

/// Cap of the topological distance matrix (`data.TOPO_DIST_CAP`).
pub const TOPO_DIST_CAP: u8 = 12;
/// Element vocabulary (`dataset.ELEMENTS`).
pub const ELEMENTS: [&str; 7] = ["<pad>", "<unk>", "C", "N", "O", "S", "P"];

fn atomic_number(el: &str) -> i32 {
    match el {
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "P" => 15,
        "S" => 16,
        _ => 0,
    }
}

/// Graph-derived arrays of one glycan.
#[derive(Clone, Debug)]
pub struct Topology {
    pub n_atoms: usize,
    pub ring_atoms: Vec<bool>,
    /// topological distance, row-major [N, N], capped at 12
    pub topo_dist: Vec<u8>,
    /// rotatable torsions (a, b, c, d), bond b->c oriented away from atom 0, sorted by depth of b
    pub quads: Vec<[usize; 4]>,
    /// atoms on the distal side of each torsion bond (ascending indices; includes c)
    pub distal: Vec<Vec<usize>>,
}

impl Topology {
    pub fn n_torsions(&self) -> usize {
        self.quads.len()
    }

    /// Distal masks as a row-major [T, N] bool array.
    pub fn distal_mask(&self) -> Vec<bool> {
        let mut m = vec![false; self.quads.len() * self.n_atoms];
        for (t, idx) in self.distal.iter().enumerate() {
            for &i in idx {
                m[t * self.n_atoms + i] = true;
            }
        }
        m
    }

    /// `structure_entry` topology from atoms and explicit bonds (root = atom 0).
    pub fn new(names: &[String], elements: &[String], bonds: &[[usize; 2]]) -> Result<Self> {
        let n = names.len();
        // adjacency in networkx insertion order
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &[a, b] in bonds {
            if a >= n || b >= n {
                return Err(Error::Topology(format!("bond ({a}, {b}) out of range")));
            }
            if a == b || adj[a].contains(&b) {
                continue;
            }
            adj[a].push(b);
            adj[b].push(a);
        }
        let bfs = |src: usize, skip: Option<(usize, usize)>, cutoff: usize| -> Vec<usize> {
            let mut dist = vec![usize::MAX; n];
            dist[src] = 0;
            let mut q = VecDeque::from([src]);
            while let Some(u) = q.pop_front() {
                if dist[u] >= cutoff {
                    continue;
                }
                for &v in &adj[u] {
                    if let Some((x, y)) = skip {
                        if (u == x && v == y) || (u == y && v == x) {
                            continue;
                        }
                    }
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        q.push_back(v);
                    }
                }
            }
            dist
        };
        if n == 0 {
            return Err(Error::Topology("no atoms".into()));
        }
        let depth = bfs(0, None, usize::MAX);
        if depth.contains(&usize::MAX) {
            return Err(Error::Topology("heavy-atom graph is disconnected".into()));
        }
        // networkx edge iteration order
        let mut edges = Vec::new();
        let mut seen = vec![false; n];
        for u in 0..n {
            for &v in &adj[u] {
                if !seen[v] {
                    edges.push((u, v));
                }
            }
            seen[u] = true;
        }
        // ring edges = edges of the cycle basis = non-bridges
        let is_ring_edge = |u: usize, v: usize| bfs(u, Some((u, v)), usize::MAX)[v] != usize::MAX;
        let mut ring_atoms = vec![false; n];
        let mut ring_edge = Vec::with_capacity(edges.len());
        for &(u, v) in &edges {
            let r = is_ring_edge(u, v);
            if r {
                ring_atoms[u] = true;
                ring_atoms[v] = true;
            }
            ring_edge.push(r);
        }
        let priority = |x: usize| (-atomic_number(&elements[x]), names[x].clone(), x);
        let mut quads = Vec::new();
        let mut distal = Vec::new();
        for (&(u, v), &ring) in edges.iter().zip(&ring_edge) {
            if ring || adj[u].len() < 2 || adj[v].len() < 2 {
                continue;
            }
            let (b, c) = if depth[u] < depth[v] { (u, v) } else { (v, u) };
            let a = adj[b]
                .iter()
                .copied()
                .filter(|&x| x != c)
                .min_by_key(|&x| priority(x))
                .unwrap();
            let d = adj[c]
                .iter()
                .copied()
                .filter(|&x| x != b)
                .min_by_key(|&x| priority(x))
                .unwrap();
            let comp = bfs(c, Some((b, c)), usize::MAX);
            if comp[0] != usize::MAX {
                return Err(Error::Topology(
                    "root on distal side of a non-ring bond".into(),
                ));
            }
            quads.push([a, b, c, d]);
            distal.push(
                (0..n)
                    .filter(|&i| comp[i] != usize::MAX)
                    .collect::<Vec<_>>(),
            );
        }
        let mut order: Vec<usize> = (0..quads.len()).collect();
        order.sort_by_key(|&k| depth[quads[k][1]]); // stable
        let quads: Vec<[usize; 4]> = order.iter().map(|&k| quads[k]).collect();
        let distal: Vec<Vec<usize>> = order.iter().map(|&k| distal[k].clone()).collect();
        let mut topo_dist = vec![TOPO_DIST_CAP; n * n];
        for src in 0..n {
            let d = bfs(src, None, TOPO_DIST_CAP as usize);
            for (dst, &l) in d.iter().enumerate() {
                if l != usize::MAX {
                    topo_dist[src * n + dst] = l as u8;
                }
            }
        }
        Ok(Self {
            n_atoms: n,
            ring_atoms,
            topo_dist,
            quads,
            distal,
        })
    }
}

/// Checkpoint vocabulary (`dataset.Vocab`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocab {
    pub atom_names: Vec<String>,
    pub residues: Vec<String>,
    pub link_codes: Vec<String>,
}

impl Vocab {
    fn lookup(table: &[impl AsRef<str>], key: &str) -> u32 {
        table
            .iter()
            .position(|x| x.as_ref() == key)
            .map(|i| i as u32)
            .unwrap_or(1)
    }

    /// Integer atom tokens [N][5]: element, atom name, residue (resname[1:]), link code
    /// (resname[:1]), 1 + ring flag; unknown -> 1.
    pub fn encode(
        &self,
        names: &[String],
        res_names: &[String],
        elements: &[String],
        ring: &[bool],
    ) -> Vec<[u32; 5]> {
        names
            .iter()
            .zip(res_names)
            .zip(elements)
            .zip(ring)
            .map(|(((name, res), el), &r)| {
                let (link, residue) = match res.char_indices().nth(1) {
                    Some((i, _)) => (&res[..i], &res[i..]),
                    None => (res.as_str(), ""),
                };
                [
                    Self::lookup(&ELEMENTS, el),
                    Self::lookup(&self.atom_names, name),
                    Self::lookup(&self.residues, residue),
                    Self::lookup(&self.link_codes, link),
                    1 + r as u32,
                ]
            })
            .collect()
    }
}

/// A glycan ready for sampling: atoms, topology and one or more centred template conformers
/// (the entry of `evaluate.build_templates` / `data.structure_entry`).
#[derive(Clone, Debug)]
pub struct Glycan {
    pub atom_names: Vec<String>,
    pub res_names: Vec<String>,
    pub res_ids: Vec<i64>,
    pub elements: Vec<String>,
    pub bonds: Vec<[usize; 2]>,
    pub res_paths: Vec<String>,
    pub topology: Topology,
    /// template conformers [F][N] (f32, each centred on its mean as in `build_templates`)
    pub templates: Vec<Vec<[f32; 3]>>,
}

/// `coords - coords.mean(axis=1)` in f32 (sequential accumulation like numpy along a strided axis).
pub fn center_f32(x: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut s = [0f32; 3];
    for p in x {
        for k in 0..3 {
            s[k] += p[k];
        }
    }
    let n = x.len() as f32;
    let m = [s[0] / n, s[1] / n, s[2] / n];
    x.iter()
        .map(|p| [p[0] - m[0], p[1] - m[1], p[2] - m[2]])
        .collect()
}

impl Glycan {
    /// Entry from one or more builds of the same sequence (e.g. different pucker states); the first
    /// build defines atoms and topology, every build gives a template conformer.
    pub fn from_builds(builds: &[BuiltGlycan]) -> Result<Self> {
        let b0 = builds
            .first()
            .ok_or_else(|| Error::Invalid("no builds".into()))?;
        for b in builds {
            if b.atom_names != b0.atom_names {
                return Err(Error::Invalid("builds have different atoms".into()));
            }
        }
        let topology = Topology::new(&b0.atom_names, &b0.elements, &b0.bonds)?;
        Ok(Self {
            atom_names: b0.atom_names.clone(),
            res_names: b0.res_names.clone(),
            res_ids: b0.res_ids.clone(),
            elements: b0.elements.clone(),
            bonds: b0.bonds.clone(),
            res_paths: b0.res_paths.clone(),
            topology,
            templates: builds.iter().map(|b| center_f32(&b.coords_f32())).collect(),
        })
    }

    pub fn n_atoms(&self) -> usize {
        self.atom_names.len()
    }

    pub fn n_torsions(&self) -> usize {
        self.topology.n_torsions()
    }

    pub fn tokens(&self, vocab: &Vocab) -> Vec<[u32; 5]> {
        vocab.encode(
            &self.atom_names,
            &self.res_names,
            &self.elements,
            &self.topology.ring_atoms,
        )
    }

    /// Torsions of a conformer (radians).
    pub fn torsions(&self, x: &[[f32; 3]]) -> Vec<f32> {
        geometry::dihedrals(x, &self.topology.quads)
    }
}
