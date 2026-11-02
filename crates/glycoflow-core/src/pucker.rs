//! Ring pucker as a coordinate: the Cremer-Pople measure and its closed-form inverse.
//!
//! Rust port of `glycoflow/pucker.py` and the ring enumeration of `glycoflow/data.py::ring_cycles`.
//!
//! A ring's shape is its out-of-plane displacement in the Cremer-Pople basis, `(q2c, q2s, q3)`.
//! [`ring_pucker`] measures it and [`set_pucker`] inverts it, the way [`crate::geometry::dihedrals`]
//! and [`crate::geometry::set_torsions`] do for torsions.
//!
//! The inverse is exact and closed form. Of a ring's `n` out-of-plane displacements the mean plane
//! fixes three (the sum and the two first-order Fourier components), leaving
//! `span{cos 4*pi*j/n, sin 4*pi*j/n, (-1)^j}` -- precisely `(q2c, q2s, q3)`. A displacement along the
//! mean-plane normal that stays in that span leaves the normal and the centroid where they were, so
//! the map is a bijection and applying it twice changes nothing.
//!
//! Each ring atom carries what hangs off it: substituents rotate with the change in their ring
//! atom's local frame, so an axial substituent becomes equatorial when a chair inverts, and no bond
//! or angle outside the ring is touched. The branch through the ring's first atom is dragged along
//! rigidly, which moves the molecule in space but changes no internal coordinate, so rings compose
//! in any order.

use std::collections::VecDeque;

use crate::linalg::{add, cross, dot, matvec, normalize, scale, sub, M3, V3};

/// Components of a Cremer-Pople vector: `(q2c, q2s, q3)`.
pub const PUCKER_DIM: usize = 3;

/// Angstrom; the per-component spread of `(q2c, q2s, q3)` over the MD, so the model works in O(1)
/// units (`glycoflow/pucker.py::PUCKER_SCALE`). Model velocities are in these units.
pub const PUCKER_SCALE: f64 = 0.35;

/// One sugar ring: its atoms in the canonical order, which fixes the sign and phase of the
/// Cremer-Pople vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ring {
    /// 5 or 6 atom indices, in canonical order.
    pub atoms: Vec<usize>,
    /// For each ring atom, the atoms that follow it rigidly when the ring changes shape
    /// (`ring_substituents`). Never includes the ring atom itself or any other ring atom.
    pub substituents: Vec<Vec<usize>>,
}

impl Ring {
    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

/// Every furanose / pyranose ring of a molecule, in the order and orientation the Python reference
/// uses (`data.py::ring_cycles`).
///
/// Each ring starts at its ring oxygen (the lowest atom index if it has none) and runs towards the
/// neighbour carrying more oxygens, which is the anomeric carbon of a sugar ring. Rings are sorted
/// by their lowest atom index. Only 5- and 6-cycles are returned: a longer cycle is not a sugar ring
/// and has no Cremer-Pople vector of this form, which is how a spurious 12-cycle through two fused
/// rings (distance-detected bonds between touching residues) is discarded.
pub fn ring_cycles(n_atoms: usize, bonds: &[[usize; 2]], elements: &[String]) -> Vec<Ring> {
    let mut adj = vec![Vec::new(); n_atoms];
    for &[i, j] in bonds {
        if i < n_atoms && j < n_atoms && i != j {
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }

    // Shortest cycle through each atom, found by BFS from it. For a 5/6-ring every member lies on
    // exactly one such cycle, so collecting them as sorted atom sets and de-duplicating enumerates
    // the sugar rings without a full cycle basis.
    let mut seen: Vec<Vec<usize>> = Vec::new();
    for start in 0..n_atoms {
        if let Some(cycle) = shortest_cycle_through(start, &adj) {
            if (5..=6).contains(&cycle.len()) {
                let mut key = cycle.clone();
                key.sort_unstable();
                if !seen.contains(&key) {
                    seen.push(key);
                }
            }
        }
    }

    let count_o = |a: usize| -> usize {
        adj[a]
            .iter()
            .filter(|&&n| elements.get(n).map(String::as_str) == Some("O"))
            .count()
    };

    let mut rings: Vec<Ring> = Vec::new();
    for key in seen {
        // recover the cyclic order from adjacency
        let Some(mut cycle) = order_cycle(&key, &adj) else {
            continue;
        };
        // start at the ring oxygen, else the lowest index
        let start = cycle
            .iter()
            .position(|&a| elements.get(a).map(String::as_str) == Some("O"))
            .unwrap_or_else(|| {
                let m = *cycle.iter().min().expect("non-empty");
                cycle.iter().position(|&a| a == m).expect("present")
            });
        cycle.rotate_left(start);
        // run towards the neighbour with more oxygens (the anomeric carbon)
        if count_o(cycle[cycle.len() - 1]) > count_o(cycle[1]) {
            cycle[1..].reverse();
        }
        let substituents = ring_substituents(&cycle, n_atoms, &adj);
        rings.push(Ring {
            atoms: cycle,
            substituents,
        });
    }
    rings.sort_by_key(|r| *r.atoms.iter().min().expect("non-empty"));
    rings
}

/// BFS for the shortest cycle containing `start`, returned as atoms in cyclic order.
fn shortest_cycle_through(start: usize, adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = adj.len();
    let mut parent = vec![usize::MAX; n];
    let mut depth = vec![usize::MAX; n];
    depth[start] = 0;
    let mut queue = VecDeque::from([start]);
    let mut best: Option<Vec<usize>> = None;
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if depth[v] == usize::MAX {
                depth[v] = depth[u] + 1;
                parent[v] = u;
                queue.push_back(v);
            } else if v != parent[u] {
                // u-v closes a cycle; keep it only if it passes through `start`
                let up = |mut x: usize| {
                    let mut path = vec![x];
                    while x != start && parent[x] != usize::MAX {
                        x = parent[x];
                        path.push(x);
                    }
                    (path, x == start)
                };
                let (pu, ok_u) = up(u);
                let (pv, ok_v) = up(v);
                if !ok_u || !ok_v {
                    continue;
                }
                // the two paths meet only at `start`, or the cycle does not contain it
                let shared = pu.iter().filter(|a| pv.contains(a)).count();
                if shared != 1 {
                    continue;
                }
                let mut cycle = pu;
                cycle.pop(); // drop the shared `start`, re-added by pv
                cycle.reverse();
                let mut tail = pv;
                tail.reverse();
                cycle.splice(0..0, tail);
                if best.as_ref().map_or(true, |b| cycle.len() < b.len()) {
                    best = Some(cycle);
                }
            }
        }
    }
    best
}

/// Put a known ring's atoms into cyclic order by walking its own edges.
fn order_cycle(atoms: &[usize], adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(atoms.len());
    out.push(atoms[0]);
    while out.len() < atoms.len() {
        let last = *out.last().expect("non-empty");
        let next = adj[last]
            .iter()
            .copied()
            .find(|n| atoms.contains(n) && !out.contains(n))?;
        out.push(next);
    }
    // the walk must close
    let first = out[0];
    let last = *out.last().expect("non-empty");
    adj[last].contains(&first).then_some(out)
}

/// For each ring atom, the atoms that follow it rigidly: its connected component once every ring
/// bond is cut, minus the atom itself.
fn ring_substituents(ring: &[usize], n_atoms: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = ring.len();
    let is_ring_edge = |u: usize, v: usize| {
        (0..n).any(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            (u == a && v == b) || (u == b && v == a)
        })
    };
    ring.iter()
        .map(|&atom| {
            let mut seen = vec![false; n_atoms];
            seen[atom] = true;
            let mut queue = VecDeque::from([atom]);
            let mut out = Vec::new();
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if !seen[v] && !is_ring_edge(u, v) {
                        seen[v] = true;
                        out.push(v);
                        queue.push_back(v);
                    }
                }
            }
            out.sort_unstable();
            out
        })
        .collect()
}

/// Cremer-Pople basis of a ring of size `n`: `(cos, sin, alternating)`, each of length `n`.
///
/// `q2c, q2s, q3 = sqrt(2/n) z.cos, sqrt(2/n) z.sin, z.alt / sqrt(n)` with the sign convention of
/// `data.cremer_pople`; the alternating mode exists only for a six-ring.
pub fn cp_basis(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut cb, mut sb, mut ab) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for j in 0..n {
        let ang = 4.0 * std::f64::consts::PI * j as f64 / n as f64;
        cb[j] = ang.cos();
        sb[j] = -ang.sin();
        ab[j] = if n == 6 {
            if j % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        };
    }
    (cb, sb, ab)
}

/// Mean-plane normal of a ring and its atoms' out-of-plane displacements.
fn mean_plane(xr: &[V3]) -> (V3, Vec<f64>) {
    let n = xr.len();
    let centroid = scale(
        xr.iter().fold([0.0; 3], |acc, &p| add(acc, p)),
        1.0 / n as f64,
    );
    let r: Vec<V3> = xr.iter().map(|&p| sub(p, centroid)).collect();
    let (mut w, mut v) = ([0.0; 3], [0.0; 3]);
    for (j, &rj) in r.iter().enumerate() {
        let ang = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
        w = add(w, scale(rj, ang.sin()));
        v = add(v, scale(rj, ang.cos()));
    }
    let normal = normalize(cross(w, v));
    let z = r.iter().map(|&rj| dot(rj, normal)).collect();
    (normal, z)
}

/// Local frame of every ring atom from its two ring neighbours; columns `[u, w x u, w]`.
fn slot_frames(xr: &[V3]) -> Vec<M3> {
    let n = xr.len();
    (0..n)
        .map(|k| {
            let a = sub(xr[(k + n - 1) % n], xr[k]);
            let b = sub(xr[(k + 1) % n], xr[k]);
            let u = normalize(add(a, b));
            let w = normalize(cross(a, b));
            let m = cross(w, u);
            // columns, to match the Python `torch.stack([u, cross(w,u), w], -1)`
            [
                [u[0], m[0], w[0]],
                [u[1], m[1], w[1]],
                [u[2], m[2], w[2]],
            ]
        })
        .collect()
}

#[inline]
fn transpose(m: &M3) -> M3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn matmul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// The Cremer-Pople vector of one ring: `(q2c, q2s, q3)` in Angstrom.
pub fn ring_pucker(x: &[V3], ring: &Ring) -> [f64; PUCKER_DIM] {
    let n = ring.len();
    let xr: Vec<V3> = ring.atoms.iter().map(|&a| x[a]).collect();
    let (_, z) = mean_plane(&xr);
    let (cb, sb, ab) = cp_basis(n);
    let scale2 = (2.0 / n as f64).sqrt();
    let dotz = |b: &[f64]| z.iter().zip(b).map(|(a, c)| a * c).sum::<f64>();
    [
        scale2 * dotz(&cb),
        scale2 * dotz(&sb),
        dotz(&ab) / (n as f64).sqrt(),
    ]
}

/// Cremer-Pople vectors of every ring.
pub fn ring_puckers(x: &[V3], rings: &[Ring]) -> Vec<[f64; PUCKER_DIM]> {
    rings.iter().map(|r| ring_pucker(x, r)).collect()
}

/// Deform every ring of `x` so that it carries the requested Cremer-Pople vector, in place.
///
/// `targets[r]` is the `(q2c, q2s, q3)` wanted for `rings[r]`, in Angstrom. Substituents follow
/// their ring atom's frame; the branch through the ring's first atom is dragged along rigidly.
/// Applying this twice with the same target changes nothing.
pub fn set_puckers(x: &mut [V3], rings: &[Ring], targets: &[[f64; PUCKER_DIM]]) {
    for (ring, target) in rings.iter().zip(targets) {
        let n = ring.len();
        if n == 0 {
            continue;
        }
        let xr: Vec<V3> = ring.atoms.iter().map(|&a| x[a]).collect();
        let (normal, z) = mean_plane(&xr);
        let (cb, sb, ab) = cp_basis(n);
        let scale2 = (2.0 / n as f64).sqrt();
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();
        let xr_new: Vec<V3> = (0..n)
            .map(|j| {
                let zt = scale2 * (target[0] * cb[j] + target[1] * sb[j]) + target[2] * ab[j] * inv_sqrt_n;
                add(xr[j], scale(normal, zt - z[j]))
            })
            .collect();

        let (old_frames, new_frames) = (slot_frames(&xr), slot_frames(&xr_new));
        for k in 0..n {
            let rot = matmul(&new_frames[k], &transpose(&old_frames[k]));
            let (pivot, pivot_new) = (xr[k], xr_new[k]);
            for &atom in &ring.substituents[k] {
                x[atom] = add(matvec(&rot, sub(x[atom], pivot)), pivot_new);
            }
        }
        for (j, &atom) in ring.atoms.iter().enumerate() {
            x[atom] = xr_new[j];
        }
    }
}
