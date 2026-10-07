//! The root ring in the other chair. C-mannose on Trp is 1C4 in crystal structures (7R84, 8CKK:
//! Cremer-Pople theta 168-176 deg), while alpha-D-Man is 4C1 in solution and GlycoFlow's library
//! has no other pucker for it; Trp sites therefore search templates in both chairs.
//!
//! The flip mirrors the six ring atoms through their mean plane and puts every exocyclic group
//! where the carbon's implicit hydrogen was (mirrored), which keeps every configuration; what lies
//! beyond the substituent (O6, N-acetyl atoms, child residues) follows rigidly, keeping its
//! torsions.

use glycoflow_core::topology::{Glycan, center_f32};

type V = [f64; 3];

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: V) -> V {
    scale(a, 1.0 / dot(a, a).sqrt())
}

/// Orthonormal frame (rows) with x along `axis` and y towards `toward` (perpendicular part).
fn frame(axis: V, toward: V) -> [V; 3] {
    let e1 = unit(axis);
    let e2 = unit(sub(toward, scale(e1, dot(toward, e1))));
    [e1, e2, cross(e1, e2)]
}

/// Ring atoms of an aldopyranose root (O5, C1..C5), if the root is one.
fn root_ring(glycan: &Glycan) -> Option<[usize; 6]> {
    let find = |name: &str| {
        (0..glycan.atom_names.len())
            .find(|&i| glycan.res_paths[i] == "r" && glycan.atom_names[i] == name)
    };
    Some([
        find("O5")?,
        find("C1")?,
        find("C2")?,
        find("C3")?,
        find("C4")?,
        find("C5")?,
    ])
}

/// Flip the root ring of template `x` (see the module docs); None when the root is not an
/// aldopyranose with one heavy substituent per ring carbon.
fn flip(glycan: &Glycan, ring: &[usize; 6], x: &[[f32; 3]]) -> Option<Vec<[f32; 3]>> {
    let n = x.len();
    let mut adj = vec![Vec::new(); n];
    for &[a, b] in &glycan.bonds {
        adj[a].push(b);
        adj[b].push(a);
    }
    let p: Vec<V> = x.iter().map(|q| q.map(f64::from)).collect();
    // mean plane of the ring
    let c = scale(ring.iter().fold([0.0; 3], |s, &i| add(s, p[i])), 1.0 / 6.0);
    let mut normal = [0.0; 3];
    for k in 0..6 {
        normal = add(
            normal,
            cross(sub(p[ring[k]], c), sub(p[ring[(k + 1) % 6]], c)),
        );
    }
    let normal = unit(normal);
    let mirror_point = |q: V| sub(q, scale(normal, 2.0 * dot(sub(q, c), normal)));
    let mirror_dir = |d: V| sub(d, scale(normal, 2.0 * dot(d, normal)));
    let mut out = p.clone();
    for &i in ring {
        out[i] = mirror_point(p[i]);
    }
    let in_ring = |i: usize| ring.contains(&i);
    for &ci in &ring[1..] {
        let ring_nb: Vec<usize> = adj[ci].iter().copied().filter(|&j| in_ring(j)).collect();
        let exo: Vec<usize> = adj[ci].iter().copied().filter(|&j| !in_ring(j)).collect();
        if ring_nb.len() != 2 || exo.len() != 1 {
            return None;
        }
        let xi = exo[0];
        // the implicit hydrogen: the fourth tetrahedral direction
        let h = unit(scale(
            add(
                add(
                    unit(sub(p[ring_nb[0]], p[ci])),
                    unit(sub(p[ring_nb[1]], p[ci])),
                ),
                unit(sub(p[xi], p[ci])),
            ),
            -1.0,
        ));
        let bond = dot(sub(p[xi], p[ci]), sub(p[xi], p[ci])).sqrt();
        out[xi] = add(out[ci], scale(mirror_dir(h), bond));
        // everything beyond the substituent moves with it, keeping ring-C-X-Y torsions
        let old = frame(sub(p[xi], p[ci]), sub(p[ring_nb[0]], p[ci]));
        let new = frame(sub(out[xi], out[ci]), sub(out[ring_nb[0]], out[ci]));
        let mut stack: Vec<usize> = adj[xi].iter().copied().filter(|&j| j != ci).collect();
        let mut seen = vec![false; n];
        seen[ci] = true;
        seen[xi] = true;
        while let Some(j) = stack.pop() {
            if seen[j] || in_ring(j) {
                continue;
            }
            seen[j] = true;
            let d = sub(p[j], p[xi]);
            let local = [dot(d, old[0]), dot(d, old[1]), dot(d, old[2])];
            out[j] = add(
                out[xi],
                add(
                    add(scale(new[0], local[0]), scale(new[1], local[1])),
                    scale(new[2], local[2]),
                ),
            );
            stack.extend(adj[j].iter().copied());
        }
    }
    Some(center_f32(
        &out.iter().map(|q| q.map(|v| v as f32)).collect::<Vec<_>>(),
    ))
}

/// Put the root ring of the templates `which` selects into the other chair. Returns whether the
/// root could be flipped (an aldopyranose); the templates are unchanged otherwise.
pub fn flip_root_chair(glycan: &mut Glycan, which: impl Fn(usize) -> bool) -> bool {
    let Some(ring) = root_ring(glycan) else {
        return false;
    };
    let flipped: Vec<Option<Vec<[f32; 3]>>> = glycan
        .templates
        .iter()
        .enumerate()
        .map(|(t, x)| {
            if which(t) {
                flip(glycan, &ring, x)
            } else {
                None
            }
        })
        .collect();
    if glycan.templates.len() > 1
        && flipped
            .iter()
            .enumerate()
            .any(|(t, f)| which(t) && f.is_none())
    {
        return false;
    }
    for (t, f) in flipped.into_iter().enumerate() {
        if let Some(x) = f {
            glycan.templates[t] = x;
        }
    }
    true
}

/// Cremer-Pople theta (degrees) of a six-ring given in order: 0 for 4C1, 180 for 1C4 (D-sugars,
/// ring O5 C1 C2 C3 C4 C5).
pub fn chair_theta(ring: &[V; 6]) -> f64 {
    let c = scale(ring.iter().fold([0.0; 3], |s, &q| add(s, q)), 1.0 / 6.0);
    let r: Vec<V> = ring.iter().map(|&q| sub(q, c)).collect();
    let angle = |j: usize, m: f64| 2.0 * std::f64::consts::PI * m * j as f64 / 6.0;
    let r1 = (0..6).fold([0.0; 3], |s, j| add(s, scale(r[j], angle(j, 1.0).sin())));
    let r2 = (0..6).fold([0.0; 3], |s, j| add(s, scale(r[j], angle(j, 1.0).cos())));
    let nrm = unit(cross(r1, r2));
    let z: Vec<f64> = r.iter().map(|q| dot(*q, nrm)).collect();
    let q3: f64 = (0..6)
        .map(|j| z[j] * if j % 2 == 0 { 1.0 } else { -1.0 })
        .sum::<f64>()
        / 6f64.sqrt();
    let q2c = (2.0 / 6.0f64).sqrt() * (0..6).map(|j| z[j] * angle(j, 2.0).cos()).sum::<f64>();
    let q2s = -(2.0 / 6.0f64).sqrt() * (0..6).map(|j| z[j] * angle(j, 2.0).sin()).sum::<f64>();
    q2c.hypot(q2s).atan2(q3).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_volume(o: V, a: V, b: V, c: V) -> f64 {
        dot(sub(a, o), cross(sub(b, o), sub(c, o)))
    }

    #[test]
    #[ignore = "needs GLYCOFLOW_MODEL (residue_library.json of the licensed model)"]
    fn flips_the_chair_and_keeps_every_configuration() {
        let path = std::path::Path::new(&std::env::var("GLYCOFLOW_MODEL").unwrap())
            .join("residue_library.json");
        let library =
            glycoflow_core::ResidueLibrary::from_json_slice(&std::fs::read(path).unwrap()).unwrap();
        for seq in ["DManpa1-OH", "DGlcpNAcb1-OH", "DGalpb1-4DGlcpNAcb1-OH"] {
            let mut g = crate::problem::build_glycan(&library, seq, 2, 0).unwrap();
            let before = g.templates[1].clone();
            assert!(flip_root_chair(&mut g, |t| t == 1), "{seq}");
            let ring = root_ring(&g).unwrap();
            let pos = |x: &Vec<[f32; 3]>, i: usize| x[i].map(f64::from);
            let theta = |x: &Vec<[f32; 3]>| chair_theta(&ring.map(|i| pos(x, i)));
            assert!(
                theta(&before) < 30.0,
                "{seq}: starts 4C1 ({})",
                theta(&before)
            );
            assert!(
                theta(&g.templates[1]) > 150.0,
                "{seq}: ends 1C4 ({})",
                theta(&g.templates[1])
            );
            assert_eq!(
                g.templates[0],
                crate::problem::build_glycan(&library, seq, 1, 0)
                    .unwrap()
                    .templates[0]
            );
            // every bond length kept, every stereocentre kept
            for &[a, b] in &g.bonds {
                let d = |x: &Vec<[f32; 3]>| {
                    dot(sub(pos(x, a), pos(x, b)), sub(pos(x, a), pos(x, b))).sqrt()
                };
                assert!(
                    (d(&before) - d(&g.templates[1])).abs() < 1e-3,
                    "{seq}: bond {a}-{b}"
                );
            }
            let n = g.atom_names.len();
            let mut adj = vec![Vec::new(); n];
            for &[a, b] in &g.bonds {
                adj[a].push(b);
                adj[b].push(a);
            }
            for i in (0..n).filter(|&i| g.elements[i] == "C" && adj[i].len() == 3) {
                let v = |x: &Vec<[f32; 3]>| {
                    signed_volume(
                        pos(x, i),
                        pos(x, adj[i][0]),
                        pos(x, adj[i][1]),
                        pos(x, adj[i][2]),
                    )
                };
                assert_eq!(
                    v(&before).signum(),
                    v(&g.templates[1]).signum(),
                    "{seq}: {} {}",
                    g.res_paths[i],
                    g.atom_names[i]
                );
            }
        }
    }
}
