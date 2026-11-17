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

/// A six-ring within this Cremer-Pople theta of its chair is in the chair (degrees).
pub const CHAIR_TOLERANCE: f64 = 30.0;

/// Bring every six-ring that is out of its chair into that chair, with its substituents. Returns
/// the number of rings changed.
///
/// The chair of an aldopyranose is the one of crystal structures, 4C1 for a D sugar and 1C4 for
/// an L sugar (the rule of `Known::chair_deviation`), taken from the first template that has it;
/// template 0 is corrected too when it is in the other chair. The majority build is not the
/// rule: beta-D-xylose is 1C4 in 55% of its MD frames and 4C1 in 381 of 383 deposited rings. The
/// chair of any other six-ring (sialic acids, Kdo) is that of the majority build (template 0).
///
/// The pucker states of the residue library and the ring shapes a model generates include the
/// inverted chair and boats of residues such as GlcNAc at a few per cent each. A fit to a map
/// takes the best of its templates, so in weak density such a ring is selected by noise, and the
/// model would carry a conformation that no map at these resolutions can show. Iduronic acid,
/// whose ring has several states in solution, is left as drawn.
pub fn chair_templates(glycan: &mut Glycan) -> usize {
    let rings: Vec<_> = glycan
        .topology
        .rings
        .iter()
        .filter(|ring| {
            let name = &glycan.res_names[ring.atoms[0]];
            ring.len() == 6 && !name.to_ascii_lowercase().ends_with("ua")
        })
        .cloned()
        .collect();
    let f64s = |x: &[[f32; 3]]| -> Vec<V> { x.iter().map(|p| p.map(f64::from)).collect() };
    let theta = |q: &[f64; 3]| q[0].hypot(q[1]).atan2(q[2]).to_degrees();
    let puckers: Vec<_> = glycan
        .templates
        .iter()
        .map(|t| glycoflow_core::ring_puckers(&f64s(t), &rings))
        .collect();
    // the chair of each ring, and whether template 0 has to be brought into it
    let mut chairs = puckers[0].clone();
    let mut first = 1;
    for (r, ring) in rings.iter().enumerate() {
        // ring atoms O5 C1..C5 of an aldopyranose; the second letter of a GLYCAM residue name
        // is upper case for a D sugar
        let named: Option<Vec<usize>> = ["O5", "C1", "C2", "C3", "C4", "C5"]
            .iter()
            .map(|name| {
                ring.atoms
                    .iter()
                    .copied()
                    .find(|&i| glycan.atom_names[i] == *name)
            })
            .collect();
        let Some(named) = named else { continue };
        let l_sugar = glycan.res_names[ring.atoms[0]]
            .chars()
            .nth(1)
            .is_some_and(|c| c.is_ascii_lowercase());
        let deviation = |t: usize| {
            let x = &glycan.templates[t];
            let ring: [V; 6] = std::array::from_fn(|k| x[named[k]].map(f64::from));
            let theta = chair_theta(&ring);
            if l_sugar { 180.0 - theta } else { theta }
        };
        if deviation(0) <= CHAIR_TOLERANCE {
            continue;
        }
        first = 0;
        chairs[r] = match (1..glycan.templates.len()).find(|&t| deviation(t) <= CHAIR_TOLERANCE) {
            Some(t) => puckers[t][r],
            None => puckers[0][r].map(|q| -q), // the inverted ring
        };
    }
    let mut changed = 0;
    for (template, shapes) in glycan.templates.iter_mut().zip(puckers).skip(first) {
        let mut shapes = shapes;
        let before = changed;
        for (shape, chair) in shapes.iter_mut().zip(&chairs) {
            if (theta(shape) - theta(chair)).abs() > CHAIR_TOLERANCE {
                *shape = *chair;
                changed += 1;
            }
        }
        if changed > before {
            let mut x = f64s(template);
            glycoflow_core::set_puckers(&mut x, &rings, &shapes);
            *template = x.iter().map(|p| p.map(|v| v as f32)).collect();
        }
    }
    changed
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

    #[test]
    #[ignore = "needs GLYCOFLOW_MODEL (residue_library.json of the licensed model)"]
    fn chair_templates_use_the_crystal_chair() {
        let path = std::path::Path::new(&std::env::var("GLYCOFLOW_MODEL").unwrap())
            .join("residue_library.json");
        let library =
            glycoflow_core::ResidueLibrary::from_json_slice(&std::fs::read(path).unwrap()).unwrap();
        // deviation of every aldopyranose ring of every template from its crystal chair
        let worst = |g: &Glycan| {
            let mut worst = 0f64;
            for ring in g.topology.rings.iter().filter(|r| r.len() == 6) {
                let named: Option<Vec<usize>> = ["O5", "C1", "C2", "C3", "C4", "C5"]
                    .iter()
                    .map(|n| ring.atoms.iter().copied().find(|&i| g.atom_names[i] == *n))
                    .collect();
                let Some(named) = named else { continue };
                let l = g.res_names[named[0]].chars().nth(1).unwrap().is_ascii_lowercase();
                for x in &g.templates {
                    let t = chair_theta(&std::array::from_fn(|k| x[named[k]].map(f64::from)));
                    worst = worst.max(if l { 180.0 - t } else { t });
                }
            }
            worst
        };
        // beta-D-xylose: the majority build of the library is 1C4, the chair is 4C1
        for seq in [
            "DXylpb1-OH",
            "DManpa1-6[DXylpb1-2]DManpb1-4DGlcpNAcb1-4[LFucpa1-3]DGlcpNAcb1-OH",
        ] {
            let mut g = crate::problem::build_glycan(&library, seq, 16, 0).unwrap();
            assert!(worst(&g) > 150.0, "{seq}: the library has the other chair");
            assert!(chair_templates(&mut g) > 0);
            assert!(worst(&g) < CHAIR_TOLERANCE, "{seq}: {}", worst(&g));
        }
        // residues whose majority build is in the crystal chair: template 0 untouched, and the
        // others as with template 0 for the rule (L sugars included)
        for seq in [
            "DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH",
            "DNeup5Aca2-6DGalpb1-4DGlcpNAcb1-OH",
            "LRhapa1-OH",
        ] {
            let mut g = crate::problem::build_glycan(&library, seq, 16, 0).unwrap();
            let before = g.templates[0].clone();
            chair_templates(&mut g);
            assert_eq!(g.templates[0], before, "{seq}");
            assert!(worst(&g) < CHAIR_TOLERANCE, "{seq}: {}", worst(&g));
        }
    }
}
