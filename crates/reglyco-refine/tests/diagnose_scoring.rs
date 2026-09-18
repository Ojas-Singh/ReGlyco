//! Phase 1 scoring diagnosis.
//!
//! Deposit coordinates are used ONLY here as benchmark diagnostics to answer
//! the mandatory decision gate: does the current fixed-ROI density objective
//! rank deposited-like structures above the currently selected wrong arms?

use std::collections::BTreeSet;
use std::path::Path;

use glysys::{BuildOptions, ResidueId, Structure, Vec3};
use reglyco_density::{DensityMap, DensityScoreOptions, DensityScorer, DensityTarget};

fn ring_points(structure: &Structure, residue: &ResidueId) -> Vec<Vec3> {
    ["C1", "C2", "C3", "C4", "C5", "O5"]
        .iter()
        .filter_map(|name| atom_position(structure, residue, name))
        .collect()
}

/// Rigid ring-frame root-mean-square deviation.  Both rings are expressed in
/// their C1/O5/O3 orthonormal frames and compared after that rigid
/// alignment.  This is an observation-only oracle for whether one rigid
/// pyranose template can represent another ring shape.
fn ring_frame_rmsd(moving: &[Vec3], target: &[Vec3], residue_name: &str) -> f64 {
    assert_eq!(moving.len(), target.len(), "oracle point sets must match");
    if moving.len() != 6 || target.len() != 6 {
        return f64::INFINITY;
    }
    let moving_frame = frame(moving[0], moving[1], moving[2]);
    let target_frame = frame(target[0], target[1], target[2]);
    let squared = moving
        .iter()
        .zip(target)
        .map(|(from, to)| {
            let relative = sub(*from, moving[0]);
            let local = Vec3 {
                x: dot(relative, moving_frame[0]),
                y: dot(relative, moving_frame[1]),
                z: dot(relative, moving_frame[2]),
            };
            let aligned = add(
                target[0],
                add(
                    scale(target_frame[0], local.x),
                    add(
                        scale(target_frame[1], local.y),
                        scale(target_frame[2], local.z),
                    ),
                ),
            );
            dot(sub(aligned, *to), sub(aligned, *to))
        })
        .sum::<f64>();
    let rmsd = (squared / moving.len().max(1) as f64).sqrt();
    if !rmsd.is_finite() {
        eprintln!("oracle-ring-frame: residue={residue_name} rmsd=unavailable");
    }
    rmsd
}

/// Oracle reachability gate for the internal-coordinate model.
///
/// Before any search result is trusted, verify that the rigid pyranose
/// templates used by the torsion model can at least represent the deposited
/// 5KZC ring shapes.  If a ring template differs from the deposited ring by
/// more than the model-complexity budget, no φ/ψ/ω search can recover the
/// deposited structure and ring flexibility is required first.
#[test]
#[ignore = "oracle reachability gate; run explicitly to verify the rigid-ring model represents deposited 5KZC geometry"]
fn oracle_5kzc_ring_representability() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .display()
        .to_string();
    let deposited = keep_only_glycan(
        &read_structure(format!(
            "{root}/.reglyco-cache/proteins/pdb-5KZC-assembly-1.pdb"
        )),
        &['I'],
    );
    let template = keep_only_glycan(
        &read_structure(format!(
            "{root}/example-output/5kzc-adaptive-current/fitted.pdb"
        )),
        &['B'],
    );
    let mut worst_ring = 0.0_f64;
    let mut worst_residue = 0_i32;
    let mut all_atom_rmsd_total = 0.0_f64;
    let mut all_atom_count = 0usize;
    eprintln!("oracle-ring-representability: residue  ring_rmsd  residue_heavy_rmsd");
    for number in 1..=9 {
        let deposited_residue = ResidueId {
            chain: "I".into(),
            number,
            insertion_code: None,
        };
        let template_residue = ResidueId {
            chain: "B".into(),
            number,
            insertion_code: None,
        };
        let deposited_ring = ring_points(&deposited, &deposited_residue);
        let template_ring = ring_points(&template, &template_residue);
        assert!(
            deposited_ring.len() == 6 && template_ring.len() == 6,
            "residue {number}: ring atom set incomplete"
        );
        let ring_rmsd = ring_frame_rmsd(&template_ring, &deposited_ring, &format!("I:{number}"));
        let template_atoms = template
            .atoms()
            .into_iter()
            .filter(|atom| {
                atom.residue == template_residue && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        let template_points = template_atoms
            .iter()
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let deposited_points = template_atoms
            .iter()
            .filter_map(|atom| atom_position(&deposited, &deposited_residue, &atom.name))
            .collect::<Vec<_>>();
        let residue_rmsd =
            if deposited_points.len() == template_points.len() && !deposited_points.is_empty() {
                ring_frame_rmsd(
                    &template_points,
                    &deposited_points,
                    &format!("I:{number} full"),
                )
            } else {
                ring_rmsd
            };
        all_atom_rmsd_total += residue_rmsd * residue_rmsd * deposited_points.len().max(1) as f64;
        all_atom_count += deposited_points.len().max(1);
        if ring_rmsd > worst_ring {
            worst_ring = ring_rmsd;
            worst_residue = number;
        }
        eprintln!(
            "oracle-ring-representability: residue={number:2} ring={ring_rmsd:.4} residue={residue_rmsd:.4}"
        );
    }
    let all_atom_rmsd = (all_atom_rmsd_total / all_atom_count.max(1) as f64).sqrt();
    eprintln!(
        "oracle-ring-representability: worst_ring={worst_ring:.4} (residue {worst_residue}) all_atom={all_atom_rmsd:.4}"
    );
    assert!(
        worst_ring < 0.35,
        "rigid template ring differs from deposited by {worst_ring:.4} A at residue {worst_residue}; ring flexibility is required before torsion search can reach the deposited geometry"
    );
}

fn read_structure(path: impl AsRef<Path>) -> Structure {
    glysys::read_pdb(path, &BuildOptions::default()).unwrap()
}

fn keep_only_glycan(structure: &Structure, chains: &[char]) -> Structure {
    let keep = structure
        .residues()
        .into_iter()
        .filter(|record| chains.contains(&record.id.chain.chars().next().unwrap_or(' ')))
        .map(|record| record.id.clone())
        .collect::<BTreeSet<_>>();
    let mut only = structure.clone();
    let remove = only
        .residues()
        .into_iter()
        .filter(|record| !keep.contains(&record.id))
        .map(|record| record.id.clone())
        .collect::<BTreeSet<_>>();
    only.remove_residues(&remove);
    only
}

fn atom_position(structure: &Structure, residue: &ResidueId, name: &str) -> Option<Vec3> {
    let id = structure.find_atom(residue, name)?;
    structure.atom(id).map(|atom| atom.position)
}

fn normalize(v: Vec3) -> Vec3 {
    let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt().max(1.0e-12);
    Vec3 {
        x: v.x / len,
        y: v.y / len,
        z: v.z / len,
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}

fn scale(a: Vec3, s: f64) -> Vec3 {
    Vec3 {
        x: a.x * s,
        y: a.y * s,
        z: a.z * s,
    }
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn frame(c1: Vec3, o5: Vec3, o3: Vec3) -> [Vec3; 3] {
    let x = normalize(sub(o5, c1));
    let y = normalize(cross(x, normalize(sub(o3, c1))));
    let z = cross(x, y);
    [x, y, z]
}

/// Rigidly align `moving`'s first residue C1/O5/O3 frame onto `reference`'s
/// `ref_residue` frame.  Returns a translated/rotated copy.
fn align_root(moving: &Structure, reference: &Structure, ref_residue: &ResidueId) -> Structure {
    let moving_residue = moving
        .residues()
        .into_iter()
        .map(|r| r.id.clone())
        .find(|id| {
            atom_position(moving, id, "C1").is_some()
                && atom_position(moving, id, "O5").is_some()
                && atom_position(moving, id, "O3").is_some()
        })
        .expect("moving structure has no complete pyranose ring");
    let (m_c1, m_o5, m_o3) = (
        atom_position(moving, &moving_residue, "C1").unwrap(),
        atom_position(moving, &moving_residue, "O5").unwrap(),
        atom_position(moving, &moving_residue, "O3").unwrap(),
    );
    let (r_c1, r_o5, r_o3) = (
        atom_position(reference, ref_residue, "C1").unwrap(),
        atom_position(reference, ref_residue, "O5").unwrap(),
        atom_position(reference, ref_residue, "O3").unwrap(),
    );
    let m = frame(m_c1, m_o5, m_o3);
    let r = frame(r_c1, r_o5, r_o3);
    let updates = moving
        .atoms()
        .into_iter()
        .map(|atom| {
            let d = sub(atom.position, m_c1);
            // Express d in the moving frame, then re-express in the reference
            // frame: local = M^T d ; world = R local + r_c1.
            let local = Vec3 {
                x: d.x * m[0].x + d.y * m[0].y + d.z * m[0].z,
                y: d.x * m[1].x + d.y * m[1].y + d.z * m[1].z,
                z: d.x * m[2].x + d.y * m[2].y + d.z * m[2].z,
            };
            (
                atom.id,
                add(
                    r_c1,
                    Vec3 {
                        x: local.x * r[0].x + local.y * r[1].x + local.z * r[2].x,
                        y: local.x * r[0].y + local.y * r[1].y + local.z * r[2].y,
                        z: local.x * r[0].z + local.y * r[1].z + local.z * r[2].z,
                    },
                ),
            )
        })
        .collect::<Vec<_>>();
    let mut aligned = moving.clone();
    aligned.set_atom_positions(updates.into_iter()).unwrap();
    aligned
}

fn split_pdb_models(contents: &str) -> Vec<Structure> {
    let options = BuildOptions::default();
    let mut models = Vec::new();
    let mut current = String::new();
    let mut inside = false;
    for line in contents.lines() {
        if line.starts_with("MODEL") {
            current.clear();
            inside = true;
        } else if line.starts_with("ENDMDL") {
            if inside {
                current.push_str("END\n");
                models.push(glysys::read_pdb_str(&current, &options).unwrap());
                inside = false;
            }
        } else if inside {
            current.push_str(line);
            current.push('\n');
        }
    }
    models
}

fn score_candidates(
    map_path: impl AsRef<Path>,
    candidates: &[(String, Structure, DensityTarget)],
) -> Vec<(String, f64, f64)> {
    let map = DensityMap::open(map_path).unwrap();
    let scorer = DensityScorer::new(
        map,
        DensityScoreOptions {
            sigma_angstrom: Some(0.65),
            periodic: true,
            ..DensityScoreOptions::default()
        },
    )
    .unwrap();
    let structures = candidates.iter().map(|(_, s, _)| s).collect::<Vec<_>>();
    let targets = candidates
        .iter()
        .map(|(_, _, t)| t.clone())
        .collect::<Vec<_>>();
    let region = scorer.fixed_region(&structures, &targets, 4.0).unwrap();
    candidates
        .iter()
        .map(|(name, structure, target)| {
            let score = scorer
                .score_fixed_region(&region, structure, std::slice::from_ref(target))
                .unwrap();
            (name.clone(), score.likelihood_gain, score.correlation)
        })
        .collect()
}

fn rmsd_to_reference(candidate: &Structure, reference: &Structure) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    let candidate_residues = candidate.residues().into_iter().collect::<Vec<_>>();
    let reference_residues = reference.residues().into_iter().collect::<Vec<_>>();
    for (candidate_residue, reference_residue) in
        candidate_residues.iter().zip(reference_residues.iter())
    {
        for atom in candidate.atoms() {
            if atom.element.eq_ignore_ascii_case("H") || atom.residue != candidate_residue.id {
                continue;
            }
            let Some(reference_id) = reference.find_atom(&reference_residue.id, &atom.name) else {
                continue;
            };
            let Some(reference_atom) = reference.atom(reference_id) else {
                continue;
            };
            let d2 = (atom.position.x - reference_atom.position.x).powi(2)
                + (atom.position.y - reference_atom.position.y).powi(2)
                + (atom.position.z - reference_atom.position.z).powi(2);
            total += d2;
            count += 1;
        }
    }
    (total / count.max(1) as f64).sqrt()
}

#[test]
#[ignore = "requires the optional cached 5KZC oracle fixtures"]
fn diagnose_5kzc_scoring() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .display()
        .to_string();
    let protein_path = format!("{root}/.reglyco-cache/proteins/pdb-5KZC-assembly-1.pdb");
    let fitted_path = format!("{root}/example-output/5kzc-adaptive-current/fitted.pdb");
    let map_path = format!("{root}/.reglyco-cache/proteins/maps/eds-5kzc.ccp4");
    let ensemble_path = format!("{root}/.reglyco-cache/573ff8b55e4c9c59.pdb");

    let protein = read_structure(&protein_path);
    let deposited = keep_only_glycan(&protein, &['I']);
    let fitted = keep_only_glycan(&read_structure(&fitted_path), &['B']);

    let models = split_pdb_models(&std::fs::read_to_string(&ensemble_path).unwrap());
    let native60 = keep_only_glycan(&models[60], &['X']);
    let native = align_root(
        &native60,
        &deposited,
        &ResidueId {
            chain: "I".into(),
            number: 1,
            insertion_code: None,
        },
    );

    let site = ResidueId {
        chain: "A".into(),
        number: 79,
        insertion_code: None,
    };
    let target_for = |structure: &Structure| DensityTarget {
        site: site.clone(),
        glycan_residues: structure
            .residues()
            .into_iter()
            .map(|record| record.id.clone())
            .collect(),
    };

    let mut candidates = vec![
        (
            "deposited".to_string(),
            deposited.clone(),
            target_for(&deposited),
        ),
        (
            "current_fit".to_string(),
            fitted.clone(),
            target_for(&fitted),
        ),
        (
            "native60_aligned".to_string(),
            native.clone(),
            target_for(&native),
        ),
    ];
    // Perturbed versions of the deposited pose.
    let tx_updates = deposited
        .atoms()
        .into_iter()
        .map(|atom| {
            (
                atom.id,
                Vec3 {
                    x: atom.position.x + 2.0,
                    y: atom.position.y,
                    z: atom.position.z,
                },
            )
        })
        .collect::<Vec<_>>();
    let mut translate = deposited.clone();
    translate
        .set_atom_positions(tx_updates.into_iter())
        .unwrap();
    candidates.push((
        "deposited_tx2A".to_string(),
        translate,
        target_for(&candidates[0].1),
    ));

    let mut rotated = deposited.clone();
    // Rotate the alpha-1,6 arm subtree (residues starting at I:4) 30 degrees
    // around the I:3 C6->O6 axis to emulate the failing terminal-arm error.
    let parent_c6 = atom_position(
        &rotated,
        &ResidueId {
            chain: "I".into(),
            number: 3,
            insertion_code: None,
        },
        "C6",
    )
    .unwrap();
    let parent_o6 = atom_position(
        &rotated,
        &ResidueId {
            chain: "I".into(),
            number: 3,
            insertion_code: None,
        },
        "O6",
    )
    .unwrap();
    let axis = normalize(sub(parent_o6, parent_c6));
    let angle = 30.0_f64.to_radians();
    let (sin, cos) = angle.sin_cos();
    let arm_updates = rotated
        .atoms()
        .into_iter()
        .filter(|atom| atom.residue.number >= 4)
        .map(|atom| {
            let d = sub(atom.position, parent_c6);
            let parallel = scale(axis, dot(d, axis));
            let perpendicular = sub(d, parallel);
            let w = cross(axis, perpendicular);
            let new_perp = add(scale(perpendicular, cos), scale(w, sin));
            (atom.id, add(add(parent_c6, parallel), new_perp))
        })
        .collect::<Vec<_>>();
    rotated.set_atom_positions(arm_updates.into_iter()).unwrap();
    candidates.push((
        "deposited_arm30".to_string(),
        rotated,
        target_for(&candidates[0].1),
    ));

    let results = score_candidates(&map_path, &candidates);
    eprintln!("5KZC scoring diagnosis (likelihood_gain, cc, rmsd_to_deposited):");
    for (name, ll, cc) in &results {
        let structure = candidates
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, s, _)| s)
            .unwrap();
        let rmsd = if name == "deposited" {
            0.0
        } else {
            rmsd_to_reference(structure, &deposited)
        };
        eprintln!("  {name:24} ll={ll:8.2} cc={cc:8.4} rmsd={rmsd:6.2}");
    }
    // Decision gate: deposited-like pose must outrank the current fit.
    let rank = |name: &str| results.iter().position(|(n, _, _)| n == name).unwrap();
    assert!(
        results[rank("deposited")].1 > results[rank("current_fit")].1,
        "deposited-like pose must outrank current fit"
    );
    assert!(
        results[rank("deposited")].1 > results[rank("deposited_tx2A")].1,
        "a 2 A translation must not outrank the deposited pose"
    );
}

#[test]
#[ignore = "requires the optional cached 5GSQ oracle fixtures"]
fn diagnose_5gsq_scoring() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .display()
        .to_string();
    let protein_path = format!("{root}/.reglyco-cache/proteins/pdb-5GSQ-assembly-1.pdb");
    let map_path = format!("{root}/.reglyco-cache/proteins/maps/eds-5gsq.ccp4");
    let protein = read_structure(&protein_path);

    // Site A: deposited chain E, fitted chain C.
    let deposited_a = keep_only_glycan(&protein, &['E']);
    let fitted_a = keep_only_glycan(
        &read_structure(&format!(
            "{root}/example-output/5gsq-adaptive-current/sites/A_297/fitted.pdb"
        )),
        &['C'],
    );
    // Site B: deposited chain F, fitted chain C (site-specific B fit).
    let deposited_b = keep_only_glycan(&protein, &['F']);
    let fitted_b = keep_only_glycan(
        &read_structure(&format!(
            "{root}/example-output/5gsq-adaptive-current/sites/B_297/fitted.pdb"
        )),
        &['C'],
    );

    let site_a = ResidueId {
        chain: "A".into(),
        number: 297,
        insertion_code: None,
    };
    let site_b = ResidueId {
        chain: "B".into(),
        number: 297,
        insertion_code: None,
    };
    let target_for = |site: &ResidueId, structure: &Structure| DensityTarget {
        site: site.clone(),
        glycan_residues: structure
            .residues()
            .into_iter()
            .map(|record| record.id.clone())
            .collect(),
    };

    let a_candidates = vec![
        (
            "A_deposited".to_string(),
            deposited_a.clone(),
            target_for(&site_a, &deposited_a),
        ),
        (
            "A_current_fit".to_string(),
            fitted_a.clone(),
            target_for(&site_a, &fitted_a),
        ),
    ];
    let b_candidates = vec![
        (
            "B_deposited".to_string(),
            deposited_b.clone(),
            target_for(&site_b, &deposited_b),
        ),
        (
            "B_current_fit".to_string(),
            fitted_b.clone(),
            target_for(&site_b, &fitted_b),
        ),
    ];

    for (label, candidates, reference) in [
        ("5GSQ A", a_candidates, &deposited_a),
        ("5GSQ B", b_candidates, &deposited_b),
    ] {
        let results = score_candidates(&map_path, &candidates);
        eprintln!("{label} scoring diagnosis (likelihood_gain, cc, rmsd_to_deposited):");
        for (name, ll, cc) in &results {
            let structure = candidates
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, s, _)| s)
                .unwrap();
            let rmsd = if name.ends_with("deposited") {
                0.0
            } else {
                rmsd_to_reference(structure, reference)
            };
            eprintln!("  {name:20} ll={ll:8.2} cc={cc:8.4} rmsd={rmsd:6.2}");
        }
        // Decision gate: deposited-like must outrank the current fit.
        let (deposited_name, current_name) = if label.starts_with("5GSQ A") {
            ("A_deposited", "A_current_fit")
        } else {
            ("B_deposited", "B_current_fit")
        };
        assert!(
            results
                .iter()
                .find(|(n, _, _)| n == deposited_name)
                .map(|(_, ll, _)| *ll)
                .unwrap()
                > results
                    .iter()
                    .find(|(n, _, _)| n == current_name)
                    .map(|(_, ll, _)| *ll)
                    .unwrap(),
            "{label}: deposited-like pose must outrank current fit"
        );
    }
}
