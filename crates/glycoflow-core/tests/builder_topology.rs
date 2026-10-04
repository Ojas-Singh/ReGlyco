//! Builder, topology, tokens, kinematics and PDB parity against the Python fixtures (no weights).

mod common;

use common::*;
use glycoflow_core::geometry::{
    dihedrals, rotate_torsions, set_torsions, torsion_gradient, torsion_jacobian,
};
use glycoflow_core::guidance::clash_energy_grad;
use glycoflow_core::pdb::{align_ensemble, core_atoms, write_pdb, PdbAtoms};
use glycoflow_core::Glycan;

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn templates_and_topology_match_python() {
    let lib = library();
    let vocab = meta().vocab;
    for name in CASES {
        let fx = Fixture::load(name);
        let built = lib.build(fx.sequence(), None).unwrap();
        assert_eq!(
            built.atom_names,
            fx.strings("atom_names"),
            "{name}: atom names"
        );
        assert_eq!(
            built.res_names,
            fx.strings("res_names"),
            "{name}: residue names"
        );
        let rid: Vec<i64> = serde_json::from_str(&fx.meta["res_ids"]).unwrap();
        assert_eq!(built.res_ids, rid, "{name}: residue ids");
        assert_eq!(built.elements, fx.strings("elements"), "{name}: elements");
        assert_eq!(
            built.res_paths,
            fx.strings("res_paths"),
            "{name}: residue paths"
        );
        let bonds: Vec<[usize; 2]> = fx
            .i64("bonds")
            .chunks(2)
            .map(|c| [c[0] as usize, c[1] as usize])
            .collect();
        assert_eq!(built.bonds, bonds, "{name}: bonds");
        let e_build = max_dist(&built.coords_f32(), &fx.p3("build_coords"));
        assert!(e_build < 1e-4, "{name}: build coords differ by {e_build}");

        // explicit pucker states (Python: fake rng picking state (j+1) mod n for the j-th ring)
        let choices: Vec<Option<usize>> = fx
            .i64("alt_choices")
            .iter()
            .map(|&k| if k < 0 { None } else { Some(k as usize) })
            .collect();
        let alt = lib
            .build_with_states(fx.sequence(), None, &choices)
            .unwrap();
        assert_eq!(alt.residue_states, choices, "{name}: states used");
        let e_alt = max_dist(&alt.coords_f32(), &fx.p3("alt_build_coords"));
        assert!(
            e_alt < 1e-4,
            "{name}: pucker-state build coords differ by {e_alt}"
        );

        let g = Glycan::from_builds(&[built]).unwrap();
        let e_tpl = max_dist(&g.templates[0], &fx.p3("template"));
        assert!(e_tpl < 1e-4, "{name}: centred template differs by {e_tpl}");
        let t = &g.topology;
        let ring: Vec<bool> = fx.u8("ring_atoms").iter().map(|&x| x != 0).collect();
        assert_eq!(t.ring_atoms, ring, "{name}: ring atoms");
        assert_eq!(t.topo_dist, fx.u8("topo"), "{name}: topological distances");
        let quads: Vec<[usize; 4]> = fx
            .i64("quads")
            .chunks(4)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize])
            .collect();
        assert_eq!(
            t.quads, quads,
            "{name}: torsion quads (selection and order)"
        );
        let distal: Vec<bool> = fx.u8("distal").iter().map(|&x| x != 0).collect();
        assert_eq!(t.distal_mask(), distal, "{name}: distal masks");
        let tokens: Vec<[u32; 5]> = fx
            .i64("tokens")
            .chunks(5)
            .map(|c| {
                [
                    c[0] as u32,
                    c[1] as u32,
                    c[2] as u32,
                    c[3] as u32,
                    c[4] as u32,
                ]
            })
            .collect();
        assert_eq!(g.tokens(&vocab), tokens, "{name}: tokens");
        let core: Vec<usize> = fx.i64("core_atoms").iter().map(|&i| i as usize).collect();
        assert_eq!(
            core_atoms(&g.res_ids, &g.bonds, 5),
            core,
            "{name}: core atoms"
        );

        let atoms = PdbAtoms {
            atom_names: &g.atom_names,
            res_names: &g.res_names,
            res_ids: &g.res_ids,
            elements: &g.elements,
        };
        let fx_tpl = fx.p3("template");
        assert_eq!(
            write_pdb(&atoms, &[&fx_tpl]),
            fx.meta["pdb_text"],
            "{name}: PDB text"
        );
        println!(
            "{name:20} N={:3} T={:2}  build {e_build:.2e} A, pucker-state build {e_alt:.2e} A, template {e_tpl:.2e} A",
            g.n_atoms(),
            g.n_torsions()
        );
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn kinematics_match_python() {
    let lib = library();
    for name in CASES {
        let fx = Fixture::load(name);
        let g = Glycan::from_builds(&[lib.build(fx.sequence(), None).unwrap()]).unwrap();
        let (q, dist) = (&g.topology.quads, &g.topology.distal);
        let n = g.n_atoms();
        let nt = g.n_torsions();
        let tpl = fx.p3("template");
        let e_dih = max_ang(&dihedrals(&tpl, q), &fx.f32("template_torsions"));
        assert!(e_dih < 1e-4, "{name}: dihedrals differ by {e_dih}");
        let (target, want) = (fx.f32("set_target"), fx.p3("set_coords"));
        let (delta, want_rot) = (fx.f32("rot_delta"), fx.p3("rot_coords"));
        let (mut e_set, mut e_set_tau, mut e_rot) = (0f32, 0f32, 0f32);
        for s in 0..target.len() / nt {
            let mut x = tpl.clone();
            set_torsions(&mut x, q, dist, &target[s * nt..(s + 1) * nt]);
            e_set = e_set.max(max_dist(&x, &want[s * n..(s + 1) * n]));
            e_set_tau = e_set_tau.max(max_ang(&dihedrals(&x, q), &target[s * nt..(s + 1) * nt]));
            let mut y = tpl.clone();
            rotate_torsions(&mut y, q, dist, &delta[s * nt..(s + 1) * nt]);
            e_rot = e_rot.max(max_dist(&y, &want_rot[s * n..(s + 1) * n]));
        }
        assert!(
            e_set < 1e-4 && e_rot < 1e-4 && e_set_tau < 1e-4,
            "{name}: set {e_set} rot {e_rot} tau {e_set_tau}"
        );
        println!("{name:20} dihedral {e_dih:.2e} rad, set_torsions {e_set:.2e} A (torsions hit to {e_set_tau:.2e} rad), rotate {e_rot:.2e} A");
    }
}

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn alignment_matches_python() {
    let lib = library();
    for name in CASES {
        let fx = Fixture::load(name);
        let g = Glycan::from_builds(&[lib.build(fx.sequence(), None).unwrap()]).unwrap();
        let n = g.n_atoms();
        let models: Vec<Vec<[f32; 3]>> = fx
            .p3("heun32_coords")
            .chunks(n)
            .map(|c| c.to_vec())
            .collect();
        let al = align_ensemble(&models, &core_atoms(&g.res_ids, &g.bonds, 5));
        let flat: Vec<[f32; 3]> = al.concat();
        let e = max_dist(&flat, &fx.p3("heun32_aligned"));
        assert!(e < 1e-4, "{name}: aligned ensemble differs by {e}");
    }
}

/// Analytic torsion Jacobian and gradient propagation vs central finite differences of the
/// (f64) kinematics, at a generic (non-zero) rotation; f32 Jacobian vs f64.
#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)"]
fn jacobian_matches_finite_differences() {
    let lib = library();
    for name in [
        "core_man3",
        "sialyl_biantennary",
        "man8",
        "araf4",
        "sulfated",
    ] {
        let fx = Fixture::load(name);
        let g = Glycan::from_builds(&[lib.build(fx.sequence(), None).unwrap()]).unwrap();
        let (q, dist) = (&g.topology.quads, &g.topology.distal);
        let nt = g.n_torsions();
        let tpl: Vec<[f64; 3]> = fx.p3("template").iter().map(|p| p.map(f64::from)).collect();
        let delta0: Vec<f64> = fx.f32("rot_delta")[..nt]
            .iter()
            .map(|&d| d as f64)
            .collect();
        let rot = |d: &[f64]| {
            let mut x = tpl.clone();
            rotate_torsions(&mut x, q, dist, d);
            x
        };
        let x0 = rot(&delta0);
        let jac = torsion_jacobian(&x0, q, dist);
        let h = 1e-4;
        let mut worst = 0f64;
        for k in 0..nt {
            let mut dp = delta0.clone();
            dp[k] += h;
            let mut dm = delta0.clone();
            dm[k] -= h;
            let (xp, xm) = (rot(&dp), rot(&dm));
            let mut dense = vec![[0f64; 3]; x0.len()];
            for &(i, v) in &jac[k] {
                dense[i] = v;
            }
            for i in 0..x0.len() {
                for c in 0..3 {
                    worst = worst.max(((xp[i][c] - xm[i][c]) / (2.0 * h) - dense[i][c]).abs());
                }
            }
        }
        assert!(worst < 1e-6, "{name}: Jacobian vs FD max error {worst}");

        // the f32 Jacobian (used by the sampler) at the same conformer
        let x0f: Vec<[f32; 3]> = x0.iter().map(|p| p.map(|v| v as f32)).collect();
        let jac32 = torsion_jacobian(&x0f, q, dist);
        let mut worst32 = 0f64;
        for (a, b) in jac32.iter().flatten().zip(jac.iter().flatten()) {
            for c in 0..3 {
                worst32 = worst32.max((a.1[c] as f64 - b.1[c]).abs());
            }
        }
        assert!(
            worst32 < 1e-4,
            "{name}: f32 Jacobian differs from f64 by {worst32}"
        );

        // dE/dtau of a soft contact energy (floors 6 A so that many pairs contribute)
        let pairs: Vec<(usize, usize, f32)> = (0..g.n_atoms())
            .flat_map(|i| (i + 1..g.n_atoms()).map(move |j| (i, j)))
            .filter(|&(i, j)| g.topology.topo_dist[i * g.n_atoms() + j] >= 4)
            .map(|(i, j)| (i, j, 6.0))
            .collect();
        let energy = |d: &[f64]| clash_energy_grad(&rot(d), &pairs).0;
        let (_, gx) = clash_energy_grad(&x0, &pairs);
        let gt = torsion_gradient(&x0, q, dist, &gx);
        let mut worst_rel = 0f64;
        for k in 0..nt {
            let mut dp = delta0.clone();
            dp[k] += h;
            let mut dm = delta0.clone();
            dm[k] -= h;
            let fd = (energy(&dp) - energy(&dm)) / (2.0 * h);
            worst_rel = worst_rel.max((fd - gt[k]).abs() / (1.0 + fd.abs()));
        }
        assert!(
            worst_rel < 1e-6,
            "{name}: dE/dtau vs FD relative error {worst_rel}"
        );
        println!("{name:20} T={nt:2}: |dx/dtau - FD| {worst:.1e} A/rad (f32 Jacobian {worst32:.1e}), dE/dtau rel. error {worst_rel:.1e}");
    }
}
