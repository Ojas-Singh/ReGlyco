//! The deterministic parts of the extend mode against the Python reference
//! (`glycoflow/fitting/ensemble.py`), on a fixture written by GlycoFlow's
//! `scripts/fitting/export_ensemble_fixture.py` (no model, no map):
//!
//! ```text
//! .venv/bin/python scripts/fitting/export_ensemble_fixture.py --out /path/to/fixtures/ensemble.json
//! GLYCOFLOW_FIT_FIXTURES=/path/to/fixtures cargo test --release -p reglyco-glycoflow \
//!     --test ensemble_parity -- --ignored --nocapture
//! ```
//!
//! Compared: the pucker templates that agree with a template on the built residues, members
//! grafted onto a built model, their clusters, and the spread and order parameter of every residue
//! beyond the built ones. The sampling itself uses different random numbers in the two engines.

use std::collections::BTreeSet;

use reglyco_glycoflow::ensemble::{CLUSTER_RMSD, MAX_CLUSTERS, PUCKER_TOLERANCE, cluster, graft, order_parameter, rmsf, templates_in_state};
use serde_json::Value;

type V3 = [f64; 3];

fn points(v: &Value) -> Vec<V3> {
    v.as_array().unwrap().iter().map(|p| [0, 1, 2].map(|k| p[k].as_f64().unwrap())).collect()
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|i| i.as_u64().unwrap() as usize).collect()
}

#[test]
#[ignore = "needs GLYCOFLOW_FIT_FIXTURES with ensemble.json (GlycoFlow export_ensemble_fixture.py)"]
fn ensemble_parity() {
    let path = std::path::Path::new(&std::env::var("GLYCOFLOW_FIT_FIXTURES").expect("set GLYCOFLOW_FIT_FIXTURES")).join("ensemble.json");
    let fx: Value = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
    let paths: Vec<String> = fx["paths"].as_array().unwrap().iter().map(|p| p.as_str().unwrap().to_string()).collect();
    let built: BTreeSet<String> = fx["built"].as_array().unwrap().iter().map(|p| p.as_str().unwrap().to_string()).collect();

    // pucker templates in the built residues' states
    let ring: Vec<bool> = fx["ring_atoms"].as_array().unwrap().iter().map(|b| b.as_bool().unwrap()).collect();
    let templates: Vec<Vec<[f32; 3]>> =
        fx["templates"].as_array().unwrap().iter().map(|t| points(t).iter().map(|p| p.map(|v| v as f32)).collect()).collect();
    let ours = templates_in_state(&templates, &paths, &ring, fx["template"].as_u64().unwrap() as usize, &built, PUCKER_TOLERANCE);
    println!("templates in the built residues' pucker states: rust {ours:?} python {}", fx["compatible"]);
    assert_eq!(ours, indices(&fx["compatible"]));

    // grafting
    let members: Vec<Vec<V3>> = fx["members"].as_array().unwrap().iter().map(points).collect();
    let x_built = points(&fx["x_built"]);
    let expected: Vec<Vec<V3>> = fx["grafted"].as_array().unwrap().iter().map(points).collect();
    let grafted: Vec<Vec<V3>> = members.iter().map(|m| graft(&paths, m, &x_built, &built)).collect();
    let worst = grafted
        .iter()
        .zip(&expected)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(p, q)| (0..3).map(|k| (p[k] - q[k]).abs()).fold(0.0, f64::max)))
        .fold(0.0, f64::max);
    println!("grafted members: max |rust - python| = {worst:.2e} A");
    assert!(worst < 1e-8, "grafted coordinates differ by {worst} A");

    // clusters (on the reference's grafted coordinates)
    let scored = indices(&fx["scored"]);
    let (clusters, radius) = cluster(&expected, &scored, CLUSTER_RMSD, MAX_CLUSTERS);
    let theirs = fx["clusters"].as_array().unwrap();
    println!(
        "clusters: rust {:?} radius {radius:.6}; python radius {}",
        clusters.iter().map(|c| (c.medoid, c.members.len())).collect::<Vec<_>>(),
        fx["cluster_rmsd"]
    );
    assert_eq!(clusters.len(), theirs.len());
    for (c, t) in clusters.iter().zip(theirs) {
        assert_eq!(c.medoid as u64, t["medoid"].as_u64().unwrap());
        assert_eq!(c.members, indices(&t["members"]));
        assert!((c.population - t["population"].as_f64().unwrap()).abs() < 1e-12);
    }
    assert!((radius - fx["cluster_rmsd"].as_f64().unwrap()).abs() < 1e-9);

    // spread and order parameter of every residue beyond the built ones
    let sigma = fx["sigma"].as_f64().unwrap();
    for (name, r) in fx["residues"].as_object().unwrap() {
        let idx = indices(&r["idx"]);
        let z: Vec<f64> = r["z"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let (order, spread) = (order_parameter(&expected, &idx, &z, sigma), rmsf(&expected, &idx));
        println!(
            "  {name:12} order rust {order:.10} python {:.10}; spread rust {spread:.8} python {:.8}",
            r["order"].as_f64().unwrap(),
            r["rmsf"].as_f64().unwrap()
        );
        assert!((order - r["order"].as_f64().unwrap()).abs() < 1e-9, "{name}: order");
        assert!((spread - r["rmsf"].as_f64().unwrap()).abs() < 1e-9, "{name}: spread");
    }
}
