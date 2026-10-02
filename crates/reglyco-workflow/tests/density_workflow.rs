//! End-to-end density workflow (GlycoFlow) with the request/asset shape of a
//! browser or service client: an uploaded model, `density.map` and the
//! GlycoFlow model files as input assets.
#![cfg(all(feature = "full", not(target_arch = "wasm32")))]

use std::path::PathBuf;

use reglyco_workflow::{AssetData, InputAssets, ReGlycoRunRequestV1, WorkflowStatus, execute};

#[test]
#[ignore = "GlycoFlow scientific regression (CPU, minutes); needs $GLYCOFLOW_MODEL and 5KZC.pdb + eds-5kzc.ccp4 under $REGLYCO_DATA"]
fn glycoflow_density_workflow_fits_5kzc_n79() {
    let data = PathBuf::from(std::env::var_os("REGLYCO_DATA").expect("set REGLYCO_DATA"));
    let model = PathBuf::from(std::env::var_os("GLYCOFLOW_MODEL").expect("set GLYCOFLOW_MODEL"));
    let protein = std::fs::read_to_string(data.join("structures/5KZC.pdb")).unwrap();
    let mut assets = InputAssets::from([
        ("protein.pdb".to_string(), AssetData::Text(protein)),
        (
            "density.map".to_string(),
            AssetData::Bytes(std::fs::read(data.join("maps/eds-5kzc.ccp4")).unwrap()),
        ),
    ]);
    for name in [
        "glycoflow.safetensors",
        "glycoflow.json",
        "residue_library.json",
    ] {
        assets.insert(
            name.to_string(),
            AssetData::Bytes(std::fs::read(model.join(name)).unwrap()),
        );
    }
    let request: ReGlycoRunRequestV1 = serde_json::from_value(serde_json::json!({
        "schemaVersion": 1,
        "workflow": "density",
        "profile": "full",
        "input": {"kind": "upload", "label": "5KZC.pdb", "asset": "protein.pdb", "sha256": "test"},
        "assignments": [{
            "id": "a79",
            "site": {"chain": "A", "residueNumber": 79},
            "residueName": "ASN",
            "glycanId": "existing",
            "provenance": "manual"
        }],
        "options": {"seed": 0},
        "createdAt": "2026-01-01T00:00:00Z"
    }))
    .unwrap();
    let bundle = execute(&request, &assets).unwrap();
    assert!(bundle.error.is_none(), "{:?}", bundle.error);
    assert!(
        bundle.warnings.iter().all(|w| !w.contains("GLYCAM")),
        "{:?}",
        bundle.warnings
    );
    assert!(matches!(bundle.status, WorkflowStatus::Succeeded));
    assert!(
        bundle
            .primary_structure
            .as_deref()
            .unwrap()
            .contains("ATOM")
    );
    let names: Vec<&str> = bundle.artifacts.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"candidates.pdb"), "{names:?}");
    assert!(names.contains(&"glycoflow-fit.json"), "{names:?}");
    assert!(!names.contains(&"glycoflow.safetensors"), "{names:?}");
    let fit = &bundle.report.analysis["density"]["fit"];
    let recovery = &fit["sites"][0]["evaluation"]["recovery"];
    eprintln!("recovery: {recovery}");
    assert!(recovery["full_rmsd"].as_f64().unwrap() < 1.3, "{recovery}");
}
