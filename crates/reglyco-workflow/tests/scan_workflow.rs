//! End-to-end NScan regression test using the same request/asset shape the
//! browser sends: an uploaded protein plus a `scan-glycan.pdb` GlycoShape
//! bundle. This exercises sequon detection, per-site search, the direct
//! joint-compatibility fast path, and the scan report fields.

use reglyco_workflow::{
    AssetData, InputAssets, ProteinInput, ProteinInputKind, ReGlycoOptions, ReGlycoProfile,
    ReGlycoRunRequestV1, SCHEMA_VERSION, WorkflowId, execute,
};
use std::collections::BTreeMap;

const PROTEIN: &str = "\
ATOM      1  N   ASN A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ASN A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ASN A   1       2.000   1.420   0.000  1.00  0.00           C
ATOM      4  O   ASN A   1       1.350   2.440   0.000  1.00  0.00           O
ATOM      5  CB  ASN A   1       1.950  -0.780  -1.220  1.00  0.00           C
ATOM      6  CG  ASN A   1       3.420  -1.020  -1.180  1.00  0.00           C
ATOM      7  OD1 ASN A   1       4.100  -0.500  -0.300  1.00  0.00           O
ATOM      8  ND2 ASN A   1       3.920  -1.860  -2.080  1.00  0.00           N
ATOM      9  CA  ALA A   2       4.000   2.000   0.000  1.00  0.00           C
ATOM     10  CA  THR A   3       5.000   3.000   0.000  1.00  0.00           C
TER
END
";

const PROTEIN_TWO_SITES: &str = "\
ATOM      1  N   ASN A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ASN A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ASN A   1       2.000   1.420   0.000  1.00  0.00           C
ATOM      4  O   ASN A   1       1.350   2.440   0.000  1.00  0.00           O
ATOM      5  CB  ASN A   1       1.950  -0.780  -1.220  1.00  0.00           C
ATOM      6  CG  ASN A   1       3.420  -1.020  -1.180  1.00  0.00           C
ATOM      7  OD1 ASN A   1       4.100  -0.500  -0.300  1.00  0.00           O
ATOM      8  ND2 ASN A   1       3.920  -1.860  -2.080  1.00  0.00           N
ATOM      9  CA  ALA A   2       4.000   2.000   0.000  1.00  0.00           C
ATOM     10  CA  THR A   3       5.000   3.000   0.000  1.00  0.00           C
ATOM     11  N   ASN A  10      30.000   0.000   0.000  1.00  0.00           N
ATOM     12  CA  ASN A  10      31.450   0.000   0.000  1.00  0.00           C
ATOM     13  C   ASN A  10      32.000   1.420   0.000  1.00  0.00           C
ATOM     14  O   ASN A  10      31.350   2.440   0.000  1.00  0.00           O
ATOM     15  CB  ASN A  10      31.950  -0.780  -1.220  1.00  0.00           C
ATOM     16  CG  ASN A  10      33.420  -1.020  -1.180  1.00  0.00           C
ATOM     17  OD1 ASN A  10      34.100  -0.500  -0.300  1.00  0.00           O
ATOM     18  ND2 ASN A  10      33.920  -1.860  -2.080  1.00  0.00           N
ATOM     19  CA  ALA A  11      34.000   2.000   0.000  1.00  0.00           C
ATOM     20  CA  THR A  12      35.000   3.000   0.000  1.00  0.00           C
TER
END
";

fn scan_request() -> ReGlycoRunRequestV1 {
    ReGlycoRunRequestV1 {
        schema_version: SCHEMA_VERSION,
        workflow: WorkflowId::NScan,
        profile: ReGlycoProfile::Public,
        input: ProteinInput {
            kind: ProteinInputKind::Upload,
            label: "mini.pdb".into(),
            source_id: None,
            asset: "protein.pdb".into(),
            sha256: "test".into(),
            source_url: None,
        },
        assignments: Vec::new(),
        options: ReGlycoOptions::default(),
        parent_job_id: None,
        created_at: "2026-01-01T00:00:00Z".into(),
    }
}

fn scan_assets(protein: &str) -> InputAssets {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scan_asset =
        std::fs::read_to_string(manifest.join("test-data/scan-glycan.pdb")).expect("scan asset");
    let mut assets: InputAssets = BTreeMap::new();
    assets.insert(String::from("protein.pdb"), AssetData::Text(protein.into()));
    assets.insert(String::from("scan-glycan.pdb"), AssetData::Text(scan_asset));
    assets
}

#[test]
fn two_site_scan_checks_joint_compatibility_directly() {
    let request = scan_request();
    let assets = scan_assets(PROTEIN_TWO_SITES);
    let bundle = execute(&request, &assets).expect("two-site scan executes");
    let analysis = &bundle.report.analysis;
    assert_eq!(analysis["structuralAccessibilityComputed"], true, "{analysis}");
    assert_eq!(analysis["jointlyCompatibleCount"], 2, "{analysis}");
}

#[test]
fn browser_shaped_scan_completes_with_budget_and_timing_report() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scan_asset =
        std::fs::read_to_string(manifest.join("test-data/scan-glycan.pdb")).expect("scan asset");
    let request = ReGlycoRunRequestV1 {
        schema_version: SCHEMA_VERSION,
        workflow: WorkflowId::NScan,
        profile: ReGlycoProfile::Public,
        input: ProteinInput {
            kind: ProteinInputKind::Upload,
            label: "mini.pdb".into(),
            source_id: None,
            asset: "protein.pdb".into(),
            sha256: "test".into(),
            source_url: None,
        },
        assignments: Vec::new(),
        options: ReGlycoOptions::default(),
        parent_job_id: None,
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    let mut assets: InputAssets = BTreeMap::new();
    assets.insert(String::from("protein.pdb"), AssetData::Text(PROTEIN.into()));
    assets.insert(String::from("scan-glycan.pdb"), AssetData::Text(scan_asset));
    let bundle = execute(&request, &assets).expect("scan executes");
    let analysis = &bundle.report.analysis;
    assert_eq!(
        analysis["structuralAccessibilityComputed"], true,
        "{analysis}"
    );
    assert_eq!(analysis["scanBudget"]["populationSize"], 32);
    assert_eq!(analysis["scanBudget"]["generations"], 25);
    assert!(analysis["timings"]["totalSeconds"].as_f64().is_some());
}
