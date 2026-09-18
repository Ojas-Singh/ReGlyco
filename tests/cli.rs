use std::process::Command;

#[test]
#[ignore = "scientific/performance regression; requires the cached 5KZC EDS map and 512-member GlycoShape ensemble"]
fn cached_5kzc_n79_adaptive_density_regression() {
    let cache = std::path::Path::new(".reglyco-cache");
    let protein = cache.join("proteins/pdb-5KZC-assembly-1.pdb");
    let map = cache.join("proteins/maps/eds-5kzc.ccp4");
    assert!(protein.is_file(), "missing cached 5KZC assembly");
    assert!(map.is_file(), "missing cached 5KZC EDS map");
    let directory = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let output = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .env("RAYON_NUM_THREADS", "4")
        .args(["refine", "--protein"])
        .arg(&protein)
        .args([
            "--replace-glycan",
            "A:79=G63337SS",
            "--level",
            "3",
            "--offline",
            "--objective",
            "density",
            "--density-map",
        ])
        .arg(&map)
        .args([
            "--density-sigma",
            "1.0",
            "--density-difference-map",
            "none",
            "--density-effort",
            "adaptive",
            "--quiet",
            "--output",
        ])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(started.elapsed().as_secs_f64() < 90.0);
    let density: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("density.json")).unwrap())
            .unwrap();
    assert!(density["evaluations"].as_u64().unwrap() < 15_000);
    assert!(density["pre_relax"]["correlation"].as_f64().unwrap() >= 0.70);
    assert!(
        density["recovery"][0]["supported_heavy_atom_rmsd_angstrom"]
            .as_f64()
            .unwrap()
            < 1.0
    );
    assert!(
        density["recovery"][0]["full_tree_heavy_atom_rmsd_angstrom"]
            .as_f64()
            .unwrap()
            < 1.0
    );
    for rmsd in density["recovery"][0]["per_residue_heavy_atom_rmsd_angstrom"]
        .as_object()
        .unwrap()
        .values()
    {
        assert!(rmsd.as_f64().unwrap() < 1.0);
    }
    let validation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("validation.json")).unwrap())
            .unwrap();
    assert_eq!(validation["valid"], true);
    assert!(validation["errors"].as_array().unwrap().is_empty());
    assert!(directory.path().join("fitted.pdb").is_file());
    assert!(directory.path().join("candidates.pdb").is_file());
    assert!(directory.path().join("candidates.json").is_file());
    assert!(
        directory
            .path()
            .join("glycoshape-density-best.pdb")
            .is_file()
    );
    assert!(
        directory
            .path()
            .join("glycoshape-nearest-fit.pdb")
            .is_file()
    );
    assert!(directory.path().join("glycoshape-baselines.json").is_file());
}

#[test]
#[ignore = "deep scientific regression; requires cached 5KZC primary/Fo-Fc maps and the 512-member GlycoShape ensemble"]
fn cached_5kzc_n79_evidence_calibrated_deep_regression() {
    let cache = std::path::Path::new(".reglyco-cache");
    let protein = cache.join("proteins/pdb-5KZC-assembly-1.pdb");
    let primary = cache.join("proteins/maps/eds-5kzc.ccp4");
    let difference = cache.join("proteins/maps/eds-5kzc-diff.ccp4");
    for input in [&protein, &primary, &difference] {
        assert!(input.is_file(), "missing cached input {}", input.display());
    }
    let directory = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let output = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .env("RAYON_NUM_THREADS", "4")
        .args(["refine", "--protein"])
        .arg(&protein)
        .args([
            "--replace-glycan",
            "A:79=G63337SS",
            "--level",
            "3",
            "--offline",
            "--objective",
            "density",
            "--density-map",
        ])
        .arg(&primary)
        .arg("--density-difference-map")
        .arg(&difference)
        .args(["--density-effort", "deep", "--quiet", "--output"])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(started.elapsed().as_secs_f64() < 600.0);
    let density: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("density.json")).unwrap())
            .unwrap();
    assert!(density["sigma_calibration"]["selected_sigma_angstrom"].is_number());
    assert_eq!(density["difference_map"]["channel"], "Fo-Fc");
    assert!(density["pre_relax"]["correlation"].as_f64().unwrap() >= 0.70);
    let recovery = &density["recovery"][0];
    assert!(recovery["root_c1_distance_angstrom"].as_f64().unwrap() < 0.5);
    assert!(
        recovery["three_residue_heavy_atom_rmsd_angstrom"]
            .as_f64()
            .unwrap()
            < 0.5
    );
    assert!(
        recovery["supported_heavy_atom_rmsd_angstrom"]
            .as_f64()
            .unwrap()
            < 1.0
    );
    assert!(
        recovery["full_tree_heavy_atom_rmsd_angstrom"]
            .as_f64()
            .unwrap()
            < 1.0
    );
    for rmsd in recovery["per_residue_heavy_atom_rmsd_angstrom"]
        .as_object()
        .unwrap()
        .values()
    {
        assert!(rmsd.as_f64().unwrap() < 1.0);
    }
    let validation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("validation.json")).unwrap())
            .unwrap();
    assert_eq!(validation["valid"], true);
    assert!(validation["errors"].as_array().unwrap().is_empty());
    for name in [
        "deep-arm-best.pdb",
        "deep-cartesian-best.pdb",
        "fitted.pdb",
        "glycoshape-density-best.pdb",
        "glycoshape-nearest-fit.pdb",
    ] {
        assert!(directory.path().join(name).is_file(), "missing {name}");
    }
}

#[test]
fn cli_builds_a_pdb_only_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "build",
            "--protein",
            "tests/fixtures/protein.pdb",
            "--site",
            "A:1",
            "--glycan",
            "tests/fixtures/glycan.pdb",
            "--output",
        ])
        .arg(directory.path())
        .arg("--no-system")
        .status()
        .unwrap();
    assert!(status.success());
    assert!(directory.path().join("glycoprotein.pdb").is_file());
    assert!(directory.path().join("report.json").is_file());
    assert!(!directory.path().join("system.top").exists());
}

#[test]
fn cli_saxs_model_writes_fit_plot_and_report() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("experiment.dat");
    std::fs::write(
        &data,
        "# q I sigma\n0.01 10.0 0.1\n0.02 5.0 0.1\n0.03 2.5 0.1\n0.04 1.2 0.1\n0.05 0.6 0.1\n",
    )
    .unwrap();
    let output = directory.path().join("model");
    let result = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args(["saxs", "model", "--data"])
        .arg(&data)
        .args([
            "--models",
            "tests/fixtures/protein.pdb",
            "--report",
            "--quiet",
            "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for file in [
        "best-model.pdb",
        "saxs-model.json",
        "saxs-diagnostic.svg",
        "report.json",
        "report.pdf",
        "report.typ",
    ] {
        assert!(output.join(file).is_file(), "missing {file}");
    }
    assert!(output.join("report-assets/saxs-diagnostic.svg").is_file());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["analysis"]["saxs"]["mode"], "single_model");
}

#[test]
fn cli_saxs_ensemble_reweight_and_occupancy_write_machine_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("experiment.dat");
    std::fs::write(
        &data,
        "# q I sigma\n0.01 10.0 0.1\n0.02 5.0 0.1\n0.03 2.5 0.1\n0.04 1.2 0.1\n0.05 0.6 0.1\n",
    )
    .unwrap();
    let ensemble_output = directory.path().join("ensemble");
    let ensemble = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args(["saxs", "ensemble", "--data"])
        .arg(&data)
        .args([
            "--models",
            "tests/fixtures/protein.pdb",
            "--quiet",
            "--output",
        ])
        .arg(&ensemble_output)
        .status()
        .unwrap();
    assert!(ensemble.success());
    assert!(ensemble_output.join("unbiased-fit.dat").is_file());
    assert!(ensemble_output.join("saxs-ensemble.json").is_file());

    let reweight_output = directory.path().join("reweight");
    let reweight = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args(["saxs", "reweight", "--data"])
        .arg(&data)
        .args([
            "--models",
            "tests/fixtures/protein.pdb",
            "--quiet",
            "--output",
        ])
        .arg(&reweight_output)
        .status()
        .unwrap();
    assert!(reweight.success());
    assert!(reweight_output.join("reweighted-fit.dat").is_file());
    assert!(reweight_output.join("weights.csv").is_file());

    let occupancy_output = directory.path().join("occupancy");
    let occupancy = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args(["saxs", "occupancy", "--data"])
        .arg(&data)
        .args([
            "--protein",
            "tests/fixtures/protein.pdb",
            "--candidate",
            "A:1=none",
            "--report",
            "--quiet",
            "--output",
        ])
        .arg(&occupancy_output)
        .status()
        .unwrap();
    assert!(occupancy.success());
    assert!(occupancy_output.join("saxs-occupancy.json").is_file());
    assert!(
        occupancy_output
            .join("report-assets/saxs-occupancy.svg")
            .is_file()
    );
    let analysis: serde_json::Value = serde_json::from_slice(
        &std::fs::read(occupancy_output.join("saxs-occupancy.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(analysis["site_occupancy"]["A:1"], 0.0);
}

#[test]
fn cli_saxs_occupancy_accepts_json_candidate_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("experiment.dat");
    std::fs::write(
        &data,
        "# q I sigma\n0.01 10.0 0.1\n0.02 5.0 0.1\n0.03 2.5 0.1\n0.04 1.2 0.1\n0.05 0.6 0.1\n",
    )
    .unwrap();
    let manifest = directory.path().join("candidates.json");
    std::fs::write(
        &manifest,
        r#"{"sites":[{"site":"A:1","candidates":[{"id":"none","prior":2.0}]}]}"#,
    )
    .unwrap();
    let output = directory.path().join("manifest");
    let result = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args(["saxs", "occupancy", "--data"])
        .arg(&data)
        .args([
            "--protein",
            "tests/fixtures/protein.pdb",
            "--candidate-manifest",
        ])
        .arg(&manifest)
        .args(["--quiet", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let analysis: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("saxs-occupancy.json")).unwrap())
            .unwrap();
    assert_eq!(analysis["combinations"][0]["prior"], 2.0);
    assert_eq!(
        analysis["combinations"][0]["assignments"][0]["candidate"],
        "none"
    );
}

#[test]
fn cli_saxs_occupancy_rejects_combinations_above_cap() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("experiment.dat");
    std::fs::write(&data, "# q I sigma\n0.01 10.0 0.1\n0.02 5.0 0.1\n").unwrap();
    let output = directory.path().join("guard");
    let result = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args(["saxs", "occupancy", "--data"])
        .arg(&data)
        .args([
            "--protein",
            "tests/fixtures/protein.pdb",
            "--candidate",
            "A:1=none,absent",
            "--candidate",
            "A:2=none,absent",
            "--max-combinations",
            "3",
            "--quiet",
            "--output",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("at least 4 combinations"), "{stderr}");
    assert!(!output.exists());
}

#[test]
fn cli_report_writes_self_contained_pdf_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "build",
            "--protein",
            "tests/fixtures/protein.pdb",
            "--attach",
            "A:1=tests/fixtures/glycan.pdb",
            "--no-system",
            "--report",
            "--output",
        ])
        .arg(directory.path())
        .args(["--population", "8", "--generations", "2", "--seed", "7"])
        .status()
        .unwrap();
    assert!(status.success());
    for file in ["report.json", "report.pdf", "report.typ"] {
        assert!(directory.path().join(file).is_file(), "missing {file}");
    }
    assert!(
        std::fs::read(directory.path().join("report.pdf"))
            .unwrap()
            .starts_with(b"%PDF")
    );
    assert!(directory.path().join("report-assets").is_dir());
}

#[test]
fn validate_report_uses_a_stem_derived_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("validation.json");
    let status = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "validate",
            "--input",
            "tests/fixtures/protein.pdb",
            "--output",
        ])
        .arg(&output)
        .arg("--report")
        .status()
        .unwrap();
    assert!(status.success());
    for file in [
        "validation.json",
        "validation-report.json",
        "validation-report.pdf",
    ] {
        assert!(directory.path().join(file).is_file(), "missing {file}");
    }
}

#[test]
fn cli_rejects_mismatched_site_and_glycan_counts() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "build",
            "--protein",
            "tests/fixtures/protein.pdb",
            "--site",
            "A:1",
            "--site",
            "B:2",
            "--glycan",
            "tests/fixtures/glycan.pdb",
            "--output",
        ])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--site values"));
}

#[test]
fn cli_search_writes_structure_and_machine_readable_reports() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "search",
            "--protein",
            "tests/fixtures/protein.pdb",
            "--attach",
            "A:1=tests/fixtures/glycan.pdb",
            "--output",
        ])
        .arg(directory.path())
        .args(["--population", "8", "--generations", "2", "--seed", "7"])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(directory.path().join("structure.pdb").is_file());
    assert!(directory.path().join("search.json").is_file());
    assert!(directory.path().join("report.json").is_file());
    let search: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("search.json")).unwrap())
            .unwrap();
    assert_eq!(search["sites"].as_array().unwrap().len(), 1);
    assert_eq!(search["seed"], 7);
}

#[test]
fn energy_build_reuses_one_topology_and_reports_local_minimization() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "build",
            "--protein",
            "tests/fixtures/protein.pdb",
            "--attach",
            "A:1=tests/fixtures/glycan.pdb",
            "--interact",
            "--min",
            "--min-iterations",
            "1",
            "--population",
            "4",
            "--generations",
            "1",
            "--no-system",
            "--report",
            "--quiet",
            "--output",
        ])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let search: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("search.json")).unwrap())
            .unwrap();
    assert_eq!(
        search["energy_diagnostics"]["topology_parameterizations"],
        1
    );
    assert!(
        search["energy_diagnostics"]["minimizations"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(search["energy_diagnostics"]["cache_hits"].as_u64().unwrap() > 0);
    assert!(
        search["energy_diagnostics"]["active_atoms"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        search["energy_diagnostics"]["neighbor_pairs"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        !search["minimized_coordinates"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(search["energy_cutoff_angstrom"], 10.0);
    assert_eq!(search["minimization_radius_angstrom"], 5.0);
    assert!(
        std::fs::read(directory.path().join("report.pdf"))
            .unwrap()
            .starts_with(b"%PDF")
    );
}

#[test]
fn cli_refine_runs_search_and_physical_relaxation() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "refine",
            "--protein",
            "tests/fixtures/protein.pdb",
            "--attach",
            "A:1=tests/fixtures/glycan.pdb",
            "--output",
        ])
        .arg(directory.path())
        .args([
            "--population",
            "4",
            "--generations",
            "1",
            "--max-iterations",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    for file in [
        "structure.pdb",
        "search.json",
        "relaxation.json",
        "report.json",
    ] {
        assert!(directory.path().join(file).is_file(), "missing {file}");
    }
}

#[test]
fn cli_scan_attaches_glcnac_to_every_sequon() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_reglyco"))
        .args([
            "scan",
            "--protein",
            "tests/fixtures/sequon-protein.pdb",
            "--glycan",
            "tests/fixtures/glycan.pdb",
            "--output",
        ])
        .arg(directory.path())
        .args(["--population", "32", "--generations", "1"])
        .status()
        .unwrap();
    assert!(status.success());
    let scan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("scan.json")).unwrap())
            .unwrap();
    assert_eq!(scan["attachment_count"], 1);
    assert_eq!(scan["sequons"][0]["motif"], "NAT");
    assert!(directory.path().join("structure.pdb").is_file());
    assert!(directory.path().join("search.json").is_file());
    assert!(directory.path().join("report.json").is_file());
}
