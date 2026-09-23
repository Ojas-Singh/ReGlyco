#![cfg(all(feature = "webgpu", not(target_arch = "wasm32")))]
use glysys::{BuildOptions, SystemBuilder, read_pdb_str};
use reglyco_core::{
    Anomer, GlycanQuery, GlycanSource, GlycosylationSite, SearchConfig, SearchSite,
};
use reglyco_ensemble::{ensemble_from_pdb, gpu, sample_attached_ensemble_with_cancel_async};
#[test]
#[ignore = "requires Vulkan; software is correctness evidence only"]
fn independent_chain_gpu_geometry_preserves_seeded_transitions() {
    pollster::block_on(async {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        };
        let protein = read_pdb_str(
            include_str!("../../../tests/fixtures/protein.pdb"),
            &options,
        )
        .unwrap();
        let builder = SystemBuilder::new(options).unwrap();
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle("fixture".into()),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let ensemble = ensemble_from_pdb(
            include_str!("../../../tests/fixtures/glycan.pdb"),
            None,
            query,
            "fixture",
        )
        .unwrap();
        let sites = vec![SearchSite {
            site: GlycosylationSite::new("A", 1),
            ensemble,
        }];
        for seed in 0..5 {
            let config = SearchConfig {
                seed,
                mh_chains: 3,
                burn_in_steps: Some(4),
                thinning_steps: Some(2),
                ..Default::default()
            };
            gpu::configure("cpu");
            let (cpu, cd) = sample_attached_ensemble_with_cancel_async(
                &protein,
                &sites,
                12,
                &config,
                &builder,
                || false,
            )
            .await
            .unwrap();
            gpu::finish();
            gpu::configure("webgpu");
            let (actual, gd) = sample_attached_ensemble_with_cancel_async(
                &protein,
                &sites,
                12,
                &config,
                &builder,
                || false,
            )
            .await
            .unwrap();
            let report = gpu::finish();
            assert!(report["gpuEvaluations"].as_u64().unwrap() > 0, "{report}");
            assert_eq!(cd.mh_proposals, gd.mh_proposals);
            assert_eq!(cd.mh_accepts, gd.mh_accepts);
            assert!(
                gd.mh_accepts > 0,
                "GPU steric proposals never committed: {report}"
            );
            for (a, b) in cpu.iter().zip(actual) {
                assert_eq!(a.structure.to_pdb_string(), b.structure.to_pdb_string());
            }
        }
    });
}
#[test]
fn minimized_collection_keeps_prepared_chemistry_and_hydrogens() {
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    };
    let protein = read_pdb_str(
        include_str!("../../../tests/fixtures/protein.pdb"),
        &options,
    )
    .unwrap();
    let builder = SystemBuilder::new(options).unwrap();
    let query = GlycanQuery {
        source: GlycanSource::LocalBundle("fixture".into()),
        anomer: Anomer::Beta,
        format: "PDB".into(),
        level: "2".into(),
    };
    let ensemble = ensemble_from_pdb(
        include_str!("../../../tests/fixtures/glycan.pdb"),
        None,
        query,
        "fixture",
    )
    .unwrap();
    let sites = vec![SearchSite {
        site: GlycosylationSite::new("A", 1),
        ensemble,
    }];
    let config = SearchConfig {
        scoring_mode: reglyco_core::SearchScoringMode::FullEnergy,
        pre_minimization: true,
        pre_minimization_iterations: 2,
        ensemble_mode: Some("conformer_collection".into()),
        population_size: 4,
        generations: 1,
        mh_burn_in_sweeps: 0,
        mh_thinning_accepted: 1,
        ..Default::default()
    };
    let (frames, _) =
        reglyco_ensemble::sample_attached_ensemble(&protein, &sites, 2, &config, &builder).unwrap();
    let system = reglyco_ensemble::prepare_frame_topology(&protein, &sites, &builder).unwrap();
    for frame in frames {
        let coordinates = glysys_energy::geometry::CoordinateMap::new(&system)
            .coordinates(&frame.structure)
            .unwrap();
        let energy = glysys_energy::EnergyEvaluator::new(
            &system,
            glysys_energy::EnergyOptions {
                cutoff: Some(config.energy_cutoff),
                obc2: None,
                ..Default::default()
            },
        )
        .unwrap()
        .energy(&coordinates)
        .unwrap()
        .total();
        assert!((energy - frame.selected_energy_kcal_per_mol.unwrap()).abs() < 1e-8);
        assert_eq!(frame.structure.atoms().len(), system.atom_count());
    }
}
#[test]
#[ignore = "requires Vulkan; software is correctness evidence only"]
fn sampled_energy_objectives_dispatch_gpu_and_export_cpu_energies() {
    pollster::block_on(async {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..Default::default()
        };
        let protein = read_pdb_str(
            include_str!("../../../tests/fixtures/protein.pdb"),
            &options,
        )
        .unwrap();
        let builder = SystemBuilder::new(options).unwrap();
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle("fixture".into()),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let ensemble = ensemble_from_pdb(
            include_str!("../../../tests/fixtures/glycan.pdb"),
            None,
            query,
            "fixture",
        )
        .unwrap();
        let sites = vec![SearchSite {
            site: GlycosylationSite::new("A", 1),
            ensemble,
        }];
        for mode in [
            reglyco_core::SearchScoringMode::FullEnergy,
            reglyco_core::SearchScoringMode::ProteinGlycanInteraction,
        ] {
            let config = SearchConfig {
                scoring_mode: mode,
                mh_chains: 3,
                burn_in_steps: Some(3),
                thinning_steps: Some(2),
                ..Default::default()
            };
            gpu::configure("webgpu");
            let (frames, d) = sample_attached_ensemble_with_cancel_async(
                &protein,
                &sites,
                6,
                &config,
                &builder,
                || false,
            )
            .await
            .unwrap();
            let report = gpu::finish();
            let name = if mode == reglyco_core::SearchScoringMode::FullEnergy {
                "energy_score"
            } else {
                "interaction_score"
            };
            assert!(
                report["stages"][name]["gpuEvaluations"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0,
                "{report}"
            );
            assert_eq!(frames.len(), 6);
            assert_eq!(d.attempts, 21);
            let mut checked = frames.clone();
            reglyco_ensemble::refresh_frames(&mut checked, &protein, &sites, &config, &builder)
                .unwrap();
            for (frame, reference) in frames.iter().zip(checked) {
                assert!(
                    (frame.selected_energy_kcal_per_mol.unwrap()
                        - reference.selected_energy_kcal_per_mol.unwrap())
                    .abs()
                        < 1e-8
                );
            }
        }
    });
}
