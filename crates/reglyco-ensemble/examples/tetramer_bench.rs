use glysys::{BuildOptions, SystemBuilder, read_pdb_str};
use reglyco_core::{
    Anomer, GlycanQuery, GlycanSource, GlycosylationSite, SearchConfig, SearchSelectionPolicy,
    SearchSite,
};
use reglyco_ensemble::{ensemble_from_pdb, linkage_priors, search_with_progress};
use std::{fs, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args().nth(1).expect("fixture directory"));
    let started = Instant::now();
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..Default::default()
    };
    let protein = read_pdb_str(
        &fs::read_to_string(root.join("Deglycosylated_tetramer.pdb"))?,
        &options,
    )?;
    let query = GlycanQuery {
        source: GlycanSource::LocalBundle(root.join("GS00178/output/level_2/PDB/beta.pdb")),
        anomer: Anomer::Beta,
        format: "PDB".into(),
        level: "2".into(),
    };
    let metadata = serde_json::from_str(&fs::read_to_string(root.join("GS00178/data.json"))?)?;
    let mut ensemble = ensemble_from_pdb(
        &fs::read_to_string(root.join("GS00178/output/level_2/PDB/beta.pdb"))?,
        Some(&metadata),
        query,
        "tetramer-local",
    )?;
    for conformer in &mut ensemble.conformers {
        if conformer.priors.phi.is_empty() || conformer.priors.psi.is_empty() {
            conformer.priors = linkage_priors("ASN");
        }
    }
    let mut sites = Vec::new();
    for chain in ["A", "B", "C", "D", "E", "F", "G", "H"] {
        let numbers: &[i32] = if ["A", "C", "E", "G"].contains(&chain) {
            &[138, 350, 402]
        } else {
            &[930]
        };
        for &number in numbers {
            sites.push(SearchSite {
                site: GlycosylationSite::new(chain, number),
                ensemble: ensemble.clone(),
            });
        }
    }
    let config = SearchConfig {
        population_size: 32,
        generations: 25,
        seed: 42,
        selection_policy: SearchSelectionPolicy::JointPriorV1,
        ..Default::default()
    };
    let builder = SystemBuilder::new(options)?;
    eprintln!("Prepared input in {:?}", started.elapsed());
    let result = search_with_progress(&protein, &sites, &config, &builder, |event| {
        eprintln!("{:?}: {:?}", started.elapsed(), event)
    })?;
    eprintln!(
        "Finished {:?}: {:?}, {}, {} sites",
        started.elapsed(),
        result.clash_status,
        result.termination_reason,
        result.sites.len()
    );
    Ok(())
}
