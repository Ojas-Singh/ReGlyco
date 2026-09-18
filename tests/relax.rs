use glysys::{BuildOptions, SystemBuilder, read_pdb_str};
use reglyco::{
    AttachmentRequest, BuildRequest, GlycanConformer, GlycosylationSite, build,
    relax::{RelaxOptions, RelaxProgress, relax_with_progress},
};

const GLYCAN: &str = include_str!("fixtures/glycan.pdb");
const PROTEIN: &str = include_str!("fixtures/protein.pdb");

#[test]
fn relaxation_keeps_protein_fixed_and_does_not_increase_energy() {
    let options = BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    };
    let builder = SystemBuilder::new(options.clone()).unwrap();
    let product = build(
        BuildRequest {
            protein: read_pdb_str(PROTEIN, &options).unwrap(),
            attachments: vec![AttachmentRequest {
                site: GlycosylationSite::new("A", 1),
                conformer: GlycanConformer::new(read_pdb_str(GLYCAN, &options).unwrap()),
            }],
            parameterize: true,
        },
        &builder,
    )
    .unwrap();
    let before = product.structure.clone();
    let mut relax_options = RelaxOptions {
        include_local_sidechains: false,
        ..RelaxOptions::default()
    };
    relax_options.lbfgs.max_iterations = 1;
    relax_options.lbfgs.initial_step = 1.0e-4;
    let mut saw_stage_start = false;
    let mut saw_iteration = false;
    let result = relax_with_progress(
        &product.structure,
        product.system.as_ref().unwrap(),
        &relax_options,
        |event| match event {
            RelaxProgress::StageStarted { .. } => saw_stage_start = true,
            RelaxProgress::Iteration { .. } => saw_iteration = true,
            _ => {}
        },
    )
    .unwrap();
    assert!(saw_stage_start);
    assert!(saw_iteration);
    assert!(
        result.diagnostics.final_energy.total()
            <= result.diagnostics.initial_energy.total() + 1.0e-6
    );
    for residue in before
        .residues()
        .iter()
        .filter(|residue| residue.id.chain == "A" || residue.id.chain == "B")
    {
        for atom in &residue.atoms {
            assert_eq!(
                before.atom(*atom).unwrap().position,
                result.structure.atom(*atom).unwrap().position
            );
        }
    }
}

#[test]
fn vacuum_is_the_default_relaxation_model() {
    assert!(RelaxOptions::default().obc2.is_none());
}
