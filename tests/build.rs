use glysys::{BuildOptions, ResidueId, SystemBuilder, read_pdb_str};
use reglyco::{
    AttachmentRequest, BuildRequest, GlycanConformer, GlycosylationSite, ReGlycoError, build,
};

const GLYCAN: &str = include_str!("fixtures/glycan.pdb");
const PROTEIN: &str = include_str!("fixtures/protein.pdb");
const ARA_GLYCAN: &str = "\
ATOM      1  O1  ROH B   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  C1  ARA B   2       1.430   0.000   0.000  1.00  0.00           C
ATOM      3  C2  ARA B   2       2.100   1.200   0.000  1.00  0.00           C
ATOM      4  C3  ARA B   2       3.400   1.000   0.000  1.00  0.00           C
ATOM      5  C4  ARA B   2       3.600  -0.400   0.000  1.00  0.00           C
ATOM      6  O4  ARA B   2       2.300  -0.900   0.000  1.00  0.00           O
ATOM      7  C5  ARA B   2       4.700  -1.100   0.000  1.00  0.00           C
ATOM      8  O5  ARA B   2       5.700  -0.300   0.000  1.00  0.00           O
END
";

fn options() -> BuildOptions {
    BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    }
}

fn site(chain: &str, number: i32) -> GlycosylationSite {
    GlycosylationSite::new(chain, number)
}

fn attachment(chain: &str, number: i32) -> AttachmentRequest {
    AttachmentRequest {
        site: site(chain, number),
        conformer: GlycanConformer::new(read_pdb_str(GLYCAN, &options()).unwrap()),
    }
}

#[test]
fn builds_deterministic_n_linked_structure_in_memory() {
    let builder = SystemBuilder::new(options()).unwrap();
    let request = || BuildRequest {
        protein: read_pdb_str(PROTEIN, &options()).unwrap(),
        attachments: vec![attachment("A", 1)],
        parameterize: false,
    };
    let first = build(request(), &builder).unwrap();
    let second = build(request(), &builder).unwrap();
    assert_eq!(
        first.structure.to_pdb_string(),
        second.structure.to_pdb_string()
    );
    assert_eq!(first.structure.metadata().glycosylation_sites.len(), 1);
    assert_eq!(first.structure.metadata().glycan_trees.len(), 1);
    assert!(
        !first
            .structure
            .residues()
            .iter()
            .any(|residue| residue.name == "ROH")
    );

    let site = ResidueId {
        chain: "A".into(),
        number: 1,
        insertion_code: None,
    };
    let nd2 = first.structure.find_atom(&site, "ND2").unwrap();
    let glycan = &first.structure.metadata().glycosylation_sites[0];
    let c1 = first
        .structure
        .find_atom(&glycan.glycan_residue, "C1")
        .unwrap();
    let nd2 = first.structure.atom(nd2).unwrap().position;
    let c1 = first.structure.atom(c1).unwrap().position;
    let distance =
        ((nd2.x - c1.x).powi(2) + (nd2.y - c1.y).powi(2) + (nd2.z - c1.z).powi(2)).sqrt();
    assert!((distance - 1.45).abs() < 1.0e-6);
}

#[test]
fn builds_and_parameterizes_multiple_n_and_o_linked_glycans() {
    let builder = SystemBuilder::new(options()).unwrap();
    let result = build(
        BuildRequest {
            protein: read_pdb_str(PROTEIN, &options()).unwrap(),
            attachments: vec![attachment("A", 1), attachment("B", 2)],
            parameterize: true,
        },
        &builder,
    )
    .unwrap();
    let system = result.system.unwrap();
    assert_eq!(system.metadata().glycosylation_sites.len(), 2);
    assert_eq!(system.metadata().glycan_trees.len(), 2);
    assert!(system.atom_count() > 50);
}

#[test]
fn rejects_invalid_and_occupied_sites() {
    let builder = SystemBuilder::new(options()).unwrap();
    let mut invalid = read_pdb_str(PROTEIN, &options()).unwrap();
    invalid
        .rename_residue(
            &ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            },
            "GLY",
        )
        .unwrap();
    let error = build(
        BuildRequest {
            protein: invalid,
            attachments: vec![attachment("A", 1)],
            parameterize: false,
        },
        &builder,
    )
    .unwrap_err();
    assert!(matches!(error, ReGlycoError::UnsupportedSite { .. }));

    let built = build(
        BuildRequest {
            protein: read_pdb_str(PROTEIN, &options()).unwrap(),
            attachments: vec![attachment("A", 1)],
            parameterize: false,
        },
        &builder,
    )
    .unwrap();
    let error = build(
        BuildRequest {
            protein: built.structure,
            attachments: vec![attachment("A", 1)],
            parameterize: false,
        },
        &builder,
    )
    .unwrap_err();
    assert!(matches!(error, ReGlycoError::OccupiedSite(_)));
}

#[test]
fn rejects_duplicate_sites_in_one_request() {
    let builder = SystemBuilder::new(options()).unwrap();
    let error = build(
        BuildRequest {
            protein: read_pdb_str(PROTEIN, &options()).unwrap(),
            attachments: vec![attachment("A", 1), attachment("A", 1)],
            parameterize: false,
        },
        &builder,
    )
    .unwrap_err();
    assert!(matches!(error, ReGlycoError::OccupiedSite(_)));
}

#[test]
fn converts_proline_and_attaches_arabinose_using_the_o4_anchor() {
    let protein = "\
ATOM      1  N   PRO A   7       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  PRO A   7       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  CB  PRO A   7       1.900   1.400   0.000  1.00 20.00           C
ATOM      4  CG  PRO A   7       0.800   2.200   0.500  1.00 20.00           C
ATOM      5  CD  PRO A   7      -0.200   1.100   0.200  1.00 20.00           C
END
";
    let options = options();
    let builder = SystemBuilder::new(options.clone()).unwrap();
    let result = build(
        BuildRequest {
            protein: read_pdb_str(protein, &options).unwrap(),
            attachments: vec![AttachmentRequest {
                site: site("A", 7),
                conformer: GlycanConformer::new(read_pdb_str(ARA_GLYCAN, &options).unwrap()),
            }],
            parameterize: false,
        },
        &builder,
    )
    .unwrap();
    let hyp_site = ResidueId {
        chain: "A".into(),
        number: 7,
        insertion_code: None,
    };
    let residue = result
        .structure
        .residues()
        .into_iter()
        .find(|residue| residue.id == hyp_site)
        .unwrap();
    assert_eq!(residue.name, "HYP");
    let od1 = result.structure.find_atom(&hyp_site, "OD1").unwrap();
    let attachment = &result.structure.metadata().glycosylation_sites[0];
    assert_eq!(attachment.protein_atom, "OD1");
    let c1 = result
        .structure
        .find_atom(&attachment.glycan_residue, "C1")
        .unwrap();
    let od1_position = result.structure.atom(od1).unwrap().position;
    let c1_position = result.structure.atom(c1).unwrap().position;
    let distance = ((od1_position.x - c1_position.x).powi(2)
        + (od1_position.y - c1_position.y).powi(2)
        + (od1_position.z - c1_position.z).powi(2))
    .sqrt();
    assert!((distance - 1.43).abs() < 1.0e-6);
}

#[test]
fn rejects_arabinose_without_the_hyp_linkage_anchor() {
    let malformed = ARA_GLYCAN
        .lines()
        .filter(|line| !line.contains(" O4 "))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let options = options();
    let builder = SystemBuilder::new(options.clone()).unwrap();
    let protein = "\
ATOM      1  N   PRO A   7       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  PRO A   7       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  CB  PRO A   7       1.900   1.400   0.000  1.00 20.00           C
ATOM      4  CG  PRO A   7       0.800   2.200   0.500  1.00 20.00           C
ATOM      5  CD  PRO A   7      -0.200   1.100   0.200  1.00 20.00           C
END
";
    let error = build(
        BuildRequest {
            protein: read_pdb_str(protein, &options).unwrap(),
            attachments: vec![AttachmentRequest {
                site: site("A", 7),
                conformer: GlycanConformer::new(read_pdb_str(&malformed, &options).unwrap()),
            }],
            parameterize: false,
        },
        &builder,
    )
    .unwrap_err();
    assert!(matches!(error, ReGlycoError::MissingGlycanAtom(atom) if atom == "O4"));
}
