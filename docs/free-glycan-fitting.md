# Fitting free and lectin-bound glycans (design)

Status: proposal, 2026-10-06. Nothing here is implemented yet.

ReGlyco's GlycoFlow fit (`docs/glycoflow-fitting.md`) places a glycan that is covalently attached to
a protein residue. This note covers glycans with no covalent anchor: a glycan bound in a lectin's
site, a glycan in a carbohydrate-binding module or antibody, or a glycan crystallised on its own. It
lists what has to change, where, and in what order.

## Short answer

It is feasible. **GlycoFlow needs no change.** The model is a free-glycan model (trained on glycans
with an `-OH` reducing end, and no protein token in its vocabulary). Today the protein-linked case
reuses the reducing-end O1 as a stand-in for Asn ND2 or Ser/Thr OG.

What is hard-wired is ReGlyco's notion of a site:
- the Asn/Ser/Thr anchor;
- a pose with two attachment angles;
- a scoring region centred on the link atom;
- a few anchor-only energy terms.

A free glycan needs a rigid-body pose (6 degrees of freedom) in place of the two attachment angles,
a scoring region centred on the ligand, and those anchor terms switched off. The rest carries over
unchanged: the density likelihood, the clash grids, the prior, guided sampling, the Cartesian
refinement and support.

## What is anchored today

Code paths are in `crates/reglyco-glycoflow/src/` unless stated otherwise.

| Area | Today | Code |
|---|---|---|
| Site | Anchor atoms CB–CG–ND2 (Asn) or CA–CB–OG/OG1 (Ser/Thr); any other residue is rejected. The deposited glycan is the sugar tree whose root C1 lies within 1.75 Å of the link atom. | `site.rs`: `site_atoms`, `anchor_of`, `sugar_tree`, `deposited_glycan`, `load_site` |
| Pose | `Pose { tau, psi, phi, template }`. psi is CB–CG–ND2–C1 and phi is CG–ND2–C1–O5, both rigid rotations of the whole glycan. `place()` = `attach(local(tau))`, which puts C1 on the link with a fixed bond and angle. | `problem.rs`: `Pose`, `attach`, `place`, `attachment_angles`, `evaluate_placed` |
| Search | Every conformer gets an attachment grid: psi ∈ {180°, ±165°} × 36 phi values. Guidance re-runs it every 4 steps, and refinement optimises `[tau, psi, phi]`. | `search.rs`: `attachment_grid`, `attach_search`, `ObjectiveGuidance`, `refine`, `method_b` |
| Scoring region | The likelihood ball, the clash-grid box, sigma calibration (protein shell 4–12 Å), the decoy null, the density levels for the 0.3 gate, and the environment radius (45 Å) are all centred on the link atom. The radius is the bond-path span from O1. | `pipeline.rs`: `density_problem`, `calibrate_sigma`; `problem.rs`: `max_span`, `null_inflation`; `infer.rs`: `DensityLevels` |
| Energy | `e_att` (Asn amide, psi near 180°). Explicit site-residue pairs, with a bond-count exclusion next to the link. O1 is not scored (`keep[0] = false`). | `problem.rs`: `placed_terms`, `SiteProblem::new` |
| Cartesian | The attachment restraint `|B − C1|`, psi measured from coordinates, and O1 frozen in place next to the link. | `cartesian.rs`: `Restraints::attachment`, `coordinate_objective`, `free = keep` |
| Deposit checks | ψN is Asn-only. The anomer check uses the link atom as the root's reference. | `deposition.rs`: `check`, `deposition_checks` |
| Inference | Four N-glycan candidates. Support is hierarchical from the reducing end, and `tree_torsions` adds 2 for the attachment angles. | `infer.rs`: `N_GLYCAN_CANDIDATES`, `fit_candidate`, `gate`; `support.rs`: `subtree_support` |
| Output | A `LINK` record to the anchor; O1 is never written. | `output.rs`: `glycan_records`, `without_overlapping_waters` |

## Changes

### 1. A ligand site

- **Site kind.** Add `SiteKind::{Linked(ResidueId), Ligand(LigandSite)}`. A `LigandSite` holds:
  - the deposited sugar residues, or nothing for de novo placement;
  - a centre: the deposited ligand's centroid, or a point the user picks.
- **Finding ligands.** Use `sugar_tree` without the anchor test: every tree of `CCD_TO_GLYCAM`
  residues whose root C1 is not within 1.75 Å of a protein atom. The root is the residue whose
  anomeric carbon has no sugar parent. The tree's own O1 or OMe stays as part of the ligand.
- **Region.** Centre the region on the ligand, and use radius = half the template's largest
  intramolecular distance + 2.5 Å, in place of the bond-path span from O1. Sigma calibration, the
  decoys and the density levels all use the shell around the ligand atoms instead of the link atom.

### 2. A rigid-body pose

- **Placement.** `Pose { tau, place: Placement, template }` with:
  - `Placement::Linked { psi, phi }`: today's pose, unchanged;
  - `Placement::Free { rot: [f64; 4], trans: [f64; 3] }`: a unit quaternion and a translation of
    the template's centroid.
- **Gradients.** The rigid part comes from the atom gradients, which `evaluate_placed` already
  sums for psi and phi: force Σg, and torque Σ(r − c)×g about the centroid. Optimise the rotation
  as a small axis-angle step that is re-normalised into the quaternion.
- **O1.** O1 becomes a real atom (`keep[0] = true`): the reducing-end hydroxyl is in the crystal.
  - Methyl glycosides (`-OME`) are common lectin ligands, for example ConA's trimannoside.
  - The residue library has no `OME` template, so `-OME` sequences fail today
    ("no template for aglycone OME") even though the CLI accepts them. Add that template.

### 3. Search

- **Replace the attachment grid with a rigid-body search per conformer.**
  - Start from a uniform SO(3) grid, about 500 orientations (roughly 15° apart), at the centre,
    scored on density and clashes only.
  - Take the best 10 orientations through a local 6-DOF descent that includes translation, then
    score them with the full objective.
  - This is about 10× the work of today's 108-point grid. It is still cheap next to the network.
- **Guidance.** `ObjectiveGuidance` runs the rigid search every few steps on the predicted
  endpoint (as it does for attachment now), with normalised rotation (rad) and translation (Å)
  steps in between.
- **Refinement.** `refine` and `polish` optimise `[tau, rotation, translation]`.
- **Starting from a deposit.** When a deposit exists (critique mode), also start from a Kabsch
  superposition onto the deposited atoms. That gives a direct "is the deposit the best explanation
  of the map" comparison, as today.

### 4. Objective and Cartesian refinement

- **Density.** Unchanged in form; only the centre changes.
- **Contacts.** Keep the clash grids. Drop the explicit site-residue pairs and the link
  bond-count exclusion: there is no covalent neighbour.
- **`e_att`.** Off.
- **Prior.** Unchanged. This is GlycoFlow's own domain, and the prior percentiles become more
  interesting: a lectin can hold a glycan in a strained, off-prior conformation, and the report
  will show it.
- **Anchoring risk.** Without the covalent bond, a ligand in weak density at low resolution can
  drift. Offer an optional weak positional restraint to the starting placement, off by default.
  Report how far the fit moved.
- **Cartesian refinement.** Drop the attachment restraint and the frozen O1. Every ligand atom is
  free, with the same bond, 1-3 and chirality restraints.

### 5. Deposit checks

- **Anomers.** Check every residue against its own reference: the root against its O1 or OMe,
  the others against the parent oxygen, as now.
- **Reducing end.** A free reducing end mutarotates. Fit both α and β, and report which one the
  map prefers and whether the CCD code (e.g. GLC vs BGC) agrees.
- **New checks:**
  - ring pucker against GlycoFlow's pucker populations (the library has them), flagging rare
    puckers such as boats;
  - occupancy and B-factors against map support;
  - clashes with the protein.
- **ψN.** Not applicable.

### 6. Inference

- **Candidates.** The N-glycan candidates do not apply. In most lectin structures the ligand's
  sequence is known, so the main mode is "fit this sequence".
- **Optional library.** For unknown ligands, add a small library of common epitopes: LacNAc,
  Lewis X/A, sialyl-Lac α2-3 and α2-6, blood group H/A/B, and Man3/Man5 fragments. Fit each and
  prune with the 0.3 gate.
- **Pruning (a real algorithmic change).** Support is pruned from the reducing end today. A bound
  glycan is usually best ordered at the epitope in the binding site, with the reducing end
  disordered. Pruning must keep the best-supported connected subtree, which need not contain the
  root (`subtree_support`, `gate`, `pruned_sequence`). `tree_torsions` loses the +2 for
  attachment and counts the rigid body (6) instead.

### 7. Output and the browser page

- **Output.** No `LINK` record. Write O1 or OMe. Keep the deposited chain and numbering for a
  refit; for a new ligand, use a new chain (or the nearest protein chain) with HETATM numbering.
  The corrected-model writer in CE already replaces residues by key.
- **CE (`/reglyco/fit`):**
  - list bound glycans next to sequons and glycosylated sites;
  - a "pick a point" mode in the viewer for de novo placement;
  - a site spec in the wasm request: `{ "ligand": [residue keys] }` or `{ "center": [x, y, z] }`;
  - verdict wording for ligands.
  - The workspace run already stores per-site results, so ligands fit in as more entries.

## Order of work

1. **Critique deposited ligands (known sequence).** Covers the ligand site, the rigid pose,
   search, objective and Cartesian changes, deposit checks and the `OME` template. This is the bulk
   of the engine work. The CE side is small: listing ligands and the request.
2. **De novo placement** from a picked point or a difference-density blob, with both anomers.
3. **Inference** from an epitope library, with subtree pruning from the epitope.

**Validation.** Use about 10 lectin–glycan complexes across resolutions, chosen from the PDB. For
example: DC-SIGN–Man4 (1K9I), concanavalin A–methyl trimannoside (1CVN; it needs `OME`), a
galectin with LacNAc, and haemagglutinin with sialyl-LacNAc analogues. Compare against the deposit
exactly as for N-glycans.

## O-linked and other sites

The Asn-specific parts that used to apply to Ser/Thr are fixed:
- the amide term;
- psi near 180 deg;
- the Asn bond count to CB;
- the 123 deg link angle.

Sites now come from one anchor table (`anchor.rs`), which covers Ser/Thr, Trp C-mannose, Tyr, Hyp, Hyl and Cys. See `docs/glycoflow-fitting.md`.
