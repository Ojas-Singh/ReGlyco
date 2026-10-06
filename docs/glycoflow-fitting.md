# GlycoFlow density fitting

`reglyco refine --objective density` fits the glycan at each `--replace-glycan` site with the
GlycoFlow flow model (crate `reglyco-glycoflow` on the open GlycoFlow engine
`crates/glycoflow-core`). It is ReGlyco's only density fitter; the earlier
GlycoShape-ensemble fitter (`--density-effort`, `--density-deep-strategy`, ...) was removed after
tag `pre-glycoflow-density`, and `--density-search` is accepted only as `glycoflow`.

```console
export GLYCOFLOW_MODEL=/path/to/model   # glycoflow.safetensors, glycoflow.json, residue_library.json
reglyco refine --protein model.pdb --density-map map.ccp4 \
  --objective density --glycoflow-device cuda \
  --replace-glycan A:79 --seed 0 -o out/
```

The model licence and the model are checked before anything else is loaded (below).
`--replace-glycan SITE` fits the deposited glycan's sequence;
`--replace-glycan SITE=<GLYCAM sequence>` fits another one. `--density-map auto` resolves a sidecar
map or, with `--pdb-id`, downloads the PDBe EDS (or, with `--density-map-source rcsb`, the RCSB
2Fo-Fc) map; `--density-sigma` and `--density-resolution` override the calibrated atom width and
the REMARK 2 resolution.

The trained GlycoFlow model is **not** part of this repository and is not covered by its
licence. A model directory holds `glycoflow.safetensors` (weights), `glycoflow.json` (network
configuration) and `residue_library.json` (residue templates and pucker states); it is licensed
under the GlycoFlow Source-Available Non-Commercial License (non-commercial academic research;
commercial use needs a separate written licence) and distributed through the gated Hugging Face
repository `Ojas-Singh/glycoflow`, where the GlycoFlow authors approve access requests. ReGlyco
uses it only after the licence is accepted (`--accept-glycoflow-license`, recorded once in the
model cache `~/.cache/reglyco/glycoflow` or `$GLYCOFLOW_CACHE`; `GLYCOFLOW_ACCEPT_LICENSE=1` for one
run). It then loads `--glycoflow-model` / `$GLYCOFLOW_MODEL`, or the cached download, or downloads
the model with the user's Hugging Face token (`$HF_TOKEN` or `hf auth login`) at a pinned revision
with SHA-256 checks (`crates/reglyco-cli/src/glycoflow_model.rs`). Build with `--features reglyco-cli/glycoflow-cuda`
(`CUDA_COMPUTE_CAP=121` on a GB10) for the GPU path; the CPU path needs no feature.

What it does: observation-guided GlycoFlow generation (384 samples, 8 Heun steps), refinement of 24 distinct
basins on one candidate-independent site likelihood (`reglyco-density::site_likelihood`) plus
contact, amide and GlycoFlow-prior terms; a final contact-validity stage; restrained Cartesian
refinement of every basin (atoms move freely with the template's bonds, 1-3 distances and chiral
volumes as restraints; basins are ranked afterwards, with the same contact escalation); a calibrated support
test that marks residues without density, which are then regenerated from the GlycoFlow prior and
labelled as prior-driven.

Sites: any residue in `reglyco_glycoflow::anchor::ANCHORS`.

| Residue | Link | Notes |
|---|---|---|
| Asn | ND2 (N-glycans) | Amide trans (psi_N near 180 deg), as before. |
| Ser, Thr | OG / OG1 (O-glycans) | Mucin GalNAc, O-Man, O-Fuc, O-Glc, O-GlcNAc. Bond 1.42 A, angle 114 deg; psi searched over the whole circle with no energy term. |
| Trp | CD1 (C-mannose) | C1 kept in the indole plane, cis to CB. Half the templates carry the mannose in 1C4, the chair seen in crystals, because GlycoFlow's library only has 4C1 for alpha-Man (`ring.rs`). |
| Tyr | OH | Glycogenin. |
| Hyp | OD1 | |
| Hyl (LYZ) | OH | |
| Cys | SG (S-glycans) | |

Geometry comes from the deposits of X-ray entries at 2.3 A or better. Inference (`infer::candidates_for`) uses each residue's common glycans:
- the four N-glycans on Asn;
- eleven O-glycans on Ser/Thr: sialylated mucin core 2, 2,6-sialyl T, O-Man core M1, fungal O-Man, extended O-Fuc, xylosylated O-Glc and O-GlcNAc, plus the root sugars GalNAc, Man, Fuc and Glc alone, since a large candidate whose root does not settle is never pruned down to that root;
- C-mannose on Trp;
- the glycogenin glucan on Tyr, and so on for the other residues.

Validation (Quick preset, deposited sequence), all with the verdict "agrees":

| Site | Glycan | Core RMSD |
|---|---|---|
| 7R84 A:7, A:10 | C-Man (1C4) | 0.13, 0.25 A |
| 7R84 A:16 | O-Fuc | 0.22 A |
| 6R2W L:52 | Xyl-Xyl-O-Glc | 0.33 A |
| 6R2W L:60 | O-Fuc | 0.16 A |
| 5T5L a:102 | GalNAc | 0.35 A |
| 3U2U A:195 | Tyr glucan | 0.47 A |
| 3M5Q A:336 | O-Man | 0.72 A |

Outputs: `fitted.pdb` (best fit; deposited waters overlapping the glycan removed),
`candidates.pdb` (best fit, density-ambiguous alternatives, prior completions; REMARK 250 labels),
`glycoflow-fit.json` (objective terms, support per subtree, alternatives, costs, recovery against
the deposited glycan when present - evaluation only) and `validation.json`.

Report bundles: the GlycoFlow fit does not write `report.json`/`report.pdf`; `glycoflow-fit.json`
is its report.

Options: `--glycoflow-samples` (default 384), `--glycoflow-steps` (default 8; 768 x 32 gave the
same core/supported recovery on 6 sites x 3 seeds at 8x the network cost), `--glycoflow-basins`, `--glycoflow-prior-weight`,
`--glycoflow-no-prior`, `--glycoflow-clash-weight` (final contact weight, default 100),
`--glycoflow-symmetry`, `--glycoflow-precision`.

Benchmarks (13 sites, 3 X-ray and 10 cryo-EM, against the adaptive fitter) and the method
comparison are in GlycoFlow's `docs/fitting_results.md`.

Workflow API (`reglyco-workflow`, `density` workflow of the `full` profile): the request's input
assets must contain `density.map` and the model files `glycoflow.safetensors`, `glycoflow.json` and
`residue_library.json` (execution-only, never exported as artifacts). Every selected assignment is
fitted; a GLYCAM `glycanId` (ending in `-OH`) sets the fitted sequence, otherwise the deposited
glycan's sequence is used. `options.densityResolution` overrides the REMARK 2 resolution. The
result carries `fitted.pdb` as the primary structure plus `candidates.pdb` and
`glycoflow-fit.json` artifacts. The browser (wasm32) build does not offer density fitting yet:
its `full` capabilities omit `density` and a density request returns an error.

Benchmark harness: `scripts/benchmark-glycoflow.sh [OUTPUT_DIR] [SITES_FILE]` (needs
`GLYCOFLOW_MODEL`; `GLYCOFLOW_DEVICE=cuda` with a CUDA build) writes per-run logs, wall time, peak
memory and a `summary.tsv` with recovery (evaluation only) and validation.
