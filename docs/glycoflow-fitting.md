# GlycoFlow density fitting

`reglyco refine --objective density --density-search glycoflow` fits the glycan at each target
site with the GlycoFlow flow model (crate `reglyco-glycoflow`, which depends on
`../GlycoFlow/rust/glycoflow-core`).

```console
export GLYCOFLOW_MODEL=/path/to/model   # glycoflow.safetensors, glycoflow.json, residue_library.json
reglyco refine --protein model.pdb --density-map map.ccp4 \
  --objective density --density-search glycoflow --glycoflow-device cuda \
  --replace-glycan A:79 --seed 0 -o out/
```

The model directory is produced by `scripts/export_rust_fixtures.py` in GlycoFlow (weights) plus
`glycoflow/resources/residue_library.json`. Build with `--features reglyco-cli/glycoflow-cuda`
(`CUDA_COMPUTE_CAP=121` on a GB10) for the GPU path; the CPU path needs no feature.

What it does: observation-guided GlycoFlow generation (768 samples), refinement of 24 distinct
basins on one candidate-independent site likelihood (`reglyco-density::site_likelihood`) plus
contact, amide and GlycoFlow-prior terms; a final contact-validity stage; a calibrated support
test that marks residues without density, which are then regenerated from the GlycoFlow prior and
labelled as prior-driven.

Outputs: `fitted.pdb` (best fit; deposited waters overlapping the glycan removed),
`candidates.pdb` (best fit, density-ambiguous alternatives, prior completions; REMARK 250 labels),
`glycoflow-fit.json` (objective terms, support per subtree, alternatives, costs, recovery against
the deposited glycan when present - evaluation only) and `validation.json`.

Options: `--glycoflow-samples`, `--glycoflow-basins`, `--glycoflow-prior-weight`,
`--glycoflow-no-prior`, `--glycoflow-clash-weight` (final contact weight, default 100),
`--glycoflow-symmetry`, `--glycoflow-precision`.

Benchmarks (13 sites, 3 X-ray and 10 cryo-EM, against the adaptive fitter) and the method
comparison are in GlycoFlow's `docs/fitting_results.md`.
