# reglyco-glycoflow

Fits a glycan of known sequence to a protein site in a density map with the frozen GlycoFlow
flow model. Rust port of the Python reference `glycoflow/fitting/` (GlycoFlow repository);
used by `reglyco refine --objective density`.

## Engine (open) and model (licensed)

The GlycoFlow inference engine (`crates/glycoflow-core`: GLYCAM builder, torsion kinematics,
network, ODE sampler with a guidance hook) is part of this workspace and open source under
ReGlyco's licence.

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
with SHA-256 checks (`crates/reglyco-cli/src/glycoflow_model.rs`).

Tests that need the model run with `--ignored`: `tests/synthetic.rs` (residue library only,
`GLYCOFLOW_MODEL`), `tests/parity.rs` (`GLYCOFLOW_MODEL`, `GLYCOFLOW_FIT_FIXTURES` from GlycoFlow's
`scripts/fitting/export_fit_fixtures.py`), and `crates/glycoflow-core/tests`
(`GLYCOFLOW_RUST_DIR`: a GlycoFlow checkout's `rust/` with fixtures and weights).

`glycoflow-core` brings candle; its `gemm` crates are optimised in dev/test builds too (root
`Cargo.toml`), because their fp16 NEON assembly does not assemble at opt-level 0 on aarch64.

## Method

One objective, used for search and for selection (`problem.rs`):

```
E = -loglik_density + w_env E_env + w_self E_self + E_attach(psi_N) + w_prior E_prior(tau)
```

* `loglik_density`: profiled least-squares site likelihood (`reglyco_density::site_likelihood`),
  in log-likelihood units through the independent-sample volume measured from the residual
  autocorrelation on the protein shell (`noise_correlation_volume`);
* `E_env`: carbon/polar-probe penalty grids over protein, ligands, other glycans and crystal
  symmetry mates (`symmetry.rs`, all settings of the 65 Sohncke space groups generated with
  gemmi), plus explicit pairs with the site residue beyond three bonds;
* `E_self`: GlycoFlow's contact energy over glycan atom pairs >= 4 bonds apart;
* `E_attach = kappa (1 + cos psi_N)`, kappa = 1/(10 deg)^2;
* `E_prior`: von Mises KDE of each torsion over 512 GlycoFlow samples of the glycan (product of
  marginals; an approximation that ignores torsion couplings).

Gradients are analytic (`dE/dx` of every term, `dE/dtau` through the torsion Jacobian,
`dE/dpsi_N` / `dE/dphi_N` as rigid rotations about CG->ND2 / ND2->C1).

Pipeline (`pipeline.rs`, = `pipeline.fit_site` + `search.method_b`): 384 observation-guided
GlycoFlow samples (Heun 8 steps; from t >= 0.3 the velocity gets `-g/rms(g)`, g = dE/dtau at the
predicted endpoint, through `glycoflow_core::sampler::Guidance`; attachment angles grid-searched
every 4 steps; batches of 256) -> attachment grid search -> 24 distinct basins (1.5 A) -> Adam refinement (150
steps) -> polish of the best 4 (300 steps) -> restrained Cartesian refinement of every basin
(`cartesian.rs`: bonds 0.02 A, 1-3 distances 0.04 A, chiral volumes 0.2 A^3; 300 Adam steps; ranking on
objective + restraints, contact weight escalated x10 while the best pose violates the floors) -> subtree support test (gain > 5 + 0.5 per torsion) -> prior completion of
unsupported subtrees (pinned-torsion GlycoFlow inpainting with clash guidance, labelled as
prior-driven) -> density-ambiguous alternatives within 5 objective units.

`observation.rs` defines the `Observation` trait (prepare per glycan; evaluate energy and `dE/dx`
per conformer, batched). The density map is its first implementation; SAXS can implement the same
interface.

## Use

```bash
# model directory: glycoflow.safetensors + glycoflow.json (GlycoFlow scripts/export_rust_fixtures.py)
# and residue_library.json (GlycoFlow glycoflow/resources/)
export GLYCOFLOW_MODEL=/path/to/model
reglyco refine --protein 5KZC.pdb --density-map eds-5kzc.ccp4 --objective density \
    --replace-glycan A:79 --seed 0 -o out/
# GPU: cargo build --release -p reglyco --features reglyco-cli/glycoflow-cuda
#      (CUDA_COMPUTE_CAP=<cc>, nvcc on PATH), then --glycoflow-device cuda
```

`--replace-glycan SITE` fits the deposited glycan's sequence (GLYCAM from crabWURCS
`write_glycam`, checked against the deposited tree); `SITE=<GLYCAM sequence>` fits another one.
Outputs: `fitted.pdb` (protein with the glycan replaced; deposited CCD residue names and numbers,
PDB atom names, CONECT for every glycan bond, LINK to the Asn), `candidates.pdb` (best fit,
density-ambiguous alternatives, prior completions; `REMARK 250` labels), `glycoflow-fit.json`
(objective terms, support classes, alternatives, completions, costs, evaluation against the
deposited glycan) and `validation.json` (`reglyco-validate` with the density score).
The model's `REMARK 2 RESOLUTION` and `CRYST1` records are read from the input PDB.

## Tests

```bash
cargo test -p reglyco-glycoflow            # offline: unit tests, finite-difference gradients on a synthetic site
# parity with the Python reference (needs the site models and maps):
#   GlycoFlow: .venv/bin/python scripts/fitting/export_fit_fixtures.py --sites sites.json \
#                  --only 5KZC_A79 5GSQ_A297 5GSQ_B297 --out-dir /path/to/fixtures
GLYCOFLOW_FIT_FIXTURES=/path/to/fixtures cargo test --release -p reglyco-glycoflow --test parity \
    parity_ -- --ignored --nocapture
# the full pipeline on the reference's templates and prior samples (removes RNG differences):
GLYCOFLOW_FIT_FIXTURES=... GLYCOFLOW_MODEL=/path/to/model cargo test --release -p reglyco-glycoflow \
    --test parity controlled_fit -- --ignored --nocapture
```

## Differences from the Python reference

* Random numbers (initial torsions, pucker draws of the 15 non-majority templates, prior samples)
  come from the engine's SplitMix64, so runs are statistically, not bitwise, equal to Python's.
* The objective is evaluated in float64 (the reference: float32 on the GPU, including
  `torch.cdist`), CPU-parallel over candidates with rayon; the network runs on candle (CPU, or
  CUDA with the `cuda` feature).
* Symmetry operators come from the embedded table: the model's CRYST1 symbol (else the map
  header's space-group number) and cell; expansion only for full-cell maps whose cell matches.
* Maps are resampled with ReGlyco's reader (non-periodic maps must contain the site box).
