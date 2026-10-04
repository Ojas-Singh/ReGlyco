# ReGlyco

ReGlyco is a native Rust glycoprotein construction, ensemble-search, and
refinement workspace. It attaches explicit glycan conformers to supported
protein glycosylation sites and passes the resulting in-memory structure to
[GlySys](https://github.com/Ojas-Singh/GlySys) for Amber/GLYCAM
parameterization and staged energy minimization. Relaxation runs in vacuum by
default for responsive interactive use; OBC2 GBSA is an explicit opt-in.

Protein inputs can be supplied in three mutually exclusive ways:

- `--protein protein.pdb` reads a local file.
- `--uniprot O15552` resolves the current versioned AlphaFold DB model.
- `--pdb-id 4HHB` downloads legacy PDB coordinates from the RCSB archive.

Downloaded proteins are cached under `.reglyco-cache/proteins`; use
`--protein-cache PATH` to move the cache or `--protein-offline` to forbid
network access.

## GlycoShape data access

Glycan ensembles named by GlyTouCan ID are fetched from the GlycoShape API at
`https://glycoshape.org` (`--api-base`). Command-line clients receive Level 1,
the public dataset, which is the default `--level`. Levels 2 and 3 are not
served to command-line clients, and requesting them fails with an explanatory
error. Local ensemble bundles (`--glycan PATH` or `--attach SITE=PATH`) are used
as supplied. glycoshape.io is retired and should not be used as `--api-base`.

## Licensing

This project is dual licensed.

The public source code is available under the **GNU Affero General Public
License v3.0 only (AGPL-3.0-only)**. See [`LICENSE`](LICENSE).

Organizations that require terms other than the AGPL—for example for
proprietary integration or commercial redistribution—may obtain a separate
commercial licence from the copyright holder. See
[`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

A separate commercial licence does not change the AGPL rights granted to users
of the public version.

Third-party dependencies remain subject to their respective licences.

## Installation

After the first public release, install the native executable from crates.io
with the locked dependency graph:

```console
cargo install reglyco --locked --version 0.1.0
```

Development checkouts can instead use `cargo run --release -- ...`. Release
publication order and the pinned Colab binary workflow are documented in
[`docs/release.md`](docs/release.md).

## Runnable examples

The commands below run from the repository root.

Download the GlcNAc ensemble used by the scan workflow. This writes both the
multi-model distribution and its first conformer. Add `--report` to create an
offline PDF summary and the figures used to make it:

```console
cargo run -- ensemble \
  --glycan G14843DJ \
  --report \
  --output example-output/ensemble
```

Scan the AlphaFold model for UniProt O15552. Each unoccupied `N-X-S/T` sequon
(`X` cannot be Pro) is optimized independently, then the independently
accessible sites are reduced to a jointly compatible subset. Its default GA
settings match Cookbook scan: population 128 and 100 generations:

```console
cargo run -- scan \
  --uniprot O15552 \
  --glycan G14843DJ \
  --seed 7 \
  --report \
  --output example-output/o15552-scan
```

At the current O15552 AlphaFold model this identifies A:151 `NTT`, A:167
`NFT`, A:239 `NVS`, and A:265 `NAS`. `scan.json` reports independent structural
accessibility and joint-subset membership for every site. `structure.pdb`
contains only the jointly compatible subset; blocked sites are never written
with placeholder coordinates. This is a structural-accessibility prediction,
not proof that a site is biologically glycosylated. Add `--system` to prepare
an Amber/GROMACS bundle after scanning.

Build two distinct glycans onto the AlphaFold O15552 model. This fixed-seed
case uses rotamers and produces a clash-free two-site model at A:151 and
A:167:

```console
cargo run --release -- build \
  --uniprot O15552 \
  --attach A:151=G98687PM \
  --attach A:167=G92042VQ \
  --rotamers \
  --population 64 \
  --generations 30 \
  --seed 7 \
  --no-system \
  --report \
  --output example-output/o15552-build
```

Sample the corresponding attached ensemble. This draws from its native
cluster/φ/ψ distributions and rejects only incompatible frames; it does not
replace the ensemble distribution with GA optimization. Protein coordinates,
including side chains, remain fixed by default:

```console
cargo run --release -- ensemble \
  --uniprot O15552 \
  --attach A:151=G98687PM \
  --attach A:167=G92042VQ \
  --frames 10 \
  --seed 7 \
  --report \
  --output example-output/o15552-ensemble
```

Add `--calculate-sasa` to write the cookbook-compatible `sasa.pdb` and
`real_sasa.pdb` residue maps. Add `--calculate-hotspots` as well to write the
binary `hotspots.pdb` binder-design mask. These outputs use the pure-Rust
GROMACS double-cubic-lattice calculation with the Cookbook's 1.4 Å rolling
probe and `-ndots 15` setting.

The sampler first performs adaptive weighted native sampling. If that phase
cannot fill the requested count, ReGlyco automatically seeds constrained
Metropolis--Hastings chains from a complete 128-member, 200-generation GA
solution. `--chains`, `--burn-in-sweeps`, and `--thinning-accepted` tune that
fallback; there is no per-frame `--max-attempts` cutoff. A successful command
writes exactly the requested number of compatible frames or reports that the
attachment geometry is genuinely infeasible. Use `--move-sidechains` only
when deliberately allowing local side-chain rotamer changes during attached
sampling (the older `--rotamers` spelling is accepted as an alias). This
changes the protein and is therefore not part of the native, protein-fixed
ensemble distribution.

Relax the optimized O15552 build with the staged protocol: glycans first,
then glycans plus nearby sidechains with the backbone fixed. This is a
scientific calculation on a full AlphaFold model. It uses vacuum force-field
energy by default and prints parameterization, stage, iteration, energy, and
gradient progress to stderr while it runs:

```console
cargo run --release -- relax \
  --input example-output/o15552-build/glycoprotein.pdb \
  --movable glycans \
  --max-iterations 200 \
  --report \
  --output example-output/o15552-relaxed
```

Add `--obc2` for the more expensive solvent-aware OBC2 GBSA objective. It is
deliberately opt-in because its global solvent derivatives are much slower on
large proteins.

Density fitting (`refine --objective density`) fits the glycan of known
sequence at each `--replace-glycan` site into a CCP4/MRC map with the GlycoFlow
flow model (crate `reglyco-glycoflow`). The GlycoFlow model (`glycoflow.safetensors`,
`glycoflow.json`, `residue_library.json`) is not part of ReGlyco: it is licensed
separately under the GlycoFlow Source-Available Non-Commercial License
(non-commercial academic research; commercial use needs a separate written
licence) and distributed through the gated Hugging Face repository
[`Ojas-Singh/glycoflow`](https://huggingface.co/Ojas-Singh/glycoflow), where the
GlycoFlow authors approve access requests. Once your request is approved:

```console
hf auth login        # or export HF_TOKEN=<token of the approved account>
# once: accept the model licence and download the model (prints the model directory)
cargo run --release -- glycoflow-model --accept-glycoflow-license
cargo run --release -- refine \
  --protein 5KZC.pdb --density-map eds-5kzc.ccp4 \
  --objective density --replace-glycan A:79 \
  --seed 0 --output example-output/5kzc-density
```

The acceptance is recorded in the model cache (`~/.cache/reglyco/glycoflow`, or
`$GLYCOFLOW_CACHE`); `refine` also takes `--accept-glycoflow-license`, and
`GLYCOFLOW_ACCEPT_LICENSE=1` accepts it for one run without recording. Without a
model directory the model is downloaded on first use (pinned revision, SHA-256
checked); `--glycoflow-model <dir>` or `$GLYCOFLOW_MODEL` use a local copy.

`--density-map auto` resolves a sidecar map next to a local model or, with
`--pdb-id`, downloads the PDBe EDS map (`--density-map-source rcsb` for the
RCSB 2Fo-Fc map). `--replace-glycan SITE=<GLYCAM sequence>` fits a different
sequence than the deposited one. The run writes `fitted.pdb`, `candidates.pdb`,
`glycoflow-fit.json` and `validation.json`; build with
`--features reglyco-cli/glycoflow-cuda` and pass `--glycoflow-device cuda` for
the GPU path. Options, outputs and the method are described in
[`docs/glycoflow-fitting.md`](docs/glycoflow-fitting.md);
[`scripts/benchmark-glycoflow.sh`](scripts/benchmark-glycoflow.sh) runs the
fit on a list of sites and records wall time and peak memory.

For a quick report-generation smoke check only, use one iteration and omit
the second stage. It verifies the workflow but is not a converged model:

```console
cargo run --release -- relax \
  --input example-output/o15552-build/glycoprotein.pdb \
  --max-iterations 1 \
  --no-local-sidechains \
  --report \
  --output example-output/o15552-relax-smoke
```

The smaller diagnostic settings below are similarly useful only for command
smoke tests, not structural predictions:

```console
cargo run -- build \
  --protein tests/fixtures/protein.pdb \
  --site A:1 \
  --glycan example-output/ensemble \
  --population 4 \
  --generations 1 \
  --seed 7 \
  --output example-output/build-smoke \
  --no-system \
  --overwrite
```

### SAXS modeling and glycoform inference

ReGlyco includes four SAXS workflows backed by the sibling
[`crabSAXS`](../crabSAXS) library. They accept a supplied PDB/mmCIF ensemble
with `--models`, or generate equal-weight Re-Glyco frames from a protein input
and repeated `--attach SITE=GLYCAN` options:

```console
reglyco saxs model \
  --data saxs.dat --models models.pdb --report --output example-output/saxs-model

reglyco saxs ensemble \
  --data saxs.dat --protein protein.pdb \
  --attach A:139=G47816DI --frames 50 \
  --report --output example-output/saxs-ensemble

reglyco saxs reweight \
  --data saxs.dat --models models.pdb \
  --kl-strength 1.0 --report --output example-output/saxs-reweight
```

`model` selects the supplied or generated structure with the lowest fitted
reduced χ². `ensemble` fits the unbiased equal-weight Re-Glyco ensemble, and
`reweight` applies maximum-entropy reweighting relative to that uniform prior;
the stored `log_native_probability` values remain provenance diagnostics and
are not multiplied into SAXS weights.

For per-site occupancy and glycoform inference, enumerate the candidate set
with shorthand such as:

```console
reglyco saxs occupancy \
  --data saxs.dat --protein protein.pdb \
  --candidate A:139=G47816DI,none \
  --candidate A:280=G47816DI,none \
  --candidate A:301=G47816DI,none \
  --candidate A:340=G47816DI,none \
  --max-combinations 16 --report \
  --output example-output/saxs-occupancy
```

The same candidates can be supplied in JSON or TOML with
`--candidate-manifest`. Searches are exhaustive and stop with an explicit
error when they exceed `--max-combinations`. χ²-derived likelihoods and
candidate priors are reported separately from the robust multi-feature rank
used to choose a stable best-supported combination. `none` means that the
site is excluded from the generated attached-site set.

Every SAXS run writes a machine-readable result, fitted curve, PDB output,
and a four-panel diagnostic SVG (log-log intensity with error bars, semilog
intensity, Kratky, and experimental/model P(r)). `--report` adds the same
figure to the JSON/Typst/PDF bundle and occupancy runs add site-occupancy and
per-site glycoform-distribution plots. These are best-fitting or
most-supported models, not claims of structural truth.

## Reading results

Every workflow writes a machine-readable `report.json`. Passing `--report`
also writes `report.pdf`, its reproducible `report.typ` source, and
`report-assets/*.svg` without requiring a Typst executable or network access.

- Build and search reports show the selected conformer/main cluster, linkage
  φ/ψ, optional rotamer, per-site steric score, the 1.1 clash-free threshold,
  and GA convergence.
- Attached ensemble reports compare observed cluster counts with native cluster
  weights, list frame-level compatibility and log probabilities, and summarize
  protein-linkage and internal glycosidic torsions. These distributions describe
  the structural ensemble, not biological occupancy.
- Relaxation reports show energy components in kcal/mol, energy histories,
  gradient units of kcal/mol/Å, movable atoms, accepted steps, and convergence
  diagnostics for each stage.
- Scan reports distinguish independently accessible sequons from the joint
  compatible subset; this remains a structural-accessibility prediction, not
  proof of biological glycosylation.
- SAXS reports distinguish χ² likelihoods from robust multi-feature ranks,
  include Rg, Dmax, Kratky, P(r), scale/background, and intensity-correlation
  diagnostics, and show effective sample size for reweighted ensembles.

SNFG diagrams and canonical glycan names are extracted with crabWURCS PDB and
rendered with crabWURCS SNFG. If a nonstandard carbohydrate cannot be
recognized, the report records a warning while preserving the numerical result.

The build, search, refine, and scan commands all accept local, AlphaFold, or
RCSB protein sources where they take a protein input.

## Commands

```console
reglyco build --protein protein.pdb \
  --attach A:42=G00028MO --seed 7 --output result
```

Repeat `--attach`, or paired `--site` and `--glycan`, for a multi-site build.
Glycans may be GlyTouCan identifiers or local ensemble bundles. Build runs the
steric GA over cluster, φ, ψ, and optional Dunbrack rotamers. It writes the
best complete constructible result found within the configured budget, even
when some steric or VMM gates remain unresolved. Use `--require-clash-free`
for strict automation; `--allow-clashes` remains a compatibility alias for
the diagnostic default.
The output directory contains `glycoprotein.pdb` and, by default, the Amber and
GROMACS system bundle written by GlySys. Use `--no-system` to write only the
constructed PDB.

Add `--energy` to rank clash-free build/search candidates by vacuum
Amber/GLYCAM energy, or `--interact` to rank by protein--glycan Lennard-Jones
plus Coulomb interaction energy. `--energy-obc2` is available only with full
energy scoring. Attached ensembles use the same flags for 300 K
energy-biased sampling; tune `--temperature-k`, `--burn-in`, and `--thinning`
when a longer Markov chain is needed.

`--min` performs a short local minimization before each energy score. It moves
all glycan atoms and every atom of complete protein residues within 5 Å, while
the rest of the protein remains fixed. Search uses an unsolvated reusable
topology and a 10 Å nonperiodic cutoff, configurable with `--min-radius` and
`--energy-cutoff`. Progress is printed after topology setup and every GA
generation; use `--quiet` for scripts.

```console
cargo run --release -- build \
  --protein protein.pdb \
  --attach A:42=G00028MO \
  --interact --min --min-iterations 3 \
  --population 64 --generations 30 \
  --report --output result
```

`search.json` and `report.json` record the LJ/Coulomb split, active local
residues, topology and evaluation timing, actual energy/minimization counts,
steric rejections, failures, and elite-cache hits. With `--report`, the PDF
adds the physical energy history and the same performance diagnostics.

New workflows use repeatable site/source specifications:

```console
reglyco ensemble --glycan G00028MO --output ensemble
reglyco search --protein protein.pdb \
  --attach A:42=G00028MO --attach B:88=/data/local-bundle \
  --seed 7 --output selected
reglyco refine --protein protein.pdb \
  --attach A:42=G00028MO --seed 7 --output refined
reglyco scan --uniprot O15552 --output scanned
reglyco relax --input glycoprotein.pdb --movable glycans --output relaxed
reglyco validate --input glycoprotein.pdb --report
```

`build`, `search`, and `refine` accept GlycoShape API/cache/offline settings,
GA population and generation controls, and real backbone-dependent Dunbrack
rotamers. An exhausted Build emits the best complete result and marks it
visibly in JSON and terminal output; `--require-clash-free` writes those
artifacts before returning a nonzero status when the strict gates are not met.

The workspace crates are:

- `reglyco-core`, shared requests, ensembles, outcomes, warnings, and errors
- `reglyco-build`, attachment chemistry and in-memory GlySys preparation
- `reglyco-ensemble`, local/remote providers, VMM sampling, MH, steric scoring,
  linkage-angle genes, and deterministic compatible-set GA search
- `reglyco-relax`, fixed/movable vacuum or opt-in OBC2 L-BFGS minimization
- `reglyco-refine`, search → build → parameterize → relax orchestration
  (steric objective)
- `reglyco-report`, common serializable reports and provenance
- `reglyco-validate`, structural and attachment-metadata checks
- `reglyco-density`, CCP4/MRC maps, map acquisition, map-agreement scoring,
  and the site likelihood used by the GlycoFlow fitter
- `glycoflow-core`, the GlycoFlow inference engine (open; the trained model is licensed separately)
- `reglyco-glycoflow`, GlycoFlow fitting of glycans into density maps
- `reglyco-saxs`, the in-memory ReGlyco adapter for the sibling crabSAXS SAXS
  modeling, reweighting, and glycoform-search APIs
- `reglyco-cli`, command implementation used by the `reglyco` binary

All directory workflows write `report.json`; `--report` adds `report.pdf`,
`report.typ`, and `report-assets/`. `validate --output validation.json --report`
writes sibling `validation-report.json`, `validation-report.pdf`,
`validation-report.typ`, and `validation-report-assets/`. Refinement also writes
`relaxation.json`; `--solvate` adds the final Amber/GROMACS system bundle after
relaxation. Workspace crates exchange `Structure` and
`ParameterizedSystem` values directly and do not use temporary molecular files.

The published build resolves GlySys and crabSAXS through their registry
versions. A checkout build may use local path overrides for development;
those overrides are kept out of published package metadata. A PyO3
compatibility layer remains a separate milestone.
