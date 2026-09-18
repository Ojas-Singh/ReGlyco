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

Density-guided replacement uses the deterministic connected-arm ensemble
search by default. In `deep` mode, a fixed protein-subtracted ROI is also
searched for map-derived ring hypotheses and topology-constrained complete
trees before the ensemble manifold is used as a weak prior/fallback. It
establishes the attachment and conserved core from
distinct GlycoShape clusters, screens every supplied conformer at that
placement, and then recombines complete correlated GlycoShape subtrees on the
same core. Parent-child torsions are refined together when independent moves
stall. Each arm is judged in one fixed density region using ring and continuous
linkage-path support, which prevents a disconnected nearby blob from receiving
credit as the intended branch.
It has no default wall-clock cutoff; `--density-time-limit`
and `--max-density-evaluations` are explicit emergency failsafes.  Ambiguous
fits are written as a posterior credible set (95% and at most ten models by
default) in `candidates.pdb` and `candidates.json`:

```console
cargo run --release -- refine \
  --pdb-id 5KZC \
  --assembly 1 \
  --replace-glycan A:79=G63337SS \
  --anomer beta \
  --level 3 \
  --objective density \
  --density-map auto \
  --density-map-source pdbe \
  --density-difference-map auto \
  --density-effort deep \
  --post-relax none \
  --report \
  --output example-output/5kzc-density-deep \
  --overwrite
```

This is the one-shot scientific example: it downloads the 5KZC biological
assembly and the PDBe EDS density map, replaces the deposited glycan at `A:79` with
the 512-conformer GlycoShape `G63337SS` Man9 ensemble, fits correlated complete
arms, and writes `fitted.pdb`, `candidates.pdb`, `candidates.json`,
`density.json`, `validation.json`, `search.json`, `report.json`, `report.pdf`,
`report.typ`, `glycoshape-density-best.pdb`, `glycoshape-nearest-fit.pdb`,
`glycoshape-baselines.json`, `density-ring-hypotheses.pdb`,
`density-ring-hypotheses.json`, `density-ring-graph.pdb`,
`density-graph-best.pdb`, `deep-internal-best.pdb`, `deep-arm-best.pdb`,
`deep-cartesian-best.pdb`,
and the three local visualization maps. To fit a different entry, change
`--pdb-id`, the `SITE=GLYCAN` pair, and (optionally) `--density-site`. When
`--density-sigma` is omitted, ReGlyco tests a deterministic width grid against
held-out voxels around nearby fixed protein atoms and uses the selected width
for every glycan candidate. A numeric value remains an expert override.
`--density-effort fast`, `adaptive` (default), and
`deep` control evidence-driven escalation without imposing a time cutoff.
Deep mode uses deterministic graph growth, parallel exact scoring of
independent ring/arm hypotheses, exact fixed-ROI torsion gradients, and
validation-gated restrained Cartesian cleanup. `density.json` records the
ring-hypothesis count, graph evaluation count, calibrated glycan B factor,
training/held-out likelihoods, and stage timings.
The experimental native detector/topology solver can be selected explicitly
with `--density-effort deep --density-deep-strategy ring-graph`; it writes
`density-ring-hypotheses.json`, `density-ring-graph.pdb`, and
`ring_graph_diagnostics` in `density.json`. The established adaptive result
is not changed by this option.
For PDB-ID workflows, `--density-difference-map auto` attempts to acquire a
PDBe Fo-Fc map and records its provenance; use `none` for a primary-map-only
fit or pass a local CCP4/MRC path explicitly.

Use `--density-search swarm` or `--density-search staged` only to reproduce
legacy diagnostic searches.  `fitted.pdb` is always the highest-posterior
complete model; deposited glycan coordinates are used only for recovery
diagnostics and never for fitting or ranking.

### Fast progressive adaptive example

For a cached PDB-ID workflow, Adaptive is fully automatic: omit both the
sigma override and the basin-index diagnostic.  ReGlyco calibrates a shared
protein-shell kernel, uses its broader capture scale to find attachment/core
basins, and compares the surviving basins on one fixed nominal ROI:

```console
RAYON_NUM_THREADS=4 cargo run --release -- refine \
  --pdb-id 5KZC --assembly 1 \
  --replace-glycan A:79=G63337SS \
  --anomer beta --level 3 \
  --objective density \
  --density-map auto --density-map-source pdbe \
  --density-difference-map none \
  --density-effort adaptive \
  --post-relax none --report \
  --density-time-limit 300 \
  --output example-output/5kzc-adaptive-auto-kernel \
  --overwrite
```

`--density-time-limit` is an optional emergency failsafe only; omitting it
lets basin competition finish through evidence exhaustion and numerical
convergence.  `--density-sigma` remains an expert reproducibility override.
`REGLYCO_ADAPTIVE_BASIN_INDEX` is intentionally undocumented and is retained
only for diagnosing a particular pilot basin; normal fitting never requires
it.

Adaptive interprets every complete branch independently after fitting.  The
report and `density.json` distinguish three evidence states:

- `density_determined`: positive fixed-ROI gain and one dominant connected
  arm mode;
- `ambiguous`: positive evidence remains distributed across distinct arm-root
  modes, which are reported as a 95% marginal credible set;
- `ensemble_prior_determined`: adding the density-driven arm has negative
  fixed-ROI gain, so `fitted.pdb` uses a chemically compatible GlycoShape
  population-prior subtree and does not claim that its coordinates came from
  density.

Arm posteriors are conditional on the fitted core and marginalize the other
arms.  Consequently a strongly determined arm cannot manufacture certainty
for an unrelated weak arm.  When a prior fallback is required, the rejected
density-maximizing complete pose remains available in `candidates.pdb` for
diagnosis, while `fitted.pdb` is the evidence-aware representative.

Adaptive first places and freezes the attachment/core, then grows each
topology-authorized branch against the residual left by the accepted prefix.
The child ROI is fixed before proposals are scored, so a distal peak cannot
move the mask or skip an intervening residue. Parent linkages are reopened
only when fixed-ROI evidence supports it. The final cleanup now includes a
bounded terminal-ring-pose search: the anomeric C1 and parent linkage remain
fixed while the terminal pyranose frame is locally rotated and accepted only
when prefix-subtracted density, held-out likelihood, and clash checks improve.

Adaptive also ships an opt-in experimental full-domain torsion supplement:
set `REGLYCO_ADAPTIVE_GLOBAL_TORSIONS=1` to add a small, space-diverse sweep
over the complete periodic torsion domain.  Those "global" modes carry no
conformer membership, so any phi/psi/omega basin outside the 512-conformer
library is reachable and must be earned by density alone.  Measured on cached
5KZC this supplement is currently inert: the beam's seed-preserving exact
selection keeps the outcome identical whether the sweep is on or off, so it
is opt-in until it demonstrates a recovery win rather than being the default.

The automatic run records both the nominal shell-calibrated width and the
broader capture width in `density.json` (`sigma_calibration`).  Adaptive uses
the capture width only to cross basin/manifold gaps, then re-scores every
finalist with the shared nominal kernel and fixed ROI.  Per-residue blur is
tested only after a connected pose exists and is accepted only with a held-out
likelihood and BIC gain.

The canonical no-override cached Adaptive benchmark is recorded in
`example-output/5kzc-parallel-default-final`.  The accepted optimization only
parallelizes independent proposal/refinement work and defers native-arm
alternatives until final reporting.  It keeps the
same candidate ordering, evidence classes, and validation outcome as the
committed implementation:

| Case | Optimization | Evaluations | CC | Full-tree RMSD | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 5KZC A:79 | 379.79 s | 53,825 | 0.6899946 | 2.56324 Å | identical to baseline |
| 5GSQ A:297 | 388.60 s | 41,312 | 0.4949787 | 2.17925 Å | identical to baseline |
| 5GSQ B:297 | 188.14 s | 27,277 | 0.5103363 | 1.27006 Å | identical to baseline |

The previous committed timings were 398.47 s, 418.83 s, and 199.91 s,
respectively, so this is an approximately 5–7% speedup without a scientific
tradeoff.  Recovery RMSD values are diagnostics only; deposited coordinates
never participate in proposal generation, fitting, or ranking.  No sigma or
basin override is required for these runs.

`density.json` separates setup, root/core capture, basin handoff, local
branch/evidence work, and finalization.  In the accepted parallel profile the
5KZC local phase is 286.60 s (down from 303.07 s); it remains the dominant
phase and is the next target for further work.

The production timing breakdown and optimization guardrails are recorded in
[`docs/density-performance.md`](docs/density-performance.md). The largest
measured cost is the adaptive local/evidence stage; capture-only shortcuts are
not accepted unless they preserve the selected pose, evidence classifications,
validation state, and fixed-ROI candidate ordering. The diagnostic environment
flags and their cleanup policy are inventoried in
[`docs/density-env-flags.md`](docs/density-env-flags.md).
The repeatable cached benchmark harness is
[`scripts/benchmark-density.sh`](scripts/benchmark-density.sh); it records the
exact command, wall-clock resource usage, and `density.json` for 5KZC and both
5GSQ sites without applying a default failsafe.

For a PDB-ID density run with no explicit attachment, ReGlyco defaults to
biological assembly 1 and discovers every carbohydrate component with an
explicit protein LINK through crabWURCS 0.3.1. Each site is resolved against
GlycoShape at level 3, checked for exact canonical topology, fitted with its
own Adaptive scorer/ROI/conformer state, and merged only after fitting. This
preflight is atomic: an unsupported attachment or missing exact ensemble stops
before the input is stripped. Use `--asymmetric-unit` for the old asymmetric
unit, or `--assembly N` for another assembly.

```console
RAYON_NUM_THREADS=4 cargo run --release -- refine \
  --pdb-id 5GSQ \
  --objective density \
  --density-effort adaptive \
  --post-relax none --report \
  --output example-output/5gsq-adaptive \
  --overwrite
```

The merged model is `fitted.pdb`; per-site models and diagnostics are under
`sites/<chain>_<residue>/`, and `discovered-glycans.json` records canonical
WURCS/IUPAC, GlyTouCan/GlycoShape identifiers, and source residues.
For a multi-site run, an explicit `--density-time-limit` is applied independently
to each site; the reported total can therefore exceed that per-site failsafe.
Sites are never jointly coordinate-optimized during this stage, and the final
merge reports any retained cross-site clash; when a density pose clearly
dominates its native alternatives, it is preserved instead of silently being
replaced by an unrelated fallback.

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
steric GA over cluster, φ, ψ, and optional Dunbrack rotamers. It requires a
clash-free complete result unless `--allow-clashes` is explicitly supplied.
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
rotamers. Exhausted search fails by default; `--allow-clashes` emits the best
complete result and marks it visibly in JSON and terminal output.

The workspace crates are:

- `reglyco-core`, shared requests, ensembles, outcomes, warnings, and errors
- `reglyco-build`, attachment chemistry and in-memory GlySys preparation
- `reglyco-ensemble`, local/remote providers, VMM sampling, MH, steric scoring,
  linkage-angle genes, and deterministic compatible-set GA search
- `reglyco-relax`, fixed/movable vacuum or opt-in OBC2 L-BFGS minimization
- `reglyco-refine`, search → build → parameterize → relax orchestration
- `reglyco-report`, common serializable reports and provenance
- `reglyco-validate`, structural and attachment-metadata checks
- `reglyco-density`, density-guided fitting and report integration
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
those overrides are kept out of published package metadata. Density-specific
scientific objectives and a PyO3 compatibility layer remain separate
milestones.
