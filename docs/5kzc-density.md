# 5KZC density-fitting regression

This is a model-biased end-to-end regression, not an independent omit-map
validation.  The PDBe EDS map was calculated from the deposited model, so it
is useful for checking coordinate transforms, replacement, deterministic
search, and report provenance rather than for claiming an unbiased fit.

Fetch biological assembly 1, replace the legacy glycan at A:79, and fit the
complete Man9 ensemble:

```text
reglyco refine \
  --pdb-id 5KZC --assembly 1 \
  --replace-glycan A:79=G63337SS \
  --level 3 \
  --objective density \
  --density-map auto \
  --density-map-source pdbe \
  --density-difference-map none \
  --density-sigma 1.0 \
  --report \
  --output example-output/5kzc-density-connected \
  --overwrite
```

`--density-map auto --density-map-source pdbe` selects the PDBe EDS map used by
this regression. Without the source option, automatic acquisition first checks
the RCSB VolumeServer 2Fo-Fc channel for a
full-cell BinaryCIF (converted and cached as floating-point CCP4); if RCSB is
unavailable it records a warning and falls back to the PDBe EDS CCP4 map.  The
output
directory contains `fitted.pdb`, `candidates.pdb`, `candidates.json`,
`glycoshape-density-best.pdb`, `glycoshape-nearest-fit.pdb`,
`glycoshape-baselines.json`,
`structure.pdb`, `density.json`, `search.json`, `relaxation.json`, and
`validation.json`.

Full-unit-cell CCP4 maps are recognized as periodic automatically, so the
deposited crystallographic coordinates may be translated by a unit-cell
vector without producing an out-of-map failure.

The regression acceptance checks are: nine residues in the replacement tree,
no validation errors, a fitted union correlation in the top 5% of the initial
poses, and chitobiose/trimannose-core heavy-atom RMSD no greater than 3.25 Å.
Those thresholds should be evaluated against the deposited reference and
reported together with the map SHA-256 and assembly provenance.

`--density-difference-map none` makes this benchmark use only the primary EDS
map. `auto` attempts to cache PDBe Fo-Fc density and uses it as a low-weight
consistency term; a local CCP4/MRC path is also accepted.

## Connected-arm adaptive fitting

The default adaptive strategy searches correlated ensemble geometry before
adding sparse torsion corrections:

1. A compact PSO places distinct conserved-core representatives, then every
   untouched GlycoShape conformer is screened at its matching root placement.
2. Native circular torsion vectors are clustered for each complete branch-point
   subtree. α1-6 and α1-3 arm modes may be recombined from different conformers
   on the same core, then parent-child torsions cross local saddles together.
3. Every candidate is evaluated in one fixed reachable-volume ROI. Profiled
   likelihood gain, ring-template agreement, and continuous linkage-path
   support prevent disconnected peak chasing; arm-removal/BIC diagnostics mark
   density-determined, ambiguous, and ensemble-prior cases separately.
4. Sparse analytic torsion refinement, targeted overlap resolution, and
   best-valid-pose preservation
   keep the reported result chemically valid even when a later proposal fails.

`density.json` reports the posterior weight, cumulative credible mass,
likelihood gain, prior contribution, active and unsupported residues, linkage
support, and which linkages were density-determined versus ensemble-prior.

Current cached 5KZC A:79 primary-map-only status (four vCPUs, no wall-clock
cutoff; August 2026): 65.0 seconds end to end, 9,730 evaluations, 0.543
pre-relax CC, nine residues, and zero validation errors. Supported and
full-tree heavy-atom RMSD to the deposited diagnostic reference are both 1.81
Å. The complete α1-6 and α1-3 arms are 1.82 Å and 2.34 Å, respectively;
per-residue and per-arm values are stored under `recovery` in `density.json`.
The untouched density-best conformer is `model-81` (CC 0.391), while
`model-12` is nearest to the fitted result (2.90 Å). This improves the earlier
full-tree RMSD of 3.20 Å, but CC >= 0.70 and every supported residue below 1.0
Å remain unmet scientific targets.

## Evidence-calibrated deep fitting

Deep mode now begins with a fixed-ROI density-first graph pass: residual map
peaks become ring hypotheses, complete topology-constrained branch proposals
are screened in parallel, and only the leading shortlist receives exact
fixed-ROI scoring. It then exhausts correlated native arm modes, preserves the
established core unless evidence reopens it, applies sparse analytic
torsion-gradient steps, and follows with validation-gated restrained Cartesian
steps. Omitting `--density-sigma`
calibrates one candidate-independent kernel width on deterministic spatial
holdout blocks of the nearby fixed protein shell. `--density-difference-map
auto` downloads the PDBe Fo-Fc channel; it is supporting evidence rather than
a second independent primary observation. Deep runs additionally write
`density-ring-hypotheses.pdb`, `density-graph-best.pdb`,
`deep-internal-best.pdb`, `deep-arm-best.pdb`, and `deep-cartesian-best.pdb`.
These stage snapshots are diagnostic; the highest-posterior
validation-clean model remains `fitted.pdb`.

The density-first graph and exact-gradient path is implemented and covered by
the density scorer tests. The current 5KZC deep search remains an expensive
scientific diagnostic on this four-vCPU host: the complete run can exceed ten
minutes because fixed-ROI likelihood evaluation is still the dominant cost.
There is no implicit wall-clock cutoff; supply `--density-time-limit` or
`--max-density-evaluations` when an explicit failsafe is wanted.

The August 2026 four-vCPU dual-map run in
`example-output/5kzc-density-evidence-deep` selected 1.00 Å from six tested
widths using 296 protein-shell atoms. It finished in 160.4 seconds with 16,487
evaluations (10,294 correlated-arm and 77 Cartesian evaluations), CC 0.541,
root C1 distance 0.300 Å, core RMSD 1.204 Å, supported RMSD 2.058 Å, and
full-tree RMSD 2.910 Å. It retained nine residues and passed final validation
with zero errors. The complete α1-6 and α1-3 arms had RMSD 2.278 Å and 5.055
Å, respectively.

The matched primary-only ablation in
`example-output/5kzc-density-evidence-deep-primary-only` took 159.2 seconds
and produced the same fit to numerical precision (CC 0.541, full-tree RMSD
2.910 Å). Thus Fo-Fc evidence neither improved nor measurably degraded the
selected fit in this example. The deep implementation and diagnostics are
operational, but this benchmark fails the intended CC 0.70 and complete-tree
RMSD 1.0 Å scientific acceptance thresholds.
