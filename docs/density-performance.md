# Density-fitting performance profile

This page records the production profile of the evidence-aware Adaptive
pipeline. It is intentionally a diagnostic document: changing a timer or an
evaluation count is not a scientific improvement unless the selected pose,
validation state, and evidence classifications remain equivalent.

## 5KZC A:79 baseline

The reference run is the cached level-3, 512-conformer `G63337SS` Man9 site
with four Rayon workers, the PDBe EDS primary map, automatic sigma
calibration, and no Fo--Fc map. The frozen evidence-selection output is in
`example-output/5kzc-evidence-selection-full`.

| Production stage | Seconds | Evaluations | Interpretation |
| --- | ---: | ---: | --- |
| Broad/root/core capture | 40.48 | 6,656 | Attachment and distinct core-basin capture. The current timer includes the conformer/core handoff and is therefore an upper bound for this stage. |
| Focused basin refinement | 0.20 | 16 | Final pilot resampling after broad capture. |
| Adaptive local/evidence refinement | 328.03 | 53,824* | **Previous cumulative profile.** It included the finalization tail and reported the cumulative evaluation count. |
| Finalization, validation, and report writing | ~33.56 | -- | Remainder between the old stage sum and the reported 402.27 s optimization total. |
| **Total optimization** | **402.27** | **53,825** | Frozen scientific reference before timing/accounting cleanup. |

The current source records a non-overlapping profile.  The accepted parallel
profile is in `example-output/5kzc-parallel-default-final`.  It parallelizes
independent fixed-ROI shortlist/refinement work and defers native-arm
alternative screens until final reporting.  The selected pose and evidence
are unchanged:

| Current production phase | Seconds | Evaluations |
| --- | ---: | ---: |
| Setup / map preparation | 33.08 | 0 |
| Capture / root + core | 41.29 | 6,656 |
| Basin pilot / handoff | 0.20 | 16 |
| Local / branch + evidence refinement | 286.60 | 44,390 |
| Finalization / candidates + baselines | 18.60 | 2,763 |
| **Total optimization** | **379.79** | **53,825** |

The run produced CC `0.6899946`, full-tree heavy-atom RMSD `2.56324 Å`, the
same arm/residue evidence, and the same final candidate ordering as the
committed reference.  This is an 18.68 s (4.7%) optimization-time reduction;
the local branch/evidence phase remains the measured bottleneck.

The same source was compared against a clean `0abed59` worktree on the two
5GSQ single-site cases:

| Case | Baseline seconds | Accepted seconds | Evaluations | RMSD / CC unchanged |
| --- | ---: | ---: | ---: | --- |
| A:297 | 418.83 | 388.60 | 41,312 | 2.17925 Å / 0.4949787 |
| B:297 | 199.91 | 188.14 | 27,277 | 1.27006 Å / 0.5103363 |

The two comparisons also matched evidence classifications, alternative
counts, and candidate ordering.  A change is considered default-safe only
when it passes this kind of baseline comparison; a candidate-generation
change that improves one map while changing another is rejected.

A rejected compact-indexed proposal scorer is intentionally not part of the
production path: although it improved the 5KZC score in one run, it changed
5GSQ A:297.  The accepted change parallelizes only independent work around the
existing exact scorer and does not add nested map-level parallelism.

The two polished basins account for most of the local stage:

| Basin | Seconds | Evaluations | State |
| --- | ---: | ---: | --- |
| `model-60@-157,167` | 112.13 | 13,176 | selected |
| `model-31@-157,167` | 194.90 | 30,098 | selected runner-up |

The current evidence-aware representative remains validation-clean and keeps
the established semantics: supported arms are selected by fixed-ROI density
gain, ambiguous arms retain alternatives, and negative-evidence arms retain a
chemically compatible GlycoShape prior mode. The raw density-max candidate is
still available in `candidates.pdb`; it is not silently substituted for the
representative.

## Optimization target

The next optimization target remains the adaptive local stage.  It is the
largest cost, but any incremental scorer must pass all three cases above before
it can replace the current exact path.  For traceability, the rejected
compact-coordinate trial took 439.5 s and 51,431 evaluations, changed the
retained basin, and is not a production result.

The required equivalence checks for each optimization are:

1. same validation-clean best pose (or numerically indistinguishable
   coordinates);
2. same arm evidence classes and selected/credible-set modes;
3. same fixed-ROI score ordering for every retained candidate;
4. no increase in serious clashes or topology/stereochemistry errors; and
5. lower wall time and/or lower full-score-equivalent evaluations.

Debug logging (`REGLYCO_DENSITY_DEBUG=1`) is not a benchmark setting because
frontier diagnostics are intentionally verbose and can dominate a run.

## Reproducing the profile

Run the checked-in harness from the repository root. It uses the cached/remote
PDB-ID workflow, four Rayon workers by default, primary-map-only scoring, and
does not impose a hidden timeout:

```console
scripts/benchmark-density.sh example-output/density-benchmarks
```

The default cases are 5KZC A:79 and the two 5GSQ sites. Each case records the
exact command, `/usr/bin/time` output, and the fitter's `density.json`. Select
individual cases by naming them after the output directory, for example
`scripts/benchmark-density.sh example-output/density-benchmarks 5kzc`. Set
`RAYON_NUM_THREADS` to compare thread counts. An explicit diagnostic failsafe
can be passed through `REGLYCO_EXTRA_ARGS`, for example
`REGLYCO_EXTRA_ARGS='--density-time-limit 60'`; this is intentionally opt-in
and must not be used for the reference profile.

The machine-readable `timings.stages` records are non-overlapping: setup/map
preparation, root/core capture, basin handoff, local branch/evidence work, and
candidate/baseline finalization. Their evaluation counts sum to the reported
optimization evaluations (apart from stages that perform no map evaluations).
