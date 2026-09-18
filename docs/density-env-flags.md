# Density environment-flag inventory

The command-line interface is the supported production interface. Environment
variables below are retained only for diagnostics and historical experiments;
none is required for the default Adaptive result. They are intentionally not
part of the stable report schema.

## Stable diagnostic controls

| Flag | Scope | Policy |
| --- | --- | --- |
| `REGLYCO_DENSITY_DEBUG` | refiner/CLI/density | Verbose progress and proposal diagnostics; never use for timing benchmarks. |
| `REGLYCO_ADAPTIVE_BASIN_INDEX` | refiner | Undocumented basin reproduction control; useful only when comparing a specific pilot basin. |
| `REGLYCO_DENSITY_DEBUG_REF` | refiner | Diagnostic reference-coordinate summaries; does not affect ranking. |

## Search experiments retained for comparison

These flags switch or disable individual historical stages. They are kept so
old scientific investigations remain reproducible, but they should not be
used in a production command and should not be added to the README command:

`REGLYCO_ADAPTIVE_ALL_ARM_MODES`, `REGLYCO_ADAPTIVE_BASINS`,
`REGLYCO_ADAPTIVE_CONFORMERS`, `REGLYCO_ADAPTIVE_DENSITY_DIRECT_RING`,
`REGLYCO_ADAPTIVE_DENSITY_RING_CAPTURE`, `REGLYCO_ADAPTIVE_DIRECT_LOCAL`,
`REGLYCO_ADAPTIVE_DISCRETE_ONLY`, `REGLYCO_ADAPTIVE_EXACT_FRONTIER`,
`REGLYCO_ADAPTIVE_EXHAUSTIVE`, `REGLYCO_ADAPTIVE_FRONTIER_EXHAUSTIVE`,
`REGLYCO_ADAPTIVE_FROZEN_FRONTIER`, `REGLYCO_ADAPTIVE_GLOBAL_TORSIONS`,
`REGLYCO_ADAPTIVE_LOCAL_POSES`, `REGLYCO_ADAPTIVE_OFFSET_CAPTURE`,
`REGLYCO_ADAPTIVE_PILOT_INDEX`, `REGLYCO_ADAPTIVE_RIGID_LINKAGE`,
`REGLYCO_ADAPTIVE_SUBTREE_EXHAUSTIVE`, `REGLYCO_ADAPTIVE_TERMINAL_EXHAUSTIVE`,
`REGLYCO_ADAPTIVE_TERMINAL_FAST`, `REGLYCO_ADAPTIVE_TRANSPLANT_ARMS`,
`REGLYCO_COMPARE_DIRECT_LOCAL`, `REGLYCO_DISABLE_COMPACT_HANDOFF`,
`REGLYCO_DISABLE_NEIGHBOR_NUISANCE`, `REGLYCO_ENABLE_COMPACT_HANDOFF`,
`REGLYCO_ENABLE_SITE_OWNERSHIP`, `REGLYCO_FRONTIER_CONTEXT_SWITCH`,
`REGLYCO_FRONTIER_DISABLE_ARM_RESCUE`, `REGLYCO_FRONTIER_DISABLE_AUTO_CAPTURE`,
`REGLYCO_FRONTIER_DISABLE_CORE_REFINEMENT`,
`REGLYCO_FRONTIER_DISABLE_CORRELATED_RESCUE`,
`REGLYCO_FRONTIER_EXHAUSTIVE_DICTIONARY`, `REGLYCO_FRONTIER_FAST_ARM_SCREEN`,
`REGLYCO_FRONTIER_LATE_CORRELATED_RESCUE`, `REGLYCO_FRONTIER_LEGACY_POLISH`,
`REGLYCO_FRONTIER_LEGACY_POSES`, `REGLYCO_FRONTIER_NATIVE_ROOT_HANDOFF`,
`REGLYCO_FRONTIER_POST_STAGE_RESCUE`, `REGLYCO_FRONTIER_PROFILED_FAST_SCORE`,
`REGLYCO_FRONTIER_RING_LATTICE`, `REGLYCO_FRONTIER_RING_PROPOSALS`,
`REGLYCO_FRONTIER_ROOT_POLISH`, `REGLYCO_USE_LEGACY_CORE_CLUSTER`,
`REGLYCO_USE_LEGACY_LOCAL_QUEUE`, and `REGLYCO_USE_LEGACY_ROOT_HANDOFF`.

## Debug/output-only flags

All `REGLYCO_DENSITY_DEBUG_*` flags are logging or coverage probes. The output
helpers `REGLYCO_DENSITY_DIRECT_SAVE`, `REGLYCO_DENSITY_GROWTH_OUT`, and
`REGLYCO_DENSITY_PROGRESSIVE_OUT` write diagnostic snapshots only. The map
experimental toggles `REGLYCO_DENSITY_RING_WEAK` and
`REGLYCO_FRONTIER_*` variants are not enabled by default and must not change
the production ranking policy.

This inventory precedes cleanup: flags will be removed only after their
scientific references and any archived benchmark scripts are migrated to
explicit CLI/configuration options.

