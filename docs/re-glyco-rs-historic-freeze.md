# Re-Glyco-rs historical freeze: what was explored

`Re-Glyco-rs` is retired. Its final working tree is committed as `062f9f6` and
tagged `historical-freeze-2026-09-22`. Nothing further should be developed
there; it is kept as a readable record of a parallel line of work.

ReGlyco is the surviving repository and the target of the rewrite. This
document exists so that work explored on the rs line is not silently lost when
the ReGlyco engine is rewritten. It records what rs did, why, and whether
ReGlyco currently has it.

## Why the two lines cannot simply be merged

- They have separate root commits and both diverged independently. rs has 120
  commits, ReGlyco 79. Neither is a fast-forward of the other.
- Licensing differs: rs is `CC-BY-NC-ND-4.0` with no `LICENSE` file; ReGlyco is
  `AGPL-3.0-only` with `COMMERCIAL-LICENSE.md`. Only ReGlyco is releasable.
- `crates/reglyco-build/src/lib.rs` is byte-identical in both. Build and
  attachment construction are *not* a differentiator: same code, only
  `Cargo.toml` differs (licence and repository URL).
- The scientific divergence is concentrated in a few files:

| file | rs | ReGlyco | diff lines |
| --- | --- | --- | --- |
| `reglyco-ensemble/src/lib.rs` | 6595 | 8761 | 2966 |
| `reglyco-ensemble/src/statistical.rs` | 746 | 1307 | 887 |
| `reglyco-workflow/src/lib.rs` | 7800 | 7524 | 844 |
| `reglyco-cli/src/lib.rs` | 4488 | 4639 | 347 |
| `reglyco-wasm/src/lib.rs` | 192 | 228 | 176 |
| `reglyco-ensemble/src/gpu.rs` | 993 | 974 | 153 |
| `reglyco-ensemble/src/geometry_gpu.rs` | 323 | 316 | 121 |
| `reglyco-ensemble/src/prepared.rs` | 928 | 953 | 99 |
| `reglyco-core/src/lib.rs` | 1214 | 1305 | 91 |
| `reglyco-refine/src/lib.rs` | 33695 | 33740 | 67 |
| `reglyco-report/src/lib.rs` | 5318 | 5326 | 8 |

Everything else under `crates/*/src` is byte-identical: `reglyco-density`,
`reglyco-relax`, `reglyco-validate`, `reglyco-saxs`, `reglyco-pdf-wasm`, and
`reglyco-ensemble/src/{dunbrack,sasa,sampling_energy}.rs`.

Roughly as many diffs have ReGlyco ahead of rs as the other way round. This is
not "rs is newer"; they are two branches. Each item below is marked with which
side currently holds it.

## 1. Search budget (Auto/Manual)

rs owns this in the workflow layer:
`crates/reglyco-workflow/src/lib.rs::{resolve_search_budget,
resolve_search_budget_values}`. ReGlyco moved the same concept into
`reglyco-ensemble::resolve_search_budget` and promoted the result type into
`reglyco-core::SearchBudgetResolution`.

rs heuristics, which the rewrite should preserve or knowingly change:

- Estimator versioning: rs stamps `manual_v0` and `auto_v1`. ReGlyco carries no
  estimator version string. Worth restoring so reports stay comparable across
  future estimator changes.
- Manual: `population = max(requested, 2)`, `generations = max(requested, 1)`,
  `required_work = population * (generations + 1)`, `capped = false`.
- Auto inputs per site: count of positive-population conformers; `min(phi, psi)`
  count of positive VMM components capped at 4; `rotamers =
  scan_rotamers ? 3 : 1`.
- `site_need = conformers * min(basins, 4) * rotamers`, clamped to `[16, 192]`.
- `pool_target = clamp(max(16 + 10 * largest_component, max(site_need)), 16, 192)`.
- `component_work = max(edges * pool_target^2 / 32, sum(site_needs) * 4)`.
- `required_work = max(128 * 101, component_work * 2)`.
- `population = clamp(64 + 8 * largest_component + max_degree, 128, 256)`,
  rounded up to a power of two.
- `generations = max(required_work / population, 100)`, hard-capped at 500, with
  `capped = uncapped_generations > 500`.
- Hard input rejections, all returning `Invalid` rather than silently shrinking
  the budget: malformed or negative `cluster_weight`, non-finite population, a
  zero-population conformer with no positive phi/psi prior component, and a
  site with no positive-population conformer at all.
- Zero-population conformers are deliberately excluded from Auto coverage: they
  may stay in an asset for provenance but cannot contribute a reachable state.

rs's tests here are the best available specification and are richer than
ReGlyco's: `auto_budget_site`, `auto_budget_ensemble`, `auto_budget_protein`
(fixtures), `auto_budget_keeps_the_floor_and_is_deterministic`,
`auto_budget_distinguishes_dense_and_disconnected_components`,
`auto_budget_counts_rotamers_and_ignores_zero_population_conformers`,
`auto_budget_rejects_missing_positive_priors_and_caps_safely`. Only the last two
concepts survive in ReGlyco, as
`auto_budget_caps_a_dense_sixteen_site_fixture_without_overflow` and
`auto_budget_keeps_small_jobs_at_the_reliability_floor`.

ReGlyco's variant is otherwise ahead: it records `conformer_counts`,
`vmm_basin_counts`, `rotamer_counts`, `site_pose_needs`, component sizes, edge
count, maximum degree and `required_compatibility_work`, and publishes
`searchBudget` in the report contract. rs publishes only a bare
`searchBudgetMode: 'auto' | 'manual'`.

## 2. Conservative conflict graph

rs: `structure_position_extent` and `conservative_conflict_components` in
`reglyco-workflow`.

- `structure_position_extent` is the bounding-sphere radius of a conformer: half
  the bounding-box diagonal, floored at 4 Angstrom.
- `conservative_conflict_components` anchors each site at the centroid of its
  protein residue atoms and joins two sites when
  `distance <= reach_i + reach_j + 12.0` Angstrom, where `reach` is the maximum
  extent over that site's positive-population conformers.
- Union-find returns component sizes, the conflict edge count and the maximum
  degree; those three feed the Auto budget.

The 12 Angstrom slack and the "max conformer reach, not mean" rule are the
conservative choices that make the estimate an upper bound. ReGlyco has the same
concept (`conservative_site_graph` plus `connected_components`), so this is a
keep-both invariant rather than a gap.

## 3. Scan workflow (GlcNAc accessibility)

Both repos carry `scan_search_config` and `scan_trial_compatible`; the design
rationale lives only in the rs doc comments and should be kept:

- A scan asks one cheap question per sequon, whether a single GlcNAc can sit
  there clash-free, and must answer it for every sequon. Running the full
  Build/Ensemble budget, or the large Auto budget, once per sequon plus once per
  joint trial makes many-site scans unusable, and the extra work does not change
  the answer: feasible single-site poses normally appear in the initial
  population, and blocked sites exhaust any budget.
- Scans therefore use a fixed first-feasible budget and never inherit the
  request's Build/Ensemble population, generations, or Auto mode:
  `population_size = 32`, `generations = 25`, `require_clash_free = false`,
  `scan_rotamers = false`, `polish_attachment_vmm = false`,
  `selection_policy = CookbookFirstFeasible`.
- Joint validity is verified cheaply and independently: `scan_trial_compatible`
  materialises the accepted per-site poses together and re-scores them, which is
  one structure build plus one steric traversal versus a complete search.

## 4. Sampler strategies and the proposal kernel

Both repos expose the same three strategy labels (`compatible_pool`,
`native_sampler`, `ga_seeded_mh`) and the same
`SearchSelectionPolicy::{CookbookFirstFeasible, JointPriorV1}`. The `target()`
function is byte-identical in both, so the stationary distribution is the same;
only the transition kernel differs.

- rs `native_sampler` (`reglyco-ensemble/src/statistical.rs`) uses a full-state
  independence proposal: every MH step redraws all sites from their priors,
  `priors.iter().map(|p| p.generate(rng))`, with delayed acceptance.
- ReGlyco replaces that with coordinated block proposals over the connected site
  graph: `conservative_site_graph` then `build_proposal_blocks` (singles,
  adjacent pairs, connected groups of 3-4, whole state), then
  `choose_proposal_block`, `propose_block` (a 50/50 mixture of independent prior
  resample and a symmetric wrapped +/-8 degree phi/psi walk),
  `incremental_steric_poses` (rebuild only the changed block, then
  changed-vs-all compatibility) and `state_vmm_allowed`.

Why this matters for the rewrite: a full-state independence proposal must redraw
every site into a jointly compatible region in one shot, so acceptance, and
therefore effective sample size per model evaluation, collapses as the site
count grows. That is the plausible mechanism behind multi-site scans and
ensembles being "very slow" on the rs line, and it is the single most important
thing not to regress. ReGlyco's `build_proposal_blocks` comment also records the
subtlety that the whole-state category must stay distinct from a same-size
connected group, or the block has two selection paths and its reverse proposal
probability becomes ambiguous.

rs's `cookbook_compatible_pool_sample` is the joint-start pooling behind the
`compatible_pool` strategy: one shared feasibility pass yielding distinct valid
joint starts for all chains, instead of running a full GA per chain.

## 5. Strict failure handling: a real behavioural divergence

This is a deliberate disagreement, not an omission.

- rs `finish_strict_failure` sets `primary_structure: None`. Its doc comment
  says the best candidate is deliberately stored as an analysis artifact only,
  and `primary_structure` stays empty so follow-up workflows cannot treat an
  out-of-contract pose as a valid parent. Enforced by
  `strict_failure_bundle_keeps_candidate_out_of_primary_structure`.
- ReGlyco sets `primary_structure: build_partial.then_some(...)`, so it can
  promote the best candidate into the primary slot. Enforced by
  `strict_build_fallback_promotes_complete_candidate_to_partial_result`.

The rewrite must pick one on purpose. rs's rule is the more conservative: a
candidate that failed both clash-free sterics and the VMM gate cannot become an
input to the next stage.

## 6. VMM compliance gates

rs carries the fuller helper set: `vmm_component_bounds_95`,
`vmm_component_within_95`, `vmm_angle_within_95` and
`probability_half_width_degrees` (7 references to `vmm_component_bounds_95` in
rs versus 4 in ReGlyco). The gating rule, documented only in rs: attachment VMM
compliance is a hard gate only for strict steric workflows (`Uniprot`,
`SiteBuild`, `Ensemble` with `SearchScoringMode::StericPrior`), while generic
energy and interaction paths still expose measured torsions as diagnostics
without turning a non-strict result into a failed workflow.

`cookbook_failure_diagnostics` (rs only) reports, per site, the chosen
conformer, the phi/psi component indices, the 95 percent bounds of each
component, and whether the accepted angle falls inside them. That is what makes
a strict rejection explainable rather than opaque.

## 7. GPU execution path and diagnostics

This is the area reported as problematic in ReGlyco, so the rs design is worth
keeping even though the implementation is being replaced.

rs `reglyco-ensemble::gpu` exposes a small control surface:

- `configure(backend: &str)` sets the requested backend for the session.
- `set_progress(callback: impl Fn(&str) + 'static)` publishes the backend that
  is actually running, so the UI can label a CPU fallback honestly.
- `finish() -> serde_json::Value` emits the final `ComputeReport`.

Behaviours encoded in rs that a rewrite should not lose:

- An explicitly requested `webgpu` that is unavailable is a hard error ("explicit
  WebGPU scoring is unavailable: {reason}"). Falling back silently would
  misreport what actually ran.
- `auto` may fall back to CPU, and records the reason.
- Result arity is checked before indexing: GPU score count versus candidate
  count, gradient presence, and gradient atom count versus prepared topology.
- The backend classification is only final after the async session completes,
  so `energy.csv` is regenerated afterwards. JSON and CSV must carry the same
  actual execution label; this is the `actualBackend` plumbing in
  `reglyco-wasm`.
- `reglyco-ensemble/src/geometry_gpu.rs` reports precise internal errors naming
  the site, rotamer slot, conformer and offset when a GPU attachment-library
  lookup fails, instead of indexing blindly.

## 8. Experiments harness (rs only, tracked)

rs tracked `experiments/bench.sh`, `experiments/EXPERIMENTS.md` and
`experiments/results.tsv`, an append-only density/search benchmark log. This is
the most valuable non-code asset on the rs line and does not exist in ReGlyco.

- `bench.sh` runs `reglyco refine --post-relax none --report --quiet` with no
  time limit by default, over the 5KZC single-site and 5GSQ multi-site fixtures,
  and appends a summary line to `results.tsv`.
- `results.tsv` columns: `run, status, wall_s, cc, evals, stopping, sites_rmsd,
  valid, n_errors`.
- Reference runs recorded in `EXPERIMENTS.md`:

| name | CC | full RMSD (A) | stopping | evals | opt s |
| --- | --- | --- | --- | --- | --- |
| `example-output/5kzc-adaptive-current` | 0.5980 | 3.445 | emergency_time_limit (300 s) | 7,794 | 326.9 |
| `/tmp/5kzc-auto-after` | 0.7482 | 1.683 | phase_complete | 28,423 | 245.8 |
| `/tmp/5gsq-A-correct-v2` | 0.561 | 4.92 (A:297) | n/a | n/a | n/a |
| `/tmp/5gsq-single-B-current` | 0.316 | 8.81 (B:297) | n/a | n/a | n/a |

- Key finding logged there: the capped 5KZC run and the uncapped run reach the
  same broad-PSO state (CC 0.59797, 6,656 evals, 7 retained); the uncapped run
  then spent 28,422 adaptive-local evals and 186.7 s climbing to CC 0.7482, while
  the capped run was cut off after 13 adaptive-local evals. The conclusion was
  that the regression was time-budget starvation, not a scoring or algorithm
  change. Re-run this harness before trusting any performance claim.

## 9. Provenance notes

- `docs/emodelg-cache.json` (rs only) records that EModelG was deliberately not
  vendored: URL `https://www.cryoseek.org.cn/EModelG_v0.zip`, retrieved
  `2026-08-02`, sha256 `b4b2aebb...`, and the note "not vendored; clean-room
  algorithmic notes only".
- The four `docs/*density*.md` files are byte-identical in both repos, as are
  `crates/reglyco-workflow/{tests,test-data}`. There is nothing to port there.

## 10. Local hacks on the rs line: do not carry over

- `crates/reglyco-refine/src/lib.rs` hardcodes the absolute path
  `/home/opc/Re-Glyco-rs` as a test fixture root, repeated about eight times,
  where ReGlyco correctly derives it from `CARGO_MANIFEST_DIR`. Never port the
  hardcoded form.
- Seven dated `browser-release-*` clone directories, roughly 132 MB each, were
  left in the repo root by per-attempt WASM builds. They are unreferenced build
  output and are not tracked in either repo. ReGlyco should keep exactly one
  gitignored `browser-release/` and use `REGLYCO_LOCAL_RELEASE_DIR` plus
  `REGLYCO_ONLY_VARIANT` instead of cloning.
- `validation-report.json` is tracked at the rs repo root as scratch output. It
  is untracked in ReGlyco. Do not adopt it.
- `example-output/` and `.reglyco-cache/` are gitignored scratch.

## 11. Where ReGlyco is already ahead of rs

Do not regress these while rewriting:

- Block and coordinated proposals with incremental steric screening (section 4).
- Search-budget resolution promoted into `reglyco-core` and published in the
  report contract; rs only exposes `searchBudgetMode`.
- Report and lifecycle fields ReGlyco has and rs lacks: `complete_output`,
  `vmm_gate_satisfied`, `termination_reason`, `clash_partners`, and
  `search_budget`.
- Joint-feasibility diagnostics: `compatibility_algorithm`,
  `compatibility_seeded_states`, `compatibility_pose_attempts`,
  `compatibility_pool_sizes`, `compatibility_pool_expansions`.
- Multi-site coverage work: `propagate_domains`, `conformer_reach`,
  `coverage_conformer_index`, `compatible_pool_target`, and the tetramer
  coverage tests `tetramer_coverage_pool_finds_joint_starts`,
  `tetramer_ensemble_output_is_complete_and_sterically_valid` and
  `steric_coverage_sampling_does_not_hide_rare_conformers`.
- AGPL plus commercial licensing and `docs/release.md`; rs has neither.

## Suggested rewrite order

1. Preserve the statistical model exactly. `target()` and the prior cache are
   already correct and identical across both lines.
2. Fix the transition kernel first (block proposals plus incremental steric),
   then re-measure with the ported `experiments/` harness before touching
   budgets.
3. Port the Auto-budget estimator, decide on estimator versioning, and decide
   the strict-failure `primary_structure` rule (section 5) explicitly.
4. Rebuild the GPU control surface from section 7 rather than patching the
   existing one, since that is where the reported failures concentrate.
