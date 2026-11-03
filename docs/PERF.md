# GlycoFlow sampling performance (browser engine)

Where the time of a GlycoFlow ensemble goes in the browser, what was changed on branch
`perf/sampling` (glycoflow-core), and what is left. Every change keeps the output **bit-identical**:
the same seed gives the same bytes before and after, in wasm and natively.

## The call that was measured

The SugarGLIDER page (GlycoShapeCE `web/src/pages/SugarGliderPage.tsx`) samples the glycan's free
ensemble before docking: Quick = 128 conformers, seed 1, 32 Heun steps (63 network evaluations per
conformer), clash guidance 0.3 from t = 0.5, 16 pucker templates, `ATOM_BUDGET` 200 000. The run is
split over `min(8, cores - 1)` module workers, each a single-threaded wasm instance of
`crates/glycoflow-wasm` (CE) running `Engine::sample_slice`, then worker 0 superposes and writes
the PDB (`assemble`) and computes rarity. With 8 workers one worker samples 16 conformers, and the
user waits on the slowest worker, so the benchmark is one worker's slice: `sample(seq, 1, 128, 0,
16, 32, 'heun', 0.3, 0.5, 16, 200000)` in Node 22, single-threaded, `+simd128`, the same flags as
CE's `scripts/build-glycoflow-wasm.sh`.

Glycans: G1 = the 1K9I Quick-dock example, `DGlcpNAcb1-2DManpa1-3[DGlcpNAcb1-2DManpa1-6]DManpa1-OH`
(5 sugars, 78 atoms, 29 torsions, 5 rings); G2 = `DGlcpNAcb1-2DManpa1-3[DGlcpNAcb1-2DManpa1-6]
DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH` (8 sugars, 124 atoms, 48 torsions). Model: the v6
release (HF revision 365e37e). Machine: the lab's DGX Spark (GB10, Arm), one core.

Harness: `scripts/perf/wasm_bench.mjs` (times `sample`, `assemble`, `member_rarity`, writes the raw
output for byte comparison). The bundle is CE's `crates/glycoflow-wasm` copied out of CE with its
`glycoflow-core` path pointed at this checkout, built with
`RUSTFLAGS='-C target-feature=+simd128 --cfg getrandom_backend="wasm_js"' cargo build --lib
--target wasm32-unknown-unknown --release` and `wasm-bindgen --target web`. The baseline bundle built
from `feat/ome-aglycone` (6c1edfd) gives the same bytes and the same time as the bundle committed in
CE (d852b9b). Profiles: `node --cpu-prof` (wasm function names are kept in the bundle).

## Where the time goes

G1, one worker (16 conformers = 63 batched network calls of 16), wasm, before -> after:

| part | before (s) | share | after (s) |
|---|---:|---:|---:|
| network: linear-layer matmuls (candle -> `gemm` simd128 12x4 kernel) | 19.0 | 45 % | 19.0 |
| network: attention and other matmuls | 1.4 | 3 % | 1.3 |
| network: bias + GELU (feed-forward) and SiLU (`BiasAct`) | 11.4 | 27 % | 4.0 |
| network: pair-bias MLP from distances (`PairBias`) | 5.5 | 13 % | 2.3 |
| network: attention softmax | 2.9 | 7 % | 1.4 |
| network: layer norms | 0.75 | 2 % | 0.17 |
| network: other candle ops, allocation (torque head, head split, gated add) | 0.8 | 2 % | 0.5 |
| builder (ring shapes, torsions each step), clash guidance, repair (`tidy`) | < 0.05 | < 0.1 % | same |
| `assemble` (superpose, PDB text) | 0.007 | | same |
| rarity | < 0.001 | | same |
| JS <-> wasm marshalling (one Float32Array back) | ~0 | | same |
| **total `sample`** (profiled runs) | **41.8** | | **28.9** |

The network is all of it. Within it, half was not matrix multiplication: the scalar libm `expf`
and `erff` (softmax, the RBF features and SiLU of the pair MLP, the GELU of every feed-forward
layer) took about 30 % of the run on their own (GELU alone: 0.64 billion `erff` calls).

## Changes (all bit-identical)

1. `vmath.rs` (new): four-lane wasm simd128 versions of `exp`, SiLU and GELU-erf that execute the
   scalar algorithms -- FreeBSD/musl `expf` (what `f32::exp` lowers to on wasm32, from
   compiler-builtins' libm) and libm's `erff` (behind candle's `erf_f32`) -- operation for
   operation in f32 (wasm has no fused multiply-add, so `a * b + c` rounds twice in both), each
   branch selected per lane, NaN/overflow lanes falling back to the scalar call. GELU sorts its
   lanes by the `erff` branch they take (SIMD left-pack) so every vector evaluates one branch only.
   Native targets keep the scalar loops (native `f32::exp` is glibc's, a different algorithm).
2. `kernels.rs`: the CPU kernels call those slice functions (bias + activation, softmax, the pair
   MLP's RBF and SiLU); the pair MLP keeps its 32/64 accumulators in registers (same order of
   operations per output) and writes the layer-major output directly instead of through a scratch
   copy; layer norm and softmax sum four rows interleaved (each row still in its own order, from
   -0.0 as `Iterator::sum`), which hides the latency of the dependent additions; the softmax row
   maximum is taken four-wide (the maximum does not depend on order; a zero's sign is lost in
   `x - max`); the bias add no longer takes `k % c` per element.

## Results

Wasm, one worker's slice (16 of 128), two runs per arm, alternating:

| glycan | before run 1 / 2 (s) | after run 1 / 2 (s) | speed-up |
|---|---:|---:|---:|
| G1 (78 atoms), 16 of 128 | 41.23 / 41.50 | 28.34 / 28.42 | 1.46x (-31 %) |
| G2 (124 atoms), 16 of 128 | 74.55 / 74.83 | 49.64 / 50.27 | 1.49x (-33 %) |
| G1, all 128 in one worker (1 run) | 331.6 | 229.3 | 1.45x |

On the page with 8 workers the ~39 s glycan step of a Quick dock should drop to about 27 s (every
worker runs the same slice size); with fewer cores (fewer workers) the saving is the same fraction.
`assemble` (7 ms) and rarity (< 1 ms) were never significant.

Native (aarch64, one thread, same slice): G1 23.2 -> 21.2 s, G2 44.3 -> 41.5 s, the sialylated trisaccharide 16.3 -> 15.6 s. Natively only the layer-norm/softmax
interleaving and the pair-MLP output layout apply (`exp` stays glibc's).

## Bit-identity evidence

- Byte comparison of the raw `sample()` output (coordinates, torsions, validity, ring shapes) and
  the `assemble()` JSON (multi-model PDB, clash counts), before vs after, in wasm: G1 slices 0-15 and 112-127 and the whole 128-run in one call; G2;
`DNeup5Aca2-3DGalpb1-4DGlcpNAcb1-OH`; Man5 (`DManpa1-6[DManpa1-3]DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4DGlcpNAcb1-OH`,
8 of 128); core 1 `DGalpb1-3DGalpNAca1-OH`; `LFucpa1-2DGalpb1-4DGlcpNAcb1-OH`: all identical (G1 and G2
again in each timing run). Before/after times of these: Neu5Ac trisaccharide 29.3 -> 20.5 s, Man5 (8
conformers) 30.2 -> 20.4 s, core 1 15.9 -> 11.3 s, Fuc-LacNAc 22.3 -> 15.8 s.
- Natively (`Engine::sample_slice`, one thread), before vs after: G1, G2 and the Neu5Ac trisaccharide (16 of 128 each): identical.
- The four-lane functions against the scalar calls on all 2^32 f32 bit patterns (exp, erf; the exp,
  SiLU and GELU slice functions) plus 30 M mixed-magnitude values each (lanes of one vector taking
  different branches): 0 mismatches. Rerun with `crates/glycoflow-core/examples/vmath_check.rs` and
  `scripts/perf/vmath_check.mjs` after any change to `vmath.rs`.
- `cargo test --release -p glycoflow-core` and the parity tests (`--ignored` with
  `GLYCOFLOW_RUST_DIR` = GlycoFlow `rust/` and `rust_rings/`): all pass.

## Not adopted (numerically different), and what is left

- **Fused multiply-add in the matmuls** (relaxed-SIMD `f32x4.relaxed_madd` in gemm's simd128
  kernel, `+relaxed-simd`, measured on an intermediate build of this branch): G1 slice 29.5 s ->
  22.3 s (-24 %); coordinates move by at most 6e-5 A on G1 (93 % of the values change in their last
  bits). Not adopted: it changes the output, and `relaxed_madd` may or may not fuse
  depending on the CPU and browser, so results would differ between visitors' machines.
- The linear-layer matmuls are now ~70 % of the run and run at about 50 GFLOP/s, close to what
  separate simd128 multiplies and adds allow on this core; gemm's blocking fixes the summation
  order, so a faster kernel would have to reproduce it exactly to stay bit-identical.
- Bigger savings need a numerical or model change: fewer network evaluations (fewer steps, Euler
  instead of Heun: 63 -> 32 calls), a smaller network (dim 256, depth 8), f16/bf16 weights, or the
  WebGPU path the page already has. On a many-core machine the page could also use more than 8
  workers (CE `defaultWorkers`), which needs no change here.
