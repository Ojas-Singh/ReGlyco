// Exhaustive bit check of glycoflow-core's four-lane activations (vmath) against the scalar calls,
// in wasm32 simd128. Build crates/glycoflow-core/examples/vmath_check.rs first (see its doc comment):
//   node scripts/perf/vmath_check.mjs <vmath_check.wasm>
// Prints one line per check and exits 1 on any mismatch (all 2^32 patterns take ~2 minutes).
import fs from 'node:fs';

const bytes = fs.readFileSync(process.argv[2]);
// the module links getrandom's wasm-bindgen glue, which these checks never call: stub every import
const stub = new Proxy({}, { get: () => () => { throw new Error('unexpected import call'); } });
const imports = Object.fromEntries(WebAssembly.Module.imports(new WebAssembly.Module(bytes)).map((i) => [i.module, stub]));
const { instance } = await WebAssembly.instantiate(bytes, imports);
const { check, check_mixed } = instance.exports;

let failed = false;
for (const [kind, name] of [[0, 'exp'], [1, 'silu'], [2, 'gelu']]) {
  const t = performance.now();
  let bad = 0n;
  const step = 2 ** 26;
  for (let lo = 0; lo < 2 ** 32; lo += step) bad += check(kind, lo >>> 0, (lo + step) % 2 ** 32);
  const mixed = check_mixed(kind, 11, 30_000_000);
  failed ||= bad !== 0n || mixed !== 0n;
  console.log(JSON.stringify({ kind: name, all_patterns_mismatches: Number(bad), mixed_mismatches: Number(mixed), seconds: +((performance.now() - t) / 1000).toFixed(1) }));
}
process.exit(failed ? 1 : 0);
