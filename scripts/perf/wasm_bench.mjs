// GLYCOFLOW_MODEL=<model dir> node scripts/perf/wasm_bench.mjs <bundle dir> <sequence> <total> <start> <count> <out prefix|-> [repeats]
// <bundle dir>: wasm-bindgen --target web output of GlycoShapeCE's crates/glycoflow-wasm built against this
// checkout (see docs/PERF.md). Prints timings and a digest of the output; writes <out>.bin (the raw
// sample() result) and <out>.json (assemble()) for byte comparison between builds.
// Reproduces one GlycoFlow worker's 'sample' request of the SugarGLIDER page (Quick: 32 Heun steps,
// guidance 0.3 from t=0.5, 16 pucker templates, seed 1, ATOM_BUDGET 200000), then assemble + rarity.
import fs from 'node:fs';
import crypto from 'node:crypto';
const [dir, seq, total, start, count, out, repeats = '1'] = process.argv.slice(2);
const mod = await import(new URL(`${dir}/glycoflow_wasm.js`, `file://${process.cwd()}/`).href);
mod.initSync({ module: fs.readFileSync(`${dir}/glycoflow_wasm_bg.wasm`) });
const M = process.env.GLYCOFLOW_MODEL;
if (!M) throw new Error('set GLYCOFLOW_MODEL to the model directory');
let t = performance.now();
const engine = new mod.GlycoFlow(fs.readFileSync(`${M}/glycoflow.safetensors`), fs.readFileSync(`${M}/glycoflow.json`, 'utf8'), fs.readFileSync(`${M}/residue_library.json`, 'utf8'));
const load = (performance.now() - t) / 1000;
const desc = JSON.parse(engine.describe(seq));
for (let rep = 0; rep < +repeats; rep++) {
  t = performance.now();
  const data = engine.sample(seq, 1, +total, +start, +count, 32, 'heun', 0.3, 0.5, 16, 200000, () => {});
  const sample = (performance.now() - t) / 1000;
  const n = desc.atoms, nt = desc.torsions.length;
  const coords = data.slice(0, +count * n * 3);
  const torsions = data.slice(+count * n * 3, +count * (n * 3 + nt));
  t = performance.now();
  const asm = engine.assemble(seq, coords, 1);
  const assemble = (performance.now() - t) / 1000;
  t = performance.now();
  const rar = mod.member_rarity(torsions, nt);
  const rarity = (performance.now() - t) / 1000;
  const h = crypto.createHash('sha256');
  h.update(Buffer.from(data.buffer, data.byteOffset, data.byteLength)); h.update(asm); h.update(Buffer.from(rar.buffer, rar.byteOffset, rar.byteLength));
  const digest = h.digest('hex').slice(0, 16);
  console.log(JSON.stringify({ bundle: dir, rep, atoms: n, torsions: nt, count: +count, load: +load.toFixed(2), sample: +sample.toFixed(2), assemble: +assemble.toFixed(3), rarity: +rarity.toFixed(3), digest }));
  if (rep === 0 && out !== '-') {
    fs.writeFileSync(`${out}.bin`, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    fs.writeFileSync(`${out}.json`, asm);
  }
}
