#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="${1:-${workspace_dir}/browser-release}"
if [[ -n "${REGLYCO_RELEASE_VERSION:-}" ]]; then
  version="${REGLYCO_RELEASE_VERSION}"
else
  version="$(git -C "${workspace_dir}" describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')"
  if [[ -z "${version}" ]]; then
    version="$(cargo metadata --no-deps --format-version 1 --manifest-path "${workspace_dir}/Cargo.toml" | sed -n 's/.*"name":"reglyco-wasm","version":"\([^"]*\)".*/\1/p')"
  fi
fi
export REGLYCO_RELEASE_VERSION="${version}"
source_commit="$(git -C "${workspace_dir}" rev-parse HEAD)"
if [[ -n "$(git -C "${workspace_dir}" status --porcelain)" ]]; then source_dirty=true; else source_dirty=false; fi
if [[ "${source_dirty}" == true ]]; then
  echo "Refusing to build a browser release from a dirty ReGlyco worktree." >&2
  exit 1
fi
rustc_version="$(rustc --version)"
wasm_bindgen_version="$(wasm-bindgen --version)"

build_variant() {
  variant="$1"
  features="$2"
  rustflags="${3:-}"
  stage_dir="${release_dir}/${variant}"
  target_dir="${workspace_dir}/target/browser-release-cache/${variant/full-gpu/full}"
  mkdir -p "${stage_dir}"
  if [[ -n "${rustflags}" ]]; then
    CARGO_TARGET_DIR="${target_dir}" RUSTFLAGS="${rustflags}" cargo build --manifest-path "${workspace_dir}/Cargo.toml" --release --target wasm32-unknown-unknown -p reglyco-wasm --no-default-features --features "${features}"
  else
    CARGO_TARGET_DIR="${target_dir}" cargo build --manifest-path "${workspace_dir}/Cargo.toml" --release --target wasm32-unknown-unknown -p reglyco-wasm --no-default-features --features "${features}"
  fi
  wasm-bindgen "${target_dir}/wasm32-unknown-unknown/release/reglyco_wasm.wasm" --target web --out-dir "${stage_dir}" --out-name reglyco
}

build_threaded_variant() {
  variant="$1"
  features="$2"
  stage_dir="${release_dir}/${variant}"
  target_dir="${workspace_dir}/target/browser-release-cache/${variant/full-gpu/full}"
  mkdir -p "${stage_dir}"
  CARGO_TARGET_DIR="${target_dir}" rustup run nightly-2026-07-15 cargo build -Z build-std=panic_abort,std \
    --config 'target.wasm32-unknown-unknown.rustflags=["-C","target-feature=+atomics,+bulk-memory,+mutable-globals","-C","link-arg=--shared-memory","-C","link-arg=--import-memory","-C","link-arg=--max-memory=4294967296","-C","link-arg=--export=__wasm_init_tls","-C","link-arg=--export=__tls_size","-C","link-arg=--export=__tls_align","-C","link-arg=--export=__tls_base","-C","link-arg=--export=__heap_base","-C","link-arg=--export=__data_end","-C","link-arg=--export=__stack_pointer","--cfg","getrandom_backend=\"custom\""]' \
    --manifest-path "${workspace_dir}/Cargo.toml" --release --target wasm32-unknown-unknown \
    -p reglyco-wasm --no-default-features --features "${features}"
  wasm-bindgen "${target_dir}/wasm32-unknown-unknown/release/reglyco_wasm.wasm" --target web --out-dir "${stage_dir}" --out-name reglyco
  # wasm-bindgen-rayon assumes a bundler can resolve the package directory.
  # Release files are served directly, so point nested workers at the
  # concrete ESM entry instead of a directory URL that an SPA serves as HTML.
  while IFS= read -r helper; do
    sed -i "s|import('../../..')|import('../../../reglyco.js')|" "${helper}"
  done < <(find "${stage_dir}/snippets" -name workerHelpers.js -type f)
}

mkdir -p "${release_dir}"
only_variant="${REGLYCO_ONLY_VARIANT:-all}"
gpu_enabled=false
if [[ "${REGLYCO_SITE_PROFILE:-}" == "development" && "${REGLYCO_WEBGPU:-}" == "true" ]]; then gpu_enabled=true; fi
if [[ "${only_variant}" == full-gpu-* && "${gpu_enabled}" != true ]]; then
  echo "GPU artifacts require REGLYCO_SITE_PROFILE=development and REGLYCO_WEBGPU=true" >&2
  exit 1
fi
[[ "${only_variant}" == "all" || "${only_variant}" == "public-single" ]] && build_variant public-single public
[[ "${only_variant}" == "all" || "${only_variant}" == "public-threaded" ]] && build_threaded_variant public-threaded 'public,threaded'
[[ "${only_variant}" == "all" || "${only_variant}" == "full-single" ]] && build_variant full-single full
[[ "${only_variant}" == "all" || "${only_variant}" == "full-threaded" ]] && build_threaded_variant full-threaded 'full,threaded'
if [[ "${gpu_enabled}" == true ]]; then
  [[ "${only_variant}" == "all" || "${only_variant}" == "full-gpu-single" ]] && build_variant full-gpu-single 'full,webgpu'
  [[ "${only_variant}" == "all" || "${only_variant}" == "full-gpu-threaded" ]] && build_threaded_variant full-gpu-threaded 'full,webgpu,threaded'
fi
if [[ "${only_variant}" == "all" || "${only_variant}" == "pdf" ]]; then
  pdf_target_dir="${workspace_dir}/target/browser-release-cache/pdf"
  CARGO_TARGET_DIR="${pdf_target_dir}" cargo build --manifest-path "${workspace_dir}/Cargo.toml" --release --target wasm32-unknown-unknown -p reglyco-pdf-wasm
  mkdir -p "${release_dir}/pdf"
  wasm-bindgen "${pdf_target_dir}/wasm32-unknown-unknown/release/reglyco_pdf_wasm.wasm" --target web --out-dir "${release_dir}/pdf" --out-name reglyco_pdf
fi
cp "${workspace_dir}/contracts/reglyco-v1.d.ts" "${release_dir}/reglyco-v1.d.ts"

checksum() { sha256sum "$1" | cut -d ' ' -f 1; }
public_js="$(checksum "${release_dir}/public-single/reglyco.js")"
public_wasm="$(checksum "${release_dir}/public-single/reglyco_bg.wasm")"
public_threaded_js="$(checksum "${release_dir}/public-threaded/reglyco.js")"
public_threaded_wasm="$(checksum "${release_dir}/public-threaded/reglyco_bg.wasm")"
full_js="$(checksum "${release_dir}/full-single/reglyco.js")"
full_wasm="$(checksum "${release_dir}/full-single/reglyco_bg.wasm")"
threaded_js="$(checksum "${release_dir}/full-threaded/reglyco.js")"
threaded_wasm="$(checksum "${release_dir}/full-threaded/reglyco_bg.wasm")"
pdf_js="$(checksum "${release_dir}/pdf/reglyco_pdf.js")"
pdf_wasm="$(checksum "${release_dir}/pdf/reglyco_pdf_bg.wasm")"

printf '%s\n' \
  '{' \
  "  \"version\": \"${version}\"," \
  '  "schemaVersion": 1,' \
  '  "contracts": "reglyco-v1.d.ts",' \
  '  "source": {' \
  '    "repository": "https://github.com/Ojas-Singh/ReGlyco",' \
  "    \"commit\": \"${source_commit}\"," \
  "    \"dirty\": ${source_dirty}" \
  '  },' \
  '  "toolchain": {' \
  "    \"rustc\": \"${rustc_version}\"," \
  "    \"wasmBindgen\": \"${wasm_bindgen_version}\"" \
  '  },' \
  '  "artifacts": {' \
  "    \"publicSingle\": { \"js\": \"public-single/reglyco.js\", \"jsSha256\": \"${public_js}\", \"wasm\": \"public-single/reglyco_bg.wasm\", \"wasmSha256\": \"${public_wasm}\" }," \
  "    \"publicThreaded\": { \"js\": \"public-threaded/reglyco.js\", \"jsSha256\": \"${public_threaded_js}\", \"wasm\": \"public-threaded/reglyco_bg.wasm\", \"wasmSha256\": \"${public_threaded_wasm}\" }," \
  "    \"fullSingle\": { \"js\": \"full-single/reglyco.js\", \"jsSha256\": \"${full_js}\", \"wasm\": \"full-single/reglyco_bg.wasm\", \"wasmSha256\": \"${full_wasm}\" }," \
  "    \"fullThreaded\": { \"js\": \"full-threaded/reglyco.js\", \"jsSha256\": \"${threaded_js}\", \"wasm\": \"full-threaded/reglyco_bg.wasm\", \"wasmSha256\": \"${threaded_wasm}\" }," \
  "    \"pdfRenderer\": { \"js\": \"pdf/reglyco_pdf.js\", \"jsSha256\": \"${pdf_js}\", \"wasm\": \"pdf/reglyco_pdf_bg.wasm\", \"wasmSha256\": \"${pdf_wasm}\" }" \
  '  }' \
  '}' > "${release_dir}/manifest.json"

if [[ "${gpu_enabled}" == true ]]; then
  python3 - "${release_dir}" <<'PYGPU'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);p=root/'manifest.json';m=json.loads(p.read_text())
for variant,key in [('full-gpu-single','fullGpuSingle'),('full-gpu-threaded','fullGpuThreaded')]:
    js=root/variant/'reglyco.js';wasm=root/variant/'reglyco_bg.wasm'
    if js.exists() and wasm.exists():
        m['artifacts'][key]={'js':str(js.relative_to(root)),'jsSha256':hashlib.sha256(js.read_bytes()).hexdigest(),'wasm':str(wasm.relative_to(root)),'wasmSha256':hashlib.sha256(wasm.read_bytes()).hexdigest()}
p.write_text(json.dumps(m,indent=2)+'\n')
PYGPU
fi
echo "Browser release ${version} written to ${release_dir}"
