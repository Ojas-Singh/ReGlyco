#!/usr/bin/env bash
# GlycoFlow density-fitting benchmark: runs `reglyco refine --objective density` on a list of
# sites and seeds, recording the log, wall time and peak memory of every run, and writes a
# summary table (summary.tsv) with the recovery against the deposited glycan (evaluation only)
# and the validation result.
#
# Usage: scripts/benchmark-glycoflow.sh [OUTPUT_DIR] [SITES_FILE]
#
#   SITES_FILE   lines "NAME MODEL MAP SITE[=GLYCAM]" ('#' comments allowed); default: the
#                3 X-ray + 10 cryo-EM sites below, read from $REGLYCO_DATA
#                (structures/, maps/, em/; default ~/reglyco-data)
#
# Environment:
#   GLYCOFLOW_MODEL     GlycoFlow model directory (required)
#   REGLYCO_BIN         reglyco binary (default: target/release/reglyco; build with
#                       --features reglyco-cli/glycoflow-cuda for the GPU)
#   GLYCOFLOW_DEVICE    cpu (default) or cuda
#   SEEDS               seeds per site (default: "0 1 2")
#   REGLYCO_EXTRA_ARGS  extra refine arguments (e.g. "--glycoflow-samples 512")
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_root}/example-output/glycoflow-benchmarks}"
sites_file="${2:-}"
bin="${REGLYCO_BIN:-${repo_root}/target/release/reglyco}"
device="${GLYCOFLOW_DEVICE:-cpu}"
seeds="${SEEDS:-0 1 2}"
data="${REGLYCO_DATA:-${HOME}/reglyco-data}"
extra_args=()
if [[ -n "${REGLYCO_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${REGLYCO_EXTRA_ARGS})
fi

if [[ -z "${GLYCOFLOW_MODEL:-}" ]]; then
    echo "set GLYCOFLOW_MODEL to the GlycoFlow model directory" >&2
    exit 2
fi
if [[ ! -x "$bin" ]]; then
    echo "reglyco binary not found: $bin (cargo build --release, or set REGLYCO_BIN)" >&2
    exit 2
fi

default_sites() {
    cat <<LIST
5KZC_A79 ${data}/structures/5KZC.pdb ${data}/maps/eds-5kzc.ccp4 A:79
5GSQ_A297 ${data}/structures/5GSQ.pdb ${data}/maps/eds-5gsq.ccp4 A:297
5GSQ_B297 ${data}/structures/5GSQ.pdb ${data}/maps/eds-5gsq.ccp4 B:297
9AVV_C161 ${data}/em/9AVV.pdb ${data}/em/emd_43924.map.gz C:161
9AVV_A161 ${data}/em/9AVV.pdb ${data}/em/emd_43924.map.gz A:161
9RGD_A111 ${data}/em/9RGD.pdb ${data}/em/emd_53948.map.gz A:111
9RGD_B149 ${data}/em/9RGD.pdb ${data}/em/emd_53948.map.gz B:149
6UDJ_G332 ${data}/em/6UDJ.pdb ${data}/em/emd_20739.map.gz G:332
6UDJ_G276 ${data}/em/6UDJ.pdb ${data}/em/emd_20739.map.gz G:276
9EA0_A761 ${data}/em/9EA0.pdb ${data}/em/emd_47823.map.gz A:761
9MJ1_A89 ${data}/em/9MJ1.pdb ${data}/em/emd_48306.map.gz A:89
9MJ1_A109 ${data}/em/9MJ1.pdb ${data}/em/emd_48306.map.gz A:109
9MJ1_a365 ${data}/em/9MJ1.pdb ${data}/em/emd_48306.map.gz a:365
LIST
}

mkdir -p "$output_root"
summary="$output_root/summary.tsv"
printf 'name\tseed\texit\twall_seconds\tmax_rss_kb\tfull_rmsd\tcore_rmsd\tvalid\n' > "$summary"

time_cmd=()
if [[ -x /usr/bin/time ]]; then
    time_cmd=(/usr/bin/time -f $'wall_seconds=%e\nmax_rss_kb=%M' -o)
fi

run_case() {
    local name="$1" model="$2" map="$3" site="$4" seed="$5"
    local output="$output_root/${name}_seed${seed}"
    mkdir -p "$output"
    local -a args=(
        refine --protein "$model" --density-map "$map" --objective density
        --replace-glycan "$site" --glycoflow-device "$device" --seed "$seed"
        --output "$output" --overwrite "${extra_args[@]}"
    )
    printf '%q ' "$bin" "${args[@]}" > "$output/command.txt"
    printf '\n' >> "$output/command.txt"
    local status=0
    local start=$SECONDS
    if ((${#time_cmd[@]})); then
        "${time_cmd[@]}" "$output/time.txt" "$bin" "${args[@]}" > "$output/log.txt" 2>&1 || status=$?
    else
        "$bin" "${args[@]}" > "$output/log.txt" 2>&1 || status=$?
        printf 'wall_seconds=%s\n' "$((SECONDS - start))" > "$output/time.txt"
    fi
    python3 - "$output" "$name" "$seed" "$status" >> "$summary" <<'PY'
import json, os, sys
out, name, seed, status = sys.argv[1:5]
times = {}
try:
    for line in open(os.path.join(out, "time.txt")):
        if "=" in line:
            key, value = line.strip().split("=", 1)
            times[key] = value
except OSError:
    pass
full = core = valid = ""
try:
    fit = json.load(open(os.path.join(out, "glycoflow-fit.json")))
    recovery = ((fit["sites"][0].get("evaluation") or {}).get("recovery")) or {}
    full = f"{recovery['full_rmsd']:.3f}" if "full_rmsd" in recovery else ""
    core = f"{recovery['core_rmsd']:.3f}" if "core_rmsd" in recovery else ""
    valid = str((fit.get("validation") or {}).get("valid", ""))
except (OSError, ValueError, KeyError, IndexError):
    pass
print("\t".join([name, seed, status, times.get("wall_seconds", ""), times.get("max_rss_kb", ""), full, core, valid]))
PY
    printf '%s seed %s: exit %s\n' "$name" "$seed" "$status"
}

while read -r name model map site; do
    [[ -z "${name}" || "${name}" == \#* ]] && continue
    for seed in $seeds; do
        run_case "$name" "$model" "$map" "$site" "$seed"
    done
done < <(if [[ -n "$sites_file" ]]; then cat "$sites_file"; else default_sites; fi)

column -t -s $'\t' "$summary" 2>/dev/null || cat "$summary"
