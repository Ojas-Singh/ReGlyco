#!/usr/bin/env bash
# Reproducible cached density benchmarks.  This script intentionally does not
# set a time/evaluation cutoff: pass one explicitly in REGLYCO_EXTRA_ARGS when
# a bounded diagnostic run is desired.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_root}/example-output/density-benchmarks}"
shift || true

threads="${RAYON_NUM_THREADS:-4}"
extra_args=()
if [[ -n "${REGLYCO_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${REGLYCO_EXTRA_ARGS})
fi

if (($# == 0)); then
    cases=(5kzc 5gsq-a 5gsq-b)
else
    cases=("$@")
fi

mkdir -p "$output_root"

run_case() {
    local name="$1"
    local output="$output_root/$name"
    local command_file="$output/command.txt"
    local time_file="$output/time.txt"
    mkdir -p "$output"

    local -a case_args=(
        --assembly 1
        --objective density
        --density-map auto
        --density-map-source pdbe
        --density-difference-map none
        --density-effort adaptive
        --post-relax none
        --report
        --output "$output"
        --overwrite
    )
    case "$name" in
        5kzc)
            case_args+=(--pdb-id 5KZC --replace-glycan A:79=G63337SS --anomer beta --level 3)
            ;;
        5gsq-a)
            case_args+=(--pdb-id 5GSQ --replace-glycan A:297)
            ;;
        5gsq-b)
            case_args+=(--pdb-id 5GSQ --replace-glycan B:297)
            ;;
        *)
            printf 'unknown benchmark case: %s\n' "$name" >&2
            return 2
            ;;
    esac

    {
        printf 'RAYON_NUM_THREADS=%q cargo run --release --manifest-path %q -- refine' \
            "$threads" "$repo_root/Cargo.toml"
        printf ' %q' "${case_args[@]}" "${extra_args[@]}"
        printf '\n'
    } | tee "$command_file"

    /usr/bin/time -f 'wall_seconds=%e\nuser_seconds=%U\nsys_seconds=%S\nmax_rss_kb=%M' \
        -o "$time_file" \
        env RAYON_NUM_THREADS="$threads" \
        cargo run --release --manifest-path "$repo_root/Cargo.toml" -- refine \
        "${case_args[@]}" "${extra_args[@]}"

    printf 'completed %s; outputs: %s\n' "$name" "$output"
}

for case in "${cases[@]}"; do
    run_case "$case"
done

printf '\nBenchmark records are in %s/<case>/command.txt, time.txt, and density.json.\n' \
    "$output_root"
