#!/usr/bin/env bash
# Check that the workspace is ready to release VERSION, and package every
# crate that scripts/publish_crates.sh will upload. Nothing is uploaded.
#
#   scripts/check_release.sh [VERSION]                # default: [workspace.package] version
#   scripts/check_release.sh --dependencies VERSION   # only the crates.io check below
#
# ReGlyco builds against ../GlySys and ../crabSAXS by path and crabWURCS from
# git. The published crates take them from crates.io, so those releases must
# be in the crates.io index first; the check stops there otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() {
  echo "check_release: $*" >&2
  exit 1
}

dependencies_only=false
if [ "${1:-}" = --dependencies ]; then
  dependencies_only=true
  shift
fi

workspace_version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' Cargo.toml)
version=${1:-$workspace_version}
version=${version#v}

declare -A wanted position
crates=()
while read -r crate pinned; do
  position[$crate]=${#crates[@]}
  crates+=("$crate")
  wanted[$crate]=${pinned:-$version}
done < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' scripts/release-crates.txt)

index_url() {
  local name=$1
  case ${#name} in
    1) echo "https://index.crates.io/1/$name" ;;
    2) echo "https://index.crates.io/2/$name" ;;
    3) echo "https://index.crates.io/3/${name:0:1}/$name" ;;
    *) echo "https://index.crates.io/${name:0:2}/${name:2:2}/$name" ;;
  esac
}
in_index() {
  curl --fail --silent "$(index_url "$1")" |
    jq --exit-status --arg version "$2" 'select(.vers == $version)' >/dev/null
}

# Reading the manifests alone does not need the sibling checkouts.
manifests=$(cargo metadata --no-deps --format-version 1)
root=$(jq -r .workspace_root <<<"$manifests")

missing=()
while IFS=$'\t' read -r dep req; do
  dep_version=${req#^}
  dep_version=${dep_version#=}
  in_index "$dep" "$dep_version" || missing+=("$dep $dep_version")
done < <(jq -r --arg root "$root/" '.packages[] | select(.publish != []) | .dependencies[]
  | select(((.path // "") != "" and ((.path + "/") | startswith($root) | not))
      or ((.source // "") | startswith("git+")))
  | [.name, .req] | @tsv' <<<"$manifests" | sort -u)
if [ "${#missing[@]}" -gt 0 ]; then
  fail "not in the crates.io index yet: ${missing[*]}.
ReGlyco $version depends on these exact releases. Publish GlySys first (and crabSAXS
or crabWURCS if listed), wait until they are in the index, then run this again."
fi
echo "GlySys, crabSAXS and crabWURCS dependencies are on crates.io"
if $dependencies_only; then
  exit 0
fi

[ "$workspace_version" = "$version" ] || fail "[workspace.package] version is $workspace_version, not $version"
grep -q "^## $version " CHANGELOG.md || fail "CHANGELOG.md has no '## $version - DATE' section"
! grep -qi '^## Unreleased' CHANGELOG.md || fail "CHANGELOG.md still has an Unreleased section"
grep -qF "cargo install reglyco --locked --version $version" README.md ||
  fail "README.md does not install version $version"

# The release workflow checks out GlySys and crabSAXS at the tags of the
# versions the manifests require.
glysys_version=$(jq -r '[.packages[].dependencies[] | select(.name == "glysys") | .req][0] | ltrimstr("^")' <<<"$manifests")
crabsaxs_version=$(jq -r '[.packages[].dependencies[] | select(.name == "crabsaxs") | .req][0] | ltrimstr("^")' <<<"$manifests")
grep -q "^  GLYSYS_REF: v$glysys_version$" .github/workflows/release.yml ||
  fail "release.yml does not check out GlySys v$glysys_version"
grep -q "^  CRABSAXS_REF: \"$crabsaxs_version\"$" .github/workflows/release.yml ||
  fail "release.yml does not check out crabSAXS $crabsaxs_version"

metadata=$(cargo metadata --no-deps --format-version 1 --locked) ||
  fail "Cargo.lock is not up to date (cargo metadata --locked failed)"

# The list holds every publishable workspace member and nothing else.
listed=$(printf '%s\n' "${crates[@]}" | sort)
publishable=$(jq -r '.packages[] | select(.publish != []) | .name' <<<"$metadata" | sort)
[ "$listed" = "$publishable" ] ||
  fail "scripts/release-crates.txt does not list the publishable workspace members:
$(diff <(echo "$listed") <(echo "$publishable") || true)"

for crate in "${crates[@]}"; do
  package=$(jq -c --arg name "$crate" '.packages[] | select(.name == $name)' <<<"$metadata")
  crate_version=$(jq -r .version <<<"$package")
  [ "$crate_version" = "${wanted[$crate]}" ] || fail "$crate is at $crate_version, not ${wanted[$crate]}"
  [ "$(jq -r .license <<<"$package")" = "AGPL-3.0-only" ] || fail "$crate is not AGPL-3.0-only"
  grep -A1 "^name = \"$crate\"$" Cargo.lock | grep -qx "version = \"$crate_version\"" ||
    fail "Cargo.lock has no $crate $crate_version"
  if in_index "$crate" "$crate_version"; then
    echo "check_release: note: $crate $crate_version is already on crates.io (publish_crates.sh skips it)" >&2
  fi
  while IFS=$'\t' read -r dep kind req source; do
    if [ -n "${position[$dep]+set}" ]; then
      # A path-only dev-dependency is dropped from the published manifest.
      [ "$kind" = dev ] && [ "$req" = '*' ] && continue
      [ "$req" = "^${wanted[$dep]}" ] || fail "$crate requires $dep $req, not ^${wanted[$dep]}"
      [ "${position[$dep]}" -lt "${position[$crate]}" ] ||
        fail "$crate is listed before its dependency $dep in scripts/release-crates.txt"
    elif [[ "$source" == git+* ]]; then
      [ "$req" != '*' ] || fail "$crate takes $dep from git without a version for crates.io"
    fi
  done < <(jq -r '.dependencies[] | [.name, (.kind // "normal"), .req, (.source // "-")] | @tsv' <<<"$package")
done

if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "check_release: warning: the working tree has uncommitted changes; cargo publish will refuse it" >&2
fi

# Packaging the crates together lets Cargo verify each one against the others
# as they will be on crates.io (and against GlySys, crabSAXS and crabWURCS
# from crates.io), before any of them is uploaded.
package_args=()
for crate in "${crates[@]}"; do
  package_args+=(-p "$crate")
done
cargo package --locked --allow-dirty "${package_args[@]}"

target_dir=$(jq -r .target_directory <<<"$metadata")
for crate in "${crates[@]}"; do
  file="$target_dir/package/$crate-${wanted[$crate]}.crate"
  size=$(stat -c %s "$file")
  [ "$size" -lt 10000000 ] || fail "$file is $size bytes; crates.io accepts at most 10 MB"
  printf '%-18s %-6s %8d KiB\n' "$crate" "${wanted[$crate]}" $((size / 1024))
done

echo "release metadata and crate packages are consistent at $version"
echo "publish order: ${crates[*]}"
