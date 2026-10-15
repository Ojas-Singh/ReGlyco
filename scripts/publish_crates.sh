#!/usr/bin/env bash
# Publish the crates of scripts/release-crates.txt to crates.io, in that
# order, waiting for each to appear in the index before the next.
#
#   scripts/publish_crates.sh VERSION
#
# Needs a crates.io token: `cargo login` on a maintainer machine, or
# CARGO_REGISTRY_TOKEN in the release workflow, and the sibling checkouts
# ../GlySys and ../crabSAXS at the released versions. A crate whose version is
# already in the index is skipped, so an interrupted run can be repeated.
# crates.io versions are permanent: run scripts/check_release.sh VERSION first.
set -euo pipefail
cd "$(dirname "$0")/.."

version=${1:?usage: scripts/publish_crates.sh VERSION}
version=${version#v}

workspace_version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' Cargo.toml)
if [ "$workspace_version" != "$version" ]; then
  echo "publish_crates: [workspace.package] version is $workspace_version, not $version" >&2
  exit 1
fi
if [ "$(git describe --exact-match --tags HEAD 2>/dev/null)" != "v$version" ]; then
  echo "publish_crates: warning: HEAD is not the tag v$version" >&2
fi

# Refuse before uploading anything when GlySys (or crabSAXS, crabWURCS) is
# not on crates.io at the version the manifests require.
scripts/check_release.sh --dependencies "$version"

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

published=()
while read -r crate pinned; do
  crate_version=${pinned:-$version}
  published+=("$crate $crate_version")
  if in_index "$crate" "$crate_version"; then
    echo "$crate $crate_version is already on crates.io"
    continue
  fi
  echo "publishing $crate $crate_version"
  cargo publish -p "$crate" --locked
  for _ in $(seq 60); do
    in_index "$crate" "$crate_version" && break
    sleep 10
  done
  if ! in_index "$crate" "$crate_version"; then
    echo "publish_crates: $crate $crate_version did not appear in the index within 10 minutes" >&2
    exit 1
  fi
  echo "$crate $crate_version is in the index"
done < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' scripts/release-crates.txt)

printf 'published: %s\n' "${published[@]}"
