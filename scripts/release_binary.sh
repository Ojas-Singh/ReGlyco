#!/usr/bin/env bash
# Build the Linux x86-64 executable of a published ReGlyco release for the
# Colab notebooks, from crates.io, and write its SHA-256.
#
#   scripts/release_binary.sh VERSION            # dist/reglyco-VERSION-linux-x86_64 and .sha256
#   scripts/release_binary.sh VERSION --upload   # also attach both to the GitHub release vVERSION
#
# Run it after `reglyco VERSION` is on crates.io, on x86-64 Linux with glibc
# 2.35 or older (Ubuntu 22.04, as Colab), so the binary runs on Colab. The
# release workflow runs it on ubuntu-22.04. --upload needs the GitHub CLI
# logged in with write access to Ojas-Singh/ReGlyco.
set -euo pipefail

version=${1:?usage: scripts/release_binary.sh VERSION [--upload]}
version=${version#v}
upload=${2:-}
repository=Ojas-Singh/ReGlyco
asset="reglyco-${version}-linux-x86_64"

if [ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]; then
  echo "release_binary: this machine is $(uname -s)-$(uname -m); the Colab binary must be built on x86-64 Linux" >&2
  exit 1
fi
glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
if [ "$(printf '%s\n2.35\n' "$glibc" | sort -V | tail -n 1)" != 2.35 ]; then
  echo "release_binary: warning: glibc $glibc is newer than Colab's 2.35; the binary may not run there" >&2
fi

index="https://index.crates.io/re/gl/reglyco"
if ! curl --fail --silent "$index" | jq --exit-status --arg v "$version" 'select(.vers == $v)' >/dev/null; then
  echo "release_binary: reglyco $version is not on crates.io yet" >&2
  exit 1
fi

root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
cargo install reglyco --locked --version "$version" --root "$root"

mkdir -p dist
install -m 0755 "$root/bin/reglyco" "dist/$asset"
"dist/$asset" --version
(cd dist && sha256sum "$asset" > "$asset.sha256")
cat "dist/$asset.sha256"

if [ "$upload" = --upload ]; then
  gh release upload "v$version" "dist/$asset" "dist/$asset.sha256" --repo "$repository"
  echo "attached to https://github.com/$repository/releases/tag/v$version"
fi
