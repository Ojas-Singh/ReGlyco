# Releasing ReGlyco

crates.io versions are permanent: a published version can be yanked but never
replaced, so a mistake needs a new patch release. Never move or reuse a
version tag.

## What is published

The crates of [`scripts/release-crates.txt`](../scripts/release-crates.txt),
in this order:

```text
glycoflow-core (own version)  reglyco-core  reglyco-build  reglyco-relax
reglyco-density  reglyco-validate  reglyco-report  reglyco-saxs
reglyco-ensemble  reglyco-refine  reglyco-glycoflow  reglyco-workflow
reglyco-cli  reglyco
```

They carry the version of `[workspace.package]` in `Cargo.toml` and depend on
each other at that exact version. `glycoflow-core`, the GlycoFlow engine that
GlycoFlow's own command line also takes by git tag (`glycoflow-core-vX.Y.Z`),
keeps its own version. The GlycoFlow model is never part of a package.
`reglyco-wasm` and `reglyco-pdf-wasm` are `publish = false`: they ship in the
browser bundles.

The workspace builds against `../GlySys`, `../crabSAXS` (path dependencies)
and crabWURCS (git tag). The published manifests take them from crates.io at
the versions the manifests carry, so the releases come in this order:

1. crabWURCS and crabSAXS, at the versions in the manifests.
2. GlySys, all its crates (its `docs/release.md`).
3. ReGlyco.

`scripts/check_release.sh` stops with a message naming what is missing when a
GlySys, crabSAXS or crabWURCS version is not in the crates.io index yet, and
so do `scripts/publish_crates.sh` and the release workflow, before anything is
uploaded.

## Prepare

1. Set `[workspace.package] version` in `Cargo.toml` and the internal
   dependency versions (`reglyco-* = { version = ..., path = ... }`) to the new
   version; bump `glycoflow-core` (its `Cargo.toml` and
   `scripts/release-crates.txt`) only when it changed since its last release.
   Point the GlySys, crabSAXS and crabWURCS requirements at their releases,
   and the `GLYSYS_REF` / `CRABSAXS_REF` tags in
   `.github/workflows/release.yml` at the same versions. Run
   `cargo update -w`.
2. Add `## VERSION - DATE` to `CHANGELOG.md` and the install line
   `cargo install reglyco --locked --version VERSION` to `README.md`.
3. Merge the release pull request after CI passes.

## Publish from a maintainer machine

This is the current route (the repository has no crates.io token secret).
The machine needs `cargo login` once with a token that may publish the crates
above, and GlySys and crabSAXS checked out next to ReGlyco at their released
tags.

```console
cd ReGlyco && git fetch origin && git checkout --detach origin/main
(cd ../GlySys && git fetch --tags && git checkout --detach v0.1.2)
(cd ../crabSAXS && git fetch --tags && git checkout --detach 0.1.1)
scripts/check_release.sh VERSION
git tag -a vVERSION -m "ReGlyco VERSION" && git push origin vVERSION
scripts/publish_crates.sh VERSION
```

`check_release.sh` checks the versions, the lock file, the changelog and
README, the crate list and its order, and that the dependencies are on
crates.io, then packages all the crates together so Cargo verifies each one
against the others and against GlySys, crabSAXS and crabWURCS from crates.io.
`publish_crates.sh` runs `cargo publish -p <crate> --locked` for each crate in
order and waits until the crate is in the crates.io index before the next; a
crate already in the index is skipped, so the script can be run again after an
interruption. Without the script, the same sequence is `cargo publish -p
<crate> --locked` for each crate of the list in order, waiting after each
until `https://index.crates.io/<first two letters>/<next two>/<crate>` lists
the new version.

Pushing the tag starts the release workflow. It skips publication with a
notice when there is no token, skips the Colab binary while `reglyco` is not
on crates.io, and creates the GitHub release.

## Colab binary

The Colab notebooks use the Linux x86-64 executable described by
`GlycoShape-Resources/colab/reglyco_release.json`: an asset
`reglyco-VERSION-linux-x86_64` on the GitHub release `vVERSION`, pinned by its
SHA-256. It is built from crates.io after publication, so it is exactly what
`cargo install reglyco --locked --version VERSION` gives, on glibc 2.35 or
older (Ubuntu 22.04, as Colab). Either:

- run the release workflow again by hand (Actions, Release, Run workflow,
  VERSION). Its `binary` job runs `scripts/release_binary.sh VERSION` on
  ubuntu-22.04 and attaches `reglyco-VERSION-linux-x86_64` and its `.sha256`
  file to the release; or
- on an x86-64 Linux machine, run

  ```console
  scripts/release_binary.sh VERSION --upload
  ```

  which runs `cargo install reglyco --locked --version VERSION` into a
  temporary root, writes `dist/reglyco-VERSION-linux-x86_64` and its
  `.sha256`, and attaches both with `gh release upload vVERSION`.

Then update `colab/reglyco_release.json` in GlycoShape-Resources: `version`,
`asset`, `url`
(`https://github.com/Ojas-Singh/ReGlyco/releases/download/vVERSION/reglyco-VERSION-linux-x86_64`),
`sha256` (from the `.sha256` asset), `source_revision` (the commit of the tag
`vVERSION`) and `source_dirty_at_build: false`. The notebook helper checks the
SHA-256 before using a downloaded binary and accepts `REGLYCO_BIN` or
`REGLYCO_SOURCE_ROOT` for local testing.

## Publish from GitHub Actions

With a `CARGO_REGISTRY_TOKEN` repository secret, pushing the tag after the
GlySys release is in the index is the whole release:
`.github/workflows/release.yml` checks the dependencies, repeats the CI checks
with GlySys and crabSAXS at `GLYSYS_REF` / `CRABSAXS_REF` and
`check_release.sh`, publishes with `publish_crates.sh`, builds the Colab
binary from crates.io and attaches it with its SHA-256 to the GitHub release.
