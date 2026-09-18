# Release checklist

The published workspace is built from a clean checkout. Local path
dependencies in the development workspace also carry their registry version,
so Cargo removes the path during publication.

Publish in dependency order:

1. `crabWURCS` crates and `crabSAXS` at the versions recorded in the manifests.
2. The GlySys workspace crates, beginning with `glysys` and then its first-party
   library dependencies.
3. ReGlyco member crates in dependency order, followed by the `reglyco` root
   package and binary.

Before each upload, run the package and verification checks from a checkout
that does not rely on sibling paths:

```console
cargo package --workspace
cargo publish --dry-run -p <package>
cargo install --locked --path .
```

The Colab notebooks use the pinned Linux x86-64 release described by
`GlycoShape-Resources/colab/reglyco_release.json`. Upload the matching binary
to the GitHub release asset URL only after the source has been committed and
the checksum has been regenerated. The notebook helper verifies the SHA-256
before using a downloaded asset and supports `REGLYCO_BIN` or
`REGLYCO_SOURCE_ROOT` for local testing.
