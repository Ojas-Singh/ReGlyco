# Changelog

## 0.2.3 - 2026-10-15

Everything since 0.2.2. The command line gains GlycoShape logins for Level 2
data and runs density fitting with GlycoFlow; it builds against GlySys 0.1.2,
crabSAXS 0.1.1 and crabWURCS 0.3.3.

### GlycoShape data

- `reglyco login`, `reglyco logout` and `reglyco whoami`: a device login
  (RFC 8628) to the GlycoShape site. `reglyco login` prints a code to enter at
  `https://glycoshape.org/device`; the login is kept for Level 2 data only in
  `credentials.json` of the configuration folder (`$REGLYCO_CONFIG_DIR`,
  `$XDG_CONFIG_HOME/reglyco`, `~/.config/reglyco`, `%APPDATA%\reglyco`),
  readable by its owner only, and sent only with Level 2 requests to the site
  that issued it. `--level 2` selects Level 2 for accounts the GlycoShape team
  has given access; a refusal says why (not logged in, no access, terms to
  accept). Level 1 needs nothing, as before.
- Ensembles are fetched from `https://glycoshape.org` (glycoshape.io is
  retired), at Level 1 by default.
- Linkage torsions follow the definitions each GlycoShape entry declares:
  entries re-analysed with the IUPAC definitions (conventions version 2) and
  older entries (version 1) are both measured on their own atoms, and a point
  is not placed in the populations of a reference with other definitions.

### Density fitting

- `refine --objective density` fits glycans into X-ray and cryo-EM maps with
  GlycoFlow (`reglyco-glycoflow`, engine `glycoflow-core`); the earlier
  density fitter and its options are removed. Fits cover N-glycans,
  O-glycans, C-mannose and the other glycosylated residues (Ser, Thr, Trp,
  Tyr, Hyp, Hyl, Cys), infer a glycan out of density, score the deposited
  glycan with the same objective, and can extend a fit past the density as an
  ensemble.
- The GlycoFlow model is not part of ReGlyco. It is used after its licence is
  accepted (`--accept-glycoflow-license`, or `GLYCOFLOW_ACCEPT_LICENSE=1` for
  one run) and is loaded from `--glycoflow-model` / `$GLYCOFLOW_MODEL`, or
  downloaded from the gated Hugging Face repository `Ojas-Singh/glycoflow`
  with the user's token, at a pinned revision with SHA-256 checks
  (`reglyco glycoflow-model`).
- `validate --site` focuses the structural checks on one site; N-acetyl bond
  lengths for PDB component names; a metal coordination bond limit.

### Fixes

- The steric test skipped the first three atoms of every glycan by count, so
  H1 and one ring atom of the first sugar were never tested. It now skips only
  the glycan atom bonded to the protein, and a side chain turned to a library
  rotamer is tested against the rest of the protein.
- Glycosidic phi rotates about the child's anomeric carbon (C2 for sialic
  acids, which used to rotate about a line through the carboxyl carbon).
- crabWURCS 0.3.3: reports show the glycans as their coordinates describe
  them and list where residue names and atoms disagree; site discovery reads
  with the names deciding, so `refine` builds what it built before and records
  each disagreement.

### Browser

- Large ensembles stay within the wasm32 heap; torsion analysis of large
  ensembles is linear; multi-model inputs are relaxed from compact frame
  storage; browser releases include the WebGPU artifacts and record their
  provenance.

### Packaging

- The published crates share the workspace version (0.2.3). `glycoflow-core`
  (0.1.1) and `reglyco-glycoflow` are published for the first time; they hold
  open-source code only, not the GlycoFlow model. `reglyco-wasm` and
  `reglyco-pdf-wasm` are built into the browser bundles and are no longer
  published to crates.io.
- Releases: `scripts/check_release.sh`, `scripts/publish_crates.sh`,
  `scripts/release_binary.sh` and a tag-triggered release workflow
  ([`docs/release.md`](docs/release.md)).
