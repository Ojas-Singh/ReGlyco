# P02763 Build audit against Cookbook

This is a frozen diagnostic comparison of the five annotated N-linked sites in
UniProt P02763 (A:33, A:56, A:72, A:93, and A:103), all assigned `G00028MO`.
It is an audit of output completeness and steric classification; it is not a
claim that the two search implementations are statistically identical.

## Frozen inputs

- Protein: AlphaFold P02763 monomer, chain A, fetched on 2026-09-18.
- Protein SHA-256: `dfbee042bc0e99398b6df9a4f4b01ca36ef62c29acf0725325249caa0b296b8c`.
- GlycoShape: PDB, beta anomer, level 2, `G00028MO`.
- Rust workspace revision before this audit: `561b1ebe873e19ef072c54993fe6d4bd20249dd5`.
- Cookbook revision: `553ae83d0c8c6f77508f7bff9769aeab8954e982`.
- Search budget: population 32, 24 generations (800 nominal candidate slots),
  20 seeds (`0..19`).

The Cookbook extension was built locally from its checked-out source in a
disposable Python environment. No Cookbook files were changed by the audit.
The Cookbook path uses its existing level-2 `sampler` and `ensemble` calls;
Rust uses the same PDB level-2 provider assets. The Rust and Cookbook cache
layouts are different, so their downloaded multiframe files are recorded by
their source/cache provenance rather than compared as raw filenames.

## Commands

Rust Build (one output directory per seed):

```text
reglyco build --uniprot P02763 \
  --attach A:33=G00028MO --attach A:56=G00028MO \
  --attach A:72=G00028MO --attach A:93=G00028MO \
  --attach A:103=G00028MO \
  --no-system --population 32 --generations 24 --level 2 \
  --seed <seed> --output <directory>
```

Cookbook was invoked with the equivalent local call:

```python
ensemble.attach(
    protein, protein_models,
    ["G00028MO"] * 5,
    ["33_A", "56_A", "72_A", "93_A", "103_A"],
    ray_size=32, max_attempts=800,
    output_format="PDB", seed=seed,
)
```

## Results

| Engine | Runs | Complete, non-placeholder output | Jointly clash-free | Site 56 | Site 93 |
| --- | ---: | ---: | ---: | --- | --- |
| Rust ReGlyco | 20 | 20/20 | 0/20 | unresolved in 20/20 | unresolved in 5/20 |
| Cookbook | 20 | 0/20 after placeholder inspection | not established | zero-coordinate placeholder in 20/20 | emitted in 20/20 |

Every Rust run exited successfully and wrote a complete five-glycan PDB. Its
status was `best_complete_clashing`; site A:56 was always reported as
unresolved, and A:93 was unresolved for five seeds. The exported PDB contains
coordinates for all requested attachment trees, so these are inspectable
diagnostic structures rather than omitted sites.

The Cookbook call reported 32 frames for every run, but its per-site log says
`site 56_A ... 0 frames recieved` and then fills that site with zero arrays.
The resulting multiframe PDB has no chain for that attachment in its first
model. Therefore its apparent frame count is not an all-site glycosylated
solution. This is the missing-conformer/zero-placeholder behavior that the
new Build result makes explicit.

The comparison does **not** show that P02763 has a jointly clash-free
five-glycan solution. It shows that the previous “successful” Cookbook output
could suppress one site, while Rust now exports every constructible glycan and
reports the unresolved site. A future search-quality comparison must use a
common valid-state representation and independently re-score exported
coordinates.

## Regression fixed during the audit

P02763 first exposed an unrelated repair panic in the Rust path. Rotamer repair
used `then_some(rotamer_pass - 1)`. Rust evaluates the argument eagerly, so
the deposited-rotamer pass (`0`) underflowed before a diagnostic Build could be
written. Repair now converts the pass with a checked helper and has a unit
regression for pass `0` and pass `1`.

