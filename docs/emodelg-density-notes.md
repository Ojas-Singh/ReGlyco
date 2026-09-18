# EModelG density-fitting notes

ReGlyco keeps the downloaded research archive in the ignored cache at
`.reglyco-cache/research/emodelg/`. The source URL is
<https://www.cryoseek.org.cn/EModelG_v0.zip> and the cached archive SHA-256 is
`b4b2aebb906d5ca7307830e647a388df6a019f436fcd780d1b70a930f4d26774`.

The archive is not vendored or executed by ReGlyco. The inspected files refer
to external CUDA/Torch, Phenix, checkpoints, SO(3) samples, and carbohydrate
templates that are not included in the archive; no license or citation file
was present. ReGlyco therefore uses only independently implemented algorithmic
ideas:

- normalize and resample maps before real-space scoring;
- detect or score pyranose-ring templates over discrete orientations;
- grow a carbohydrate outward from a known attachment/root;
- score density continuity along glycosidic connection paths;
- reject overlapping or weakly supported placements and rescore after fitting.

The current density objective applies the root-first and continuity ideas
without requiring external executables or machine-learning weights.
