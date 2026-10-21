# Benchmark data

`data/hard` and `data/full` are derived from the source datasets' annotations and are therefore not
distributed: `construction/rebuild_benchmark.sh` generates them from your copy of the raw datasets
and the saved tracker outputs, and installs them only if all files match `CHECKSUMS.sha256`, the
SHA-256 of every file of the benchmark used in the paper. This directory then contains:

```
CHECKSUMS.sha256                        checksums of the released build (distributed)
sources/bacteria_sequence_maps/         local bacterial sequence id -> original recording (distributed)
data/<benchmark>/                       benchmark = hard (paper leaderboard) | full  (generated)
  benchmark_manifest.json               benchmark id and where each split's items live
  <split>_items.json                    split = validation | test: one entry per neighbourhood
  <split>/<task>/answer_key.jsonl       ground-truth answer per item, for each of the three tasks
  items/<split>/<item_id>/metadata.json everything that defines and scores one item (see below)
  references/<split>.json               precomputed reference graphs (see below)
  build/<split>/                        provenance of the item build (selection parameters, counts)
```

| Benchmark | Validation | Test | Datasets with items (val / test) | Selection |
|---|---:|---:|---:|---|
| `hard` | 205 | 244 | 10 / 13 | top-64 hard-mined links per dataset and split |
| `full` | 626 | 911 | 11 / 13 | top-256 hard-mined links per dataset and split |

Validation and test come from disjoint sequences. Validation was used only to choose the harness;
every leaderboard number is on `hard/test`.

**Item metadata.** `metadata.json` holds the frame pair, the crop (`crop_yxyx`) of the full frame
that is rendered, the display ids of the five source cells and their candidate targets
(`source_display`, `target_display`, `candidate_map`), the ground-truth subgraph (`gt_edges`), the
HOCT proposal used by repair and inspection (`hoct_edges`), the gold edits and review flags
(`operations`, `inspection`), and how the item was selected (`selection`: which trackers erred on the
anchor link and its hardness). The `paths` fields name preview images from the build that are not
distributed; the harness renders every image from the raw sequences instead.

**Identity.** The runner hashes every item entry, metadata file and answer key into the benchmark
identity written into each result row. A verified build reproduces the identities of the paper's
runs exactly: `celltrack-bench-consensus-v2-3bdb6d31bf98f548` (hard/test) and
`...-6e05798663aaf734` (hard/validation); the full benchmark is
`celltrack-bench-consensus-v2-xl-d1da48bcf453e840` (test) and `...-xl-6c150fb52cea70ee` (validation).
`consensus` is the internal name of the hard-mining selection described in the paper.

**Reference graphs.** `references/<split>.json` contains, per item,
- `nearest_centroid`: each source cell's nearest-centroid target, computed from the ground-truth
  masks. It defines the nearest-centroid baseline, ΔNN and NN-residual F1, and is precomputed so
  that scoring needs no raw images.
- `trackers.hungarian`, `trackers.trackastra`: the neighbourhood subgraphs predicted by the two other
  benchmark trackers, reconstructed from the error mining that selected the items (every
  reconstruction reproduces its item's recorded tracker verdict on the anchor link). The HOCT
  reconstruction reproduces `hoct_edges` on all hard-set items and on all but one item of each
  full-set split; scoring always uses `hoct_edges` itself.

These files are derived from the Cell Tracking Challenge and van Vliet et al. (2018) data and remain
subject to their terms; do not redistribute them.
