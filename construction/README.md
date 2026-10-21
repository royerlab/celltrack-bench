# Building the benchmark

`data/hard` and `data/full` are generated from the saved outputs of the three trackers
(`tracker_outputs/`) and your copy of the raw sequences. `rebuild_benchmark.sh` runs the whole
construction, checks every resulting file against `data/CHECKSUMS.sha256` (the benchmark used in the
paper), and installs the verified files into `data/`.

```bash
pip install -r construction/requirements.txt -e .      # Python 3.11
RAW_DATA=/path/to/raw_data construction/rebuild_benchmark.sh   # a few minutes on CPU
```

`RAW_DATA` must contain `datasets/<dataset>/<sequence>` and `<sequence>_GT/TRA` (with `man_track.txt`)
in Cell Tracking Challenge layout for all 35 sequences the trackers were run on: sequences `01` and
`02` of the ten CTC datasets (BF-C2DL-HSC, BF-C2DL-MuSC, DIC-C2DH-HeLa, Fluo-C2DL-Huh7, Fluo-C2DL-MSC,
Fluo-N2DH-GOWT1, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, PhC-C2DH-U373, PhC-C2DL-PSC), and Bacteria-cib
`01`–`02`, Bacteria-metA `01`–`05`, Bacteria-pheA `05`, Bacteria-rpsM `01`, `02`, `07` and
Bacteria-trpL `03`–`06` (see `data/sources/bacteria_sequence_maps/`). The script checks for all of
them before starting.

## What it runs

The trackers themselves are not re-run: each was applied to the ground-truth detections of every
sequence under 3-fold, dataset-level cross-validation (each dataset predicted by the model that did
not see it in training), and `tracker_outputs/` holds their outputs. From there:

| Step | Script | Output |
|---|---|---|
| 1. error analysis | `scripts/analyze_three_tracker_errors.py` | per-link membership of each tracker's errors (`error_membership.csv.gz`) and the dependence statistics |
| 2. hard-mining selection | `scripts/select_three_tracker_error_pool_by_split.py` | sequence-disjoint validation/test split; per dataset and split, the top 64 (hard set) or 256 (full set) links ranked by the number of trackers in error, then by hardness h(e) |
| 3. item build | `scripts/build_hoct_graph_benchmarks.py`, `scripts/validate_selected_hoct_graph_benchmarks.py` | the five-cell neighbourhoods, answer keys and metadata (seed 20260709) |
| 4. export | `export_release.py` | `data/<benchmark>/` in the released layout, including the reference graphs |

Steps 1–3 are the original construction scripts, unchanged. They resolve all paths relative to the
workspace root, and some of those relative paths are recorded in the items (and so in the benchmark
identity), which is why the workspace (`runs/rebuild/`) mirrors the original directory layout.

## Tracker outputs

| `tracker_outputs/evaluation/...` | Content |
|---|---|
| `hoct_groupcv/full_graphs/group_<k>/<dataset>/<sequence>.geff` | HOCT graphs: node positions, candidate links with similarities, selected solution, orphan probabilities |
| `trackastra_node_only_groupcv/full_graphs/...` | Trackastra graphs (link similarities) |
| `errors_gt_linking/<tracker>/all_groups_link_errors.csv` | AOGM link errors (EA/ED/EC) of HOCT, Trackastra and the Hungarian matcher against the ground truth |

The graphs originally also carried per-node features computed from the images and the ground-truth
segmentation (masks, bounding boxes, intensity and shape statistics); the construction never reads
them, so they were removed. `make_tracker_outputs.py` documents exactly how `tracker_outputs/` was
produced from the original tracking runs.
