# CellTrack-Bench

Code for **CellTrack-Bench: Microscopic Cell Tracking Benchmark for Vision–Language Models**
(Ilan Theodoro, Jordão Bragantini, Loïc A. Royer), [arXiv:2610.31566](https://arxiv.org/abs/2610.31566).

CellTrack-Bench evaluates whether vision–language models (VLMs) can make the cell-tracking decisions
that specialized trackers get wrong. It draws from 15 cell-tracking datasets across four imaging
modalities and turns the links that three trackers with different inductive biases fail on together
into local two-frame problems, each posed as three tasks:

| Task | The model is given | It returns |
|---|---|---|
| `cell_linking` | five source cells in frame *t* and their candidate targets in frame *t+1* | every ground-truth link (continuations and divisions) |
| `track_repair` | the same neighbourhood plus the HOCT tracker's proposed graph | the add / delete / change-semantic edits that make it correct |
| `track_inspection` | the same neighbourhood plus the HOCT proposal | every proposed or missing link a human should review |

The source datasets cannot be redistributed, so neither can the benchmark items derived from them.
Instead, this repository ships what is needed to **generate the benchmark locally** from your own copy
of the datasets, and to check that the result is byte-identical to the benchmark used in the paper:

- `tracker_outputs/` — the saved outputs of the three trackers the benchmark is mined from.
- `construction/` — the pipeline that builds `data/hard` (the paper's leaderboard set: 205 validation /
  244 test neighbourhoods) and `data/full` (626 / 911) from them, verified against
  `data/CHECKSUMS.sha256`.
- `celltrack_bench/` — the evaluation harness (rendering, the eight evidence conditions, the
  multi-turn runner for hosted and self-hosted models) and every metric reported in the paper.
- `predictions/` — the stored outputs of all 22 models in the paper, plus the validation runs used
  for harness selection, so every table can be recomputed without an API key.
- `scripts/eval_openweights.sh` — the vLLM serving settings used for the open-weight models.

## 1. Install

```bash
git clone git@github.com:royerlab/celltrack-bench.git && cd celltrack-bench
pip install -e ".[run]"                    # harness, metrics, openai/anthropic clients
pip install -r construction/requirements.txt  # only needed to build the benchmark (Python 3.11)
```

## 2. Get the raw datasets

Everything is rendered from the original microscopy sequences and their ground-truth tracking
annotations, in [Cell Tracking Challenge](https://celltrackingchallenge.net) layout under one root:

```
$RAW_DATA/datasets/<dataset>/<sequence>/t000.tif ...
$RAW_DATA/datasets/<dataset>/<sequence>_GT/TRA/man_track000.tif ... man_track.txt
```

| Dataset | Sequences | Source |
|---|---|---|
| BF-C2DL-HSC, BF-C2DL-MuSC, DIC-C2DH-HeLa, Fluo-C2DL-Huh7, Fluo-C2DL-MSC, Fluo-N2DH-GOWT1, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, PhC-C2DH-U373, PhC-C2DL-PSC | `01`, `02` | Cell Tracking Challenge training datasets (sequences and `TRA` ground truth) |
| Bacteria-cib | `01`, `02` | van Vliet et al., *Cell Systems* 2018 |
| Bacteria-metA | `01`–`05` | 〃 |
| Bacteria-pheA | `05` | 〃 |
| Bacteria-rpsM | `01`, `02`, `07` | 〃 |
| Bacteria-trpL | `03`–`06` | 〃 |

<!-- TODO: say where to obtain the bacterial sequences with TRA-style ground truth, and how to
convert them. -->
The bacterial sequences use local ids; `data/sources/bacteria_sequence_maps/` maps each one to the
original recording (e.g. `Bacteria-cib/01` = `140409-03`). In every dataset, `01` (or the lower
half of the sequence ids) is validation and the rest is test.

## 3. Build the benchmark

```bash
RAW_DATA=/path/to/raw_data construction/rebuild_benchmark.sh
```

This reruns the error analysis, the hard-mining selection and the item builds (a few minutes on CPU),
checks all 2,016 resulting files against `data/CHECKSUMS.sha256`, and installs `data/hard` and
`data/full`. See [`construction/README.md`](construction/README.md) for the steps and
[`data/README.md`](data/README.md) for the format.

## 4. Reproduce the paper's tables

```bash
python -m celltrack_bench.reproduce
```

This recomputes, from `predictions/` and `data/hard/` alone (about 10 s, no images, no API calls),
the leaderboard (Table 2), the harness-selection study and Table 1's NN-residual columns, and the
appendix tables (repair decomposition, review-budget triage, difficulty strata, validation–test
robustness, task correlations). Each of the 635 recomputed cells is compared with the value printed
in the paper (`reproduction/paper_values.json`), and all agree.

`pytest` additionally checks every leaderboard metric for all 22 systems against the unrounded
outputs of the original analysis scripts (`tests/fixtures/original_outputs.json`); tests that need
`data/` are skipped until the benchmark has been built.

## 5. Evaluate a model

The runner's defaults are the frozen protocol used for every leaderboard row: condition **c1**
(unannotated two-frame overview, normalized bounding boxes for every cell, and a `crop_and_zoom`
tool), low reasoning effort, an 8192-token total output budget per item with 2048 tokens reserved
for the final answer, temperature 1.0 and top-p 0.95. It renders from the raw data, found through
`$CELLTRACK_DATA_ROOT` (default `./raw_data`) or `--source-data-root`.

```bash
export CELLTRACK_DATA_ROOT=/path/to/raw_data

# hosted models (OPENAI_API_KEY / ANTHROPIC_API_KEY)
python -m celltrack_bench.runner --model gpt-5.5 --split test
python -m celltrack_bench.runner --backend anthropic --model claude-opus-5 --split test

# any OpenAI-compatible server, e.g. vLLM
python -m celltrack_bench.runner --backend chat --base-url http://127.0.0.1:8000/v1 \
    --model my-model --label my-model --split test

# or let the script start vLLM with the paper's per-model settings
MODEL=gemma-4-31b-it MODEL_PATH=google/gemma-4-31b-it scripts/eval_openweights.sh
```

A test run is 244 items × 3 tasks = 732 conversations. Use `--sample N` for a quick check spread
across datasets, `--dry-run` to build every prompt and image without calling a model, and
`--split-dir data/full` for the full benchmark. Responses are cached in `runs/cache/`, so an
interrupted run resumes; each run writes a compact `runs/<...>.jsonl`, its `.config.json`, and full
per-call traces in `runs/traces/` with every image the model saw in `runs/images/`.

```bash
python -m celltrack_bench.leaderboard --split test --results runs/test_<...>.jsonl
```

prints the run's full Table 2 row next to the reference graphs (nearest-centroid, Hungarian,
Trackastra, HOCT). `python -m celltrack_bench.score --results <run>.jsonl --split test` gives the
per-task primary scores with dataset-level bootstrap intervals and health checks (truncation, JSON
validity, tool failures); with several conditions in one file it also applies the harness-selection
rule.

## Metrics

| Column | Meaning |
|---|---|
| Link / Repair / Inspect F1, Mean | task-primary F1, averaged per dataset and then across datasets |
| NN-res | linking F1 restricted to links a nearest-centroid rule does not already explain |
| Div | linking F1 on division edges |
| LNK<sub>sub</sub>, ΔNN | CTC-style graph-edit score of the predicted links on the local neighbourhoods, and its margin over the nearest-centroid graph |
| Repair ΔLNK | change in LNK<sub>sub</sub> after applying the model's edits to the HOCT proposal |
| Inspect ΔLNK, Δ/1k | LNK<sub>sub</sub> gain when every flagged source is corrected by an oracle, in total and per 1000 flags |

All metrics use condition 1 unless stated otherwise; definitions are in the paper and in
`celltrack_bench/subgraph_lnk.py` and `celltrack_bench/nn_residual.py`.

## Harness conditions

The runner also implements the eight evidence presentations compared on the validation split
(`--conditions 1 2 ... 8`); c1 was selected and frozen before any test evaluation.

| | Cells identified by | Closer look | Overview |
|---|---|---|---|
| c1 | normalized bounding boxes | `crop_and_zoom` tool | unannotated |
| c2 | outlines + ids on the image | `crop_and_zoom` tool | annotated |
| c3 | outlines + ids | `zoom(cell_id)` tool | annotated |
| c4 | outlines + ids | pre-supplied outlined crops | annotated |
| c5 | outlines + ids | pre-supplied crops, no outline | annotated |
| c6 | boxes for sources, outlines for targets | `crop_and_zoom` tool | targets annotated |
| c7 | — | pre-supplied outlined crops only | none (information-loss ablation) |
| c8 | boxes and outlines | `crop_and_zoom` tool | annotated |

## Citation

```bibtex
@article{theodoro2026celltrackbench,
  title   = {{CellTrack-Bench}: Microscopic Cell Tracking Benchmark for Vision--Language Models},
  author  = {Theodoro, Ilan and Bragantini, Jord{\~a}o and Royer, Lo{\"\i}c A.},
  journal = {arXiv preprint arXiv:2610.31566},
  year    = {2026}
}
```

## License

<!-- TODO: code license. The source datasets (Cell Tracking Challenge; van Vliet et al., 2018) and
anything derived from their annotations remain subject to their own terms. -->
