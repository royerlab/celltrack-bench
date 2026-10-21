#!/usr/bin/env python3
"""NN-residual and the other debiased metrics for one or more sweeps, with paired inference.

Why this exists. `celltrack_bench.score` reports the pooled primary metric, and ~68% of validation
links are solved by a nearest-centroid rule alone, so a model can raise pooled f1 without getting any
better at the decisions that matter. This computes the proximity-resistant view instead, which is the
check that distinguishes learning from reward hacking.

Metrics, all macro (per-item mean, then averaged) so views are comparable:
  nn_resid   f1 restricted to links that nearest-centroid does NOT already solve
  divisions  f1 over division edges only
  nn_agree   fraction of sources where the model picked the nearest centroid (diagnostic, not a score)
  pooled     ordinary f1 over all links, i.e. the proximity-inflated number

TWO RULES, both learned the hard way:
  * Always filter by `condition`. Keying predictions by item_id alone silently keeps whichever of the
    8 conditions appears last in the file, and that mix differs per model -- it invalidated an entire
    round of cross-checkpoint comparisons.
  * Never mix harness revisions. v10 rewrote the division contract, so a v7 file and a v10 file are not
    comparable.

Use `python -m celltrack_bench.leaderboard` to report these metrics for a run.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from celltrack_bench import benchmark_data as B
from celltrack_bench import render as R
from celltrack_bench.answer_scoring import select_output_object

DIVISION_WORDS = {"division", "divide", "divided", "mitosis", "daughter", "split"}


def semantic(value) -> str:
    return "division" if str(value or "").strip().lower() in DIVISION_WORDS else "continuation"


def flatten(value):
    """Yield scalar ids. Guards against Ellipsis, which a truncated answer can produce and which is truthy."""
    if value is Ellipsis or value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten(item)
    elif not isinstance(value, dict):
        yield str(value)


def link_list(parsed):
    value = parsed.get("links") if isinstance(parsed, dict) else None
    return value if isinstance(value, (list, tuple)) else []


def link_set(parsed) -> set[tuple[str, str, str]]:
    counts: dict[str, int] = defaultdict(int)
    raw = []
    for link in link_list(parsed):
        if not isinstance(link, dict):
            continue
        for src in flatten(link.get("source")):
            for tgt in flatten(link.get("target")):
                raw.append((src, tgt, link.get("semantic")))
                counts[src] += 1
    out = set()
    for src, tgt, sem in raw:
        kind = semantic(sem) if sem is not None else ("division" if counts[src] > 1 else "continuation")
        out.add((src, tgt, kind))
    return out


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


def load_truth(split: str, split_dir=None):
    truth = {}
    base = Path(split_dir) if split_dir else (R.REPO_ROOT / "data" / "hard")
    path = base / split / "cell_linking" / "answer_key.jsonl"
    for line in B.open_text(path):
        row = json.loads(line)
        truth[row["item_id"]] = link_set(row["answer"])
    return truth


def nearest_centroid(item_ids, items_root=None):
    """Per item: for each source, the target whose centroid is closest. This is the shortcut to remove."""
    out = {}
    for item in item_ids:
        meta = R.load_meta(item, items_root=items_root) if items_root else R.load_meta(item)
        _images, panels = R.overview(meta, annotate=())
        src = panels["t"].normalized_bboxes()
        tgt = panels["t1"].normalized_bboxes()
        if not tgt:
            out[item] = {}
            continue
        centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
        dist = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5  # noqa: E731
        out[item] = {
            name: min(tgt, key=lambda k: dist(centre(box), centre(tgt[k]))) for name, box in src.items()
        }
    return out


def per_item(path: Path, condition: int, truth, nn, residual):
    revisions, per = set(), {}
    truncated = valid = total = 0
    for line in B.open_text(path):
        row = json.loads(line)
        if row.get("task") != "cell_linking" or row.get("condition") != condition:
            continue
        revisions.add(row.get("harness_revision"))
        total += 1
        truncated += 1 if row.get("truncated") else 0
        try:
            parsed, _ = select_output_object(row.get("text") or "", "cell_linking")
            got = link_set(parsed)
            valid += 1
        except Exception:
            got = set()
        item = row["item_id"]
        if item not in truth:
            continue
        gold, res = truth[item], residual[item]
        sources = {s for s, _t, _k in res}
        gold_div = {x for x in gold if x[2] == "division"}
        got_div = {x for x in got if x[2] == "division"}
        by_source = defaultdict(set)
        for src, tgt, _k in got:
            by_source[src].add(tgt)
        shared = [s for s in nn[item] if s in by_source]
        per[item] = {
            "nn_resid": f1(len(res & got),
                           len({x for x in got if x[0] in sources and x not in gold}),
                           len(res - got)) if res else None,
            "divisions": f1(len(gold_div & got), len(got_div - gold), len(gold_div - got)) if gold_div else None,
            "pooled": f1(len(gold & got), len(got - gold), len(gold - got)),
            "nn_agree": statistics.fmean(
                1.0 if nn[item][s] in by_source[s] else 0.0 for s in shared) if shared else None,
        }
    return per, revisions, {"n": total, "valid_json": valid / total if total else 0.0,
                            "truncated": truncated / total if total else 0.0}


def mean(per, key):
    vals = [v[key] for v in per.values() if v.get(key) is not None]
    return statistics.fmean(vals) if vals else float("nan"), len(vals)


def paired(a, b, key, rng, draws=4000):
    pairs = [(a[i][key], b[i][key]) for i in a
             if i in b and a[i].get(key) is not None and b[i].get(key) is not None]
    if len(pairs) < 8:
        return None
    delta = statistics.fmean(y - x for x, y in pairs)
    boots = sorted(statistics.fmean(y - x for x, y in (pairs[rng.randrange(len(pairs))] for _ in pairs))
                   for _ in range(draws))
    return delta, boots[int(0.05 * draws)], boots[int(0.95 * draws) - 1], len(pairs)
