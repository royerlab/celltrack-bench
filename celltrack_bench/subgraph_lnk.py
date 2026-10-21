#!/usr/bin/env python3
"""subgraph-LNK: an AOGM/CTC-style linking score restricted to benchmark neighborhoods.

WHY. The pooled f1 in `harness.score` and the debiased views in nn_residual.py are set-overlap
metrics; the tracking community reports AOGM-derived scores (Matula et al. 2015, the CTC "TRA/LNK"
family) where errors are weighted graph-edit operations. This computes the LNK analogue on the
benchmark's local neighborhoods, so numbers are comparable in spirit to CTC LNK while remaining
computable from the frozen result files -- no new evals.

METRIC. Per item, GT edge set E_gt and predicted edge set E_pred, each edge (source, target, kind),
kind in {continuation, division}, kind derived exactly as nn_residual.link_set(): explicit
`semantic` wins, otherwise division iff the source has >1 predicted targets. AOGM-A cost:
  * edge addition   (in E_gt, endpoints missing in E_pred)                     w_EA = 1.5
  * edge deletion   (in E_pred, endpoints not in E_gt) -- counted ONLY if the
    source has at least one GT edge, mirroring nn_residual's FP scoping        w_ED = 1.0
  * semantic change (endpoints match, kind wrong; counts once, NOT add+del)   w_ES = 1.0
Benchmark level (disjoint-union semantics, sums over items):
  LNK_sub = 1 - min(sum AOGM_A, sum AOGM_0) / sum AOGM_0,   AOGM_0 = 1.5 * |E_gt| per item.
Items with a missing/unparseable model row count as EMPTY predictions for cell_linking (full cost)
and as NO-OP for repair/inspection (the proposal stands), so every model is scored over the same
sum of AOGM_0. Per-item LNK_sub (used for --compare pairing and the Spearman check) caps per item.

TASKS.
  cell_linking     E_pred = the model's links. Reported: LNK_sub and dNN = LNK_sub(model) -
                   LNK_sub(nearest-centroid baseline), where the baseline links every GT source to
                   the nearest-centroid target as a continuation (nn_residual.nearest_centroid()).
  track_repair     The model returns edit operations; they are applied to the HOCT proposal graph
                   and the corrected graph is scored. Reported: dLNK = LNK_sub(corrected) -
                   LNK_sub(proposal).
  track_inspection Oracle triage: for every edge the model flags, the flagged edge's SOURCE gets a
                   simulated human fix (all its proposal out-edges replaced with its GT out-edges).
                   Reported: dLNK and dLNK per 1000 flagged edges (triage efficiency).

SCHEMA FINDINGS (verified on data/hard test, 244 items per task, and its items/<item_id>/metadata.json):
  * cell_linking answer:    {"links": [{source, target, semantic}], "unlinked_sources": [...]}.
    1426 GT edges, 424 divisions; 7 items have unlinked sources (their spurious out-edges are
    outside the deletion scope, same as nn_residual).
  * track_repair answer:    {"operations": [{op: add|delete|change_semantic, source, target,
    semantic, [from_semantic, to_semantic]}]} -- edits RELATIVE to the proposal. Model outputs use
    the same format (the prompt embeds the proposal and the JSON contract).
  * track_inspection answer: {"links_to_inspect": [{source, target, hoct_status:
    selected|unselected, issue: missing|spurious|wrong_semantic}]}. Model outputs use the same
    format; entries with an "ok"-like issue are not counted as flags.
  * proposal graph:         metadata_private.json has `hoct_edges` (the proposal), `gt_edges`,
    plus the gold `operations`/`inspection`. Verified for all 244 items: applying the answer-key
    operations to hoct_edges reproduces gt_edges exactly, and gold-flag oracle triage reproduces
    gt_edges exactly, so the proposal graph is fully recoverable for both tasks.

RULE CARRIED OVER FROM nn_residual.py: always filter by condition (the frozen protocol on this
benchmark is condition 1). Use `python -m celltrack_bench.leaderboard` to report these metrics.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from celltrack_bench import benchmark_data as B
from celltrack_bench import render as R
from celltrack_bench.answer_scoring import (
    normalized_label,
    normalized_operation,
    normalized_semantic,
    select_output_object,
)

W_EA, W_ED, W_ES = 1.5, 1.0, 1.0
TASKS = ("cell_linking", "track_repair", "track_inspection")
DEFAULT_SPLIT_DIR = R.REPO_ROOT / "data" / "hard"
DEFAULT_ITEMS_ROOT = DEFAULT_SPLIT_DIR / "items" / "test"
# issues that mean "this link is fine" -- listing a link with one of these is not a flag
OK_ISSUES = {"ok", "okay", "none", "correct", "fine", "good", "valid", "no_issue", "keep"}

def semantic(value) -> str:
    return normalized_semantic(value) if normalized_semantic(value) == "division" else "continuation"


def flatten(value):
    """Yield scalar ids. Guards against Ellipsis, which a truncated answer can produce."""
    if value is Ellipsis or value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten(item)
    elif not isinstance(value, dict):
        yield value


def parse_links(parsed) -> dict[tuple[str, str], set[str]]:
    """Model/GT links -> {(source, target): {kind, ...}}. Kind derivation mirrors nn_residual.link_set:
    explicit semantic wins, otherwise division iff the source has >1 raw targets. Labels are
    normalized (s1/1 -> S1) so ragged model spellings still score."""
    links = parsed.get("links") if isinstance(parsed, dict) else None
    if not isinstance(links, (list, tuple)):
        return {}
    counts: dict[str, int] = defaultdict(int)
    raw = []
    for link in links:
        if not isinstance(link, dict):
            continue
        for src in flatten(link.get("source") if "source" in link else link.get("source_id")):
            src = normalized_label(src, "S")
            if src is None:
                continue
            for tgt in flatten(link.get("target") if "target" in link else link.get("target_id")):
                tgt = normalized_label(tgt, "T")
                if tgt is None:
                    continue
                raw.append((src, tgt, link.get("semantic") if "semantic" in link else link.get("type")))
                counts[src] += 1
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for src, tgt, sem in raw:
        kind = semantic(sem) if sem is not None else ("division" if counts[src] > 1 else "continuation")
        out[(src, tgt)].add(kind)
    return dict(out)


def canonical(edges) -> dict[tuple[str, str], str]:
    """GT/proposal edge list from an answer key or metadata_private.json -> {(s, t): kind}."""
    return {(e["source"], e["target"]): semantic(e.get("semantic")) for e in edges}


def aogm(gt: dict[tuple[str, str], str], pred: dict[tuple[str, str], set[str]]) -> tuple[float, float]:
    """(AOGM_A cost, AOGM_0). Deletions scoped to sources that have at least one GT edge."""
    gt_sources = {s for s, _t in gt}
    cost = 0.0
    for pair, kind in gt.items():
        kinds = pred.get(pair)
        if kinds is None:
            cost += W_EA
        elif kind not in kinds:
            cost += W_ES
    for s, _t in pred:
        if (s, _t) not in gt and s in gt_sources:
            cost += W_ED
    return cost, W_EA * len(gt)


def lnk(records, cost_key: str = "cost") -> float:
    cost = sum(r[cost_key] for r in records)
    denom = sum(r["aogm0"] for r in records)
    return 1.0 - min(cost, denom) / denom if denom else float("nan")


def apply_operations(operations, proposal: dict[tuple[str, str], str]) -> dict[tuple[str, str], set[str]]:
    """track_repair: apply add/delete/change_semantic to the proposal, in listed order.
    add without semantic gets its kind derived from final out-degree; change_semantic without a
    target kind toggles; change_semantic on an absent edge is ignored."""
    graph: dict[tuple[str, str], str | None] = dict(proposal)
    if isinstance(operations, (list, tuple)):
        for row in operations:
            if not isinstance(row, dict):
                continue
            op = normalized_operation(row.get("op") or row.get("operation") or row.get("action"))
            src = normalized_label(row.get("source") or row.get("source_id"), "S")
            tgt = normalized_label(row.get("target") or row.get("target_id"), "T")
            if src is None or tgt is None:
                continue
            sem = normalized_semantic(row.get("semantic") or row.get("type"))
            if op == "add":
                graph[(src, tgt)] = sem
            elif op == "delete":
                graph.pop((src, tgt), None)
            elif op == "change_semantic":
                to = normalized_semantic(row.get("to_semantic") or row.get("to")) or sem
                if (src, tgt) in graph:
                    current = graph[(src, tgt)]
                    graph[(src, tgt)] = to or ("continuation" if current == "division" else "division")
    degree: dict[str, int] = defaultdict(int)
    for s, _t in graph:
        degree[s] += 1
    return {
        pair: {semantic(kind) if kind is not None else ("division" if degree[pair[0]] > 1 else "continuation")}
        for pair, kind in graph.items()
    }


def parse_flags(parsed) -> set[tuple[str, str]]:
    """track_inspection: the set of flagged (source, target) pairs. Any listed link counts as a
    flag unless its issue is an explicit ok-alias."""
    rows = parsed.get("links_to_inspect") if isinstance(parsed, dict) else None
    flags: set[tuple[str, str]] = set()
    if not isinstance(rows, (list, tuple)):
        return flags
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = normalized_label(row.get("source") or row.get("source_id"), "S")
        tgt = normalized_label(row.get("target") or row.get("target_id"), "T")
        issue = str(row.get("issue") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if src is None or tgt is None or issue in OK_ISSUES:
            continue
        flags.add((src, tgt))
    return flags


def triage(flags: set[tuple[str, str]], proposal, gt) -> dict[tuple[str, str], set[str]]:
    """Oracle triage: every flagged source gets its proposal out-edges replaced by its GT out-edges."""
    sources = {s for s, _t in flags}
    graph = {pair: kind for pair, kind in proposal.items() if pair[0] not in sources}
    graph.update({pair: kind for pair, kind in gt.items() if pair[0] in sources})
    return {pair: {kind} for pair, kind in graph.items()}


def load_truth(split: str, split_dir=None) -> dict[str, dict[tuple[str, str], str]]:
    base = Path(split_dir) if split_dir else DEFAULT_SPLIT_DIR
    truth = {}
    for line in B.open_text(base / split / "cell_linking" / "answer_key.jsonl"):
        row = json.loads(line)
        truth[row["item_id"]] = canonical(row["answer"]["links"])
    return truth


def load_graphs(item_ids, items_root) -> dict[str, tuple[dict, dict]]:
    """item -> (gt, proposal) from metadata_private.json (verified against the answer keys)."""
    out = {}
    for item in item_ids:
        meta = R.load_meta(item, items_root=Path(items_root))
        out[item] = (canonical(meta["gt_edges"]), canonical(meta["hoct_edges"]))
    return out


def load_references(split_dir, split: str) -> dict:
    """Precomputed reference graphs shipped with the benchmark (data/<benchmark>/references/<split>.json).

    `nearest_centroid` maps item -> {source: nearest-centroid target}; it was computed with
    `nearest_centroid()` below from the GT masks, so scoring needs no raw images. `trackers` holds the
    Trackastra and Hungarian reference subgraphs reconstructed from the benchmark's error mining."""
    return json.loads((Path(split_dir) / "references" / f"{split}.json").read_text())


def nearest_centroid_from_references(item_ids, split_dir, split: str):
    nn = load_references(split_dir, split)["nearest_centroid"]
    missing = [item for item in item_ids if item not in nn]
    if missing:
        raise KeyError(f"no nearest-centroid reference for {len(missing)} items, e.g. {missing[:3]}")
    return {item: nn[item] for item in item_ids}


def nearest_centroid(item_ids, items_root, source_data_root=None):
    """Per item: each source -> nearest-centroid target, exactly as nn_residual.nearest_centroid.
    Renders the GT masks, so it needs the raw data; scoring uses nearest_centroid_from_references."""
    out = {}
    root = Path(source_data_root) if source_data_root is not None else R.SOURCE_DATA_ROOT
    for item in item_ids:
        meta = R.load_meta(item, items_root=Path(items_root), source_data_root=root)
        _images, panels = R.overview(meta, annotate=())
        src = panels["t"].normalized_bboxes()
        tgt = panels["t1"].normalized_bboxes()
        if not tgt:
            out[item] = {}
            continue
        centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
        dist = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5  # noqa: E731
        out[item] = {name: min(tgt, key=lambda k: dist(centre(box), centre(tgt[k]))) for name, box in src.items()}
    return out


def nn_records(truth, nn) -> dict[str, dict]:
    per = {}
    for item, gt in truth.items():
        pred = {(s, nn[item][s]): {"continuation"} for s, _t in gt if s in nn[item]}
        cost, aogm0 = aogm(gt, pred)
        per[item] = {"cost": cost, "aogm0": aogm0}
    return per


def parse_row_object(text: str, task: str):
    try:
        parsed, _src = select_output_object(text or "", task)
        return parsed
    except Exception:
        return None


def per_item(path: Path, task: str, condition: int, truth, graphs):
    """Per-item records for one results file. Missing rows: empty prediction for cell_linking,
    no-op (proposal stands / zero flags) for repair and inspection."""
    per, revisions = {}, set()
    truncated = valid = total = 0
    for line in B.open_text(path):
        row = json.loads(line)
        if row.get("task") != task or row.get("condition") != condition:
            continue
        item = row["item_id"]
        if item not in truth:
            continue
        revisions.add(row.get("harness_revision"))
        total += 1
        truncated += 1 if row.get("truncated") else 0
        parsed = parse_row_object(row.get("text") or "", task)
        if parsed is not None:
            valid += 1
        gt = truth[item]
        if task == "cell_linking":
            pred = parse_links(parsed) if parsed else {}
            cost, aogm0 = aogm(gt, pred)
            gold = {(s, t, k) for (s, t), k in gt.items()}
            got = {(s, t, k) for (s, t), kinds in pred.items() for k in kinds}
            per[item] = {"cost": cost, "aogm0": aogm0,
                         "pooled_f1": _f1(len(gold & got), len(got - gold), len(gold - got))}
        else:
            gt_m, proposal = graphs[item]
            base_cost, aogm0 = aogm(gt_m, {p: {k} for p, k in proposal.items()})
            if task == "track_repair":
                pred = apply_operations(parsed.get("operations") if parsed else None, proposal)
                cost, _ = aogm(gt_m, pred)
                per[item] = {"cost": cost, "aogm0": aogm0, "base_cost": base_cost}
            else:
                flags = parse_flags(parsed) if parsed else set()
                cost, _ = aogm(gt_m, triage(flags, proposal, gt_m))
                per[item] = {"cost": cost, "aogm0": aogm0, "base_cost": base_cost, "flags": len(flags)}
    # items with no row at this condition
    for item in truth:
        if item in per:
            continue
        gt = truth[item]
        if task == "cell_linking":
            cost, aogm0 = aogm(gt, {})
            per[item] = {"cost": cost, "aogm0": aogm0, "pooled_f1": 0.0, "missing": True}
        else:
            gt_m, proposal = graphs[item]
            base_cost, aogm0 = aogm(gt_m, {p: {k} for p, k in proposal.items()})
            per[item] = {"cost": base_cost, "aogm0": aogm0, "base_cost": base_cost, "flags": 0, "missing": True}
    health = {"n": total, "n_items": len(truth),
              "valid_json": valid / total if total else 0.0,
              "truncated": truncated / total if total else 0.0}
    return per, revisions, health


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


def summarize(task: str, per: dict, nn_per=None) -> dict:
    records = list(per.values())
    out = {"lnk_sub": lnk(records), "n_missing": sum(1 for r in records if r.get("missing"))}
    if task == "cell_linking" and nn_per is not None:
        out["d_nn"] = out["lnk_sub"] - lnk(list(nn_per.values()))
    if task in ("track_repair", "track_inspection"):
        out["lnk_proposal"] = lnk(records, "base_cost")
        out["d_lnk"] = out["lnk_sub"] - out["lnk_proposal"]
    if task == "track_inspection":
        out["flags"] = sum(r.get("flags", 0) for r in records)
        out["eff_per_1k"] = 1000.0 * out["d_lnk"] / out["flags"] if out["flags"] else float("nan")
    return out


def item_lnk(rec) -> float | None:
    return 1.0 - min(rec["cost"], rec["aogm0"]) / rec["aogm0"] if rec["aogm0"] else None


def metric_key(task: str) -> str:
    return "lnk_sub" if task == "cell_linking" else "d_lnk"


def resample_metric(task: str, records) -> float:
    value = lnk(records)
    return value if task == "cell_linking" else value - lnk(records, "base_cost")


def paired(task: str, a: dict, b: dict, rng: random.Random, draws: int = 4000):
    """Paired item-level bootstrap of the BENCHMARK-level metric (ratio of sums recomputed per draw)."""
    items = sorted(set(a) & set(b))
    if len(items) < 8:
        return None
    delta = resample_metric(task, [b[i] for i in items]) - resample_metric(task, [a[i] for i in items])
    boots = []
    for _ in range(draws):
        sample = [items[rng.randrange(len(items))] for _ in items]
        boots.append(resample_metric(task, [b[i] for i in sample]) - resample_metric(task, [a[i] for i in sample]))
    boots.sort()
    return delta, boots[int(0.05 * draws)], boots[int(0.95 * draws) - 1], len(items)


def spearman(xs, ys) -> float:
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rank = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rank[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rank
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")
