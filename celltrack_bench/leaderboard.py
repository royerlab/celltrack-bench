#!/usr/bin/env python3
"""Paper metrics (the leaderboard row of Table 2, plus the appendix breakdowns) for result files.

Every number is computed from the runner's JSONL output and the benchmark files alone; no raw images
are needed, because the nearest-centroid baseline ships precomputed in data/<benchmark>/references/.

  Primary task scores  dataset-macro primary F1 per task (celltrack_bench.score), and their mean
  Shortcut resistance  NN-residual F1 and division F1 (item means, nn_residual), linking LNK_sub and
                       dNN = LNK_sub - LNK_sub(nearest-centroid graph) (subgraph_lnk)
  Graph utility        repair dLNK and inspection dLNK relative to the HOCT proposal, and inspection
                       dLNK per 1000 flagged edges (subgraph_lnk)

Files passed together with --system are replicates: each metric is averaged over them.

  python -m celltrack_bench.leaderboard --split test --results runs/test_..._v10_<run_id>.jsonl
  python -m celltrack_bench.leaderboard --split validation --split-dir data/full --results run.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from celltrack_bench import benchmark_data as B
from celltrack_bench import nn_residual as NR
from celltrack_bench import render as R
from celltrack_bench import score as SC
from celltrack_bench import subgraph_lnk as SL

TASKS = SL.TASKS
BUDGETS = (0.10, 0.25, 0.50)
FROZEN_CONDITION = 1


@dataclass
class Context:
    """Everything about one benchmark split that the metrics need, loaded once."""

    split_dir: Path
    split: str
    items_root: Path
    items: dict[str, dict[str, Any]]
    answers: dict[tuple[str, str], dict[str, Any]]
    truth: dict[str, dict[tuple[str, str], str]]
    graphs: dict[str, tuple[dict, dict]]
    metas: dict[str, dict[str, Any]]
    dataset_of: dict[str, str]
    nn: dict[str, dict[str, str]]
    nn_per: dict[str, dict[str, float]]
    nn_lnk: float
    prop_lnk: float
    truth_nr: dict[str, set[tuple[str, str, str]]]
    residual: dict[str, set[tuple[str, str, str]]]
    gold_add: dict[str, set[tuple[str, str]]]
    gold_del: dict[str, set[tuple[str, str]]]
    gold_chg: dict[str, set[tuple[str, str]]]
    gold_flags: dict[str, list[tuple[str, str]]]
    budget_k: dict[float, dict[str, int]]


def load_context(split_dir: Path, split: str) -> Context:
    split_dir = Path(split_dir).expanduser().resolve()
    items_root = B.resolve_items_root(split_dir, split, None, R.DEFAULT_ITEMS_ROOT)
    item_list, answers = B.validate_benchmark(split_dir, items_root, split, TASKS)
    items = {str(item["item_id"]): item for item in item_list}
    truth = SL.load_truth(split, split_dir)
    graphs = SL.load_graphs(truth, items_root)
    metas = {item: B.load_item_metadata(items_root, item) for item in truth}
    nn = SL.nearest_centroid_from_references(truth, split_dir, split)
    nn_per = SL.nn_records(truth, nn)
    prop_rec = [
        {"cost": SL.aogm(gt, {p: {k} for p, k in proposal.items()})[0], "aogm0": SL.W_EA * len(gt)}
        for gt, proposal in graphs.values()
    ]
    truth_nr = {}
    for line in B.open_text(split_dir / split / "cell_linking" / "answer_key.jsonl"):
        row = json.loads(line)
        truth_nr[row["item_id"]] = NR.link_set(row["answer"])
    residual = {i: truth_nr[i] - {(s, t, "continuation") for s, t in nn[i].items()} for i in truth_nr}
    ops = {i: metas[i]["operations"] for i in truth}
    return Context(
        split_dir=split_dir,
        split=split,
        items_root=items_root,
        items=items,
        answers=answers,
        truth=truth,
        graphs=graphs,
        metas=metas,
        dataset_of={i: metas[i]["dataset"] for i in truth},
        nn=nn,
        nn_per=nn_per,
        nn_lnk=SL.lnk(list(nn_per.values())),
        prop_lnk=SL.lnk(prop_rec),
        truth_nr=truth_nr,
        residual=residual,
        gold_add={i: {(o["source"], o["target"]) for o in ops[i] if o["op"] == "add"} for i in truth},
        gold_del={i: {(o["source"], o["target"]) for o in ops[i] if o["op"] == "delete"} for i in truth},
        gold_chg={i: {(o["source"], o["target"]) for o in ops[i] if o["op"] == "change_semantic"} for i in truth},
        gold_flags={i: [(e["source"], e["target"]) for e in metas[i]["inspection"]] for i in truth},
        budget_k={f: {i: math.ceil(f * len(graphs[i][1])) for i in truth} for f in BUDGETS},
    )


# ------------------------------------------------------------------------------------ parsing helpers

def ordered_flags(parsed) -> list[tuple[str, str]]:
    """Inspection flags in emission order (stored answers carry no confidence), ok-aliases dropped."""
    rows = parsed.get("links_to_inspect") if isinstance(parsed, dict) else None
    out, seen = [], set()
    if not isinstance(rows, (list, tuple)):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = SL.normalized_label(row.get("source") or row.get("source_id"), "S")
        tgt = SL.normalized_label(row.get("target") or row.get("target_id"), "T")
        issue = str(row.get("issue") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if src is None or tgt is None or issue in SL.OK_ISSUES:
            continue
        if (src, tgt) not in seen:
            seen.add((src, tgt))
            out.append((src, tgt))
    return out


def parse_ops(parsed) -> tuple[set, set, set]:
    adds, dels, chgs = set(), set(), set()
    ops = parsed.get("operations") if isinstance(parsed, dict) else None
    if isinstance(ops, (list, tuple)):
        for row in ops:
            if not isinstance(row, dict):
                continue
            op = SL.normalized_operation(row.get("op") or row.get("operation") or row.get("action"))
            src = SL.normalized_label(row.get("source") or row.get("source_id"), "S")
            tgt = SL.normalized_label(row.get("target") or row.get("target_id"), "T")
            if src is None or tgt is None:
                continue
            if op == "add":
                adds.add((src, tgt))
            elif op == "delete":
                dels.add((src, tgt))
            elif op == "change_semantic":
                chgs.add((src, tgt))
    return adds, dels, chgs


def triage_lnk(ctx: Context, flag_lists: dict[str, list[tuple[str, str]]]) -> float:
    recs = []
    for i in ctx.truth:
        gt, proposal = ctx.graphs[i]
        cost, _ = SL.aogm(gt, SL.triage(set(flag_lists.get(i, [])), proposal, gt))
        recs.append({"cost": cost, "aogm0": SL.W_EA * len(gt)})
    return SL.lnk(recs)


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


# ------------------------------------------------------------------------------------ one result file

def primary_scores(ctx: Context, path: Path, condition: int = FROZEN_CONDITION) -> dict[str, float | None]:
    """Dataset-macro primary F1 per task, exactly as `python -m celltrack_bench.score` reports it."""
    raw_rows = [row for row in B.read_jsonl(path) if row.get("condition") == condition]
    scored, errors = SC.score_rows(raw_rows, ctx.answers, ctx.items, ctx.items_root)
    if errors:
        raise ValueError(f"{path}: {len(errors)} scoring errors, e.g. {errors[:3]}")
    out = {task: SC.dataset_macro(scored, condition, task, "paper")[0] for task in TASKS}
    values = [out[task] for task in TASKS]
    out["mean"] = statistics.fmean(values) if all(v is not None for v in values) else None
    return out


def file_metrics(ctx: Context, path: Path, condition: int = FROZEN_CONDITION) -> dict[str, Any]:
    path = Path(path)
    per = {task: SL.per_item(path, task, condition, ctx.truth, ctx.graphs)[0] for task in TASKS}
    nr, _rev, _health = NR.per_item(path, condition, ctx.truth_nr, ctx.nn, ctx.residual)

    ops_by_item: dict[str, tuple[set, set, set]] = {}
    flags_by_item: dict[str, list[tuple[str, str]]] = {}
    rows = trunc = 0
    for line in B.open_text(path):
        row = json.loads(line)
        if row.get("condition") != condition or row.get("item_id") not in ctx.truth:
            continue
        rows += 1
        trunc += bool(row.get("truncated"))
        parsed = SL.parse_row_object(row.get("text") or "", row.get("task"))
        if row.get("task") == "track_repair":
            ops_by_item[row["item_id"]] = parse_ops(parsed) if parsed else (set(), set(), set())
        elif row.get("task") == "track_inspection":
            flags_by_item[row["item_id"]] = ordered_flags(parsed) if parsed else []

    m: dict[str, Any] = {f"{task}_f1": value for task, value in primary_scores(ctx, path, condition).items()}
    m["primary_mean"] = m.pop("mean_f1")
    m["rows"] = rows
    m["truncated_fraction"] = _ratio(trunc, rows)

    # linking: shortcut resistance
    m["lnk_sub"] = SL.lnk(list(per["cell_linking"].values()))
    m["d_nn"] = m["lnk_sub"] - ctx.nn_lnk
    m["nn_residual_f1"] = statistics.fmean([v["nn_resid"] for v in nr.values() if v.get("nn_resid") is not None])
    m["division_f1"] = statistics.fmean([v["divisions"] for v in nr.values() if v.get("divisions") is not None])

    # repair: graph benefit and edit decomposition
    recs = list(per["track_repair"].values())
    m["repair_d_lnk"] = SL.lnk(recs) - SL.lnk(recs, "base_cost")
    counts = Counter()
    better = worse = 0
    for i in ctx.truth:
        adds, dels, chgs = ops_by_item.get(i, (set(), set(), set()))
        for name, got, gold in (("add", adds, ctx.gold_add[i]), ("del", dels, ctx.gold_del[i]),
                                ("chg", chgs, ctx.gold_chg[i])):
            counts[f"{name}_tp"] += len(got & gold)
            counts[f"{name}_fp"] += len(got - gold)
            counts[f"{name}_fn"] += len(gold - got)
        rec = per["track_repair"][i]
        delta = (SL.item_lnk(rec) or 0.0) - (1.0 - min(rec["base_cost"], rec["aogm0"]) / rec["aogm0"])
        better += delta > 1e-9
        worse += delta < -1e-9
    for name in ("add", "del", "chg"):
        tp, fp, fn = counts[f"{name}_tp"], counts[f"{name}_fp"], counts[f"{name}_fn"]
        m[f"repair_{name}_precision"] = _ratio(tp, tp + fp)
        m[f"repair_{name}_recall"] = _ratio(tp, tp + fn)
    n = len(ctx.truth)
    m["repair_better"], m["repair_worse"] = better / n, worse / n
    m["repair_unchanged"] = 1.0 - m["repair_better"] - m["repair_worse"]

    # inspection: oracle triage, with and without a review budget
    summary = SL.summarize("track_inspection", per["track_inspection"])
    m["inspect_d_lnk"] = summary["d_lnk"]
    m["inspect_flags"] = summary["flags"]
    m["inspect_d_lnk_per_1k_flags"] = summary["eff_per_1k"]
    for f in BUDGETS:
        capped = {i: flags_by_item.get(i, [])[: ctx.budget_k[f][i]] for i in ctx.truth}
        m[f"inspect_d_lnk_budget{int(f * 100)}"] = triage_lnk(ctx, capped) - ctx.prop_lnk
    return m


def system_metrics(ctx: Context, paths: Iterable[Path], condition: int = FROZEN_CONDITION) -> dict[str, Any]:
    """Mean of file_metrics over replicate files."""
    runs = [file_metrics(ctx, Path(p), condition) for p in paths]
    out: dict[str, Any] = {"replicates": len(runs)}
    for key in runs[0]:
        values = [run[key] for run in runs]
        if all(isinstance(v, (int, float)) for v in values):
            out[key] = statistics.fmean(values)
        else:
            out[key] = values[0] if len(set(map(str, values))) == 1 else values
    return out


def per_item_linking_lnk(ctx: Context, paths: Iterable[Path], condition: int = FROZEN_CONDITION) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    for path in paths:
        per = SL.per_item(Path(path), "cell_linking", condition, ctx.truth, ctx.graphs)[0]
        for i, rec in per.items():
            vals.setdefault(i, []).append(SL.item_lnk(rec) or 0.0)
    return {i: statistics.fmean(v) for i, v in vals.items()}


# ------------------------------------------------------------------------------------ reference rows

def _reference_link_f1(ctx: Context, edges_by_item: dict[str, list[dict[str, str]]]) -> float:
    """Score a reference graph as if it were a cell_linking answer (dataset-macro strict F1)."""
    rows = [
        {
            "item_id": i, "task": "cell_linking", "condition": FROZEN_CONDITION, "dataset": ctx.dataset_of[i],
            "text": json.dumps({"links": edges_by_item[i]}), "usage": {}, "truncated": False,
            "budget_exhausted": False, "hit_safety_cap": False, "benchmark_identity": None, "run_id": None,
        }
        for i in ctx.truth
    ]
    scored, errors = SC.score_rows(rows, ctx.answers, ctx.items, ctx.items_root)
    assert not errors, errors[:3]
    return SC.dataset_macro(scored, FROZEN_CONDITION, "cell_linking", "paper")[0]


def reference_metrics(ctx: Context) -> dict[str, dict[str, float | None]]:
    """Link F1 and LNK_sub of the reference graphs: nearest-centroid, the two other benchmark
    trackers, and the HOCT proposal that repair and inspection start from."""
    trackers = SL.load_references(ctx.split_dir, ctx.split)["trackers"]
    graphs = {
        "hungarian": {i: trackers["hungarian"][i] for i in ctx.truth},
        "trackastra": {i: trackers["trackastra"][i] for i in ctx.truth},
        "hoct": {i: ctx.metas[i]["hoct_edges"] for i in ctx.truth},
    }
    out: dict[str, dict[str, float | None]] = {"nearest_centroid": {"link_f1": None, "lnk_sub": ctx.nn_lnk}}
    for name, edges in graphs.items():
        recs = [{"cost": SL.aogm(ctx.truth[i], {p: {k} for p, k in SL.canonical(edges[i]).items()})[0],
                 "aogm0": SL.W_EA * len(ctx.truth[i])} for i in ctx.truth]
        out[name] = {"link_f1": _reference_link_f1(ctx, edges), "lnk_sub": SL.lnk(recs)}
    return out


def triage_references(ctx: Context, seed: int = 20260826, draws: int = 100) -> dict[str, dict[str, float]]:
    """Gold-oracle and random reviewers at each review budget (the appendix triage table)."""
    rng = random.Random(seed)
    gold, rand = {}, {}
    for f in BUDGETS:
        key = f"budget{int(f * 100)}"
        gold[key] = triage_lnk(ctx, {i: ctx.gold_flags[i][: ctx.budget_k[f][i]] for i in ctx.truth}) - ctx.prop_lnk
        values = []
        for _ in range(draws):
            flags = {}
            for i in ctx.truth:
                edges = list(ctx.graphs[i][1])
                flags[i] = rng.sample(edges, min(ctx.budget_k[f][i], len(edges)))
            values.append(triage_lnk(ctx, flags) - ctx.prop_lnk)
        rand[key] = statistics.fmean(values)
    gold["full"] = triage_lnk(ctx, ctx.gold_flags) - ctx.prop_lnk
    return {"gold": gold, "random": rand}


# ------------------------------------------------------------------------------------ CLI

COLUMNS = (
    ("cell_linking_f1", "Link F1", "{:.3f}"),
    ("track_repair_f1", "Repair F1", "{:.3f}"),
    ("track_inspection_f1", "Inspect F1", "{:.3f}"),
    ("primary_mean", "Mean", "{:.3f}"),
    ("nn_residual_f1", "NN-res", "{:.3f}"),
    ("division_f1", "Div", "{:.3f}"),
    ("lnk_sub", "LNK_sub", "{:.3f}"),
    ("d_nn", "dNN", "{:+.3f}"),
    ("repair_d_lnk", "Repair dLNK", "{:+.3f}"),
    ("inspect_d_lnk", "Inspect dLNK", "{:+.3f}"),
    ("inspect_d_lnk_per_1k_flags", "d/1k", "{:.3f}"),
)


def format_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    width = max([len(name) for name, _ in rows] + [6])
    head = f"{'system':<{width}}  " + "  ".join(f"{label:>12}" for _k, label, _f in COLUMNS)
    lines = [head, "-" * len(head)]
    for name, m in rows:
        cells = []
        for key, _label, fmt in COLUMNS:
            value = m.get(key)
            cells.append(f"{fmt.format(value) if isinstance(value, (int, float)) else '---':>12}")
        lines.append(f"{name:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", "--split-dir", dest="benchmark", type=Path, default=R.REPO_ROOT / "data" / "hard",
                   help="benchmark directory (data/hard or data/full)")
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--results", nargs="+", type=Path, required=True,
                   help="result JSONL file(s); several files are treated as replicates of one system")
    p.add_argument("--system", default=None, help="display name for the row")
    p.add_argument("--condition", type=int, default=FROZEN_CONDITION)
    p.add_argument("--json", type=Path, default=None, help="also write all metrics as JSON")
    args = p.parse_args()

    try:
        ctx = load_context(args.benchmark, args.split)
    except B.BenchmarkDataError as exc:
        raise SystemExit(str(exc)) from exc
    metrics = system_metrics(ctx, args.results, args.condition)
    refs = reference_metrics(ctx)
    name = args.system or Path(args.results[0]).name.split(".jsonl")[0]
    rows = [(name, metrics)] + [
        (f"[{ref}]", {"cell_linking_f1": v["link_f1"], "lnk_sub": v["lnk_sub"]}) for ref, v in refs.items()
    ]
    print(f"{args.benchmark.name}/{args.split}: {len(ctx.truth)} items, condition c{args.condition}, "
          f"{metrics['replicates']} replicate(s), {metrics['rows']:.0f} rows, "
          f"truncated {metrics['truncated_fraction']:.1%}")
    print(format_table(rows))
    if args.json:
        args.json.write_text(json.dumps({"system": name, "metrics": metrics, "references": refs}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
