#!/usr/bin/env python3
"""Score a harness sweep with dataset-macro paper metrics and paired inference."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from celltrack_bench.answer_scoring import (
    f1_from_counts,
    normalized_label,
    normalized_operation,
    normalized_semantic,
    score_output,
    select_output_object,
)

from celltrack_bench import benchmark_data as B
from celltrack_bench import output_schema as S
from celltrack_bench import render as R

ROOT = R.REPO_ROOT

TASKS = B.TASKS
PAPER_PRIMARY = {
    "cell_linking": "strict_linking_semantic_f1",
    "track_repair": "repaired_graph_semantic_f1",
    "track_inspection": "primary_f1",
}
REQUIRED_RESULT_FIELDS = {
    "item_id",
    "task",
    "condition",
    "dataset",
    "text",
    "usage",
    "truncated",
    "budget_exhausted",
    "hit_safety_cap",
    "benchmark_identity",
    "run_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "hard")
    parser.add_argument(
        "--benchmark-items-root",
        "--benchmark-item-root",
        dest="benchmark_items_root",
        type=Path,
        default=None,
    )
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument(
        "--metric",
        default="paper",
        help="'paper' uses task-specific primaries; any other value names one scored-row field",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="expected tasks; defaults to run config")
    parser.add_argument("--conditions", nargs="+", type=int, default=None, help="expected conditions")
    parser.add_argument("--sample", type=int, default=0, help="expected dataset-spread smoke item count")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--out", type=Path, default=None, help="per-call scored JSONL")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--require-healthy", action="store_true", help="exit nonzero when any health gate fails")
    parser.add_argument(
        "--require-clean-output",
        action="store_true",
        help="also fail for strict-JSON, parse, or schema errors (intended for smoke runs)",
    )
    return parser.parse_args()


def strict_json_object(text: str) -> bool:
    """True only when the complete response is one standards-compliant JSON object."""
    try:
        return isinstance(json.loads((text or "").strip()), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def _edge(row: dict[str, Any]) -> tuple[str, str, str | None] | None:
    source = normalized_label(row.get("source") or row.get("source_id"), "S")
    target = normalized_label(row.get("target") or row.get("target_id"), "T")
    semantic = normalized_semantic(row.get("semantic") or row.get("type"))
    if source is None or target is None:
        return None
    return source, target, semantic


def _metadata_graph(rows: Any) -> dict[tuple[str, str], str | None]:
    graph: dict[tuple[str, str], str | None] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        edge = _edge(row)
        if edge is not None:
            source, target, semantic = edge
            graph[(source, target)] = semantic
    return graph


def repaired_graph_metrics(
    parsed: dict[str, Any] | None, meta: dict[str, Any]
) -> dict[str, Any]:
    """Apply predicted operations to HOCT's graph, then score the resulting semantic edges."""
    if not isinstance(meta.get("hoct_edges"), list) or not isinstance(meta.get("gt_edges"), list):
        raise ValueError(f"{meta.get('item_id')}: metadata needs hoct_edges and gt_edges lists")
    graph_map = _metadata_graph(meta.get("hoct_edges"))
    target_graph = _metadata_graph(meta.get("gt_edges"))
    graph = {(source, target, semantic) for (source, target), semantic in graph_map.items()}
    operations = parsed.get("operations") if isinstance(parsed, dict) else None
    if isinstance(operations, list):
        for row in operations:
            if not isinstance(row, dict):
                continue
            operation = normalized_operation(row.get("op") or row.get("operation") or row.get("action"))
            edge = _edge(row)
            if edge is None:
                continue
            source, target, semantic = edge
            if operation == "add" and semantic is not None:
                graph.add((source, target, semantic))
            elif operation == "delete" and semantic is not None:
                graph.discard((source, target, semantic))
            elif operation == "change_semantic":
                from_semantic = normalized_semantic(row.get("from_semantic") or row.get("from"))
                to_semantic = normalized_semantic(
                    row.get("to_semantic") or row.get("to") or row.get("semantic")
                )
                if from_semantic is not None and to_semantic is not None:
                    graph.discard((source, target, from_semantic))
                    graph.add((source, target, to_semantic))

    predicted_endpoints = {(source, target) for source, target, _semantic in graph}
    target_endpoints = set(target_graph)
    predicted_semantic = graph
    target_semantic = {
        (source, target, semantic) for (source, target), semantic in target_graph.items()
    }

    def block(predicted: set[Any], target: set[Any], prefix: str) -> dict[str, Any]:
        tp = len(predicted & target)
        fp = len(predicted - target)
        fn = len(target - predicted)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        return {
            f"{prefix}_tp": tp,
            f"{prefix}_fp": fp,
            f"{prefix}_fn": fn,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
        }

    return {
        **block(predicted_endpoints, target_endpoints, "repaired_graph_endpoint"),
        **block(predicted_semantic, target_semantic, "repaired_graph_semantic"),
        "repaired_graph_exact_match": predicted_semantic == target_semantic,
        "repaired_graph_predicted_count": len(predicted_semantic),
        "repaired_graph_target_count": len(target_semantic),
    }


def _explicit_link_edges(payload: dict[str, Any] | None) -> set[tuple[str, str, str | None]]:
    """Parse links without the shared scorer's source-degree semantic inference."""
    if not isinstance(payload, dict):
        return set()
    raw_links = payload.get("links")
    edges: set[tuple[str, str, str | None]] = set()
    if isinstance(raw_links, list):
        rows = raw_links
    elif isinstance(raw_links, dict):
        rows = [
            {"source": source, **(value if isinstance(value, dict) else {"target": value})}
            for source, value in raw_links.items()
        ]
    else:
        return edges
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = normalized_label(row.get("source") or row.get("source_id"), "S")
        raw_target = row.get("target") if "target" in row else row.get("target_id")
        targets = raw_target if isinstance(raw_target, list) else [raw_target]
        semantic = normalized_semantic(row.get("semantic") or row.get("type"))
        for value in targets:
            target = normalized_label(value, "T")
            if source is not None and target is not None:
                edges.add((source, target, semantic))
    return edges


def _explicit_unlinked_sources(payload: dict[str, Any] | None) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    raw = payload.get("unlinked_sources")
    values = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    return {
        source
        for value in values
        if (source := normalized_label(value, "S")) is not None
    }


def strict_linking_metrics(
    parsed: dict[str, Any] | None, answer: dict[str, Any]
) -> dict[str, Any]:
    """Score explicit semantic labels; unlike the legacy scorer, infer none from degree."""
    predicted_semantic = _explicit_link_edges(parsed)
    target_semantic = _explicit_link_edges(answer)
    predicted_endpoints = {(source, target) for source, target, _ in predicted_semantic}
    target_endpoints = {(source, target) for source, target, _ in target_semantic}

    def block(predicted: set[Any], target: set[Any], prefix: str) -> dict[str, Any]:
        tp = len(predicted & target)
        fp = len(predicted - target)
        fn = len(target - predicted)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        return {
            f"{prefix}_tp": tp,
            f"{prefix}_fp": fp,
            f"{prefix}_fn": fn,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
        }

    predicted_unlinked = _explicit_unlinked_sources(parsed)
    target_unlinked = _explicit_unlinked_sources(answer)
    result = {
        **block(predicted_endpoints, target_endpoints, "strict_linking_endpoint"),
        **block(predicted_semantic, target_semantic, "strict_linking_semantic"),
        "strict_linking_unlinked_exact": predicted_unlinked == target_unlinked,
        "strict_linking_exact_match": (
            predicted_semantic == target_semantic
            and predicted_unlinked == target_unlinked
        ),
    }
    for semantic in ("continuation", "division"):
        result.update(
            block(
                {edge for edge in predicted_semantic if edge[2] == semantic},
                {edge for edge in target_semantic if edge[2] == semantic},
                f"{semantic}_edge",
            )
        )
    return result


def strict_division_metrics(
    parsed: dict[str, Any] | None, answer: dict[str, Any]
) -> dict[str, Any]:
    """Count a division only when its complete, exclusively division-labelled daughter set matches."""
    pred_edges = _explicit_link_edges(parsed)
    gt_edges = _explicit_link_edges(answer)
    pred_by_source: dict[str, set[tuple[str, str]]] = defaultdict(set)
    gt_divisions: dict[str, set[str]] = defaultdict(set)
    for source, target, semantic in pred_edges:
        pred_by_source[source].add((target, semantic))
    for source, target, semantic in gt_edges:
        if semantic == "division":
            gt_divisions[source].add(target)

    predicted_parents = {
        source
        for source, edges in pred_by_source.items()
        if any(semantic == "division" for _target, semantic in edges)
    }
    def exact_sources(target_events: dict[str, set[str]]) -> set[str]:
        exact: set[str] = set()
        for source, daughters in target_events.items():
            predicted = pred_by_source.get(source, set())
            if (
                {target for target, _semantic in predicted} == daughters
                and len(predicted) == len(daughters)
                and all(semantic == "division" for _target, semantic in predicted)
            ):
                exact.add(source)
        return exact

    correct_sources = exact_sources(gt_divisions)
    correct = len(correct_sources)
    tp = correct
    fp = len(predicted_parents) - correct
    fn = len(gt_divisions) - correct
    precision, recall, f1 = f1_from_counts(tp, fp, fn)

    binary_divisions = {
        source: daughters for source, daughters in gt_divisions.items() if len(daughters) == 2
    }
    binary_correct = exact_sources(binary_divisions)
    # Predictions on annotation-defined non-binary branches belong to the all-branch metric,
    # not to this canonical two-daughter diagnostic. False alarms on non-division sources remain.
    binary_predicted_parents = {
        source
        for source in predicted_parents
        if source in binary_divisions or source not in gt_divisions
    }
    binary_tp = len(binary_correct)
    binary_fp = len(binary_predicted_parents) - binary_tp
    binary_fn = len(binary_divisions) - binary_tp
    binary_precision, binary_recall, binary_f1 = f1_from_counts(
        binary_tp, binary_fp, binary_fn
    )
    return {
        "division_event_tp": tp,
        "division_event_fp": fp,
        "division_event_fn": fn,
        "division_event_precision": precision,
        "division_event_recall": recall,
        "division_event_f1": f1,
        "division_event_predicted": len(predicted_parents),
        "division_event_target": len(gt_divisions),
        "division_event_exact_match": fp == 0 and fn == 0,
        "binary_division_event_tp": binary_tp,
        "binary_division_event_fp": binary_fp,
        "binary_division_event_fn": binary_fn,
        "binary_division_event_precision": binary_precision,
        "binary_division_event_recall": binary_recall,
        "binary_division_event_f1": binary_f1,
        "binary_division_event_predicted": len(binary_predicted_parents),
        "binary_division_event_target": len(binary_divisions),
        "binary_division_event_exact_match": binary_fp == 0 and binary_fn == 0,
    }


def _load_run_config(results: Path) -> dict[str, Any]:
    path = B.run_config_path(results)
    if not path.exists():
        return {}
    value = B.read_json(path)
    return value if isinstance(value, dict) else {}


def _selected_metric(row: dict[str, Any], task: str, metric: str) -> float | None:
    field = PAPER_PRIMARY[task] if metric == "paper" else metric
    value = row.get(field)
    return float(value) if isinstance(value, (int, float, bool)) else None


def dataset_macro(
    rows: Iterable[dict[str, Any]], condition: int, task: str, metric: str
) -> tuple[float | None, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] != condition or row["task"] != task:
            continue
        value = _selected_metric(row, task, metric)
        if value is not None:
            grouped[str(row["dataset"])].append(value)
    per_dataset = {dataset: statistics.fmean(values) for dataset, values in grouped.items() if values}
    return (statistics.fmean(per_dataset.values()) if per_dataset else None), per_dataset


def overall_score(
    rows: Iterable[dict[str, Any]], condition: int, tasks: Iterable[str], metric: str
) -> float | None:
    values = [dataset_macro(rows, condition, task, metric)[0] for task in tasks]
    return statistics.fmean(value for value in values if value is not None) if all(
        value is not None for value in values
    ) else None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    index = min(len(values) - 1, max(0, int(fraction * len(values))))
    return sorted(values)[index]


def hierarchical_score_ci(
    rows: list[dict[str, Any]],
    condition: int,
    tasks: tuple[str, ...],
    metric: str,
    n: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] != condition or row["task"] not in tasks:
            continue
        value = _selected_metric(row, row["task"], metric)
        if value is not None:
            grouped[(str(row["dataset"]), row["task"])].append(value)
    datasets = sorted(
        dataset
        for dataset in {key[0] for key in grouped}
        if all(grouped.get((dataset, task)) for task in tasks)
    )
    if not datasets or n <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n):
        sampled_datasets = rng.choices(datasets, k=len(datasets))
        task_scores: list[float] = []
        for task in tasks:
            dataset_scores = []
            for dataset in sampled_datasets:
                values = grouped[(dataset, task)]
                dataset_scores.append(statistics.fmean(rng.choices(values, k=len(values))))
            task_scores.append(statistics.fmean(dataset_scores))
        draws.append(statistics.fmean(task_scores))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def hierarchical_paired_ci(
    rows: list[dict[str, Any]],
    condition_a: int,
    condition_b: int,
    tasks: tuple[str, ...],
    metric: str,
    n: int,
    seed: int,
) -> tuple[float, float, float, int, int, list[float]]:
    values: dict[tuple[str, str, int], tuple[str, float]] = {}
    for row in rows:
        if row["task"] not in tasks or row["condition"] not in (condition_a, condition_b):
            continue
        value = _selected_metric(row, row["task"], metric)
        if value is not None:
            values[(row["item_id"], row["task"], row["condition"])] = (
                str(row["dataset"]), value
            )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    paired_rows: list[tuple[str, str, float]] = []
    for (item_id, task, condition), (dataset, value_a) in values.items():
        if condition != condition_a:
            continue
        other = values.get((item_id, task, condition_b))
        if other is None or other[0] != dataset:
            continue
        diff = other[1] - value_a
        grouped[(dataset, task)].append(diff)
        paired_rows.append((dataset, task, diff))
    datasets = sorted(
        dataset
        for dataset in {key[0] for key in grouped}
        if all(grouped.get((dataset, task)) for task in tasks)
    )
    if not datasets:
        return float("nan"), float("nan"), float("nan"), 0, 0, []
    flat_diffs = [diff for dataset, _task, diff in paired_rows if dataset in datasets]
    point = statistics.fmean(
        statistics.fmean(statistics.fmean(grouped[(dataset, task)]) for dataset in datasets)
        for task in tasks
    )
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n):
        sampled_datasets = rng.choices(datasets, k=len(datasets))
        task_scores = []
        for task in tasks:
            dataset_scores = []
            for dataset in sampled_datasets:
                diffs = grouped[(dataset, task)]
                dataset_scores.append(statistics.fmean(rng.choices(diffs, k=len(diffs))))
            task_scores.append(statistics.fmean(dataset_scores))
        draws.append(statistics.fmean(task_scores))
    return (
        point,
        _percentile(draws, 0.025),
        _percentile(draws, 0.975),
        len(flat_diffs),
        len(datasets),
        flat_diffs,
    )


def sign_test_p(diffs: list[float]) -> float:
    pos = sum(diff > 0 for diff in diffs)
    neg = sum(diff < 0 for diff in diffs)
    count = pos + neg
    if count == 0:
        return 1.0
    from math import comb

    tail = sum(comb(count, index) for index in range(min(pos, neg) + 1)) / (2**count)
    return min(1.0, 2 * tail)


def select_harness(
    rows: list[dict[str, Any]],
    conditions: tuple[int, ...],
    tasks: tuple[str, ...],
    metric: str,
    bootstrap: int,
    seed: int,
    ineligible: set[int] | None = None,
) -> dict[str, Any]:
    """Apply the pre-specified score, paired-CI, then cost selection rule."""
    excluded = ineligible or set()
    scores = {
        condition: overall_score(rows, condition, tasks, metric)
        for condition in conditions
        if condition not in excluded
    }
    scores = {condition: score for condition, score in scores.items() if score is not None}
    if not scores:
        raise ValueError("no eligible condition has a complete overall score")
    point_best = max(scores, key=lambda condition: (scores[condition], -condition))
    comparisons: dict[int, dict[str, Any]] = {}
    contenders = [point_best]
    for condition in sorted(scores):
        if condition == point_best:
            continue
        point, lo, hi, item_count, dataset_count, _diffs = hierarchical_paired_ci(
            rows,
            condition,
            point_best,
            tasks,
            metric,
            bootstrap,
            seed + condition * 100 + point_best,
        )
        unresolved = lo <= 0 <= hi
        comparisons[condition] = {
            "best_minus_condition": point,
            "ci95": [lo, hi],
            "paired_items": item_count,
            "datasets": dataset_count,
            "statistically_unresolved": unresolved,
        }
        if unresolved:
            contenders.append(condition)

    costs = {}
    for condition in sorted(scores):
        subset = [row for row in rows if row["condition"] == condition]
        costs[condition] = {
            "mean_input_tokens": statistics.fmean(
                float(row.get("input_tokens", 0)) for row in subset
            ),
            "mean_tool_calls": statistics.fmean(
                float(row.get("tool_calls", 0)) for row in subset
            ),
        }
    selected = min(
        contenders,
        key=lambda condition: (
            costs[condition]["mean_input_tokens"],
            costs[condition]["mean_tool_calls"],
            condition,
        ),
    )
    return {
        "selected_condition": selected,
        "point_best_condition": point_best,
        "eligible_conditions": sorted(scores),
        "ineligible_conditions": sorted(set(conditions) & excluded),
        "statistically_unresolved_contenders": sorted(contenders),
        "overall_scores": scores,
        "costs": costs,
        "comparisons_to_point_best": comparisons,
        "rule": "highest overall score; among conditions unresolved from the best, minimize mean input tokens then mean tool calls",
    }


def division_summary(
    rows: list[dict[str, Any]], condition: int, prefix: str = "division_event"
) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row["task"] == "cell_linking" and row["condition"] == condition
    ]
    per_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    total: Counter[str] = Counter()
    for row in subset:
        counts = {
            name: int(row.get(f"{prefix}_{name}", 0)) for name in ("tp", "fp", "fn")
        }
        per_dataset[str(row["dataset"])].update(counts)
        total.update(counts)
    precision, recall, f1 = f1_from_counts(total["tp"], total["fp"], total["fn"])
    dataset_f1 = []
    for counts in per_dataset.values():
        # tp + fn is the fixed number of annotated events in this dataset.
        # Do not let a condition's false alarms change the macro denominator.
        if counts["tp"] + counts["fn"]:
            dataset_f1.append(f1_from_counts(counts["tp"], counts["fp"], counts["fn"])[2])
    return {
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dataset_macro_f1": statistics.fmean(dataset_f1) if dataset_f1 else None,
        "datasets": len(dataset_f1),
        "false_positives_in_datasets_without_events": sum(
            counts["fp"]
            for counts in per_dataset.values()
            if counts["tp"] + counts["fn"] == 0
        ),
    }


def score_rows(
    raw_rows: list[dict[str, Any]],
    answers: dict[tuple[str, str], dict[str, Any]],
    items: dict[str, dict[str, Any]],
    items_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    scored: list[dict[str, Any]] = []
    errors: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str, int]] = set()
    for index, row in enumerate(raw_rows, 1):
        missing_fields = sorted(REQUIRED_RESULT_FIELDS - set(row))
        if missing_fields:
            errors.append(f"row {index}: missing fields {missing_fields}")
        try:
            key = (str(row["item_id"]), str(row["task"]), int(row["condition"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {index}: invalid key fields: {exc}")
            continue
        if key in seen:
            errors.append(f"duplicate result key {key}")
            continue
        seen.add(key)
        answer_key = (key[0], key[1])
        if answer_key not in answers:
            errors.append(f"{key}: missing answer key")
            continue
        if key[0] not in items:
            errors.append(f"{key}: item absent from split manifest")
            continue
        if key[0] not in metadata:
            metadata[key[0]] = B.load_item_metadata(items_root, key[0])
        meta = metadata[key[0]]
        text = row.get("text") or ""
        parsed, _source = select_output_object(text, key[1])
        metrics = score_output(key[1], parsed, answers[answer_key])
        strict_schema_errors = S.validate_output(key[1], parsed, meta)
        extras: dict[str, Any] = {
            "strict_json": strict_json_object(text),
            "strict_schema_valid": not strict_schema_errors,
            "strict_schema_errors": strict_schema_errors,
        }
        if key[1] == "track_repair":
            extras.update(
                {
                    "operation_f1": metrics.get("primary_f1", 0.0),
                    "operation_precision": metrics.get("primary_precision", 0.0),
                    "operation_recall": metrics.get("primary_recall", 0.0),
                    "operation_exact_match": metrics.get("exact_match", False),
                }
            )
            try:
                extras.update(repaired_graph_metrics(parsed, meta))
            except ValueError as exc:
                errors.append(f"{key}: {exc}")
        elif key[1] == "cell_linking":
            extras.update(strict_linking_metrics(parsed, answers[answer_key]))
            extras.update(strict_division_metrics(parsed, answers[answer_key]))
        manifest_item = items[key[0]]
        dataset = str(manifest_item["dataset"])
        if str(row.get("dataset")) != dataset:
            errors.append(f"{key}: row dataset {row.get('dataset')!r} != manifest {dataset!r}")
        scored.append(
            {
                "item_id": key[0],
                "task": key[1],
                "condition": key[2],
                "dataset": dataset,
                "sequence": str(manifest_item["sequence"]),
                "proxy_strata": row.get("proxy_strata"),
                "benchmark_identity": row.get("benchmark_identity"),
                "run_id": row.get("run_id"),
                "tool_calls": row.get("tool_calls", 0),
                "tool_failures": sum(
                    not entry.get("ok", False) for entry in (row.get("tool_log") or [])
                ),
                "truncated": float(bool(row.get("truncated"))),
                "recovered_incomplete_turns": int(
                    row.get("recovered_incomplete_turns", 0)
                ),
                "budget_exhausted": float(bool(row.get("budget_exhausted"))),
                "hit_safety_cap": float(bool(row.get("hit_safety_cap"))),
                "input_tokens": (row.get("usage") or {}).get("input_tokens", 0),
                "output_tokens": (row.get("usage") or {}).get("output_tokens", 0),
                "reasoning_tokens": (row.get("usage") or {}).get("reasoning_tokens", 0),
                **{
                    name: float(value)
                    for name, value in metrics.items()
                    if isinstance(value, (int, float, bool))
                },
                **{
                    name: float(value) if isinstance(value, (int, float, bool)) else value
                    for name, value in extras.items()
                },
            }
        )
    return scored, errors


def health_report(
    scored: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    errors: list[str],
    expected_items: list[dict[str, Any]],
    tasks: tuple[str, ...],
    conditions: tuple[int, ...],
    benchmark_identity: str,
) -> dict[str, Any]:
    expected = {
        (str(item["item_id"]), task, condition)
        for item in expected_items
        for task in tasks
        for condition in conditions
    }
    observed = {(row["item_id"], row["task"], row["condition"]) for row in scored}
    unexpected = observed - expected
    missing = expected - observed
    counts = Counter((row["task"], row["condition"]) for row in scored if (row["item_id"], row["task"], row["condition"]) in expected)
    imbalance = {
        f"{task}/c{condition}": counts[(task, condition)]
        for task in tasks
        for condition in conditions
        if counts[(task, condition)] != len(expected_items)
    }
    identity_mismatch = sum(
        row.get("benchmark_identity") != benchmark_identity for row in raw_rows
    )
    totals = {
        "strict_json_failures": sum(not bool(row.get("strict_json")) for row in scored),
        "lenient_json_failures": sum(not bool(row.get("valid_json")) for row in scored),
        "schema_failures": sum(not bool(row.get("strict_schema_valid")) for row in scored),
        "truncated": sum(bool(row.get("truncated")) for row in scored),
        "recovered_incomplete_turns": sum(
            int(row.get("recovered_incomplete_turns", 0)) for row in scored
        ),
        "budget_exhausted": sum(bool(row.get("budget_exhausted")) for row in scored),
        "safety_cap_hits": sum(bool(row.get("hit_safety_cap")) for row in scored),
        "tool_failures": sum(int(row.get("tool_failures", 0)) for row in scored),
        "identity_mismatches": identity_mismatch,
    }
    by_condition: dict[int, dict[str, Any]] = {}
    for condition in conditions:
        subset = [row for row in scored if row["condition"] == condition]
        condition_totals = {
            "strict_json_failures": sum(not bool(row.get("strict_json")) for row in subset),
            "lenient_json_failures": sum(not bool(row.get("valid_json")) for row in subset),
            "schema_failures": sum(not bool(row.get("strict_schema_valid")) for row in subset),
            "truncated": sum(bool(row.get("truncated")) for row in subset),
            "recovered_incomplete_turns": sum(
                int(row.get("recovered_incomplete_turns", 0)) for row in subset
            ),
            "budget_exhausted": sum(bool(row.get("budget_exhausted")) for row in subset),
            "safety_cap_hits": sum(bool(row.get("hit_safety_cap")) for row in subset),
            "tool_failures": sum(int(row.get("tool_failures", 0)) for row in subset),
        }
        expected_for_condition = len(expected_items) * len(tasks)
        infrastructure_failures = {
            name: condition_totals[name]
            for name in ("truncated", "budget_exhausted", "safety_cap_hits", "tool_failures")
        }
        output_failures = {
            name: condition_totals[name]
            for name in ("strict_json_failures", "lenient_json_failures", "schema_failures")
        }
        by_condition[condition] = {
            "passed": len(subset) == expected_for_condition and not any(infrastructure_failures.values()),
            "clean_output": not any(output_failures.values()),
            "rows": len(subset),
            "expected_rows": expected_for_condition,
            **condition_totals,
        }
    infrastructure_total_names = (
        "truncated",
        "budget_exhausted",
        "safety_cap_hits",
        "tool_failures",
        "identity_mismatches",
    )
    output_total_names = (
        "strict_json_failures",
        "lenient_json_failures",
        "schema_failures",
    )
    passed = not (
        errors
        or missing
        or unexpected
        or imbalance
        or any(totals[name] for name in infrastructure_total_names)
    )
    clean_output = not any(totals[name] for name in output_total_names)
    return {
        "passed": passed,
        "clean_output": clean_output,
        "expected_rows": len(expected),
        "observed_rows": len(observed),
        "missing_rows": len(missing),
        "unexpected_rows": len(unexpected),
        "imbalanced_cells": imbalance,
        "scoring_errors": errors,
        "by_condition": by_condition,
        **totals,
    }


def main() -> None:
    args = parse_args()
    if args.sample and args.limit:
        raise SystemExit("--sample and --limit are mutually exclusive")
    try:
        raw_rows = B.read_jsonl(args.results)
    except B.BenchmarkDataError as exc:
        raise SystemExit(str(exc)) from exc
    if not raw_rows:
        raise SystemExit(f"no rows in {args.results}")
    run_config = _load_run_config(args.results)
    configured_tasks = tuple(run_config.get("tasks") or ())
    configured_conditions = tuple(int(value) for value in (run_config.get("conditions") or ()))
    tasks = tuple(args.tasks or configured_tasks or sorted({str(row.get("task")) for row in raw_rows}))
    conditions = tuple(
        args.conditions
        or configured_conditions
        or sorted({int(row.get("condition")) for row in raw_rows if row.get("condition") is not None})
    )
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise SystemExit(f"unknown tasks in expected selection: {unknown_tasks}")

    split_dir = args.split_dir.expanduser().resolve()
    items_root = B.resolve_items_root(
        split_dir, args.split, args.benchmark_items_root, R.DEFAULT_ITEMS_ROOT
    )
    try:
        identity = B.benchmark_identity(
            split_dir, items_root, args.split, label=args.benchmark_id
        )
        all_items, answers = B.validate_benchmark(split_dir, items_root, args.split, TASKS)
        expected_items = B.select_items(all_items, sample=args.sample, limit=args.limit)
    except B.BenchmarkDataError as exc:
        raise SystemExit(f"benchmark validation failed: {exc}") from exc
    item_map = {str(item["item_id"]): item for item in all_items}
    scored, scoring_errors = score_rows(raw_rows, answers, item_map, items_root)
    health = health_report(
        scored,
        raw_rows,
        scoring_errors,
        expected_items,
        tasks,
        conditions,
        identity.value,
    )

    stem = args.results.name.removesuffix(".gz").removesuffix(".jsonl")
    out = args.out or args.results.with_name(stem + "_scored.jsonl")
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in scored))

    print(
        f"\n  {len(scored)} scored calls | benchmark {identity.value} | "
        f"conditions {list(conditions)} | metric {args.metric}\n"
    )
    print("  dataset-macro primary (equal weight per dataset)")
    print(f"  {'task':<18}" + "".join(f"  c{condition:<10}" for condition in conditions))
    task_scores: dict[str, dict[int, float | None]] = {}
    for task in tasks:
        task_scores[task] = {}
        cells = []
        for condition in conditions:
            value, per_dataset = dataset_macro(scored, condition, task, args.metric)
            task_scores[task][condition] = value
            cells.append(f"  {value:.3f} (d={len(per_dataset):<2})" if value is not None else "  --          ")
        print(f"  {task:<18}" + "".join(cells))
    overall: dict[int, float | None] = {}
    print(f"  {'OVERALL':<18}", end="")
    for condition in conditions:
        value = overall_score(scored, condition, tasks, args.metric)
        overall[condition] = value
        print(f"  {value:.3f}       " if value is not None else "  --          ", end="")
    print()

    print("\n  hierarchical 95% CIs for the unweighted task mean")
    for condition in conditions:
        lo, hi = hierarchical_score_ci(
            scored, condition, tasks, args.metric, args.bootstrap, args.seed + condition
        )
        print(f"  c{condition}: [{lo:.4f}, {hi:.4f}]")

    print("\n  strict complete-daughter division events")
    division: dict[int, dict[str, Any]] = {}
    binary_division: dict[int, dict[str, Any]] = {}
    for condition in conditions:
        summary = division_summary(scored, condition)
        division[condition] = summary
        binary_summary = division_summary(scored, condition, "binary_division_event")
        binary_division[condition] = binary_summary
        macro = summary["dataset_macro_f1"]
        macro_text = f"{macro:.3f}" if macro is not None else "--"
        binary_macro = binary_summary["dataset_macro_f1"]
        binary_macro_text = f"{binary_macro:.3f}" if binary_macro is not None else "--"
        print(
            f"  c{condition}: P={summary['precision']:.3f} R={summary['recall']:.3f} "
            f"F1={summary['f1']:.3f} dataset-macro={macro_text} "
            f"TP/FP/FN={summary['tp']}/{summary['fp']}/{summary['fn']} | "
            f"binary F1={binary_summary['f1']:.3f} dataset-macro={binary_macro_text} "
            f"TP/FP/FN={binary_summary['tp']}/{binary_summary['fp']}/{binary_summary['fn']}"
        )

    if "track_repair" in tasks:
        print("\n  track repair diagnostics (dataset-macro)")
        for condition in conditions:
            operation, _ = dataset_macro(scored, condition, "track_repair", "operation_f1")
            repaired, _ = dataset_macro(
                scored, condition, "track_repair", "repaired_graph_semantic_f1"
            )
            operation_text = f"{operation:.3f}" if operation is not None else "--"
            repaired_text = f"{repaired:.3f}" if repaired is not None else "--"
            print(f"  c{condition}: operation_F1={operation_text} repaired_graph_F1={repaired_text}")

    print("\n  harness health gates")
    print(
        f"  {'cond':<6}{'gate':>7}{'rows':>10}{'strict':>9}{'lenient':>9}"
        f"{'schema':>9}{'trunc':>8}{'recover':>9}{'budget':>8}{'cap':>6}{'tool':>7}"
    )
    for condition in conditions:
        condition_health = health["by_condition"][condition]
        row_count = f"{condition_health['rows']}/{condition_health['expected_rows']}"
        print(
            f"  c{condition:<5}{('PASS' if condition_health['passed'] else 'FAIL'):>7}"
            f"{row_count:>10}"
            f"{condition_health['strict_json_failures']:>9}"
            f"{condition_health['lenient_json_failures']:>9}"
            f"{condition_health['schema_failures']:>9}"
            f"{condition_health['truncated']:>8}"
            f"{condition_health['recovered_incomplete_turns']:>9}"
            f"{condition_health['budget_exhausted']:>8}"
            f"{condition_health['safety_cap_hits']:>6}"
            f"{condition_health['tool_failures']:>7}"
        )
    print(
        f"  infrastructure={'PASS' if health['passed'] else 'FAIL'} "
        f"output={'CLEAN' if health['clean_output'] else 'INVALID_ROWS_PRESENT'} | "
        f"rows {health['observed_rows']}/"
        f"{health['expected_rows']} | strict_json_fail={health['strict_json_failures']} | "
        f"lenient_json_fail={health['lenient_json_failures']} | schema_fail={health['schema_failures']}"
    )
    print(
        f"  truncated={health['truncated']} "
        f"recovered_incomplete_turns={health['recovered_incomplete_turns']} "
        f"budget_exhausted={health['budget_exhausted']} "
        f"safety_cap={health['safety_cap_hits']} tool_failures={health['tool_failures']} "
        f"identity_mismatch={health['identity_mismatches']}"
    )
    if health["missing_rows"] or health["unexpected_rows"] or health["imbalanced_cells"]:
        print(
            f"  missing={health['missing_rows']} unexpected={health['unexpected_rows']} "
            f"imbalanced={health['imbalanced_cells']}"
        )
    for error in health["scoring_errors"][:10]:
        print(f"  ERROR {error}")

    valid_overall = {condition: value for condition, value in overall.items() if value is not None}
    pairs = [(a, b) for a, b in zip(conditions, conditions[1:])]
    if len(valid_overall) > 2:
        best = max(valid_overall, key=lambda condition: valid_overall[condition])
        pairs += [
            (condition, best)
            for condition in conditions
            if condition != best and (condition, best) not in pairs
        ]
    pairs = list(dict.fromkeys(pairs))
    comparisons: list[dict[str, Any]] = []
    print("\n  paired hierarchical comparisons (second minus first)")
    print(f"  {'pair':<10}{'items':>7}{'data':>6}{'diff':>10}{'95% CI':>22}{'sign p':>10}  verdict")
    for a, b in pairs:
        point, lo, hi, item_count, dataset_count, diffs = hierarchical_paired_ci(
            scored, a, b, tasks, args.metric, args.bootstrap, args.seed + a * 100 + b
        )
        if not diffs:
            continue
        p_value = sign_test_p(diffs)
        verdict = (
            "inconclusive"
            if lo <= 0 <= hi
            else (f"c{b} better" if point > 0 else f"c{a} better")
        )
        print(
            f"  c{a}->c{b}{'':<4}{item_count:>7}{dataset_count:>6}{point:>+10.4f}"
            f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p_value:>10.3f}  {verdict}"
        )
        comparisons.append(
            {
                "condition_a": a,
                "condition_b": b,
                "mean_difference": point,
                "ci95": [lo, hi],
                "paired_items": item_count,
                "datasets": dataset_count,
                "sign_test_p": p_value,
                "verdict": verdict,
            }
        )

    selection = select_harness(
        scored,
        conditions,
        tasks,
        args.metric,
        args.bootstrap,
        args.seed,
        ineligible={7},
    )
    print("\n  pre-specified harness selection")
    print(
        f"  point best: c{selection['point_best_condition']} | "
        f"selected: c{selection['selected_condition']} | "
        f"unresolved contenders: {selection['statistically_unresolved_contenders']}"
    )

    summary_path = args.summary_json or args.results.with_name(stem + "_summary.json")
    summary_payload = {
        "schema_version": 1,
        "results": str(args.results.resolve()),
        "benchmark_identity": identity.value,
        "split": args.split,
        "metric": args.metric,
        "tasks": list(tasks),
        "conditions": list(conditions),
        "task_dataset_macro": task_scores,
        "overall_unweighted_tasks": overall,
        "division_events": division,
        "binary_division_events": binary_division,
        "health": health,
        "paired_comparisons": comparisons,
        "harness_selection": selection,
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    print(f"\n  -> {out}")
    print(f"  -> {summary_path}")
    if args.require_healthy and not health["passed"]:
        raise SystemExit(2)
    if args.require_clean_output and not health["clean_output"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
