#!/usr/bin/env python3
"""Validate benchmark builds and gate smoke results before a paid full sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from celltrack_bench import benchmark_data as B
from celltrack_bench import conditions as C
from celltrack_bench import output_schema as S
from celltrack_bench import render as R
from celltrack_bench.answer_scoring import score_output, select_output_object

ROOT = R.REPO_ROOT


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
    parser.add_argument("--source-data-root", type=Path, default=R.SOURCE_DATA_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(C.TASKS))
    parser.add_argument("--conditions", nargs="+", type=int, default=list(C.CONDITIONS))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-budget", type=int, default=8192)
    parser.add_argument("--results", type=Path, default=None, help="smoke JSONL to health-check")
    parser.add_argument(
        "--allow-invalid-output",
        action="store_true",
        help="report model JSON/schema failures without treating them as infrastructure failures",
    )
    return parser.parse_args()


def strict_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads((text or "").strip()), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def expected_keys(
    items: list[dict[str, Any]], tasks: list[str], conditions: list[int]
) -> set[tuple[str, str, int]]:
    return {
        (str(item["item_id"]), task, condition)
        for item in items
        for task in tasks
        for condition in conditions
    }


def check_result_rows(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    answers: dict[tuple[str, str], dict[str, Any]],
    tasks: list[str],
    conditions: list[int],
    benchmark_identity: str,
    metadata: dict[str, dict[str, Any]] | None = None,
    require_clean_output: bool = True,
) -> list[str]:
    errors: list[str] = []
    expected = expected_keys(items, tasks, conditions)
    item_by_id = {str(item["item_id"]): item for item in items}
    seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    counts: Counter[tuple[str, int]] = Counter()

    for index, row in enumerate(rows, 1):
        missing_fields = sorted(REQUIRED_RESULT_FIELDS - set(row))
        if missing_fields:
            errors.append(f"row {index}: missing fields {missing_fields}")
            continue
        key = (str(row["item_id"]), str(row["task"]), int(row["condition"]))
        if key in seen:
            errors.append(f"duplicate result key {key}")
            continue
        seen[key] = row
        counts[(key[1], key[2])] += 1
        if key not in expected:
            errors.append(f"unexpected result key {key}")
            continue
        item = item_by_id[key[0]]
        if str(row["dataset"]) != str(item["dataset"]):
            errors.append(f"{key}: dataset {row['dataset']!r} != {item['dataset']!r}")
        if row.get("benchmark_identity") != benchmark_identity:
            errors.append(f"{key}: benchmark identity mismatch")
        for flag in ("truncated", "budget_exhausted", "hit_safety_cap"):
            if row.get(flag):
                errors.append(f"{key}: {flag}=true")
        failed_tools = [entry for entry in row.get("tool_log") or [] if not entry.get("ok", False)]
        if failed_tools:
            errors.append(f"{key}: {len(failed_tools)} failed tool call(s)")
        if require_clean_output and not strict_json_object(row.get("text") or ""):
            errors.append(f"{key}: final response is not a strict JSON object")
        parsed, _source = select_output_object(row.get("text") or "", key[1])
        schema_errors = S.validate_output(
            key[1], parsed, (metadata or {}).get(key[0])
        )
        if require_clean_output and schema_errors:
            errors.append(f"{key}: strict schema errors: {schema_errors[:3]}")
        metrics = score_output(key[1], parsed, answers[(key[0], key[1])])
        if require_clean_output and not metrics.get("valid_json"):
            errors.append(f"{key}: no leniently parseable answer object")
        if require_clean_output and not metrics.get("schema_valid"):
            errors.append(f"{key}: answer schema is invalid")

    missing = sorted(expected - set(seen))
    if missing:
        errors.append(f"missing {len(missing)} expected rows; first={missing[:5]}")
    expected_per_cell = len(items)
    for task in tasks:
        for condition in conditions:
            actual = counts[(task, condition)]
            if actual != expected_per_cell:
                errors.append(
                    f"task/condition imbalance: {task}/c{condition} has {actual}, "
                    f"expected {expected_per_cell}"
                )
    return errors


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.expanduser().resolve()
    source_root = args.source_data_root.expanduser().resolve()
    items_root = B.resolve_items_root(
        split_dir, args.split, args.benchmark_items_root, R.DEFAULT_ITEMS_ROOT
    )
    try:
        identity = B.benchmark_identity(
            split_dir, items_root, args.split, label=args.benchmark_id
        )
        all_items, all_answers = B.validate_benchmark(
            split_dir, items_root, args.split, args.tasks
        )
        items = B.select_items(all_items, sample=args.sample, limit=args.limit)
    except B.BenchmarkDataError as exc:
        raise SystemExit(f"PRECHECK FAILED: {exc}") from exc

    build_errors: list[str] = []
    for item in items:
        try:
            meta = R.load_meta(
                str(item["item_id"]),
                items_root=items_root,
                source_data_root=source_root,
            )
        except Exception as exc:
            build_errors.append(
                f"{item['item_id']}: metadata/render load: {type(exc).__name__}: {exc}"
            )
            continue
        for task in args.tasks:
            for condition in args.conditions:
                try:
                    C.build(meta, task, condition, output_budget=args.output_budget)
                except Exception as exc:
                    build_errors.append(
                        f"{item['item_id']} {task} c{condition}: {type(exc).__name__}: {exc}"
                    )
    expected = len(items) * len(args.tasks) * len(args.conditions)
    print(
        f"build preflight: {expected - len(build_errors)}/{expected} combinations healthy "
        f"for {identity.value}"
    )
    if build_errors:
        for error in build_errors[:20]:
            print(f"  ERROR {error}")
        raise SystemExit(2)

    if args.results is None:
        print("PRECHECK PASSED (no API calls made)")
        return

    try:
        rows = B.read_jsonl(args.results)
    except B.BenchmarkDataError as exc:
        raise SystemExit(f"SMOKE HEALTH FAILED: {exc}") from exc
    answers = {
        key: answer
        for key, answer in all_answers.items()
        if key[0] in {str(item["item_id"]) for item in items}
    }
    errors = check_result_rows(
        rows,
        items,
        answers,
        args.tasks,
        args.conditions,
        identity.value,
        {
            str(item["item_id"]): B.load_item_metadata(items_root, str(item["item_id"]))
            for item in items
        },
        require_clean_output=not args.allow_invalid_output,
    )
    if errors:
        print(f"SMOKE HEALTH FAILED: {len(errors)} issue(s)")
        for error in errors[:40]:
            print(f"  ERROR {error}")
        raise SystemExit(2)
    print(f"SMOKE HEALTH PASSED: {len(rows)}/{expected} rows")


if __name__ == "__main__":
    main()
