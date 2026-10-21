#!/usr/bin/env python3
"""Validate split isolation and graph labels for consensus-selected HOCT tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PARTS = ("cell_linking", "track_repair", "track_inspection")
TRACKERS = ("trackastra", "hoct", "hungarian")
ERROR_TO_OPERATION = {"EA": "add", "ED": "delete", "EC": "change_semantic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-selection-rank", type=int, default=64,
                        help="Upper bound for split_selection_rank; 64 for the classic build, 256 for -xl.")
    parser.add_argument("--data-split", choices=("validation", "test"), required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def edge_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source"]), str(row["target"])


def expected_operations(sample: dict[str, Any]) -> dict[tuple[str, str], str]:
    gt = {edge_key(edge): edge for edge in sample["gt_edges"]}
    solution = {edge_key(edge): edge for edge in sample["hoct_edges"]}
    operations: dict[tuple[str, str], str] = {}
    for pair in set(gt) - set(solution):
        operations[pair] = "add"
    for pair in set(solution) - set(gt):
        operations[pair] = "delete"
    for pair in set(gt) & set(solution):
        if gt[pair]["semantic"] != solution[pair]["semantic"]:
            operations[pair] = "change_semantic"
    return operations


def validate_sample(sample: dict[str, Any], data_split: str, max_selection_rank: int = 64) -> list[str]:
    item_id = sample.get("item_id", "<missing-item-id>")
    failures: list[str] = []
    selection = sample.get("selection", {})
    if sample.get("data_split") != data_split or selection.get("data_split") != data_split:
        failures.append(f"{item_id}: wrong or missing data_split")
    trackers = selection.get("trackers", {})
    if set(trackers) != set(TRACKERS):
        failures.append(f"{item_id}: incomplete tracker provenance")
        return failures
    observed_error_count = sum(bool(trackers[tracker].get("error")) for tracker in TRACKERS)
    if observed_error_count != selection.get("tracker_error_count"):
        failures.append(f"{item_id}: tracker_error_count mismatch")
    if not 1 <= int(selection.get("selection_rank", 0)) <= max_selection_rank:
        failures.append(f"{item_id}: invalid selection rank")

    candidates = {edge_key(edge) for edge in sample.get("candidate_edges", [])}
    gt = {edge_key(edge) for edge in sample.get("gt_edges", [])}
    if not gt.issubset(candidates):
        failures.append(f"{item_id}: ground-truth edge outside candidate graph")
    expected = expected_operations(sample)
    actual = {edge_key(row): row["op"] for row in sample.get("operations", [])}
    if not actual:
        failures.append(f"{item_id}: empty HOCT repair")
    if actual != expected:
        failures.append(f"{item_id}: repair operations do not equal the HOCT-vs-GT diff")
    if sample.get("local_error_count") != len(actual):
        failures.append(f"{item_id}: local_error_count mismatch")

    source_display = {int(label): display for label, display in sample["source_display"].items()}
    target_display = {int(label): display for label, display in sample["target_display"].items()}
    proxy = sample["proxy"]
    anchor_pair = (
        source_display[int(proxy["source_label"])],
        target_display[int(proxy["target_label"])],
    )
    hoct = trackers["hoct"]
    if hoct["error"]:
        expected_op = ERROR_TO_OPERATION.get(hoct.get("error_type"))
        if expected_op is None or actual.get(anchor_pair) != expected_op:
            failures.append(f"{item_id}: declared HOCT anchor error is not reproduced")
    elif anchor_pair in actual:
        failures.append(f"{item_id}: anchor is a HOCT error despite a false provenance flag")
    return failures


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "shared_items.json")
    summary = load_json(root / "build_summary.json")
    skipped = load_json(root / "skipped_anchors.json")
    failures: list[str] = []

    item_ids = [row["item_id"] for row in manifest]
    if len(item_ids) != len(set(item_ids)):
        failures.append("shared_items.json contains duplicate item IDs")
    if any(row.get("data_split") != args.data_split for row in manifest):
        failures.append("shared_items.json mixes data splits")
    if any(row.get("data_split") != args.data_split for row in skipped):
        failures.append("skipped_anchors.json mixes data splits")

    metadata_paths = sorted((root / "items").glob("*/metadata_private.json"))
    samples = [load_json(path) for path in metadata_paths]
    if {sample["item_id"] for sample in samples} != set(item_ids):
        failures.append("private metadata IDs differ from shared manifest IDs")
    for sample in samples:
        failures.extend(validate_sample(sample, args.data_split, max_selection_rank=args.max_selection_rank))
        prompt_dir = root / "items" / sample["item_id"] / "prompts"
        for part in PARTS:
            prompt = (prompt_dir / f"{part}.md").read_text(encoding="utf-8")
            if "one or more visible" not in prompt:
                failures.append(f"{sample['item_id']}/{part}: lineage-cardinality contract missing")

    task_counts: dict[str, int] = {}
    for part in PARTS:
        tasks = load_jsonl(root / part / "tasks.jsonl")
        answers = load_jsonl(root / part / "answer_key.jsonl")
        task_counts[part] = len(tasks)
        if {row["item_id"] for row in tasks} != set(item_ids):
            failures.append(f"{part}: task item IDs differ from shared manifest")
        if {row["task_id"] for row in tasks} != {row["task_id"] for row in answers}:
            failures.append(f"{part}: task and answer IDs differ")
        if any(row.get("data_split") != args.data_split for row in tasks):
            failures.append(f"{part}: tasks mix data splits")

    if summary.get("anchor_mode") != "selected_consensus_first":
        failures.append("build summary does not identify selected-anchor mode")
    if summary.get("data_split") != args.data_split:
        failures.append("build summary has the wrong data split")
    if summary.get("samples") != len(samples):
        failures.append("build summary sample count mismatch")

    division_cardinalities: Counter[int] = Counter()
    for sample in samples:
        daughters: dict[str, set[str]] = {}
        for edge in sample["gt_edges"]:
            if edge["semantic"] == "division":
                daughters.setdefault(edge["source"], set()).add(edge["target"])
        division_cardinalities.update(len(targets) for targets in daughters.values())

    report = {
        "valid": not failures,
        "data_split": args.data_split,
        "samples": len(samples),
        "task_counts": task_counts,
        "hoct_anchor_errors": sum(sample["proxy"]["hoct_error"] for sample in samples),
        "non_hoct_anchors": sum(not sample["proxy"]["hoct_error"] for sample in samples),
        "operation_counts": dict(
            Counter(operation["op"] for sample in samples for operation in sample["operations"])
        ),
        "division_event_cardinality_counts": {
            str(cardinality): count
            for cardinality, count in sorted(division_cardinalities.items())
        },
        "failures": failures,
    }
    report_path = args.report or (root / "selected_validation_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
