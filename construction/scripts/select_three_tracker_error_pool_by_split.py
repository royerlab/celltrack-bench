#!/usr/bin/env python3
"""Split sequences first, then select up to 64 errors per dataset and split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from select_three_tracker_error_pool import (  # noqa: E402
    MODELS,
    REGIONS,
    SCENARIOS,
    add_hardness,
    graph_paths,
    physical_sort_key,
    read_membership,
    region,
    write_csv,
)


SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    analysis_root = (
        REPO_ROOT
        / "evaluation"
        / "errors_gt_linking"
        / "trackastra_hoct_hungarian_analysis"
    )
    parser.add_argument("--membership", type=Path, default=analysis_root / "error_membership.csv.gz")
    parser.add_argument("--datasets-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--trackastra-graphs",
        type=Path,
        default=REPO_ROOT / "evaluation" / "trackastra_node_only_groupcv" / "full_graphs",
    )
    parser.add_argument(
        "--hoct-graphs",
        type=Path,
        default=REPO_ROOT / "evaluation" / "hoct_groupcv" / "full_graphs",
    )
    parser.add_argument("--per-dataset-per-split", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=analysis_root / "selected_top64_per_split_per_dataset",
    )
    return parser.parse_args()


def build_sequence_split(graph_root: Path) -> dict[str, dict[str, list[str]]]:
    sequences: dict[str, set[str]] = defaultdict(set)
    for dataset, sequence in graph_paths(graph_root):
        sequences[dataset].add(sequence)

    assignments: dict[str, dict[str, list[str]]] = {}
    for dataset, values in sorted(sequences.items()):
        ordered = sorted(values)
        if len(ordered) == 1:
            validation: list[str] = []
            test = ordered
        else:
            cut = len(ordered) // 2
            validation = ordered[:cut]
            test = ordered[cut:]
        assignments[dataset] = {"validation": validation, "test": test}
    return assignments


def split_lookup(assignments: dict[str, dict[str, list[str]]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for dataset, splits in assignments.items():
        for split, sequences in splits.items():
            for sequence in sequences:
                key = (dataset, sequence)
                if key in lookup:
                    raise ValueError(f"Sequence assigned more than once: {key}")
                lookup[key] = split
    return lookup


def ordered_candidates(rows: list[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    if scenario == "hardness_only":
        return sorted(rows, key=lambda row: (-row["hardness"], physical_sort_key(row)))
    if scenario == "tracker_count_then_hardness":
        return sorted(
            rows,
            key=lambda row: (
                -row["tracker_error_count"],
                -row["hardness"],
                physical_sort_key(row),
            ),
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def select(
    rows: list[dict[str, Any]],
    assignments: dict[str, dict[str, list[str]]],
    scenario: str,
    limit: int,
) -> list[dict[str, Any]]:
    lookup = split_lookup(assignments)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = lookup[(row["dataset"], row["sequence"])]
        grouped[(row["dataset"], split)].append(row)

    selected: list[dict[str, Any]] = []
    for dataset in sorted(assignments):
        for split in SPLITS:
            candidates = ordered_candidates(grouped.get((dataset, split), []), scenario)
            for rank, row in enumerate(candidates[:limit], start=1):
                output = dict(row)
                output.update(
                    scenario=scenario,
                    data_split=split,
                    split_selection_rank=rank,
                )
                selected.append(output)
    return selected


def count_record(rows: list[dict[str, Any]]) -> dict[str, int]:
    regions = Counter(region(row) for row in rows)
    record = {
        "selected_items": len(rows),
        **{name: regions[name] for name in REGIONS},
    }
    for model in MODELS:
        record[f"{model}_inclusive"] = sum(int(row[f"{model}_error"]) for row in rows)
    record.update(
        trackastra_hoct_inclusive=regions["trackastra_hoct_only"] + regions["all_three"],
        trackastra_hungarian_inclusive=(
            regions["trackastra_hungarian_only"] + regions["all_three"]
        ),
        hoct_hungarian_inclusive=regions["hoct_hungarian_only"] + regions["all_three"],
        all_three_inclusive=regions["all_three"],
    )
    return record


def summarize(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    assignments: dict[str, dict[str, list[str]]],
    scenario: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    chosen: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    lookup = split_lookup(assignments)
    for row in rows:
        available[(row["dataset"], lookup[(row["dataset"], row["sequence"])])].append(row)
    for row in selected:
        chosen[(row["dataset"], row["data_split"])].append(row)

    per_dataset: list[dict[str, Any]] = []
    for dataset in sorted(assignments):
        for split in SPLITS:
            key = (dataset, split)
            record: dict[str, Any] = {
                "scenario": scenario,
                "data_split": split,
                "dataset": dataset,
                "sequences": ";".join(assignments[dataset][split]),
                "available_union_items": len(available.get(key, [])),
                **count_record(chosen.get(key, [])),
            }
            per_dataset.append(record)

    split_summary = {
        split: count_record([row for row in selected if row["data_split"] == split])
        for split in SPLITS
    }
    split_summary["combined"] = count_record(selected)
    return split_summary, per_dataset


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Sequence-Disjoint Top-64 Selection",
        "",
        "Sequences are partitioned before ranking. Each dataset contributes up to 64 items",
        "independently to validation and test under each ordering.",
        "",
        "For datasets with two sequences, `01` is validation and `02` is test. For bacterial",
        "datasets with more sequences, the sorted lower half is validation and the upper half",
        "is test. Bacteria-pheA has only sequence `05`, so it is test-only.",
        "",
        "| Ordering | Split | Selected | Trackastra | HOCT | Hungarian | All three |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for split in SPLITS:
            values = payload["scenarios"][scenario][split]
            lines.append(
                f"| {scenario.replace('_', ' ')} | {split} | {values['selected_items']:,} "
                f"| {values['trackastra_inclusive']:,} | {values['hoct_inclusive']:,} "
                f"| {values['hungarian_inclusive']:,} | {values['all_three_inclusive']:,} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.per_dataset_per_split <= 0:
        raise ValueError("--per-dataset-per-split must be positive")

    rows = read_membership(args.membership)
    scoring = add_hardness(rows, args.datasets_dir, args.trackastra_graphs, args.hoct_graphs)
    membership_items = len(rows)
    rows = [row for row in rows if row["in_common_decision_universe"]]
    scoring.update(
        membership_items=membership_items,
        excluded_negative_links_outside_common_candidates=membership_items - len(rows),
    )
    assignments = build_sequence_split(args.trackastra_graphs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_summaries: dict[str, Any] = {}
    all_per_dataset: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        selected = select(
            rows,
            assignments,
            scenario,
            args.per_dataset_per_split,
        )
        summary, per_dataset = summarize(rows, selected, assignments, scenario)
        scenario_summaries[scenario] = summary
        all_per_dataset.extend(per_dataset)
        write_csv(args.output_dir / f"{scenario}.csv", selected)

    payload = {
        "scope": "Sequence-disjoint selection from the common three-tracker error universe.",
        "selection_order": "Split sequences first, then rank independently within dataset and split.",
        "per_dataset_per_split_limit": args.per_dataset_per_split,
        "available_union_items": len(rows),
        "sequence_assignments": assignments,
        "singleton_policy": "A dataset with one represented sequence is assigned to test only.",
        "hardness_definition": "Mean Trackastra/HOCT physical-link discrepancy.",
        "scoring_diagnostics": scoring,
        "scenarios": scenario_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_dataset.csv", all_per_dataset)
    write_report(args.output_dir / "REPORT.md", payload)
    print(json.dumps(scenario_summaries, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
