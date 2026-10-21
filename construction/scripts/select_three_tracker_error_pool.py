#!/usr/bin/env python3
"""Select a capped three-tracker error pool under two hardness orderings."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_tracker_error_dependence import (  # noqa: E402
    graph_arrays,
    labels_at_centroids,
    node_indices,
)


MODELS = ("trackastra", "hoct", "hungarian")
REGIONS = (
    "trackastra_only",
    "hoct_only",
    "hungarian_only",
    "trackastra_hoct_only",
    "trackastra_hungarian_only",
    "hoct_hungarian_only",
    "all_three",
)
SCENARIOS = ("hardness_only", "tracker_count_then_hardness")
KEY_FIELDS = ("dataset", "sequence", "frame_from", "label_from", "frame_to", "label_to")


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
    parser.add_argument("--per-dataset", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=analysis_root / "selected_top64_per_dataset")
    return parser.parse_args()


def read_membership(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "dataset": row["dataset"],
            "sequence": row["sequence"],
            "frame_from": int(row["frame_from"]),
            "label_from": int(row["label_from"]),
            "frame_to": int(row["frame_to"]),
            "label_to": int(row["label_to"]),
        }
        for model in MODELS:
            error_type = row.get(f"{model}_error_type", "")
            record[f"{model}_error_type"] = error_type
            record[f"{model}_error"] = bool(error_type)
        record["tracker_error_count"] = sum(record[f"{model}_error"] for model in MODELS)
        if record["tracker_error_count"] != int(row["tracker_error_count"]):
            raise ValueError(f"Tracker count mismatch: {row}")
        key = tuple(record[field] for field in KEY_FIELDS)
        if key in seen:
            raise ValueError(f"Duplicate physical error location: {key}")
        seen.add(key)
        output.append(record)
    return output


def graph_paths(root: Path) -> dict[tuple[str, str], Path]:
    paths = {
        (path.parent.name, path.stem): path
        for path in sorted(root.glob("group_*/*/*.geff"))
    }
    if not paths:
        raise FileNotFoundError(f"No candidate graphs under {root}")
    return paths


def graph_scores(graph_dir: Path, labels: np.ndarray) -> dict[tuple[int, int, int, int], float]:
    graph = graph_arrays(graph_dir)
    similarities = np.asarray(
        zarr.open_group(str(graph_dir), mode="r")["edges/props/similarity/values"][:],
        dtype=np.float64,
    ).reshape(-1)
    if len(similarities) != len(graph["edges"]):
        raise ValueError(f"Similarity/edge length mismatch: {graph_dir}")

    source_indices, target_indices = node_indices(graph["node_ids"], graph["edges"])
    source_times = graph["t"][source_indices]
    target_times = graph["t"][target_indices]
    source_labels = labels[source_indices]
    target_labels = labels[target_indices]
    direct = (
        (target_times == source_times + 1)
        & (source_labels > 0)
        & (target_labels > 0)
        & np.isfinite(similarities)
    )

    scores: dict[tuple[int, int, int, int], float] = {}
    indices = np.flatnonzero(direct)
    for index in indices:
        key = (
            int(source_times[index]),
            int(source_labels[index]),
            int(target_times[index]),
            int(target_labels[index]),
        )
        score = max(0.0, min(1.0, float(similarities[index])))
        scores[key] = max(scores.get(key, 0.0), score)
    return scores


def model_difficulty(score: float, is_gt: bool, error_type: str) -> float:
    if error_type == "EC":
        return max(score, 1.0 - score)
    return 1.0 - score if is_gt else score


def add_hardness(
    rows: list[dict[str, Any]],
    datasets_dir: Path,
    trackastra_root: Path,
    hoct_root: Path,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sequence"])].append(row)

    trackastra_paths = graph_paths(trackastra_root)
    hoct_paths = graph_paths(hoct_root)
    if set(grouped) != set(trackastra_paths) or set(grouped) != set(hoct_paths):
        raise ValueError("Membership sequences and paired candidate graphs do not match")

    absent = Counter()
    for index, sequence_key in enumerate(sorted(grouped), start=1):
        dataset, sequence = sequence_key
        print(f"[{index:02d}/{len(grouped):02d}] scoring {dataset}/{sequence}", flush=True)
        ta_graph = graph_arrays(trackastra_paths[sequence_key])
        labels = labels_at_centroids(
            ta_graph, datasets_dir / dataset / f"{sequence}_GT" / "TRA"
        )
        ta_scores = graph_scores(trackastra_paths[sequence_key], labels)
        ho_scores = graph_scores(hoct_paths[sequence_key], labels)

        for row in grouped[sequence_key]:
            key = (
                row["frame_from"],
                row["label_from"],
                row["frame_to"],
                row["label_to"],
            )
            error_types = {
                row[f"{model}_error_type"]
                for model in MODELS
                if row[f"{model}_error_type"]
            }
            positive = bool(error_types & {"EA", "EC"})
            negative = "ED" in error_types
            if positive == negative:
                raise ValueError(f"Cannot classify physical link as GT/non-GT: {sequence_key + key}")

            ta_present = key in ta_scores
            ho_present = key in ho_scores
            ta_score = ta_scores.get(key, 0.0)
            ho_score = ho_scores.get(key, 0.0)
            absent.update(
                {
                    "trackastra": int(not ta_present),
                    "hoct": int(not ho_present),
                }
            )
            ta_difficulty = model_difficulty(
                ta_score, positive, row["trackastra_error_type"]
            )
            ho_difficulty = model_difficulty(ho_score, positive, row["hoct_error_type"])
            row.update(
                is_ground_truth_link=positive,
                trackastra_candidate_present=ta_present,
                hoct_candidate_present=ho_present,
                in_common_decision_universe=(positive or (ta_present and ho_present)),
                trackastra_score=ta_score,
                hoct_score=ho_score,
                trackastra_difficulty=ta_difficulty,
                hoct_difficulty=ho_difficulty,
                hardness=(ta_difficulty + ho_difficulty) / 2.0,
            )
    return {"absent_candidate_counts": dict(absent)}


def physical_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def select(rows: list[dict[str, Any]], scenario: str, limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    selected: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        candidates = grouped[dataset]
        if scenario == "hardness_only":
            ordered = sorted(candidates, key=lambda row: (-row["hardness"], physical_sort_key(row)))
        elif scenario == "tracker_count_then_hardness":
            ordered = sorted(
                candidates,
                key=lambda row: (
                    -row["tracker_error_count"],
                    -row["hardness"],
                    physical_sort_key(row),
                ),
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        for rank, row in enumerate(ordered[:limit], start=1):
            output = dict(row)
            output.update(scenario=scenario, dataset_selection_rank=rank)
            selected.append(output)
    return selected


def region(row: dict[str, Any]) -> str:
    failed = tuple(model for model in MODELS if row[f"{model}_error"])
    mapping = {
        ("trackastra",): "trackastra_only",
        ("hoct",): "hoct_only",
        ("hungarian",): "hungarian_only",
        ("trackastra", "hoct"): "trackastra_hoct_only",
        ("trackastra", "hungarian"): "trackastra_hungarian_only",
        ("hoct", "hungarian"): "hoct_hungarian_only",
        MODELS: "all_three",
    }
    return mapping[failed]


def summarize_selection(
    rows: list[dict[str, Any]], selected: list[dict[str, Any]], scenario: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available_by_dataset = Counter(row["dataset"] for row in rows)
    selected_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_dataset[row["dataset"]].append(row)

    overall_regions = Counter(region(row) for row in selected)
    tracker_totals = Counter()
    for row in selected:
        for model in MODELS:
            tracker_totals[model] += int(row[f"{model}_error"])

    per_dataset: list[dict[str, Any]] = []
    for dataset in sorted(available_by_dataset):
        dataset_rows = selected_by_dataset[dataset]
        regions = Counter(region(row) for row in dataset_rows)
        record: dict[str, Any] = {
            "scenario": scenario,
            "dataset": dataset,
            "available_union_items": available_by_dataset[dataset],
            "selected_items": len(dataset_rows),
        }
        record.update({name: regions[name] for name in REGIONS})
        for model in MODELS:
            record[f"{model}_inclusive"] = sum(
                int(row[f"{model}_error"]) for row in dataset_rows
            )
        record.update(
            trackastra_hoct_inclusive=(
                regions["trackastra_hoct_only"] + regions["all_three"]
            ),
            trackastra_hungarian_inclusive=(
                regions["trackastra_hungarian_only"] + regions["all_three"]
            ),
            hoct_hungarian_inclusive=(
                regions["hoct_hungarian_only"] + regions["all_three"]
            ),
            all_three_inclusive=regions["all_three"],
        )
        per_dataset.append(record)

    summary = {
        "selected_items": len(selected),
        "datasets": len(selected_by_dataset),
        "venn_regions": {name: overall_regions[name] for name in REGIONS},
        "inclusive_tracker_counts": {model: tracker_totals[model] for model in MODELS},
        "inclusive_intersections": {
            "trackastra_hoct": sum(
                row["trackastra_error"] and row["hoct_error"] for row in selected
            ),
            "trackastra_hungarian": sum(
                row["trackastra_error"] and row["hungarian_error"] for row in selected
            ),
            "hoct_hungarian": sum(
                row["hoct_error"] and row["hungarian_error"] for row in selected
            ),
            "all_three": overall_regions["all_three"],
        },
        "tracker_error_count_distribution": dict(
            sorted(Counter(row["tracker_error_count"] for row in selected).items())
        ),
        "mean_hardness": float(np.mean([row["hardness"] for row in selected])),
    }
    return summary, per_dataset


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Three-Tracker Top-64 Error Selection",
        "",
        "Each dataset contributes at most 64 unique physical error links from the union of the",
        "Trackastra node-only, HOCT, and distance-Hungarian error sets. As in the three-tracker",
        "comparison, negative links are restricted to the shared learned-model candidate graph.",
        "",
        "Hardness is the mean Trackastra/HOCT discrepancy on the same physical link. GT links",
        "expect score 1 and non-GT links expect score 0; EC follows the existing extractor's",
        "semantic-error rule. Hungarian is excluded from this score because it has no calibrated",
        "edge confidence.",
        "",
        "| Venn region | Hardness only | Tracker count, then hardness |",
        "|---|---:|---:|",
    ]
    first = payload["scenarios"]["hardness_only"]["venn_regions"]
    second = payload["scenarios"]["tracker_count_then_hardness"]["venn_regions"]
    for name in REGIONS:
        lines.append(f"| {name.replace('_', ' ')} | {first[name]:,} | {second[name]:,} |")
    lines.extend(
        [
            "",
            "| Total | Hardness only | Tracker count, then hardness |",
            "|---|---:|---:|",
            f"| Selected union items | {payload['scenarios']['hardness_only']['selected_items']:,} | {payload['scenarios']['tracker_count_then_hardness']['selected_items']:,} |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be positive")
    rows = read_membership(args.membership)
    scoring = add_hardness(rows, args.datasets_dir, args.trackastra_graphs, args.hoct_graphs)
    membership_items = len(rows)
    rows = [row for row in rows if row["in_common_decision_universe"]]
    scoring["membership_items"] = membership_items
    scoring["excluded_negative_links_outside_common_candidates"] = membership_items - len(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_summaries: dict[str, Any] = {}
    all_per_dataset: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        selected = select(rows, scenario, args.per_dataset)
        summary, per_dataset = summarize_selection(rows, selected, scenario)
        scenario_summaries[scenario] = summary
        all_per_dataset.extend(per_dataset)
        write_csv(args.output_dir / f"{scenario}.csv", selected)

    payload = {
        "scope": "Union of physical link errors in the common three-tracker comparison universe, capped independently per dataset.",
        "per_dataset_limit": args.per_dataset,
        "hardness_definition": {
            "score": "Mean of Trackastra and HOCT physical-link discrepancy.",
            "ground_truth_link": "1 - candidate similarity; absent candidate similarity is 0.",
            "non_ground_truth_link": "candidate similarity; absent candidate similarity is 0.",
            "semantic_error_EC": "max(score, 1 - score), matching the existing error extractor.",
            "hungarian": "Not included because the baseline has no calibrated edge confidence.",
        },
        "tie_breaking": {
            "hardness_only": "Descending hardness, then stable physical key only.",
            "tracker_count_then_hardness": "Descending tracker error count, descending hardness, then stable physical key.",
        },
        "available_union_items": len(rows),
        "scoring_diagnostics": scoring,
        "scenarios": scenario_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_dataset.csv", all_per_dataset)
    write_report(args.output_dir / "REPORT.md", payload)
    print(json.dumps(payload["scenarios"], indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
