#!/usr/bin/env python3
"""Analyze Trackastra, HOCT, and distance-Hungarian link errors together."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import special, stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_tracker_error_dependence import (
    edge_encoder,
    graph_arrays,
    ground_truth_edges,
    labels_at_centroids,
    node_indices,
    parse_track_file,
    physical_key,
    read_csv,
    verify_common_nodes,
)


ERROR_ROOT = REPO_ROOT / "evaluation" / "errors_gt_linking"
MODELS = ("trackastra", "hoct", "hungarian")
PAIRS = (("trackastra", "hoct"), ("trackastra", "hungarian"), ("hoct", "hungarian"))
UNIVERSES = (
    "ground_truth_links",
    "ground_truth_continuation_links",
    "ground_truth_division_links",
    "negative_candidate_links",
    "combined",
)
REGIONS = (
    "none",
    "trackastra_only",
    "hoct_only",
    "hungarian_only",
    "trackastra_hoct_only",
    "trackastra_hungarian_only",
    "hoct_hungarian_only",
    "all_three",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--trackastra-errors",
        type=Path,
        default=ERROR_ROOT / "trackastra_node_only_groupcv" / "all_groups_link_errors.csv",
    )
    parser.add_argument(
        "--hoct-errors",
        type=Path,
        default=ERROR_ROOT / "hoct_groupcv" / "all_groups_link_errors.csv",
    )
    parser.add_argument(
        "--hungarian-errors",
        type=Path,
        default=ERROR_ROOT / "hungarian_distance" / "all_groups_link_errors.csv",
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ERROR_ROOT / "trackastra_hoct_hungarian_analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def group_rows(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sequence"])].append(row)
    return dict(grouped)


def error_maps(rows: list[dict[str, str]]) -> tuple[dict[tuple[int, int, int, int], str], dict[tuple[int, int, int, int], str]]:
    positive: dict[tuple[int, int, int, int], str] = {}
    negative: dict[tuple[int, int, int, int], str] = {}
    for row in rows:
        key = physical_key(row)
        destination = positive if row["type"] in {"EA", "EC"} else negative
        if key in destination:
            raise ValueError(f"Duplicate physical error: {key}")
        destination[key] = row["type"]
    return positive, negative


def direct_candidate_codes(
    graph: dict[str, np.ndarray],
    labels: np.ndarray,
    encode_arrays,
) -> np.ndarray:
    source_indices, target_indices = node_indices(graph["node_ids"], graph["edges"])
    source_times = graph["t"][source_indices]
    target_times = graph["t"][target_indices]
    direct = (
        (target_times == source_times + 1)
        & (labels[source_indices] > 0)
        & (labels[target_indices] > 0)
    )
    return np.unique(
        encode_arrays(
            source_times[direct], labels[source_indices[direct]], labels[target_indices[direct]]
        )
    )


def restrict_keys(
    keys: set[tuple[int, int, int, int]],
    candidates: np.ndarray,
    encode_key,
) -> tuple[set[tuple[int, int, int, int]], int]:
    ordered = sorted(keys)
    codes = np.fromiter((encode_key(key) for key in ordered), dtype=np.uint64, count=len(ordered))
    keep = np.isin(codes, candidates)
    return {key for key, selected in zip(ordered, keep, strict=True) if selected}, int(
        np.count_nonzero(~keep)
    )


def pair_table(
    universe_size: int,
    first: set[tuple[int, int, int, int]],
    second: set[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    both = len(first & second)
    table = (both, len(first) - both, len(second) - both, universe_size - len(first | second))
    if table[3] < 0:
        raise ValueError("Pairwise error union exceeds its decision universe")
    return table


def three_regions(
    universe_size: int,
    errors: dict[str, set[tuple[int, int, int, int]]],
) -> dict[str, int]:
    ta = errors["trackastra"]
    ho = errors["hoct"]
    hu = errors["hungarian"]
    all_three = ta & ho & hu
    regions = {
        "trackastra_only": len(ta - ho - hu),
        "hoct_only": len(ho - ta - hu),
        "hungarian_only": len(hu - ta - ho),
        "trackastra_hoct_only": len((ta & ho) - hu),
        "trackastra_hungarian_only": len((ta & hu) - ho),
        "hoct_hungarian_only": len((ho & hu) - ta),
        "all_three": len(all_three),
    }
    regions["none"] = universe_size - sum(regions.values())
    if regions["none"] < 0:
        raise ValueError("Three-way error union exceeds its decision universe")
    return {region: regions[region] for region in REGIONS}


def add_pair_tables(tables: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    values = list(tables)
    return tuple(sum(table[index] for table in values) for index in range(4))


def add_regions(regions: Iterable[dict[str, int]]) -> dict[str, int]:
    values = list(regions)
    return {region: sum(value[region] for value in values) for region in REGIONS}


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def pair_metrics(table: tuple[int, int, int, int], first: str, second: str) -> dict[str, Any]:
    a, b, c, d = table
    total = sum(table)
    first_errors = a + b
    second_errors = a + c
    second_given_first = safe_ratio(a, first_errors)
    second_given_first_correct = safe_ratio(c, c + d)
    first_given_second = safe_ratio(a, second_errors)
    first_given_second_correct = safe_ratio(b, b + d)
    expected = safe_ratio(first_errors * second_errors, total)
    fisher = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "both_error": a,
        f"{first}_only_error": b,
        f"{second}_only_error": c,
        "neither_error": d,
        "decisions": total,
        f"{first}_errors": first_errors,
        f"{second}_errors": second_errors,
        f"{second}_error_rate_given_{first}_error": second_given_first,
        f"{second}_error_rate_given_{first}_correct": second_given_first_correct,
        f"{second}_conditional_risk_ratio": (
            safe_ratio(second_given_first, second_given_first_correct)
            if second_given_first is not None and second_given_first_correct is not None
            else None
        ),
        f"{first}_error_rate_given_{second}_error": first_given_second,
        f"{first}_error_rate_given_{second}_correct": first_given_second_correct,
        f"{first}_conditional_risk_ratio": (
            safe_ratio(first_given_second, first_given_second_correct)
            if first_given_second is not None and first_given_second_correct is not None
            else None
        ),
        "odds_ratio": safe_ratio(a * d, b * c),
        "jaccard": safe_ratio(a, a + b + c),
        "expected_overlap_under_independence": expected,
        "overlap_enrichment": safe_ratio(a, expected) if expected is not None else None,
        "fisher_odds_ratio": float(fisher.statistic),
        "fisher_p_one_sided": float(fisher.pvalue),
    }


def stratified_pair_test(
    records: list[dict[str, Any]], pair: tuple[str, str], universe: str
) -> dict[str, Any]:
    observed = expected = variance = mh_numerator = mh_denominator = 0.0
    valid = 0
    for record in records:
        a, b, c, d = record["pair_tables"][pair][universe]
        total = a + b + c + d
        if total <= 1:
            continue
        first_errors = a + b
        second_errors = a + c
        expected_i = first_errors * second_errors / total
        variance_i = (
            first_errors
            * second_errors
            * (total - first_errors)
            * (total - second_errors)
            / (total * total * (total - 1))
        )
        observed += a
        expected += expected_i
        variance += variance_i
        mh_numerator += a * d / total
        mh_denominator += b * c / total
        valid += 1
    z_score = safe_ratio(observed - expected, math.sqrt(variance))
    log_p = special.log_ndtr(-z_score) if z_score is not None else float("nan")
    return {
        "strata": valid,
        "observed_overlap": observed,
        "expected_overlap_conditional_on_sequence_margins": expected,
        "overlap_enrichment": safe_ratio(observed, expected),
        "z_score": z_score,
        "one_sided_p": float(math.exp(log_p)) if log_p > math.log(np.finfo(float).tiny) else 0.0,
        "one_sided_log10_p": float(log_p / math.log(10.0)),
        "mantel_haenszel_common_odds_ratio": safe_ratio(mh_numerator, mh_denominator),
    }


def dataset_pair_direction(
    records: list[dict[str, Any]], pair: tuple[str, str], universe: str
) -> dict[str, Any]:
    effects = []
    for record in records:
        effect = pair_metrics(record["pair_tables"][pair][universe], *pair)["overlap_enrichment"]
        if effect is not None and math.isfinite(effect) and effect != 1.0:
            effects.append(effect)
    above = sum(effect > 1.0 for effect in effects)
    test = (
        stats.binomtest(above, len(effects), 0.5, alternative="greater")
        if effects
        else None
    )
    return {
        "datasets_with_defined_non_tied_effect": len(effects),
        "datasets_with_enrichment_above_one": above,
        "one_sided_sign_test_p": float(test.pvalue) if test is not None else None,
    }


def dataset_pair_bootstrap(
    records: list[dict[str, Any]],
    pair: tuple[str, str],
    universe: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    cells = np.asarray([record["pair_tables"][pair][universe] for record in records], dtype=np.int64)
    estimates = np.empty(replicates, dtype=np.float64)
    count = len(cells)
    for start in range(0, replicates, 2_000):
        stop = min(replicates, start + 2_000)
        sampled = cells[rng.integers(0, count, size=(stop - start, count))].sum(axis=1)
        a, b, c, d = sampled.T.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimates[start:stop] = (a / (a + b)) / (c / (c + d))
    finite = estimates[np.isfinite(estimates)]
    interval = (
        [float(value) for value in np.quantile(finite, [0.025, 0.975])]
        if len(finite)
        else [None, None]
    )
    return {
        "replicates": replicates,
        "finite_replicates": len(finite),
        "positive_infinity_replicates": int(np.count_nonzero(np.isposinf(estimates))),
        "risk_ratio_median": float(np.median(finite)) if len(finite) else None,
        "risk_ratio_ci95": interval,
        "fraction_at_or_below_one": float(np.mean(finite <= 1.0)) if len(finite) else 0.0,
    }


def three_metrics(regions: dict[str, int]) -> dict[str, Any]:
    total = sum(regions.values())
    ta = (
        regions["trackastra_only"]
        + regions["trackastra_hoct_only"]
        + regions["trackastra_hungarian_only"]
        + regions["all_three"]
    )
    ho = (
        regions["hoct_only"]
        + regions["trackastra_hoct_only"]
        + regions["hoct_hungarian_only"]
        + regions["all_three"]
    )
    hu = (
        regions["hungarian_only"]
        + regions["trackastra_hungarian_only"]
        + regions["hoct_hungarian_only"]
        + regions["all_three"]
    )
    any_error = total - regions["none"]
    at_least_two = (
        regions["trackastra_hoct_only"]
        + regions["trackastra_hungarian_only"]
        + regions["hoct_hungarian_only"]
        + regions["all_three"]
    )
    expected = safe_ratio(ta * ho * hu, total * total)
    return {
        **regions,
        "decisions": total,
        "trackastra_errors": ta,
        "hoct_errors": ho,
        "hungarian_errors": hu,
        "any_tracker_error": any_error,
        "at_least_two_trackers_error": at_least_two,
        "all_three_error": regions["all_three"],
        "all_three_jaccard": safe_ratio(regions["all_three"], any_error),
        "expected_all_three_under_independence": expected,
        "all_three_overlap_enrichment": (
            safe_ratio(regions["all_three"], expected) if expected is not None else None
        ),
    }


def stratified_three_test(records: list[dict[str, Any]], universe: str) -> dict[str, Any]:
    observed = expected = variance = 0.0
    valid = 0
    for record in records:
        metrics = three_metrics(record["three_regions"][universe])
        total = metrics["decisions"]
        if total <= 1:
            continue
        counts = [metrics[f"{model}_errors"] for model in MODELS]
        p_one = math.prod(count / total for count in counts)
        p_two = math.prod(
            count * (count - 1) / (total * (total - 1)) for count in counts
        )
        observed += metrics["all_three_error"]
        expected += total * p_one
        variance += total * p_one * (1 - p_one) + total * (total - 1) * (p_two - p_one**2)
        valid += 1
    z_score = safe_ratio(observed - expected, math.sqrt(max(variance, 0.0)))
    log_p = special.log_ndtr(-z_score) if z_score is not None else float("nan")
    return {
        "strata": valid,
        "observed_all_three_overlap": observed,
        "expected_all_three_conditional_on_sequence_margins": expected,
        "all_three_overlap_enrichment": safe_ratio(observed, expected),
        "z_score": z_score,
        "one_sided_p": float(math.exp(log_p)) if log_p > math.log(np.finfo(float).tiny) else 0.0,
        "one_sided_log10_p": float(log_p / math.log(10.0)),
    }


def dataset_three_direction(records: list[dict[str, Any]], universe: str) -> dict[str, Any]:
    effects = []
    for record in records:
        effect = three_metrics(record["three_regions"][universe])["all_three_overlap_enrichment"]
        if effect is not None and math.isfinite(effect) and effect != 1.0:
            effects.append(effect)
    above = sum(effect > 1.0 for effect in effects)
    test = (
        stats.binomtest(above, len(effects), 0.5, alternative="greater")
        if effects
        else None
    )
    return {
        "datasets_with_defined_non_tied_effect": len(effects),
        "datasets_with_enrichment_above_one": above,
        "one_sided_sign_test_p": float(test.pvalue) if test is not None else None,
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record["metadata"])
    for pair in PAIRS:
        pair_name = "_vs_".join(pair)
        for universe in UNIVERSES:
            metrics = pair_metrics(record["pair_tables"][pair][universe], *pair)
            for field, value in metrics.items():
                row[f"{pair_name}_{universe}_{field}"] = value
    for universe in UNIVERSES:
        for field, value in three_metrics(record["three_regions"][universe]).items():
            row[f"three_way_{universe}_{field}"] = value
    return row


def aggregate_records(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["metadata"][field])].append(record)
    output = []
    for value, items in sorted(grouped.items()):
        metadata: dict[str, Any] = {field: value, "sequences": len(items)}
        sum_fields = (
            "ground_truth_links",
            "ground_truth_continuation_links",
            "ground_truth_division_links",
            "trackastra_candidates_t_plus_1",
            "hoct_candidates_t_plus_1",
            "common_candidates_t_plus_1",
            *(f"{model}_{error_type}" for model in MODELS for error_type in ("EA", "ED", "EC")),
        )
        for sum_field in sum_fields:
            metadata[sum_field] = sum(int(item["metadata"][sum_field]) for item in items)
        for model in MODELS:
            metadata[f"{model}_errors"] = sum(
                metadata[f"{model}_{error_type}"] for error_type in ("EA", "ED", "EC")
            )
        output.append(
            {
                "metadata": metadata,
                "pair_tables": {
                    pair: {
                        universe: add_pair_tables(
                            item["pair_tables"][pair][universe] for item in items
                        )
                        for universe in UNIVERSES
                    }
                    for pair in PAIRS
                },
                "three_regions": {
                    universe: add_regions(item["three_regions"][universe] for item in items)
                    for universe in UNIVERSES
                },
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_membership(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "sequence",
        "frame_from",
        "label_from",
        "frame_to",
        "label_to",
        "trackastra_error_type",
        "hoct_error_type",
        "hungarian_error_type",
        "tracker_error_count",
    )
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows = {
        "trackastra": read_csv(args.trackastra_errors),
        "hoct": read_csv(args.hoct_errors),
        "hungarian": read_csv(args.hungarian_errors),
    }
    grouped = {model: group_rows(rows) for model, rows in all_rows.items()}
    graph_dirs = sorted(args.trackastra_graphs.glob("group_*/*/*.geff"))
    records: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    pair_categories: Counter[tuple[str, str, str, str]] = Counter()
    triple_categories: Counter[tuple[str, str, str]] = Counter()

    for index, ta_graph_dir in enumerate(graph_dirs, start=1):
        relative = ta_graph_dir.relative_to(args.trackastra_graphs)
        ho_graph_dir = args.hoct_graphs / relative
        dataset = ta_graph_dir.parent.name
        sequence = ta_graph_dir.stem
        key = (dataset, sequence)
        print(f"[{index:02d}/{len(graph_dirs):02d}] {dataset}/{sequence}", flush=True)
        ta_graph = graph_arrays(ta_graph_dir)
        ho_graph = graph_arrays(ho_graph_dir)
        verify_common_nodes(ta_graph, ho_graph, f"{dataset}/{sequence}")
        gt_dir = args.datasets_dir / dataset / f"{sequence}_GT" / "TRA"
        gt_edge_map = ground_truth_edges(parse_track_file(gt_dir / "man_track.txt"))
        gt_edges = set(gt_edge_map)
        gt_continuations = {
            edge for edge, semantic in gt_edge_map.items() if semantic == "continuation"
        }
        gt_divisions = {edge for edge, semantic in gt_edge_map.items() if semantic == "division"}
        labels = labels_at_centroids(ta_graph, gt_dir)

        ta_source, ta_target = node_indices(ta_graph["node_ids"], ta_graph["edges"])
        ho_source, ho_target = node_indices(ho_graph["node_ids"], ho_graph["edges"])
        all_times = np.concatenate((ta_graph["t"][ta_source], ho_graph["t"][ho_source]))
        all_sources = np.concatenate((labels[ta_source], labels[ho_source]))
        all_targets = np.concatenate((labels[ta_target], labels[ho_target]))
        encode_arrays, encode_key = edge_encoder(all_times, all_sources, all_targets, gt_edges)
        candidates = {
            "trackastra": direct_candidate_codes(ta_graph, labels, encode_arrays),
            "hoct": direct_candidate_codes(ho_graph, labels, encode_arrays),
        }
        pair_candidates = {
            ("trackastra", "hoct"): np.intersect1d(
                candidates["trackastra"], candidates["hoct"], assume_unique=True
            ),
            ("trackastra", "hungarian"): candidates["trackastra"],
            ("hoct", "hungarian"): candidates["hoct"],
        }
        common_candidates = pair_candidates[("trackastra", "hoct")]
        gt_codes = np.fromiter(
            (encode_key(edge) for edge in gt_edges), dtype=np.uint64, count=len(gt_edges)
        )

        positive_maps: dict[str, dict[tuple[int, int, int, int], str]] = {}
        negative_maps: dict[str, dict[tuple[int, int, int, int], str]] = {}
        for model in MODELS:
            positive_maps[model], negative_maps[model] = error_maps(grouped[model].get(key, []))
            if not set(positive_maps[model]) <= gt_edges:
                raise ValueError(f"{model} positive error outside GT: {dataset}/{sequence}")
        for model in ("trackastra", "hoct"):
            own_codes = np.fromiter(
                (encode_key(edge) for edge in negative_maps[model]),
                dtype=np.uint64,
                count=len(negative_maps[model]),
            )
            if len(own_codes) and not np.all(np.isin(own_codes, candidates[model])):
                raise ValueError(f"{model} ED outside own candidates: {dataset}/{sequence}")

        pair_tables: dict[tuple[str, str], dict[str, tuple[int, int, int, int]]] = {}
        for pair in PAIRS:
            candidate_codes = pair_candidates[pair]
            gt_in_candidates = int(np.count_nonzero(np.isin(gt_codes, candidate_codes)))
            negative_size = len(candidate_codes) - gt_in_candidates
            restricted = {}
            for model in pair:
                restricted[model], _excluded = restrict_keys(
                    set(negative_maps[model]), candidate_codes, encode_key
                )
            positive_table = pair_table(
                len(gt_edges), set(positive_maps[pair[0]]), set(positive_maps[pair[1]])
            )
            continuation_table = pair_table(
                len(gt_continuations),
                set(positive_maps[pair[0]]) & gt_continuations,
                set(positive_maps[pair[1]]) & gt_continuations,
            )
            division_table = pair_table(
                len(gt_divisions),
                set(positive_maps[pair[0]]) & gt_divisions,
                set(positive_maps[pair[1]]) & gt_divisions,
            )
            negative_table = pair_table(
                negative_size, restricted[pair[0]], restricted[pair[1]]
            )
            pair_tables[pair] = {
                "ground_truth_links": positive_table,
                "ground_truth_continuation_links": continuation_table,
                "ground_truth_division_links": division_table,
                "negative_candidate_links": negative_table,
                "combined": add_pair_tables((positive_table, negative_table)),
            }

        common_gt_count = int(np.count_nonzero(np.isin(gt_codes, common_candidates)))
        common_negative_size = len(common_candidates) - common_gt_count
        three_positive = {model: set(positive_maps[model]) for model in MODELS}
        three_negative = {
            model: restrict_keys(set(negative_maps[model]), common_candidates, encode_key)[0]
            for model in MODELS
        }
        three_regions_by_universe = {
            "ground_truth_links": three_regions(len(gt_edges), three_positive),
            "ground_truth_continuation_links": three_regions(
                len(gt_continuations),
                {model: errors & gt_continuations for model, errors in three_positive.items()},
            ),
            "ground_truth_division_links": three_regions(
                len(gt_divisions),
                {model: errors & gt_divisions for model, errors in three_positive.items()},
            ),
            "negative_candidate_links": three_regions(common_negative_size, three_negative),
        }
        three_regions_by_universe["combined"] = add_regions(
            (
                three_regions_by_universe["ground_truth_links"],
                three_regions_by_universe["negative_candidate_links"],
            )
        )

        all_error_maps = {
            model: {**positive_maps[model], **negative_maps[model]} for model in MODELS
        }
        union_keys = set().union(*(set(mapping) for mapping in all_error_maps.values()))
        for edge in sorted(union_keys):
            types = {model: all_error_maps[model].get(edge, "") for model in MODELS}
            error_count = sum(bool(value) for value in types.values())
            memberships.append(
                {
                    "dataset": dataset,
                    "sequence": sequence,
                    "frame_from": edge[0],
                    "label_from": edge[1],
                    "frame_to": edge[2],
                    "label_to": edge[3],
                    "trackastra_error_type": types["trackastra"],
                    "hoct_error_type": types["hoct"],
                    "hungarian_error_type": types["hungarian"],
                    "tracker_error_count": error_count,
                }
            )
            for first, second in PAIRS:
                if types[first] and types[second]:
                    pair_categories[(first, second, types[first], types[second])] += 1
            if error_count == 3:
                triple_categories[
                    (types["trackastra"], types["hoct"], types["hungarian"])
                ] += 1

        records.append(
            {
                "metadata": {
                    "dataset": dataset,
                    "sequence": sequence,
                    "ground_truth_links": len(gt_edges),
                    "ground_truth_continuation_links": len(gt_continuations),
                    "ground_truth_division_links": len(gt_divisions),
                    "trackastra_candidates_t_plus_1": len(candidates["trackastra"]),
                    "hoct_candidates_t_plus_1": len(candidates["hoct"]),
                    "common_candidates_t_plus_1": len(common_candidates),
                    **{
                        f"{model}_{error_type}": sum(
                            row["type"] == error_type for row in grouped[model].get(key, [])
                        )
                        for model in MODELS
                        for error_type in ("EA", "ED", "EC")
                    },
                },
                "pair_tables": pair_tables,
                "three_regions": three_regions_by_universe,
            }
        )

    expected_keys = {(record["metadata"]["dataset"], record["metadata"]["sequence"]) for record in records}
    for model in MODELS:
        if set(grouped[model]) - expected_keys:
            raise ValueError(f"{model} error CSV includes sequences outside the graph catalog")

    dataset_records = aggregate_records(records, "dataset")
    overall_record = {
        "metadata": {"sequences": len(records), "datasets": len(dataset_records)},
        "pair_tables": {
            pair: {
                universe: add_pair_tables(record["pair_tables"][pair][universe] for record in records)
                for universe in UNIVERSES
            }
            for pair in PAIRS
        },
        "three_regions": {
            universe: add_regions(record["three_regions"][universe] for record in records)
            for universe in UNIVERSES
        },
    }
    rng = np.random.default_rng(args.seed)
    pairwise_summary: dict[str, Any] = {}
    for pair in PAIRS:
        pair_name = "_vs_".join(pair)
        pairwise_summary[pair_name] = {}
        for universe in UNIVERSES:
            pairwise_summary[pair_name][universe] = {
                "paired_table": pair_metrics(overall_record["pair_tables"][pair][universe], *pair),
                "sequence_stratified_test": stratified_pair_test(records, pair, universe),
                "dataset_direction_test": dataset_pair_direction(dataset_records, pair, universe),
                "dataset_cluster_bootstrap": dataset_pair_bootstrap(
                    dataset_records, pair, universe, args.bootstrap_replicates, rng
                ),
            }
    three_way_summary = {
        universe: {
            "regions": three_metrics(overall_record["three_regions"][universe]),
            "sequence_stratified_test": stratified_three_test(records, universe),
            "dataset_direction_test": dataset_three_direction(dataset_records, universe),
        }
        for universe in UNIVERSES
    }
    pair_category_summary = {}
    for first, second in PAIRS:
        counts = {
            f"{first_type}_to_{second_type}": pair_categories.get(
                (first, second, first_type, second_type), 0
            )
            for first_type in ("EA", "ED", "EC")
            for second_type in ("EA", "ED", "EC")
        }
        pair_category_summary["_vs_".join((first, second))] = {
            "shared_physical_error_locations": sum(counts.values()),
            "same_error_category": sum(
                pair_categories.get((first, second, error_type, error_type), 0)
                for error_type in ("EA", "ED", "EC")
            ),
            "category_crosstab": counts,
        }
    triple_shared = sum(triple_categories.values())
    triple_same_category = sum(
        triple_categories.get((error_type, error_type, error_type), 0)
        for error_type in ("EA", "ED", "EC")
    )
    ta_candidate_total = sum(
        record["metadata"]["trackastra_candidates_t_plus_1"] for record in records
    )
    ho_candidate_total = sum(
        record["metadata"]["hoct_candidates_t_plus_1"] for record in records
    )
    common_candidate_total = sum(
        record["metadata"]["common_candidates_t_plus_1"] for record in records
    )

    summary = {
        "scope": "Trackastra node-only, HOCT, and ungated distance-Hungarian on the same 35 holdout sequences and GT TRA detections.",
        "decision_universes": {
            "ground_truth_links": "All GT links; EA or EC is an error.",
            "ground_truth_continuation_links": "GT continuation links only; EA or EC is an error.",
            "ground_truth_division_links": "Both daughter edges of every GT division; EA or EC is an error.",
            "negative_candidate_links": "Pairwise candidate intersection for pairwise tests; Trackastra-HOCT intersection for three-way tests; ED is an error.",
            "combined": "GT links plus the applicable non-GT candidate intersection.",
        },
        "error_counts": {
            model: {
                "total": len(rows),
                "types": dict(sorted(Counter(row["type"] for row in rows).items())),
            }
            for model, rows in all_rows.items()
        },
        "pairwise": pairwise_summary,
        "three_way": three_way_summary,
        "error_category_overlap": {
            "pairwise": pair_category_summary,
            "all_three_shared_physical_error_locations": triple_shared,
            "all_three_same_error_category": triple_same_category,
        },
        "candidate_graph_comparison": {
            "trackastra_physical_t_plus_1_candidates": ta_candidate_total,
            "hoct_physical_t_plus_1_candidates": ho_candidate_total,
            "shared_physical_t_plus_1_candidates": common_candidate_total,
            "jaccard": safe_ratio(
                common_candidate_total,
                ta_candidate_total + ho_candidate_total - common_candidate_total,
            ),
        },
        "provenance": {
            "trackastra_errors": str(args.trackastra_errors.resolve()),
            "hoct_errors": str(args.hoct_errors.resolve()),
            "hungarian_errors": str(args.hungarian_errors.resolve()),
            "sequences": len(records),
            "datasets": len(dataset_records),
            "bootstrap_seed": args.seed,
        },
        "interpretation_limit": "These are associations among three fixed trackers sharing detections and data, not a causal claim or a population estimate for arbitrary trackers.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_sequence.csv", [flatten_record(record) for record in records])
    write_csv(
        args.output_dir / "per_dataset.csv", [flatten_record(record) for record in dataset_records]
    )
    write_csv(
        args.output_dir / "per_dataset_error_counts.csv",
        [record["metadata"] for record in dataset_records],
    )
    write_csv(
        args.output_dir / "pairwise_error_category_crosstab.csv",
        [
            {
                "first_model": first,
                "second_model": second,
                "first_error_type": first_type,
                "second_error_type": second_type,
                "count": count,
            }
            for (first, second, first_type, second_type), count in sorted(pair_categories.items())
        ],
    )
    write_csv(
        args.output_dir / "triple_error_category_crosstab.csv",
        [
            {
                "trackastra_error_type": ta_type,
                "hoct_error_type": ho_type,
                "hungarian_error_type": hu_type,
                "count": count,
            }
            for (ta_type, ho_type, hu_type), count in sorted(triple_categories.items())
        ],
    )
    write_membership(args.output_dir / "error_membership.csv.gz", memberships)
    print(json.dumps(three_way_summary["combined"], indent=2))
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
