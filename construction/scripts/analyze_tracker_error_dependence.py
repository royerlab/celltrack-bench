#!/usr/bin/env python3
"""Test whether Trackastra and HOCT errors co-locate more than expected.

The analysis treats each physical t -> t+1 link as a paired binary decision.
Ground-truth links test missing or wrong-semantic links (EA/EC), while shared
non-ground-truth candidate links test erroneous additions (ED).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
import zarr
from scipy import special, stats


REPO_ROOT = Path(__file__).resolve().parent.parent
ERROR_ROOT = REPO_ROOT / "evaluation" / "errors_gt_linking"
DEFAULT_OUTPUT = ERROR_ROOT / "trackastra_node_only_vs_hoct" / "error_dependence"
UNIVERSES = ("ground_truth_links", "negative_candidate_links", "combined")


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
        "--trackastra-graphs",
        type=Path,
        default=REPO_ROOT / "evaluation" / "trackastra_node_only_groupcv" / "full_graphs",
    )
    parser.add_argument(
        "--hoct-graphs",
        type=Path,
        default=REPO_ROOT / "evaluation" / "hoct_groupcv" / "full_graphs",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sequence_key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["sequence"]


def physical_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(float(row["frame_from"])),
        int(float(row["gt_label_from"])),
        int(float(row["frame_to"])),
        int(float(row["gt_label_to"])),
    )


def group_errors(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[sequence_key(row)].append(row)
    return dict(grouped)


def parse_track_file(path: Path) -> dict[int, tuple[int, int, int]]:
    tracks: dict[int, tuple[int, int, int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) >= 4:
                label, begin, end, parent = map(int, fields[:4])
                tracks[label] = (begin, end, parent)
    return tracks


def ground_truth_edges(tracks: dict[int, tuple[int, int, int]]) -> dict[tuple[int, int, int, int], str]:
    edges: dict[tuple[int, int, int, int], str] = {}
    for label, (begin, end, parent) in tracks.items():
        for frame in range(begin, end):
            edges[(frame, label, frame + 1, label)] = "continuation"
        if parent > 0:
            edges[(max(0, begin - 1), parent, begin, label)] = "division"
    return edges


def frame_from_mask_path(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot determine frame from {path}")
    return int(digits)


def mask_paths(gt_tra_dir: Path) -> dict[int, Path]:
    paths = sorted(gt_tra_dir.glob("man_track*.tif"))
    return {frame_from_mask_path(path): path for path in paths}


def graph_arrays(graph_dir: Path) -> dict[str, np.ndarray]:
    group = zarr.open_group(str(graph_dir), mode="r")
    arrays = {
        "node_ids": np.asarray(group["nodes/ids"][:], dtype=np.int64),
        "t": np.asarray(group["nodes/props/t/values"][:], dtype=np.int64),
        "x": np.asarray(group["nodes/props/x/values"][:], dtype=np.float64),
        "y": np.asarray(group["nodes/props/y/values"][:], dtype=np.float64),
        "z": np.asarray(group["nodes/props/z/values"][:], dtype=np.float64),
        "edges": np.asarray(group["edges/ids"][:], dtype=np.int64),
    }
    node_count = len(arrays["node_ids"])
    if any(len(arrays[field]) != node_count for field in ("t", "x", "y", "z")):
        raise ValueError(f"Inconsistent node arrays in {graph_dir}")
    return arrays


def verify_common_nodes(trackastra: dict[str, np.ndarray], hoct: dict[str, np.ndarray], label: str) -> None:
    for field in ("node_ids", "t", "x", "y", "z"):
        if not np.array_equal(trackastra[field], hoct[field], equal_nan=True):
            raise ValueError(f"Candidate graph node mismatch for {label}: {field}")


def labels_at_centroids(
    graph: dict[str, np.ndarray],
    gt_tra_dir: Path,
) -> np.ndarray:
    labels = np.zeros(len(graph["node_ids"]), dtype=np.int64)
    paths = mask_paths(gt_tra_dir)
    for frame in np.unique(graph["t"]):
        frame = int(frame)
        path = paths.get(frame)
        if path is None:
            raise FileNotFoundError(f"Missing GT mask for frame {frame}: {gt_tra_dir}")
        mask = np.asarray(tifffile.imread(path))
        indices = np.flatnonzero(graph["t"] == frame)
        yy = np.rint(graph["y"][indices]).astype(np.int64)
        xx = np.rint(graph["x"][indices]).astype(np.int64)
        if mask.ndim == 2:
            yy = np.clip(yy, 0, mask.shape[0] - 1)
            xx = np.clip(xx, 0, mask.shape[1] - 1)
            labels[indices] = mask[yy, xx]
        elif mask.ndim == 3:
            zz = np.rint(graph["z"][indices]).astype(np.int64)
            zz = np.clip(zz, 0, mask.shape[0] - 1)
            yy = np.clip(yy, 0, mask.shape[1] - 1)
            xx = np.clip(xx, 0, mask.shape[2] - 1)
            labels[indices] = mask[zz, yy, xx]
        else:
            raise ValueError(f"Unsupported GT mask shape {mask.shape}: {path}")

        missing = indices[labels[indices] <= 0]
        for index in missing:
            y = int(round(float(graph["y"][index])))
            x = int(round(float(graph["x"][index])))
            z = int(round(float(graph["z"][index])))
            plane = mask if mask.ndim == 2 else mask[np.clip(z, 0, mask.shape[0] - 1)]
            y = int(np.clip(y, 0, plane.shape[0] - 1))
            x = int(np.clip(x, 0, plane.shape[1] - 1))
            values = plane[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
            values = values[values > 0]
            if values.size:
                unique, counts = np.unique(values, return_counts=True)
                labels[index] = int(unique[np.argmax(counts)])

    return labels


def node_indices(node_ids: np.ndarray, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(node_ids)
    sorted_ids = node_ids[order]
    flat = endpoints.reshape(-1)
    positions = np.searchsorted(sorted_ids, flat)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], flat):
        raise ValueError("Graph edge references an unknown node ID")
    indices = order[positions].reshape(endpoints.shape)
    return indices[:, 0], indices[:, 1]


def edge_encoder(
    times: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    gt_edges: Iterable[tuple[int, int, int, int]],
):
    gt_edges = list(gt_edges)
    max_label = max(
        int(source_labels.max(initial=0)),
        int(target_labels.max(initial=0)),
        max((max(edge[1], edge[3]) for edge in gt_edges), default=0),
    )
    base = max_label + 1
    max_time = max(int(times.max(initial=0)), max((edge[0] for edge in gt_edges), default=0))
    max_code = (max_time * base + max_label) * base + max_label
    if max_code > np.iinfo(np.uint64).max:
        raise OverflowError("Link labels cannot be encoded in uint64")

    def encode_arrays(t: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (
            (t.astype(np.uint64) * np.uint64(base) + source.astype(np.uint64))
            * np.uint64(base)
            + target.astype(np.uint64)
        )

    def encode_key(key: tuple[int, int, int, int]) -> int:
        return (key[0] * base + key[1]) * base + key[3]

    return encode_arrays, encode_key


def contingency(universe_size: int, trackastra_errors: set[Any], hoct_errors: set[Any]) -> dict[str, int]:
    both = len(trackastra_errors & hoct_errors)
    trackastra_only = len(trackastra_errors) - both
    hoct_only = len(hoct_errors) - both
    neither = universe_size - both - trackastra_only - hoct_only
    if neither < 0:
        raise ValueError("Error union is larger than the decision universe")
    return {
        "both_error": both,
        "trackastra_only_error": trackastra_only,
        "hoct_only_error": hoct_only,
        "neither_error": neither,
    }


def add_tables(*tables: dict[str, int]) -> dict[str, int]:
    fields = ("both_error", "trackastra_only_error", "hoct_only_error", "neither_error")
    return {field: sum(table[field] for table in tables) for field in fields}


def table_values(table: dict[str, int]) -> tuple[int, int, int, int]:
    return (
        table["both_error"],
        table["trackastra_only_error"],
        table["hoct_only_error"],
        table["neither_error"],
    )


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def table_metrics(table: dict[str, int], include_fisher: bool = True) -> dict[str, Any]:
    a, b, c, d = table_values(table)
    total = a + b + c + d
    trackastra_errors = a + b
    hoct_errors = a + c
    p_h_given_t = safe_ratio(a, trackastra_errors)
    p_h_given_not_t = safe_ratio(c, c + d)
    p_t_given_h = safe_ratio(a, hoct_errors)
    p_t_given_not_h = safe_ratio(b, b + d)
    expected = safe_ratio(trackastra_errors * hoct_errors, total)
    odds_ratio = safe_ratio(a * d, b * c)
    risk_ratio_h = (
        safe_ratio(p_h_given_t, p_h_given_not_t)
        if p_h_given_t is not None and p_h_given_not_t is not None
        else None
    )
    risk_ratio_t = (
        safe_ratio(p_t_given_h, p_t_given_not_h)
        if p_t_given_h is not None and p_t_given_not_h is not None
        else None
    )
    result: dict[str, Any] = {
        **table,
        "decisions": total,
        "trackastra_errors": trackastra_errors,
        "hoct_errors": hoct_errors,
        "hoct_error_rate_given_trackastra_error": p_h_given_t,
        "hoct_error_rate_given_trackastra_correct": p_h_given_not_t,
        "trackastra_error_rate_given_hoct_error": p_t_given_h,
        "trackastra_error_rate_given_hoct_correct": p_t_given_not_h,
        "hoct_conditional_risk_ratio": risk_ratio_h,
        "trackastra_conditional_risk_ratio": risk_ratio_t,
        "odds_ratio": odds_ratio,
        "expected_overlap_under_independence": expected,
        "overlap_enrichment": safe_ratio(a, expected) if expected is not None else None,
    }
    if include_fisher:
        fisher = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        result["fisher_odds_ratio"] = float(fisher.statistic)
        result["fisher_p_one_sided"] = float(fisher.pvalue)
    return result


def flatten_tables(
    identity: dict[str, Any],
    tables: dict[str, dict[str, int]],
) -> dict[str, Any]:
    row = dict(identity)
    for universe in UNIVERSES:
        metrics = table_metrics(tables[universe], include_fisher=False)
        for field, value in metrics.items():
            row[f"{universe}_{field}"] = value
    return row


def sequence_record(
    dataset: str,
    sequence: str,
    trackastra_graph_dir: Path,
    hoct_graph_dir: Path,
    datasets_dir: Path,
    trackastra_rows: list[dict[str, str]],
    hoct_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    trackastra_graph = graph_arrays(trackastra_graph_dir)
    hoct_graph = graph_arrays(hoct_graph_dir)
    verify_common_nodes(trackastra_graph, hoct_graph, f"{dataset}/{sequence}")

    gt_tra_dir = datasets_dir / dataset / f"{sequence}_GT" / "TRA"
    tracks = parse_track_file(gt_tra_dir / "man_track.txt")
    gt_edge_map = ground_truth_edges(tracks)
    gt_edge_set = set(gt_edge_map)
    labels = labels_at_centroids(trackastra_graph, gt_tra_dir)

    mapped_nodes = labels > 0
    node_pairs = np.stack((trackastra_graph["t"][mapped_nodes], labels[mapped_nodes]), axis=1)
    unique_node_pairs = len(np.unique(node_pairs, axis=0))

    def direct_candidate_components(graph: dict[str, np.ndarray]):
        source_indices, target_indices = node_indices(graph["node_ids"], graph["edges"])
        source_times = graph["t"][source_indices]
        target_times = graph["t"][target_indices]
        direct = (
            (target_times == source_times + 1)
            & (labels[source_indices] > 0)
            & (labels[target_indices] > 0)
        )
        return source_times[direct], labels[source_indices[direct]], labels[target_indices[direct]]

    ta_times, ta_sources, ta_targets = direct_candidate_components(trackastra_graph)
    ho_times, ho_sources, ho_targets = direct_candidate_components(hoct_graph)
    encode_arrays, encode_key = edge_encoder(
        np.concatenate((ta_times, ho_times)),
        np.concatenate((ta_sources, ho_sources)),
        np.concatenate((ta_targets, ho_targets)),
        gt_edge_set,
    )
    ta_candidate_codes = np.unique(encode_arrays(ta_times, ta_sources, ta_targets))
    ho_candidate_codes = np.unique(encode_arrays(ho_times, ho_sources, ho_targets))
    common_candidate_codes = np.intersect1d(
        ta_candidate_codes, ho_candidate_codes, assume_unique=True
    )
    gt_codes = np.fromiter(
        (encode_key(edge) for edge in gt_edge_set), dtype=np.uint64, count=len(gt_edge_set)
    )
    ta_gt_candidate_count = int(np.count_nonzero(np.isin(gt_codes, ta_candidate_codes)))
    ho_gt_candidate_count = int(np.count_nonzero(np.isin(gt_codes, ho_candidate_codes)))
    common_gt_candidate_count = int(np.count_nonzero(np.isin(gt_codes, common_candidate_codes)))
    negative_candidate_count = len(common_candidate_codes) - common_gt_candidate_count

    ta_positive = {physical_key(row) for row in trackastra_rows if row["type"] in {"EA", "EC"}}
    ho_positive = {physical_key(row) for row in hoct_rows if row["type"] in {"EA", "EC"}}
    ta_negative = {physical_key(row) for row in trackastra_rows if row["type"] == "ED"}
    ho_negative = {physical_key(row) for row in hoct_rows if row["type"] == "ED"}

    if not ta_positive <= gt_edge_set or not ho_positive <= gt_edge_set:
        raise ValueError(f"EA/EC outside GT links: {dataset}/{sequence}")
    if (ta_negative | ho_negative) & gt_edge_set:
        raise ValueError(f"ED overlaps a GT link: {dataset}/{sequence}")
    def restrict_to_candidates(
        errors: set[tuple[int, int, int, int]], candidate_codes: np.ndarray, model: str
    ) -> tuple[set[tuple[int, int, int, int]], int]:
        ordered = sorted(errors)
        codes = np.fromiter(
            (encode_key(edge) for edge in ordered), dtype=np.uint64, count=len(ordered)
        )
        own_candidates = ta_candidate_codes if model == "Trackastra" else ho_candidate_codes
        if len(codes) and not np.all(np.isin(codes, own_candidates)):
            raise ValueError(f"{model} ED outside its candidate graph: {dataset}/{sequence}")
        in_common = np.isin(codes, candidate_codes)
        return {edge for edge, keep in zip(ordered, in_common, strict=True) if keep}, int(
            np.count_nonzero(~in_common)
        )

    ta_negative_common, ta_negative_excluded = restrict_to_candidates(
        ta_negative, common_candidate_codes, "Trackastra"
    )
    ho_negative_common, ho_negative_excluded = restrict_to_candidates(
        ho_negative, common_candidate_codes, "HOCT"
    )

    positive = contingency(len(gt_edge_set), ta_positive, ho_positive)
    negative = contingency(negative_candidate_count, ta_negative_common, ho_negative_common)
    tables = {
        "ground_truth_links": positive,
        "negative_candidate_links": negative,
        "combined": add_tables(positive, negative),
    }
    metadata = {
        "dataset": dataset,
        "sequence": sequence,
        "nodes": len(trackastra_graph["node_ids"]),
        "nodes_unmapped_to_gt": int(np.count_nonzero(~mapped_nodes)),
        "nodes_after_gt_object_mapping": unique_node_pairs,
        "node_to_gt_mapping_collisions": int(np.count_nonzero(mapped_nodes))
        - unique_node_pairs,
        "trackastra_candidate_edges_all_time_gaps": len(trackastra_graph["edges"]),
        "hoct_candidate_edges_all_time_gaps": len(hoct_graph["edges"]),
        "trackastra_candidate_edges_t_plus_1_before_gt_mapping_dedup": len(ta_times),
        "hoct_candidate_edges_t_plus_1_before_gt_mapping_dedup": len(ho_times),
        "trackastra_candidate_edges_t_plus_1": len(ta_candidate_codes),
        "hoct_candidate_edges_t_plus_1": len(ho_candidate_codes),
        "common_candidate_edges_t_plus_1": len(common_candidate_codes),
        "trackastra_only_candidate_edges_t_plus_1": len(ta_candidate_codes)
        - len(common_candidate_codes),
        "hoct_only_candidate_edges_t_plus_1": len(ho_candidate_codes)
        - len(common_candidate_codes),
        "candidate_topology_identical": bool(
            np.array_equal(ta_candidate_codes, ho_candidate_codes)
        ),
        "ground_truth_links": len(gt_edge_set),
        "trackastra_ground_truth_links_in_candidate_graph": ta_gt_candidate_count,
        "hoct_ground_truth_links_in_candidate_graph": ho_gt_candidate_count,
        "common_ground_truth_links_in_candidate_graph": common_gt_candidate_count,
        "trackastra_ground_truth_links_absent_from_candidate_graph": len(gt_edge_set)
        - ta_gt_candidate_count,
        "hoct_ground_truth_links_absent_from_candidate_graph": len(gt_edge_set)
        - ho_gt_candidate_count,
        "common_negative_candidate_links": negative_candidate_count,
        "trackastra_ED_excluded_outside_common_candidates": ta_negative_excluded,
        "hoct_ED_excluded_outside_common_candidates": ho_negative_excluded,
    }
    return metadata, tables


def aggregate_by(
    sequence_tables: list[tuple[dict[str, Any], dict[str, dict[str, int]]]],
    field: str,
) -> list[tuple[dict[str, Any], dict[str, dict[str, int]]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, dict[str, int]]]]] = defaultdict(list)
    for record in sequence_tables:
        grouped[str(record[0][field])].append(record)
    output = []
    for value, records in sorted(grouped.items()):
        tables = {
            universe: add_tables(*(record[1][universe] for record in records))
            for universe in UNIVERSES
        }
        metadata = {field: value, "sequences": len(records)}
        output.append((metadata, tables))
    return output


def aggregate_all(
    records: list[tuple[dict[str, Any], dict[str, dict[str, int]]]],
) -> dict[str, dict[str, int]]:
    return {
        universe: add_tables(*(record[1][universe] for record in records))
        for universe in UNIVERSES
    }


def stratified_test(
    records: list[tuple[dict[str, Any], dict[str, dict[str, int]]]],
    universe: str,
) -> dict[str, Any]:
    observed = 0.0
    expected = 0.0
    variance = 0.0
    mh_numerator = 0.0
    mh_denominator = 0.0
    valid_strata = 0
    for _metadata, tables in records:
        a, b, c, d = table_values(tables[universe])
        total = a + b + c + d
        if total <= 1:
            continue
        trackastra_errors = a + b
        hoct_errors = a + c
        expected_i = trackastra_errors * hoct_errors / total
        variance_i = (
            trackastra_errors
            * hoct_errors
            * (total - trackastra_errors)
            * (total - hoct_errors)
            / (total * total * (total - 1))
        )
        observed += a
        expected += expected_i
        variance += variance_i
        mh_numerator += a * d / total
        mh_denominator += b * c / total
        valid_strata += 1

    z_score = safe_ratio(observed - expected, math.sqrt(variance))
    log_p = special.log_ndtr(-z_score) if z_score is not None else float("nan")
    return {
        "strata": valid_strata,
        "observed_overlap": observed,
        "expected_overlap_conditional_on_sequence_margins": expected,
        "overlap_enrichment": safe_ratio(observed, expected),
        "z_score": z_score,
        "one_sided_p": float(math.exp(log_p)) if log_p > math.log(np.finfo(float).tiny) else 0.0,
        "one_sided_log10_p": float(log_p / math.log(10.0)),
        "mantel_haenszel_common_odds_ratio": safe_ratio(mh_numerator, mh_denominator),
    }


def dataset_direction_test(
    dataset_records: list[tuple[dict[str, Any], dict[str, dict[str, int]]]],
    universe: str,
) -> dict[str, Any]:
    enrichments: list[float] = []
    for _metadata, tables in dataset_records:
        metrics = table_metrics(tables[universe], include_fisher=False)
        enrichment = metrics["overlap_enrichment"]
        if enrichment is not None and math.isfinite(enrichment) and enrichment != 1.0:
            enrichments.append(enrichment)
    above = sum(value > 1.0 for value in enrichments)
    sign_test = stats.binomtest(above, len(enrichments), 0.5, alternative="greater")
    return {
        "datasets_with_defined_non_tied_effect": len(enrichments),
        "datasets_with_overlap_enrichment_above_one": above,
        "one_sided_sign_test_p": float(sign_test.pvalue),
    }


def dataset_bootstrap(
    dataset_records: list[tuple[dict[str, Any], dict[str, dict[str, int]]]],
    universe: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    cells = np.asarray(
        [table_values(tables[universe]) for _metadata, tables in dataset_records],
        dtype=np.int64,
    )
    estimates = np.empty(replicates, dtype=np.float64)
    dataset_count = len(cells)
    for start in range(0, replicates, 2_000):
        stop = min(replicates, start + 2_000)
        indices = rng.integers(0, dataset_count, size=(stop - start, dataset_count))
        totals = cells[indices].sum(axis=1)
        a, b, c, d = totals.T.astype(np.float64)
        estimates[start:stop] = (a / (a + b)) / (c / (c + d))
    finite = estimates[np.isfinite(estimates)]
    return {
        "replicates": replicates,
        "finite_replicates": len(finite),
        "risk_ratio_median": float(np.median(finite)),
        "risk_ratio_ci95": [float(value) for value in np.quantile(finite, [0.025, 0.975])],
        "fraction_at_or_below_one": float(np.mean(finite <= 1.0)),
    }


def leave_one_dataset_out(
    dataset_records: list[tuple[dict[str, Any], dict[str, dict[str, int]]]],
    universe: str,
) -> dict[str, Any]:
    estimates = []
    for excluded, _tables in dataset_records:
        included = [record for record in dataset_records if record[0]["dataset"] != excluded["dataset"]]
        table = add_tables(*(record[1][universe] for record in included))
        metrics = table_metrics(table, include_fisher=False)
        estimates.append((excluded["dataset"], metrics["hoct_conditional_risk_ratio"]))
    finite = [(dataset, value) for dataset, value in estimates if value is not None and math.isfinite(value)]
    minimum = min(finite, key=lambda item: item[1])
    maximum = max(finite, key=lambda item: item[1])
    return {
        "minimum_risk_ratio": minimum[1],
        "minimum_when_excluding": minimum[0],
        "maximum_risk_ratio": maximum[1],
        "maximum_when_excluding": maximum[0],
    }


def main() -> None:
    args = parse_args()
    trackastra_rows = read_csv(args.trackastra_errors)
    hoct_rows = read_csv(args.hoct_errors)
    ta_grouped = group_errors(trackastra_rows)
    ho_grouped = group_errors(hoct_rows)

    graph_dirs = sorted(args.trackastra_graphs.glob("group_*/*/*.geff"))
    if not graph_dirs:
        raise FileNotFoundError(f"No Trackastra graphs under {args.trackastra_graphs}")

    sequence_tables: list[tuple[dict[str, Any], dict[str, dict[str, int]]]] = []
    for index, trackastra_graph_dir in enumerate(graph_dirs, start=1):
        relative = trackastra_graph_dir.relative_to(args.trackastra_graphs)
        hoct_graph_dir = args.hoct_graphs / relative
        dataset = trackastra_graph_dir.parent.name
        sequence = trackastra_graph_dir.stem
        if not hoct_graph_dir.exists():
            raise FileNotFoundError(f"Missing paired HOCT graph: {hoct_graph_dir}")
        print(f"[{index:02d}/{len(graph_dirs):02d}] {dataset}/{sequence}", flush=True)
        sequence_tables.append(
            sequence_record(
                dataset=dataset,
                sequence=sequence,
                trackastra_graph_dir=trackastra_graph_dir,
                hoct_graph_dir=hoct_graph_dir,
                datasets_dir=args.datasets_dir,
                trackastra_rows=ta_grouped.get((dataset, sequence), []),
                hoct_rows=ho_grouped.get((dataset, sequence), []),
            )
        )

    seen_sequences = {(metadata["dataset"], metadata["sequence"]) for metadata, _ in sequence_tables}
    if set(ta_grouped) - seen_sequences or set(ho_grouped) - seen_sequences:
        raise ValueError("Error CSV contains sequences without paired candidate graphs")

    dataset_records = aggregate_by(sequence_tables, "dataset")
    overall_tables = aggregate_all(sequence_tables)
    rng = np.random.default_rng(args.seed)
    overall: dict[str, Any] = {}
    for universe in UNIVERSES:
        overall[universe] = {
            "paired_table": table_metrics(overall_tables[universe]),
            "sequence_stratified_test": stratified_test(sequence_tables, universe),
            "dataset_direction_test": dataset_direction_test(dataset_records, universe),
            "dataset_cluster_bootstrap": dataset_bootstrap(
                dataset_records, universe, args.bootstrap_replicates, rng
            ),
            "leave_one_dataset_out": leave_one_dataset_out(dataset_records, universe),
        }

    combined = overall_tables["combined"]
    ta_excluded = sum(
        metadata["trackastra_ED_excluded_outside_common_candidates"]
        for metadata, _tables in sequence_tables
    )
    ho_excluded = sum(
        metadata["hoct_ED_excluded_outside_common_candidates"]
        for metadata, _tables in sequence_tables
    )
    if combined["both_error"] != 8_480:
        raise ValueError(
            f"Physical overlap changed: expected 8480, found {combined['both_error']}"
        )
    if combined["both_error"] + combined["trackastra_only_error"] != len(trackastra_rows) - ta_excluded:
        raise ValueError("Trackastra physical error count does not match the source CSV")
    if combined["both_error"] + combined["hoct_only_error"] != len(hoct_rows) - ho_excluded:
        raise ValueError("HOCT physical error count does not match the source CSV")

    sequence_output = [flatten_tables(metadata, tables) for metadata, tables in sequence_tables]
    dataset_output = [flatten_tables(metadata, tables) for metadata, tables in dataset_records]
    summary = {
        "question": "Does an error by one tracker increase the probability of an error by the other tracker at the same physical link?",
        "scope": "Trackastra node-only and HOCT group-CV holdouts on 35 sequences from 15 datasets.",
        "decision_unit": "Physical t to t+1 link; error categories may differ between trackers.",
        "universes": {
            "ground_truth_links": "All GT links; EA and EC are errors.",
            "negative_candidate_links": "Intersection of both models' t+1 candidate links, excluding GT; ED is an error.",
            "combined": "Union of all GT links and shared non-GT t+1 candidates.",
        },
        "provenance": {
            "trackastra_errors": str(args.trackastra_errors.resolve()),
            "hoct_errors": str(args.hoct_errors.resolve()),
            "trackastra_graphs": str(args.trackastra_graphs.resolve()),
            "hoct_graphs": str(args.hoct_graphs.resolve()),
            "paired_graphs": len(sequence_tables),
            "datasets": len(dataset_records),
            "paired_graphs_with_identical_t_plus_1_candidates": sum(
                metadata["candidate_topology_identical"] for metadata, _tables in sequence_tables
            ),
            "trackastra_ED_excluded_outside_common_candidates": ta_excluded,
            "hoct_ED_excluded_outside_common_candidates": ho_excluded,
            "bootstrap_seed": args.seed,
        },
        "overall": overall,
        "interpretation_limit": "Association for this tracker pair and benchmark; it does not establish causation or a universal rate for unseen tracker architectures.",
    }

    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "per_sequence.csv", sequence_output)
    write_csv(args.output_dir / "per_dataset.csv", dataset_output)
    print(json.dumps(overall["combined"], indent=2))
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
