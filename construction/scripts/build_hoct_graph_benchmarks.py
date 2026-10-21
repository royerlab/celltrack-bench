#!/usr/bin/env python3
"""Build HOCT-anchored cell-linking, track-repair, and inspection benchmarks.

Each shared sample contains five cells at frame t and the union of their
frame-(t+1) neighbors from the saved HOCT candidate graph. A legacy hard/easy
HOCT error or a consensus-selected physical error anchors the sample, while
labels include every HOCT-vs-GT edge error in the induced two-frame graph.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import shutil
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "hoct_graph_benchmarks"
DEFAULT_HARD = (
    REPO_ROOT
    / "evaluation"
    / "errors_gt_linking"
    / "hoct_groupcv"
    / "selected_top72_per_dataset"
    / "selected_top72_per_dataset.csv"
)
DEFAULT_EASY = (
    REPO_ROOT
    / "evaluation"
    / "errors_gt_linking"
    / "hoct_groupcv"
    / "selected_easy8_per_dataset"
    / "selected_easy8_per_dataset.csv"
)
DEFAULT_ALL_ERRORS = (
    REPO_ROOT / "evaluation" / "errors_gt_linking" / "hoct_groupcv" / "all_groups_link_errors.csv"
)
DEFAULT_SELECTED_ANCHORS = (
    REPO_ROOT
    / "evaluation"
    / "errors_gt_linking"
    / "trackastra_hoct_hungarian_analysis"
    / "selected_top64_per_split_per_dataset"
    / "tracker_count_then_hardness.csv"
)
DEFAULT_FULL_GRAPHS = REPO_ROOT / "evaluation" / "hoct_groupcv" / "full_graphs"
DEFAULT_DATASETS = REPO_ROOT / "datasets"

PARTS = ("cell_linking", "track_repair", "track_inspection")
ERROR_TO_OPERATION = {"EA": "add", "ED": "delete", "EC": "change_semantic"}
TRACKERS = ("trackastra", "hoct", "hungarian")


@dataclass(frozen=True)
class TrackRecord:
    track_id: int
    start: int
    end: int
    parent: int


@dataclass(frozen=True)
class GraphNode:
    node_id: int
    frame: int
    label: int
    y: float
    x: float


@dataclass(frozen=True)
class GraphEdge:
    source_id: int
    target_id: int
    source_label: int
    target_label: int
    frame_from: int
    frame_to: int
    similarity: float
    solution: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--hard-anchors", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--easy-anchors", type=Path, default=DEFAULT_EASY)
    parser.add_argument(
        "--selected-anchors",
        type=Path,
        nargs="?",
        const=DEFAULT_SELECTED_ANCHORS,
        default=None,
        help=(
            "Use a consensus-first selected-anchor CSV instead of the legacy hard/easy CSVs. "
            "When passed without a path, uses the top-64 split selection."
        ),
    )
    parser.add_argument(
        "--data-split",
        choices=("validation", "test"),
        default=None,
        help="Required with --selected-anchors; only anchors from this sequence split are loaded.",
    )
    parser.add_argument(
        "--full-graphs-dir",
        type=Path,
        default=DEFAULT_FULL_GRAPHS,
        help="HOCT group-CV graph root used to resolve selected dataset/sequence rows.",
    )
    parser.add_argument("--all-errors", type=Path, default=DEFAULT_ALL_ERRORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset allowlist.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--source-count", type=int, default=5)
    parser.add_argument("--anchor-center-probability", type=float, default=0.20)
    parser.add_argument(
        "--neighbor-pool",
        type=int,
        default=4,
        help="Closest eligible proxy neighbors from which a non-proxy sampling center may be drawn.",
    )
    parser.add_argument("--panel-size", type=int, default=768)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Reference size used by the source compactness limits; rendered crops are candidate-tight.",
    )
    parser.add_argument("--crop-margin", type=int, default=48)
    parser.add_argument(
        "--max-source-radius-fraction",
        type=float,
        default=0.40,
        help="Maximum source distance from the sampling center as a fraction of crop size.",
    )
    parser.add_argument(
        "--max-source-diameter-fraction",
        type=float,
        default=0.50,
        help="Maximum source pairwise distance as a fraction of crop size.",
    )
    parser.add_argument(
        "--max-normalized-source-diameter",
        type=float,
        default=5.0,
        help="Maximum source diameter in local nearest-frame-cell spacing units.",
    )
    parser.add_argument(
        "--pixel-compactness-exempt-datasets",
        default="PhC-C2DH-U373",
        help="Comma-separated datasets exempt from pixel radius/diameter limits (normalized limit still applies).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_or_abs(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list(value: Any, default: list[float] | None = None) -> list[float]:
    if default is None:
        default = []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return list(default)
    if not isinstance(parsed, (list, tuple)):
        return list(default)
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return list(default)


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def frame_number(path: Path) -> int:
    digits = ""
    for char in reversed(path.stem):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    if not digits:
        raise ValueError(f"Cannot parse frame number from {path}")
    return int(digits)


def frame_path_map(root: Path, pattern: str) -> dict[int, Path]:
    return {frame_number(path): path for path in sorted(root.glob(pattern))}


def squeeze_2d(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)
    while out.ndim > 2 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 3 and out.shape[-1] in (3, 4):
        out = out[..., :3].mean(axis=-1)
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {out.shape}")
    return out


def load_image(path: Path) -> np.ndarray:
    return squeeze_2d(tifffile.imread(str(path)))


def load_mask(path: Path) -> np.ndarray:
    return squeeze_2d(tifffile.imread(str(path))).astype(np.int64, copy=False)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image_f = np.asarray(image, dtype=np.float32)
    finite = image_f[np.isfinite(image_f)]
    if finite.size == 0:
        return np.zeros(image_f.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.7])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(image_f.shape, dtype=np.uint8)
    scaled = (np.clip(image_f, lo, hi) - lo) / (hi - lo)
    return np.asarray(np.round(scaled * 255.0), dtype=np.uint8)


def read_tracks(path: Path) -> dict[int, TrackRecord]:
    tracks: dict[int, TrackRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        track_id, start, end, parent = (int(value) for value in fields[:4])
        tracks[track_id] = TrackRecord(track_id, start, end, parent)
    return tracks


def gt_edges_by_transition(tracks: dict[int, TrackRecord], transitions: set[int]) -> dict[int, dict[tuple[int, int], str]]:
    result: dict[int, dict[tuple[int, int], str]] = {frame: {} for frame in transitions}
    for record in tracks.values():
        for frame in transitions:
            if record.start <= frame < record.end:
                result[frame][(record.track_id, record.track_id)] = "continuation"
        division_frame = record.start - 1
        if record.parent > 0 and division_frame in result:
            result[division_frame][(record.parent, record.track_id)] = "division"
    return result


def label_at(mask: np.ndarray, y: float, x: float) -> int | None:
    yy = max(0, min(mask.shape[0] - 1, int(round(y))))
    xx = max(0, min(mask.shape[1] - 1, int(round(x))))
    value = int(mask[yy, xx])
    if value > 0:
        return value
    radius = 4
    patch = mask[max(0, yy - radius) : yy + radius + 1, max(0, xx - radius) : xx + radius + 1]
    values = patch[patch > 0]
    if not values.size:
        return None
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[int(np.argmax(counts))])


def canonical_anchor(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["sequence"],
        row["type"],
        parse_int(row["frame_from"]),
        parse_int(row["gt_label_from"]),
        parse_int(row["frame_to"]),
        parse_int(row["gt_label_to"]),
    )


def load_anchors(hard_path: Path, easy_path: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for stratum, path in (("hard", hard_path), ("easy", easy_path)):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = canonical_anchor(row)
                if key not in by_key:
                    item = dict(row)
                    item["proxy_strata"] = [stratum]
                    by_key[key] = item
                elif stratum not in by_key[key]["proxy_strata"]:
                    by_key[key]["proxy_strata"].append(stratum)
    anchors = list(by_key.values())
    anchors.sort(
        key=lambda row: (
            row["dataset"],
            row["sequence"],
            parse_int(row["frame_from"]),
            parse_int(row.get("dataset_selection_rank", 0)),
            row["type"],
        )
    )
    return anchors


def full_graph_index(root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    duplicates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(root.glob("group_*/*/*.geff")):
        key = (path.parent.name, path.stem)
        if key in index:
            duplicates[key].extend((index[key], path))
        else:
            index[key] = path
    if duplicates:
        details = "; ".join(
            f"{dataset}/{sequence}: {sorted({str(path) for path in paths})}"
            for (dataset, sequence), paths in sorted(duplicates.items())
        )
        raise ValueError(f"HOCT full graphs are not unique: {details}")
    if not index:
        raise FileNotFoundError(f"No group-CV HOCT graphs found under {root}")
    return index


def selected_anchor_type(row: dict[str, Any]) -> str:
    is_gt = parse_bool(row.get("is_ground_truth_link"))
    errors: list[tuple[str, str]] = []
    for tracker in TRACKERS:
        is_error = parse_bool(row.get(f"{tracker}_error"))
        error_type = str(row.get(f"{tracker}_error_type", "")).strip()
        if is_error and error_type not in ERROR_TO_OPERATION:
            raise ValueError(
                f"Selected row has invalid {tracker} error type {error_type!r}: "
                f"{row.get('dataset')}/{row.get('sequence')}"
            )
        if is_error:
            errors.append((tracker, error_type))
    if not errors:
        raise ValueError(
            f"Selected row is not an error for any tracker: "
            f"{row.get('dataset')}/{row.get('sequence')}"
        )
    if parse_int(row.get("tracker_error_count"), -1) != len(errors):
        raise ValueError(
            f"tracker_error_count mismatch for {row.get('dataset')}/{row.get('sequence')}: "
            f"declared={row.get('tracker_error_count')} observed={len(errors)}"
        )
    allowed = {"EA", "EC"} if is_gt else {"ED"}
    incompatible = [(tracker, error_type) for tracker, error_type in errors if error_type not in allowed]
    if incompatible:
        raise ValueError(
            f"Selected row has error types incompatible with ground-truth membership: {incompatible}"
        )
    if not is_gt:
        return "ED"
    for preferred_tracker in ("hoct", "trackastra", "hungarian"):
        for tracker, error_type in errors:
            if tracker == preferred_tracker:
                return error_type
    raise AssertionError("unreachable")


def load_selected_anchors(path: Path, data_split: str, graph_root: Path) -> list[dict[str, Any]]:
    graph_paths = full_graph_index(graph_root)
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "sequence",
            "frame_from",
            "label_from",
            "frame_to",
            "label_to",
            "tracker_error_count",
            "is_ground_truth_link",
            "hoct_candidate_present",
            "hardness",
            "data_split",
            "split_selection_rank",
            *(f"{tracker}_error" for tracker in TRACKERS),
            *(f"{tracker}_error_type" for tracker in TRACKERS),
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Selected-anchor CSV is missing columns: {missing}")
        for row in reader:
            if row["data_split"] != data_split:
                continue
            dataset, sequence = row["dataset"], row["sequence"]
            graph_path = graph_paths.get((dataset, sequence))
            if graph_path is None:
                raise FileNotFoundError(
                    f"No unique HOCT full graph for selected row {dataset}/{sequence}"
                )
            frame_from = parse_int(row["frame_from"])
            frame_to = parse_int(row["frame_to"])
            label_from = parse_int(row["label_from"])
            label_to = parse_int(row["label_to"])
            key = (dataset, sequence, frame_from, label_from, frame_to, label_to)
            if key in seen:
                raise ValueError(f"Duplicate selected physical link: {key}")
            seen.add(key)
            item = dict(row)
            item.update(
                {
                    "anchor_mode": "selected_consensus_first",
                    "type": selected_anchor_type(row),
                    "gt_label_from": str(label_from),
                    "gt_label_to": str(label_to),
                    "candidate_present": row["hoct_candidate_present"],
                    "difficulty": row["hardness"],
                    "full_graph_path": repo_or_abs(graph_path),
                    "proxy_strata": ["consensus_first"],
                }
            )
            anchors.append(item)
    if not anchors:
        raise ValueError(f"No selected anchors found for data split {data_split!r} in {path}")
    anchors.sort(
        key=lambda row: (
            row["dataset"],
            row["sequence"],
            parse_int(row["split_selection_rank"]),
            parse_int(row["frame_from"]),
            parse_int(row["gt_label_from"]),
            parse_int(row["gt_label_to"]),
        )
    )
    return anchors


def selected_anchor_metadata(anchor: dict[str, Any]) -> dict[str, Any] | None:
    if anchor.get("anchor_mode") != "selected_consensus_first":
        return None
    return {
        "mode": "consensus_first",
        "scenario": anchor.get("scenario") or "tracker_count_then_hardness",
        "data_split": anchor["data_split"],
        "selection_rank": parse_int(anchor["split_selection_rank"]),
        "tracker_error_count": parse_int(anchor["tracker_error_count"]),
        "hardness": parse_float(anchor["hardness"]),
        "is_ground_truth_link": parse_bool(anchor["is_ground_truth_link"]),
        "in_common_decision_universe": parse_bool(anchor.get("in_common_decision_universe")),
        "trackers": {
            tracker: {
                "error": parse_bool(anchor.get(f"{tracker}_error")),
                "error_type": str(anchor.get(f"{tracker}_error_type", "")).strip() or None,
            }
            for tracker in TRACKERS
        },
    }


def load_error_index(path: Path) -> dict[tuple[str, str, int], list[dict[str, str]]]:
    index: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["sequence"], parse_int(row["frame_from"]))
            index[key].append(row)
    return index


def load_graph_transition_data(
    graph_path: Path,
    frames: set[int],
    mask_paths: dict[int, Path],
) -> tuple[dict[tuple[int, int], GraphNode], dict[int, list[GraphEdge]]]:
    import tracksdata as td

    graph, _metadata = td.graph.InMemoryGraph.from_geff(str(graph_path))
    mask_cache = {frame: load_mask(mask_paths[frame]) for frame in sorted(frames) if frame in mask_paths}
    node_keys = [
        td.DEFAULT_ATTR_KEYS.NODE_ID,
        td.DEFAULT_ATTR_KEYS.T,
        td.DEFAULT_ATTR_KEYS.Y,
        td.DEFAULT_ATTR_KEYS.X,
    ]
    node_df = graph.node_attrs(attr_keys=node_keys)
    node_by_id: dict[int, GraphNode] = {}
    node_by_key: dict[tuple[int, int], GraphNode] = {}
    for row in node_df.iter_rows(named=True):
        frame = int(row[td.DEFAULT_ATTR_KEYS.T])
        if frame not in frames or frame not in mask_cache:
            continue
        y = float(row[td.DEFAULT_ATTR_KEYS.Y])
        x = float(row[td.DEFAULT_ATTR_KEYS.X])
        label = label_at(mask_cache[frame], y, x)
        if label is None:
            continue
        node = GraphNode(
            node_id=int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]),
            frame=frame,
            label=label,
            y=y,
            x=x,
        )
        node_by_id[node.node_id] = node
        node_by_key[(frame, label)] = node

    edge_keys = [
        td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        "similarity",
        "solution",
        "delta_t",
    ]
    edge_df = graph.edge_attrs(attr_keys=edge_keys)
    edges_by_frame: dict[int, list[GraphEdge]] = defaultdict(list)
    for row in edge_df.iter_rows(named=True):
        source = node_by_id.get(int(row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]))
        target = node_by_id.get(int(row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]))
        if source is None or target is None or target.frame != source.frame + 1:
            continue
        edge = GraphEdge(
            source_id=source.node_id,
            target_id=target.node_id,
            source_label=source.label,
            target_label=target.label,
            frame_from=source.frame,
            frame_to=target.frame,
            similarity=parse_float(row.get("similarity")),
            solution=bool(row.get("solution", False)),
        )
        edges_by_frame[source.frame].append(edge)
    return node_by_key, edges_by_frame


def sample_sources(
    anchor_source: int,
    eligible_sources: list[int],
    node_by_key: dict[tuple[int, int], GraphNode],
    frame: int,
    rng: random.Random,
    count: int,
    center_probability: float,
    neighbor_pool: int,
) -> tuple[list[int], int, str, int] | None:
    if anchor_source not in eligible_sources or len(eligible_sources) < count:
        return None
    anchor_node = node_by_key[(frame, anchor_source)]

    def distance(label_a: int, label_b: int) -> float:
        node_a = node_by_key[(frame, label_a)]
        node_b = node_by_key[(frame, label_b)]
        return math.hypot(node_a.y - node_b.y, node_a.x - node_b.x)

    neighbors = sorted(
        (label for label in eligible_sources if label != anchor_source),
        key=lambda label: (distance(anchor_source, label), label),
    )
    centered = rng.random() < center_probability
    if centered or not neighbors:
        sampling_center = anchor_source
        center_mode = "proxy_anchor"
        center_neighbor_rank = 0
    else:
        local_neighbors = neighbors[: min(neighbor_pool, count - 1, len(neighbors))]
        sampling_center = rng.choice(local_neighbors)
        center_mode = "random_neighbor"
        center_neighbor_rank = neighbors.index(sampling_center) + 1

    chosen = [anchor_source]
    if sampling_center != anchor_source:
        chosen.append(sampling_center)

    # Grow a compact group around the selected center.  The first objective
    # limits the diameter of the five-cell set; center distance and total
    # distance provide deterministic tie-breakers.
    while len(chosen) < count:
        remaining = [label for label in eligible_sources if label not in chosen]
        if not remaining:
            break
        next_label = min(
            remaining,
            key=lambda label: (
                max(distance(label, selected) for selected in chosen),
                distance(sampling_center, label),
                sum(distance(label, selected) for selected in chosen),
                label,
            ),
        )
        chosen.append(next_label)
    if len(chosen) != count:
        return None
    return chosen, sampling_center, center_mode, center_neighbor_rank


def source_compactness(
    source_labels: list[int],
    sampling_center: int,
    node_by_key: dict[tuple[int, int], GraphNode],
    frame: int,
) -> dict[str, float]:
    def distance(label_a: int, label_b: int) -> float:
        node_a = node_by_key[(frame, label_a)]
        node_b = node_by_key[(frame, label_b)]
        return math.hypot(node_a.y - node_b.y, node_a.x - node_b.x)

    pairwise = [
        distance(label_a, label_b)
        for index, label_a in enumerate(source_labels)
        for label_b in source_labels[index + 1 :]
    ]
    center_distances = [distance(sampling_center, label) for label in source_labels]
    frame_sources = [label for (node_frame, label) in node_by_key if node_frame == frame]
    nearest_frame_cell = []
    for label in source_labels:
        distances = [distance(label, other) for other in frame_sources if other != label]
        positive = [value for value in distances if value > 0]
        if positive:
            nearest_frame_cell.append(min(positive))
    local_scale = float(np.median(nearest_frame_cell)) if nearest_frame_cell else 1.0
    max_pairwise = max(pairwise, default=0.0)
    return {
        "max_pairwise_distance_px": round(max_pairwise, 7),
        "mean_pairwise_distance_px": round(float(np.mean(pairwise)) if pairwise else 0.0, 7),
        "max_radius_from_sampling_center_px": round(max(center_distances, default=0.0), 7),
        "median_nearest_frame_cell_distance_px": round(local_scale, 7),
        "normalized_max_pairwise_distance": round(max_pairwise / max(local_scale, 1e-9), 7),
    }


def label_bbox(mask: np.ndarray, labels: set[int]) -> tuple[int, int, int, int] | None:
    keep = np.isin(mask, list(labels))
    ys, xs = np.nonzero(keep)
    if not ys.size:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def crop_for_sample(
    mask_t: np.ndarray,
    mask_t1: np.ndarray,
    source_labels: set[int],
    target_labels: set[int],
    margin: int,
) -> tuple[int, int, int, int]:
    height, width = mask_t.shape
    boxes = [
        box
        for box in (label_bbox(mask_t, source_labels), label_bbox(mask_t1, target_labels))
        if box is not None
    ]
    if not boxes:
        return 0, 0, height, width
    req_y0 = max(0, min(box[0] for box in boxes) - margin)
    req_x0 = max(0, min(box[1] for box in boxes) - margin)
    req_y1 = min(height, max(box[2] for box in boxes) + margin)
    req_x1 = min(width, max(box[3] for box in boxes) + margin)
    side = max(req_y1 - req_y0, req_x1 - req_x0)
    crop_h = min(height, side)
    crop_w = min(width, side)
    desired_top = int(round((req_y0 + req_y1 - crop_h) / 2.0))
    desired_left = int(round((req_x0 + req_x1 - crop_w) / 2.0))
    min_top = max(0, req_y1 - crop_h)
    max_top = min(req_y0, height - crop_h)
    min_left = max(0, req_x1 - crop_w)
    max_left = min(req_x0, width - crop_w)
    top = min(max(desired_top, min_top), max_top) if min_top <= max_top else min_top
    left = min(max(desired_left, min_left), max_left) if min_left <= max_left else min_left
    top = max(0, min(top, height - crop_h))
    left = max(0, min(left, width - crop_w))
    return top, left, top + crop_h, left + crop_w


def resize_crop_to_panel(
    image_crop: np.ndarray,
    mask_crop: np.ndarray,
    panel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    crop_h, crop_w = image_crop.shape
    scale = min(panel_size / crop_h, panel_size / crop_w)
    resized_h = max(1, int(round(crop_h * scale)))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_image = cv2.resize(image_crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_crop, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    top = (panel_size - resized_h) // 2
    left = (panel_size - resized_w) // 2
    image_panel = np.full((panel_size, panel_size), 18, dtype=np.uint8)
    mask_panel = np.zeros((panel_size, panel_size), dtype=np.int32)
    image_panel[top : top + resized_h, left : left + resized_w] = resized_image
    mask_panel[top : top + resized_h, left : left + resized_w] = resized_mask
    return image_panel, mask_panel


def draw_text(canvas: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render_frame(
    image: np.ndarray,
    mask: np.ndarray,
    crop: tuple[int, int, int, int],
    label_to_display: dict[int, str],
    role: str,
    frame: int,
    panel_size: int,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    y0, x0, y1, x1 = crop
    image_crop = normalize_image(image)[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1].astype(np.int32)
    image_panel, mask_panel = resize_crop_to_panel(image_crop, mask_crop, panel_size)
    panel = np.repeat(image_panel[..., None], 3, axis=2).astype(np.float32)
    fill_color = np.asarray((45, 185, 245) if role == "source" else (75, 215, 95), dtype=np.float32)
    for label in label_to_display:
        region = mask_panel == label
        panel[region] = 0.48 * panel[region] + 0.52 * fill_color
    panel_u8 = np.clip(panel, 0, 255).astype(np.uint8)
    centers: dict[str, tuple[int, int]] = {}
    for label, display in label_to_display.items():
        region = (mask_panel == label).astype(np.uint8)
        if not np.any(region):
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel_u8, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
        ys, xs = np.nonzero(region)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        centers[display] = (cx, cy + 58)
        draw_text(panel_u8, display, (max(2, cx - 18), max(20, cy + 7)), 0.56, (255, 255, 255))
    header = np.full((58, panel_size, 3), 24, dtype=np.uint8)
    title = f"Frame t ({frame}) - five source cells" if role == "source" else f"Frame t+1 ({frame}) - candidate target cells"
    draw_text(header, title, (12, 37), 0.58, (238, 238, 238))
    return np.concatenate([header, panel_u8], axis=0), centers


def combine_panels(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, int]:
    divider = np.full((left.shape[0], 16, 3), 45, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1), left.shape[1] + divider.shape[1]


def draw_solution_edges(
    clean_sheet: np.ndarray,
    right_offset: int,
    source_centers: dict[str, tuple[int, int]],
    target_centers: dict[str, tuple[int, int]],
    solution_edges: list[dict[str, str]],
) -> np.ndarray:
    canvas = clean_sheet.copy()
    overlay = canvas.copy()
    for edge in solution_edges:
        source = source_centers.get(edge["source"])
        target = target_centers.get(edge["target"])
        if source is None or target is None:
            continue
        target_abs = (target[0] + right_offset, target[1])
        color = (230, 80, 230) if edge["semantic"] == "division" else (20, 220, 255)
        cv2.arrowedLine(overlay, source, target_abs, color, 3, cv2.LINE_AA, tipLength=0.025)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    draw_text(canvas, "HOCT selected links", (12, canvas.shape[0] - 14), 0.52, (250, 250, 250))
    return canvas


def display_edge(pair: tuple[int, int], semantic: str, source_display: dict[int, str], target_display: dict[int, str]) -> dict[str, str]:
    return {
        "source": source_display[pair[0]],
        "target": target_display[pair[1]],
        "semantic": semantic,
    }


def candidate_map_text(candidate_map: dict[str, list[str]]) -> str:
    return "\n".join(f"- {source}: {', '.join(targets)}" for source, targets in sorted(candidate_map.items()))


def linking_prompt(candidate_map: dict[str, list[str]]) -> str:
    return (
        "Two consecutive microscopy frames are shown. Frame t contains exactly five labeled source cells "
        "(S1-S5). Frame t+1 contains the union of their HOCT candidate target cells (T labels).\n\n"
        "Candidate targets allowed for each source:\n"
        f"{candidate_map_text(candidate_map)}\n\n"
        "A continuation joins a source to the same track in the next frame. A division edge joins an annotated "
        "parent to each visible daughter track; adjacent-frame annotations can contain one or more visible "
        "daughters, although two is the usual case.\n\n"
        "Return the complete ground-truth linking for all five sources. A source may disappear (zero targets), "
        "continue (one target), or branch to one or more visible daughter targets. Use semantic=division for "
        "every daughter edge and do not assume there are exactly two. Do not omit links simply because they are "
        "not the most obvious error in the image.\n\n"
        "Return only JSON:\n"
        '{"links": [{"source": "S1", "target": "T2", "semantic": "continuation"}], '
        '"unlinked_sources": ["S4"], "confidence": 0.0, "reason": "short visual justification"}'
    )


def repair_prompt(candidate_map: dict[str, list[str]], solution_edges: list[dict[str, str]]) -> str:
    return (
        "The image shows the HOCT-selected links between five source cells at frame t and their candidate targets "
        "at frame t+1.\n\n"
        "Candidate targets allowed for each source:\n"
        f"{candidate_map_text(candidate_map)}\n\n"
        "A continuation joins a source to the same track in the next frame. A division edge joins an annotated "
        "parent to each visible daughter track; adjacent-frame annotations can contain one or more visible "
        "daughters, although two is the usual case.\n\n"
        "HOCT currently selected these links:\n"
        f"{json.dumps(solution_edges, sort_keys=True)}\n\n"
        "Return every operation required to transform this local HOCT graph into the complete ground-truth graph. "
        "Valid operations are add, delete, and change_semantic. The graph may contain multiple errors; repair all "
        "of them.\n\n"
        "Return only JSON:\n"
        '{"operations": [{"op": "add", "source": "S1", "target": "T2", "semantic": "continuation"}, '
        '{"op": "delete", "source": "S3", "target": "T4", "semantic": "continuation"}, '
        '{"op": "change_semantic", "source": "S5", "target": "T7", "from_semantic": "continuation", '
        '"to_semantic": "division"}], "confidence": 0.0, "reason": "short visual justification"}'
    )


def inspection_prompt(candidate_map: dict[str, list[str]], solution_edges: list[dict[str, str]]) -> str:
    return (
        "The image shows the HOCT-selected links between five source cells at frame t and their candidate targets "
        "at frame t+1.\n\n"
        "Candidate targets allowed for each source:\n"
        f"{candidate_map_text(candidate_map)}\n\n"
        "A continuation joins a source to the same track in the next frame. A division edge joins an annotated "
        "parent to each visible daughter track; adjacent-frame annotations can contain one or more visible "
        "daughters, although two is the usual case.\n\n"
        "HOCT currently selected these links:\n"
        f"{json.dumps(solution_edges, sort_keys=True)}\n\n"
        "Select every source-target relationship that should be inspected by a human. Include selected links that "
        "are spurious or have the wrong semantic type, and unselected candidate links that should exist. The graph "
        "may contain multiple errors.\n\n"
        "Return only JSON:\n"
        '{"links_to_inspect": [{"source": "S1", "target": "T2", "hoct_status": "unselected", '
        '"issue": "missing"}, {"source": "S3", "target": "T4", "hoct_status": "selected", '
        '"issue": "spurious"}, {"source": "S5", "target": "T7", "hoct_status": "selected", '
        '"issue": "wrong_semantic"}], "confidence": 0.0, "reason": "short visual justification"}'
    )


def build_sample(
    anchor: dict[str, Any],
    node_by_key: dict[tuple[int, int], GraphNode],
    edges_by_frame: dict[int, list[GraphEdge]],
    gt_edges_by_frame: dict[int, dict[tuple[int, int], str]],
    local_error_rows: list[dict[str, str]],
    image_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    dataset, sequence = anchor["dataset"], anchor["sequence"]
    frame = parse_int(anchor["frame_from"])
    frame_to = parse_int(anchor["frame_to"])
    anchor_source = parse_int(anchor["gt_label_from"])
    anchor_target = parse_int(anchor["gt_label_to"])
    anchor_type = anchor["type"]
    selection = selected_anchor_metadata(anchor)
    is_selected_anchor = selection is not None
    if not is_selected_anchor and anchor_type not in {"EA", "EC"}:
        return None, "proxy_has_no_gt_edge"
    if is_selected_anchor and anchor_type not in ERROR_TO_OPERATION:
        return None, "selected_anchor_has_invalid_error_type"
    if frame_to != frame + 1:
        return None, "proxy_not_adjacent"
    if not parse_bool(anchor.get("candidate_present")):
        return None, "proxy_not_in_candidate_graph"
    if frame not in image_paths or frame_to not in image_paths or frame not in mask_paths or frame_to not in mask_paths:
        return None, "missing_frame_files"

    transition_edges = edges_by_frame.get(frame, [])
    by_source: dict[int, list[GraphEdge]] = defaultdict(list)
    for edge in transition_edges:
        by_source[edge.source_label].append(edge)
    candidate_pairs = {(edge.source_label, edge.target_label) for edge in transition_edges}
    anchor_pair = (anchor_source, anchor_target)
    if anchor_pair not in candidate_pairs:
        return None, "proxy_candidate_edge_unmapped"

    gt_pairs = gt_edges_by_frame.get(frame, {})
    anchor_is_gt = (
        selection["is_ground_truth_link"] if selection is not None else anchor_type in {"EA", "EC"}
    )
    if anchor_is_gt and anchor_pair not in gt_pairs:
        return None, "proxy_gt_edge_missing"
    if not anchor_is_gt and anchor_pair in gt_pairs:
        return None, "proxy_non_gt_edge_is_ground_truth"
    eligible_sources: list[int] = []
    for source, edges in by_source.items():
        if (frame, source) not in node_by_key:
            continue
        candidate_targets = {edge.target_label for edge in edges}
        required_targets = {target for (src, target), _semantic in gt_pairs.items() if src == source}
        if required_targets.issubset(candidate_targets):
            eligible_sources.append(source)

    anchor_key = "|".join(str(part) for part in canonical_anchor(anchor))
    if selection is not None:
        anchor_key += f"|{selection['data_split']}|rank={selection['selection_rank']}"
    rng = random.Random(deterministic_seed(args.seed, anchor_key))
    sampled = sample_sources(
        anchor_source,
        eligible_sources,
        node_by_key,
        frame,
        rng,
        args.source_count,
        args.anchor_center_probability,
        args.neighbor_pool,
    )
    if sampled is None:
        return None, "fewer_than_five_solvable_sources"
    source_labels, sampling_center, center_mode, center_neighbor_rank = sampled
    compactness = source_compactness(
        source_labels,
        sampling_center,
        node_by_key,
        frame,
    )
    compactness_limits = {
        "max_radius_from_sampling_center_px": args.crop_size * args.max_source_radius_fraction,
        "max_pairwise_distance_px": args.crop_size * args.max_source_diameter_fraction,
        "normalized_max_pairwise_distance": args.max_normalized_source_diameter,
    }
    compactness_exemptions = (
        {"max_radius_from_sampling_center_px", "max_pairwise_distance_px"}
        if dataset in args.pixel_compactness_exempt_datasets
        else set()
    )
    if any(
        compactness[key] > limit
        for key, limit in compactness_limits.items()
        if key not in compactness_exemptions
    ):
        return None, "source_group_not_compact"
    source_set = set(source_labels)
    target_set = {
        edge.target_label
        for source in source_labels
        for edge in by_source[source]
    }
    if not target_set:
        return None, "empty_target_pool"
    local_gt = {
        pair: semantic
        for pair, semantic in gt_pairs.items()
        if pair[0] in source_set
    }
    if any(pair[1] not in target_set for pair in local_gt):
        return None, "unsatisfied_gt_target"
    local_candidates = [
        edge
        for edge in transition_edges
        if edge.source_label in source_set and edge.target_label in target_set
    ]
    solution_candidates = [edge for edge in local_candidates if edge.solution]
    solution_outdegree = Counter(edge.source_label for edge in solution_candidates)

    errors_by_pair: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for error in local_error_rows:
        pair = (parse_int(error["gt_label_from"]), parse_int(error["gt_label_to"]))
        if pair[0] in source_set and pair[1] in target_set and parse_int(error["frame_to"]) == frame_to:
            errors_by_pair[pair].append(error)

    local_solution: dict[tuple[int, int], str] = {}
    for edge in solution_candidates:
        pair = (edge.source_label, edge.target_label)
        semantic = None
        for error in errors_by_pair.get(pair, []):
            if error["type"] == "EC":
                semantic = error.get("pred_semantic") or "continuation"
            elif error["type"] == "ED":
                semantic = error.get("semantic") or "continuation"
        if semantic is None and pair in local_gt:
            semantic = local_gt[pair]
        if semantic is None:
            semantic = "division" if solution_outdegree[edge.source_label] > 1 else "continuation"
        local_solution[pair] = semantic

    operations: list[dict[str, Any]] = []
    for pair in sorted(set(local_gt) - set(local_solution)):
        operations.append({"op": "add", "pair": pair, "semantic": local_gt[pair]})
    for pair in sorted(set(local_solution) - set(local_gt)):
        operations.append({"op": "delete", "pair": pair, "semantic": local_solution[pair]})
    for pair in sorted(set(local_solution) & set(local_gt)):
        if local_solution[pair] != local_gt[pair]:
            operations.append(
                {
                    "op": "change_semantic",
                    "pair": pair,
                    "from_semantic": local_solution[pair],
                    "to_semantic": local_gt[pair],
                }
            )
    if not operations:
        return None, "no_local_errors"
    anchor_operation = next(
        (operation for operation in operations if operation["pair"] == anchor_pair),
        None,
    )
    anchor_is_hoct_error = (
        selection["trackers"]["hoct"]["error"] if selection is not None else True
    )
    if anchor_is_hoct_error:
        hoct_anchor_type = (
            selection["trackers"]["hoct"]["error_type"] if selection is not None else anchor_type
        )
        if hoct_anchor_type not in ERROR_TO_OPERATION:
            return None, "hoct_anchor_error_type_missing"
        expected_anchor_op = ERROR_TO_OPERATION[hoct_anchor_type]
        if anchor_operation is None or anchor_operation["op"] != expected_anchor_op:
            return None, "proxy_error_not_reproduced"
    elif anchor_operation is not None:
        return None, "hoct_anchor_error_flag_mismatch"

    source_order = list(source_labels)
    target_order = sorted(target_set)
    rng.shuffle(source_order)
    rng.shuffle(target_order)
    source_display = {label: f"S{index + 1}" for index, label in enumerate(source_order)}
    target_display = {label: f"T{index + 1}" for index, label in enumerate(target_order)}
    candidate_map = {
        source_display[source]: sorted(
            {target_display[edge.target_label] for edge in by_source[source] if edge.target_label in target_set},
            key=lambda value: int(value[1:]),
        )
        for source in source_order
    }
    gt_display_edges = [
        display_edge(pair, semantic, source_display, target_display)
        for pair, semantic in sorted(local_gt.items())
    ]
    solution_display_edges = [
        display_edge(pair, semantic, source_display, target_display)
        for pair, semantic in sorted(local_solution.items())
    ]
    display_operations: list[dict[str, str]] = []
    inspection: list[dict[str, str]] = []
    for operation in operations:
        source = source_display[operation["pair"][0]]
        target = target_display[operation["pair"][1]]
        if operation["op"] == "add":
            display_operations.append(
                {"op": "add", "source": source, "target": target, "semantic": operation["semantic"]}
            )
            inspection.append(
                {"source": source, "target": target, "hoct_status": "unselected", "issue": "missing"}
            )
        elif operation["op"] == "delete":
            display_operations.append(
                {"op": "delete", "source": source, "target": target, "semantic": operation["semantic"]}
            )
            inspection.append(
                {"source": source, "target": target, "hoct_status": "selected", "issue": "spurious"}
            )
        else:
            display_operations.append(
                {
                    "op": "change_semantic",
                    "source": source,
                    "target": target,
                    "from_semantic": operation["from_semantic"],
                    "to_semantic": operation["to_semantic"],
                }
            )
            inspection.append(
                {"source": source, "target": target, "hoct_status": "selected", "issue": "wrong_semantic"}
            )

    item_hash = hashlib.sha256(
        f"{anchor_key}|{','.join(str(label) for label in sorted(source_labels))}".encode("utf-8")
    ).hexdigest()[:14]
    item_id = f"{dataset}_{sequence}_f{frame:04d}_{item_hash}"
    center_node = node_by_key[(frame, sampling_center)]
    anchor_node = node_by_key[(frame, anchor_source)]
    sample = {
        "item_id": item_id,
        "dataset": dataset,
        "sequence": sequence,
        "frame": frame,
        "frame_to": frame_to,
        "graph_path": anchor["full_graph_path"],
        "proxy": {
            "type": anchor_type,
            "source_label": anchor_source,
            "target_label": anchor_target,
            "difficulty": parse_float(anchor.get("difficulty")),
            "strata": sorted(anchor["proxy_strata"]),
            "candidate_present": parse_bool(anchor.get("candidate_present")),
            "is_ground_truth_link": anchor_is_gt,
            "hoct_error": anchor_is_hoct_error,
            "operation_reproduced": anchor_operation is not None,
        },
        "source_labels": source_order,
        "target_labels": target_order,
        "source_display": source_display,
        "target_display": target_display,
        "candidate_map": candidate_map,
        "candidate_edges": [
            {
                "source_label": edge.source_label,
                "target_label": edge.target_label,
                "source": source_display[edge.source_label],
                "target": target_display[edge.target_label],
                "similarity": round(edge.similarity, 7),
                "hoct_selected": edge.solution,
            }
            for edge in sorted(local_candidates, key=lambda edge: (edge.source_label, edge.target_label))
        ],
        "gt_edges": gt_display_edges,
        "hoct_edges": solution_display_edges,
        "operations": display_operations,
        "inspection": inspection,
        "error_counts": dict(Counter(operation["op"] for operation in display_operations)),
        "local_error_count": len(display_operations),
        "incidental_error_count": len(display_operations) - int(anchor_operation is not None),
        "sampling_center_label": sampling_center,
        "center_mode": center_mode,
        "sampling_center_neighbor_rank": center_neighbor_rank,
        "sampling_neighbor_pool": args.neighbor_pool,
        "source_compactness": compactness,
        "source_compactness_limits": {
            key: round(value, 7) for key, value in compactness_limits.items()
        },
        "source_compactness_exemptions": sorted(compactness_exemptions),
        "sampling_center_yx": [center_node.y, center_node.x],
        "proxy_source_yx": [anchor_node.y, anchor_node.x],
    }
    if selection is not None:
        sample["data_split"] = selection["data_split"]
        sample["selection"] = selection
    return sample, None


def materialize_sample(
    sample: dict[str, Any],
    image_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    item_dir = output_dir / "items" / sample["item_id"]
    prompts_dir = item_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    frame = sample["frame"]
    frame_to = sample["frame_to"]
    image_t, image_t1 = load_image(image_paths[frame]), load_image(image_paths[frame_to])
    mask_t, mask_t1 = load_mask(mask_paths[frame]), load_mask(mask_paths[frame_to])
    crop = crop_for_sample(
        mask_t,
        mask_t1,
        set(sample["source_labels"]),
        set(sample["target_labels"]),
        args.crop_margin,
    )
    y0, x0, y1, x1 = crop
    source_labels_present = set(np.unique(mask_t[y0:y1, x0:x1]))
    target_labels_present = set(np.unique(mask_t1[y0:y1, x0:x1]))
    missing_sources = sorted(set(sample["source_labels"]) - source_labels_present)
    missing_targets = sorted(set(sample["target_labels"]) - target_labels_present)
    if missing_sources or missing_targets:
        raise ValueError(
            f"Candidate-tight crop for {sample['item_id']} misses "
            f"sources={missing_sources}, targets={missing_targets}"
        )
    source_map = {int(label): display for label, display in sample["source_display"].items()}
    target_map = {int(label): display for label, display in sample["target_display"].items()}
    frame_t_path = item_dir / "frame_t.jpg"
    frame_t1_path = item_dir / "frame_t1.jpg"
    linking_view_path = item_dir / "cell_linking.jpg"
    solution_view_path = item_dir / "hoct_solution.jpg"
    if not args.skip_render:
        panel_t, source_centers = render_frame(
            image_t, mask_t, crop, source_map, "source", frame, args.panel_size
        )
        panel_t1, target_centers = render_frame(
            image_t1, mask_t1, crop, target_map, "target", frame_to, args.panel_size
        )
        clean_sheet, right_offset = combine_panels(panel_t, panel_t1)
        solution_sheet = draw_solution_edges(
            clean_sheet,
            right_offset,
            source_centers,
            target_centers,
            sample["hoct_edges"],
        )
        expected_source_displays = set(source_map.values())
        expected_target_displays = set(target_map.values())
        if set(source_centers) != expected_source_displays or set(target_centers) != expected_target_displays:
            raise ValueError(
                f"Panel rendering dropped labels for {sample['item_id']}: "
                f"sources={sorted(expected_source_displays - set(source_centers))}, "
                f"targets={sorted(expected_target_displays - set(target_centers))}"
            )
        cv2.imwrite(str(frame_t_path), panel_t, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        cv2.imwrite(str(frame_t1_path), panel_t1, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        cv2.imwrite(str(linking_view_path), clean_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        cv2.imwrite(str(solution_view_path), solution_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

    prompt_paths = {
        "cell_linking": prompts_dir / "cell_linking.md",
        "track_repair": prompts_dir / "track_repair.md",
        "track_inspection": prompts_dir / "track_inspection.md",
    }
    prompt_paths["cell_linking"].write_text(linking_prompt(sample["candidate_map"]) + "\n", encoding="utf-8")
    prompt_paths["track_repair"].write_text(
        repair_prompt(sample["candidate_map"], sample["hoct_edges"]) + "\n", encoding="utf-8"
    )
    prompt_paths["track_inspection"].write_text(
        inspection_prompt(sample["candidate_map"], sample["hoct_edges"]) + "\n", encoding="utf-8"
    )
    proxy_y, proxy_x = sample["proxy_source_yx"]
    crop_center_y, crop_center_x = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    sample["crop_yxyx"] = list(crop)
    sample["crop_mode"] = "tight_source_candidate_union"
    sample["crop_shape_yx"] = [y1 - y0, x1 - x0]
    sample["crop_coverage"] = {
        "source_labels": len(sample["source_labels"]),
        "candidate_target_labels": len(sample["target_labels"]),
        "all_sources_present": not missing_sources,
        "all_candidate_targets_present": not missing_targets,
    }
    sample["crop_area_fraction"] = round(
        ((y1 - y0) * (x1 - x0)) / max(1, mask_t.shape[0] * mask_t.shape[1]),
        7,
    )
    sample["proxy_offset_from_crop_center"] = [
        round((proxy_y - crop_center_y) / max(1.0, y1 - y0), 7),
        round((proxy_x - crop_center_x) / max(1.0, x1 - x0), 7),
    ]
    sample["paths"] = {
        "frame_t": repo_or_abs(frame_t_path),
        "frame_t1": repo_or_abs(frame_t1_path),
        "cell_linking": repo_or_abs(linking_view_path),
        "hoct_solution": repo_or_abs(solution_view_path),
        "prompts": {part: repo_or_abs(path) for part, path in prompt_paths.items()},
    }
    write_json(item_dir / "metadata_private.json", sample)
    return sample


def task_and_answer(sample: dict[str, Any], part: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = f"{part}::{sample['item_id']}"
    image_path = sample["paths"]["cell_linking" if part == "cell_linking" else "hoct_solution"]
    task = {
        "task_id": task_id,
        "item_id": sample["item_id"],
        "dataset": sample["dataset"],
        "sequence": sample["sequence"],
        "part": part,
        "prompt_path": sample["paths"]["prompts"][part],
        "image_paths": [image_path],
        "contact_sheet_path": image_path,
        "sequence_frames": [sample["frame"], sample["frame_to"]],
        "source_count": len(sample["source_labels"]),
        "target_count": len(sample["target_labels"]),
        "candidate_edge_count": len(sample["candidate_edges"]),
        "proxy_strata": sample["proxy"]["strata"],
        "proxy_difficulty": sample["proxy"]["difficulty"],
        "local_error_count": sample["local_error_count"],
    }
    if "selection" in sample:
        task.update(
            data_split=sample["data_split"],
            tracker_error_count=sample["selection"]["tracker_error_count"],
            selection_rank=sample["selection"]["selection_rank"],
        )
    if part == "cell_linking":
        linked_sources = {edge["source"] for edge in sample["gt_edges"]}
        all_sources = sorted(sample["source_display"].values(), key=lambda value: int(value[1:]))
        answer_payload = {
            "links": sample["gt_edges"],
            "unlinked_sources": [source for source in all_sources if source not in linked_sources],
        }
    elif part == "track_repair":
        answer_payload = {"operations": sample["operations"]}
    else:
        answer_payload = {"links_to_inspect": sample["inspection"]}
    answer = {
        "task_id": task_id,
        "item_id": sample["item_id"],
        "part": part,
        "answer": answer_payload,
    }
    return task, answer


def skip_record(anchor: dict[str, Any], reason: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {"anchor": canonical_anchor(anchor), "reason": reason}
    selection = selected_anchor_metadata(anchor)
    if selection is not None:
        record.update(
            data_split=selection["data_split"],
            selection_rank=selection["selection_rank"],
            tracker_error_count=selection["tracker_error_count"],
            hardness=selection["hardness"],
            tracker_errors=selection["trackers"],
        )
    return record


def shared_manifest_item(sample: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "item_id": sample["item_id"],
        "dataset": sample["dataset"],
        "sequence": sample["sequence"],
        "frames": [sample["frame"], sample["frame_to"]],
        "proxy_strata": sample["proxy"]["strata"],
        "proxy_difficulty": sample["proxy"]["difficulty"],
        "local_error_count": sample["local_error_count"],
        "paths": sample["paths"],
    }
    if "selection" in sample:
        selection = sample["selection"]
        record.update(
            data_split=selection["data_split"],
            tracker_error_count=selection["tracker_error_count"],
            hardness=selection["hardness"],
            selection_rank=selection["selection_rank"],
            is_ground_truth_link=selection["is_ground_truth_link"],
            tracker_errors=selection["trackers"],
        )
    return record


def main() -> None:
    args = parse_args()
    args.pixel_compactness_exempt_datasets = {
        value.strip()
        for value in args.pixel_compactness_exempt_datasets.split(",")
        if value.strip()
    }
    if args.source_count != 5:
        raise ValueError("This benchmark specification requires exactly five source cells.")
    if not (0.0 <= args.anchor_center_probability <= 1.0):
        raise ValueError("--anchor-center-probability must be in [0, 1].")
    if args.neighbor_pool < 1:
        raise ValueError("--neighbor-pool must be at least 1.")
    if not (0.0 < args.max_source_radius_fraction <= 1.0):
        raise ValueError("--max-source-radius-fraction must be in (0, 1].")
    if not (0.0 < args.max_source_diameter_fraction <= 1.0):
        raise ValueError("--max-source-diameter-fraction must be in (0, 1].")
    if args.max_normalized_source_diameter <= 0.0:
        raise ValueError("--max-normalized-source-diameter must be positive.")
    if args.crop_margin < 0:
        raise ValueError("--crop-margin cannot be negative.")
    if args.selected_anchors is not None and args.data_split is None:
        raise ValueError("--data-split is required with --selected-anchors.")
    if args.selected_anchors is None and args.data_split is not None:
        raise ValueError("--data-split is only valid with --selected-anchors.")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    elif args.selected_anchors is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Selected-anchor builds require an empty --output-dir or --overwrite to prevent split mixing."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "items").mkdir(parents=True, exist_ok=True)

    allowlist = None
    if args.datasets:
        allowlist = {part.strip() for part in args.datasets.split(",") if part.strip()}
    if args.selected_anchors is not None:
        anchors = load_selected_anchors(
            args.selected_anchors,
            args.data_split,
            args.full_graphs_dir,
        )
        anchor_mode = "selected_consensus_first"
    else:
        anchors = load_anchors(args.hard_anchors, args.easy_anchors)
        anchor_mode = "legacy_hard_easy"
    if allowlist is not None:
        anchors = [anchor for anchor in anchors if anchor["dataset"] in allowlist]
    if not anchors:
        raise ValueError("No anchors remain after applying input and dataset filters.")
    if args.selected_anchors is not None and {
        anchor["data_split"] for anchor in anchors
    } != {args.data_split}:
        raise ValueError("Selected-anchor input mixed sequence splits after filtering.")
    error_index = load_error_index(args.all_errors)
    anchors_by_graph: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        anchors_by_graph[(anchor["dataset"], anchor["sequence"], anchor["full_graph_path"])].append(anchor)

    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_sample_signatures: set[tuple[Any, ...]] = set()
    stop = False
    for (dataset, sequence, graph_path_raw), graph_anchors in sorted(anchors_by_graph.items()):
        if stop:
            break
        dataset_dir = args.datasets_dir / dataset
        image_paths = frame_path_map(dataset_dir / sequence, "*.tif")
        mask_paths = frame_path_map(dataset_dir / f"{sequence}_GT" / "TRA", "man_track*.tif")
        tracks = read_tracks(dataset_dir / f"{sequence}_GT" / "TRA" / "man_track.txt")
        transitions = {parse_int(anchor["frame_from"]) for anchor in graph_anchors}
        frames = transitions | {frame + 1 for frame in transitions}
        missing_frames = sorted(frame for frame in frames if frame not in mask_paths)
        if missing_frames:
            for anchor in graph_anchors:
                skipped.append(skip_record(anchor, "missing_mask_frames"))
            continue
        graph_path = Path(graph_path_raw)
        if not graph_path.is_absolute():
            graph_path = REPO_ROOT / graph_path
        print(f"Loading {dataset}/{sequence}: {len(graph_anchors)} anchors", flush=True)
        node_by_key, edges_by_frame = load_graph_transition_data(graph_path, frames, mask_paths)
        gt_by_frame = gt_edges_by_transition(tracks, transitions)
        for anchor in graph_anchors:
            if args.max_items is not None and len(samples) >= args.max_items:
                stop = True
                break
            frame = parse_int(anchor["frame_from"])
            sample, reason = build_sample(
                anchor,
                node_by_key,
                edges_by_frame,
                gt_by_frame,
                error_index.get((dataset, sequence, frame), []),
                image_paths,
                mask_paths,
                args,
            )
            if sample is None:
                skipped.append(skip_record(anchor, reason))
                continue
            signature = (
                dataset,
                sequence,
                frame,
                tuple(sorted(sample["source_labels"])),
                tuple(sorted(sample["target_labels"])),
            )
            if signature in seen_sample_signatures:
                skipped.append(skip_record(anchor, "duplicate_induced_graph"))
                continue
            seen_sample_signatures.add(signature)
            samples.append(materialize_sample(sample, image_paths, mask_paths, args.output_dir, args))

    if args.selected_anchors is not None:
        retained_splits = {sample["data_split"] for sample in samples}
        if retained_splits and retained_splits != {args.data_split}:
            raise ValueError(f"Built samples mix data splits: {sorted(retained_splits)}")

    tasks_by_part: dict[str, list[dict[str, Any]]] = {part: [] for part in PARTS}
    answers_by_part: dict[str, list[dict[str, Any]]] = {part: [] for part in PARTS}
    for sample in samples:
        for part in PARTS:
            task, answer = task_and_answer(sample, part)
            tasks_by_part[part].append(task)
            answers_by_part[part].append(answer)
    for part in PARTS:
        part_dir = args.output_dir / part
        write_jsonl(part_dir / "tasks.jsonl", tasks_by_part[part])
        write_jsonl(part_dir / "answer_key.jsonl", answers_by_part[part])
        write_json(part_dir / "items.json", tasks_by_part[part])

    center_counts = Counter(sample["center_mode"] for sample in samples)
    normalized_spreads = [
        sample["source_compactness"]["normalized_max_pairwise_distance"] for sample in samples
    ]
    crop_area_fractions = [sample["crop_area_fraction"] for sample in samples]
    local_error_counts = Counter()
    for sample in samples:
        for operation, count in sample["error_counts"].items():
            local_error_counts[operation] += count
    summary = {
        "benchmarks": list(PARTS),
        "samples": len(samples),
        "tasks_per_benchmark": {part: len(tasks_by_part[part]) for part in PARTS},
        "input_anchor_rows": len(anchors),
        "skipped_anchors": len(skipped),
        "anchor_mode": anchor_mode,
        "data_split": args.data_split,
        "anchor_source": (
            repo_or_abs(args.selected_anchors)
            if args.selected_anchors is not None
            else {
                "hard": repo_or_abs(args.hard_anchors),
                "easy": repo_or_abs(args.easy_anchors),
            }
        ),
        "seed": args.seed,
        "source_count": args.source_count,
        "anchor_center_probability": args.anchor_center_probability,
        "neighbor_pool": args.neighbor_pool,
        "max_source_radius_fraction": args.max_source_radius_fraction,
        "max_source_diameter_fraction": args.max_source_diameter_fraction,
        "max_normalized_source_diameter": args.max_normalized_source_diameter,
        "pixel_compactness_exempt_datasets": sorted(args.pixel_compactness_exempt_datasets),
        "source_sampling_policy": (
            "Choose the proxy as center with the configured probability; otherwise choose from its closest "
            "eligible neighbors, then greedily minimize maximum pairwise source distance."
        ),
        "center_mode_counts": dict(center_counts),
        "realized_anchor_center_fraction": (
            center_counts["proxy_anchor"] / len(samples) if samples else None
        ),
        "samples_with_multiple_errors": sum(sample["local_error_count"] > 1 for sample in samples),
        "max_local_errors": max((sample["local_error_count"] for sample in samples), default=0),
        "local_error_operation_counts": dict(local_error_counts),
        "source_compactness": {
            "normalized_max_pairwise_median": (
                float(np.median(normalized_spreads)) if normalized_spreads else None
            ),
            "normalized_max_pairwise_p90": (
                float(np.quantile(normalized_spreads, 0.90)) if normalized_spreads else None
            ),
            "normalized_max_pairwise_p95": (
                float(np.quantile(normalized_spreads, 0.95)) if normalized_spreads else None
            ),
            "normalized_max_pairwise_max": max(normalized_spreads, default=None),
            "samples_over_5_nearest_neighbor_units": sum(value > 5.0 for value in normalized_spreads),
            "samples_over_10_nearest_neighbor_units": sum(value > 10.0 for value in normalized_spreads),
        },
        "candidate_tight_crops": {
            "margin_px": args.crop_margin,
            "area_fraction_median": (
                float(np.median(crop_area_fractions)) if crop_area_fractions else None
            ),
            "area_fraction_p90": (
                float(np.quantile(crop_area_fractions, 0.90)) if crop_area_fractions else None
            ),
            "area_fraction_max": max(crop_area_fractions, default=None),
        },
        "proxy_strata_counts": {
            stratum: sum(stratum in sample["proxy"]["strata"] for sample in samples)
            for stratum in ("hard", "easy", "consensus_first")
        },
        "retained_split_counts": dict(Counter(sample.get("data_split", "unsplit") for sample in samples)),
        "retained_anchor_gt_membership": dict(
            Counter(str(sample["proxy"]["is_ground_truth_link"]) for sample in samples)
        ),
        "retained_anchor_hoct_error": dict(
            Counter(str(sample["proxy"]["hoct_error"]) for sample in samples)
        ),
        "retained_tracker_error_counts": dict(
            Counter(
                sample["selection"]["tracker_error_count"]
                for sample in samples
                if "selection" in sample
            )
        ),
        "dataset_counts": dict(Counter(sample["dataset"] for sample in samples)),
        "skip_reason_counts": dict(Counter(row["reason"] for row in skipped)),
        "rendered": not args.skip_render,
    }
    write_json(
        args.output_dir / "shared_items.json",
        [shared_manifest_item(sample) for sample in samples],
    )
    write_json(args.output_dir / "skipped_anchors.json", skipped)
    write_json(args.output_dir / "build_summary.json", summary)
    (args.output_dir / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # HOCT Graph Benchmarks

            Three benchmarks share `{len(samples)}` induced two-frame graphs:

            - `cell_linking`: recover all GT links for five source cells.
            - `track_repair`: return all operations needed to repair the HOCT solution.
            - `track_inspection`: identify every selected or missing relationship needing review.

            Each benchmark directory contains `tasks.jsonl`, `answer_key.jsonl`, and `items.json`.
            Shared visual assets and private audit metadata live under `items/`.
            Source groups are compact neighborhoods: the proxy is the sampling center with probability
            `{args.anchor_center_probability:.2f}`; otherwise the center is drawn from its
            `{min(args.neighbor_pool, args.source_count - 1)}` nearest eligible neighbors, and remaining
            sources greedily minimize the group's maximum pairwise distance. Groups are retained only when
            their maximum center radius is at most `{args.max_source_radius_fraction:.2f}` of the crop size
            and their maximum pairwise diameter is at most `{args.max_source_diameter_fraction:.2f}`. The
            diameter must also be at most `{args.max_normalized_source_diameter:.1f}` local nearest-cell
            spacing units. Datasets listed in `{sorted(args.pixel_compactness_exempt_datasets)}` retain the
            normalized requirement but are exempt from pixel radius/diameter limits.

            Every rendered pair uses one candidate-tight crop containing all five source cells and the full
            union of their HOCT candidate targets, with `{args.crop_margin}` pixels of context where image
            boundaries permit. Rectangular boundary-limited crops are letterboxed rather than stretched.
            """
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
