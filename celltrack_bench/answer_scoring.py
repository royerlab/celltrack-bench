"""Answer parsing and per-task set scoring shared by every CellTrack-Bench metric.

Vendored verbatim from the HOCT graph-benchmark evaluator (``evaluate_hoct_graph_benchmarks.py``),
keeping only the parsing and scoring functions; the local-model inference loop is not needed here.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any

PARTS = ("cell_linking", "track_repair", "track_inspection")
PREFERRED_FIELDS = {
    "cell_linking": "links",
    "track_repair": "operations",
    "track_inspection": "links_to_inspect",
}


def parse_json_candidate(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        pyish = (
            candidate.replace(": null", ": None")
            .replace(": true", ": True")
            .replace(": false", ": False")
        )
        value = ast.literal_eval(pyish)
        return value if isinstance(value, dict) else None
    except (SyntaxError, TypeError, ValueError):
        return None


def json_objects(text: str) -> list[tuple[dict[str, Any], str]]:
    objects: list[tuple[dict[str, Any], str]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    source = text[start : end + 1]
                    parsed = parse_json_candidate(source)
                    if parsed is not None:
                        objects.append((parsed, source))
                    break
    return objects


def select_output_object(text: str, part: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = json_objects(text)
    preferred = PREFERRED_FIELDS[part]
    for parsed, source in reversed(candidates):
        if preferred in parsed:
            return parsed, source
        answer = parsed.get("answer")
        if isinstance(answer, dict) and preferred in answer:
            return answer, source
    return candidates[-1] if candidates else (None, None)


def normalized_label(value: Any, prefix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NULL", "N/A", "NO LINK", "UNLINKED"}:
        return None
    if text.startswith(prefix):
        suffix = text[len(prefix) :].strip()
        if suffix.isdigit():
            return f"{prefix}{int(suffix)}"
    if text.isdigit():
        return f"{prefix}{int(text)}"
    return text


def normalized_semantic(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"continuation", "continue", "continued", "same_cell", "link"}:
        return "continuation"
    if text in {"division", "divide", "divided", "mitosis", "daughter"}:
        return "division"
    return text or None


def normalized_operation(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "insert": "add",
        "create": "add",
        "remove": "delete",
        "del": "delete",
        "change": "change_semantic",
        "relabel": "change_semantic",
        "change_type": "change_semantic",
    }
    return aliases.get(text, text or None)


def normalized_issue(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "missing_link": "missing",
        "false_negative": "missing",
        "extra": "spurious",
        "extra_link": "spurious",
        "false_positive": "spurious",
        "semantic": "wrong_semantic",
        "wrong_type": "wrong_semantic",
    }
    return aliases.get(text, text or None)


def normalized_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"selected", "present", "active"}:
        return "selected"
    if text in {"unselected", "missing", "absent", "inactive"}:
        return "unselected"
    return text or None


def set_counts(predicted: set[tuple[Any, ...]], target: set[tuple[Any, ...]]) -> tuple[int, int, int]:
    return len(predicted & target), len(predicted - target), len(target - predicted)


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == fp == fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def metric_block(
    predicted: set[tuple[Any, ...]],
    target: set[tuple[Any, ...]],
    prefix: str,
) -> dict[str, Any]:
    tp, fp, fn = set_counts(predicted, target)
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    return {
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def scalar_or_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def linking_sets(payload: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]], set[str], bool]:
    raw_links = payload.get("links")
    if raw_links is None and payload and all(str(key).upper().startswith("S") for key in payload):
        raw_links = payload
    records: list[tuple[str, str, str | None]] = []
    unlinked: set[str] = set()
    schema_valid = isinstance(raw_links, (dict, list))
    if isinstance(raw_links, dict):
        for source_raw, target_raw in raw_links.items():
            source = normalized_label(source_raw, "S")
            if source is None:
                continue
            semantic: str | None = None
            if isinstance(target_raw, dict):
                semantic = normalized_semantic(target_raw.get("semantic") or target_raw.get("type"))
                target_raw = target_raw.get("target") or target_raw.get("target_id")
            targets = scalar_or_list(target_raw)
            if not targets:
                unlinked.add(source)
            for target_value in targets:
                target = normalized_label(target_value, "T")
                if target is not None:
                    records.append((source, target, semantic or ("division" if len(targets) > 1 else None)))
    elif isinstance(raw_links, list):
        for row in raw_links:
            if not isinstance(row, dict):
                continue
            source = normalized_label(row.get("source") or row.get("source_id"), "S")
            target_value = row.get("target") if "target" in row else row.get("target_id")
            targets = scalar_or_list(target_value)
            if source is None:
                continue
            if not targets:
                unlinked.add(source)
            semantic = normalized_semantic(row.get("semantic") or row.get("type"))
            for target_raw in targets:
                target = normalized_label(target_raw, "T")
                if target is not None:
                    records.append((source, target, semantic or ("division" if len(targets) > 1 else None)))
    for source_raw in scalar_or_list(payload.get("unlinked_sources")):
        source = normalized_label(source_raw, "S")
        if source is not None:
            unlinked.add(source)
    source_counts = Counter(source for source, _target, _semantic in records)
    endpoints = {(source, target) for source, target, _semantic in records}
    semantic_edges = {
        (source, target, semantic or ("division" if source_counts[source] > 1 else "continuation"))
        for source, target, semantic in records
    }
    return endpoints, semantic_edges, unlinked, schema_valid


def repair_sets(payload: dict[str, Any]) -> tuple[set[tuple[str, str, str]], set[tuple[Any, ...]], bool]:
    rows = payload.get("operations")
    endpoint_actions: set[tuple[str, str, str]] = set()
    full_actions: set[tuple[Any, ...]] = set()
    schema_valid = isinstance(rows, list)
    if not isinstance(rows, list):
        return endpoint_actions, full_actions, schema_valid
    for row in rows:
        if not isinstance(row, dict):
            continue
        operation = normalized_operation(row.get("op") or row.get("operation") or row.get("action"))
        source = normalized_label(row.get("source") or row.get("source_id"), "S")
        target = normalized_label(row.get("target") or row.get("target_id"), "T")
        if operation not in {"add", "delete", "change_semantic"} or source is None or target is None:
            continue
        endpoint_actions.add((operation, source, target))
        if operation == "change_semantic":
            full_actions.add(
                (
                    operation,
                    source,
                    target,
                    normalized_semantic(row.get("from_semantic") or row.get("from")),
                    normalized_semantic(row.get("to_semantic") or row.get("to") or row.get("semantic")),
                )
            )
        else:
            full_actions.add((operation, source, target, normalized_semantic(row.get("semantic") or row.get("type"))))
    return endpoint_actions, full_actions, schema_valid


def inspection_sets(payload: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[Any, ...]], bool]:
    rows = payload.get("links_to_inspect")
    endpoints: set[tuple[str, str]] = set()
    full_rows: set[tuple[Any, ...]] = set()
    schema_valid = isinstance(rows, list)
    if not isinstance(rows, list):
        return endpoints, full_rows, schema_valid
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = normalized_label(row.get("source") or row.get("source_id"), "S")
        target = normalized_label(row.get("target") or row.get("target_id"), "T")
        if source is None or target is None:
            continue
        endpoints.add((source, target))
        full_rows.add(
            (
                source,
                target,
                normalized_status(row.get("hoct_status") or row.get("status")),
                normalized_issue(row.get("issue") or row.get("reason")),
            )
        )
    return endpoints, full_rows, schema_valid


def score_output(part: str, parsed: dict[str, Any] | None, answer: dict[str, Any]) -> dict[str, Any]:
    valid_json = parsed is not None and PREFERRED_FIELDS[part] in parsed
    payload = parsed or {}
    if part == "cell_linking":
        pred_endpoints, pred_full, pred_unlinked, schema_valid = linking_sets(payload)
        target_endpoints, target_full, target_unlinked, _ = linking_sets(answer)
        metrics = {
            **metric_block(pred_endpoints, target_endpoints, "endpoint"),
            **metric_block(pred_full, target_full, "primary"),
            "unlinked_exact": pred_unlinked == target_unlinked,
            "exact_match": pred_full == target_full and pred_unlinked == target_unlinked,
            "predicted_count": len(pred_full),
            "target_count": len(target_full),
        }
    elif part == "track_repair":
        pred_endpoints, pred_full, schema_valid = repair_sets(payload)
        target_endpoints, target_full, _ = repair_sets(answer)
        metrics = {
            **metric_block(pred_endpoints, target_endpoints, "endpoint"),
            **metric_block(pred_full, target_full, "primary"),
            "exact_match": pred_full == target_full,
            "predicted_count": len(pred_full),
            "target_count": len(target_full),
        }
    elif part == "track_inspection":
        pred_endpoints, pred_full, schema_valid = inspection_sets(payload)
        target_endpoints, target_full, _ = inspection_sets(answer)
        metrics = {
            **metric_block(pred_endpoints, target_endpoints, "endpoint"),
            **metric_block(pred_full, target_full, "primary"),
            "exact_match": pred_full == target_full,
            "predicted_count": len(pred_full),
            "target_count": len(target_full),
        }
    else:
        raise ValueError(part)
    metrics["valid_json"] = valid_json
    metrics["schema_valid"] = schema_valid
    return metrics
