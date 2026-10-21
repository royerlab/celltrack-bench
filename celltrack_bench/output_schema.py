"""Strict final-answer schema checks shared by smoke gating and paper scoring."""

from __future__ import annotations

import re
from typing import Any


TASK_FIELDS = {
    "cell_linking": "links",
    "track_repair": "operations",
    "track_inspection": "links_to_inspect",
}
SEMANTICS = {"continuation", "division"}
OPERATIONS = {"add", "delete", "change_semantic"}
STATUSES = {"selected", "unselected"}
ISSUES = {"missing", "spurious", "wrong_semantic"}


def _cell_id(value: Any, prefix: str) -> str | None:
    if not isinstance(value, str):
        return None
    canonical = value.strip().upper()
    return canonical if re.fullmatch(rf"{prefix}[1-9][0-9]*", canonical) else None


def _enum(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    canonical = value.strip().lower()
    return canonical if canonical in allowed else None


def _endpoint(
    row: dict[str, Any],
    index: int,
) -> tuple[str | None, str | None, list[str]]:
    errors: list[str] = []
    source = _cell_id(row.get("source"), "S")
    target = _cell_id(row.get("target"), "T")
    if source is None:
        errors.append(f"row {index}: invalid or missing source")
    if target is None:
        errors.append(f"row {index}: invalid or missing target")
    return source, target, errors


def validate_output(
    task: str, payload: dict[str, Any] | None, meta: dict[str, Any] | None = None
) -> list[str]:
    """Return schema errors without judging whether the proposed graph is correct."""
    if task not in TASK_FIELDS:
        return [f"unknown task {task!r}"]
    if not isinstance(payload, dict):
        return ["final answer is not a JSON object"]
    field = TASK_FIELDS[task]
    rows = payload.get(field)
    if not isinstance(rows, list):
        return [f"{field} must be a list"]

    errors: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    linked_sources: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: expected an object")
            continue
        source, target, endpoint_errors = _endpoint(row, index)
        errors.extend(endpoint_errors)

        if task == "cell_linking":
            semantic = _enum(row.get("semantic"), SEMANTICS)
            if semantic is None:
                errors.append(f"row {index}: semantic must be continuation or division")
            key = (source, target, semantic)
            if source is not None:
                linked_sources.add(source)
        elif task == "track_repair":
            operation = _enum(row.get("op"), OPERATIONS)
            if operation is None:
                errors.append(f"row {index}: invalid or missing op")
            if operation == "change_semantic":
                from_semantic = _enum(row.get("from_semantic"), SEMANTICS)
                to_semantic = _enum(row.get("to_semantic"), SEMANTICS)
                if from_semantic is None or to_semantic is None:
                    errors.append(
                        f"row {index}: change_semantic requires valid from_semantic and to_semantic"
                    )
                elif from_semantic == to_semantic:
                    errors.append(f"row {index}: semantic change has identical values")
                key = (operation, source, target, from_semantic, to_semantic)
            else:
                semantic = _enum(row.get("semantic"), SEMANTICS)
                if semantic is None:
                    errors.append(f"row {index}: {operation or 'operation'} requires semantic")
                key = (operation, source, target, semantic)
        else:
            status = _enum(row.get("hoct_status"), STATUSES)
            issue = _enum(row.get("issue"), ISSUES)
            if status is None:
                errors.append(f"row {index}: invalid or missing hoct_status")
            if issue is None:
                errors.append(f"row {index}: invalid or missing issue")
            if status == "unselected" and issue not in {None, "missing"}:
                errors.append(f"row {index}: unselected links must have issue=missing")
            if status == "selected" and issue == "missing":
                errors.append(f"row {index}: selected links cannot have issue=missing")
            key = (source, target, status, issue)

        if key in seen:
            errors.append(f"row {index}: duplicate decision")
        seen.add(key)

    if task == "cell_linking":
        raw_unlinked = payload.get("unlinked_sources")
        if not isinstance(raw_unlinked, list):
            errors.append("unlinked_sources must be a list")
        else:
            unlinked: set[str] = set()
            for index, value in enumerate(raw_unlinked):
                source = _cell_id(value, "S")
                if source is None:
                    errors.append(f"unlinked_sources[{index}]: invalid source")
                    continue
                if source in unlinked:
                    errors.append(f"unlinked_sources[{index}]: duplicate source {source}")
                unlinked.add(source)
            overlap = linked_sources & unlinked
            if overlap:
                errors.append(f"sources cannot be both linked and unlinked: {sorted(overlap)}")
    return errors
