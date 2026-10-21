"""Versioned benchmark manifests, identities, and deterministic item selection."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TASKS = ("cell_linking", "track_repair", "track_inspection")
MANIFEST_NAME = "benchmark_manifest.json"


MISSING_HINT = (
    "the benchmark files are generated locally from the raw datasets: "
    "see construction/README.md (RAW_DATA=... construction/rebuild_benchmark.sh)"
)


class BenchmarkDataError(ValueError):
    """Raised when benchmark files are incomplete or internally inconsistent."""


@dataclass(frozen=True)
class BenchmarkIdentity:
    label: str
    digest: str

    @property
    def value(self) -> str:
        return f"{self.label}-{self.digest}"


def safe_slug(value: str, *, fallback: str = "benchmark", max_length: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (slug or fallback)[:max_length]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def read_text(path: Path) -> str:
    """Read a text file, transparently decompressing ``*.gz``."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return handle.read()
    return path.read_text()


def open_text(path: Path):
    """Open a text file for line iteration, transparently decompressing ``*.gz``."""
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def run_config_path(results: Path) -> Path:
    """``run.jsonl`` and ``run.jsonl.gz`` both keep their config in ``run.jsonl.config.json``."""
    results = Path(results)
    if results.suffix == ".gz":
        results = results.with_suffix("")
    return results.with_suffix(results.suffix + ".config.json")


def read_json(path: Path) -> Any:
    try:
        return json.loads(read_text(path))
    except FileNotFoundError as exc:
        raise BenchmarkDataError(f"missing benchmark file: {path}; {MISSING_HINT}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkDataError(f"invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = read_text(path).splitlines()
    except FileNotFoundError as exc:
        raise BenchmarkDataError(f"missing benchmark file: {path}; {MISSING_HINT}") from exc
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkDataError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
        if not isinstance(row, dict):
            raise BenchmarkDataError(f"expected an object at {path}:{lineno}")
        rows.append(row)
    return rows


def load_split_items(split_dir: Path, split: str) -> list[dict[str, Any]]:
    path = Path(split_dir) / f"{split}_items.json"
    items = read_json(path)
    if not isinstance(items, list) or not items:
        raise BenchmarkDataError(f"{path} must contain a non-empty JSON list")
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise BenchmarkDataError(f"{path}: item {index} is not an object")
        missing = [key for key in ("item_id", "dataset", "sequence") if key not in item]
        if missing:
            raise BenchmarkDataError(f"{path}: item {index} is missing {missing}")
        item_id = str(item["item_id"])
        if item_id in seen:
            raise BenchmarkDataError(f"{path}: duplicate item_id {item_id!r}")
        seen.add(item_id)
    return items


def select_items(
    items: Iterable[dict[str, Any]], *, sample: int = 0, limit: int = 0
) -> list[dict[str, Any]]:
    selected = list(items)
    if sample and limit:
        raise BenchmarkDataError("--sample and --limit are mutually exclusive")
    if sample:
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for item in selected:
            by_dataset.setdefault(str(item["dataset"]), []).append(item)
        picked: list[dict[str, Any]] = []
        while len(picked) < sample and any(by_dataset.values()):
            for dataset in sorted(by_dataset):
                if by_dataset[dataset]:
                    picked.append(by_dataset[dataset].pop(0))
                    if len(picked) >= sample:
                        break
        return picked
    return selected[:limit] if limit else selected


def load_answers(
    split_dir: Path, split: str, tasks: Iterable[str] = TASKS
) -> dict[tuple[str, str], dict[str, Any]]:
    answers: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        path = Path(split_dir) / split / task / "answer_key.jsonl"
        for row in read_jsonl(path):
            if "item_id" not in row or "answer" not in row:
                raise BenchmarkDataError(f"{path}: every row needs item_id and answer")
            key = (str(row["item_id"]), task)
            if key in answers:
                raise BenchmarkDataError(f"{path}: duplicate answer for {key[0]!r}")
            answers[key] = row["answer"]
    return answers


def load_manifest(split_dir: Path) -> dict[str, Any]:
    invalidated = Path(split_dir) / "INVALIDATED.md"
    if invalidated.exists():
        raise BenchmarkDataError(
            f"benchmark split is explicitly invalidated: {invalidated}"
        )
    path = Path(split_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise BenchmarkDataError(f"{path} must contain a JSON object")
    return manifest


def resolve_items_root(
    split_dir: Path,
    split: str,
    explicit_root: Path | None,
    legacy_root: Path,
) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()
    manifest = load_manifest(split_dir)
    split_config = (manifest.get("splits") or {}).get(split) or {}
    configured = split_config.get("items_root")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(split_dir) / path
        return path.resolve()
    return Path(legacy_root).expanduser().resolve()


def metadata_path(items_root: Path, item_id: str) -> Path:
    root = Path(items_root)
    candidates = (
        root / item_id / "metadata_private.json",
        root / item_id / "metadata.json",
        root / "items" / item_id / "metadata_private.json",
        root / "items" / item_id / "metadata.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise BenchmarkDataError(
        f"metadata for {item_id!r} was not found under benchmark item root {root}"
    )


def load_item_metadata(items_root: Path, item_id: str) -> dict[str, Any]:
    path = metadata_path(items_root, item_id)
    meta = read_json(path)
    if not isinstance(meta, dict):
        raise BenchmarkDataError(f"{path} must contain a JSON object")
    actual_id = str(meta.get("item_id", item_id))
    if actual_id != item_id:
        raise BenchmarkDataError(
            f"{path}: metadata item_id {actual_id!r} does not match manifest {item_id!r}"
        )
    return meta


def validate_benchmark(
    split_dir: Path,
    items_root: Path,
    split: str,
    tasks: Iterable[str] = TASKS,
    *,
    check_metadata: bool = True,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    tasks = tuple(tasks)
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise BenchmarkDataError(f"unknown tasks: {unknown}")
    items = load_split_items(split_dir, split)
    answers = load_answers(split_dir, split, tasks)
    item_ids = {str(item["item_id"]) for item in items}
    for task in tasks:
        answer_ids = {item_id for item_id, answer_task in answers if answer_task == task}
        missing = sorted(item_ids - answer_ids)
        extra = sorted(answer_ids - item_ids)
        if missing or extra:
            raise BenchmarkDataError(
                f"{split}/{task} answer keys disagree with the item manifest: "
                f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
            )
    if check_metadata:
        for item in items:
            meta = load_item_metadata(items_root, str(item["item_id"]))
            for key in ("dataset", "sequence"):
                if str(meta.get(key)) != str(item[key]):
                    raise BenchmarkDataError(
                        f"{item['item_id']}: metadata {key}={meta.get(key)!r} "
                        f"does not match manifest {item[key]!r}"
                    )
    return items, answers


def benchmark_identity(
    split_dir: Path,
    items_root: Path,
    split: str,
    *,
    label: str | None = None,
) -> BenchmarkIdentity:
    """Hash every prompt- and score-defining benchmark object for this split."""
    items, answers = validate_benchmark(split_dir, items_root, split, TASKS)
    manifest = load_manifest(split_dir)
    chosen_label = label or manifest.get("benchmark_id") or Path(items_root).parent.name
    digest = hashlib.sha256()
    digest.update(b"experiment-harness-benchmark-v1\0")
    digest.update(split.encode())
    for item in items:
        item_id = str(item["item_id"])
        digest.update(canonical_json(item))
        digest.update(canonical_json(load_item_metadata(items_root, item_id)))
        for task in TASKS:
            digest.update(task.encode())
            digest.update(canonical_json(answers[(item_id, task)]))
    return BenchmarkIdentity(safe_slug(str(chosen_label)), digest.hexdigest()[:16])
