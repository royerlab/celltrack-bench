#!/usr/bin/env python3
"""Merge disjoint condition shards into one provenance-consistent result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from celltrack_bench import benchmark_data as B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def config_path(results: Path) -> Path:
    return B.run_config_path(results)


def digest(value: Any, length: int = 12) -> str:
    import hashlib

    return hashlib.sha256(B.canonical_json(value)).hexdigest()[:length]


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output_config = config_path(output)
    if output.exists() or output_config.exists():
        raise SystemExit(f"refusing to overwrite merged output: {output}")

    configs = [B.read_json(config_path(path.expanduser().resolve())) for path in args.inputs]
    if not configs or any(not isinstance(config, dict) for config in configs):
        raise SystemExit("every input must have a result config")
    reference = configs[0]
    invariant_fields = ("schema_version", "benchmark", "response_config", "response_id", "tasks")
    for path, config in zip(args.inputs[1:], configs[1:]):
        mismatches = [field for field in invariant_fields if config.get(field) != reference.get(field)]
        if mismatches:
            raise SystemExit(f"incompatible shard {path}: fields differ: {mismatches}")

    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    conditions: set[int] = set()
    for path, config in zip(args.inputs, configs):
        declared = {int(value) for value in config.get("conditions", [])}
        conditions.update(declared)
        for row in B.read_jsonl(path.expanduser().resolve()):
            key = (str(row["item_id"]), str(row["task"]), int(row["condition"]))
            if key[2] not in declared:
                raise SystemExit(f"{path}: row condition c{key[2]} is not declared")
            if key in rows_by_key and rows_by_key[key] != row:
                raise SystemExit(f"conflicting duplicate result key: {key}")
            rows_by_key[key] = row

    merged_config = {field: reference[field] for field in invariant_fields}
    merged_config["conditions"] = sorted(conditions)
    merged_config["run_id"] = digest(merged_config)
    merged_run_id = merged_config["run_id"]
    rows = []
    for key in sorted(rows_by_key):
        row = dict(rows_by_key[key])
        row["source_run_id"] = row.get("run_id")
        row["run_id"] = merged_run_id
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    output_config.write_text(json.dumps(merged_config, indent=2, sort_keys=True) + "\n")
    print(
        f"merged {len(args.inputs)} shards, {len(rows)} rows, "
        f"conditions={sorted(conditions)} -> {output}"
    )


if __name__ == "__main__":
    main()
