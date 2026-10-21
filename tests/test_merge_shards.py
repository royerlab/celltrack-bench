from __future__ import annotations

import json
import sys
from pathlib import Path

from celltrack_bench import merge_shards as merge


def _write_shard(path: Path, condition: int) -> None:
    config = {
        "schema_version": 1,
        "benchmark": {"identity": "benchmark-v1"},
        "response_config": {"model": "model"},
        "response_id": "response",
        "tasks": ["cell_linking"],
        "conditions": [condition],
        "run_id": f"shard-{condition}",
    }
    row = {
        "item_id": "item",
        "task": "cell_linking",
        "condition": condition,
        "run_id": config["run_id"],
    }
    path.write_text(json.dumps(row) + "\n")
    merge.config_path(path).write_text(json.dumps(config) + "\n")


def test_merge_condition_shards_rewrites_combined_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "c1.jsonl"
    second = tmp_path / "c2.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_shard(first, 1)
    _write_shard(second, 2)
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_shards.py", "--inputs", str(first), str(second), "--output", str(output)],
    )

    merge.main()

    config = json.loads(merge.config_path(output).read_text())
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert config["conditions"] == [1, 2]
    assert len(rows) == 2
    assert {row["source_run_id"] for row in rows} == {"shard-1", "shard-2"}
    assert {row["run_id"] for row in rows} == {config["run_id"]}
