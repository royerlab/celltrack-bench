from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from celltrack_bench import benchmark_data as B
from celltrack_bench import render as R
from conftest import requires_benchmark

DATA = R.REPO_ROOT / "data"

# Content digests of the benchmark splits the paper's runs were made on (they are part of every
# result row's `benchmark_identity`). The released files must reproduce them exactly.
PAPER_IDENTITIES = {
    ("hard", "test"): "celltrack-bench-consensus-v2-3bdb6d31bf98f548",
    ("hard", "validation"): "celltrack-bench-consensus-v2-6e05798663aaf734",
    # the full benchmark has no leaderboard runs; these are the identities of the released build
    ("full", "test"): "celltrack-bench-consensus-v2-xl-d1da48bcf453e840",
    ("full", "validation"): "celltrack-bench-consensus-v2-xl-6c150fb52cea70ee",
}


def _write_split(root: Path, item_id: str) -> Path:
    """A one-item benchmark directory in the released layout."""
    item = {"item_id": item_id, "dataset": "demo", "sequence": "01"}
    (root / "items" / "test" / item_id).mkdir(parents=True)
    (root / "items" / "test" / item_id / "metadata.json").write_text(
        json.dumps({**item, "frame": 0, "frame_to": 1, "candidate_map": {"S1": ["T1"]},
                    "hoct_edges": [], "gt_edges": [{"source": "S1", "target": "T1", "semantic": "continuation"}]})
    )
    (root / "test_items.json").write_text(json.dumps([item]))
    (root / "benchmark_manifest.json").write_text(
        json.dumps({"benchmark_id": "demo", "splits": {"test": {"items_root": "items/test"}}})
    )
    answers = {
        "cell_linking": {"links": [{"source": "S1", "target": "T1", "semantic": "continuation"}], "unlinked_sources": []},
        "track_repair": {"operations": [{"op": "add", "source": "S1", "target": "T1", "semantic": "continuation"}]},
        "track_inspection": {"links_to_inspect": [{"source": "S1", "target": "T1", "hoct_status": "unselected", "issue": "missing"}]},
    }
    for task, answer in answers.items():
        (root / "test" / task).mkdir(parents=True)
        (root / "test" / task / "answer_key.jsonl").write_text(json.dumps({"item_id": item_id, "answer": answer}) + "\n")
    return root


def test_select_items_round_robins_across_datasets() -> None:
    items = [
        {"item_id": "a1", "dataset": "a", "sequence": "01"},
        {"item_id": "a2", "dataset": "a", "sequence": "01"},
        {"item_id": "b1", "dataset": "b", "sequence": "01"},
        {"item_id": "b2", "dataset": "b", "sequence": "01"},
    ]
    assert [item["item_id"] for item in B.select_items(items, sample=3)] == ["a1", "b1", "a2"]
    with pytest.raises(B.BenchmarkDataError):
        B.select_items(items, sample=1, limit=1)


def test_identity_covers_item_metadata(tmp_path: Path) -> None:
    root = _write_split(tmp_path / "bench", "item-1")
    items_root = B.resolve_items_root(root, "test", None, Path("unused"))
    assert items_root == (root / "items" / "test").resolve()
    before = B.benchmark_identity(root, items_root, "test")
    meta_path = items_root / "item-1" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["candidate_map"] = {"S1": []}
    meta_path.write_text(json.dumps(meta))
    assert B.benchmark_identity(root, items_root, "test").value != before.value


def test_load_manifest_rejects_invalidated_split(tmp_path: Path) -> None:
    (tmp_path / "benchmark_manifest.json").write_text("{}")
    (tmp_path / "INVALIDATED.md").write_text("This benchmark must not be used.\n")
    with pytest.raises(B.BenchmarkDataError, match="explicitly invalidated"):
        B.load_manifest(tmp_path)


def test_gzip_results_are_read_transparently(tmp_path: Path) -> None:
    rows = [{"item_id": "a", "task": "cell_linking"}, {"item_id": "b", "task": "track_repair"}]
    plain = tmp_path / "run.jsonl"
    plain.write_text("".join(json.dumps(r) + "\n" for r in rows))
    packed = tmp_path / "packed.jsonl.gz"
    with gzip.open(packed, "wt") as handle:
        handle.write(plain.read_text())
    assert B.read_jsonl(packed) == B.read_jsonl(plain) == rows
    assert [json.loads(line) for line in B.open_text(packed)] == rows
    assert B.run_config_path(packed) == tmp_path / "packed.jsonl.config.json"
    assert B.run_config_path(plain) == tmp_path / "run.jsonl.config.json"


@pytest.mark.parametrize("benchmark,split", [
    pytest.param(b, s, marks=requires_benchmark(b)) for b, s in sorted(PAPER_IDENTITIES)])
def test_released_splits_reproduce_the_paper_identity(benchmark: str, split: str) -> None:
    split_dir = DATA / benchmark
    items_root = B.resolve_items_root(split_dir, split, None, Path("unused"))
    assert B.benchmark_identity(split_dir, items_root, split).value == PAPER_IDENTITIES[(benchmark, split)]


@pytest.mark.parametrize("benchmark,split,count", [
    pytest.param(b, s, n, marks=requires_benchmark(b))
    for b, s, n in (("hard", "test", 244), ("hard", "validation", 205), ("full", "test", 911), ("full", "validation", 626))
])
def test_released_splits_are_complete(benchmark: str, split: str, count: int) -> None:
    split_dir = DATA / benchmark
    items_root = B.resolve_items_root(split_dir, split, None, Path("unused"))
    items, answers = B.validate_benchmark(split_dir, items_root, split)
    assert len(items) == count and len(answers) == 3 * count
    refs = json.loads((split_dir / "references" / f"{split}.json").read_text())
    ids = {item["item_id"] for item in items}
    assert set(refs["nearest_centroid"]) == ids
    assert set(refs["trackers"]["hungarian"]) == ids == set(refs["trackers"]["trackastra"])
