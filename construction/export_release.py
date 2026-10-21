#!/usr/bin/env python3
"""Turn a workspace's validation/test item builds into a released benchmark directory (data/<name>/).

Writes the split manifests, answer keys, per-item metadata, build provenance, and the reference
graphs (nearest-centroid assignments from the GT masks; Trackastra and Hungarian subgraphs from the
mined link errors), in exactly the format of the shipped data/ directory.

  python construction/export_release.py --workspace WORK --build celltrack_bench_consensus_v2 \
      --benchmark-id celltrack-bench-consensus-v2 --output WORK/release/hard
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from celltrack_bench import benchmark_data as B
from celltrack_bench import subgraph_lnk as SL

SPLITS = ("validation", "test")
TRACKER_ERRORS = {
    "trackastra": "evaluation/errors_gt_linking/trackastra_node_only_groupcv/all_groups_link_errors.csv",
    "hungarian": "evaluation/errors_gt_linking/hungarian_distance/all_groups_link_errors.csv",
    "hoct": "evaluation/errors_gt_linking/hoct_groupcv/all_groups_link_errors.csv",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load_build(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], str]:
    """One split's build: items, answers in item order, and the build digest (as assembled for the paper)."""
    items = B.read_json(root / "shared_items.json")
    if not isinstance(items, list) or not items:
        raise B.BenchmarkDataError(f"{root / 'shared_items.json'} must be a non-empty list")
    ids = [str(item["item_id"]) for item in items]
    if len(set(ids)) != len(ids):
        raise B.BenchmarkDataError(f"{root}: duplicate item ids")
    for item_id in ids:
        B.metadata_path(root / "items", item_id)
    answers: dict[str, list[dict[str, Any]]] = {}
    for task in B.TASKS:
        by_id = {str(row["item_id"]): row for row in B.read_jsonl(root / task / "answer_key.jsonl")}
        if set(by_id) != set(ids):
            raise B.BenchmarkDataError(f"{root}/{task}: answers do not match the items")
        answers[task] = [by_id[item_id] for item_id in ids]
    digest = hashlib.sha256()
    digest.update(B.canonical_json(items))
    for task in B.TASKS:
        digest.update(task.encode())
        digest.update(B.canonical_json(answers[task]))
    return items, answers, digest.hexdigest()[:16]


# -- tracker reference subgraphs: reconstruction from the mined AOGM link errors ----------------------

def load_error_index(path: Path) -> dict[tuple[str, str, int], list[dict]]:
    index: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            index[(row["dataset"], row["sequence"], int(row["frame_from"]))].append(row)
    return dict(index)


def displays(meta: dict) -> tuple[dict[int, str], dict[int, str]]:
    return ({int(k): v for k, v in meta["source_display"].items()},
            {int(k): v for k, v in meta["target_display"].items()})


def reconstruct(meta: dict, rows: list[dict]) -> dict[tuple[str, str], str]:
    """Tracker's predicted neighbourhood subgraph {(S#, T#): kind} = (gt minus EA) plus ED, EC overrides."""
    src_disp, tgt_disp = displays(meta)
    pred = {(e["source"], e["target"]): SL.semantic(e.get("semantic")) for e in meta["gt_edges"]}
    for row in rows:
        s, t = int(row["gt_label_from"]), int(row["gt_label_to"])
        if s not in src_disp or t not in tgt_disp:
            continue
        pair = (src_disp[s], tgt_disp[t])
        if row["type"] == "EA":
            pred.pop(pair, None)
        elif row["type"] == "EC":
            pred[pair] = SL.semantic(row["pred_semantic"])
        elif row["type"] == "ED":
            pred[pair] = SL.semantic(row["semantic"])
    return pred


def anchor_verdict(meta: dict, pred: dict[tuple[str, str], str]) -> str | None:
    src_disp, tgt_disp = displays(meta)
    pair = (src_disp[int(meta["proxy"]["source_label"])], tgt_disp[int(meta["proxy"]["target_label"])])
    gt = {(e["source"], e["target"]): SL.semantic(e.get("semantic")) for e in meta["gt_edges"]}
    if pair in gt:
        if pair not in pred:
            return "EA"
        return "EC" if pred[pair] != gt[pair] else None
    return "ED" if pair in pred else None


def references(out: Path, split: str, workspace: Path, source_data_root: Path, errors: dict) -> dict[str, Any]:
    truth = SL.load_truth(split, out)
    items_root = out / "items" / split
    metas = {i: B.load_item_metadata(items_root, i) for i in truth}
    refs: dict[str, Any] = {"nearest_centroid": SL.nearest_centroid(truth, items_root, source_data_root),
                            "trackers": {}}
    checks: dict[str, Any] = {"items": len(truth)}
    for tracker in ("hoct", "trackastra", "hungarian"):
        preds, agree = {}, 0
        for i, m in metas.items():
            preds[i] = reconstruct(m, errors[tracker].get((m["dataset"], m["sequence"], int(m["frame"])), []))
            sel = m["selection"]["trackers"][tracker]
            agree += anchor_verdict(m, preds[i]) == (sel.get("error_type") if sel.get("error") else None)
        checks[f"{tracker}_anchor_agreement"] = agree
        if tracker == "hoct":
            checks["hoct_identical_to_proposal"] = sum(SL.canonical(metas[i]["hoct_edges"]) == preds[i] for i in metas)
            continue
        refs["trackers"][tracker] = {
            i: [{"source": s, "target": t, "semantic": k} for (s, t), k in sorted(p.items())]
            for i, p in preds.items()}
    (out / "references").mkdir(parents=True, exist_ok=True)
    (out / "references" / f"{split}.json").write_text(json.dumps(refs, sort_keys=True) + "\n")
    return checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--build", required=True, help="build directory name under <workspace>/evaluation/")
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-data-root", type=Path, default=None,
                   help="root containing datasets/ (default: the workspace)")
    args = p.parse_args()
    workspace = args.workspace.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    source_root = (args.source_data_root or workspace).expanduser().resolve()

    builds = {split: workspace / "evaluation" / args.build / split for split in SPLITS}
    payloads = {split: load_build(root) for split, root in builds.items()}
    ids = [{str(i["item_id"]) for i in payloads[s][0]} for s in SPLITS]
    seqs = [{(str(i["dataset"]), str(i["sequence"])) for i in payloads[s][0]} for s in SPLITS]
    if ids[0] & ids[1] or seqs[0] & seqs[1]:
        raise SystemExit("validation and test overlap")

    for split, (items, answers, _digest) in payloads.items():
        _write_json(out / f"{split}_items.json", items)
        for task in B.TASKS:
            _write_jsonl(out / split / task / "answer_key.jsonl", answers[task])
        for item in items:
            src = B.metadata_path(builds[split] / "items", str(item["item_id"]))
            dst = out / "items" / split / str(item["item_id"]) / "metadata.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        for name in ("README.md", "build_summary.json"):
            (out / "build" / split).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(builds[split] / name, out / "build" / split / name)
    _write_json(out / B.MANIFEST_NAME, {
        "schema_version": 1,
        "benchmark_id": B.safe_slug(args.benchmark_id),
        "splits": {split: {"items_root": f"items/{split}", "item_count": len(payloads[split][0]),
                           "build_digest": payloads[split][2]} for split in SPLITS},
    })

    errors = {t: load_error_index(workspace / path) for t, path in TRACKER_ERRORS.items()}
    for split in SPLITS:
        checks = references(out, split, workspace, source_root, errors)
        identity = B.benchmark_identity(out, out / "items" / split, split)
        print(f"{out.name}/{split}: {len(payloads[split][0])} items, identity {identity.value}, checks {json.dumps(checks)}")


if __name__ == "__main__":
    main()
