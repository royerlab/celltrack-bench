#!/usr/bin/env python3
"""Collect the saved tracker outputs the benchmark is built from into tracker_outputs/ (maintainer tool).

  python construction/make_tracker_outputs.py /path/to/tracking-benchmark tracker_outputs

Copies, under evaluation/ (the relative paths are part of each item's recorded provenance):
  hoct_groupcv/full_graphs/group_<k>/<dataset>/<sequence>.geff     HOCT graphs: candidate links, link
                                                                    similarities, selected solution
  trackastra_node_only_groupcv/full_graphs/...                     Trackastra graphs (link similarities)
  errors_gt_linking/{hoct_groupcv,trackastra_node_only_groupcv,hungarian_distance}/
      all_groups_link_errors.csv                                    each tracker's AOGM link errors

The trackers ran on ground-truth detections, so their graphs also carry per-node features computed
from the source images and the ground-truth segmentation (masks, bounding boxes, intensity and shape
statistics). The construction never reads them, so they are removed here; node positions, the
network predictions and every edge attribute are kept.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

GRAPH_ROOTS = ("evaluation/hoct_groupcv/full_graphs", "evaluation/trackastra_node_only_groupcv/full_graphs")
ERROR_TABLES = tuple(
    f"evaluation/errors_gt_linking/{tracker}/all_groups_link_errors.csv"
    for tracker in ("hoct_groupcv", "trackastra_node_only_groupcv", "hungarian_distance")
)
# node features derived from the images / GT segmentation; unused by the construction
DROPPED_NODE_PROPS = (
    "mask", "bbox", "intensity_min", "intensity_max", "intensity_mean", "intensity_std",
    "inertia_tensor", "equivalent_diameter_area", "border_dist",
)


def strip_graph(geff: Path) -> None:
    meta_path = geff / "zarr.json"
    meta = json.loads(meta_path.read_text())
    props = meta["attributes"]["geff"]["node_props_metadata"]
    for name in DROPPED_NODE_PROPS:
        props.pop(name, None)
        shutil.rmtree(geff / "nodes" / "props" / name, ignore_errors=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    graphs = 0
    for root in GRAPH_ROOTS:
        shutil.copytree(src / root, out / root)
        for geff in sorted((out / root).glob("group_*/*/*.geff")):
            strip_graph(geff)
            graphs += 1
    for table in ERROR_TABLES:
        (out / table).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / table, out / table)
    print(f"wrote {out}: {graphs} graphs (dropped node features: {', '.join(DROPPED_NODE_PROPS)}), "
          f"{len(ERROR_TABLES)} error tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
