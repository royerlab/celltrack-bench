#!/usr/bin/env python3
"""Render harness images for a HOCT benchmark item, from source data.

Why not reuse the item's own JPEGs: `frame_t.jpg`, `frame_t1.jpg` and `cell_linking.jpg` all have
filled colour overlays and id text **baked into the pixels** (verified by zooming into frame_t.jpg,
which shows an orange blob labelled "S4" covering the cell). Using them would give every condition
condition-2's visual marks for free and confound the comparison. So we re-render from
`datasets/<DS>/<SEQ>/t####.tif` plus the GT mask `<SEQ>_GT/TRA/man_track####.tif`, reusing the
benchmark's own geometry so cell ids and coordinates stay consistent with its answer keys.
The source-data root is ``$CELLTRACK_DATA_ROOT`` (default ``<repo>/raw_data``); see the README.

Geometry is fixed by the benchmark build:
PANEL_SIZE=768, HEADER_PX=58, DIVIDER_PX=16, and the item's `crop_yxyx` region of the full frame.

Design decisions:
- outlines are a 1px colour contour over a 3px black halo, so the cell interior is never covered
- ids are black text with a white halo, placed OUTSIDE the cell bbox; placement tries 8 offsets and
  rejects any that overlap a cell body or an already-placed label
- per-cell crops pad by `pad_frac` x the cell's larger side (default 1.5). Condition 4 outlines the
  subject inside the crop; condition 5 does not, which is exactly the pair that measures whether
  the outline is what makes a crop interpretable when it contains more than one cell
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA_ROOT = Path(os.environ.get("CELLTRACK_DATA_ROOT", str(REPO_ROOT / "raw_data")))
DEFAULT_ITEMS_ROOT = REPO_ROOT / "data" / "hard" / "items" / "test"
# Kept as an import-compatible alias for older scripts.
BENCH_ROOT = SOURCE_DATA_ROOT
PANEL_SIZE = 768
HEADER_PX = 58
DIVIDER_PX = 16
SOURCE_COLOUR = (0, 190, 255)  # amber, BGR
TARGET_COLOUR = (90, 255, 120)  # green, BGR
FONT = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE = 0.62
LABEL_HALO = 4
LABEL_CORE = 2
# Zoom ceiling. Cells here are ~30px across, so past ~4x every added pixel is
# interpolation rather than detail; the cap keeps requests honest and images small.
MAX_ZOOM = 4.0
# Context padding for per-cell crops, as a multiple of the cell's larger side. 1.5 means
# the crop spans ~4x the cell, so neighbours and local texture are visible.
DEFAULT_PAD_FRAC = 1.5


@dataclass
class Panel:
    """One rendered frame panel plus the label mask aligned to it."""

    image: np.ndarray  # (PANEL, PANEL, 3) uint8, BGR, no annotation
    mask: np.ndarray  # (PANEL, PANEL) int32 GT label ids
    display: dict[int, str]  # label id -> "S1" / "T3"

    def bbox(self, label: int) -> tuple[int, int, int, int] | None:
        ys, xs = np.nonzero(self.mask == label)
        if not ys.size:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def normalized_bboxes(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for label, name in self.display.items():
            bb = self.bbox(label)
            if bb is not None:
                out[name] = [round(v / PANEL_SIZE, 4) for v in bb]
        return out


@lru_cache(maxsize=512)
def _frame_path_map(directory: str, prefix: str) -> dict[int, str]:
    """frame number -> file, by globbing rather than assuming a zero-pad width.

    The CTC layout pads to the sequence length: BF-C2DL-HSC has >1000 frames so it uses t0000.tif,
    while every other dataset here uses t000.tif. Hardcoding "%04d" silently matched exactly one
    dataset and failed 675 of 863 renders on the rest.
    """
    out: dict[int, str] = {}
    for path in Path(directory).glob(f"{prefix}*.tif"):
        digits = path.stem[len(prefix):]
        if digits.isdigit():
            out[int(digits)] = str(path)
    return out


def resolve_frame(directory: Path, prefix: str, frame: int) -> str:
    mapping = _frame_path_map(str(directory), prefix)
    if frame not in mapping:
        raise FileNotFoundError(
            f"frame {frame} not found as {directory}/{prefix}*.tif "
            f"({len(mapping)} frames present, range "
            f"{min(mapping) if mapping else '-'}..{max(mapping) if mapping else '-'})"
        )
    return mapping[frame]


@lru_cache(maxsize=64)
def _read_2d(path: str) -> np.ndarray:
    array = tifffile.imread(path)
    while array.ndim > 2:
        array = array[0]
    return array


def _stretch(array: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    values = array.astype(np.float32)
    lo, hi = np.percentile(values, (lo_pct, hi_pct))
    scaled = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _source_data_root(meta: dict[str, Any], root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    configured = meta.get("_harness_source_data_root")
    return Path(configured) if configured else SOURCE_DATA_ROOT


def load_panel(meta: dict[str, Any], which: str, root: Path | None = None) -> Panel:
    """`which` is "t" or "t1"."""
    root = _source_data_root(meta, root)
    dataset, sequence = str(meta["dataset"]), str(meta["sequence"])
    frame = int(meta["frame"] if which == "t" else meta["frame_to"])
    display = {int(k): str(v) for k, v in (meta["source_display"] if which == "t" else meta["target_display"]).items()}
    y0, x0, y1, x1 = (int(v) for v in meta["crop_yxyx"])

    raw = _read_2d(resolve_frame(root / "datasets" / dataset / sequence, "t", frame))
    image = cv2.resize(_stretch(raw[y0:y1, x0:x1]), (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC)
    labels = _read_2d(
        resolve_frame(root / "datasets" / dataset / f"{sequence}_GT" / "TRA", "man_track", frame)
    )
    mask = cv2.resize(labels[y0:y1, x0:x1].astype(np.int32), (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

    missing = [name for label, name in display.items() if not (mask == label).any()]
    if missing:
        raise ValueError(f"{meta['item_id']}: cells absent from the rendered {which} panel: {missing}")
    return Panel(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), mask, display)


def _text(canvas: np.ndarray, text: str, org: tuple[int, int]) -> None:
    cv2.putText(canvas, text, org, FONT, LABEL_SCALE, (255, 255, 255), LABEL_HALO, cv2.LINE_AA)
    cv2.putText(canvas, text, org, FONT, LABEL_SCALE, (0, 0, 0), LABEL_CORE, cv2.LINE_AA)


def draw_outlines(panel: Panel, colour: tuple[int, int, int]) -> np.ndarray:
    canvas = panel.image.copy()
    for label in panel.display:
        contours, _ = cv2.findContours((panel.mask == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.drawContours(canvas, contours, -1, colour, 1, cv2.LINE_AA)
    return canvas


def draw_labels(canvas: np.ndarray, panel: Panel) -> None:
    """Place every id outside its cell, avoiding cell bodies and other labels."""
    occupied = (panel.mask > 0).astype(np.uint8)
    placed: list[tuple[int, int, int, int]] = []
    for label, name in sorted(panel.display.items(), key=lambda kv: kv[1]):
        bb = panel.bbox(label)
        if bb is None:
            continue
        bx0, by0, bx1, by1 = bb
        (tw, th), _ = cv2.getTextSize(name, FONT, LABEL_SCALE, LABEL_CORE)
        chosen = None
        candidates = (
            (bx1 + 6, by0 - 6), (bx1 + 6, by0 + th + 6), (bx0 - tw - 6, by0 - 6), (bx0 - tw - 6, by0 + th + 6),
            (bx0, by0 - 10), (bx0, by1 + th + 10), (bx1 + 6, (by0 + by1 + th) // 2), (bx0 - tw - 6, (by0 + by1 + th) // 2),
        )
        for ox, oy in candidates:
            if ox < 2 or oy - th < 2 or ox + tw > PANEL_SIZE - 2 or oy > PANEL_SIZE - 2:
                continue
            box = (ox - 2, oy - th - 2, ox + tw + 2, oy + 2)
            if occupied[max(0, box[1]) : box[3], max(0, box[0]) : box[2]].any():
                continue
            if any(not (box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3]) for p in placed):
                continue
            chosen = (ox, oy, box)
            break
        if chosen is None:  # last resort: nudge inside the frame and accept an overlap
            ox = min(bx1 + 6, PANEL_SIZE - tw - 4)
            oy = max(by0 - 6, th + 4)
            chosen = (ox, oy, (ox - 2, oy - th - 2, ox + tw + 2, oy + 2))
        _text(canvas, name, (chosen[0], chosen[1]))
        placed.append(chosen[2])


def _header(width: int, text: str) -> np.ndarray:
    band = np.full((HEADER_PX, width, 3), 18, np.uint8)
    cv2.putText(band, text, (12, 38), FONT, 0.78, (255, 255, 255), 1, cv2.LINE_AA)
    return band


def compose(left: np.ndarray, right: np.ndarray, left_text: str, right_text: str) -> np.ndarray:
    gap_body = np.full((PANEL_SIZE, DIVIDER_PX, 3), 18, np.uint8)
    gap_head = np.full((HEADER_PX, DIVIDER_PX, 3), 18, np.uint8)
    head = np.hstack([_header(PANEL_SIZE, left_text), gap_head, _header(PANEL_SIZE, right_text)])
    return np.vstack([head, np.hstack([left, gap_body, right])])


def overview(
    meta: dict[str, Any],
    *,
    annotate: bool | Iterable[str],
    root: Path | None = None,
) -> tuple[np.ndarray, dict[str, Panel]]:
    """The two-panel overview.

    `annotate=False` is condition 1's clean image and `annotate=True` annotates both panels. It also
    accepts an iterable of panel keys so a condition can annotate one side only -- condition 6 marks
    the targets visually while describing the sources as text bboxes, which needs ("t1",).
    """
    t, t1 = load_panel(meta, "t", root), load_panel(meta, "t1", root)
    if isinstance(annotate, bool):
        which = ("t", "t1") if annotate else ()
    else:
        which = tuple(annotate)

    parts, suffix = {}, {}
    for key, panel, colour in (("t", t, SOURCE_COLOUR), ("t1", t1, TARGET_COLOUR)):
        if key in which:
            drawn = draw_outlines(panel, colour)
            draw_labels(drawn, panel)
            parts[key] = drawn
            suffix[key] = f" - {'source' if key == 't' else 'target'} cells outlined"
        else:
            parts[key] = panel.image.copy()
            suffix[key] = ""
    image = compose(parts["t"], parts["t1"],
                    f"Frame t ({meta['frame']}){suffix['t']}",
                    f"Frame t+1 ({meta['frame_to']}){suffix['t1']}")
    return image, {"t": t, "t1": t1}


def cell_crop(
    panel: Panel,
    label: int,
    *,
    pad_frac: float = DEFAULT_PAD_FRAC,
    out_px: int = 320,
    outline: bool = True,
    colour: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Close-up centred on one cell, padded by `pad_frac` x its larger side.

    The subject is outlined because at this padding a crop routinely contains other cells; without
    it the model cannot tell which cell the id refers to.
    """
    bb = panel.bbox(label)
    if bb is None:
        raise ValueError(f"label {label} absent from panel")
    bx0, by0, bx1, by1 = bb
    pad = int(round(max(bx1 - bx0 + 1, by1 - by0 + 1) * pad_frac))
    cx0, cy0 = max(0, bx0 - pad), max(0, by0 - pad)
    cx1, cy1 = min(PANEL_SIZE, bx1 + pad + 1), min(PANEL_SIZE, by1 + pad + 1)

    base = panel.image
    if outline:
        base = base.copy()
        contours, _ = cv2.findContours((panel.mask == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, contours, -1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.drawContours(base, contours, -1, colour or SOURCE_COLOUR, 1, cv2.LINE_AA)

    patch = base[cy0:cy1, cx0:cx1]
    scale = out_px / max(patch.shape[:2])
    return cv2.resize(patch, (int(patch.shape[1] * scale), int(patch.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)


def crop_region(panel: Panel, bbox_norm: list[float], zoom: float, *, out_cap: int = 768) -> np.ndarray:
    """crop_and_zoom: normalized bbox on this panel, upscaled by `zoom`, capped so requests stay sane."""
    x0, y0, x1, y1 = (float(v) for v in bbox_norm)
    x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty bbox {bbox_norm}")
    px0, py0 = int(x0 * PANEL_SIZE), int(y0 * PANEL_SIZE)
    px1, py1 = max(px0 + 1, int(x1 * PANEL_SIZE)), max(py0 + 1, int(y1 * PANEL_SIZE))
    patch = panel.image[py0:py1, px0:px1]
    zoom = max(1.0, min(MAX_ZOOM, float(zoom)))
    target = min(out_cap, int(max(patch.shape[:2]) * zoom))
    scale = target / max(patch.shape[:2])
    return cv2.resize(patch, (max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)


def load_meta(
    item_id: str,
    items_root: Path = DEFAULT_ITEMS_ROOT,
    source_data_root: Path = SOURCE_DATA_ROOT,
) -> dict[str, Any]:
    from .benchmark_data import load_item_metadata

    meta = load_item_metadata(Path(items_root), item_id)
    # conditions.build calls overview without a root argument. Keep the source-data
    # location attached to this in-memory metadata object, separate from the item build.
    meta["_harness_source_data_root"] = str(Path(source_data_root).expanduser().resolve())
    return meta
