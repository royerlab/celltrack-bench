#!/usr/bin/env python3
"""The eight harness conditions, and the server-side tools three of them use.

Factorised so each adjacent pair isolates one variable:

    cond  cells pointed at by        closer look via                    isolates
    1     text: normalized bboxes    crop_and_zoom(bbox, zoom)          baseline
    2     visual: outlines + ids     crop_and_zoom(bbox, zoom)          1->2 text coords vs visual marks
    3     visual: outlines + ids     zoom(cell_id, zoom)                2->3 free region vs cell-addressed
    4     visual: outlines + ids     pre-supplied crops, outlined       3->4 agency vs everything up front
    5     visual: outlines + ids     pre-supplied crops, NOT outlined   4->5 does the in-crop outline matter
    6     sources text bboxes,       crop_and_zoom(bbox, zoom)          1->6 target coords, exact vs visual
          targets outlined + ids
    7     crops only, NO overview    pre-supplied crops                 5->7 remove global position entirely
    8     text bboxes AND outlines   crop_and_zoom(bbox, zoom)          1->8 do visual marks add to coords

Conditions 6 and 7 exist because every one of 1-5 turned out to follow a nearest-neighbour proximity
rule (0.655-0.910 of links), c1 most precisely of all because it is handed
exact coordinates for both frames.

  - c6 keeps c1's textual source addressing but forces the target positions to be read off the image,
    so proximity matching becomes as imprecise as it is for the visual conditions.
  - c7 withholds the overview altogether. With no whole-frame image there is no global position for any
    cell, so proximity is not computable at all and only appearance can drive the match. That makes it
    an ablation rather than a deployable harness: proximity is a legitimate physical prior for cell
    tracking, and c7 deliberately removes real signal to isolate perception.

Everything else -- task text, answer schema, item, scorer -- is identical across conditions.

Tool calls are unlimited by design; TOOL_CALL_SAFETY_CAP only stops a runaway loop from spending
money forever, and a run that reaches it is reported rather than silently truncated.

Overview and per-cell crops are always sent as SEPARATE images in the request, never composited into
one sheet -- id-to-image association is then positional and unambiguous.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import cv2

from . import render as R

TASKS = ("cell_linking", "track_repair", "track_inspection")
TOOL_CALL_SAFETY_CAP = 40  # runaway guard only; not a budget the model is told about
CONDITIONS = (1, 2, 3, 4, 5, 6, 7, 8)

# c7 outlines the subject inside each crop even though c5 does not. Without the overview there is no
# second view to disambiguate which cell an id refers to, and at pad_frac 1.5 a crop routinely contains
# neighbours -- an un-outlined crop would add an identification failure that has nothing to do with the
# position bias c7 exists to remove. Outlined, c7 differs from c4 in exactly one thing: no global
# position. Set False to pair it against c5 instead.
C7_OUTLINE_CROPS = True

TASK_BLOCK = {
    "cell_linking": (
        "Return the complete ground-truth linking for all {n_source} sources. A source may disappear "
        "(zero targets), continue (one target), or branch to one or more visible daughter targets. "
        "Use semantic=division for every daughter edge in a branch. The number of visible daughters "
        "is defined by the lineage annotation; do not assume it is exactly two. Do not omit links "
        "simply because they are not the most obvious error in the image."
    ),
    "track_repair": (
        "A candidate tracking graph has been proposed for these frames. Return every operation "
        "required to transform it into the complete ground-truth graph. Valid operations are add, "
        "delete and change_semantic."
    ),
    "track_inspection": (
        "A candidate tracking graph has been proposed for these frames. Select every source-target "
        "relationship that a human should inspect: selected links that are spurious or have the wrong "
        "semantic type, and unselected candidate links that should have been selected."
    ),
}

SCHEMA = {
    "cell_linking": (
        '{"links": [{"source": "S1", "target": "T2", "semantic": "continuation"}], '
        '"unlinked_sources": ["S4"], "confidence": 0.0, "reason": "short visual justification"}'
    ),
    "track_repair": (
        '{"operations": [{"op": "add", "source": "S1", "target": "T2", "semantic": "continuation"}, '
        '{"op": "delete", "source": "S3", "target": "T4", "semantic": "continuation"}, '
        '{"op": "change_semantic", "source": "S5", "target": "T7", '
        '"from_semantic": "continuation", "to_semantic": "division"}]}'
    ),
    "track_inspection": (
        '{"links_to_inspect": [{"source": "S1", "target": "T2", "hoct_status": "unselected", '
        '"issue": "missing"}, {"source": "S3", "target": "T4", "hoct_status": "selected", '
        '"issue": "spurious"}, {"source": "S5", "target": "T7", "hoct_status": "selected", '
        '"issue": "wrong_semantic"}]}'
    ),
}

EVIDENCE = {
    1: (
        "The image is unannotated. Each cell's location is given below as a normalized bounding box "
        "[x0, y0, x1, y1], where 0.0-1.0 is relative to that cell's own panel (the left panel for S "
        "ids, the right panel for T ids):\n{bboxes}\n"
        "You may call crop_and_zoom(panel, bbox, zoom) to inspect any region more closely, as many "
        "times as you need. zoom is 1.0-4.0."
    ),
    2: (
        "Every cell is outlined in the image and labelled with its id next to the outline. The outline "
        "is the exact segmentation boundary and the cell interior is unmodified.\n"
        "You may call crop_and_zoom(panel, bbox, zoom) to inspect any region more closely, as many "
        "times as you need. zoom is 1.0-4.0."
    ),
    3: (
        "Every cell is outlined in the image and labelled with its id next to the outline.\n"
        "You may call zoom(cell_id, zoom) to get a close-up centred on any labelled cell, as many "
        "times as you need. zoom is 1.0-4.0."
    ),
    4: (
        "Every cell is outlined in the image and labelled with its id next to the outline. After the "
        "overview you are given one close-up per cell as a separate image, in id order, each centred "
        "on its cell with the surrounding context included and that cell outlined. No tools are "
        "available: this is the complete visual evidence."
    ),
    5: (
        "Every cell is outlined in the image and labelled with its id next to the outline. After the "
        "overview you are given one close-up per cell as a separate image, in id order. Each close-up "
        "is centred on its cell and includes the surrounding context; the close-ups themselves carry "
        "no outline, so the cell in question is the one at the centre of the crop. Neighbouring cells "
        "may also be visible. No tools are available: this is the complete visual evidence."
    ),
    6: (
        "In frame t the image is unannotated and each source cell's location is given below as a "
        "normalized bounding box [x0, y0, x1, y1], where 0.0-1.0 is relative to the left panel:\n"
        "{bboxes}\n"
        "In frame t+1 no coordinates are given: every target cell is outlined in the image and "
        "labelled with its id next to the outline, so you must read the target positions off the "
        "picture.\n"
        "You may call crop_and_zoom(panel, bbox, zoom) to inspect any region more closely, as many "
        "times as you need. zoom is 1.0-4.0."
    ),
    8: (
        "Each cell's location is given below as a normalized bounding box [x0, y0, x1, y1], where "
        "0.0-1.0 is relative to that cell's own panel (the left panel for S ids, the right panel for T "
        "ids):\n{bboxes}\n"
        "The same cells are ALSO outlined in the image and labelled with their id next to the outline, "
        "so every cell is identified twice: once numerically above and once visually in the picture.\n"
        "You may call crop_and_zoom(panel, bbox, zoom) to inspect any region more closely, as many "
        "times as you need. zoom is 1.0-4.0."
    ),
    7: (
        "You are given no whole-frame image. Instead you receive one close-up per cell as a separate "
        "image, in id order: first every source cell from frame t, then every target cell from frame "
        "t+1. Each close-up is centred on its cell, includes the immediate surrounding context, and has "
        "that cell outlined; neighbouring cells may also be visible but are not the subject.\n"
        "Because there is no overview, you cannot tell where in the frame any cell lies, and you "
        "cannot use proximity between frames to match cells. Decide from appearance -- shape, size, "
        "texture, intensity, and the local arrangement visible inside each crop.\n"
        "No tools are available: this is the complete visual evidence."
    ),
}

CROP_AND_ZOOM_TOOL = {
    "type": "function",
    "name": "crop_and_zoom",
    "description": "Return a magnified crop of a region of one panel.",
    "parameters": {
        "type": "object",
        "properties": {
            "panel": {"type": "string", "enum": ["t", "t1"], "description": "'t' = left/frame t, 't1' = right/frame t+1"},
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "[x0, y0, x1, y1] normalized 0.0-1.0 within that panel",
            },
            "zoom": {"type": "number", "description": "magnification, 1.0-4.0"},
        },
        "required": ["panel", "bbox", "zoom"],
        "additionalProperties": False,
    },
}

ZOOM_TOOL = {
    "type": "function",
    "name": "zoom",
    "description": "Return a close-up centred on one labelled cell, with surrounding context.",
    "parameters": {
        "type": "object",
        "properties": {
            "cell_id": {"type": "string", "description": "an id shown in the image, e.g. 'S1' or 'T3'"},
            "zoom": {"type": "number", "description": "magnification, 1.0-4.0"},
        },
        "required": ["cell_id", "zoom"],
        "additionalProperties": False,
    },
}


def png_data_url(image) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@dataclass
class Built:
    """Everything the runner needs for one (item, task, condition)."""

    text: str
    images: list[Any]
    tools: list[dict]
    panels: dict[str, R.Panel]
    label_of: dict[str, tuple[str, int]] = field(default_factory=dict)  # "S1" -> ("t", label)


def budget_line(total: int) -> str:
    return (
        f"Output budget: you have {total} output tokens in total for this item, counting your "
        f"reasoning and every tool-calling turn, not just the final answer. Spend them so that the "
        f"JSON answer still fits. You will be told when 20% remains."
    )


def build(meta: dict[str, Any], task: str, condition: int, output_budget: int = 8192) -> Built:
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")

    # c1 sends a clean image; c6 marks only the targets; c7's overview is built for its panels but
    # never sent, so its annotation is irrelevant.
    annotate: bool | tuple[str, ...]
    if condition in (1, 7):
        annotate = ()
    elif condition == 6:
        annotate = ("t1",)
    else:
        annotate = True
    overview_img, panels = R.overview(meta, annotate=annotate)
    images = [] if condition == 7 else [overview_img]

    label_of = {name: (which, label) for which in ("t", "t1") for label, name in panels[which].display.items()}

    if condition in (1, 6, 8):
        # c1 gives coordinates for both frames, which makes nearest-neighbour matching exactly
        # computable. c6 gives them for the source frame only, so the targets must be located visually.
        # c8 gives both frames AND annotates the image, to test whether visual marks add anything once
        # exact coordinates are already present.
        panels_with_coords = ("t",) if condition == 6 else ("t", "t1")
        lines = []
        for which in panels_with_coords:
            for name, bb in sorted(panels[which].normalized_bboxes().items()):
                lines.append(f"  {name}: [{bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.3f}, {bb[3]:.3f}]")
        evidence = EVIDENCE[condition].format(bboxes="\n".join(lines))
    else:
        evidence = EVIDENCE[condition]

    if condition in (4, 5, 7):
        outline = condition == 4 or (condition == 7 and C7_OUTLINE_CROPS)
        for which, colour in (("t", R.SOURCE_COLOUR), ("t1", R.TARGET_COLOUR)):
            panel = panels[which]
            for label, _name in sorted(panel.display.items(), key=lambda kv: kv[1]):
                images.append(R.cell_crop(panel, label, out_px=320, outline=outline, colour=colour))

    tools = []
    if condition in (1, 2, 6, 8):
        tools = [CROP_AND_ZOOM_TOOL]
    elif condition == 3:
        tools = [ZOOM_TOOL]

    candidate_map = meta.get("candidate_map") or {}
    cm = "\n".join(f"- {s}: {', '.join(candidate_map[s])}" for s in sorted(candidate_map, key=lambda x: (len(x), x)))
    n_source = len(meta.get("source_display") or {})

    if condition == 7:
        # No overview is sent, so promising two visible frames would be a lie.
        scene = (f"This item comes from two consecutive microscopy frames of a cell-tracking sequence, "
                 f"frame t and frame t+1. Frame t contains {n_source} labelled source cells; frame t+1 "
                 f"contains their candidate target cells. The frames themselves are not shown.")
    else:
        scene = (f"Two consecutive microscopy frames from a cell-tracking sequence are shown: frame t "
                 f"(left) and frame t+1 (right). Frame t contains {n_source} labelled source cells; "
                 f"frame t+1 contains their candidate target cells.")
    hoct_block = ""
    if task in ("track_repair", "track_inspection"):
        hoct = meta.get("hoct_edges") or []
        rows = json.dumps([
            {"semantic": e.get("semantic"), "source": e.get("source"), "target": e.get("target")}
            for e in hoct
        ])
        hoct_block = f"HOCT currently selected these links:\n{rows}\n\n"

    header = (
        f"{budget_line(output_budget)}\n\n"
        f"{scene}\n\n"
        f"Candidate targets allowed for each source:\n{cm}\n\n"
        "A continuation joins a source to the same track in the next frame. A division edge joins "
        "an annotated parent to each visible daughter track; adjacent-frame annotations can contain "
        "one or more visible daughters, although two is the usual case.\n\n"
        f"{hoct_block}"
    )

    text = (
        header
        + TASK_BLOCK[task].format(n_source=n_source)
        + "\n\n"
        + evidence
        + "\n\nThink it through, then output only the JSON object, with no text after it.\n"
        + SCHEMA[task]
    )
    return Built(text=text, images=images, tools=tools, panels=panels, label_of=label_of)


def run_tool(built: Built, name: str, arguments: str | dict) -> tuple[str, Any | None]:
    """Execute a tool call server-side. Returns (text_for_model, image_or_None)."""
    args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    if name == "crop_and_zoom":
        which = "t" if str(args.get("panel", "t")).lower() in ("t", "t0", "left") else "t1"
        try:
            image = R.crop_region(built.panels[which], args["bbox"], float(args.get("zoom", 2.0)))
        except Exception as exc:  # malformed request is information, not a crash
            return f"crop_and_zoom failed: {exc}", None
        return f"Crop of panel {which}, bbox {args['bbox']}, zoom {args.get('zoom')}.", image
    if name == "zoom":
        cell = str(args.get("cell_id", "")).strip().upper()
        if cell not in built.label_of:
            return f"zoom failed: unknown cell_id {cell!r}. Valid ids: {', '.join(sorted(built.label_of))}", None
        which, label = built.label_of[cell]
        zoom = max(1.0, min(R.MAX_ZOOM, float(args.get("zoom", 3.0))))
        colour = R.SOURCE_COLOUR if which == "t" else R.TARGET_COLOUR
        image = R.cell_crop(
            built.panels[which], label, pad_frac=max(0.3, R.DEFAULT_PAD_FRAC * 2.0 / zoom),
            out_px=320, colour=colour
        )
        return f"Close-up of {cell} at zoom {zoom}.", image
    return f"unknown tool {name}", None
