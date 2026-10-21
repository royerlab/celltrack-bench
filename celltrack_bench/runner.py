#!/usr/bin/env python3
"""Run a model through the CellTrack-Bench harness over items x tasks x conditions, with a response cache.

The defaults are the paper's frozen protocol: condition c1, low reasoning effort, an 8192-token total
output budget with 2048 tokens reserved for the final answer, temperature 1.0 and top-p 0.95.

Every call is cached on disk keyed by (item, task, condition, model, effort, harness revision), so a
re-run costs nothing and an interrupted sweep resumes. Tool calls are executed server-side by
harness.conditions.run_tool and fed back as images.

WHAT IS PERSISTED, per call:
  runs/<split>_<benchmark>_<model>_<tag>_<rev>_<run_id>.jsonl
                                  compact index: final text, tool log, usage, metadata
  runs/traces/<key>.json          the FULL trace -- every raw API response (reasoning summaries
                                  included), every request turn, and the sha of every image sent or
                                  returned
  runs/images/<sha256>.png        each distinct image, written once and referenced by hash from the
                                  trace, so a trace reconstructs exactly what the model saw

Traces are the point of the experiment: the ranking says which harness wins, the traces say why.

    python -m celltrack_bench.runner --model gpt-5.5 --split test
    python -m celltrack_bench.runner --model gpt-5.5 --split validation --conditions 1 2 3 4 --sample 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from celltrack_bench import conditions as C
from celltrack_bench import benchmark_data as B
from celltrack_bench import render as R

ROOT = R.REPO_ROOT

HARNESS_REVISION = "v10"  # v10: serialize reserved final answers without new hidden reasoning.
# v9 enforces the final-answer reserve on every tool-enabled turn.
# v8 makes annotation-defined branch cardinality explicit in every task.
# v4: max_output_tokens 4096 -> 16384 and per-turn truncation detection.
# v6 reserves --answer-reserve tokens for the final answer and withdraws tools before the budget is
# gone, because v5's fixed budget truncated exactly the conditions it should not have: c1 83%
# truncated / valid_json 0.167, c3 25%, against 0% for the tool-free c4 and c5. NOTE this makes the
# comparison a value-per-token one: under a fixed total budget a tool-using harness necessarily buys
# less exploration, which is a real property but not the same question as "best with unlimited budget".
#
# v5 caps TOTAL output tokens per item at --output-budget (8192) across every turn, tells the model
# that budget up front, and injects a warning turn once 80% is spent. A per-response cap cannot be
# warned about mid-generation, so the budget has to be cumulative for the warning to mean anything --
# and cumulative is also what actually bounds spend per item.
#
# v3 systematically penalised conditions 4 and 5: with 12-14 images, medium-effort reasoning consumed
# the whole 4096-token budget (out=4096, reasoning=4096) and the answer came back empty, so 2/4 and
# 3/4 of their calls failed to produce parseable JSON for a reason that had nothing to do with the
# harness being compared. Any run must therefore assert `truncated` is false before its numbers mean
# anything.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "data" / "hard",
        help="benchmark directory containing <split>_items.json and <split>/<task>/answer_key.jsonl "
             "(data/hard = the paper's hard set, data/full = the full benchmark)",
    )
    p.add_argument(
        "--benchmark-items-root",
        "--benchmark-item-root",
        dest="benchmark_items_root",
        type=Path,
        default=None,
        help="directory containing <item_id>/metadata_private.json; a split manifest may provide it",
    )
    p.add_argument(
        "--benchmark-id",
        default=None,
        help="human-readable dataset version; always combined with a content digest",
    )
    p.add_argument(
        "--source-data-root",
        type=Path,
        default=R.SOURCE_DATA_ROOT,
        help="root containing datasets/<dataset>/<sequence> in Cell Tracking Challenge layout "
             "(default $CELLTRACK_DATA_ROOT or <repo>/raw_data)",
    )
    p.add_argument("--tasks", nargs="+", default=list(C.TASKS))
    p.add_argument("--conditions", nargs="+", type=int, default=[1],
                   help="harness conditions (1-8); the frozen protocol is condition 1")
    p.add_argument("--model", required=True, help="model name as the endpoint expects it")
    p.add_argument("--reasoning-effort", default="low",
                   help="hosted backends only; 'none' disables thinking on the Anthropic backend")
    p.add_argument("--limit", type=int, default=0,
                   help="first N items. NOTE the split is dataset-ordered, so this is BF-C2DL-HSC only "
                        "-- that is how a path bug hid behind 177 healthy-looking calls. Prefer --sample.")
    p.add_argument("--sample", type=int, default=0,
                   help="N items spread across datasets, round-robin, deterministic. Use for pilots so "
                        "a failure specific to one dataset shows up immediately.")
    p.add_argument("--output-budget", type=int, default=8192,
                   help="TOTAL output tokens per item across all turns, reasoning included")
    p.add_argument("--results-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "cache")
    p.add_argument("--output", type=Path, default=None, help="exact results JSONL path")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=8,
                   help="parallel in-flight conversations. Raise until 429s appear, then back off.")
    p.add_argument("--answer-reserve", type=int, default=2048,
                   help="output tokens withheld for the final answer; tools are withdrawn once the "
                        "remaining budget falls to this, so a tool-heavy condition loses exploration "
                        "rather than returning a truncated, unparseable answer")
    p.add_argument(
        "--final-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
        help="reasoning effort after tools are withdrawn; 'none' preserves the reserved tokens for JSON",
    )
    p.add_argument("--backend", choices=("responses", "anthropic", "chat"), default="responses",
                   help="'responses' = hosted OpenAI Responses API; 'anthropic' = native Anthropic "
                        "Messages API; 'chat' = any OpenAI-compatible Chat Completions endpoint, "
                        "e.g. a local vLLM server")
    p.add_argument("--base-url", default=None, help="endpoint for --backend chat (vLLM /v1)")
    p.add_argument("--api-key", default="EMPTY", help="ignored by vLLM but required by the client")
    p.add_argument("--label", default=None,
                   help="filename tag replacing reasoning_effort for local models, e.g. 'q35-base'")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--chat-template-kwargs", default=None,
                   help='JSON passed through as extra_body.chat_template_kwargs, e.g. \'{"enable_thinking": true}\'')
    p.add_argument("--pid-file", default=None, help="optional file to write the runner pid to")
    p.add_argument("--dry-run", action="store_true", help="build prompts and images, make no API call")
    return p.parse_args()


def store_image(image, images_dir: Path) -> str:
    """Write an image once, keyed by content hash; return the hash."""
    ok, buf = __import__("cv2").imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encode failed")
    blob = buf.tobytes()
    sha = hashlib.sha256(blob).hexdigest()[:24]
    path = images_dir / f"{sha}.png"
    if not path.exists():
        path.write_bytes(blob)
    return sha


def _plain(value: Any) -> Any:
    """Best-effort JSON view of an SDK object, so raw responses survive in the trace."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(B.canonical_json(value)).hexdigest()[:length]


def harness_source_digest() -> str:
    """Bind response caches to prompt, renderer, tool, and request construction code."""
    digest = hashlib.sha256()
    for path in sorted((Path(__file__), Path(C.__file__), Path(R.__file__))):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def response_config(args: argparse.Namespace, benchmark: B.BenchmarkIdentity) -> dict[str, Any]:
    """Fields that can change one model response and therefore its cache identity."""
    return {
        "harness_revision": HARNESS_REVISION,
        "harness_source_digest": harness_source_digest(),
        "benchmark_identity": benchmark.value,
        "source_data_root": str(Path(args.source_data_root).expanduser().resolve()),
        "model": args.model,
        "label": args.label,
        "reasoning_effort": args.reasoning_effort,
        "output_budget": args.output_budget,
        "answer_reserve": args.answer_reserve,
        "final_reasoning_effort": args.final_reasoning_effort,
        "backend": args.backend,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "chat_template_kwargs": args.chat_template_kwargs,
        "tool_call_safety_cap": C.TOOL_CALL_SAFETY_CAP,
    }


def cache_key(
    item_id: str,
    task: str,
    condition: int,
    response_id: str,
) -> str:
    blob = f"{response_id}|{item_id}|{task}|{condition}"
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def run_config(
    args: argparse.Namespace,
    benchmark: B.BenchmarkIdentity,
    items_root: Path,
) -> dict[str, Any]:
    response = response_config(args, benchmark)
    response_id = _digest(response)
    core = {
        "schema_version": 1,
        "benchmark": {
            "identity": benchmark.value,
            "label": benchmark.label,
            "digest": benchmark.digest,
            "split": args.split,
            "split_dir": str(Path(args.split_dir).expanduser().resolve()),
            "items_root": str(Path(items_root).expanduser().resolve()),
            "source_data_root": response["source_data_root"],
        },
        "response_config": response,
        "response_id": response_id,
        "tasks": list(args.tasks),
        "conditions": list(args.conditions),
    }
    core["run_id"] = _digest(core, 12)
    return core


def default_output_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    benchmark = config["benchmark"]
    tag = B.safe_slug(args.label or args.reasoning_effort, fallback="run", max_length=32)
    model = B.safe_slug(args.model, fallback="model", max_length=48)
    name = (
        f"{args.split}_{benchmark['identity']}_{model}_{tag}_"
        f"{HARNESS_REVISION}_{config['run_id']}.jsonl"
    )
    return Path(args.results_dir) / name


def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses-style tool dicts -> Chat Completions' nested {"function": {...}} form."""
    out = []
    for t in tools:
        out.append({"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }})
    return out


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses-style tool definitions -> Anthropic Messages tool definitions."""
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }
        for tool in tools
    ]


def _anthropic_image_block(image) -> dict[str, Any]:
    data = C.png_data_url(image).split(",", 1)[1]
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _anthropic_effort(value: str) -> str:
    aliases = {"minimal": "low", "xhigh": "max"}
    effort = aliases.get(value, value)
    if effort not in {"low", "medium", "high", "max"}:
        raise ValueError(f"Anthropic backend does not support reasoning effort {value!r}")
    return effort


def _turn_policy(
    remaining: int,
    answer_reserve: int,
    tools_available: bool,
    force_final: bool,
) -> tuple[int, bool]:
    """Return the per-turn token cap and whether tools may be offered.

    Reserving tokens only by withdrawing tools on the *next* turn is too late: a
    reasoning model can consume the entire cumulative budget before emitting its
    first tool call. Keep the final-answer reserve outside every tool-enabled
    response cap, then release it when tools are withdrawn.
    """
    offer_tools = (
        tools_available
        and not force_final
        and remaining >= answer_reserve + 256
    )
    max_tokens = remaining - answer_reserve if offer_tools else remaining
    return max(1, max_tokens), offer_tools


def call_once_chat(client, built: C.Built, args: argparse.Namespace, images_dir: Path) -> dict[str, Any]:
    """Chat Completions equivalent of call_once, for locally served models (vLLM).

    Deliberately mirrors call_once turn for turn -- same cumulative budget, same answer reserve, same
    80% warning, same trace shape -- so a local model's numbers are comparable with the hosted ones.
    Only the wire format differs, and it differs in three ways that matter:

      * images ride in ``{"type": "image_url"}`` parts rather than ``input_image``
      * a ``tool`` role message may only carry text, so a returned crop is appended as a following
        ``user`` message (the same thing the Responses path does, for a different reason)
      * truncation is ``finish_reason == "length"`` rather than ``status == "incomplete"``
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": built.text}]
    prompt_image_shas = [store_image(image, images_dir) for image in built.images]
    for image in built.images:
        content.append({"type": "image_url", "image_url": {"url": C.png_data_url(image)}})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

    tool_calls = 0
    tool_log: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    text = ""
    truncated = False
    budget = int(args.output_budget)
    warned = False
    budget_exhausted = False
    force_final = False
    recovered_incomplete_turns = 0
    extra_body: dict[str, Any] = {}
    if args.chat_template_kwargs:
        extra_body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    trace: dict[str, Any] = {
        "prompt_text": built.text, "prompt_image_shas": prompt_image_shas,
        "tools_offered": [t["name"] for t in built.tools], "turns": [],
    }

    for _ in range(C.TOOL_CALL_SAFETY_CAP + 1):
        remaining = budget - usage["output_tokens"]
        if remaining <= 0:
            budget_exhausted = True
            break
        max_tokens, offer_tools = _turn_policy(
            remaining,
            args.answer_reserve,
            bool(built.tools) and tool_calls < C.TOOL_CALL_SAFETY_CAP,
            force_final,
        )
        request: dict[str, Any] = {
            "model": args.model, "messages": messages,
            "max_tokens": max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
        }
        if offer_tools:
            request["tools"] = _chat_tools(built.tools)
        if extra_body:
            request["extra_body"] = extra_body

        response = client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        finish = getattr(choice, "finish_reason", None)
        u = getattr(response, "usage", None)
        if u is not None:
            usage["input_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            usage["output_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            details = getattr(u, "completion_tokens_details", None)
            if details is not None:
                usage["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)

        calls = list(getattr(message, "tool_calls", None) or [])
        text = (getattr(message, "content", "") or "").strip()
        reasoning = getattr(message, "reasoning_content", None)
        trace["turns"].append({
            "raw_output": _plain(message), "usage": _plain(u), "status": finish,
            "incomplete_details": None, "output_text": text,
            "reasoning_content": reasoning,
            "output_tokens_cumulative": usage["output_tokens"],
            "budget_remaining": budget - usage["output_tokens"],
            "n_function_calls": len(calls),
        })
        if not calls and finish == "length" and offer_tools and budget - usage["output_tokens"] > 0:
            recovered_incomplete_turns += 1
            force_final = True
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({
                "role": "user",
                "content": "The bounded inspection turn has ended. Output the final JSON object now; no tools remain.",
            })
            continue
        if not calls:
            truncated = finish == "length"
            break

        messages.append({
            "role": "assistant", "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in calls],
        })
        # Every tool result must be emitted BEFORE any image, because a `tool` message following a
        # `user` message is invalid in strict chat templates -- Mistral rejects it outright with
        # "Unexpected role 'tool' after role 'user'". With two or more tool calls in one turn, the
        # naive interleaving (tool, image, tool) produces exactly that. So: all tool messages first,
        # then one user message carrying every crop returned this turn.
        pending_images = []
        for call in calls:
            if tool_calls >= C.TOOL_CALL_SAFETY_CAP:
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": "Runaway-loop guard reached. Answer now with the JSON object."})
                continue
            tool_calls += 1
            note, image = C.run_tool(built, call.function.name, call.function.arguments)
            sha = store_image(image, images_dir) if image is not None else None
            tool_log.append({"name": call.function.name, "arguments": call.function.arguments,
                             "ok": image is not None, "note": note, "returned_image_sha": sha})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": note})
            if image is not None:
                pending_images.append(image)
        if pending_images:
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": C.png_data_url(im)}} for im in pending_images]})

        if finish == "length":
            recovered_incomplete_turns += 1
            force_final = True
        if budget - usage["output_tokens"] <= args.answer_reserve:
            force_final = True
        if force_final:
            messages.append({
                "role": "user",
                "content": "Inspection is complete. Output the final JSON object now; no tools remain.",
            })

        used = usage["output_tokens"]
        if not warned and used >= 0.8 * budget:
            warned = True
            pct = max(0, round(100 * (budget - used) / budget))
            messages.append({"role": "user", "content": [{"type": "text", "text":
                f"You have used {used} of your {budget} output tokens; about {pct}% remains. "
                f"Stop inspecting and output the final JSON answer now."}]})
            trace["budget_warning_at"] = used

    return {"text": text, "tool_calls": tool_calls, "tool_log": tool_log, "usage": usage,
            "hit_safety_cap": tool_calls >= C.TOOL_CALL_SAFETY_CAP,
            "truncated": truncated, "budget_exhausted": budget_exhausted,
            "budget_warned": warned,
            "recovered_incomplete_turns": recovered_incomplete_turns,
            "trace": trace}


def call_once_anthropic(client, built: C.Built, args: argparse.Namespace, images_dir: Path) -> dict[str, Any]:
    """Native Anthropic Messages equivalent of the OpenAI Responses conversation."""
    content: list[dict[str, Any]] = [{"type": "text", "text": built.text}]
    prompt_image_shas = [store_image(image, images_dir) for image in built.images]
    content.extend(_anthropic_image_block(image) for image in built.images)
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

    tool_calls = 0
    tool_log: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    text = ""
    truncated = False
    budget = int(args.output_budget)
    warned = False
    budget_exhausted = False
    force_final = False
    recovered_incomplete_turns = 0
    # CC: effort "none" = plain request for models without adaptive thinking / output_config
    # (e.g. claude-haiku-4-5 rejects thinking type "adaptive"). Thinking is disabled on every turn
    # and no output_config is sent; all other efforts behave exactly as before.
    plain_mode = args.reasoning_effort == "none"
    base_effort = None if plain_mode else _anthropic_effort(args.reasoning_effort)
    trace: dict[str, Any] = {
        "prompt_text": built.text,
        "prompt_image_shas": prompt_image_shas,
        "tools_offered": [tool["name"] for tool in built.tools],
        "turns": [],
    }

    for _ in range(C.TOOL_CALL_SAFETY_CAP + 1):
        remaining = budget - usage["output_tokens"]
        if remaining <= 0:
            budget_exhausted = True
            break
        max_tokens, offer_tools = _turn_policy(
            remaining,
            args.answer_reserve,
            bool(built.tools) and tool_calls < C.TOOL_CALL_SAFETY_CAP,
            force_final,
        )
        turn_reasoning_effort = (
            args.final_reasoning_effort
            if built.tools and not offer_tools
            else args.reasoning_effort
        )
        request: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not plain_mode:  # CC: see note at base_effort
            # Opus 4.8 defaults to high effort, so low must be explicit on every turn.
            request["output_config"] = {
                "effort": (
                    base_effort
                    if turn_reasoning_effort == "none"
                    else _anthropic_effort(turn_reasoning_effort)
                )
            }
            request["thinking"] = {
                "type": "disabled" if turn_reasoning_effort == "none" else "adaptive"
            }
        if offer_tools:
            request["tools"] = _anthropic_tools(built.tools)

        response = client.messages.create(**request)
        blocks = list(getattr(response, "content", None) or [])
        stop_reason = getattr(response, "stop_reason", None)
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            usage["input_tokens"] += int(getattr(response_usage, "input_tokens", 0) or 0)
            usage["output_tokens"] += int(getattr(response_usage, "output_tokens", 0) or 0)

        calls = [block for block in blocks if getattr(block, "type", "") == "tool_use"]
        text = "".join(
            getattr(block, "text", "")
            for block in blocks
            if getattr(block, "type", "") == "text"
        ).strip()
        trace["turns"].append(
            {
                "raw_output": _plain(blocks),
                "usage": _plain(response_usage),
                "status": stop_reason,
                "incomplete_details": (
                    {"reason": "max_tokens"} if stop_reason == "max_tokens" else None
                ),
                "reasoning_effort": turn_reasoning_effort,
                "output_text": text,
                "output_tokens_cumulative": usage["output_tokens"],
                "budget_remaining": budget - usage["output_tokens"],
                "n_function_calls": len(calls),
                "n_thinking_blocks": sum(
                    getattr(block, "type", "") in {"thinking", "redacted_thinking"}
                    for block in blocks
                ),
            }
        )
        if (
            not calls
            and stop_reason == "max_tokens"
            and offer_tools
            and budget - usage["output_tokens"] > 0
        ):
            recovered_incomplete_turns += 1
            force_final = True
            messages.append({"role": "assistant", "content": blocks})
            messages.append(
                {
                    "role": "user",
                    "content": "The bounded inspection turn has ended. Output the final JSON object now; no tools remain.",
                }
            )
            continue
        if not calls:
            truncated = stop_reason == "max_tokens"
            break

        # Thinking blocks and signatures must be returned byte-for-byte around tool use.
        messages.append({"role": "assistant", "content": blocks})
        tool_results: list[dict[str, Any]] = []
        for call in calls:
            if tool_calls >= C.TOOL_CALL_SAFETY_CAP:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": "Runaway-loop guard reached. Answer now with the JSON object.",
                        "is_error": True,
                    }
                )
                continue
            tool_calls += 1
            arguments = getattr(call, "input", {}) or {}
            note, image = C.run_tool(built, call.name, arguments)
            sha = store_image(image, images_dir) if image is not None else None
            tool_log.append(
                {
                    "name": call.name,
                    "arguments": arguments,
                    "ok": image is not None,
                    "note": note,
                    "returned_image_sha": sha,
                }
            )
            result_content: list[dict[str, Any]] = [{"type": "text", "text": note}]
            if image is not None:
                result_content.append(_anthropic_image_block(image))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result_content,
                    "is_error": image is None,
                }
            )
        messages.append({"role": "user", "content": tool_results})

        if stop_reason == "max_tokens":
            recovered_incomplete_turns += 1
            force_final = True
        if budget - usage["output_tokens"] <= args.answer_reserve:
            force_final = True
        if force_final:
            messages.append(
                {
                    "role": "user",
                    "content": "Inspection is complete. Output the final JSON object now; no tools remain.",
                }
            )

        used = usage["output_tokens"]
        if not warned and used >= 0.8 * budget:
            warned = True
            pct = max(0, round(100 * (budget - used) / budget))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have used {used} of your {budget} output tokens; about {pct}% remains. "
                        "Stop inspecting and output the final JSON answer now."
                    ),
                }
            )
            trace["budget_warning_at"] = used

    return {
        "text": text,
        "tool_calls": tool_calls,
        "tool_log": tool_log,
        "usage": usage,
        "hit_safety_cap": tool_calls >= C.TOOL_CALL_SAFETY_CAP,
        "truncated": truncated,
        "budget_exhausted": budget_exhausted,
        "budget_warned": warned,
        "recovered_incomplete_turns": recovered_incomplete_turns,
        "trace": trace,
    }


def call_once(client, built: C.Built, args: argparse.Namespace, images_dir: Path) -> dict[str, Any]:
    """One conversation: initial message, then tool calls until the model answers or hits the cap."""
    content: list[dict[str, Any]] = [{"type": "input_text", "text": built.text}]
    prompt_image_shas = [store_image(image, images_dir) for image in built.images]
    for image in built.images:
        content.append({"type": "input_image", "image_url": C.png_data_url(image)})
    messages: list[Any] = [{"role": "user", "content": content}]

    tool_calls = 0
    tool_log: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    text = ""
    truncated = False
    budget = int(args.output_budget)
    warned = False
    budget_exhausted = False
    force_final = False
    recovered_incomplete_turns = 0
    trace: dict[str, Any] = {
        "prompt_text": built.text,
        "prompt_image_shas": prompt_image_shas,
        "tools_offered": [t["name"] for t in built.tools],
        "turns": [],
    }

    for _ in range(C.TOOL_CALL_SAFETY_CAP + 1):
        remaining = budget - usage["output_tokens"]
        if remaining <= 0:
            budget_exhausted = True
            break
        max_output_tokens, offer_tools = _turn_policy(
            remaining,
            args.answer_reserve,
            bool(built.tools) and tool_calls < C.TOOL_CALL_SAFETY_CAP,
            force_final,
        )
        request: dict[str, Any] = {
            "model": args.model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
        }
        # Withdraw tools while there is still an answer's worth of budget left. Without this, a
        # tool-heavy condition burns the whole budget on turns and its answer is truncated to
        # unparseable -- which is noise, not a real penalty. Measured at v5 without a reserve:
        # c1 truncated on 83% of calls (valid_json 0.167) while the tool-free c4/c5 truncated on 0%.
        if offer_tools:
            request["tools"] = built.tools
        turn_reasoning_effort = (
            args.final_reasoning_effort
            if built.tools and not offer_tools
            else args.reasoning_effort
        )
        if turn_reasoning_effort:
            # summary="auto" so the reasoning trace is recoverable, not just the token count
            request["reasoning"] = {
                "effort": turn_reasoning_effort,
                "summary": "auto",
            }

        response = client.responses.create(**request)
        status = getattr(response, "status", None)
        incomplete = _plain(getattr(response, "incomplete_details", None))
        turn: dict[str, Any] = {
            "raw_output": _plain(response.output), "usage": _plain(getattr(response, "usage", None)),
            "status": status, "incomplete_details": incomplete,
            "reasoning_effort": turn_reasoning_effort,
        }
        if getattr(response, "usage", None):
            u = response.usage
            usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
            usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
            details = getattr(u, "output_tokens_details", None)
            if details is not None:
                usage["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0)

        calls = [o for o in (response.output or []) if getattr(o, "type", "") == "function_call"]
        text = (getattr(response, "output_text", "") or "").strip()
        turn["output_text"] = text
        turn["output_tokens_cumulative"] = usage["output_tokens"]
        turn["budget_remaining"] = budget - usage["output_tokens"]
        turn["n_function_calls"] = len(calls)
        trace["turns"].append(turn)
        if not calls and status == "incomplete" and offer_tools and budget - usage["output_tokens"] > 0:
            recovered_incomplete_turns += 1
            force_final = True
            messages += list(response.output or [])
            messages.append({"role": "user", "content": [{"type": "input_text", "text":
                "The bounded inspection turn has ended. Output the final JSON object now; no tools remain."
            }]})
            continue
        if not calls:
            truncated = status == "incomplete"
            break

        messages += list(response.output)
        for call in calls:
            if tool_calls >= C.TOOL_CALL_SAFETY_CAP:
                messages.append(
                    {"type": "function_call_output", "call_id": call.call_id,
                     "output": "Runaway-loop guard reached. Answer now with the JSON object."}
                )
                continue
            tool_calls += 1
            note, image = C.run_tool(built, call.name, call.arguments)
            sha = store_image(image, images_dir) if image is not None else None
            tool_log.append({"name": call.name, "arguments": call.arguments, "ok": image is not None,
                             "note": note, "returned_image_sha": sha})
            messages.append({"type": "function_call_output", "call_id": call.call_id, "output": note})
            if image is not None:
                messages.append(
                    {"role": "user", "content": [{"type": "input_image", "image_url": C.png_data_url(image)}]}
                )

        if status == "incomplete":
            recovered_incomplete_turns += 1
            force_final = True
        if budget - usage["output_tokens"] <= args.answer_reserve:
            force_final = True
        if force_final:
            messages.append({"role": "user", "content": [{"type": "input_text", "text":
                "Inspection is complete. Output the final JSON object now; no tools remain."
            }]})

        used = usage["output_tokens"]
        if not warned and used >= 0.8 * budget:
            warned = True
            pct = max(0, round(100 * (budget - used) / budget))
            messages.append({"role": "user", "content": [{"type": "input_text", "text":
                f"You have used {used} of your {budget} output tokens; about {pct}% remains. "
                f"Stop inspecting and output the final JSON answer now."}]})
            trace["budget_warning_at"] = used
    return {"text": text, "tool_calls": tool_calls, "tool_log": tool_log, "usage": usage,
            "hit_safety_cap": tool_calls >= C.TOOL_CALL_SAFETY_CAP,
            # Only a terminal incomplete response is a failed call. A bounded inspection turn may
            # hit its cap and recover through the reserved final-answer turn above.
            "truncated": truncated, "budget_exhausted": budget_exhausted,
            "budget_warned": warned,
            "recovered_incomplete_turns": recovered_incomplete_turns,
            "trace": trace}


def main() -> None:
    args = parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(C.TASKS))
    unknown_conditions = sorted(set(args.conditions) - set(C.CONDITIONS))
    if unknown_tasks or unknown_conditions:
        raise SystemExit(
            f"invalid selection: unknown tasks={unknown_tasks}, conditions={unknown_conditions}"
        )
    if args.sample and args.limit:
        raise SystemExit("--sample and --limit are mutually exclusive")

    args.split_dir = args.split_dir.expanduser().resolve()
    args.source_data_root = args.source_data_root.expanduser().resolve()
    items_root = B.resolve_items_root(
        args.split_dir,
        args.split,
        args.benchmark_items_root,
        R.DEFAULT_ITEMS_ROOT,
    )
    try:
        benchmark = B.benchmark_identity(
            args.split_dir,
            items_root,
            args.split,
            label=args.benchmark_id,
        )
        all_items = B.load_split_items(args.split_dir, args.split)
        items = B.select_items(all_items, sample=args.sample, limit=args.limit)
    except B.BenchmarkDataError as exc:
        raise SystemExit(f"benchmark preflight failed: {exc}") from exc
    if not (args.source_data_root / "datasets").is_dir():
        raise SystemExit(
            f"source data not found: {args.source_data_root / 'datasets'} does not exist. "
            "Download the raw sequences (see README, 'Raw image data') and set CELLTRACK_DATA_ROOT "
            "or pass --source-data-root."
        )

    config = run_config(args, benchmark, items_root)
    response_id = config["response_id"]
    out_path = args.output.expanduser().resolve() if args.output else default_output_path(args, config)
    config_path = B.run_config_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise SystemExit(f"run configuration conflicts with existing {config_path}")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    if args.pid_file:
        Path(args.pid_file).write_text(f"{os.getpid()}\n")
        print(f"  runner pid {os.getpid()} -> {args.pid_file}")
    print(
        f"  benchmark {benchmark.value} | split {args.split} | "
        f"items {len(items)}/{len(all_items)}"
    )
    print(f"  manifests {args.split_dir}")
    print(f"  item metadata {items_root}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_path.parent
    images_dir = artifact_dir / "images"
    traces_dir = artifact_dir / "traces"
    images_dir.mkdir(exist_ok=True)
    traces_dir.mkdir(exist_ok=True)

    client = None
    if not args.dry_run:
        if args.backend == "chat":
            from openai import OpenAI

            if not args.base_url:
                raise SystemExit("--backend chat requires --base-url (e.g. http://127.0.0.1:8000/v1)")
            client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800, max_retries=0)
        elif args.backend == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(timeout=1800, max_retries=0)
        else:
            from openai import OpenAI

            client = OpenAI()

    done = set()
    if out_path.exists() and not args.dry_run:
        for line in out_path.read_text().splitlines():
            if line:
                row = json.loads(line)
                if row.get("run_id") != config["run_id"] or row.get("benchmark_identity") != benchmark.value:
                    raise SystemExit(f"results provenance conflicts with {out_path}")
                key = (row["item_id"], row["task"], row["condition"])
                if key in done:
                    raise SystemExit(f"duplicate result key {key} in {out_path}")
                done.add(key)

    work = [
        (item, task, condition)
        for item in items
        for task in args.tasks
        for condition in args.conditions
        if (item["item_id"], task, condition) not in done
    ]
    total = len(items) * len(args.tasks) * len(args.conditions)
    skipped = total - len(work)
    print(f"  {total} combinations | {skipped} already in results | {len(work)} to run "
          f"| concurrency {args.concurrency}")

    state = {"n": 0, "fresh": 0, "cached": 0, "failed": 0}
    write_lock = threading.Lock()   # serialises the JSONL append and the progress line
    started = time.time()
    sink = None if args.dry_run else out_path.open("a")

    def process(job: tuple[dict, str, int]) -> None:
        item, task, condition = job
        try:
            meta = R.load_meta(
                item["item_id"],
                items_root=items_root,
                source_data_root=args.source_data_root,
            )
            built = C.build(meta, task, condition, output_budget=args.output_budget)
        except Exception as exc:
            with write_lock:
                state["failed"] += 1
                state["n"] += 1
                print(f"  [{state['n']}/{len(work)}] BUILD FAIL {item['item_id']} {task} c{condition}: {exc}")
            return
        if args.dry_run:
            with write_lock:
                state["n"] += 1
                print(f"  [{state['n']}/{len(work)}] {item['item_id'][:34]:<34} {task:<17} c{condition} "
                      f"{len(built.images)} images, {len(built.text)} chars, {len(built.tools)} tools")
            return

        key = cache_key(item["item_id"], task, condition, response_id)
        cache_file = args.cache_dir / f"{key}.json"
        from_cache = cache_file.exists()
        if from_cache:
            result = json.loads(cache_file.read_text())
        else:
            result = None
            for attempt in range(args.retries):
                try:
                    if args.backend == "chat":
                        caller = call_once_chat
                    elif args.backend == "anthropic":
                        caller = call_once_anthropic
                    else:
                        caller = call_once
                    result = caller(client, built, args, images_dir)
                    break
                except Exception as exc:
                    wait = min(60, 2 ** attempt) + (attempt * 0.7)
                    with write_lock:
                        print(f"  retry {attempt + 1}/{args.retries} {item['item_id'][:22]} {task} "
                              f"c{condition} after {type(exc).__name__}: {str(exc)[:100]} (sleep {wait:.0f}s)")
                    time.sleep(wait)
            if result is None:
                with write_lock:
                    state["failed"] += 1
                    state["n"] += 1
                return
            cache_file.write_text(json.dumps(result) + "\n")

        trace = result.pop("trace", None)
        if trace is not None:
            (traces_dir / f"{key}.json").write_text(
                json.dumps({"item_id": item["item_id"], "task": task, "condition": condition,
                            "model": args.model, "backend": args.backend,
                            "reasoning_effort": args.reasoning_effort,
                            "final_reasoning_effort": args.final_reasoning_effort,
                            "harness_revision": HARNESS_REVISION,
                            "benchmark_identity": benchmark.value,
                            "run_id": config["run_id"], **trace}, indent=2) + "\n"
            )
        row = {
            "item_id": item["item_id"], "task": task, "condition": condition,
            "dataset": item["dataset"], "sequence": item["sequence"],
            "proxy_strata": item.get("proxy_strata"), "local_error_count": item.get("local_error_count"),
            "model": args.model, "backend": args.backend,
            "reasoning_effort": args.reasoning_effort,
            "final_reasoning_effort": args.final_reasoning_effort,
            "harness_revision": HARNESS_REVISION,
            "benchmark_identity": benchmark.value, "run_id": config["run_id"],
            "n_images": len(built.images), "trace_file": f"traces/{key}.json", **result,
        }
        with write_lock:
            assert sink is not None
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            state["n"] += 1
            state["cached" if from_cache else "fresh"] += 1
            rate = state["n"] / max(1e-6, time.time() - started)
            eta = (len(work) - state["n"]) / max(1e-6, rate) / 60
            print(f"  [{state['n']}/{len(work)}] {item['item_id'][:28]:<28} {task:<17} c{condition} "
                  f"tools={result['tool_calls']:<2} out={result['usage']['output_tokens']:<5} "
                  f"{'TRUNC ' if result.get('truncated') else ''}"
                  f"{'BUDGET ' if result.get('budget_exhausted') else ''}"
                  f"{'warn ' if result.get('budget_warned') else ''}| {rate * 60:.1f}/min eta {eta:.0f}m")

    if args.dry_run:
        for job in work:
            process(job)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(process, work))
    if sink is not None:
        sink.close()
    n, fresh, cached, failed = state["n"], state["fresh"], state["cached"], state["failed"]

    print(f"\n  {n} combinations | fresh {fresh} | cached {cached} | failed {failed} "
          f"| {time.time() - started:.0f}s")
    print(f"  -> {'dry-run only; planned output ' if args.dry_run else ''}{out_path}")
    print(f"  -> {config_path}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
