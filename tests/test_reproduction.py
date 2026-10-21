"""The released predictions must reproduce the original analysis scripts' unrounded outputs."""

from __future__ import annotations

import json
import math

import pytest

from celltrack_bench import benchmark_data as B
from celltrack_bench import leaderboard as L
from celltrack_bench import reproduce as RP
from celltrack_bench import score as SC
from conftest import requires_benchmark

pytestmark = requires_benchmark("hard")

ORIGINAL = json.loads((RP.ROOT / "tests" / "fixtures" / "original_outputs.json").read_text())
REGISTRY = RP.load_registry()
TEST_SYSTEMS = {entry["name"]: entry for entry in REGISTRY["splits"]["test"]}


@pytest.fixture(scope="module")
def ctx_test() -> L.Context:
    return L.load_context(RP.ROOT / "data" / "hard", "test")


def _close(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9


def test_reference_rows(ctx_test: L.Context) -> None:
    refs = L.reference_metrics(ctx_test)
    assert round(refs["nearest_centroid"]["lnk_sub"], 3) == 0.493
    assert {k: round(v["lnk_sub"], 3) for k, v in refs.items() if k != "nearest_centroid"} == {
        "hungarian": 0.492, "trackastra": 0.484, "hoct": 0.385}
    assert {k: round(v["link_f1"], 3) for k, v in refs.items() if k != "nearest_centroid"} == {
        "hungarian": 0.681, "trackastra": 0.698, "hoct": 0.622}


@pytest.mark.parametrize("name", sorted(TEST_SYSTEMS))
def test_system_matches_original_scripts(ctx_test: L.Context, name: str) -> None:
    ours = L.system_metrics(ctx_test, RP.files_of(TEST_SYSTEMS[name]))
    expected = ORIGINAL["test"][name]
    diffs = {k: (ours[k], v) for k, v in expected.items() if not _close(float(ours[k]), float(v))}
    assert not diffs


def test_gold_triage(ctx_test: L.Context) -> None:
    gold = L.triage_references(ctx_test, draws=1)["gold"]
    assert all(_close(gold[k], v) for k, v in ORIGINAL["test_triage_gold"].items())


def test_harness_selection_matches_original_scoring() -> None:
    ctx = L.load_context(RP.ROOT / "data" / "hard", "validation")
    sweep = next(e for e in REGISTRY["splits"]["validation"] if e["name"] == "gpt-5.5")
    scored, errors = SC.score_rows(B.read_jsonl(RP.files_of(sweep)[0]), ctx.answers, ctx.items, ctx.items_root)
    assert not errors
    expected = ORIGINAL["validation_harness"]
    for task, by_condition in expected["task_dataset_macro"].items():
        for condition, value in by_condition.items():
            assert _close(SC.dataset_macro(scored, int(condition), task, "paper")[0], value)
    selection = SC.select_harness(scored, tuple(range(1, 9)), L.TASKS, "paper", 5000, 20260730, ineligible={7})
    assert selection["selected_condition"] == 1
    c8 = selection["comparisons_to_point_best"][8]
    assert _close(c8["best_minus_condition"], expected["c8_vs_best"]["best_minus_condition"])
    assert all(_close(a, b) for a, b in zip(c8["ci95"], expected["c8_vs_best"]["ci95"]))
