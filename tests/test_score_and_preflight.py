from __future__ import annotations

import json

import pytest

from celltrack_bench.preflight import check_result_rows
from celltrack_bench.output_schema import validate_output
from celltrack_bench.score import (
    dataset_macro,
    division_summary,
    health_report,
    overall_score,
    repaired_graph_metrics,
    select_harness,
    strict_division_metrics,
    strict_json_object,
    strict_linking_metrics,
)


def _metadata() -> dict:
    return {
        "item_id": "item-1",
        "hoct_edges": [{"source": "S1", "target": "T1", "semantic": "continuation"}],
        "gt_edges": [
            {"source": "S1", "target": "T1", "semantic": "division"},
            {"source": "S1", "target": "T2", "semantic": "division"},
        ],
    }


def test_repaired_graph_f1_applies_complete_semantic_operations() -> None:
    operations = {
        "operations": [
            {
                "op": "change_semantic",
                "source": "S1",
                "target": "T1",
                "from_semantic": "continuation",
                "to_semantic": "division",
            },
            {"op": "add", "source": "S1", "target": "T2", "semantic": "division"},
        ]
    }
    perfect = repaired_graph_metrics(operations, _metadata())
    assert perfect["repaired_graph_semantic_f1"] == 1.0
    assert perfect["repaired_graph_exact_match"] is True

    wrong_operation = {
        "operations": [{"op": "add", "source": "S1", "target": "T1", "semantic": "division"}]
    }
    wrong = repaired_graph_metrics(wrong_operation, _metadata())
    assert wrong["repaired_graph_semantic_f1"] < 1.0
    assert wrong["repaired_graph_exact_match"] is False


def test_division_event_requires_both_explicitly_labelled_daughters() -> None:
    answer = {"links": _metadata()["gt_edges"]}
    assert strict_division_metrics(answer, answer)["division_event_f1"] == 1.0
    one_daughter = {"links": [answer["links"][0]]}
    assert strict_division_metrics(one_daughter, answer)["division_event_recall"] == 0.0
    no_semantics = {
        "links": [{"source": edge["source"], "target": edge["target"]} for edge in answer["links"]]
    }
    assert strict_division_metrics(no_semantics, answer)["division_event_f1"] == 0.0


def test_nonbinary_lineage_branches_are_scored_and_separated_from_binary_divisions() -> None:
    answer = {
        "links": [
            {"source": "S1", "target": target, "semantic": "division"}
            for target in ("T1", "T2", "T3")
        ]
    }
    perfect = strict_division_metrics(answer, answer)
    assert perfect["division_event_f1"] == 1.0
    assert perfect["division_event_target"] == 1
    assert perfect["binary_division_event_target"] == 0


def test_linking_reports_semantic_class_edge_f1() -> None:
    answer = {
        "links": _metadata()["gt_edges"]
        + [{"source": "S2", "target": "T3", "semantic": "continuation"}],
        "unlinked_sources": [],
    }
    one_daughter = {"links": [answer["links"][0]], "unlinked_sources": []}
    scored = strict_linking_metrics(one_daughter, answer)
    assert scored["division_edge_precision"] == 1.0
    assert scored["division_edge_recall"] == 0.5
    assert scored["continuation_edge_f1"] == 0.0


def test_linking_primary_does_not_infer_missing_semantics() -> None:
    answer = {
        "links": _metadata()["gt_edges"],
        "unlinked_sources": ["S2"],
    }
    perfect = strict_linking_metrics(answer, answer)
    assert perfect["strict_linking_semantic_f1"] == 1.0
    assert perfect["strict_linking_exact_match"] is True

    missing_semantics = {
        "links": [
            {"source": edge["source"], "target": edge["target"]}
            for edge in answer["links"]
        ],
        "unlinked_sources": ["S2"],
    }
    scored = strict_linking_metrics(missing_semantics, answer)
    assert scored["strict_linking_endpoint_f1"] == 1.0
    assert scored["strict_linking_semantic_f1"] == 0.0
    assert scored["strict_linking_exact_match"] is False


def test_strict_output_schema_requires_semantics_but_scores_candidate_violations() -> None:
    meta = {"candidate_map": {"S1": ["T1", "T2"]}}
    assert validate_output(
        "cell_linking",
        {
            "links": [{"source": "S1", "target": "T1", "semantic": "continuation"}],
            "unlinked_sources": [],
        },
        meta,
    ) == []
    errors = validate_output(
        "cell_linking",
        {"links": [{"source": "S1", "target": "T3"}], "unlinked_sources": []},
        meta,
    )
    assert any("semantic must be" in error for error in errors)
    assert validate_output(
        "cell_linking",
        {
            "links": [{"source": "S1", "target": "T3", "semantic": "continuation"}],
            "unlinked_sources": [],
        },
        meta,
    ) == []


def test_strict_output_schema_checks_repair_and_inspection_fields() -> None:
    repair_errors = validate_output(
        "track_repair",
        {
            "operations": [
                {
                    "op": "change_semantic",
                    "source": "S1",
                    "target": "T1",
                    "to_semantic": "division",
                }
            ]
        },
    )
    assert any("requires valid from_semantic" in error for error in repair_errors)

    inspection_errors = validate_output(
        "track_inspection",
        {
            "links_to_inspect": [
                {
                    "source": "S1",
                    "target": "T1",
                    "hoct_status": "unselected",
                    "issue": "spurious",
                }
            ]
        },
    )
    assert any("unselected links must have issue=missing" in error for error in inspection_errors)


def test_strict_json_is_separate_from_extractable_json() -> None:
    assert strict_json_object('{"links": []}')
    assert not strict_json_object('answer: {"links": []}')
    assert not strict_json_object("{'links': []}")


def test_dataset_macro_prevents_large_dataset_domination() -> None:
    rows = []
    for index in range(9):
        rows.append(
            {"item_id": f"a{index}", "dataset": "large", "task": "cell_linking", "condition": 1, "strict_linking_semantic_f1": 1.0}
        )
    rows.append(
        {"item_id": "b0", "dataset": "small", "task": "cell_linking", "condition": 1, "strict_linking_semantic_f1": 0.0}
    )
    value, per_dataset = dataset_macro(rows, 1, "cell_linking", "paper")
    assert per_dataset == {"large": 1.0, "small": 0.0}
    assert value == 0.5


def test_overall_is_unweighted_across_tasks() -> None:
    rows = [
        {"item_id": "i", "dataset": "d", "task": "cell_linking", "condition": 1, "strict_linking_semantic_f1": 1.0},
        {
            "item_id": "i",
            "dataset": "d",
            "task": "track_repair",
            "condition": 1,
            "primary_f1": 1.0,
            "repaired_graph_semantic_f1": 0.0,
        },
        {"item_id": "i", "dataset": "d", "task": "track_inspection", "condition": 1, "primary_f1": 0.5},
    ]
    assert overall_score(rows, 1, ("cell_linking", "track_repair", "track_inspection"), "paper") == 0.5


def test_smoke_health_rejects_truncation_and_non_strict_json() -> None:
    item = {"item_id": "item-1", "dataset": "demo", "sequence": "01"}
    answer = {"links": [], "unlinked_sources": []}
    base = {
        "item_id": "item-1",
        "task": "cell_linking",
        "condition": 1,
        "dataset": "demo",
        "text": json.dumps(answer),
        "usage": {},
        "truncated": False,
        "budget_exhausted": False,
        "hit_safety_cap": False,
        "benchmark_identity": "bench-123",
        "run_id": "run-1",
        "tool_log": [],
    }
    answers = {("item-1", "cell_linking"): answer}
    assert check_result_rows([base], [item], answers, ["cell_linking"], [1], "bench-123") == []

    bad = dict(base, truncated=True, text=f"answer: {json.dumps(answer)}")
    errors = check_result_rows([bad], [item], answers, ["cell_linking"], [1], "bench-123")
    assert any("truncated=true" in error for error in errors)
    assert any("strict JSON" in error for error in errors)


def test_full_health_separates_model_format_errors_from_infrastructure() -> None:
    scored = [
        {
            "item_id": "item-1",
            "task": "cell_linking",
            "condition": 1,
            "strict_json": 0.0,
            "valid_json": 0.0,
            "strict_schema_valid": 0.0,
            "truncated": 0.0,
            "recovered_incomplete_turns": 1,
            "budget_exhausted": 0.0,
            "hit_safety_cap": 0.0,
            "tool_failures": 0,
        }
    ]
    raw = [{"benchmark_identity": "bench"}]
    health = health_report(
        scored,
        raw,
        [],
        [{"item_id": "item-1"}],
        ("cell_linking",),
        (1,),
        "bench",
    )
    assert health["passed"] is True
    assert health["clean_output"] is False
    assert health["recovered_incomplete_turns"] == 1


def test_division_macro_uses_fixed_ground_truth_dataset_denominator() -> None:
    rows = [
        {
            "dataset": "with-events",
            "task": "cell_linking",
            "condition": 1,
            "division_event_tp": 0,
            "division_event_fp": 0,
            "division_event_fn": 1,
        },
        {
            "dataset": "without-events",
            "task": "cell_linking",
            "condition": 1,
            "division_event_tp": 0,
            "division_event_fp": 1,
            "division_event_fn": 0,
        },
    ]
    summary = division_summary(rows, 1)
    assert summary["datasets"] == 1
    assert summary["dataset_macro_f1"] == 0.0
    assert summary["false_positives_in_datasets_without_events"] == 1


def test_harness_selection_uses_cost_only_among_unresolved_eligible_conditions() -> None:
    rows = []
    for condition, input_tokens in ((1, 100), (2, 50), (7, 1)):
        rows.append(
            {
                "item_id": "item",
                "dataset": "dataset",
                "task": "cell_linking",
                "condition": condition,
                "score": 0.5,
                "input_tokens": input_tokens,
                "tool_calls": 0,
            }
        )
    selection = select_harness(
        rows,
        (1, 2, 7),
        ("cell_linking",),
        "score",
        bootstrap=100,
        seed=1,
        ineligible={7},
    )
    assert selection["point_best_condition"] == 1
    assert selection["statistically_unresolved_contenders"] == [1, 2]
    assert selection["selected_condition"] == 2
