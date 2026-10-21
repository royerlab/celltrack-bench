#!/usr/bin/env python3
"""Recompute the paper's tables from the released predictions and compare them with the paper.

Uses only files in this repository (predictions/, data/hard/); no API calls and no raw images.
Every recomputed cell is rounded the way the paper prints it and compared with the printed value
in reproduction/paper_values.json; mismatches are listed at the end.

  python -m celltrack_bench.reproduce                # all tables
  python -m celltrack_bench.reproduce --tables table2 harness
  python -m celltrack_bench.reproduce --json out.json --strict
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from celltrack_bench import benchmark_data as B
from celltrack_bench import leaderboard as L
from celltrack_bench import nn_residual as NR
from celltrack_bench import render as R
from celltrack_bench import score as SC
from celltrack_bench import subgraph_lnk as SL

ROOT = R.REPO_ROOT
TABLES = ("table2", "harness", "repair", "triage", "strata", "robustness", "spearman")


class Checker:
    """Collects (table, row, column, ours, paper) comparisons at the paper's printed precision."""

    def __init__(self) -> None:
        self.checked = 0
        self.mismatches: list[tuple[str, str, str, float, float]] = []

    def __call__(self, table: str, row: str, col: str, ours: float | None, paper: Any, digits: int = 3) -> str:
        if not isinstance(paper, (int, float)) or ours is None:
            return "   "
        self.checked += 1
        if round(ours, digits) == round(float(paper), digits) or abs(ours - float(paper)) <= 0.5 * 10 ** -digits + 1e-9:
            return " ok"
        self.mismatches.append((table, row, col, ours, float(paper)))
        return " !!"


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "---"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def load_registry() -> dict[str, Any]:
    return json.loads((ROOT / "predictions" / "systems.json").read_text())


def files_of(entry: dict[str, Any]) -> list[Path]:
    return [ROOT / f for f in entry["files"]]


# ------------------------------------------------------------------------------------ Table 2

def table2(ctx: L.Context, registry: dict, paper: dict, check: Checker) -> dict[str, Any]:
    print("\n=== Table 2: frozen-c1 hard-test leaderboard ===")
    systems = {}
    for entry in registry["splits"]["test"]:
        systems[entry["name"]] = L.system_metrics(ctx, files_of(entry))
        systems[entry["name"]]["display"] = entry["display"]
        systems[entry["name"]]["access"] = entry["access"]
    refs = L.reference_metrics(ctx)
    expected = paper["table2"]
    order = sorted(systems, key=lambda n: (-systems[n]["lnk_sub"], -systems[n]["repair_d_lnk"]))
    header = f"{'System':<26}{'Acc.':<8}" + "".join(f"{label:>13}" for _k, label, _f in L.COLUMNS)
    print(header + "\n" + "-" * len(header))
    for name in order:
        m, exp = systems[name], expected["systems"].get(name, {})
        cells = []
        for key, _label, fmt in L.COLUMNS:
            mark = check("table2", name, key, m[key], exp.get(key))
            cells.append(f"{fmt.format(m[key]):>10}{mark}")
        print(f"{m['display']:<26}{m['access']:<8}" + "".join(cells))
    for ref, values in refs.items():
        exp = expected["references"].get(ref, {})
        f1 = values["link_f1"]
        mark_f1 = check("table2", ref, "link_f1", f1, exp.get("link_f1"))
        mark_lnk = check("table2", ref, "lnk_sub", values["lnk_sub"], exp.get("lnk_sub"))
        print(f"[{ref}]".ljust(34) + f"{_fmt(f1):>10}{mark_f1}" + " " * 13 * 5 + f"{_fmt(values['lnk_sub']):>10}{mark_lnk}")
    return {"systems": systems, "references": refs}


# ------------------------------------------------------------------------------------ harness selection

def harness(ctx_val: L.Context, registry: dict, paper: dict, check: Checker) -> dict[str, Any]:
    print("\n=== Harness selection (validation, gpt-5.5, c1-c8) and Table 1 NN-residual F1 ===")
    by_name = {e["name"]: e for e in registry["splits"]["validation"]}
    sweep = files_of(by_name["gpt-5.5"])[0]
    raw = B.read_jsonl(sweep)
    scored, errors = SC.score_rows(raw, ctx_val.answers, ctx_val.items, ctx_val.items_root)
    assert not errors, errors[:3]
    conditions = tuple(range(1, 9))
    table = {}
    for c in conditions:
        row = {task: SC.dataset_macro(scored, c, task, "paper")[0] for task in L.TASKS}
        row["mean"] = statistics.fmean(row[t] for t in L.TASKS)
        table[f"c{c}"] = row
    selection = SC.select_harness(scored, conditions, L.TASKS, "paper", 5000, 20260730, ineligible={7})

    def nn_res(path: Path, condition: int) -> tuple[float, float, int]:
        per, _rev, _health = NR.per_item(path, condition, ctx_val.truth_nr, ctx_val.nn, ctx_val.residual)
        vals = [v["nn_resid"] for v in per.values() if v.get("nn_resid") is not None]
        return statistics.fmean(vals), statistics.stdev(vals) / len(vals) ** 0.5, len(vals)

    opus = files_of(by_name["claude-opus-4.8"])[0]
    print(f"{'cond':<6}{'Linking':>11}{'Repair':>11}{'Inspect':>11}{'Mean':>11}"
          f"{'NN-res gpt-5.5':>22}{'NN-res opus-4.8':>22}")
    nr_out: dict[str, dict[str, list[float]]] = {"gpt-5.5": {}, "claude-opus-4.8": {}}
    for c in conditions:
        key, row, exp = f"c{c}", table[f"c{c}"], paper["harness_validation"][f"c{c}"]
        cells = "".join(
            f"{_fmt(row[t]):>8}{check('harness', key, t, row[t], exp[t])}"
            for t in (*L.TASKS, "mean")
        )
        nr_cells = ""
        for model, path in (("gpt-5.5", sweep), ("claude-opus-4.8", opus)):
            exp_nr = paper["table1_nn_residual"][model].get(key)
            if model == "claude-opus-4.8" and exp_nr is None:
                nr_cells += f"{'---':>22}"
                continue
            mean, se, n = nn_res(path, c)
            nr_out[model][key] = [mean, se, n]
            m1 = check("table1", f"{model} {key}", "nn_residual_f1", mean, exp_nr[0] if exp_nr else None)
            m2 = check("table1", f"{model} {key}", "nn_residual_se", se, exp_nr[1] if exp_nr else None)
            nr_cells += f"{mean:>9.3f}{m1} +-{se:.3f}{m2}"
        print(f"{key:<6}{cells}{nr_cells}")
    comp = selection["comparisons_to_point_best"][8]
    exp = paper["harness_selection"]
    print(f"selected c{selection['selected_condition']} (point best c{selection['point_best_condition']}); "
          f"c1 - c8 = {comp['best_minus_condition']:.3f}{check('harness', 'c1-c8', 'difference', comp['best_minus_condition'], exp['difference'])} "
          f"95% CI [{comp['ci95'][0]:.3f}{check('harness', 'c1-c8', 'ci_lo', comp['ci95'][0], exp['ci95'][0])}, "
          f"{comp['ci95'][1]:.3f}{check('harness', 'c1-c8', 'ci_hi', comp['ci95'][1], exp['ci95'][1])}]; "
          f"NN-residual items: {nr_out['gpt-5.5']['c1'][2]}")
    return {"scores": table, "selection": selection, "nn_residual": nr_out}


# ------------------------------------------------------------------------------------ appendix

def repair(t2: dict, paper: dict, check: Checker) -> None:
    print("\n=== Appendix: track-repair decomposition ===")
    print(f"{'System':<26}{'Repair F1':>12}{'dLNK':>12}{'Add P/R':>16}{'Delete P/R':>16}{'Better':>10}{'Worse':>10}")
    systems = t2["systems"]
    for name in sorted(systems, key=lambda n: -systems[n]["repair_d_lnk"]):
        m, exp = systems[name], paper["repair_breakdown"].get(name, {})
        c = lambda key, digits=2: check("repair", name, key, m[key], exp.get(key), digits)  # noqa: E731
        print(f"{m['display']:<26}{m['track_repair_f1']:>9.3f}{c('track_repair_f1', 3)}"
              f"{m['repair_d_lnk']:>+9.3f}{c('repair_d_lnk', 3)}"
              f"{m['repair_add_precision']:>8.2f}{c('repair_add_precision')}/{m['repair_add_recall']:.2f}{c('repair_add_recall')}"
              f"{m['repair_del_precision']:>8.2f}{c('repair_del_precision')}/{m['repair_del_recall']:.2f}{c('repair_del_recall')}"
              f"{100 * m['repair_better']:>6.0f}%{c('repair_better')}{100 * m['repair_worse']:>6.0f}%{c('repair_worse')}")


def triage(ctx: L.Context, t2: dict, paper: dict, check: Checker) -> dict[str, Any]:
    print("\n=== Appendix: inspection under fixed review budgets ===")
    refs = L.triage_references(ctx)
    exp = paper["triage_budget"]
    keys = ("inspect_d_lnk_budget10", "inspect_d_lnk_budget25", "inspect_d_lnk_budget50", "inspect_d_lnk")
    print(f"{'System':<26}{'Inspect F1':>12}{'Flags':>9}{'@10%':>12}{'@25%':>12}{'@50%':>12}{'All flags':>12}")
    gold = refs["gold"]
    gold_vals = [gold["budget10"], gold["budget25"], gold["budget50"], gold["full"]]
    print(f"{'Gold-flag oracle':<26}{'--':>12}{sum(len(v) for v in ctx.gold_flags.values()):>9}" + "".join(
        f"{v:>+9.3f}{check('triage', 'gold', k, v, exp['gold-flag'].get(k))}" for k, v in zip(keys, gold_vals)))
    systems = t2["systems"]
    for name in sorted(systems, key=lambda n: -systems[n]["inspect_d_lnk_budget10"]):
        m, e = systems[name], exp["systems"].get(name, {})
        print(f"{m['display']:<26}{m['track_inspection_f1']:>9.3f}{check('triage', name, 'track_inspection_f1', m['track_inspection_f1'], e.get('track_inspection_f1'))}"
              f"{m['inspect_flags']:>9.0f}" + "".join(
                  f"{m[k]:>+9.3f}{check('triage', name, k, m[k], e.get(k))}" for k in keys))
    rand = refs["random"]
    print(f"{'Random reviewer':<26}{'--':>12}{'--':>9}" + "".join(
        f"{rand[k.replace('inspect_d_lnk_', '')]:>+9.3f}{check('triage', 'random', k, rand[k.replace('inspect_d_lnk_', '')], exp['random'].get(k))}"
        for k in keys[:3]))
    return refs


def strata(ctx: L.Context, registry: dict, paper: dict, check: Checker) -> None:
    print("\n=== Appendix: per-neighborhood linking LNK_sub by difficulty stratum (dataset-macro) ===")
    entries = {e["name"]: e for e in registry["splits"]["test"]}
    values = {name: L.per_item_linking_lnk(ctx, files_of(entries[name])) for name in ("gpt-5.6-sol", "gemma-4-31b")}
    values["nn-baseline"] = {i: SL.item_lnk(ctx.nn_per[i]) or 0.0 for i in ctx.truth}
    metas = ctx.metas
    hardness = {i: metas[i]["selection"]["hardness"] for i in ctx.truth}
    hs = sorted(hardness.values())
    cuts = [hs[int(len(hs) * k / 5)] for k in range(1, 5)]
    quint = {i: 1 + sum(hardness[i] >= c for c in cuts) for i in ctx.truth}
    nn_solved = {i for i in ctx.truth if ctx.nn_per[i]["cost"] == 0}
    divisions = {i for i in ctx.truth if any(e["semantic"] == "division" for e in metas[i]["gt_edges"])}
    proxy = {i: metas[i]["proxy"]["type"] for i in ctx.truth}
    groups = {
        "NN-solved": [i for i in ctx.truth if i in nn_solved],
        "NN-resistant": [i for i in ctx.truth if i not in nn_solved],
        **{f"Hardness Q{k}" + (" (easiest)" if k == 1 else " (hardest)" if k == 5 else ""):
           [i for i in ctx.truth if quint[i] == k] for k in range(1, 6)},
        "Proxy EA (missing)": [i for i in ctx.truth if proxy[i] == "EA"],
        "Proxy EC (semantic)": [i for i in ctx.truth if proxy[i] == "EC"],
        "Proxy ED (spurious)": [i for i in ctx.truth if proxy[i] == "ED"],
        "Division-bearing": [i for i in ctx.truth if i in divisions],
        "No division": [i for i in ctx.truth if i not in divisions],
    }
    print(f"{'Stratum':<24}{'n':>6}{'NN baseline':>15}{'GPT-5.6 (sol)':>15}{'Gemma 4':>15}")
    for label, items in groups.items():
        exp = paper["difficulty_strata"].get(label, {})
        cells = ""
        for name in ("nn-baseline", "gpt-5.6-sol", "gemma-4-31b"):
            by_ds: dict[str, list[float]] = {}
            for i in items:
                by_ds.setdefault(ctx.dataset_of[i], []).append(values[name][i])
            macro = statistics.fmean(statistics.fmean(v) for v in by_ds.values())
            cells += f"{macro:>12.3f}{check('strata', label, name, macro, exp.get(name))}"
        n_mark = check("strata", label, "n", len(items), exp.get("n"), 0)
        print(f"{label:<24}{len(items):>3}{n_mark}{cells}")


def robustness(ctx_val: L.Context, ctx_test: L.Context, registry: dict, t2: dict, triage_test: dict,
               paper: dict, check: Checker) -> None:
    print("\n=== Appendix: validation-test robustness for GPT-5.6 (terra) ===")
    val_entry = next(e for e in registry["splits"]["validation"] if e["name"] == "gpt-5.6-terra")
    rows = {
        "validation": (ctx_val, L.system_metrics(ctx_val, files_of(val_entry)), L.triage_references(ctx_val)),
        "test": (ctx_test, t2["systems"]["gpt-5.6-terra"], triage_test),
    }
    keys = ("nn_lnk", "hoct_lnk", "lnk_sub", "d_nn", "primary_mean", "track_repair_f1", "repair_d_lnk",
            "inspect_d_lnk_budget10", "gold_budget10")
    print(f"{'Split':<12}" + "".join(f"{k:>16}" for k in keys))
    for split, (ctx, m, refs) in rows.items():
        values = {"nn_lnk": ctx.nn_lnk, "hoct_lnk": ctx.prop_lnk, "gold_budget10": refs["gold"]["budget10"],
                  **{k: m[k] for k in keys if k in m}}
        exp = paper["validation_robustness"][split]
        print(f"{split:<12}" + "".join(f"{values[k]:>13.3f}{check('robustness', split, k, values[k], exp.get(k))}" for k in keys))


def spearman(t2: dict, paper: dict, check: Checker) -> dict[str, float]:
    print("\n=== Task relationships: system-level Spearman over the Table 2 systems ===")
    systems = list(t2["systems"].values())
    cols = {
        "link": [m["lnk_sub"] for m in systems],
        "repair": [m["repair_d_lnk"] for m in systems],
        "inspect": [m["inspect_d_lnk"] for m in systems],
    }
    out = {
        "link_repair": SL.spearman(cols["link"], cols["repair"]),
        "link_inspect": SL.spearman(cols["link"], cols["inspect"]),
        "repair_inspect": SL.spearman(cols["repair"], cols["inspect"]),
    }
    for key, value in out.items():
        print(f"  {key:<16} rho = {value:.3f}{check('spearman', key, 'rho', value, paper['spearman'][key], 2)}"
              f"   (paper {paper['spearman'][key]:.2f}, n = {len(systems)} systems)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    p.add_argument("--json", type=Path, default=None, help="write every recomputed number as JSON")
    p.add_argument("--strict", action="store_true", help="exit 1 if any cell differs from the paper")
    args = p.parse_args()

    registry = load_registry()
    paper = json.loads((ROOT / "reproduction" / "paper_values.json").read_text())
    bench = ROOT / registry["benchmark"]
    check = Checker()
    out: dict[str, Any] = {}
    want = set(args.tables)
    try:
        ctx_test = L.load_context(bench, "test")
    except B.BenchmarkDataError as exc:
        raise SystemExit(str(exc)) from exc
    needs_t2 = want & {"table2", "repair", "triage", "robustness", "spearman"}
    if needs_t2:
        out["table2"] = t2 = table2(ctx_test, registry, paper, check)
    if want & {"harness", "robustness"}:
        ctx_val = L.load_context(bench, "validation")
    if "harness" in want:
        out["harness"] = harness(ctx_val, registry, paper, check)
    if "repair" in want:
        repair(t2, paper, check)
    triage_test = None
    if want & {"triage", "robustness"}:
        triage_test = triage(ctx_test, t2, paper, check) if "triage" in want else L.triage_references(ctx_test)
    if "strata" in want:
        strata(ctx_test, registry, paper, check)
    if "robustness" in want:
        robustness(ctx_val, ctx_test, registry, t2, triage_test, paper, check)
    if "spearman" in want:
        out["spearman"] = spearman(t2, paper, check)

    print(f"\n{check.checked} cells compared with the paper, {len(check.mismatches)} differ:")
    for table, row, col, ours, theirs in check.mismatches:
        print(f"  {table:<11} {row:<26} {col:<28} recomputed {ours:+.4f}   paper {theirs:+.4f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 1 if args.strict and check.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
