"""Global low-capacity fusion using only training-period OOF probabilities."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import INTEGRATED, dump, metrics, sha


def mean_cells(frame, column):
    values = []
    for (_, _), group in frame.groupby(["symbol", "month"]):
        score = metrics(group.label, group[column])
        if score["BA"] is not None:
            values.append(score)
    return {
        "cells": len(values),
        "mean_BA": float(np.mean([row["BA"] for row in values])),
        "mean_Brier": float(np.mean([row["Brier"] for row in values])),
        "mean_ECE10": float(np.mean([row["ECE10"] for row in values])),
    }


def grids(size):
    for integers in itertools.product(range(5), repeat=size):
        if sum(integers) == 4:
            yield np.asarray(integers, dtype=float) / 4


def choose_weights(training, components, baseline):
    candidates = []
    values = training[components].to_numpy()
    for weights in grids(len(components)):
        probability = values @ weights
        scratch = training.assign(candidate=probability)
        score = mean_cells(scratch, "candidate")
        row = {
            "components": components,
            "weights": weights.tolist(),
            **score,
            "eligible": score["mean_Brier"] <= baseline["mean_Brier"] + 0.002,
        }
        candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return None, candidates
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["mean_BA"],
            row["mean_Brier"],
            sum(value > 0 for value in row["weights"]),
            row["weights"],
        ),
    )[0]
    return selected, candidates


def assemble(old, price, llm, phase):
    base = old[old.phase.eq(phase)].copy() if "phase" in old else old.copy()
    columns = ["key", "symbol", "day", "month", "label", "has_news", "split", "title", "body", "semantic"]
    base = base[columns]
    p = price[price.phase.eq(phase)][["key", "R0", "R1", "R2", "S2", "S3"]]
    l = llm[llm.phase.eq(phase)][["key", "paragraph_selected"]]
    result = base.merge(p, on="key", validate="one_to_one").merge(l, on="key", validate="one_to_one")
    if len(result) != len(base):
        raise AssertionError("fusion alignment dropped rows")
    result = result.rename(columns={"paragraph_selected": "llm"})
    for component in ("title", "body", "semantic"):
        result.loc[result.has_news.eq(0), component] = result.loc[result.has_news.eq(0), "R1"]
    if not np.allclose(result.loc[result.has_news.eq(0), "llm"], result.loc[result.has_news.eq(0), "R1"]):
        raise AssertionError("LLM strict fallback failed")
    return result


def main(price_dir: Path, calibration_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    old_oof = pd.read_csv(INTEGRATED / "runs/v1/oof.csv")
    old_oof["phase"] = "outer"
    old_eval = pd.read_csv(INTEGRATED / "runs/v1/predictions.csv")
    price = pd.read_csv(price_dir / "predictions.csv")
    llm = pd.read_csv(calibration_dir / "predictions.csv")
    training = assemble(old_oof, price, llm, "outer")
    evaluation = assemble(old_eval, price, llm, "frozen")
    if len(training) != 765 or len(evaluation) != 609:
        raise AssertionError("wrong fusion row count")
    training["F0"] = training.title
    training["F1"] = training.body
    training["F2"] = training.semantic
    evaluation["F0"] = evaluation.title
    evaluation["F1"] = evaluation.body
    evaluation["F2"] = evaluation.semantic
    baseline = mean_cells(training, "F0")
    definitions = {
        "F3": ["body", "semantic"],
        "F4": ["body", "llm"],
        "F5": ["body", "semantic", "llm"],
        "F6": ["body", "semantic", "llm", "R1"],
    }
    selection, all_candidates = {}, []
    for name, components in definitions.items():
        chosen, candidates = choose_weights(training, components, baseline)
        for row in candidates:
            all_candidates.append({"method": name, **row})
        if chosen is None:
            training[name] = training.F0
            evaluation[name] = evaluation.F0
            selection[name] = {"action": "fallback_F0", "reason": "no weight met Brier guardrail", "components": components}
        else:
            weights = np.asarray(chosen["weights"])
            training[name] = training[components].to_numpy() @ weights
            evaluation[name] = evaluation[components].to_numpy() @ weights
            selection[name] = {"action": "selected", **chosen}
    selection["F7"] = {"action": "NOT_RUN", "reason": "No qualified cutoff-aligned market minute data"}
    for frame in (training, evaluation):
        no_news = frame.has_news.eq(0)
        for method in ("F0", "F1", "F2", "F3", "F4", "F5", "F6"):
            if not np.allclose(frame.loc[no_news, method], frame.loc[no_news, "R1"]):
                raise AssertionError(f"strict no-news fallback failed: {method}")
    training["phase"] = "outer"
    evaluation["phase"] = "frozen"
    predictions = pd.concat([training, evaluation], ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(all_candidates).to_csv(out / "weight_grid.csv", index=False)
    dump(out / "selection.json", {"F0_training_baseline": baseline, "methods": selection, "selection_scope": "combined stocks, March-August forward OOF cells"})
    rows = []
    methods = ["R0", "R1", "R2", "S2", "S3", "title", "body", "semantic", "llm", "F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    for (phase, symbol), group in predictions.groupby(["phase", "symbol"]):
        periods = [("all", group)] + [(month, sample) for month, sample in group.groupby("month")]
        if phase == "frozen":
            periods += [(split, sample) for split, sample in group.groupby("split")]
        for period, sample in periods:
            for method in methods:
                rows.append({"phase": phase, "symbol": symbol, "period": period, "method": method, **metrics(sample.label, sample[method])})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    transitions = []
    for (phase, symbol), group in predictions.groupby(["phase", "symbol"]):
        baseline_correct = (group.F0.ge(0.5).astype(int) == group.label)
        for method in ("F1", "F2", "F3", "F4", "F5", "F6"):
            current = group[method].ge(0.5).astype(int) == group.label
            transitions.append({"phase": phase, "symbol": symbol, "method": method, "baseline_wrong_new_right": int((~baseline_correct & current).sum()), "baseline_right_new_wrong": int((baseline_correct & ~current).sum()), "both_right": int((baseline_correct & current).sum()), "both_wrong": int((~baseline_correct & ~current).sum())})
    pd.DataFrame(transitions).to_csv(out / "transitions.csv", index=False)
    dump(out / "status.json", {"status": "COMPLETE", "training_rows": len(training), "evaluation_rows": len(evaluation), "all_evaluation_periods_exposed": True, "market_F7": "NOT_RUN", "strict_no_news_fallback": True, "source_hashes": {str(INTEGRATED / "runs/v1/oof.csv"): sha(INTEGRATED / "runs/v1/oof.csv"), str(INTEGRATED / "runs/v1/predictions.csv"): sha(INTEGRATED / "runs/v1/predictions.csv"), str(price_dir / "predictions.csv"): sha(price_dir / "predictions.csv"), str(calibration_dir / "predictions.csv"): sha(calibration_dir / "predictions.csv")}})
    print(json.dumps(selection, indent=2))
    print(pd.DataFrame(rows).query("phase == 'frozen' and period in ['validation','test'] and method in ['F0','F1','F2','F3','F4','F5','F6']")[['symbol','period','method','BA','Brier']].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.price, args.calibration, args.out)
