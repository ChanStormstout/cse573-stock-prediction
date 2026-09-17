"""Past-only calibration, no-news policies and global paragraph selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from common import dump, jsonl, logit, metrics, sha, sigmoid

VARIANTS = ("P0", "P1", "P2", "P3")
METHODS = ("raw", "platt", "temperature")


def fit_calibrator(y, p, method):
    y = np.asarray(y, dtype=int)
    x = logit(p)
    if len(y) < 20 or len(set(y)) < 2:
        return {"method": "raw", "reason": "insufficient labels"}
    if method == "raw":
        return {"method": "raw"}
    if method == "platt":
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=3000, random_state=573).fit(x[:, None], y)
        return {"method": "platt", "coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
    if method == "temperature":
        def loss(log_temperature):
            q = sigmoid(x / np.exp(log_temperature))
            return float(-np.mean(y * np.log(np.clip(q, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1))))
        result = minimize_scalar(loss, bounds=(-4, 4), method="bounded", options={"xatol": 1e-9})
        if not result.success:
            raise RuntimeError("temperature optimization failed")
        return {"method": "temperature", "temperature": float(np.exp(result.x)), "objective": float(result.fun)}
    raise ValueError(method)


def apply_calibrator(p, fitted):
    if fitted["method"] == "raw":
        return np.asarray(p, dtype=float)
    x = logit(p)
    if fitted["method"] == "platt":
        return sigmoid(fitted["coef"] * x + fitted["intercept"])
    return sigmoid(x / fitted["temperature"])


def cell_score(frame, column):
    cells = []
    for (_, _), group in frame.groupby(["symbol", "month"]):
        score = metrics(group.label, group[column])
        if score["BA"] is not None:
            cells.append(score)
    return {
        "cells": len(cells),
        "mean_BA": float(np.mean([row["BA"] for row in cells])),
        "mean_Brier": float(np.mean([row["Brier"] for row in cells])),
        "mean_ECE10": float(np.mean([row["ECE10"] for row in cells])),
    }


def main(paragraphs: Path, llm: Path, price: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    paragraph_manifest = json.loads((paragraphs / "manifest.json").read_text())
    llm_summary = json.loads((llm / "summary.json").read_text())
    if llm_summary["status"] != "COMPLETE" or llm_summary["labels_loaded"]:
        raise ValueError("LLM inference not complete or label-free contract failed")
    labels = pd.read_pickle(paragraphs / "labels.pkl").copy()
    labels["start_utc"] = pd.to_datetime(labels.start_utc, utc=True)
    labels["end_utc"] = pd.to_datetime(labels.end_utc, utc=True)
    labels["cutoff_utc"] = pd.to_datetime(labels.cutoff_utc, utc=True)
    labels["month"] = labels.start_utc.dt.strftime("%Y-%m")
    scores = labels.copy()
    input_rows = {
        variant: {row["key"]: row for row in jsonl(paragraphs / f"{variant}.jsonl")}
        for variant in VARIANTS
    }
    for variant in VARIANTS:
        rows = jsonl(llm / f"{variant}.jsonl")
        if len(rows) != 1607 or sha(llm / f"{variant}.jsonl") != llm_summary["variants"][variant]["output_sha256"]:
            raise ValueError(f"LLM score seal mismatch: {variant}")
        mapping = {row["key"]: row["p"] for row in rows}
        scores[f"{variant}_observed"] = scores.key.map(mapping)
    # Batch composition can change quantized token logits even for a byte-for-byte
    # identical prompt. Keep every observed score for audit, but for each controlled
    # comparison carry forward the preceding variant's normalized score whenever
    # the complete input is identical. Thus P0->P1 measures content selection,
    # P1->P2 measures context expansion and P2->P3 measures deduplication rather
    # than a different batch composition. This rule uses no outcome label.
    scores["P0_raw"] = scores.P0_observed
    same_prompt_normalization = {"P0": {"reference": None, "rows": 1607, "changed_observed_scores": 0, "max_absolute_observed_difference": 0.0}}
    for variant, reference in (("P1", "P0"), ("P2", "P1"), ("P3", "P2")):
        scores[f"{variant}_raw"] = scores[f"{variant}_observed"]
        same = scores.key.map(lambda key: input_rows[variant][key] == input_rows[reference][key]).to_numpy()
        difference = np.abs(scores.loc[same, f"{variant}_observed"].to_numpy() - scores.loc[same, f"{reference}_observed"].to_numpy())
        scores.loc[same, f"{variant}_raw"] = scores.loc[same, f"{reference}_raw"]
        same_prompt_normalization[variant] = {
            "reference": reference,
            "rows": int(same.sum()),
            "changed_observed_scores": int((difference > 0).sum()),
            "max_absolute_observed_difference": float(difference.max()) if len(difference) else 0.0,
        }
    if scores.filter(like="_raw").isna().any().any():
        raise ValueError("missing LLM score")
    price_predictions = pd.read_csv(price / "predictions.csv")
    price_map = price_predictions.set_index("key")["R1"]
    scores["R1"] = scores.key.map(price_map)
    if scores.loc[scores.month >= "2018-03", "R1"].isna().any():
        raise ValueError("missing R1 score")

    calibration_records, fit_records = [], []
    for variant in VARIANTS:
        raw_col = f"{variant}_raw"
        for method in METHODS:
            target = f"{variant}_{method}"
            if method == "raw":
                continue
            scores[target] = np.nan
            for symbol in ("AAPL", "AMZN"):
                stock = scores.symbol.eq(symbol)
                for month in pd.period_range("2018-03", "2018-08", freq="M").astype(str):
                    train = scores[stock & (scores.month < month)]
                    evaluate = scores[stock & scores.month.eq(month)]
                    if train.end_utc.max() >= evaluate.cutoff_utc.min():
                        raise AssertionError("calibration time leakage")
                    fitted = fit_calibrator(train.label, train[raw_col], method)
                    scores.loc[evaluate.index, target] = apply_calibrator(evaluate[raw_col], fitted)
                    fit_records.append({"variant": variant, "requested_method": method, "symbol": symbol, "period": month, "train_n": len(train), "train_label_end_max": train.end_utc.max().isoformat(), "eval_cutoff_min": evaluate.cutoff_utc.min().isoformat(), **fitted})
                train = scores[stock & (scores.month < "2018-09")]
                evaluate = scores[stock & (scores.month >= "2018-09")]
                if train.end_utc.max() >= evaluate.cutoff_utc.min():
                    raise AssertionError("frozen calibration leakage")
                fitted = fit_calibrator(train.label, train[raw_col], method)
                scores.loc[evaluate.index, target] = apply_calibrator(evaluate[raw_col], fitted)
                fit_records.append({"variant": variant, "requested_method": method, "symbol": symbol, "period": "frozen", "train_n": len(train), "train_label_end_max": train.end_utc.max().isoformat(), "eval_cutoff_min": evaluate.cutoff_utc.min().isoformat(), **fitted})

    train_oof = scores[scores.month.between("2018-03", "2018-08")].copy()
    selection = {}
    for variant in VARIANTS:
        candidates = []
        for method in METHODS:
            result = {"method": method, **cell_score(train_oof, f"{variant}_{method}")}
            candidates.append(result)
            calibration_records.append({"variant": variant, "scope": "selection_oof", **result})
        chosen = sorted(candidates, key=lambda row: (row["mean_Brier"], -row["mean_BA"], METHODS.index(row["method"])))[0]
        selection[variant] = chosen
        scores[f"{variant}_cal"] = scores[f"{variant}_{chosen['method']}"]

    # Compare three no-news policies without dropping any window.
    for variant in VARIANTS:
        scores[f"{variant}_policy_raw"] = scores[f"{variant}_cal"]
        no_news = scores.has_news.eq(0)
        scores.loc[no_news, f"{variant}_policy_raw"] = scores.loc[no_news, f"{variant}_raw"]
        scores[f"{variant}_policy_price"] = scores[f"{variant}_cal"]
        scores.loc[no_news, f"{variant}_policy_price"] = scores.loc[no_news, "R1"]
        scores[f"{variant}_policy_learned"] = scores[f"{variant}_cal"]
        for symbol in ("AAPL", "AMZN"):
            stock = scores.symbol.eq(symbol)
            for month in list(pd.period_range("2018-03", "2018-08", freq="M").astype(str)) + ["frozen"]:
                eval_mask = stock & (scores.month.eq(month) if month != "frozen" else scores.month.ge("2018-09")) & no_news
                past_mask = stock & no_news & (scores.month.lt(month) if month != "frozen" else scores.month.lt("2018-09"))
                train = scores[past_mask]
                evaluate = scores[eval_mask]
                enough = len(train) >= 30 and train.label.nunique() == 2
                if enough:
                    fitted = fit_calibrator(train.label, train[f"{variant}_raw"], "platt")
                    values = apply_calibrator(evaluate[f"{variant}_raw"], fitted)
                    action = "no_news_platt"
                else:
                    fitted = {"method": "R1_fallback"}
                    values = evaluate.R1.to_numpy()
                    action = "R1_fallback"
                scores.loc[evaluate.index, f"{variant}_policy_learned"] = values
                fit_records.append({"variant": variant, "requested_method": "no_news_learned", "symbol": symbol, "period": month, "train_no_news_n": len(train), "action": action, **fitted})

    for variant in VARIANTS:
        for policy in ("policy_raw", "policy_price", "policy_learned"):
            calibration_records.append({"variant": variant, "scope": "no_news_policy_oof", "method": policy, **cell_score(train_oof.assign(**{f"{variant}_{policy}": scores.loc[train_oof.index, f"{variant}_{policy}"]}), f"{variant}_{policy}")})

    # Paragraph selection is one shared rule for both stocks and uses strict R1 fallback.
    for variant in VARIANTS:
        scores[f"{variant}_strict"] = scores[f"{variant}_cal"]
        scores.loc[scores.has_news.eq(0), f"{variant}_strict"] = scores.loc[scores.has_news.eq(0), "R1"]
    train_oof = scores[scores.month.between("2018-03", "2018-08")].copy()
    base = cell_score(train_oof, "P0_strict")
    paragraph_candidates = []
    for variant in VARIANTS:
        result = {"variant": variant, **cell_score(train_oof, f"{variant}_strict")}
        result["eligible"] = result["mean_Brier"] <= base["mean_Brier"] + 0.002
        paragraph_candidates.append(result)
    eligible = [row for row in paragraph_candidates if row["eligible"]]
    paragraph_choice = sorted(eligible, key=lambda row: (-row["mean_BA"], row["mean_Brier"], VARIANTS.index(row["variant"])))[0]
    selection["paragraph"] = {"baseline": base, "candidates": paragraph_candidates, "chosen": paragraph_choice}
    scores["paragraph_selected"] = scores[f"{paragraph_choice['variant']}_strict"]

    phase = np.where(scores.month.between("2018-03", "2018-08"), "outer", np.where(scores.month >= "2018-09", "frozen", "calibration_train"))
    scores["phase"] = phase
    scores.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(out / "calibration_metrics.csv", index=False)
    dump(out / "calibration_fits.json", fit_records)
    dump(out / "selection.json", selection)
    metric_rows = []
    columns = [f"{variant}_{suffix}" for variant in VARIANTS for suffix in ("raw", "cal", "policy_raw", "policy_price", "policy_learned", "strict")]
    columns += ["R1", "paragraph_selected"]
    for (symbol, split), group in scores[scores.month >= "2018-09"].groupby(["symbol", "split"]):
        for column in columns:
            metric_rows.append({"symbol": symbol, "period": split, "method": column, **metrics(group.label, group[column])})
        for month, sample in group.groupby("month"):
            for column in columns:
                metric_rows.append({"symbol": symbol, "period": month, "method": column, **metrics(sample.label, sample[column])})
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)

    # Same-window P0 -> P1/P2/P3 transitions. These descriptive rows are kept
    # separate from the training-only paragraph-selection rule above.
    transition_rows = []
    later = scores[scores.month >= "2018-09"]
    for (symbol, split), group in later.groupby(["symbol", "split"]):
        periods = [(split, group)] + [(month, sample) for month, sample in group.groupby("month")]
        for period, sample in periods:
            y = sample.label.to_numpy(dtype=int)
            baseline_probability = sample.P0_strict.to_numpy(dtype=float)
            baseline_direction = baseline_probability >= 0.5
            baseline_correct = baseline_direction == y
            baseline_score = metrics(y, baseline_probability)
            for variant in ("P1", "P2", "P3"):
                probability = sample[f"{variant}_strict"].to_numpy(dtype=float)
                direction = probability >= 0.5
                correct = direction == y
                score = metrics(y, probability)
                transition_rows.append(
                    {
                        "symbol": symbol,
                        "period": period,
                        "variant": variant,
                        "n": len(sample),
                        "direction_changed": int((direction != baseline_direction).sum()),
                        "P0_wrong_new_right": int((~baseline_correct & correct).sum()),
                        "P0_right_new_wrong": int((baseline_correct & ~correct).sum()),
                        "both_right": int((baseline_correct & correct).sum()),
                        "both_wrong": int((~baseline_correct & ~correct).sum()),
                        "mean_absolute_probability_change": float(np.mean(np.abs(probability - baseline_probability))),
                        "P0_BA": baseline_score["BA"],
                        "new_BA": score["BA"],
                        "BA_difference": None if score["BA"] is None or baseline_score["BA"] is None else score["BA"] - baseline_score["BA"],
                        "P0_Brier": baseline_score["Brier"],
                        "new_Brier": score["Brier"],
                        "Brier_difference": score["Brier"] - baseline_score["Brier"],
                    }
                )
    pd.DataFrame(transition_rows).to_csv(out / "paragraph_transitions.csv", index=False)
    dump(out / "status.json", {"status": "COMPLETE", "rows": len(scores), "calibration_selection_period": "2018-03 through 2018-08 forward predictions", "paragraph_choice": paragraph_choice, "same_prompt_score_normalization": same_prompt_normalization, "same_window_transition_rows": len(transition_rows), "all_later_periods_exposed": True, "source_hashes": {str(paragraphs / "manifest.json"): sha(paragraphs / "manifest.json"), str(llm / "summary.json"): sha(llm / "summary.json"), str(price / "status.json"): sha(price / "status.json")}})
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraphs", required=True, type=Path)
    parser.add_argument("--llm", required=True, type=Path)
    parser.add_argument("--price", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.paragraphs, args.llm, args.price, args.out)
