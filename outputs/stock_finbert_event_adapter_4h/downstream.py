"""Strict gated residual corrections for the fixed four-hour task."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import NEXTGEN, PARAGRAPHS, PUBLIC, SEEDS, classification_metrics, dump, jsonl, logit, paired_block_interval, sha, sigmoid
from events import FEATURES, aggregate_events

PARAMETERS = ("C0_shared", "C1_independent", "C2_partial_shared")
CS = (0.01, 0.1, 1.0)


def load_facts(path: Path):
    rows = jsonl(path)
    return rows, {(row["symbol"], row["record_key"]): row for row in rows}


def window_features(fact_lookup: dict, windows: dict) -> pd.DataFrame:
    rows = []
    for key, window in windows.items():
        articles = []
        identities = []
        for article in window["news"]:
            matched = [fact_lookup[(window["symbol"], member)] for member in article.get("cluster_members", [article["record_key"]]) if (window["symbol"], member) in fact_lookup]
            if not matched:
                continue
            events, seen = [], set()
            for item in matched:
                for event in item["events"]:
                    signature = (event["kind"], event["action"], event.get("old"), event.get("new"))
                    if signature not in seen:
                        seen.add(signature)
                        events.append(event)
            if events:
                members = sorted(article.get("cluster_members", [article["record_key"]]))
                articles.append(
                    {
                        "record_key": article["record_key"],
                        "available_at": article["available_at"],
                        "event_group": "|".join(sorted(article.get("cluster_members", [article["record_key"]]))),
                        "reports": len(article.get("cluster_members", [article["record_key"]])),
                        "events": events,
                    }
                )
                for event in events:
                    identities.append(
                        {
                            "members": members,
                            "signature": [event.get("kind"), event.get("action"), event.get("old"), event.get("new")],
                        }
                    )
        vector, gate, details = aggregate_events(articles, pd.Timestamp(window["cutoff"]))
        rows.append(
            {
                "key": key,
                "event_gate": gate,
                "unique_events": details["unique_groups"],
                "event_identities": json.dumps(identities, sort_keys=True),
                **{f"z_{name}": float(value) for name, value in zip(FEATURES, vector)},
            }
        )
    return pd.DataFrame(rows)


def count_unique_events(frame: pd.DataFrame, symbol: str) -> int:
    """Count event components while merging overlapping causal report groups."""
    by_signature = collections.defaultdict(list)

    def normalize(value):
        if value is None:
            return None
        try:
            return ("number", float(str(value).replace(",", "")))
        except ValueError:
            return ("text", " ".join(str(value).lower().replace("-", " ").split()))

    sample = frame[frame.symbol.eq(symbol)]
    for raw in sample.event_identities:
        for item in json.loads(raw):
            kind, action, old, new = item["signature"]
            signature = (kind, action, normalize(old), normalize(new))
            members = set(item["members"])
            merged = members
            keep = []
            for previous in by_signature[signature]:
                if merged & previous:
                    merged |= previous
                else:
                    keep.append(previous)
            # A newly enlarged component can bridge more than one prior group.
            changed = True
            while changed:
                changed = False
                remainder = []
                for previous in keep:
                    if merged & previous:
                        merged |= previous
                        changed = True
                    else:
                        remainder.append(previous)
                keep = remainder
            by_signature[signature] = [*keep, merged]
    return sum(len(groups) for groups in by_signature.values())


def eligibility_counts(frame: pd.DataFrame) -> dict:
    unique = {symbol: count_unique_events(frame, symbol) for symbol in ("AAPL", "AMZN")}
    event_windows = frame.groupby("symbol").event_gate.sum().astype(int).to_dict()
    eligible = {
        symbol: unique[symbol] >= 30 and event_windows.get(symbol, 0) >= 60
        for symbol in ("AAPL", "AMZN")
    }
    return {"unique_event_groups": unique, "event_windows": event_windows, "independent_eligible": eligible}


def prepare_design(train: pd.DataFrame, evaluation: pd.DataFrame, parameterization: str):
    columns = [f"z_{name}" for name in FEATURES]
    active = train.event_gate.eq(1)
    if not active.any():
        return None
    mean = train.loc[active, columns].mean().to_numpy(float)
    scale = train.loc[active, columns].std(ddof=0).replace(0, 1).to_numpy(float)
    x_train = (train[columns].to_numpy(float) - mean) / scale
    x_eval = (evaluation[columns].to_numpy(float) - mean) / scale
    if parameterization == "C2_partial_shared":
        train_amzn = train.symbol.eq("AMZN").to_numpy(float)[:, None]
        eval_amzn = evaluation.symbol.eq("AMZN").to_numpy(float)[:, None]
        # One strongly shrunk stock-specific event-window offset.  Event
        # effects themselves remain shared, which is materially safer than a
        # second 16-coefficient AMZN model under sparse supervision.
        x_train = np.concatenate([x_train, 0.25 * train_amzn], axis=1)
        x_eval = np.concatenate([x_eval, 0.25 * eval_amzn], axis=1)
    return x_train, x_eval, mean, scale


def fit_beta(x, y, offset, c):
    x, y, offset = np.asarray(x, float), np.asarray(y, float), np.asarray(offset, float)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return None

    def objective(beta):
        score = offset + x @ beta
        loss = np.mean(np.logaddexp(0, score) - y * score) + np.sum(beta * beta) / (2 * c * len(y))
        gradient = x.T @ (sigmoid(score) - y) / len(y) + beta / (c * len(y))
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True)
    return result.x if result.success else None


def apply_fit(train: pd.DataFrame, evaluation: pd.DataFrame, parameterization: str, c: float, independent_eligible: dict):
    probability = evaluation.F1.to_numpy(float).copy()
    models = {}
    if parameterization == "C1_independent":
        for symbol in ("AAPL", "AMZN"):
            if not independent_eligible[symbol]:
                models[symbol] = {"status": "disabled_minimum_counts"}
                continue
            train_symbol = train[train.symbol.eq(symbol)].copy()
            eval_mask = evaluation.symbol.eq(symbol).to_numpy()
            eval_symbol = evaluation.loc[eval_mask].copy()
            design = prepare_design(train_symbol, eval_symbol, "C0_shared")
            if design is None:
                models[symbol] = {"status": "no_events"}
                continue
            x_train, x_eval, mean, scale = design
            active_train = train_symbol.event_gate.to_numpy(bool)
            beta = fit_beta(x_train[active_train], train_symbol.loc[active_train, "label"], logit(train_symbol.loc[active_train, "F1"]), c)
            if beta is None:
                models[symbol] = {"status": "single_class_or_fit_failure"}
                continue
            active_eval = eval_symbol.event_gate.to_numpy(bool)
            local = eval_symbol.F1.to_numpy(float).copy()
            local[active_eval] = sigmoid(logit(local[active_eval]) + x_eval[active_eval] @ beta)
            probability[eval_mask] = local
            models[symbol] = {"status": "fit", "beta": beta.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}
        return probability, models
    design = prepare_design(train, evaluation, parameterization)
    if design is None:
        return probability, {"status": "no_events"}
    x_train, x_eval, mean, scale = design
    active_train = train.event_gate.to_numpy(bool)
    beta = fit_beta(x_train[active_train], train.loc[active_train, "label"], logit(train.loc[active_train, "F1"]), c)
    if beta is None:
        return probability, {"status": "single_class_or_fit_failure"}
    active_eval = evaluation.event_gate.to_numpy(bool)
    probability[active_eval] = sigmoid(logit(probability[active_eval]) + x_eval[active_eval] @ beta)
    return probability, {"status": "fit", "beta": beta.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}


def metric_groups(frame: pd.DataFrame, probability) -> list[dict]:
    scratch = frame.assign(candidate=np.asarray(probability, float))
    rows = []
    for (month, symbol), group in scratch.groupby(["month", "symbol"]):
        metric = classification_metrics(group.label, group.candidate)
        rows.append({"month": month, "symbol": symbol, **metric})
    return rows


def forward_candidate(frame: pd.DataFrame, parameterization: str, c: float):
    output = frame.F1.to_numpy(float).copy()
    fits = []
    months = sorted(frame.month.unique())
    for month in months:
        train = frame[frame.month.lt(month)]
        evaluation = frame[frame.month.eq(month)]
        if train.empty:
            fits.append({"month": month, "status": "base_no_prior_month"})
            continue
        current_counts = eligibility_counts(train)
        probability, model = apply_fit(train, evaluation, parameterization, c, current_counts["independent_eligible"])
        output[evaluation.index.to_numpy()] = probability
        fits.append({"month": month, "model": model, "counts_available_before_month": current_counts})
    return output, fits


def choose_residual(frame: pd.DataFrame):
    counts = eligibility_counts(frame)
    base_monthly = metric_groups(frame, frame.F1)
    base_ba = float(np.mean([row["BA"] for row in base_monthly if row["BA"] is not None]))
    base_brier = float(np.mean([row["Brier"] for row in base_monthly]))
    candidates = [{"parameterization": "BASE", "C": None, "mean_monthly_BA": base_ba, "mean_monthly_Brier": base_brier, "delta_Brier": 0.0, "eligible": True}]
    prediction_lookup = {("BASE", None): frame.F1.to_numpy(float)}
    fit_lookup = {}
    for parameterization in PARAMETERS:
        for c in CS:
            probability, fits = forward_candidate(frame, parameterization, c)
            monthly = metric_groups(frame, probability)
            ba = float(np.mean([row["BA"] for row in monthly if row["BA"] is not None]))
            brier = float(np.mean([row["Brier"] for row in monthly]))
            row = {"parameterization": parameterization, "C": c, "mean_monthly_BA": ba, "mean_monthly_Brier": brier, "delta_Brier": brier - base_brier, "eligible": brier - base_brier <= 0.002}
            candidates.append(row)
            prediction_lookup[(parameterization, c)] = probability
            fit_lookup[(parameterization, c)] = fits
    capacity = {"BASE": 0, "C0_shared": 1, "C2_partial_shared": 2, "C1_independent": 3}
    selected = sorted([row for row in candidates if row["eligible"]], key=lambda row: (-row["mean_monthly_BA"], row["mean_monthly_Brier"], capacity[row["parameterization"]], -1 if row["C"] is None else row["C"]))[0]
    key = (selected["parameterization"], selected["C"])
    return selected, candidates, prediction_lookup[key], fit_lookup.get(key, []), counts


def final_probability(train: pd.DataFrame, evaluation: pd.DataFrame, selected: dict, eligibility: dict):
    if selected["parameterization"] == "BASE":
        return evaluation.F1.to_numpy(float), {"status": "base_selected"}
    return apply_fit(train, evaluation, selected["parameterization"], selected["C"], eligibility["independent_eligible"])


def transitions(frame: pd.DataFrame, challenger: str, baseline: str = "D1"):
    base = frame[baseline].to_numpy() >= 0.5
    new = frame[challenger].to_numpy() >= 0.5
    y = frame.label.to_numpy(bool)
    changed = base != new
    return {"changed": int(changed.sum()), "changed_right": int((changed & (new == y) & (base != y)).sum()), "changed_wrong": int((changed & (new != y) & (base == y)).sum())}


def main(extraction: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    adapter_selection = json.loads((extraction.parent / "adapter_runs/selection.json").read_text())
    selected_config = adapter_selection["selected_config"]
    windows = {row["key"]: row for row in jsonl(PARAGRAPHS / "P3.jsonl")}
    oof = pd.read_csv(NEXTGEN / "OOF_PREDICTIONS.csv").reset_index(drop=True)
    frozen = pd.read_csv(NEXTGEN / "PREDICTIONS.csv").reset_index(drop=True)
    frozen["phase"] = frozen.split.map({"validation": "development", "test": "later"})
    sources = {
        "rules": extraction / "facts_rules.jsonl",
        "A0_ensemble": extraction / "facts_A0_ensemble.jsonl",
        f"{selected_config}_ensemble": extraction / f"facts_{selected_config}_ensemble.jsonl",
    }
    for config in {"A0", selected_config}:
        for seed in SEEDS:
            sources[f"{config}_seed{seed}"] = extraction / f"facts_{config}_seed{seed}.jsonl"

    features, fact_rows = {}, {}
    for name, path in sources.items():
        facts, lookup = load_facts(path)
        fact_rows[name] = facts
        features[name] = window_features(lookup, windows).set_index("key")

    main_source = {"D2": "rules", "D3": "A0_ensemble", "D4": f"{selected_config}_ensemble"}
    selections, grids, oof_probabilities, frozen_probabilities, fit_models, coverage = {}, {}, {}, {}, {}, {}
    base_columns = ["key", "symbol", "day", "month", "split", "label", "has_news", "F0", "R1", "F1", "F2", "F6"]
    oof_result = oof[base_columns].copy()
    frozen_result = frozen[base_columns + ["phase"]].copy()
    oof_result["D0"] = oof_result.R1
    oof_result["D1"] = oof_result.F1
    frozen_result["D0"] = frozen_result.R1
    frozen_result["D1"] = frozen_result.F1

    for method, source in main_source.items():
        defaults = {"event_gate": 0, "unique_events": 0, "event_identities": "[]", **{f"z_{name}": 0 for name in FEATURES}}
        train = oof_result.merge(features[source].reset_index(), on="key", how="left", validate="one_to_one").fillna(defaults)
        evaluation = frozen_result.merge(features[source].reset_index(), on="key", how="left", validate="one_to_one").fillna(defaults)
        selected, candidates, forward_probability, fits, counts = choose_residual(train)
        final, model = final_probability(train, evaluation, selected, counts)
        oof_result[method] = forward_probability
        oof_result[f"{method}_event_gate"] = train.event_gate.astype(int).to_numpy()
        frozen_result[method] = final
        frozen_result[f"{method}_event_gate"] = evaluation.event_gate.astype(int).to_numpy()
        selections[method] = {"event_source": source, "selected": selected, "counts": counts}
        grids[method] = candidates
        fit_models[method] = {"forward_fits": fits, "final_fit": model}
        coverage[method] = {"train_event_windows": int(train.event_gate.sum()), "evaluation_event_windows": int(evaluation.event_gate.sum())}
        np.testing.assert_array_equal(oof_result.loc[train.event_gate.eq(0), method].to_numpy(), oof_result.loc[train.event_gate.eq(0), "D1"].to_numpy())
        np.testing.assert_array_equal(frozen_result.loc[evaluation.event_gate.eq(0), method].to_numpy(), frozen_result.loc[evaluation.event_gate.eq(0), "D1"].to_numpy())

    seed_rows = []
    for method, config in (("D3", "A0"), ("D4", selected_config)):
        selected = selections[method]["selected"]
        for seed in SEEDS:
            source = f"{config}_seed{seed}"
            defaults = {"event_gate": 0, "unique_events": 0, "event_identities": "[]", **{f"z_{name}": 0 for name in FEATURES}}
            train = oof_result.merge(features[source].reset_index(), on="key", how="left", validate="one_to_one").fillna(defaults)
            evaluation = frozen_result.merge(features[source].reset_index(), on="key", how="left", validate="one_to_one").fillna(defaults)
            counts = eligibility_counts(train)
            # Seed variation uses the ensemble-selected structure and C, refit on each seed's facts.
            if selected["parameterization"] == "BASE":
                forward_probability = train.F1.to_numpy(float)
                final = evaluation.F1.to_numpy(float)
            else:
                forward_probability, _ = forward_candidate(train, selected["parameterization"], selected["C"])
                final, _ = apply_fit(train, evaluation, selected["parameterization"], selected["C"], counts["independent_eligible"])
            for phase, sample, probability in (("train_forward_oof", train, forward_probability), ("development", evaluation[evaluation.phase.eq("development")], final[evaluation.phase.eq("development")]), ("later", evaluation[evaluation.phase.eq("later")], final[evaluation.phase.eq("later")])):
                for symbol in ("AAPL", "AMZN"):
                    mask = sample.symbol.eq(symbol).to_numpy()
                    metric = classification_metrics(sample.loc[mask, "label"], np.asarray(probability)[mask])
                    seed_rows.append({"method": method, "config": config, "seed": seed, "phase": phase, "symbol": symbol, **metric})

    predictions = pd.concat([oof_result.assign(phase="train_forward_oof"), frozen_result], ignore_index=True)
    methods = ["F0", "R1", "F1", "F2", "F6", "D0", "D1", "D2", "D3", "D4"]
    metric_rows, monthly_rows, subgroup_rows, transition_rows, interval_rows = [], [], [], [], []
    for phase in ("train_forward_oof", "development", "later"):
        phase_frame = predictions[predictions.phase.eq(phase)]
        for symbol in ("AAPL", "AMZN"):
            sample = phase_frame[phase_frame.symbol.eq(symbol)]
            for method in methods:
                metric_rows.append({"phase": phase, "symbol": symbol, "method": method, **classification_metrics(sample.label, sample[method])})
            for month, group in sample.groupby("month"):
                for method in methods:
                    monthly_rows.append({"phase": phase, "symbol": symbol, "month": month, "method": method, **classification_metrics(group.label, group[method])})
            for method in ("D2", "D3", "D4"):
                source = main_source[method]
                joined = sample.merge(features[source][["event_gate"]].reset_index(), on="key", how="left").fillna({"event_gate": 0})
                for gate, group in joined.groupby("event_gate"):
                    subgroup_rows.append({"phase": phase, "symbol": symbol, "method": method, "event_status": "event" if gate else "no_event", **classification_metrics(group.label, group[method])})
                transition_rows.append({"phase": phase, "symbol": symbol, "method": method, **transitions(sample, method)})
                for days in (1, 5):
                    interval_rows.append({"phase": phase, "symbol": symbol, "method": method, **paired_block_interval(sample, method, "D1", days)})

    seed_frame = pd.DataFrame(seed_rows)
    seed_summary = []
    for keys, group in seed_frame.groupby(["method", "config", "phase", "symbol"]):
        method, config, phase, symbol = keys
        seed_summary.append({"method": method, "config": config, "phase": phase, "symbol": symbol, "seeds": len(group), "BA_mean": group.BA.mean(), "BA_std": group.BA.std(ddof=1), "MCC_mean": group.MCC.mean(), "MCC_std": group.MCC.std(ddof=1), "Brier_mean": group.Brier.mean(), "Brier_std": group.Brier.std(ddof=1)})

    gate_columns = [f"{method}_event_gate" for method in ("D2", "D3", "D4")]
    predictions[["key", "symbol", "day", "month", "phase", "label", "has_news", *methods, *gate_columns]].to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "monthly.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(out / "subgroups.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(out / "transitions.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out / "paired_intervals.csv", index=False)
    seed_frame.to_csv(out / "seed_metrics.csv", index=False)
    pd.DataFrame(seed_summary).to_csv(out / "seed_variation.csv", index=False)
    dump(out / "selection.json", {"adapter_selected_config": selected_config, "residual_selections": selections, "candidate_grids": grids, "selection_period": "March-August train-forward OOF only", "development_and_later_used": False})
    dump(out / "fit_models.json", fit_models)
    status = {
        "status": "COMPLETE",
        "selected_adapter": selected_config,
        "methods": main_source,
        "coverage": coverage,
        "exact_no_event_fallback_verified": True,
        "independent_human_review_passed": False,
        "source_hashes": {"OOF": sha(NEXTGEN / "OOF_PREDICTIONS.csv"), "frozen": sha(NEXTGEN / "PREDICTIONS.csv"), "P3": sha(PARAGRAPHS / "P3.jsonl")},
    }
    dump(out / "status.json", status)
    print(json.dumps({"status": status, "selections": selections}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.extraction, args.out)
