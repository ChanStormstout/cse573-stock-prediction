"""Execute the preregistered G0--G3 reliability gate after explicit approval."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from common import CONTROLLER_NAMES, INTERACTION_COLUMNS, STATE_COLUMNS, design_matrix, map_advantage, penalty_for, sha, weighted_mean, weighted_ridge, write_json
from prepare import EVIDENCE, INPUTS, PREDICTIONS, load_inputs, validate_expert_evidence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE_ROOT = ROOT / "work" / "stock-data" / "stock_specific_gate_4h"
OUTER_MONTHS = ["2018-06", "2018-07", "2018-08"]
FORWARD_MONTHS = [f"2018-{m:02d}" for m in range(3, 9)]


def metric(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); q = p >= 0.5
    return {"n": int(len(y)), "BA": float(balanced_accuracy_score(y, q)) if len(np.unique(y)) == 2 else None, "MCC": float(matthews_corrcoef(y, q)), "Brier": float(brier_score_loss(y, p)), "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "up_recall": float(q[y == 1].mean()) if (y == 1).any() else None, "down_recall": float((~q[y == 0]).mean()) if (y == 0).any() else None, "pred_up": float(q.mean()), "constant": bool(len(np.unique(q)) == 1)}


def state_fit(train):
    x = train[STATE_COLUMNS].to_numpy(float)
    med = np.nanmedian(x, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isfinite(x), x, med)
    mean = np.mean(x, axis=0); scale = np.std(x, axis=0); scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return mean, scale


def state_transform(frame, mean, scale):
    x = frame[STATE_COLUMNS].to_numpy(float); x = np.where(np.isfinite(x), x, mean); return (x - mean) / scale


def fit_controller(train, evaluate, method, fold):
    eligible = train[train.eligible_news.eq(1)].copy()
    counts = eligible.groupby(["symbol", "day"], dropna=False).size()
    weights = (1.0 / eligible.groupby(["symbol", "day"], dropna=False)["advantage"].transform("size").to_numpy(float)) if len(eligible) else np.array([])
    support = {"eligible_rows": int(len(eligible)), "unique_stock_days": int(len(counts)), "stock_rows": {s: int((eligible.symbol == s).sum()) for s in ("AAPL", "AMZN")}}
    enough_global = support["eligible_rows"] >= 60 and support["unique_stock_days"] >= 20
    enough_stock = all(x >= 20 for x in support["stock_rows"].values())
    fallback = ""
    coefs = []
    if method == "G0":
        if enough_global:
            d_hat = np.full(len(evaluate), weighted_mean(eligible.advantage, weights)); coefs = [{"feature": "global_advantage", "coefficient": float(d_hat[0])}]
        else:
            d_hat = np.zeros(len(evaluate)); fallback = "R1_SUPPORT_BELOW_60_ROWS_OR_20_STOCK_DAYS"
    elif method == "G1":
        if not enough_global:
            d_hat = np.zeros(len(evaluate)); fallback = "R1_SUPPORT_BELOW_60_ROWS_OR_20_STOCK_DAYS"
        elif not enough_stock:
            d_hat = np.full(len(evaluate), weighted_mean(eligible.advantage, weights)); fallback = "G0_PER_STOCK_SUPPORT_BELOW_20"
            coefs = [{"feature": "global_advantage", "coefficient": float(d_hat[0])}]
        else:
            means = {s: weighted_mean(eligible.loc[eligible.symbol == s, "advantage"], weights[eligible.symbol.to_numpy() == s]) for s in ("AAPL", "AMZN")}
            d_hat = evaluate.symbol.map(means).to_numpy(float); coefs = [{"feature": f"{s}_static_advantage", "coefficient": float(means[s])} for s in ("AAPL", "AMZN")]
    else:
        if not enough_global:
            d_hat = np.zeros(len(evaluate)); fallback = "R1_SUPPORT_BELOW_60_ROWS_OR_20_STOCK_DAYS"
        elif method == "G3" and not enough_stock:
            # Fit the nested shared-slope model as the exact registered G3
            # fallback, rather than silently dropping the fold.
            fitted = fit_controller(train, evaluate, "G2", fold); fitted["fallback"] = "G2_PER_STOCK_SUPPORT_BELOW_20"; return fitted
        else:
            mean, scale = state_fit(eligible); z_train = state_transform(eligible, mean, scale); z_eval = state_transform(evaluate, mean, scale)
            stock_train = (eligible.symbol == "AMZN").astype(float).to_numpy(); stock_eval = (evaluate.symbol == "AMZN").astype(float).to_numpy()
            variant = method; X = design_matrix(z_train, stock_train, variant); Z = design_matrix(z_eval, stock_eval, variant)
            beta = weighted_ridge(X, eligible.advantage.to_numpy(float), weights, penalty_for(variant)); d_hat = Z @ beta
            names = ["intercept", "AMZN_intercept"] + (STATE_COLUMNS if variant == "G2" else STATE_COLUMNS + INTERACTION_COLUMNS)
            coefs = [{"feature": n, "coefficient": float(v)} for n, v in zip(names, beta)]
            return {"d_hat": d_hat, "coefs": coefs, "fallback": fallback, "support": support, "state_mean": mean.tolist(), "state_scale": scale.tolist(), "fit_rows": int(len(eligible)), "weights_sum": float(weights.sum()), "variant": method}
    return {"d_hat": d_hat, "coefs": coefs, "fallback": fallback, "support": support, "state_mean": None, "state_scale": None, "fit_rows": int(len(eligible)), "weights_sum": float(weights.sum()) if len(weights) else 0.0, "variant": method}


def prediction_rows(frame, fitted, method, fold):
    p, w = map_advantage(fitted["d_hat"], frame.p_price.to_numpy(float), frame.p_text.to_numpy(float), frame.has_news.to_numpy(int))
    if np.any(frame.has_news.to_numpy(int) == 0):
        no = frame.has_news.to_numpy(int) == 0
        if not np.array_equal(p[no], frame.loc[no, "p_price"].to_numpy(float)):
            raise AssertionError("no-news probability is not exact R1")
        w[no] = 0.0
    out = frame[["key", "symbol", "day", "month", "phase", "start_utc", "end_utc", "cutoff_utc", "label", "has_news", "eligible_news", "p_price", "p_text"]].copy()
    out["controller"] = method; out["fold"] = fold; out["d_hat"] = fitted["d_hat"]; out["w_text"] = w; out["p_final"] = p; out["fallback"] = fitted["fallback"] or "NONE"; return out


def oracle_table(frame):
    rows = []
    for (phase, symbol, month), g in frame.groupby(["phase", "symbol", "month"]):
        g = g[g.p_price.notna() & g.p_text.notna()].copy(); y = g.label.to_numpy(int); a = (g.p_price.to_numpy(float) >= .5).astype(int); b = (g.p_text.to_numpy(float) >= .5).astype(int); ca = a == y; cb = b == y; disagree = a != b; oracle = np.where(disagree & ca, a, np.where(disagree & cb, b, a))
        rows.append({"phase": phase, "symbol": symbol, "month": month, "n": len(g), "disagreement": int(disagree.sum()), "disagreement_rate": float(disagree.mean()) if len(g) else 0.0, "R1_wrong_F1_right": int((~ca & cb).sum()), "R1_right_F1_wrong": int((ca & ~cb).sum()), "both_right": int((ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()), "oracle_BA": float(balanced_accuracy_score(y, oracle)) if len(np.unique(y)) == 2 else None, "label": "HINDSIGHT DIAGNOSTIC ORACLE — NOT A MODEL"})
    return pd.DataFrame(rows)


def make_metrics(pred):
    rows = []; monthly = []; methods = list(CONTROLLER_NAMES)
    for (phase, symbol), g in pred.groupby(["phase", "symbol"]):
        for m in methods:
            z = g[g.p_final.notna() & (g.controller == m)]; base = z.p_price.to_numpy(float); final = z.p_final.to_numpy(float)
            q = metric(z.label, final); q.update({"phase": phase, "symbol": symbol, "method": m, "text_weight_mean": float(z.w_text.mean()), "text_weight_std": float(z.w_text.std(ddof=0)), "text_weight_gt_half": float((z.w_text > .5).mean()), "direction_changed_vs_R1": int(((final >= .5) != (base >= .5)).sum()), "repaired_errors": int((((final >= .5) == z.label) & ((base >= .5) != z.label)).sum()), "new_errors": int((((final >= .5) != z.label) & ((base >= .5) == z.label)).sum())}); rows.append(q)
        for month, x in g.groupby("month"):
            for m in methods:
                z = x[x.controller == m]; q = metric(z.label, z.p_final); q.update({"phase": phase, "symbol": symbol, "month": month, "method": m, "text_weight_mean": float(z.w_text.mean()), "text_weight_std": float(z.w_text.std(ddof=0)), "text_weight_gt_half": float((z.w_text > .5).mean()), "direction_changed_vs_R1": int(((z.p_final >= .5) != (z.p_price >= .5)).sum()), "repaired_errors": int((((z.p_final >= .5) == z.label) & ((z.p_price >= .5) != z.label)).sum()), "new_errors": int((((z.p_final >= .5) != z.label) & ((z.p_price >= .5) == z.label)).sum())}); monthly.append(q)
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def advancement(monthly, pred):
    out = []; contrasts = [("G1", "G0"), ("G2", "G1"), ("G3", "G2")]
    for new, base in contrasts:
        a = monthly[(monthly.phase == "train_forward_oof") & monthly.month.isin(OUTER_MONTHS) & (monthly.method == new)].set_index(["symbol", "month"]); b = monthly[(monthly.phase == "train_forward_oof") & monthly.month.isin(OUTER_MONTHS) & (monthly.method == base)].set_index(["symbol", "month"]); j = a.join(b, lsuffix="_new", rsuffix="_base"); j["delta_BA"] = j.BA_new - j.BA_base; j["delta_Brier"] = j.Brier_new - j.Brier_base
        stock = j.groupby(level=0).delta_BA.mean(); months = j.groupby(level=1).delta_BA.mean(); brier = j.groupby(level=0).delta_Brier.mean(); constants = monthly[(monthly.phase == "train_forward_oof") & monthly.month.isin(OUTER_MONTHS) & (monthly.method == new)].groupby("symbol").constant.any()
        outer = pred[(pred.phase == "train_forward_oof") & pred.month.isin(OUTER_MONTHS)].drop_duplicates(["key"])
        headroom = {s: int(((outer[outer.symbol == s].p_price - outer[outer.symbol == s].p_text).abs() > 1e-12).sum()) for s in ("AAPL", "AMZN")}
        enough_headroom = min(headroom.values()) >= 10
        out.append({"contrast": f"{new}_vs_{base}", "AAPL_mean_delta_BA": float(stock.get("AAPL", np.nan)), "AMZN_mean_delta_BA": float(stock.get("AMZN", np.nan)), "weaker_stock_mean_delta_BA": float(stock.min()) if len(stock) else None, "macro_delta_BA": float(j.delta_BA.mean()) if len(j) else None, "positive_outer_months": int((months > 0).sum()), "max_stock_mean_delta_Brier": float(brier.max()) if len(brier) else None, "constant_any_stock": bool(constants.any()) if len(constants) else True, "expert_difference_rows": headroom, "routing_headroom_sufficient": enough_headroom, "outer_cells": int(len(j)), "passes": bool(len(j) == 6 and stock.min() >= .01 and stock.max() >= -.01 and (months > 0).sum() >= 2 and brier.max() <= .002 and not constants.any() and enough_headroom)})
    return pd.DataFrame(out)


def run(output: Path):
    d, provenance = load_inputs(); evidence = validate_expert_evidence(d); all_predictions = []; fits = []; coefficients = []; fallbacks = []; start = time.monotonic()
    for fold in FORWARD_MONTHS:
        ev = d[(d.month == fold) & d.p_price.notna()].copy(); train = d[(d.month < fold) & d.eligible_news.eq(1) & (d.end_utc < ev.cutoff_utc.min())].copy()
        for method in CONTROLLER_NAMES:
            fit = fit_controller(train, ev, method, fold); all_predictions.append(prediction_rows(ev, fit, method, fold));
            fits.append({"fold": fold, "method": method, "train_rows": int(len(train)), "eligible_rows": fit["fit_rows"], "eval_rows": int(len(ev)), "train_target_end_max": train.end_utc.max().isoformat() if len(train) else None, "eval_cutoff_min": ev.cutoff_utc.min().isoformat() if len(ev) else None, "time_safe": bool(not len(train) or train.end_utc.max() < ev.cutoff_utc.min()), "fallback": fit["fallback"] or "NONE", "support": fit["support"], "weights_sum": fit["weights_sum"], "state_fit_rows": fit["fit_rows"]})
            fallbacks.append({"fold": fold, "method": method, "fallback": fit["fallback"] or "NONE", **fit["support"]})
            for c in fit["coefs"]: coefficients.append({"fold": fold, "method": method, **c})
    final_ev = d[(d.month >= "2018-09") & d.p_price.notna()].copy()
    final_train = d[(d.month >= "2018-03") & (d.month < "2018-09") & d.eligible_news.eq(1) & (d.end_utc < final_ev.cutoff_utc.min())]
    for method in CONTROLLER_NAMES:
        fit = fit_controller(final_train, final_ev, method, "august_freeze"); all_predictions.append(prediction_rows(final_ev, fit, method, "august_freeze")); fits.append({"fold": "august_freeze", "method": method, "train_rows": int(len(final_train)), "eligible_rows": fit["fit_rows"], "eval_rows": int(len(final_ev)), "train_target_end_max": final_train.end_utc.max().isoformat() if len(final_train) else None, "eval_cutoff_min": final_ev.cutoff_utc.min().isoformat() if len(final_ev) else None, "time_safe": bool(not len(final_train) or final_train.end_utc.max() < final_ev.cutoff_utc.min()), "fallback": fit["fallback"] or "NONE", "support": fit["support"], "weights_sum": fit["weights_sum"], "state_fit_rows": fit["fit_rows"], "frozen_no_exposed_label_update": True}); fallbacks.append({"fold": "august_freeze", "method": method, "fallback": fit["fallback"] or "NONE", **fit["support"]}); [coefficients.append({"fold": "august_freeze", "method": method, **c}) for c in fit["coefs"]]
    pred = pd.concat(all_predictions, ignore_index=True); pred.to_csv(output / "predictions.csv", index=False); metrics, monthly = make_metrics(pred); metrics.to_csv(output / "metrics.csv", index=False); monthly.to_csv(output / "monthly_metrics.csv", index=False); oracle = oracle_table(d[d.month >= "2018-03"].copy()); oracle.to_csv(output / "oracle_ceiling.csv", index=False); oracle.to_csv(output / "disagreement_audit.csv", index=False); pd.DataFrame(coefficients).to_csv(output / "controller_coefficients.csv", index=False); pd.DataFrame(fallbacks).to_csv(output / "fallback_audit.csv", index=False); write_json(output / "controller_training_evidence.json", {"fits": fits, "expert_provenance": provenance, "expert_evidence": evidence, "state_columns": STATE_COLUMNS, "protocol_sha256": sha(HERE / "PRE_REGISTRATION.md")}); adv = advancement(monthly, pred); adv.to_json(output / "advancement.json", orient="records", indent=2); write_json(output / "protocol_fingerprint.json", {"inputs_sha256": sha(INPUTS), "predictions_sha256": sha(PREDICTIONS), "evidence_sha256": sha(EVIDENCE), "runner_sha256": sha(HERE / "run_gate.py"), "preregistration_sha256": sha(HERE / "PRE_REGISTRATION.md"), "runtime_seconds": time.monotonic() - start}); return pred, metrics, monthly, adv


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--approve-gate-run", action="store_true"); parser.add_argument("--output", type=Path, default=HERE / "v1"); args = parser.parse_args()
    if not args.approve_gate_run:
        raise SystemExit("Phase B is preregistered but paused. Re-run only with --approve-gate-run after explicit approval.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("stock-specific gate output must be a new directory")
    private = PRIVATE_ROOT / args.output.name
    if private.exists():
        raise FileExistsError("private stock-specific gate directory already exists")
    args.output.mkdir(parents=True); private.mkdir(parents=True)
    pred, metrics, monthly, adv = run(args.output); print(json.dumps({"status": "COMPLETE", "rows": int(len(pred)), "advancement": adv.to_dict("records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
