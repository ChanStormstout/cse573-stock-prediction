"""Time-safe v2 reaction probe.

The v1 reaction files remain historical. v2 makes the information boundary
auditable: pre-event context uses completed bars only, AR1 preprocessing is fit
inside each past fold, AR2 is explicitly NOT_RUN when the target-context
FinBERT binary is unavailable, and the downstream W0--W3 protocol is frozen in
the v2 preregistration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work" / "stock-data"
PUBLIC = HERE / "v2"
PRIVATE = WORK / "reaction_features_4h" / "v2"
HORIZONS = (30, 60, 120, 240)
CS = (0.01, 0.1, 1.0)
NUMERIC = [f"pre_{m}m_{x}" for m in (5, 15, 30, 60) for x in ("return", "rv")] + ["minutes_from_open"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    def clean(x):
        if hasattr(x, "item"):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def metric(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); q = p >= 0.5
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, q)) if len(np.unique(y)) == 2 else None,
        "MCC": float(matthews_corrcoef(y, q)),
        "Brier": float(brier_score_loss(y, p)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "up_recall": float(q[y == 1].mean()) if (y == 1).any() else None,
        "down_recall": float((~q[y == 0]).mean()) if (y == 0).any() else None,
        "pred_up": float(q.mean()),
        "constant": bool(len(np.unique(q)) == 1),
    }


def context_text(row) -> str:
    import re
    target = "Apple" if row.symbol == "AAPL" else "Amazon"
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", str(row.body or "")) if x.strip()]
    hits = [i for i, s in enumerate(sentences) if target.casefold() in s.casefold() or row.symbol in s]
    evidence = " ".join(sentences[max(0, hits[0] - 1):min(len(sentences), hits[0] + 2)]) if hits else " ".join(sentences[:2])
    return f"TARGET={row.symbol}\nTITLE={row.title}\nEVIDENCE={evidence}"[:6000]


def fit_design(train: pd.DataFrame, evaluate: pd.DataFrame, method: str):
    """Fit numeric and lexical transforms on train only, then transform eval."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[NUMERIC])
    x_num = imputer.transform(train[NUMERIC]); z_num = imputer.transform(evaluate[NUMERIC])
    scaler = StandardScaler().fit(x_num); x_num = scaler.transform(x_num); z_num = scaler.transform(z_num)
    evidence = {"numeric_columns": NUMERIC, "numeric_imputer_fit_rows": int(len(train)), "numeric_scaler_fit_rows": int(len(train))}
    if method == "AR0":
        return x_num, z_num, evidence
    vectorizer = CountVectorizer(binary=True, min_df=3)
    raw = vectorizer.fit_transform(train.context.fillna("")); terms = vectorizer.get_feature_names_out()
    if raw.shape[1]:
        scores = chi2(raw, train.label.to_numpy(int))[0]
        keep = np.lexsort((terms, -np.nan_to_num(scores, nan=-np.inf)))[:min(500, raw.shape[1])]
    else:
        keep = np.array([], dtype=int)
    x_txt = raw[:, keep].toarray() if len(keep) else np.zeros((len(train), 0))
    z_raw = vectorizer.transform(evaluate.context.fillna("")); z_txt = z_raw[:, keep].toarray() if len(keep) else np.zeros((len(evaluate), 0))
    evidence.update({"lexical_fit_rows": int(len(train)), "lexical_vocabulary_size": int(raw.shape[1]), "lexical_selected_features": int(len(keep)), "lexical_selection": "chi2_on_past_fold_then_lexicographic_tie_break"})
    return np.c_[x_num, x_txt], np.c_[z_num, z_txt], evidence


def train_predict(train: pd.DataFrame, evaluate: pd.DataFrame, method: str, c_value: float, model_path: Path):
    if train.empty or evaluate.empty:
        raise ValueError("empty reaction fold")
    if pd.to_datetime(train.reaction_end_utc, utc=True).max() >= pd.to_datetime(evaluate.available_utc, utc=True).min():
        raise AssertionError("reaction label is not mature before evaluation availability")
    x, z, prep = fit_design(train, evaluate, method)
    group_n = train.groupby(["symbol", "session_day"])["group_key"].transform("count").to_numpy(float); weights = 1.0 / group_n
    model = LogisticRegression(C=c_value, solver="liblinear", max_iter=3000, tol=1e-8, random_state=573).fit(x, train.label.to_numpy(int), sample_weight=weights)
    p = model.predict_proba(z)[:, 1]
    model_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"model": model, "method": method, "preprocessing": prep}, model_path)
    loaded = joblib.load(model_path); reload_p = loaded["model"].predict_proba(z)[:, 1]; assert np.max(np.abs(p - reload_p)) <= 1e-12
    return p, {"method": method, "C": float(c_value), "train_n": int(len(train)), "eval_n": int(len(evaluate)), "train_target_end_max": pd.to_datetime(train.reaction_end_utc, utc=True).max().isoformat(), "eval_available_min": pd.to_datetime(evaluate.available_utc, utc=True).min().isoformat(), "weight_sum": float(weights.sum()), "weight_rule": "1 / count(symbol, session_day)", "model_sha256": sha(model_path), "preprocessing": prep, "reload_max_abs_error": float(np.max(np.abs(p - reload_p)))}


def prepare_reactions():
    rx = pd.read_pickle(PRIVATE / "article_reactions_all_time.pkl"); pairs = pd.read_pickle(PRIVATE / "canonical_pairs_all_time.pkl")
    rx["available_utc"] = pd.to_datetime(rx.available_utc, utc=True)
    for h in HORIZONS:
        rx[f"reaction_{h}m_end_utc"] = pd.to_datetime(rx[f"reaction_{h}m_end_utc"], utc=True, errors="coerce")
        rx[f"reaction_{h}m_start_utc"] = pd.to_datetime(rx[f"reaction_{h}m_start_utc"], utc=True, errors="coerce")
    # The reaction table already carries the canonical title and pair key;
    # merge only the missing body to avoid pandas title_x/title_y ambiguity.
    pairs = pairs[["symbol", "group_key", "body"]]
    rx = rx.merge(pairs, on=["symbol", "group_key"], how="left", validate="one_to_one")
    rx = rx[(rx.available_utc >= pd.Timestamp("2018-01-01", tz="UTC")) & (rx.available_utc < pd.Timestamp("2018-09-01", tz="UTC"))].copy()
    rx["month"] = rx.available_utc.dt.strftime("%Y-%m"); rx["context"] = rx.apply(context_text, axis=1)
    return rx


def run_article_models(rx: pd.DataFrame, public: Path, private: Path):
    predictions = []; selections = []; fits = []
    for horizon in HORIZONS:
        valid_col = f"reaction_{horizon}m_valid"; end_col = f"reaction_{horizon}m_end_utc"
        # train_predict uses horizon-neutral names; set them only for this
        # loop so no horizon can accidentally borrow another label/end time.
        rx["label"] = (rx[f"reaction_{horizon}m"] > 0).astype(int)
        rx["reaction_end_utc"] = pd.to_datetime(rx[end_col], utc=True, errors="coerce")
        for method in ("AR0", "AR1"):
            inner_scores = []
            for c_value in CS:
                for month in ("2018-03", "2018-04", "2018-05"):
                    for symbol in ("AAPL", "AMZN"):
                        ev = rx[(rx.symbol == symbol) & (rx.month == month) & (rx[valid_col] == 1)].copy()
                        if ev.empty:
                            train = rx.iloc[0:0].copy()
                        else:
                            earliest_eval = ev.available_utc.min()
                            train = rx[(rx.symbol == symbol) & (rx.month < month) &
                                       (rx[valid_col] == 1) & (rx[end_col] < earliest_eval)].copy()
                        if len(train) < 30 or ev.empty: continue
                        p, evidence = train_predict(train, ev, method, c_value, private / "models" / f"inner_{horizon}_{method}_{symbol}_{month}_{c_value}.joblib")
                        fits.append({**evidence, "phase": "inner", "horizon_minutes": horizon, "symbol": symbol, "month": month}); inner_scores.append({"horizon_minutes": horizon, "method": method, "C": c_value, "symbol": symbol, "month": month, **metric(ev.label, p)})
            choices = []
            for c_value in CS:
                rows = [r for r in inner_scores if r["horizon_minutes"] == horizon and r["method"] == method and r["C"] == c_value]
                choices.append((c_value, float(np.mean([r["BA"] for r in rows])) if rows else -1.0))
            chosen = min(choices, key=lambda x: (-x[1], x[0]))[0] if choices else 0.1
            selections.append({"horizon_minutes": horizon, "method": method, "C": chosen, "inner_scores": choices, "selection_period": ["2018-03", "2018-04", "2018-05"]})
            for month, phase in [(m, "inner") for m in ("2018-03", "2018-04", "2018-05")] + [(m, "outer") for m in ("2018-06", "2018-07", "2018-08")]:
                for symbol in ("AAPL", "AMZN"):
                    ev = rx[(rx.symbol == symbol) & (rx.month == month) & (rx[valid_col] == 1)].copy()
                    if ev.empty:
                        train = rx.iloc[0:0].copy()
                    else:
                        earliest_eval = ev.available_utc.min()
                        train = rx[(rx.symbol == symbol) & (rx.month < month) &
                                   (rx[valid_col] == 1) & (rx[end_col] < earliest_eval)].copy()
                    if len(train) < 30 or ev.empty: continue
                    p, evidence = train_predict(train, ev, method, chosen, private / "models" / f"{phase}_{horizon}_{method}_{symbol}_{month}.joblib")
                    fits.append({**evidence, "phase": phase, "horizon_minutes": horizon, "symbol": symbol, "month": month})
                    q = ev[["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "session_day", "label", end_col, "price_context_max_used_bar_end_utc", "price_context_complete"]].copy(); q["phase"] = phase; q["horizon_minutes"] = horizon; q["method"] = method; q["prediction"] = p; predictions.append(q)
    result = pd.concat(predictions, ignore_index=True); result.to_csv(public / "reaction_predictions.csv", index=False); pd.DataFrame(selections).to_json(public / "reaction_selection.json", orient="records", indent=2)
    write_json(public / "reaction_training_evidence.json", {"status": "COMPLETE", "fits": fits, "fit_count": len(fits), "selection": selections, "AR2": {"status": "NOT_RUN_MODEL_BINARY_UNAVAILABLE", "reason": "registered target-context FinBERT representation cannot be reconstructed from the available private artifact"}})
    return result


def article_gate(predictions: pd.DataFrame, public: Path):
    predictions = predictions.copy(); predictions["month"] = pd.to_datetime(predictions.available_utc, utc=True).dt.strftime("%Y-%m")
    rows = []
    for (phase, horizon, symbol, method, month), g in predictions.groupby(["phase", "horizon_minutes", "symbol", "method", "month"]):
        rows.append({"phase": phase, "horizon_minutes": int(horizon), "symbol": symbol, "method": method, "month": month, **metric(g.label, g.prediction), "valid_rows": int(len(g)), "context_complete_rate": float(g.price_context_complete.mean())})
    metrics = pd.DataFrame(rows); metrics.to_csv(public / "reaction_metrics.csv", index=False); promotions = []
    for horizon in HORIZONS:
        ar0 = metrics[(metrics.phase == "outer") & (metrics.horizon_minutes == horizon) & (metrics.method == "AR0")].set_index(["symbol", "month"]); ar1 = metrics[(metrics.phase == "outer") & (metrics.horizon_minutes == horizon) & (metrics.method == "AR1")].set_index(["symbol", "month"]); common = ar0.index.intersection(ar1.index); joined = ar0.loc[common].join(ar1.loc[common], lsuffix="_ar0", rsuffix="_ar1"); joined["delta_BA"] = joined.BA_ar1 - joined.BA_ar0; joined["delta_Brier"] = joined.Brier_ar1 - joined.Brier_ar0
        monthly = [{"symbol": symbol, "month": month, "delta_BA": float(r.delta_BA), "delta_Brier": float(r.delta_Brier), "AR1_n": int(r.n_ar1), "AR0_n": int(r.n_ar0), "coverage": 1.0, "horizon_complete": True} for (symbol, month), r in joined.iterrows()]; month_df = pd.DataFrame(monthly); month_df.to_csv(public / f"promotion_h{horizon}m_monthly.csv", index=False)
        stock_rows = [{"symbol": symbol, "mean_delta_BA": float(g.delta_BA.mean()), "max_delta_Brier": float(g.delta_Brier.max()), "positive_months": int((g.delta_BA > 0).sum()), "months": int(len(g)), "min_month_rows": int(g.AR1_n.min()), "coverage_min": float(g.coverage.min()), "horizon_complete": bool(g.horizon_complete.all())} for symbol, g in month_df.groupby("symbol")]
        stock_df = pd.DataFrame(stock_rows); enough = bool(len(stock_df) == 2 and len(month_df) == 6 and (stock_df.min_month_rows >= 30).all() and (stock_df.coverage_min >= 0.8).all() and stock_df.horizon_complete.all()); passes = bool(enough and (stock_df.mean_delta_BA >= 0.01).all() and (stock_df.max_delta_Brier <= 0.002).all() and (stock_df.positive_months >= 2).all())
        promotions.append({"horizon_minutes": horizon, "candidate": "AR1", "gate_formula": "each stock: mean_delta_BA>=0.01, max_delta_Brier<=0.002, positive_months>=2/3; both stocks have 3 complete outer months, >=30 rows/month, coverage>=0.8", "stock_rows": stock_rows, "passes": passes, "status": "PASS" if passes else "FAIL"})
    write_json(public / "promotion_gate.json", {"status": "COMPLETE", "outer_months": ["2018-06", "2018-07", "2018-08"], "promotions": promotions}); return metrics, promotions


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def downstream_protocol(predictions: pd.DataFrame, promotions, public: Path):
    selected = [int(x["horizon_minutes"]) for x in promotions if x["passes"]]
    if not selected:
        write_json(public / "downstream_status.json", {"status": "NOT_RUN_NO_PROMOTED_HORIZON", "protocol": "W0 fixed F1; W1 single promoted horizon; W2 all promoted horizons; W3 residual with strict coverage fallback"}); return {"status": "NOT_RUN_NO_PROMOTED_HORIZON"}
    inputs = pd.read_pickle(WORK / "paper_methods_4h" / "v1" / "inputs.pkl").copy(); inputs["key"] = inputs.symbol + "|" + pd.to_datetime(inputs.start_utc, utc=True).astype(str); inputs["start_utc"] = pd.to_datetime(inputs.start_utc, utc=True); inputs["cutoff_utc"] = pd.to_datetime(inputs.cutoff_utc, utc=True)
    base = pd.read_csv(ROOT / "outputs" / "stock_goal60_4h" / "v1" / "predictions.csv")[["key", "F1"]].rename(columns={"F1": "W0"}); windows = inputs.merge(base, on="key", validate="one_to_one"); windows["month"] = windows.start_utc.dt.strftime("%Y-%m"); windows["phase"] = np.where(windows.month < "2018-03", "warmup", np.where(windows.month < "2018-09", "train_forward_oof", "exposed"))
    manifest_rows = []; features = {}
    for r in windows.itertuples(index=False):
        for h in selected:
            q = predictions[(predictions.method == "AR1") & (predictions.horizon_minutes == h) & (predictions.symbol == r.symbol) & (predictions.available_utc <= r.cutoff_utc) & (predictions.available_utc > r.cutoff_utc - pd.Timedelta("4h"))]; count = len(q); mean = float(q.prediction.mean()) if count else .5; std = float(q.prediction.std(ddof=0)) if count else 0.0; manifest_rows.append({"key": r.key, "symbol": r.symbol, "cutoff_utc": r.cutoff_utc.isoformat(), "horizon_minutes": h, "n_articles": count, "coverage": 1.0 if count else 0.0, "horizon_complete": bool(count > 0), "used_group_count": int(q.group_key.nunique())}); features[(r.key, h)] = (mean, count, std)
    pd.DataFrame(manifest_rows).to_csv(public / "downstream_input_manifest.csv", index=False); outputs = []; metric_rows = []
    for symbol in ("AAPL", "AMZN"):
        train = windows[(windows.symbol == symbol) & (windows.phase == "train_forward_oof") & windows.month.isin(["2018-03", "2018-04", "2018-05"])].copy(); ev = windows[(windows.symbol == symbol) & windows.month.isin(["2018-06", "2018-07", "2018-08"])].copy()
        def vector(frame, hs, include_std=False):
            vals = [logit(frame.W0.to_numpy(float))]
            for h in hs:
                vals.extend([[features[(k, h)][0] for k in frame.key],
                             [np.log1p(features[(k, h)][1]) for k in frame.key]])
                if include_std:
                    vals.append([features[(k, h)][2] for k in frame.key])
            return np.column_stack(vals)
        scaler1 = StandardScaler().fit(vector(train, [selected[0]])); m1 = LogisticRegression(C=.1, solver="liblinear", max_iter=3000, random_state=573).fit(scaler1.transform(vector(train, [selected[0]])), train.label); w1 = m1.predict_proba(scaler1.transform(vector(ev, [selected[0]])))[:, 1]
        if len(selected) > 1:
            scaler2 = StandardScaler().fit(vector(train, selected)); m2 = LogisticRegression(C=.1, solver="liblinear", max_iter=3000, random_state=573).fit(scaler2.transform(vector(train, selected)), train.label); w2 = m2.predict_proba(scaler2.transform(vector(ev, selected)))[:, 1]
        else: w2 = w1.copy()
        scaler3 = StandardScaler().fit(vector(train, [selected[0]], include_std=True)); m3 = LogisticRegression(C=.1, solver="liblinear", max_iter=3000, random_state=573).fit(scaler3.transform(vector(train, [selected[0]], include_std=True)), train.label); w3_adjusted = m3.predict_proba(scaler3.transform(vector(ev, [selected[0]], include_std=True)))[:, 1]
        count = np.array([features[(k, selected[0])][1] for k in ev.key]); w3 = np.where(count >= 2, w3_adjusted, ev.W0.to_numpy(float)); out = ev[["key", "symbol", "month", "label", "W0"]].copy(); out["W1"] = w1; out["W2"] = w2; out["W3"] = w3; out["W3_fallback"] = count < 2; outputs.append(out)
        for name in ("W0", "W1", "W2", "W3"): metric_rows.append({"symbol": symbol, "method": name, **metric(out.label, out[name])})
    p = pd.concat(outputs, ignore_index=True); p.to_csv(public / "reaction_downstream_predictions.csv", index=False); pd.DataFrame(metric_rows).to_csv(public / "reaction_downstream_metrics.csv", index=False); return {"status": "COMPLETE", "promoted_horizons": selected, "protocol": "W0 fixed F1; W1 single promoted horizon; W2 all promoted horizons; W3 W1 with n_articles>=2 else exact W0", "rows": int(len(p))}


def main(public_path: Path = PUBLIC, private_path: Path = PRIVATE):
    gate = json.loads((public_path / "reaction_gate.json").read_text())
    if gate.get("status") != "PASS": raise SystemExit("reaction feasibility gate failed; stop before predictive training")
    t0 = time.monotonic(); rx = prepare_reactions(); pred = run_article_models(rx, public_path, private_path); _, promotions = article_gate(pred, public_path); downstream = downstream_protocol(pred, promotions, public_path)
    write_json(public_path / "reaction_probe_gate.json", {"status": "COMPLETE", "promotion": promotions, "downstream": downstream, "AR2": {"status": "NOT_RUN_MODEL_BINARY_UNAVAILABLE"}, "runtime_seconds": time.monotonic() - t0, "protocol_sha256": sha(public_path.parent / "PRE_REGISTRATION_v2.md"), "run_sha256": sha(HERE / "run_reaction_probe_v2.py")}); print(json.dumps({"status": "COMPLETE", "promotion": promotions, "downstream": downstream}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--public", type=Path, default=PUBLIC); parser.add_argument("--private", type=Path, default=PRIVATE); args = parser.parse_args(); main(args.public, args.private)
