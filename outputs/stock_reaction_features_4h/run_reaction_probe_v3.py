"""Reviewer-protocol reaction probe: AR0/AR1 audit and explicit AR2 stop."""
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
SOURCE_PRIVATE = WORK / "reaction_features_4h" / "v2"
PUBLIC = HERE / "v3"
PRIVATE = WORK / "reaction_features_4h" / "v3"
HORIZONS = (60, 240)
CS = (0.01, 0.1, 1.0)
NUMERIC = [f"pre_{m}m_{x}" for m in (5, 15, 30, 60) for x in ("return", "rv")] + ["minutes_from_open"]


def write_json(path, value):
    def clean(x):
        if hasattr(x, "item"): return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        return x
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def metric(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); q = p >= .5
    return {"n": int(len(y)), "BA": float(balanced_accuracy_score(y, q)) if len(np.unique(y)) == 2 else None, "MCC": float(matthews_corrcoef(y, q)), "Brier": float(brier_score_loss(y, p)), "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "up_recall": float(q[y == 1].mean()) if (y == 1).any() else None, "down_recall": float((~q[y == 0]).mean()) if (y == 0).any() else None, "pred_up": float(q.mean()), "constant": bool(len(np.unique(q)) == 1)}


def context_text(row):
    import re
    target = "Apple" if row.symbol == "AAPL" else "Amazon"
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", str(row.body or "")) if x.strip()]
    hits = [i for i, s in enumerate(sentences) if target.casefold() in s.casefold() or row.symbol in s]
    evidence = " ".join(sentences[max(0, hits[0] - 1):min(len(sentences), hits[0] + 2)]) if hits else " ".join(sentences[:2])
    return f"TARGET={row.symbol}\nTITLE={row.title}\nCONTEXT={evidence}"[:6000]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(start="2018-01-01", end=None):
    rx = pd.read_pickle(SOURCE_PRIVATE / "article_reactions_all_time.pkl"); pairs = pd.read_pickle(SOURCE_PRIVATE / "canonical_pairs_all_time.pkl")
    rx["available_utc"] = pd.to_datetime(rx.available_utc, utc=True)
    for h in HORIZONS:
        rx[f"reaction_{h}m_end_utc"] = pd.to_datetime(rx[f"reaction_{h}m_end_utc"], utc=True, errors="coerce")
        rx[f"reaction_{h}m_start_utc"] = pd.to_datetime(rx[f"reaction_{h}m_start_utc"], utc=True, errors="coerce")
    rx = rx.merge(pairs[["symbol", "group_key", "body"]], on=["symbol", "group_key"], how="left", validate="one_to_one")
    rx = rx[rx.available_utc >= pd.Timestamp(start, tz="UTC")].copy()
    if end is not None:
        rx = rx[rx.available_utc < pd.Timestamp(end, tz="UTC")].copy()
    rx["month"] = rx.available_utc.dt.strftime("%Y-%m"); rx["context"] = rx.apply(context_text, axis=1); return rx


def fit_design(train, evaluate, method):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[NUMERIC]); x_num = imputer.transform(train[NUMERIC]); z_num = imputer.transform(evaluate[NUMERIC]); scaler = StandardScaler().fit(x_num); x_num = scaler.transform(x_num); z_num = scaler.transform(z_num)
    evidence = {"numeric_columns": NUMERIC, "numeric_imputer_fit_rows": int(len(train)), "numeric_scaler_fit_rows": int(len(train))}
    if method == "AR0": return x_num, z_num, evidence
    vec = CountVectorizer(binary=True, min_df=3); raw = vec.fit_transform(train.context.fillna("") ); terms = vec.get_feature_names_out(); scores = chi2(raw, train.label.to_numpy(int))[0] if raw.shape[1] else np.array([]); keep = np.lexsort((terms, -np.nan_to_num(scores, nan=-np.inf)))[:min(500, raw.shape[1])] if raw.shape[1] else np.array([], int); x_text = raw[:, keep].toarray() if len(keep) else np.zeros((len(train), 0)); z_text = vec.transform(evaluate.context.fillna(""))[:, keep].toarray() if len(keep) else np.zeros((len(evaluate), 0)); evidence.update({"lexical_fit_rows": int(len(train)), "lexical_vocabulary_size": int(raw.shape[1]), "lexical_selected_features": int(len(keep)), "lexical_selection": "chi2_on_past_fold_then_lexicographic_tie_break"}); return np.c_[x_num, x_text], np.c_[z_num, z_text], evidence


def train_predict(train, evaluate, method, c, path):
    if train.empty or evaluate.empty: raise ValueError("empty fold")
    if train.reaction_end_utc.max() >= evaluate.available_utc.min(): raise AssertionError("reaction label is not mature before evaluation availability")
    x, z, prep = fit_design(train, evaluate, method); weights = 1.0 / train.groupby(["symbol", "session_day"], dropna=False)["label"].transform("size").to_numpy(float); model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, tol=1e-8, random_state=573).fit(x, train.label.to_numpy(int), sample_weight=weights); p = model.predict_proba(z)[:, 1]; path.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"model": model, "method": method, "preprocessing": prep}, path); saved = joblib.load(path); reload = saved["model"].predict_proba(z)[:, 1]; assert np.max(np.abs(p - reload)) <= 1e-12
    return p, {"method": method, "C": c, "train_n": int(len(train)), "eval_n": int(len(evaluate)), "train_target_end_max": train.reaction_end_utc.max().isoformat(), "eval_available_min": evaluate.available_utc.min().isoformat(), "weight_sum": float(weights.sum()), "preprocessing": prep, "reload_max_abs_error": float(np.max(np.abs(p - reload)))}


def run_article_models(rx):
    predictions = []; selections = []; fits = []
    for h in HORIZONS:
        end_col = f"reaction_{h}m_end_utc"; valid_col = f"reaction_{h}m_valid"; rx["label"] = (rx[f"reaction_{h}m"] > 0).astype(int); rx["reaction_end_utc"] = pd.to_datetime(rx[end_col], utc=True, errors="coerce")
        for method in ("AR0", "AR1"):
            inner = []
            for c in CS:
                for month in ("2018-03", "2018-04", "2018-05"):
                    for symbol in ("AAPL", "AMZN"):
                        ev = rx[(rx.symbol == symbol) & (rx.month == month) & (rx[valid_col] == 1)].copy()
                        train = rx[(rx.symbol == symbol) & (rx.month < month) & (rx[valid_col] == 1) & (rx.reaction_end_utc < ev.available_utc.min())].copy() if len(ev) else rx.iloc[0:0].copy()
                        if len(train) < 30 or ev.empty: continue
                        p, e = train_predict(train, ev, method, c, PRIVATE / "models" / f"inner_{h}_{method}_{symbol}_{month}_{c}.joblib"); fits.append({**e, "phase": "inner", "horizon_minutes": h, "symbol": symbol, "month": month}); inner.append({"horizon_minutes": h, "method": method, "C": c, "symbol": symbol, "month": month, **metric(ev.label, p)})
            choices = [(c, float(np.mean([r["BA"] for r in inner if r["C"] == c])) if [r for r in inner if r["C"] == c] else -1) for c in CS]; chosen = min(choices, key=lambda x: (-x[1], x[0]))[0]; selections.append({"horizon_minutes": h, "method": method, "C": chosen, "inner_scores": choices})
            for month, phase in [(m, "inner") for m in ("2018-03", "2018-04", "2018-05")] + [(m, "outer") for m in ("2018-06", "2018-07", "2018-08")]:
                for symbol in ("AAPL", "AMZN"):
                    ev = rx[(rx.symbol == symbol) & (rx.month == month) & (rx[valid_col] == 1)].copy(); train = rx[(rx.symbol == symbol) & (rx.month < month) & (rx[valid_col] == 1) & (rx.reaction_end_utc < ev.available_utc.min())].copy() if len(ev) else rx.iloc[0:0].copy()
                    if len(train) < 30 or ev.empty: continue
                    p, e = train_predict(train, ev, method, chosen, PRIVATE / "models" / f"{phase}_{h}_{method}_{symbol}_{month}.joblib"); fits.append({**e, "phase": phase, "horizon_minutes": h, "symbol": symbol, "month": month}); q = ev[["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "session_day", "label", end_col, "price_context_max_used_bar_end_utc", "price_context_complete"]].copy(); q["phase"] = phase; q["horizon_minutes"] = h; q["method"] = method; q["prediction"] = p; predictions.append(q)
    return pd.concat(predictions, ignore_index=True), selections, fits


def article_gate(pred, selections, public):
    """Apply the independent AR1 gate and record unavailable AR2 explicitly."""
    x = pred.copy()
    x["month"] = pd.to_datetime(x.available_utc, utc=True, format="mixed").dt.strftime("%Y-%m")
    rows = []
    for (phase, h, symbol, method, month), g in x.groupby(
        ["phase", "horizon_minutes", "symbol", "method", "month"]
    ):
        rows.append({
            "phase": phase, "horizon_minutes": int(h), "symbol": symbol,
            "method": method, "month": month, **metric(g.label, g.prediction),
            "valid_rows": int(len(g)),
            "context_complete_rate": float(g.price_context_complete.mean()),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(public / "article_reaction_metrics.csv", index=False)
    metrics.to_csv(public / "article_reaction_monthly.csv", index=False)
    promotions = []
    for h in HORIZONS:
        ar0 = metrics[(metrics.phase == "outer") & (metrics.horizon_minutes == h) & (metrics.method == "AR0")].set_index(["symbol", "month"])
        ar1 = metrics[(metrics.phase == "outer") & (metrics.horizon_minutes == h) & (metrics.method == "AR1")].set_index(["symbol", "month"])
        common = ar0.index.intersection(ar1.index)
        joined = ar0.loc[common].join(ar1.loc[common], lsuffix="_ar0", rsuffix="_ar1")
        joined["delta_BA"] = joined.BA_ar1 - joined.BA_ar0
        joined["delta_Brier"] = joined.Brier_ar1 - joined.Brier_ar0
        joined["delta_AUC"] = joined.AUC_ar1 - joined.AUC_ar0
        monthly = [{
            "symbol": symbol, "month": month,
            "delta_BA": float(r.delta_BA), "delta_Brier": float(r.delta_Brier),
            "delta_AUC": float(r.delta_AUC), "AR1_n": int(r.n_ar1),
            "AR0_n": int(r.n_ar0), "coverage": 1.0,
            "horizon_complete": True,
            "AR1_constant": bool(r.constant_ar1),
        } for (symbol, month), r in joined.iterrows()]
        month_df = pd.DataFrame(monthly)
        month_df.to_csv(public / f"promotion_h{h}m_monthly.csv", index=False)
        stock_rows = []
        for symbol, g in month_df.groupby("symbol"):
            stock_rows.append({
                "symbol": symbol,
                "mean_delta_BA": float(g.delta_BA.mean()),
                "max_delta_Brier": float(g.delta_Brier.max()),
                "positive_months": int((g.delta_BA > 0).sum()),
                "months": int(len(g)), "min_month_rows": int(g.AR1_n.min()),
                "coverage_min": float(g.coverage.min()),
                "horizon_complete": bool(g.horizon_complete.all()),
                "constant_direction": bool(g.AR1_constant.any()),
            })
        macro = {
            "macro_auc_ar0": float(ar0.loc[common].AUC.mean()),
            "macro_auc_ar1": float(ar1.loc[common].AUC.mean()),
            "macro_auc_delta": float(joined.delta_AUC.mean()),
            "positive_macro_months": int(
                (joined.groupby(level="month").delta_BA.mean() > 0).sum()
            ),
        }
        enough = bool(
            len(stock_rows) == 2 and len(month_df) == 6
            and all(r["months"] == 3 and r["min_month_rows"] >= 30
                    and r["coverage_min"] >= 0.8
                    and r["horizon_complete"] for r in stock_rows)
        )
        passes = bool(
            enough
            and all(r["mean_delta_BA"] >= 0.01 for r in stock_rows)
            and all(r["max_delta_Brier"] <= 0.002 for r in stock_rows)
            and all(r["positive_months"] >= 2 for r in stock_rows)
            and not any(r["constant_direction"] for r in stock_rows)
            and macro["macro_auc_delta"] > 0
            and macro["positive_macro_months"] >= 2
        )
        c_rows = [s for s in selections if s["horizon_minutes"] == h and s["method"] == "AR1"]
        promotions.append({
            "candidate": "AR1", "horizon_minutes": h,
            "C": c_rows[0]["C"] if c_rows else None,
            "gate_formula": "both stocks mean_delta_BA>=0.01; max_delta_Brier<=0.002; positive stock months>=2/3; macro AUC delta>0; positive macro BA months>=2; complete/coverage/no constant collapse",
            "stock_rows": stock_rows, "macro": macro,
            "passes": passes, "status": "PASS" if passes else "FAIL",
        })
    ar2 = {
        "status": "NOT_RUN_MODEL_UNAVAILABLE",
        "reason": "target-context FinBERT binary unavailable; no cached title vectors substituted",
        "eligible": False,
    }
    write_json(public / "promotion_gate.json", {
        "status": "COMPLETE_AR1_GATE_AR2_NOT_RUN",
        "outer_months": ["2018-06", "2018-07", "2018-08"],
        "promotions": promotions, "ar1_candidates": promotions, "ar2": ar2,
    })
    return metrics, promotions, ar2


def choose_horizon(promotions):
    passed = [x for x in promotions if x["passes"]]
    if not passed:
        return None
    def rank(x):
        gains = [r["mean_delta_BA"] for r in x["stock_rows"]]
        return (-min(gains), -x["macro"]["macro_auc_delta"], int(x["horizon_minutes"]))
    return sorted(passed, key=rank)[0]


def build_later_article_predictions(rx_all, horizon, c_value, public, private):
    """Freeze the selected article model at August for exposed windows."""
    freeze = pd.Timestamp("2018-09-01", tz="UTC")
    rx = rx_all.copy()
    end_col = f"reaction_{horizon}m_end_utc"
    valid_col = f"reaction_{horizon}m_valid"
    rx["label"] = (rx[f"reaction_{horizon}m"] > 0).astype(int)
    rx["reaction_end_utc"] = pd.to_datetime(rx[end_col], utc=True, errors="coerce")
    rows = []; fit_evidence = []
    for symbol in ("AAPL", "AMZN"):
        train = rx[(rx.symbol == symbol) & (rx.available_utc < freeze)
                   & (rx[valid_col] == 1) & (rx.reaction_end_utc < freeze)].copy()
        ev = rx[(rx.symbol == symbol) & (rx.available_utc >= freeze)
                & (rx[valid_col] == 1)].copy()
        if len(train) < 30 or ev.empty:
            continue
        p, evidence = train_predict(
            train, ev, "AR1", c_value,
            private / "models" / f"frozen_aug_{horizon}_AR1_{symbol}.joblib",
        )
        q = ev[["article_key", "target_pair_key", "symbol", "group_key", "available_utc",
                "session_day", end_col, "price_context_max_used_bar_end_utc",
                "price_context_complete"]].copy()
        q["horizon_minutes"] = horizon; q["method"] = "AR1_frozen_aug"
        q["phase"] = "frozen_aug"; q["prediction"] = p
        rows.append(q)
        fit_evidence.append({**evidence, "phase": "frozen_aug", "horizon_minutes": horizon, "symbol": symbol})
    if not rows:
        result = pd.DataFrame(columns=["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "prediction"])
    else:
        result = pd.concat(rows, ignore_index=True)
    result.to_csv(public / "reaction_later_predictions.csv", index=False)
    return result, fit_evidence


def fit_head(train, evaluate, columns, c_value, path=None):
    """Fit one small downstream head; every transform is fit on past rows."""
    from sklearn.impute import SimpleImputer
    a = train.dropna(subset=columns + ["label"]).copy()
    b = evaluate.dropna(subset=columns).copy()
    if a.empty or b.empty or len(a.label.unique()) < 2:
        return b.index, np.full(len(b), np.nan), {"train_n": len(a), "eval_n": len(b)}
    imp = SimpleImputer(strategy="median").fit(a[columns])
    scaler = StandardScaler().fit(imp.transform(a[columns]))
    xa = scaler.transform(imp.transform(a[columns])); xb = scaler.transform(imp.transform(b[columns]))
    model = LogisticRegression(C=float(c_value), solver="liblinear", max_iter=3000, random_state=573).fit(xa, a.label.to_numpy(int))
    p = model.predict_proba(xb)[:, 1]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "columns": columns, "imputer": imp, "scaler": scaler}, path)
    return b.index, p, {"train_n": len(a), "eval_n": len(b), "columns": columns}


def official_windows(score, selected_horizon, public, private):
    """Run the exact W0--W3 protocol after an article candidate passes."""
    inputs = pd.read_pickle(WORK / "paper_methods_4h" / "v1" / "inputs.pkl").copy()
    inputs["start_utc"] = pd.to_datetime(inputs.start_utc, utc=True)
    inputs["cutoff_utc"] = pd.to_datetime(inputs.cutoff_utc, utc=True)
    inputs["key"] = inputs.symbol + "|" + inputs.start_utc.astype(str)
    base = pd.read_csv(ROOT / "outputs" / "stock_goal60_4h" / "v1" / "predictions.csv")
    base["key"] = base.key.astype(str)
    base = base[["key", "phase", "label", "R1", "F1"]]
    windows = inputs.merge(base, on="key", how="left", validate="one_to_one", suffixes=("", "_goal"))
    windows["month"] = windows.start_utc.dt.strftime("%Y-%m")
    windows["phase"] = windows.phase_goal
    score = score.copy()
    if not score.empty:
        score["available_utc"] = pd.to_datetime(score.available_utc, utc=True, format="mixed")
    groups = {s: g.sort_values("available_utc") for s, g in score.groupby("symbol")} if not score.empty else {}
    manifest = []
    for row in windows.itertuples(index=False):
        g = groups.get(row.symbol, pd.DataFrame())
        if len(g):
            q = g[(g.available_utc <= row.cutoff_utc)
                  & (g.available_utc > row.cutoff_utc - pd.Timedelta(hours=24))]
        else:
            q = g
        count = int(len(q)); has = int(count > 0)
        if count:
            p = q.prediction.astype(float).to_numpy()
            mean = float(p.mean()); signed = p - 0.5
            max_signed = float(signed[np.argmax(np.abs(signed))])
            newest_age = float((row.cutoff_utc - q.available_utc.max()).total_seconds() / 3600.0)
        else:
            mean = 0.5; max_signed = 0.0; newest_age = 24.0
        manifest.append({
            "key": row.key, "symbol": row.symbol, "start_utc": row.start_utc.isoformat(),
            "cutoff_utc": row.cutoff_utc.isoformat(), "month": row.month,
            "phase": row.phase, "reaction_article_count": count,
            "log1p_reaction_article_count": float(np.log1p(count)),
            "newest_reaction_article_age_hours": newest_age,
            "has_reaction_article": has, "reaction_mean": mean,
            "reaction_max_abs_signed": max_signed,
        })
    manifest = pd.DataFrame(manifest)
    manifest.to_csv(public / "downstream_input_manifest.csv", index=False)
    windows = windows.merge(manifest, on=["key", "symbol", "month", "phase"], validate="one_to_one")
    windows["W0"] = windows.R1
    w1_cols = ["R1", "log1p_reaction_article_count", "newest_reaction_article_age_hours", "has_reaction_article"]
    w2_cols = ["R1", "reaction_mean", "reaction_max_abs_signed", "log1p_reaction_article_count", "newest_reaction_article_age_hours", "has_reaction_article"]
    w3_cols = ["F1"] + w2_cols
    specs = {"W1": w1_cols, "W2": w2_cols, "W3": w3_cols}
    selection = {}
    for name, cols in specs.items():
        scores = []
        for c in CS:
            fold = []
            for month in ("2018-04", "2018-05"):
                train = windows[(windows.phase == "train_forward_oof") & (windows.month < month)]
                ev = windows[(windows.phase == "train_forward_oof") & (windows.month == month)]
                for symbol in ("AAPL", "AMZN"):
                    tr = train[train.symbol == symbol]; ee = ev[ev.symbol == symbol]
                    idx, p, _ = fit_head(tr, ee, cols, c)
                    if len(idx): fold.append(metric(ee.loc[idx, "label"], p)["BA"])
            scores.append({"C": c, "mean_BA": float(np.mean(fold)) if fold else None, "folds": len(fold)})
        selection[name] = min(scores, key=lambda r: (-(r["mean_BA"] if r["mean_BA"] is not None else -1), r["C"]))["C"]
        selection[name + "_scores"] = scores
    # Outer June--August: train on March--May only.
    outputs = windows[["key", "symbol", "month", "phase", "label", "R1", "F1",
                       "reaction_mean", "reaction_max_abs_signed", "log1p_reaction_article_count",
                       "newest_reaction_article_age_hours", "has_reaction_article"]].copy()
    for name in specs: outputs[name] = np.nan
    outer = windows[(windows.phase == "train_forward_oof") & windows.month.isin(["2018-06", "2018-07", "2018-08"])]
    outer_train = windows[(windows.phase == "train_forward_oof") & windows.month.isin(["2018-03", "2018-04", "2018-05"])]
    exposed = windows[windows.phase.isin(["development", "later"])]
    exposed_train = windows[(windows.phase == "train_forward_oof") & windows.month <= "2018-08"]
    fit_records = []
    for name, cols in specs.items():
        for symbol in ("AAPL", "AMZN"):
            tr = outer_train[outer_train.symbol == symbol]; ev = outer[outer.symbol == symbol]
            idx, p, e = fit_head(tr, ev, cols, selection[name], private / "downstream_models" / f"outer_{name}_{symbol}.joblib")
            outputs.loc[idx, name] = p; fit_records.append({"scope": "outer", "method": name, "symbol": symbol, "C": selection[name], **e})
            tr = exposed_train[exposed_train.symbol == symbol]; ev = exposed[exposed.symbol == symbol]
            idx, p, e = fit_head(tr, ev, cols, selection[name], private / "downstream_models" / f"frozen_aug_{name}_{symbol}.joblib")
            outputs.loc[idx, name] = p; fit_records.append({"scope": "exposed_frozen_aug", "method": name, "symbol": symbol, "C": selection[name], **e})
    outputs.to_csv(public / "reaction_downstream_predictions.csv", index=False)
    metric_rows = []
    for phase in ("train_forward_oof", "development", "later"):
        for symbol in ("AAPL", "AMZN"):
            g = outputs[(outputs.phase == phase) & (outputs.symbol == symbol)]
            for name in ("W0", "W1", "W2", "W3"):
                h = g.dropna(subset=[name])
                if len(h): metric_rows.append({"phase": phase, "symbol": symbol, "method": name, **metric(h.label, h[name])})
    pd.DataFrame(metric_rows).to_csv(public / "reaction_downstream_metrics.csv", index=False)
    # Downstream gate is descriptive and uses only June--August; it never
    # chooses an article horizon or an exposed-period winner.
    gate_rows = []
    outer_out = outputs[(outputs.phase == "train_forward_oof") & outputs.month.isin(["2018-06", "2018-07", "2018-08"])]
    for candidate, matched in (("W2", "W0"), ("W3", "W1")):
        for (symbol, month), g in outer_out.groupby(["symbol", "month"]):
            h = g.dropna(subset=[candidate, matched])
            if len(h):
                a = metric(h.label, h[matched]); b = metric(h.label, h[candidate])
                gate_rows.append({"candidate": candidate, "matched_base": matched, "symbol": symbol, "month": month, "n": len(h), "delta_BA": b["BA"] - a["BA"], "delta_Brier": b["Brier"] - a["Brier"], "delta_AUC": (b["AUC"] - a["AUC"]) if a["AUC"] is not None and b["AUC"] is not None else None})
    gate_df = pd.DataFrame(gate_rows); gate_df.to_csv(public / "downstream_gate_monthly.csv", index=False)
    downstream_gate = []
    for candidate, matched in (("W2", "W0"), ("W3", "W1")):
        d = gate_df[gate_df.candidate == candidate]
        stock = d.groupby("symbol").delta_BA.mean() if len(d) else pd.Series(dtype=float)
        macro = d.groupby("month").delta_BA.mean() if len(d) else pd.Series(dtype=float)
        passed = bool(len(stock) == 2 and stock.min() >= .01 and stock.min() >= -.01 and (macro > 0).sum() >= 2)
        downstream_gate.append({"candidate": candidate, "matched_base": matched, "stock_mean_delta_BA": stock.to_dict(), "positive_macro_months": int((macro > 0).sum()), "passes_descriptive_gate": passed})
    write_json(public / "downstream_gate.json", {"status": "COMPLETE", "rows": downstream_gate, "selection_not_from_exposed": True})
    write_json(public / "downstream_selection.json", {"status": "COMPLETE", "selected_horizon": selected_horizon, "heads": selection, "fits": fit_records, "protocol": "W0=R1; W1=R1+coverage; W2=R1+five reaction features; W3=F1+R1+five reaction features"})
    return {"status": "COMPLETE", "rows": int(len(outputs)), "selected_horizon": selected_horizon, "head_selection": selection}


def main(public=PUBLIC, private=PRIVATE):
    if public.exists() and any(public.iterdir()): raise FileExistsError(public)
    if private.exists(): raise FileExistsError(private)
    public.mkdir(parents=True); (private / "models").mkdir(parents=True); t0 = time.monotonic()
    protocol_path = HERE / "PRE_REGISTRATION_v3.md"
    write_json(public / "protocol.json", {"protocol": "reaction_v3_corrected", "preregistration_sha256": sha(protocol_path), "run_sha256": sha(HERE / "run_reaction_probe_v3.py"), "horizons": list(HORIZONS), "article_gate": "AR1 independent; AR2 optional", "official_protocol": "W0=R1; W1=R1+coverage; W2=R1+five reaction features; W3=F1+R1+five reaction features"})
    rx_all = prepare()
    rx_train = rx_all[rx_all.available_utc < pd.Timestamp("2018-09-01", tz="UTC")].copy()
    pred, selections, fits = run_article_models(rx_train)
    pred.to_csv(public / "article_reaction_predictions.csv", index=False)
    metrics, promotions, ar2 = article_gate(pred, selections, public)
    chosen = choose_horizon(promotions)
    downstream = {"status": "NOT_RUN_NO_PROMOTED_ARTICLE_HORIZON"}
    if chosen is not None:
        later, later_fits = build_later_article_predictions(rx_all, int(chosen["horizon_minutes"]), float(chosen["C"]), public, private)
        fits.extend(later_fits)
        selected = pred[(pred.method == "AR1") & (pred.horizon_minutes == int(chosen["horizon_minutes"]))][["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "prediction"]].copy()
        selected["available_utc"] = pd.to_datetime(selected.available_utc, utc=True, format="mixed")
        score = pd.concat([selected, later[["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "prediction"]]], ignore_index=True)
        downstream = official_windows(score, int(chosen["horizon_minutes"]), public, private)
        write_json(public / "official_window_status.json", {"status": "COMPLETE_AR1_ELIGIBLE", "selected_article_candidate": "AR1", "selected_horizon_minutes": int(chosen["horizon_minutes"]), "protocol": "W0=R1; W1=R1+coverage; W2=R1+five reaction features; W3=F1+R1+five reaction features", "selection_source": "article_gate_only"})
    else:
        write_json(public / "official_window_status.json", {"status": "NOT_RUN_NO_PROMOTED_ARTICLE_HORIZON", "protocol": "W0=R1; W1=R1+coverage; W2=R1+five reaction features; W3=F1+R1+five reaction features"})
    write_json(public / "training_evidence.json", {"status": "ARTICLE_AR0_AR1_COMPLETE_AR2_NOT_RUN", "fits": fits, "fit_count": len(fits), "selection": selections, "AR2": ar2, "runtime_seconds": time.monotonic() - t0})
    write_json(public / "reaction_probe_gate.json", {"status": "COMPLETE", "promotion": promotions, "AR2": ar2, "selected_article_candidate": "AR1" if chosen is not None else None, "selected_horizon_minutes": int(chosen["horizon_minutes"]) if chosen is not None else None, "downstream": downstream})
    print(json.dumps({"status": "COMPLETE", "rows": len(pred), "fit_count": len(fits), "selected": int(chosen["horizon_minutes"]) if chosen is not None else None, "downstream": downstream.get("status")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--public", type=Path, default=PUBLIC); ap.add_argument("--private", type=Path, default=PRIVATE); a = ap.parse_args(); main(a.public, a.private)
