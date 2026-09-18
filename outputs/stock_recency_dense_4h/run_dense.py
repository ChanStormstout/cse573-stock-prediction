"""Evaluate official controls and augmented 30-minute price windows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import INPUTS, OLD, PUBLIC, R1_COLS, WORK, dump, load_data, metric_rows, scores, sha, standardize_fit, standardize_transform


def fit(train, evaluate, weights, name, private):
    if train.end_utc.max() >= evaluate.cutoff_utc.min():
        raise AssertionError("dense model saw a future training label")
    mean, scale = standardize_fit(train, R1_COLS)
    x = standardize_transform(train, R1_COLS, mean, scale)
    z = standardize_transform(evaluate, R1_COLS, mean, scale)
    model = LogisticRegression(C=0.1, solver="liblinear", max_iter=3000, tol=1e-8, random_state=573)
    model.fit(x, train.label.to_numpy(dtype=int), sample_weight=np.asarray(weights, dtype=float))
    p = model.predict_proba(z)[:, 1]
    path = private / "models" / f"{name}.joblib"; path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mean": mean, "scale": scale, "columns": R1_COLS, "model": model}, path)
    loaded = joblib.load(path); np.testing.assert_allclose(p, loaded["model"].predict_proba(z)[:, 1], atol=1e-12, rtol=0)
    return p, {"name": name, "train_n": int(len(train)), "eval_n": int(len(evaluate)), "weight_sum": float(np.sum(weights)), "model_sha256": sha(path), "reload_max_abs_error": 0.0}


def run(public: Path, private: Path):
    start = time.monotonic()
    dense = pd.read_pickle(private / "dense_windows.pkl")
    official = load_data()
    reference = pd.read_csv(PUBLIC.parent / "stock_goal60_4h" / "v1" / "predictions.csv").set_index("key")
    dense_off = dense[dense.official == 1].copy().set_index("key")
    if not set(official.key[official.month <= "2018-08"]).issubset(dense_off.index):
        raise AssertionError("dense official coverage is incomplete")
    fits = []; chunks = []; outer = []
    # Outer months evaluate only official rows; augmented rows are training data.
    for month in ["2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08"]:
        for symbol in ("AAPL", "AMZN"):
            ev = dense[(dense.symbol == symbol) & (dense.month == month) & (dense.official == 1)].copy()
            off_train = dense[(dense.symbol == symbol) & (dense.month < month) & (dense.official == 1)].copy()
            dense_train = dense[(dense.symbol == symbol) & (dense.month < month)].copy()
            if ev.empty: continue
            row = ev[["key", "symbol", "day", "month", "label", "target_return"]].copy(); row["phase"] = "train_forward_oof"
            # D0 is the exact previously saved R1 prediction for parity.
            row["D0_equal"] = reference.loc[row.key, "R1"].to_numpy(dtype=float)
            p0d, e0 = fit(off_train, ev, off_train.official_day_weight.to_numpy(), f"{symbol}_{month}_D0_day", private); fits.append({**e0, "method": "D0_day", "symbol": symbol, "month": month}); row["D0_day"] = p0d
            p1, e1 = fit(dense_train, ev, dense_train.day_weight.to_numpy(), f"{symbol}_{month}_D1", private); fits.append({**e1, "method": "D1", "symbol": symbol, "month": month}); row["D1"] = p1
            chunks.append(row)
            for method in ["D0_equal", "D0_day", "D1"]:
                outer.append({"method": method, "symbol": symbol, "month": month, **scores(ev.label, row[method])})
    # Final Jan-Aug fit and separate exposed development/later scores.
    for symbol in ("AAPL", "AMZN"):
        ev = official[(official.symbol == symbol) & official.phase.isin(["development", "later"])].copy()
        off_train = dense[(dense.symbol == symbol) & (dense.month < "2018-09") & (dense.official == 1)].copy()
        dense_train = dense[(dense.symbol == symbol) & (dense.month < "2018-09")].copy()
        # Features for post-August official rows are the original cutoff-safe R1 columns.
        ev_features = ev.copy()
        row = ev[["key", "symbol", "day", "month", "label", "target_return", "phase"]].copy()
        row["D0_equal"] = reference.loc[row.key, "R1"].to_numpy(dtype=float)
        p0d, e0 = fit(off_train, ev_features, off_train.official_day_weight.to_numpy(), f"{symbol}_final_D0_day", private); fits.append({**e0, "method": "D0_day", "symbol": symbol, "month": "final"}); row["D0_day"] = p0d
        p1, e1 = fit(dense_train, ev_features, dense_train.day_weight.to_numpy(), f"{symbol}_final_D1", private); fits.append({**e1, "method": "D1", "symbol": symbol, "month": "final"}); row["D1"] = p1
        chunks.append(row)
    preds = pd.concat(chunks, ignore_index=True)
    preds.to_csv(public / "dense_predictions.csv", index=False)
    metrics, monthly = metric_rows(preds, ["D0_equal", "D0_day", "D1"])
    metrics.to_csv(public / "dense_metrics.csv", index=False); monthly.to_csv(public / "dense_monthly_metrics.csv", index=False)
    out = pd.DataFrame(outer)
    base = out[out.method == "D0_day"]; new = out[out.method == "D1"]
    q = base.merge(new, on=["symbol", "month"], suffixes=("_base", "_new")); q["delta_BA"] = q.BA_new - q.BA_base
    by_stock = q.groupby("symbol").delta_BA.mean(); by_month = q.groupby("month").delta_BA.mean()
    gate = {"baseline": "D0_day", "candidate": "D1", "AAPL_delta_BA": float(by_stock.get("AAPL", np.nan)), "AMZN_delta_BA": float(by_stock.get("AMZN", np.nan)), "macro_delta_BA": float(q.delta_BA.mean()), "positive_outer_months": int((by_month > 0).sum()), "passes": bool(by_stock.min() >= 0.01 and by_stock.max() >= -0.01 and (by_month > 0).sum() >= 2)}
    rec = json.loads((PUBLIC.parent / "stock_recency_dense_4h" / "v2" / "recency_selection.json").read_text()) if (PUBLIC.parent / "stock_recency_dense_4h" / "v2" / "recency_selection.json").exists() else {"gate": []}
    r1_pass = any(x.get("method") == "R1" and x.get("passes") for x in rec.get("gate", []))
    summary = {"status": "COMPLETE", "dense_gate": gate, "D2_status": "RUN_NOT_RUN_RECENCY_GATE_FAILED" if not r1_pass else "ELIGIBLE_TO_RUN", "D2_reason": "R1 recency gate did not pass" if not r1_pass else "R1 gate passed", "rows": int(len(dense)), "official_rows": int(dense.official.sum()), "fits": len(fits), "runtime_seconds": time.monotonic() - start, "text_extension": "NOT_RUN_DENSE_GATE_FAILED" if not gate["passes"] else "ELIGIBLE_BUT_NOT_RUN_IN_THIS_BOUNDED_RUN"}
    dump(public / "dense_training_summary.json", summary); dump(public / "dense_training_evidence.json", {"fits": fits, "summary": summary, "private_model_dir": str(private / "models")})
    out.to_csv(public / "dense_outer_scores.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v3"); p.add_argument("--private", default="work/stock-data/recency_dense_4h/v3"); a = p.parse_args(); run(Path(a.public), Path(a.private))
