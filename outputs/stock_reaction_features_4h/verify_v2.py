"""Verify the repaired v2 reaction run, including its scientific contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = HERE / "v2"
PRIVATE = ROOT / "work" / "stock-data" / "reaction_features_4h" / "v2"
HORIZONS = (30, 60, 120, 240)


def load_builder():
    spec = importlib.util.spec_from_file_location("reaction_builder_v2_verify", HERE / "build_reaction_dataset_v2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import v2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    gate = json.loads((PUBLIC / "reaction_gate.json").read_text())
    probe = json.loads((PUBLIC / "reaction_probe_gate.json").read_text())
    rx = pd.read_pickle(PRIVATE / "article_reactions_all_time.pkl")
    canonical = pd.read_pickle(PRIVATE / "canonical_pairs_all_time.pkl")
    pred_path = PUBLIC / "reaction_predictions.csv"
    checks = {}

    checks["feasibility_pass"] = gate.get("status") == "PASS"
    checks["target_pair_key_unique"] = not canonical.target_pair_key.duplicated().any()
    checks["target_pair_key_present"] = set(canonical.columns) >= {"article_key", "symbol", "target_pair_key"}
    checks["reaction_columns_present"] = all(
        f"reaction_{h}m_{suffix}" in rx.columns
        for h in HORIZONS for suffix in ("start_utc", "end_utc", "matured", "valid")
    )
    checks["context_columns_present"] = all(
        f"pre_{m}m_{suffix}" in rx.columns
        for m in (5, 15, 30, 60) for suffix in ("valid", "used_bar_end_utc")
    ) and {"price_context_max_used_bar_end_utc", "price_context_complete"} <= set(rx.columns)

    available = pd.to_datetime(rx.available_utc, utc=True)
    all_context_safe = True
    for m in (5, 15, 30, 60):
        used = pd.to_datetime(rx[f"pre_{m}m_used_bar_end_utc"], utc=True, errors="coerce")
        valid = rx[f"pre_{m}m_valid"].astype(bool)
        all_context_safe &= bool((used[valid] <= available[valid]).all())
    checks["completed_bar_context_safe"] = all_context_safe
    max_used = pd.to_datetime(rx.price_context_max_used_bar_end_utc, utc=True, errors="coerce")
    checks["row_max_context_safe"] = bool((max_used.dropna() <= available[max_used.notna()]).all())

    builder = load_builder()
    sessions = builder.schedule(); sessions["day"] = sessions.open.dt.strftime("%Y-%m-%d")
    time_safe = True
    for h in HORIZONS:
        valid = rx[f"reaction_{h}m_valid"].astype(bool)
        start = pd.to_datetime(rx[f"reaction_{h}m_start_utc"], utc=True, errors="coerce")
        end = pd.to_datetime(rx[f"reaction_{h}m_end_utc"], utc=True, errors="coerce")
        time_safe &= bool((start[valid] >= available[valid]).all())
        time_safe &= bool((end[valid] > start[valid]).all())
        for _, row in rx.loc[valid, ["symbol", "session_day"]].drop_duplicates().iterrows():
            close = sessions[sessions.day == row.session_day].close
            if len(close):
                mask = valid & (rx.symbol == row.symbol) & (rx.session_day == row.session_day)
                time_safe &= bool((end[mask] <= close.iloc[0]).all())
    checks["reaction_start_maturity_safe"] = time_safe

    future_replay = False
    sample = rx[rx.price_context_complete.astype(bool)].head(1)
    if len(sample):
        r = sample.iloc[0]
        bars = builder._load_bars(r.symbol)
        session = sessions[sessions.day == r.session_day].iloc[0]
        t = pd.Timestamp(r.available_utc)
        before = builder._completed_context(bars, session, t, 60)
        future = bars.copy()
        future.loc[future.index >= t, "close"] = future.loc[future.index >= t, "close"] * 3.0 + 7.0
        after = builder._completed_context(future, session, t, 60)
        future_replay = before == after
    checks["future_price_perturbation_context_invariant"] = bool(future_replay)

    if pred_path.exists():
        pcols = set(pd.read_csv(pred_path, nrows=1).columns)
        checks["public_text_not_published"] = not bool(pcols & {"body", "title", "text", "context"})
        pred = pd.read_csv(pred_path)
        checks["prediction_pair_keys_unique"] = not pred.duplicated(["article_key", "target_pair_key", "horizon_minutes", "method"]).any()
    else:
        checks["public_text_not_published"] = True
        checks["prediction_pair_keys_unique"] = True

    evidence = json.loads((PUBLIC / "reaction_training_evidence.json").read_text())
    fits = evidence.get("fits", [])
    checks["fit_count_instrumented"] = evidence.get("fit_count") == len(fits) and len(fits) == len(list((PRIVATE / "models").glob("*.joblib")))
    checks["fold_preprocessing_recorded"] = bool(fits) and all(
        int(x.get("preprocessing", {}).get("numeric_imputer_fit_rows", 0)) == int(x.get("train_n", 0))
        for x in fits
    )
    checks["ar2_explicit_not_run"] = probe.get("AR2", {}).get("status") == "NOT_RUN_MODEL_BINARY_UNAVAILABLE"

    promotions = json.loads((PUBLIC / "promotion_gate.json").read_text())
    checks["promotion_gate_complete"] = all(
        {"horizon_minutes", "stock_rows", "gate_formula", "passes", "status"} <= set(x)
        and all({"symbol", "mean_delta_BA", "max_delta_Brier", "positive_months", "months", "min_month_rows", "coverage_min", "horizon_complete"} <= set(row) for row in x["stock_rows"])
        for x in promotions.get("promotions", [])
    ) and len(promotions.get("promotions", [])) == 4
    checks["protocol_fingerprint_present"] = bool(probe.get("protocol_sha256"))

    if probe.get("downstream", {}).get("status") == "COMPLETE":
        d = pd.read_csv(PUBLIC / "reaction_downstream_predictions.csv")
        checks["w3_exact_fallback"] = bool((d.loc[d.W3_fallback, "W3"] == d.loc[d.W3_fallback, "W0"]).all())
        checks["w0_present"] = {"W0", "W1", "W2", "W3"} <= set(d.columns)
    else:
        checks["w3_exact_fallback"] = True
        checks["w0_present"] = True

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate_status": gate.get("status"),
        "probe_status": probe.get("status"),
        "ar2": probe.get("AR2"),
        "fit_count": len(fits),
    }
    (PUBLIC / "verification_v2.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
