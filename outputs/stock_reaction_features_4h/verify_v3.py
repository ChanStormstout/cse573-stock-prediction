"""Verify the v3 reviewer-protocol article reaction audit.

The v3 run intentionally stops before AR2 and before official W0--W3.  The
large source corpus is kept in the private v2 preparation directory; this
verifier checks its time-safety contracts and the public v3 audit outputs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = HERE / "v3"
PRIVATE = ROOT / "work" / "stock-data" / "reaction_features_4h" / "v3"
SOURCE_PRIVATE = ROOT / "work" / "stock-data" / "reaction_features_4h" / "v2"
HORIZONS = (30, 60, 120, 240)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "reaction_builder_v2_verify_v3", HERE / "build_reaction_dataset_v2.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import reaction builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rx = pd.read_pickle(SOURCE_PRIVATE / "article_reactions_all_time.pkl")
    canonical = pd.read_pickle(SOURCE_PRIVATE / "canonical_pairs_all_time.pkl")
    builder = load_builder()
    sessions = builder.schedule()
    sessions["day"] = sessions.open.dt.strftime("%Y-%m-%d")
    available = pd.to_datetime(rx.available_utc, utc=True)
    checks: dict[str, bool] = {}

    checks["target_pair_key_present"] = {
        "article_key", "symbol", "target_pair_key", "group_key"
    } <= set(canonical.columns)
    checks["target_pair_key_unique"] = not canonical.target_pair_key.duplicated().any()
    checks["reaction_columns_present"] = all(
        f"reaction_{h}m_{suffix}" in rx.columns
        for h in HORIZONS
        for suffix in ("start_utc", "end_utc", "matured", "valid")
    )
    checks["context_columns_present"] = all(
        f"pre_{m}m_{suffix}" in rx.columns
        for m in (5, 15, 30, 60)
        for suffix in ("valid", "used_bar_end_utc")
    ) and {"price_context_max_used_bar_end_utc", "price_context_complete"} <= set(rx.columns)

    context_safe = True
    for m in (5, 15, 30, 60):
        used = pd.to_datetime(rx[f"pre_{m}m_used_bar_end_utc"], utc=True, errors="coerce")
        valid = rx[f"pre_{m}m_valid"].astype(bool)
        context_safe &= bool((used[valid] <= available[valid]).all())
    max_used = pd.to_datetime(rx.price_context_max_used_bar_end_utc, utc=True, errors="coerce")
    context_safe &= bool((max_used.dropna() <= available[max_used.notna()]).all())
    checks["completed_bar_context_safe"] = context_safe

    maturity_safe = True
    for h in HORIZONS:
        valid = rx[f"reaction_{h}m_valid"].astype(bool)
        start = pd.to_datetime(rx[f"reaction_{h}m_start_utc"], utc=True, errors="coerce")
        end = pd.to_datetime(rx[f"reaction_{h}m_end_utc"], utc=True, errors="coerce")
        maturity_safe &= bool((start[valid] >= available[valid]).all())
        maturity_safe &= bool((end[valid] > start[valid]).all())
        for _, row in rx.loc[valid, ["symbol", "session_day"]].drop_duplicates().iterrows():
            close = sessions[sessions.day == row.session_day].close
            if len(close):
                mask = valid & (rx.symbol == row.symbol) & (rx.session_day == row.session_day)
                maturity_safe &= bool((end[mask] <= close.iloc[0]).all())
    checks["reaction_start_maturity_safe"] = maturity_safe

    sample = rx[rx.price_context_complete.astype(bool)].head(1)
    future_replay = False
    if len(sample):
        row = sample.iloc[0]
        bars = builder._load_bars(row.symbol)
        session = sessions[sessions.day == row.session_day].iloc[0]
        cutoff = pd.Timestamp(row.available_utc)
        before = builder._completed_context(bars, session, cutoff, 60)
        future = bars.copy()
        future.loc[future.index >= cutoff, "close"] = future.loc[future.index >= cutoff, "close"] * 3.0 + 7.0
        after = builder._completed_context(future, session, cutoff, 60)
        future_replay = before == after
    checks["future_price_perturbation_context_invariant"] = bool(future_replay)

    predictions = PUBLIC / "article_reaction_predictions.csv"
    pred = pd.read_csv(predictions)
    checks["public_text_not_published"] = not bool(
        {"body", "title", "text", "context"} & set(pred.columns)
    )
    checks["prediction_pair_keys_unique"] = not pred.duplicated(
        ["article_key", "target_pair_key", "horizon_minutes", "method"]
    ).any()
    checks["prediction_horizons_are_registered"] = set(pred.horizon_minutes.unique()) <= {60, 240}
    checks["prediction_phases_are_registered"] = set(pred.phase.unique()) <= {"inner", "outer"}

    evidence = json.loads((PUBLIC / "training_evidence.json").read_text())
    fits = evidence.get("fits", [])
    checks["fit_count_instrumented"] = (
        evidence.get("fit_count") == len(fits)
        and len(fits) == len(list((PRIVATE / "models").glob("*.joblib")))
    )
    checks["fold_numeric_preprocessing_recorded"] = bool(fits) and all(
        int(row.get("preprocessing", {}).get("numeric_imputer_fit_rows", 0))
        == int(row.get("train_n", 0))
        for row in fits
    )
    checks["ar2_explicit_not_run"] = evidence.get("AR2", {}).get("status") == "NOT_RUN_MODEL_UNAVAILABLE"
    gate = json.loads((PUBLIC / "promotion_gate.json").read_text())
    checks["promotion_gate_recorded"] = gate.get("status") == "COMPLETE_AR1_GATE_AR2_NOT_RUN"
    official = json.loads((PUBLIC / "official_window_status.json").read_text())
    checks["official_window_not_run"] = official.get("status") == "NOT_RUN_NO_PROMOTED_ARTICLE_HORIZON"
    checks["no_official_window_outputs"] = not any(
        (PUBLIC / name).exists()
        for name in ("reaction_downstream_predictions.csv", "reaction_downstream_metrics.csv")
    )

    metrics = pd.read_csv(PUBLIC / "article_reaction_metrics.csv")
    checks["monthly_metrics_have_distinct_months"] = (
        len(metrics) == 48
        and metrics[["phase", "horizon_minutes", "symbol", "method", "month"]]
        .duplicated()
        .sum() == 0
    )
    checks["protocol_fingerprint_present"] = (PUBLIC / "protocol.json").exists()

    result = {
        "status": "PASS" if all(bool(v) for v in checks.values()) else "FAIL",
        "checks": {k: bool(v) for k, v in checks.items()},
        "fit_count": len(fits),
        "prediction_rows": len(pred),
        "ar2": evidence.get("AR2"),
        "promotion_status": gate.get("status"),
        "official_window_status": official.get("status"),
    }
    (PUBLIC / "verification_v3.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
