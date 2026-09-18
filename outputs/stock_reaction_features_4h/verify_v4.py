"""Verify v4 reaction data contracts and gate evidence without article text."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work" / "stock-data"
PUBLIC = HERE / "v4"
PRIVATE = WORK / "reaction_features_4h" / "v4"
HORIZONS = (30, 60, 120, 240)
CONTINUOUS = [
    f"pre_{minutes}m_{kind}"
    for minutes in (5, 15, 30, 60)
    for kind in ("return", "rv")
] + ["minutes_from_open"]
VALIDITY_FLAGS = [f"pre_{minutes}m_valid" for minutes in (5, 15, 30, 60)]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "reaction_builder_v4_verify", HERE / "build_reaction_dataset_v4.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import reaction v4 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_future_replay(builder, reactions, sessions) -> bool:
    sample = reactions[reactions.price_context_complete.astype(bool)].head(1)
    if sample.empty:
        return False
    row = sample.iloc[0]
    base = builder.load_base()
    bars = base._load_bars(row.symbol)
    session = sessions[sessions.day == row.session_day].iloc[0]
    cutoff = pd.Timestamp(row.available_utc)
    before = builder._completed_context_v4(bars, session, cutoff, 60)
    future = bars.copy()
    future.loc[future.index >= cutoff, "close"] = (
        future.loc[future.index >= cutoff, "close"] * 3.0 + 7.0
    )
    after = builder._completed_context_v4(future, session, cutoff, 60)
    return bool(
        before["valid"] == after["valid"]
        and before["used_bar_end_utc"] == after["used_bar_end_utc"]
        and np.isclose(before["return"], after["return"])
        and np.isclose(before["rv"], after["rv"])
    )


def main() -> None:
    reactions = pd.read_pickle(PRIVATE / "article_reactions_all_time.pkl")
    canonical = pd.read_pickle(PRIVATE / "canonical_pairs_all_time.pkl")
    available = pd.to_datetime(reactions.available_utc, utc=True)
    builder = load_builder()
    base = builder.load_base()
    sessions = base.schedule()
    sessions["day"] = sessions.open.dt.strftime("%Y-%m-%d")
    checks: dict[str, bool] = {}

    checks["all_registered_horizons_present"] = all(
        f"reaction_{horizon}m_{suffix}" in reactions.columns
        for horizon in HORIZONS
        for suffix in ("start_utc", "end_utc", "matured", "valid")
    )
    checks["canonical_target_pair_keys_unique"] = (
        {"article_key", "symbol", "target_pair_key", "group_key"} <= set(canonical.columns)
        and not canonical.target_pair_key.duplicated().any()
    )
    checks["required_numeric_and_validity_columns_present"] = (
        set(CONTINUOUS + VALIDITY_FLAGS) <= set(reactions.columns)
    )
    context_safe = True
    missingness_safe = True
    for minutes in (5, 15, 30, 60):
        valid = reactions[f"pre_{minutes}m_valid"].astype(int)
        used = pd.to_datetime(
            reactions[f"pre_{minutes}m_used_bar_end_utc"], utc=True, errors="coerce"
        )
        ret = reactions[f"pre_{minutes}m_return"].to_numpy(float)
        rv = reactions[f"pre_{minutes}m_rv"].to_numpy(float)
        context_safe &= bool((used[valid.eq(1)] <= available[valid.eq(1)]).all())
        missingness_safe &= bool(
            np.isnan(ret[valid.eq(0).to_numpy()]).all()
            and np.isnan(rv[valid.eq(0).to_numpy()]).all()
            and np.isfinite(ret[valid.eq(1).to_numpy()]).all()
            and np.isfinite(rv[valid.eq(1).to_numpy()]).all()
            and set(valid.unique()) <= {0, 1}
        )
    max_used = pd.to_datetime(
        reactions.price_context_max_used_bar_end_utc, utc=True, errors="coerce"
    )
    context_safe &= bool((max_used.dropna() <= available[max_used.notna()]).all())
    checks["completed_bar_context_safe"] = context_safe
    checks["missing_context_is_nan_with_binary_flags"] = missingness_safe

    reaction_safe = True
    for horizon in HORIZONS:
        valid = reactions[f"reaction_{horizon}m_valid"].astype(bool)
        start = pd.to_datetime(reactions[f"reaction_{horizon}m_start_utc"], utc=True, errors="coerce")
        end = pd.to_datetime(reactions[f"reaction_{horizon}m_end_utc"], utc=True, errors="coerce")
        reaction_safe &= bool((start[valid] >= available[valid]).all() and (end[valid] > start[valid]).all())
        for _, item in reactions.loc[valid, ["symbol", "session_day"]].drop_duplicates().iterrows():
            close = sessions[sessions.day == item.session_day].close
            if len(close):
                mask = valid & reactions.symbol.eq(item.symbol) & reactions.session_day.eq(item.session_day)
                reaction_safe &= bool((end[mask] <= close.iloc[0]).all())
    checks["reaction_start_and_maturity_safe"] = reaction_safe
    checks["future_price_perturbation_context_invariant"] = context_future_replay(
        builder, reactions, sessions
    )

    build = json.loads((PUBLIC / "build_evidence.json").read_text())
    private_cards = PRIVATE / "ambiguous_AAPL_acronym_review_cards.jsonl"
    checks["aapl_acronym_audit_recorded"] = (
        "ambiguous_AAPL_acronym_rejected_count" in build
        and private_cards.exists()
        and int(build["private_ambiguous_AAPL_review_card_count"])
        == len(private_cards.read_text().splitlines())
    )

    prediction = pd.read_csv(PUBLIC / "article_reaction_predictions.csv", low_memory=False)
    checks["public_article_text_not_published"] = not bool(
        {"body", "title", "text", "context"} & set(prediction.columns)
    )
    checks["prediction_horizons_exactly_registered"] = set(prediction.horizon_minutes.unique()) == set(HORIZONS)
    checks["prediction_keys_unique"] = not prediction.duplicated(
        ["article_key", "target_pair_key", "horizon_minutes", "method"]
    ).any()
    checks["prediction_phases_registered"] = set(prediction.phase.unique()) <= {"inner", "outer"}

    evidence = json.loads((PUBLIC / "training_evidence.json").read_text())
    fits = evidence.get("fits", [])
    model_count = len(list((PRIVATE / "models").glob("*.joblib")))
    checks["fit_count_instrumented"] = evidence.get("fit_count") == len(fits) == model_count
    checks["past_fold_preprocessing_recorded"] = bool(fits) and all(
        x.get("preprocessing", {}).get("continuous_columns") == CONTINUOUS
        and x.get("preprocessing", {}).get("binary_validity_columns") == VALIDITY_FLAGS
        and x.get("preprocessing", {}).get("binary_validity_transform") == "unscaled explicit 0/1 indicators"
        and int(x.get("preprocessing", {}).get("numeric_imputer_fit_rows", 0)) == int(x.get("train_n", 0))
        and int(x.get("preprocessing", {}).get("numeric_scaler_fit_rows", 0)) == int(x.get("train_n", 0))
        for x in fits
    )
    checks["ar2_explicitly_not_run"] = evidence.get("AR2", {}).get("status") == "NOT_RUN_MODEL_UNAVAILABLE"

    gate = json.loads((PUBLIC / "promotion_gate.json").read_text())
    candidates = gate.get("ar1_candidates", [])
    checks["all_horizon_promotion_gates_recorded"] = (
        len(candidates) == 4
        and {int(x["horizon_minutes"]) for x in candidates} == set(HORIZONS)
        and all("both stocks mean_delta_BA>=0.01" in x.get("gate_formula", "") for x in candidates)
    )
    official = json.loads((PUBLIC / "official_window_status.json").read_text())
    passed = [x for x in candidates if x.get("passes")]
    if passed:
        expected = {f"official_AR1_h{int(x['horizon_minutes'])}m" for x in passed}
        actual = {p.name for p in PUBLIC.glob("official_AR1_h*m") if p.is_dir()}
        checks["official_windows_follow_article_gate_only"] = (
            official.get("status") == "COMPLETE_FOR_EACH_PROMOTED_AR1" and actual == expected
        )
    else:
        checks["official_windows_follow_article_gate_only"] = (
            official.get("status") == "NOT_RUN_NO_PROMOTED_ARTICLE_HORIZON"
            and not list(PUBLIC.glob("official_AR1_h*m"))
        )
    checks["protocol_fingerprint_present"] = (PUBLIC / "protocol.json").exists()

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "article_prediction_rows": int(len(prediction)),
        "fit_count": int(len(fits)),
        "promotion_status": gate.get("status"),
        "official_window_status": official.get("status"),
    }
    (PUBLIC / "verification_v4.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
