"""Preflight contracts for Phase B. This file never runs G0--G3 replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import STATE_COLUMNS, design_matrix, map_advantage
from prepare import load_inputs, validate_expert_evidence
from run_gate import fit_controller, prediction_rows, state_fit, state_transform

HERE = Path(__file__).resolve().parent
OUT = HERE / "preflight_verification.json"
FORWARD_MONTHS = [f"2018-{month:02d}" for month in range(3, 9)]


def synthetic_rows(aapl_rows: int, amzn_rows: int, day_count: int) -> pd.DataFrame:
    records = []
    for symbol, count in (("AAPL", aapl_rows), ("AMZN", amzn_rows)):
        for index in range(count):
            record = {
                "symbol": symbol,
                "day": f"2018-01-{(index % day_count) + 1:02d}",
                "eligible_news": 1,
                "advantage": float((index % 5) - 2) / 20.0,
                "p_price": 0.4 + 0.01 * (index % 6),
                "p_text": 0.6 - 0.01 * (index % 6),
                "has_news": 1,
                "key": f"{symbol}-{index}",
                "month": "2018-01",
                "phase": "synthetic",
                "start_utc": pd.Timestamp("2018-01-02", tz="UTC"),
                "end_utc": pd.Timestamp("2018-01-02 04:00", tz="UTC"),
                "cutoff_utc": pd.Timestamp("2018-01-02", tz="UTC"),
                "label": index % 2,
            }
            record.update({column: float(index % 3) for column in STATE_COLUMNS})
            records.append(record)
    return pd.DataFrame(records)


def fallback_checks() -> dict[str, bool]:
    evaluation = synthetic_rows(2, 2, 2)
    insufficient = synthetic_rows(3, 3, 3)
    r1 = fit_controller(insufficient, evaluation, "G0", "synthetic_r1")
    r1_rows = prediction_rows(evaluation, r1, "G0", "synthetic_r1")
    checks = {
        "exact_r1_support_fallback": bool(
            r1["force_price"]
            and np.array_equal(r1_rows.p_final.to_numpy(float), evaluation.p_price.to_numpy(float))
            and np.array_equal(r1_rows.w_text.to_numpy(float), np.zeros(len(evaluation)))
        )
    }
    g1 = fit_controller(synthetic_rows(60, 0, 20), evaluation, "G1", "synthetic_g1")
    checks["g1_to_g0_fallback"] = bool(
        g1["fallback"] == "G0_PER_STOCK_SUPPORT_BELOW_20" and not g1["force_price"]
    )
    g3_to_g2 = fit_controller(synthetic_rows(50, 10, 20), evaluation, "G3", "synthetic_g3")
    checks["g3_to_g2_fallback"] = bool(
        g3_to_g2["fallback"] == "G2_PER_STOCK_SUPPORT_BELOW_20"
        and g3_to_g2["variant"] == "G2"
        and not g3_to_g2["force_price"]
    )
    chained = fit_controller(synthetic_rows(5, 3, 3), evaluation, "G3", "synthetic_chain")
    chained_rows = prediction_rows(evaluation, chained, "G3", "synthetic_chain")
    checks["g3_to_g2_to_r1_chained_fallback"] = bool(
        chained["force_price"]
        and np.array_equal(chained_rows.p_final.to_numpy(float), evaluation.p_price.to_numpy(float))
    )
    return checks


def real_preparation_checks(frame: pd.DataFrame) -> tuple[dict[str, bool], list[dict]]:
    checks: dict[str, bool] = {}
    expected_advantage = (
        (frame.label.to_numpy(float) - frame.p_text.to_numpy(float)) ** 2
        - (frame.label.to_numpy(float) - frame.p_price.to_numpy(float)) ** 2
    )
    checks["canonical_key_count_1607"] = len(frame) == 1607 and not frame.key.duplicated().any()
    checks["advantage_present_and_recomputed"] = (
        "advantage" in frame.columns
        and float(np.nanmax(np.abs(frame.advantage.to_numpy(float) - expected_advantage))) <= 1e-15
        and np.isfinite(frame.loc[frame.eligible_news.eq(1), "advantage"]).all()
    )
    checks["state_columns_exact"] = list(STATE_COLUMNS) == [
        "recent_return_15", "recent_return_60", "recent_rv_15", "recent_rv_60",
        "minutes_from_open", "overnight_gap", "overnight_gap_missing",
        "log_news_count", "abs_expert_gap", "abs_price_centered",
    ]
    checks["no_forbidden_state_feature"] = not bool(
        set(STATE_COLUMNS) & {"label", "end_utc", "target_return", "p_final", "advantage"}
    )
    records = []
    all_good = True
    for fold in FORWARD_MONTHS:
        evaluate = frame[(frame.month == fold) & frame.p_price.notna()].copy()
        train = frame[
            (frame.month < fold)
            & frame.eligible_news.eq(1)
            & (frame.end_utc < evaluate.cutoff_utc.min())
        ].copy()
        eligible = train[train.eligible_news.eq(1)].copy()
        if eligible.empty:
            records.append({
                "fold": fold,
                "train_rows": 0,
                "eval_rows": int(len(evaluate)),
                "preprocessing": "not fit; registered exact-R1 support fallback",
                "independent_transform_max_abs_error": None,
            })
            continue
        median, mean, scale = state_fit(eligible)
        transformed = state_transform(evaluate, median, mean, scale)
        raw = evaluate[STATE_COLUMNS].to_numpy(float)
        independent = (np.where(np.isfinite(raw), raw, median) - mean) / scale
        error = float(np.max(np.abs(transformed - independent))) if len(evaluate) else 0.0
        all_good &= error <= 1e-15
        records.append({
            "fold": fold,
            "train_rows": int(len(eligible)),
            "eval_rows": int(len(evaluate)),
            "training_median": median.tolist(),
            "training_mean_after_imputation": mean.tolist(),
            "training_scale": scale.tolist(),
            "independent_transform_max_abs_error": error,
        })
    checks["fold_imputation_reconstruction_uses_training_median"] = all_good
    return checks, records


def static_checks() -> tuple[dict[str, bool], dict, list[dict]]:
    frame, provenance = load_inputs()
    expert = validate_expert_evidence(frame)
    state = np.arange(8 * len(STATE_COLUMNS), dtype=float).reshape(8, len(STATE_COLUMNS))
    stock = np.array([0, 1] * 4, dtype=float)
    g2 = design_matrix(state, stock, "G2")
    g3 = design_matrix(state, stock, "G3")
    g3_zero = design_matrix(state, stock, "G3", interaction_zero=True)
    checks, records = real_preparation_checks(frame)
    checks.update(fallback_checks())
    checks["g3_zero_nested_design"] = bool(
        np.array_equal(g3[:, : g2.shape[1]], g2)
        and np.all(g3_zero[:, g2.shape[1] :] == 0)
    )
    mapped, weights = map_advantage(
        np.linspace(-2, 2, 8), np.full(8, 0.5), np.full(8, 0.6), np.ones(8)
    )
    checks["map_weights_finite_and_bounded"] = bool(
        np.isfinite(mapped).all()
        and np.isfinite(weights).all()
        and ((weights >= 0) & (weights <= 1)).all()
    )
    prereg = (HERE / "PRE_REGISTRATION.md").read_text()
    checks["no_news_expert_contract_documented"] = "For no-news rows, `w_text=0` and `p_final=p_price` exactly." in prereg
    checks["both_expert_provenances_verified"] = bool(
        expert.get("all_expert_provenance_verified")
        and provenance["experts"]["R1"]["probability_parity_max_abs_error"] <= 1e-15
        and set(expert.get("experts", {})) == {"R1", "F1_new"}
    )
    return checks, {"provenance": provenance, "expert_evidence": expert}, records


def main() -> None:
    checks, evidence, records = static_checks()
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": "preflight_only_no_G0_G3_predictive_execution",
        "checks": checks,
        "expert_provenance": evidence["provenance"],
        "expert_evidence": evidence["expert_evidence"],
        "fold_preprocessing_records": records,
    }
    def clean(value):
        if hasattr(value, "item"):
            return clean(value.item())
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    result = clean(result)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
