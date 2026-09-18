"""Preflight contracts for Phase B. This file never runs G0--G3 replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from common import CONTROLLER_NAMES, INTERACTION_COLUMNS, STATE_COLUMNS, TAU, design_matrix, map_advantage
from prepare import load_inputs, validate_expert_evidence
from run_gate import fit_controller, oracle_table, prediction_rows, state_fit, state_transform

HERE = Path(__file__).resolve().parent
OUT = HERE / "preflight_verification.json"
RESULT_DIR = HERE / "v1"
FORWARD_MONTHS = [f"2018-{month:02d}" for month in range(3, 9)]
OUTER_MONTHS = ["2018-06", "2018-07", "2018-08"]


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


def headroom_semantics_check() -> bool:
    """Regression cases: probability differences alone cannot repair BA."""
    same_side_price, same_side_text = 0.70, 0.60
    crossing_price, crossing_text = 0.70, 0.40
    same_side_probability_diff = abs(same_side_price - same_side_text) > 1e-12
    same_side_direction_diff = (same_side_price >= 0.5) != (same_side_text >= 0.5)
    crossing_direction_diff = (crossing_price >= 0.5) != (crossing_text >= 0.5)
    return bool(
        same_side_probability_diff
        and not same_side_direction_diff
        and crossing_direction_diff
    )


def oracle_no_news_contract_check() -> bool:
    sample = pd.DataFrame({
        "phase": ["synthetic", "synthetic"],
        "symbol": ["AAPL", "AAPL"],
        "month": ["2018-01", "2018-01"],
        "eligible_news": [0, 0],
        "p_price": [0.40, 0.60],
        "p_text": [0.99, 0.01],
        "label": [0, 1],
    })
    full = oracle_table(sample)
    if set(full.oracle_scope) != {"FULL_SYSTEM_R1_ON_NO_NEWS"}:
        return False
    row = full.iloc[0]
    return bool(
        row.eligible_row_count == 0
        and row.direction_disagreement == 0
        and row.R1_wrong_F1_right == 0
        and row.R1_right_F1_wrong == 0
        and row.label == "HINDSIGHT DIAGNOSTIC ORACLE — NOT A MODEL"
    )


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
    checks["headroom_uses_direction_disagreement"] = headroom_semantics_check()
    checks["hindsight_oracle_forces_r1_on_no_news"] = oracle_no_news_contract_check()
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


def preflight_main() -> dict:
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
    return result


def independent_metric(y, p) -> dict:
    """Verifier-local metrics; intentionally not the runner's metric helper."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    direction = p >= 0.5
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, direction)) if len(np.unique(y)) == 2 else None,
        "MCC": float(matthews_corrcoef(y, direction)),
        "Brier": float(brier_score_loss(y, p)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "up_recall": float(direction[y == 1].mean()) if (y == 1).any() else None,
        "down_recall": float((~direction[y == 0]).mean()) if (y == 0).any() else None,
        "pred_up": float(direction.mean()),
        "constant": bool(len(np.unique(direction)) == 1),
    }


def independent_metric_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = []
    monthly = []
    group_specs = [(["phase", "symbol"], fields), (["phase", "symbol", "month"], monthly)]
    for keys, destination in group_specs:
        for values, group in pred.groupby(keys):
            values = (values,) if not isinstance(values, tuple) else values
            context = dict(zip(keys, values))
            for controller in CONTROLLER_NAMES:
                rows = group[group.controller.eq(controller)].copy()
                metric = independent_metric(rows.label, rows.p_final)
                final = rows.p_final.to_numpy(float) >= 0.5
                price = rows.p_price.to_numpy(float) >= 0.5
                metric.update(context)
                metric.update({
                    "method": controller,
                    "text_weight_mean": float(rows.w_text.mean()),
                    "text_weight_std": float(rows.w_text.std(ddof=0)),
                    "text_weight_gt_half": float((rows.w_text > 0.5).mean()),
                    "direction_changed_vs_R1": int((final != price).sum()),
                    "repaired_errors": int(((final == rows.label.to_numpy(int)) & (price != rows.label.to_numpy(int))).sum()),
                    "new_errors": int(((final != rows.label.to_numpy(int)) & (price == rows.label.to_numpy(int))).sum()),
                })
                destination.append(metric)
    return pd.DataFrame(fields), pd.DataFrame(monthly)


def tables_match(expected: pd.DataFrame, observed: pd.DataFrame, keys: list[str], fields: list[str]) -> bool:
    if expected.duplicated(keys).any() or observed.duplicated(keys).any():
        return False
    if set(map(tuple, expected[keys].to_numpy())) != set(map(tuple, observed[keys].to_numpy())):
        return False
    joined = expected.merge(observed, on=keys, suffixes=("_expected", "_observed"), validate="one_to_one")
    for field in fields:
        if field not in observed.columns:
            return False
        left, right = joined[f"{field}_expected"], joined[f"{field}_observed"]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right):
            if not (left.astype(bool) == right.astype(bool)).all():
                return False
        elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            present = left.notna() & right.notna()
            if not (left.isna() == right.isna()).all():
                return False
            if present.any() and not np.allclose(left[present], right[present], atol=1e-12, rtol=0.0):
                return False
        elif not (left.fillna("__NA__").astype(str) == right.fillna("__NA__").astype(str)).all():
            return False
    return True


def independent_advancement(monthly: pd.DataFrame, pred: pd.DataFrame) -> list[dict]:
    records = []
    for new, base in (("G1", "G0"), ("G2", "G1"), ("G3", "G2")):
        subset = monthly[
            monthly.phase.eq("train_forward_oof")
            & monthly.month.isin(OUTER_MONTHS)
            & monthly.method.isin([new, base])
        ]
        candidate = subset[subset.method.eq(new)].set_index(["symbol", "month"])
        reference = subset[subset.method.eq(base)].set_index(["symbol", "month"])
        joined = candidate.join(reference, lsuffix="_new", rsuffix="_base", how="inner")
        joined["delta_BA"] = joined.BA_new - joined.BA_base
        joined["delta_Brier"] = joined.Brier_new - joined.Brier_base
        stock_ba = joined.groupby(level=0).delta_BA.mean()
        month_ba = joined.groupby(level=1).delta_BA.mean()
        stock_brier = joined.groupby(level=0).delta_Brier.mean()
        constants = candidate.groupby(level=0).constant.any()
        eligible = pred[
            pred.phase.eq("train_forward_oof")
            & pred.month.isin(OUTER_MONTHS)
            & pred.controller.eq("G0")
            & pred.eligible_news.eq(1)
        ]
        probability_difference = {
            symbol: int(((eligible.loc[eligible.symbol.eq(symbol), "p_price"] - eligible.loc[eligible.symbol.eq(symbol), "p_text"]).abs() > 1e-12).sum())
            for symbol in ("AAPL", "AMZN")
        }
        direction_disagreement = {
            symbol: int(((eligible.loc[eligible.symbol.eq(symbol), "p_price"] >= 0.5) != (eligible.loc[eligible.symbol.eq(symbol), "p_text"] >= 0.5)).sum())
            for symbol in ("AAPL", "AMZN")
        }
        headroom = min(direction_disagreement.values()) >= 10
        records.append({
            "contrast": f"{new}_vs_{base}",
            "AAPL_mean_delta_BA": float(stock_ba.get("AAPL", np.nan)),
            "AMZN_mean_delta_BA": float(stock_ba.get("AMZN", np.nan)),
            "weaker_stock_mean_delta_BA": float(stock_ba.min()) if len(stock_ba) else None,
            "macro_delta_BA": float(joined.delta_BA.mean()) if len(joined) else None,
            "positive_outer_months": int((month_ba > 0).sum()),
            "max_stock_mean_delta_Brier": float(stock_brier.max()) if len(stock_brier) else None,
            "constant_any_stock": bool(constants.any()) if len(constants) else True,
            "expert_probability_difference_rows": probability_difference,
            "expert_direction_disagreement_rows": direction_disagreement,
            "routing_headroom_sufficient": bool(headroom),
            "outer_cells": int(len(joined)),
            "passes": bool(
                len(joined) == 6
                and stock_ba.min() >= 0.01
                and stock_ba.max() >= -0.01
                and (month_ba > 0).sum() >= 2
                and stock_brier.max() <= 0.002
                and not constants.any()
                and headroom
            ),
        })
    return records


def record_match(expected, observed) -> bool:
    if not isinstance(observed, dict) or set(expected) != set(observed):
        return False
    for key, value in expected.items():
        other = observed[key]
        if isinstance(value, dict):
            if not record_match(value, other):
                return False
        elif isinstance(value, bool):
            if bool(other) != value:
                return False
        elif value is None:
            if other is not None and not pd.isna(other):
                return False
        elif isinstance(value, (float, int)):
            if other is None or not np.isclose(float(value), float(other), atol=1e-12, rtol=0.0, equal_nan=True):
                return False
        elif value != other:
            return False
    return True


def verify_saved_run(result_dir: Path) -> tuple[dict[str, bool], dict]:
    required = [
        "predictions.csv", "metrics.csv", "monthly_metrics.csv", "advancement.json",
        "controller_training_evidence.json", "controller_coefficients.csv",
    ]
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        return {"required_result_artifacts_present": False}, {"missing": missing}
    pred = pd.read_csv(result_dir / "predictions.csv")
    metrics = pd.read_csv(result_dir / "metrics.csv")
    monthly = pd.read_csv(result_dir / "monthly_metrics.csv")
    evidence = json.loads((result_dir / "controller_training_evidence.json").read_text())
    saved_advancement = json.loads((result_dir / "advancement.json").read_text())
    coefficients = pd.read_csv(result_dir / "controller_coefficients.csv")
    frozen, _ = load_inputs()
    checks: dict[str, bool] = {"required_result_artifacts_present": True}

    source = frozen[(frozen.month >= "2018-03") & frozen.p_price.notna()].copy()
    expected_keys = {fold: set(source.loc[source.month.eq(fold), "key"]) for fold in FORWARD_MONTHS}
    expected_keys["august_freeze"] = set(source.loc[source.month.ge("2018-09"), "key"])
    checks["controllers_exactly_g0_to_g3"] = set(pred.controller.unique()) == set(CONTROLLER_NAMES)
    checks["no_duplicate_prediction_key_controller"] = not pred.duplicated(["key", "controller"]).any()
    checks["folds_and_exposed_freeze_exact"] = (
        set(pred.fold.unique()) == set(FORWARD_MONTHS + ["august_freeze"])
        and all(
            set(pred.loc[pred.fold.eq(fold), "key"]) == keys
            and set(pred.loc[pred.fold.eq(fold), "phase"]) == {"train_forward_oof"}
            for fold, keys in expected_keys.items() if fold != "august_freeze"
        )
        and set(pred.loc[pred.fold.eq("august_freeze"), "key"]) == expected_keys["august_freeze"]
        and set(pred.loc[pred.fold.eq("august_freeze"), "phase"]) <= {"development", "later"}
    )
    checks["all_expected_controller_key_rows_present"] = all(
        set(pred.loc[pred.fold.eq(fold) & pred.controller.eq(controller), "key"]) == keys
        for fold, keys in expected_keys.items() for controller in CONTROLLER_NAMES
    )
    source_columns = ["key", "label", "has_news", "eligible_news", "p_price", "p_text"]
    saved_source = pred[source_columns].drop_duplicates("key")
    paired = source[source_columns].merge(saved_source, on="key", suffixes=("_expected", "_saved"), validate="one_to_one")
    checks["prediction_rows_match_frozen_experts"] = bool(
        len(paired) == len(source)
        and np.array_equal(paired.label_expected.to_numpy(int), paired.label_saved.to_numpy(int))
        and np.array_equal(paired.has_news_expected.to_numpy(int), paired.has_news_saved.to_numpy(int))
        and np.array_equal(paired.eligible_news_expected.to_numpy(int), paired.eligible_news_saved.to_numpy(int))
        and np.allclose(paired.p_price_expected, paired.p_price_saved, atol=1e-12, rtol=0.0)
        and np.allclose(paired.p_text_expected, paired.p_text_saved, atol=1e-12, rtol=0.0)
    )
    no_news = pred.has_news.eq(0)
    exact_r1 = pred.force_price.eq(1) | pred.fallback.astype(str).str.startswith("R1_SUPPORT")
    checks["no_news_exact_r1_probability_and_weight"] = bool(
        np.array_equal(pred.loc[no_news, "p_final"].to_numpy(float), pred.loc[no_news, "p_price"].to_numpy(float))
        and np.array_equal(pred.loc[no_news, "w_text"].to_numpy(float), np.zeros(int(no_news.sum())))
    )
    checks["support_fallback_exact_r1_probability_and_weight"] = bool(
        np.array_equal(pred.loc[exact_r1, "p_final"].to_numpy(float), pred.loc[exact_r1, "p_price"].to_numpy(float))
        and np.array_equal(pred.loc[exact_r1, "w_text"].to_numpy(float), np.zeros(int(exact_r1.sum())))
    )
    ordinary = ~(no_news | exact_r1)
    d_hat = pred.loc[ordinary, "d_hat"].to_numpy(float)
    expected_weight = 1.0 / (1.0 + np.exp(np.clip(d_hat / TAU, -60.0, 60.0)))
    expected_probability = (1.0 - expected_weight) * pred.loc[ordinary, "p_price"].to_numpy(float) + expected_weight * pred.loc[ordinary, "p_text"].to_numpy(float)
    checks["ordinary_mapping_independently_reconstructed"] = bool(
        np.allclose(pred.loc[ordinary, "w_text"].to_numpy(float), expected_weight, atol=1e-12, rtol=0.0)
        and np.allclose(pred.loc[ordinary, "p_final"].to_numpy(float), expected_probability, atol=1e-12, rtol=0.0)
    )
    expected_metrics, expected_monthly = independent_metric_tables(pred)
    metric_fields = [
        "n", "BA", "MCC", "Brier", "AUC", "up_recall", "down_recall", "pred_up", "constant",
        "text_weight_mean", "text_weight_std", "text_weight_gt_half", "direction_changed_vs_R1",
        "repaired_errors", "new_errors",
    ]
    checks["metrics_independently_reconstructed"] = tables_match(expected_metrics, metrics, ["phase", "symbol", "method"], metric_fields)
    checks["monthly_metrics_independently_reconstructed"] = tables_match(expected_monthly, monthly, ["phase", "symbol", "month", "method"], metric_fields)
    expected_advancement = independent_advancement(expected_monthly, pred)
    saved_by_contrast = {row.get("contrast"): row for row in saved_advancement}
    checks["advancement_has_exact_six_outer_cells_per_contrast"] = all(int(row["outer_cells"]) == 6 for row in expected_advancement)
    checks["advancement_independently_reconstructed"] = bool(
        len(saved_by_contrast) == 3
        and all(record_match(row, saved_by_contrast.get(row["contrast"], {})) for row in expected_advancement)
    )
    fits = evidence.get("fits", [])
    expected_fit_keys = {(fold, method) for fold in FORWARD_MONTHS + ["august_freeze"] for method in CONTROLLER_NAMES}
    fit_map = {(fit.get("fold"), fit.get("method")): fit for fit in fits}
    forward_safe = True
    freeze_safe = True
    preproc_safe = True
    for (fold, _method), fit in fit_map.items():
        end, cutoff = fit.get("train_target_end_max"), fit.get("eval_cutoff_min")
        if fold in FORWARD_MONTHS and end is not None:
            forward_safe &= pd.Timestamp(end) < pd.Timestamp(cutoff)
        if fold == "august_freeze":
            months = fit.get("training_months", [])
            freeze_safe &= bool(fit.get("frozen_no_exposed_label_update"))
            freeze_safe &= bool(months) and all(month <= "2018-08" for month in months)
            freeze_safe &= end is not None and pd.Timestamp(end) < pd.Timestamp("2018-09-01", tz="UTC")
        if fit.get("variant") in {"G2", "G3"} and not fit.get("force_price"):
            state = fit.get("state_preprocessing", {})
            names = {"training_median", "training_mean_after_imputation", "training_scale"}
            preproc_safe &= set(state) == names and all(
                isinstance(state.get(name), list)
                and len(state[name]) == len(STATE_COLUMNS)
                and np.isfinite(np.asarray(state[name], float)).all()
                for name in names
            )
    checks["controller_fit_structure_exact"] = set(fit_map) == expected_fit_keys and len(fit_map) == len(fits)
    checks["controller_forward_chronology_safe"] = bool(forward_safe)
    checks["august_freeze_has_no_exposed_label_update"] = bool(freeze_safe)
    checks["controller_preprocessing_is_training_only"] = bool(preproc_safe)
    expected_g2 = ["intercept", "AMZN_intercept", *STATE_COLUMNS]
    expected_g3 = [*expected_g2, *INTERACTION_COLUMNS]
    coefficient_safe = True
    for (fold, method), fit in fit_map.items():
        present = coefficients.loc[coefficients.fold.eq(fold) & coefficients.method.eq(method), "feature"].tolist()
        if fit.get("force_price"):
            coefficient_safe &= present == []
        elif fit.get("variant") == "G2":
            coefficient_safe &= present == expected_g2
        elif fit.get("variant") == "G3":
            coefficient_safe &= present == expected_g3
        else:
            coefficient_safe &= all(name in {"global_advantage", "AAPL_static_advantage", "AMZN_static_advantage"} for name in present)
    checks["coefficient_structure_and_no_extra_features"] = bool(coefficient_safe)
    return checks, {
        "prediction_rows": int(len(pred)),
        "controller_fit_records": int(len(fits)),
        "advancement_recomputed": expected_advancement,
    }


def main() -> None:
    preflight = preflight_main()
    if not (RESULT_DIR / "predictions.csv").is_file():
        print(json.dumps(preflight, indent=2))
        if preflight["status"] != "PASS":
            raise SystemExit(1)
        return
    checks, evidence = verify_saved_run(RESULT_DIR)
    result = clean({
        "status": "PASS" if preflight["status"] == "PASS" and all(checks.values()) else "FAIL",
        "mode": "postrun_independent_verification",
        "preflight_status": preflight["status"],
        "checks": checks,
        "evidence": evidence,
    })
    (RESULT_DIR / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
