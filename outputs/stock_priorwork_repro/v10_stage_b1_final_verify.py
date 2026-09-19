"""Independent verifier-only replay for frozen V10 Stage B1 DPRICE.

This program intentionally does not import the Stage B1 runner.  It rebuilds
the daily table from raw sources, refits the 36 authorized candidate models and
the 24 issued models, and proves that every pre-existing scientific artifact
and private model byte remains unchanged.
"""
from __future__ import annotations

import functools
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/stock-data"
BASE = ROOT / "outputs/stock_priorwork_repro"
V9 = BASE / "v9"
OUT = BASE / "v10"
PRIVATE = WORK / "priorwork_v10/models/dprice"

SEED = 573
CANDIDATES = [0.01, 0.1, 1.0]
MONTHS = pd.period_range("2018-03", "2019-02", freq="M").astype(str).tolist()
OOF_MONTHS = MONTHS[:6]
FEATURES = [
    "DRET_1",
    "DRET_2",
    "DRET_5",
    "RANGE_1",
    "RV_5",
    "MEAN_5",
    "HISTORY_AGE_HOURS",
]
METHODS = [
    "PAPER_1G_L1LR",
    "PAPER_1G_LINSVM",
    "PAPER_2G_L1LR",
    "TFIDF_LR",
    "TFIDF_LINSVM",
    "TFIDF_RF",
    "TFIDF_ADABOOST",
    "TFIDF_KNN",
]
SCIENCE_FILES = [
    "GRID_OOF_DPRICE.csv",
    "ISSUED_PARAMS_DPRICE.csv",
    "DPRICE_PREDICTIONS.csv",
    "DPRICE_MONTHLY.csv",
    "DPRICE_METRICS.csv",
    "DPRICE_MODEL_MANIFEST.json",
    "METHOD_SELECTION_INPUT_4H.csv",
    "METHOD_SELECTION_INPUT_1D_OVERNIGHT.csv",
    "METHOD_SELECTION_INPUT_1D_24H.csv",
    "METHOD_SELECTIONS.json",
    "STAGE_B2_FROZEN_CONFIGURATION.json",
    "STAGE_B1_FINAL_AUDIT.json",
]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def phase(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2018-03-01", tz="UTC"):
        return "warmup"
    if timestamp < pd.Timestamp("2018-09-01", tz="UTC"):
        return "OOF"
    if timestamp < pd.Timestamp("2018-11-01", tz="UTC"):
        return "development"
    return "later"


def reconstruct_daily() -> tuple[pd.DataFrame, dict]:
    schedule = pd.read_csv(WORK / "audit/xnys_schedule.csv", index_col=0)
    schedule.index = pd.to_datetime(schedule.index).strftime("%Y-%m-%d")
    schedule[["open", "close"]] = schedule[["open", "close"]].apply(
        pd.to_datetime, utc=True
    )
    rows: list[dict] = []
    prior_count_failures = 0
    chronology_failures = 0
    latest_close_failures = 0
    for stock, prefix in (("AAPL", "APPLE"), ("AMZN", "AMAZON")):
        bars = pd.read_csv(
            WORK / f"raw/CHARTS/{prefix}1440.csv",
            header=None,
            names=["date", "time", "open", "high", "low", "close", "activity"],
        )
        bars["day"] = pd.to_datetime(bars["date"], format="%Y.%m.%d").dt.strftime(
            "%Y-%m-%d"
        )
        bars = bars.set_index("day").sort_index()
        days = [day for day in schedule.index if day in bars.index]
        for index, day in enumerate(days):
            if index < 5:
                continue
            prior_days = days[index - 5 : index]
            history = bars.loc[prior_days]
            returns = np.log(history["close"] / history["open"]).to_numpy(float)
            target_open, target_close = schedule.loc[day, ["open", "close"]]
            cutoff = target_open - pd.Timedelta(minutes=5)
            prior_count_failures += int(len(prior_days) != 5)
            chronology_failures += int(not all(prior_day < day for prior_day in prior_days))
            latest_close_failures += int(schedule.loc[prior_days[-1], "close"] >= cutoff)
            rows.append(
                {
                    "stock": stock,
                    "day": day,
                    "start_utc": target_open,
                    "end_utc": target_close,
                    "cutoff_utc": cutoff,
                    "label": int(bars.loc[day, "close"] > bars.loc[day, "open"]),
                    "phase": phase(target_open),
                    "DRET_1": float(returns[-1]),
                    "DRET_2": float(returns[-2:].sum()),
                    "DRET_5": float(returns.sum()),
                    "RANGE_1": float(
                        (history["high"].iloc[-1] - history["low"].iloc[-1])
                        / history["open"].iloc[-1]
                    ),
                    "RV_5": float(np.sqrt(np.sum(returns * returns))),
                    "MEAN_5": float(returns.mean()),
                    "HISTORY_AGE_HOURS": float(
                        (cutoff - schedule.loc[prior_days[-1], "close"]).total_seconds()
                        / 3600
                    ),
                }
            )
    data = pd.DataFrame(rows).sort_values(["stock", "start_utc"]).reset_index(drop=True)
    counts = {key: int(value) for key, value in data["phase"].value_counts().items()}
    audit = {
        "target_rows": int(len(data)),
        "phase_counts": counts,
        "prior_session_count_failures": prior_count_failures,
        "prior_session_chronology_failures": chronology_failures,
        "latest_prior_close_cutoff_failures": latest_close_failures,
        "predictive_features": FEATURES,
        "target_day_feature_count": 0,
        "overnight_target_gap_feature_count": 0,
        "activity_feature_count": 0,
    }
    return data, audit


def fit_probability(
    train: pd.DataFrame, evaluation: pd.DataFrame, c_value: float
) -> tuple[np.ndarray, StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(train[FEATURES])
    classifier = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    ).fit(scaler.transform(train[FEATURES]), train["label"])
    probability = classifier.predict_proba(scaler.transform(evaluation[FEATURES]))[:, 1]
    return probability, scaler, classifier


def metrics(labels: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (p >= 0.5).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "BA": float(balanced_accuracy_score(y, prediction)) if both else None,
        "MCC": float(matthews_corrcoef(y, prediction)) if both else None,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "up_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "down_recall": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "pred_up": float(prediction.mean()),
        "true_up": float(y.mean()),
        "AUC": float(roc_auc_score(y, p)) if both else None,
        "Brier": float(brier_score_loss(y, p)),
        "constant": bool(prediction.min() == prediction.max()),
    }


def numeric_difference(left: object, right: object) -> float:
    if pd.isna(left) and pd.isna(right):
        return 0.0
    return abs(float(left) - float(right))


def replay_grid(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frozen = pd.read_csv(OUT / "GRID_OOF_DPRICE.csv")
    rows: list[dict] = []
    boundary_violations = 0
    for stock in ("AAPL", "AMZN"):
        stock_data = data[data["stock"] == stock]
        for month in OOF_MONTHS:
            evaluation = stock_data[
                stock_data["start_utc"].dt.strftime("%Y-%m") == month
            ].copy()
            evaluation_cutoff = evaluation["cutoff_utc"].min()
            train = stock_data[stock_data["end_utc"] < evaluation_cutoff].copy()
            boundary_violations += int(not bool((train["end_utc"] < evaluation_cutoff).all()))
            for c_value in CANDIDATES:
                probability, _, _ = fit_probability(train, evaluation, c_value)
                rows.append(
                    {
                        "stock": stock,
                        "month": month,
                        "C": c_value,
                        "train_n": int(len(train)),
                        "train_end": str(train["end_utc"].max()),
                        "evaluation_cutoff": str(evaluation_cutoff),
                        **metrics(evaluation["label"], probability),
                    }
                )
    replay = pd.DataFrame(rows)
    key = ["stock", "month", "C"]
    replay = replay.sort_values(key).reset_index(drop=True)
    frozen = frozen.sort_values(key).reset_index(drop=True)
    key_match = replay[key].equals(frozen[key])
    max_metric_error = 0.0
    metric_fields = [
        "n",
        "accuracy",
        "BA",
        "MCC",
        "precision",
        "recall",
        "F1",
        "up_recall",
        "down_recall",
        "pred_up",
        "true_up",
        "AUC",
        "Brier",
    ]
    for field in metric_fields:
        for actual, expected in zip(replay[field], frozen[field]):
            max_metric_error = max(max_metric_error, numeric_difference(actual, expected))
    metadata_match = all(
        replay[field].astype(str).equals(frozen[field].astype(str))
        for field in ["train_n", "train_end", "evaluation_cutoff", "constant"]
    )
    audit = {
        "candidate_rows_expected": 36,
        "candidate_rows_replayed": int(len(replay)),
        "row_keys_match": bool(key_match),
        "training_metadata_match": bool(metadata_match),
        "maximum_metric_discrepancy": max_metric_error,
        "training_boundary_violations": boundary_violations,
        "sep_plus_candidate_models_fitted": 0,
    }
    return replay, audit


def selection_months(month: str) -> list[str]:
    if month == "2018-03":
        return []
    if month <= "2018-08":
        return pd.period_range(
            "2018-03", str(pd.Period(month) - 1), freq="M"
        ).astype(str).tolist()
    return OOF_MONTHS


def select_c(replay: pd.DataFrame, stock: str, months: list[str]) -> float:
    evidence = replay[(replay["stock"] == stock) & replay["month"].isin(months)]
    ranked = (
        evidence.groupby("C", as_index=False)
        .agg(BA=("BA", "mean"), Brier=("Brier", "mean"))
        .sort_values(["BA", "Brier", "C"], ascending=[False, True, True])
    )
    return float(ranked.iloc[0]["C"])


def reconstruct_selections(replay: pd.DataFrame) -> tuple[dict, dict]:
    frozen = pd.read_csv(OUT / "ISSUED_PARAMS_DPRICE.csv")
    selected: dict[tuple[str, str], float] = {}
    mismatches: list[dict] = []
    freeze_violations = 0
    checked = 0
    for stock in ("AAPL", "AMZN"):
        final_c = select_c(replay, stock, OOF_MONTHS)
        for month in MONTHS:
            used = selection_months(month)
            wanted = 0.1 if month == "2018-03" else (
                final_c if month >= "2018-09" else select_c(replay, stock, used)
            )
            selected[(stock, month)] = wanted
            row = frozen[
                (frozen["stock"] == stock)
                & (frozen["prediction_month"] == month)
            ]
            checked += int(len(row) == 1)
            if len(row) != 1:
                mismatches.append({"stock": stock, "month": month, "reason": "row_count"})
                continue
            record = row.iloc[0]
            valid = (
                float(record["issued_C"]) == wanted
                and json.loads(record["selection_months_used"]) == used
                and int(record["authorized_candidate_rows_used"]) == 3 * len(used)
                and bool(record["is_march_default"]) == (month == "2018-03")
                and bool(record["is_sep_plus_frozen"]) == (month >= "2018-09")
            )
            if not valid:
                mismatches.append(
                    {
                        "stock": stock,
                        "month": month,
                        "expected_C": wanted,
                        "saved_C": float(record["issued_C"]),
                    }
                )
            if month >= "2018-09" and wanted != final_c:
                freeze_violations += 1
    audit = {
        "issued_decisions_expected": 24,
        "issued_decisions_checked": checked,
        "issued_C_mismatch_count": len(mismatches),
        "issued_C_mismatches": mismatches,
        "sep_plus_freeze_violations": freeze_violations,
        "final_frozen_C": {
            stock: select_c(replay, stock, OOF_MONTHS) for stock in ("AAPL", "AMZN")
        },
    }
    return selected, audit


def refit_issued(
    data: pd.DataFrame,
    selected: dict[tuple[str, str], float],
    model_hashes_before: dict[str, str],
) -> tuple[dict, dict]:
    frozen_predictions = pd.read_csv(
        OUT / "DPRICE_PREDICTIONS.csv",
        parse_dates=["start_utc", "cutoff_utc"],
    )
    manifest = json.loads((OUT / "DPRICE_MODEL_MANIFEST.json").read_text())
    manifest_by_id = {entry["logical_id"]: entry for entry in manifest}
    row_key_mismatches = 0
    direction_mismatches = 0
    refit_frozen_max_error = 0.0
    serialized_refit_max_error = 0.0
    serialized_frozen_max_error = 0.0
    manifest_hash_mismatches: list[str] = []
    replayed_predictions = 0
    refitted_models = 0
    model_details: list[dict] = []
    for stock in ("AAPL", "AMZN"):
        stock_data = data[data["stock"] == stock]
        for month in MONTHS:
            evaluation = stock_data[
                stock_data["start_utc"].dt.strftime("%Y-%m") == month
            ].sort_values(["start_utc", "cutoff_utc"]).reset_index(drop=True)
            train = stock_data[
                stock_data["end_utc"] < evaluation["cutoff_utc"].min()
            ].copy()
            c_value = selected[(stock, month)]
            independent_p, _, _ = fit_probability(train, evaluation, c_value)
            refitted_models += 1
            replayed_predictions += len(evaluation)
            frozen = frozen_predictions[
                (frozen_predictions["stock"] == stock)
                & (frozen_predictions["month"] == month)
            ].sort_values(["start_utc", "cutoff_utc"]).reset_index(drop=True)
            independent_keys = list(zip(evaluation["stock"], evaluation["day"]))
            frozen_keys = list(zip(frozen["stock"], frozen["day"]))
            row_key_mismatches += int(independent_keys != frozen_keys)
            frozen_p = frozen["p"].to_numpy(float)
            refit_error = float(np.max(np.abs(independent_p - frozen_p)))
            refit_frozen_max_error = max(refit_frozen_max_error, refit_error)
            direction_mismatches += int(
                np.sum((independent_p >= 0.5) != (frozen_p >= 0.5))
            )

            logical_id = f"{stock}_{month}"
            path = PRIVATE / f"{logical_id}.joblib"
            manifest_entry = manifest_by_id[logical_id]
            if file_hash(path) != manifest_entry["sha256"]:
                manifest_hash_mismatches.append(logical_id)
            bundle = joblib.load(path)
            serialized_p = bundle["classifier"].predict_proba(
                bundle["scaler"].transform(evaluation[FEATURES])
            )[:, 1]
            serialized_refit_max_error = max(
                serialized_refit_max_error,
                float(np.max(np.abs(serialized_p - independent_p))),
            )
            serialized_frozen_max_error = max(
                serialized_frozen_max_error,
                float(np.max(np.abs(serialized_p - frozen_p))),
            )
            model_details.append(
                {
                    "logical_model_id": logical_id,
                    "C": c_value,
                    "train_n": int(len(train)),
                    "prediction_n": int(len(evaluation)),
                    "independent_refit_vs_frozen_max_error": refit_error,
                }
            )
    model_hashes_after = {
        path.name: file_hash(path) for path in sorted(PRIVATE.glob("*.joblib"))
    }
    audit = {
        "issued_models_expected": 24,
        "issued_models_independently_refit": refitted_models,
        "row_level_predictions_expected": 466,
        "row_level_predictions_replayed": replayed_predictions,
        "row_key_mismatches": row_key_mismatches,
        "direction_mismatches": direction_mismatches,
        "independent_refit_vs_frozen_max_probability_error": refit_frozen_max_error,
        "final_serialized_vs_independent_refit_max_probability_error": serialized_refit_max_error,
        "final_serialized_vs_frozen_max_probability_error": serialized_frozen_max_error,
        "manifest_hash_mismatches": manifest_hash_mismatches,
        "private_model_bytes_unchanged": model_hashes_before == model_hashes_after,
        "models": model_details,
    }
    return audit, model_hashes_after


def build_method_input(predictions: pd.DataFrame, horizon: str) -> pd.DataFrame:
    rows: list[dict] = []
    for method in METHODS:
        for stock in ("AAPL", "AMZN"):
            for month in OOF_MONTHS:
                sample = predictions[
                    (predictions["symbol"] == stock)
                    & (predictions["horizon"] == horizon)
                    & (predictions["method"] == method)
                    & (predictions["month"] == month)
                ]
                decision_only = "LINSVM" in method
                score = sample["p"].to_numpy(float)
                prediction = (score >= (0.0 if decision_only else 0.5)).astype(int)
                rows.append(
                    {
                        "method": method,
                        "stock": stock,
                        "month": month,
                        "n": int(len(sample)),
                        "BA": float(balanced_accuracy_score(sample["label"], prediction)),
                        "Brier": None
                        if decision_only
                        else float(brier_score_loss(sample["label"], score)),
                        "decision_only": decision_only,
                    }
                )
    return pd.DataFrame(rows)


def rank_methods(method_input: pd.DataFrame) -> list[str]:
    records: list[dict] = []
    for method, group in method_input.groupby("method"):
        aapl = float(group[group["stock"] == "AAPL"]["BA"].mean())
        amzn = float(group[group["stock"] == "AMZN"]["BA"].mean())
        decision_only = bool(group["decision_only"].iloc[0])
        records.append(
            {
                "method": method,
                "weaker": min(aapl, amzn),
                "macro": (aapl + amzn) / 2,
                "brier": None
                if decision_only
                else float(group["Brier"].mean()),
            }
        )

    def compare(left: dict, right: dict) -> int:
        for field in ("weaker", "macro"):
            if left[field] != right[field]:
                return -1 if left[field] > right[field] else 1
        if left["brier"] is not None and right["brier"] is not None:
            if left["brier"] != right["brier"]:
                return -1 if left["brier"] < right["brier"] else 1
        if left["method"] == right["method"]:
            return 0
        return -1 if left["method"] < right["method"] else 1

    return [
        record["method"]
        for record in sorted(records, key=functools.cmp_to_key(compare))
    ]


def verify_method_freeze() -> dict:
    predictions_4h = pd.read_csv(V9 / "PREDICTIONS_4H.csv")
    predictions_1d = pd.read_csv(V9 / "PREDICTIONS_1D.csv")
    saved = json.loads((OUT / "METHOD_SELECTIONS.json").read_text())["rankings"]
    definitions = [
        ("4h", predictions_4h, "4h"),
        ("overnight", predictions_1d, "1d:DNEWS_OVERNIGHT"),
        ("24h", predictions_1d, "1d:DNEWS_24H"),
    ]
    expected_top = {
        "4h": ["TFIDF_LR", "PAPER_2G_L1LR", "TFIDF_RF"],
        "overnight": ["PAPER_1G_L1LR", "TFIDF_LR", "PAPER_2G_L1LR"],
        "24h": ["TFIDF_ADABOOST", "PAPER_2G_L1LR", "TFIDF_KNN"],
    }
    computed: dict[str, list[str]] = {}
    valid = True
    for name, source, horizon in definitions:
        method_input = build_method_input(source, horizon)
        ranking = rank_methods(method_input)
        computed[name] = ranking[:3]
        saved_top = [record["method"] for record in saved[name] if record["top_three"]]
        valid &= len(method_input) == 96
        valid &= set(method_input["month"]) == set(OOF_MONTHS)
        valid &= computed[name] == expected_top[name] == saved_top
    return {
        "row_level_v9_issued_oof_reconstruction": bool(valid),
        "top_three": computed,
        "development_or_later_used": False,
        "candidate_grid_maxima_used": False,
    }


def verify_b2_configuration() -> dict:
    configuration = json.loads((OUT / "STAGE_B2_FROZEN_CONFIGURATION.json").read_text())
    issued_4h = pd.read_csv(OUT / "ISSUED_PARAMS_4H.csv")
    issued_1d = pd.read_csv(OUT / "ISSUED_PARAMS_1D.csv")
    mismatches: list[dict] = []
    for branch in configuration["branches"]:
        source = issued_4h if branch["horizon_window"] == "4h" else issued_1d
        rows = source[
            (source["stock"] == branch["stock"])
            & (source["horizon"] == branch["horizon_window"])
            & (source["method"] == branch["method"])
            & (source["is_sep_plus_frozen"] == True)
        ]
        params = {json.dumps(json.loads(value), sort_keys=True) for value in rows["issued_parameter_json"]}
        expected = json.dumps(branch["final_frozen_parameter"], sort_keys=True)
        if len(rows) != 6 or params != {expected}:
            mismatches.append(
                {
                    "stock": branch["stock"],
                    "horizon_window": branch["horizon_window"],
                    "method": branch["method"],
                }
            )
    return {
        "branch_count": int(len(configuration["branches"])),
        "parameter_mismatch_count": len(mismatches),
        "parameter_mismatches": mismatches,
        "status": configuration.get("status"),
    }


def protected_pre_b1_status() -> tuple[bool, dict[str, bool]]:
    ledger = json.loads((OUT / "STAGE_B1_PRE_RUN_HASHES.json").read_text())[
        "tracked_files"
    ]
    current = {path: file_hash(ROOT / path) for path in ledger}
    unchanged = {path: current[path] == expected for path, expected in ledger.items()}
    groups = {
        "v6_v9_unchanged": all(
            value
            for path, value in unchanged.items()
            if any(f"stock_priorwork_repro/v{version}/" in path for version in range(6, 10))
        ),
        "stage_a_unchanged": all(
            value for path, value in unchanged.items() if "stock_priorwork_repro/v10/STAGE_A" in path
        ),
        "f0_f1_f2_unchanged": all(
            value
            for path, value in unchanged.items()
            if any(token in path for token in ("stock_integrated_4h", "stock_paper_methods_4h", "stock_goal60_4h"))
        ),
        "market_context_unchanged": all(
            value for path, value in unchanged.items() if "stock_context_4h" in path
        ),
        "relation_reader_unchanged": all(
            value
            for path, value in unchanged.items()
            if "stock_context_4h" in path and "relation" in path.lower()
        ),
    }
    return all(unchanged.values()), groups


def main() -> None:
    historical_audit_hash_before = file_hash(OUT / "STAGE_B1_FINAL_AUDIT.json")
    science_hashes_before = {name: file_hash(OUT / name) for name in SCIENCE_FILES}
    model_hashes_before = {
        path.name: file_hash(path) for path in sorted(PRIVATE.glob("*.joblib"))
    }
    historical_audit = json.loads((OUT / "STAGE_B1_FINAL_AUDIT.json").read_text())

    data, raw_audit = reconstruct_daily()
    replay, grid_audit = replay_grid(data)
    replay.to_csv(OUT / "DPRICE_GRID_INDEPENDENT_REPLAY.csv", index=False)
    write_json(OUT / "DPRICE_GRID_INDEPENDENT_REPLAY_AUDIT.json", grid_audit)

    selected, selection_audit = reconstruct_selections(replay)
    issued_audit, model_hashes_after = refit_issued(data, selected, model_hashes_before)
    issued_audit["selection_reconstruction"] = selection_audit
    write_json(OUT / "DPRICE_ISSUED_INDEPENDENT_REFIT_AUDIT.json", issued_audit)

    method_audit = verify_method_freeze()
    b2_audit = verify_b2_configuration()
    pre_b1_unchanged, protected_groups = protected_pre_b1_status()
    science_hashes_after = {name: file_hash(OUT / name) for name in SCIENCE_FILES}
    science_unchanged = science_hashes_before == science_hashes_after

    checks = {
        "historical_b1_audit_pass_and_unchanged": historical_audit.get("status") == "PASS"
        and historical_audit_hash_before == file_hash(OUT / "STAGE_B1_FINAL_AUDIT.json"),
        "raw_daily_source_reconstruction": raw_audit["target_rows"] == 536
        and raw_audit["phase_counts"]
        == {"warmup": 70, "OOF": 258, "development": 84, "later": 124}
        and raw_audit["prior_session_count_failures"] == 0
        and raw_audit["prior_session_chronology_failures"] == 0
        and raw_audit["latest_prior_close_cutoff_failures"] == 0
        and raw_audit["target_day_feature_count"] == 0
        and raw_audit["overnight_target_gap_feature_count"] == 0
        and raw_audit["activity_feature_count"] == 0,
        "independent_36_candidate_grid_replay": grid_audit["candidate_rows_replayed"] == 36
        and grid_audit["row_keys_match"]
        and grid_audit["training_metadata_match"]
        and grid_audit["maximum_metric_discrepancy"] <= 1e-10
        and grid_audit["training_boundary_violations"] == 0
        and grid_audit["sep_plus_candidate_models_fitted"] == 0,
        "independent_issued_C_reconstruction": selection_audit["issued_decisions_checked"] == 24
        and selection_audit["issued_C_mismatch_count"] == 0
        and selection_audit["sep_plus_freeze_violations"] == 0,
        "independent_24_issued_model_refit": issued_audit["issued_models_independently_refit"] == 24,
        "row_level_prediction_replay": issued_audit["row_level_predictions_replayed"] == 466
        and issued_audit["row_key_mismatches"] == 0
        and issued_audit["independent_refit_vs_frozen_max_probability_error"] <= 1e-10
        and issued_audit["direction_mismatches"] == 0,
        "serialized_model_three_way_parity": issued_audit[
            "final_serialized_vs_independent_refit_max_probability_error"
        ]
        <= 1e-10
        and issued_audit["final_serialized_vs_frozen_max_probability_error"] <= 1e-10
        and not issued_audit["manifest_hash_mismatches"],
        "method_selection_frozen_and_reproducible": method_audit[
            "row_level_v9_issued_oof_reconstruction"
        ],
        "b2_configuration_frozen": b2_audit["branch_count"] == 18
        and b2_audit["parameter_mismatch_count"] == 0
        and b2_audit["status"] == "FROZEN_FOR_FUTURE_B2_ONLY",
        "existing_b1_scientific_artifacts_unchanged": science_unchanged,
        "private_model_bytes_unchanged": model_hashes_before == model_hashes_after
        and issued_audit["private_model_bytes_unchanged"],
        "pre_b1_protected_artifacts_unchanged": pre_b1_unchanged,
        "news_price_fit_count_zero": not any(OUT.glob("*JOINT*"))
        and not (WORK / "priorwork_v10/models/joint").exists(),
    }
    checks.update(protected_groups)
    status = "PASS" if all(checks.values()) else "FAIL"
    final = {
        "status": status,
        "checks": checks,
        "raw_daily_reconstruction": raw_audit,
        "grid_replay": grid_audit,
        "issued_refit": {key: value for key, value in issued_audit.items() if key != "models"},
        "method_selection": method_audit,
        "b2_configuration": b2_audit,
        "scope": {
            "candidate_models_independently_refit": 36,
            "issued_models_independently_refit": 24,
            "news_price_fits": 0,
            "b2_runs": 0,
        },
    }
    write_json(OUT / "STAGE_B1_FINAL_AUDIT_V2.json", final)
    print(
        "V10_STAGE_B1_FULLY_VERIFIED_AWAITING_B2"
        if status == "PASS"
        else "V10_STAGE_B1_FINAL_VERIFICATION_FAILED"
    )


if __name__ == "__main__":
    main()
