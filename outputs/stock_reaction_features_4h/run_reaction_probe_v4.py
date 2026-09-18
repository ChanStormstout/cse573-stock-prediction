"""Run the repaired v4 AR0/AR1 article gate; downstream remains conditional."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work" / "stock-data"
PUBLIC = HERE / "v4"
PRIVATE = WORK / "reaction_features_4h" / "v4"
SOURCE_PRIVATE = PRIVATE
HORIZONS = (30, 60, 120, 240)
CS = (0.01, 0.1, 1.0)
CONTINUOUS = [
    f"pre_{minutes}m_{kind}"
    for minutes in (5, 15, 30, 60)
    for kind in ("return", "rv")
] + ["minutes_from_open"]
VALIDITY_FLAGS = [f"pre_{minutes}m_valid" for minutes in (5, 15, 30, 60)]


def load_base():
    spec = importlib.util.spec_from_file_location(
        "reaction_probe_v3_for_v4", HERE / "run_reaction_probe_v3.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import reaction v3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_design_v4(train, evaluate, method):
    """Past-fold transforms; binary context-validity flags remain unscaled."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        train[CONTINUOUS]
    )
    train_continuous = imputer.transform(train[CONTINUOUS])
    eval_continuous = imputer.transform(evaluate[CONTINUOUS])
    scaler = StandardScaler().fit(train_continuous)
    train_continuous = scaler.transform(train_continuous)
    eval_continuous = scaler.transform(eval_continuous)
    train_flags = train[VALIDITY_FLAGS].to_numpy(float)
    eval_flags = evaluate[VALIDITY_FLAGS].to_numpy(float)
    if not np.isin(train_flags, (0.0, 1.0)).all() or not np.isin(eval_flags, (0.0, 1.0)).all():
        raise AssertionError("pre-context validity flags must be binary")
    x_numeric = np.c_[train_continuous, train_flags]
    z_numeric = np.c_[eval_continuous, eval_flags]
    evidence = {
        "continuous_columns": CONTINUOUS,
        "binary_validity_columns": VALIDITY_FLAGS,
        "numeric_imputer_fit_rows": int(len(train)),
        "numeric_scaler_fit_rows": int(len(train)),
        "continuous_transform": "SimpleImputer(median) then StandardScaler fit on past fold only",
        "binary_validity_transform": "unscaled explicit 0/1 indicators",
    }
    if method == "AR0":
        return x_numeric, z_numeric, evidence
    vectorizer = CountVectorizer(binary=True, min_df=3)
    raw = vectorizer.fit_transform(train.context.fillna(""))
    terms = vectorizer.get_feature_names_out()
    scores = chi2(raw, train.label.to_numpy(int))[0] if raw.shape[1] else np.array([])
    keep = (
        np.lexsort((terms, -np.nan_to_num(scores, nan=-np.inf)))[: min(500, raw.shape[1])]
        if raw.shape[1]
        else np.array([], dtype=int)
    )
    train_text = raw[:, keep].toarray() if len(keep) else np.zeros((len(train), 0))
    eval_text = (
        vectorizer.transform(evaluate.context.fillna(""))[:, keep].toarray()
        if len(keep)
        else np.zeros((len(evaluate), 0))
    )
    evidence.update(
        {
            "lexical_fit_rows": int(len(train)),
            "lexical_vocabulary_size": int(raw.shape[1]),
            "lexical_selected_features": int(len(keep)),
            "lexical_selection": "chi2_on_past_fold_then_lexicographic_tie_break",
        }
    )
    return np.c_[x_numeric, train_text], np.c_[z_numeric, eval_text], evidence


def write_report(public: Path, promotions: list[dict], downstream: list[dict]) -> None:
    lines = ["# Reaction v4 report", "", "## Scope", "", "v4 reruns all preregistered 30/60/120/240 minute AR0/AR1 article candidates with time-safe completed-bar context. Missing continuous context is imputed fold-locally and has explicit unscaled validity flags. AR2 is not run because its registered target-context FinBERT binary is unavailable.", "", "## Article gate"]
    for row in promotions:
        stock = {x["symbol"]: x["mean_delta_BA"] for x in row["stock_rows"]}
        lines.append(
            f"- {row['candidate']} {row['horizon_minutes']}m: {'PASS' if row['passes'] else 'FAIL'}; "
            f"AAPL mean ΔBA={stock.get('AAPL')}, AMZN mean ΔBA={stock.get('AMZN')}, "
            f"macro AUC Δ={row['macro']['macro_auc_delta']}."
        )
    lines += ["", "## Downstream status", ""]
    if downstream:
        lines.extend(
            f"- {x['candidate']} {x['horizon_minutes']}m: {x['status']}."
            for x in downstream
        )
    else:
        lines.append("- No AR1 horizon passed its full preregistered article gate; W0--W3 was not run.")
    lines += ["", "All values are exposed exploratory historical backtests. v3 is preserved and superseded; no Phase B G0--G3 execution occurred.", ""]
    (public / "REPORT.md").write_text("\n".join(lines))


def main(public: Path = PUBLIC, private: Path = PRIVATE) -> None:
    # v4 preparation owns the new directories; this runner may add only model
    # and gate artifacts to that prepared v4 corpus. It never resumes a partly
    # written predictive run.
    if not public.exists() or not private.exists():
        raise FileNotFoundError("run build_reaction_dataset_v4.py before v4 article fitting")
    if (public / "article_reaction_predictions.csv").exists() or (public / "protocol.json").exists() or (private / "models").exists():
        raise FileExistsError("reaction v4 article run must not overwrite or resume an existing fit")
    base = load_base()
    # Configure the imported historical implementation only in this process.
    base.PUBLIC = public
    base.PRIVATE = private
    base.SOURCE_PRIVATE = SOURCE_PRIVATE
    base.HORIZONS = HORIZONS
    base.CS = CS
    base.fit_design = fit_design_v4
    (private / "models").mkdir(parents=True)
    protocol = HERE / "PRE_REGISTRATION_v4.md"
    base.write_json(
        public / "protocol.json",
        {
            "protocol": "reaction_v4_full_registered_horizons_missingness_repair",
            "preregistration_sha256": base.sha(protocol),
            "runner_sha256": base.sha(HERE / "run_reaction_probe_v4.py"),
            "builder_sha256": base.sha(HERE / "build_reaction_dataset_v4.py"),
            "horizons": list(HORIZONS),
            "AR2": "NOT_RUN_MODEL_UNAVAILABLE",
            "downstream_rule": "run every independently promoted candidate; never choose by W0-W3 performance",
        },
    )
    started = time.monotonic()
    all_rows = base.prepare()
    train_rows = all_rows[all_rows.available_utc < base.pd.Timestamp("2018-09-01", tz="UTC")].copy()
    prediction, selections, fits = base.run_article_models(train_rows)
    prediction.to_csv(public / "article_reaction_predictions.csv", index=False)
    metrics, promotions, ar2 = base.article_gate(prediction, selections, public)

    downstream: list[dict] = []
    passed = [row for row in promotions if row["passes"]]
    for candidate in passed:
        horizon = int(candidate["horizon_minutes"])
        c_value = float(candidate["C"])
        candidate_public = public / f"official_AR1_h{horizon}m"
        candidate_private = private / f"official_AR1_h{horizon}m"
        candidate_public.mkdir(parents=True)
        candidate_private.mkdir(parents=True)
        later, later_fits = base.build_later_article_predictions(
            all_rows, horizon, c_value, candidate_public, candidate_private
        )
        fits.extend(later_fits)
        selected = prediction[
            (prediction.method == "AR1") & (prediction.horizon_minutes == horizon)
        ][["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "prediction"]].copy()
        selected["available_utc"] = base.pd.to_datetime(selected.available_utc, utc=True, format="mixed")
        score = base.pd.concat(
            [selected, later[["article_key", "target_pair_key", "symbol", "group_key", "available_utc", "prediction"]]],
            ignore_index=True,
        )
        status = base.official_windows(score, horizon, candidate_public, candidate_private)
        downstream.append({"candidate": "AR1", "horizon_minutes": horizon, **status})

    base.write_json(
        public / "official_window_status.json",
        {
            "status": "COMPLETE_FOR_EACH_PROMOTED_AR1" if passed else "NOT_RUN_NO_PROMOTED_ARTICLE_HORIZON",
            "promoted_candidates": [
                {"candidate": "AR1", "horizon_minutes": int(x["horizon_minutes"]), "C": x["C"]}
                for x in passed
            ],
            "protocol": "W0=R1; W1=R1+coverage; W2=R1+five reaction features; W3=F1+R1+five reaction features",
            "selection_source": "article_gate_only; no downstream performance selection",
        },
    )
    base.write_json(
        public / "training_evidence.json",
        {
            "status": "ARTICLE_AR0_AR1_COMPLETE_AR2_NOT_RUN",
            "fits": fits,
            "fit_count": len(fits),
            "selection": selections,
            "AR2": ar2,
            "continuous_columns": CONTINUOUS,
            "binary_validity_columns": VALIDITY_FLAGS,
            "runtime_seconds": time.monotonic() - started,
        },
    )
    base.write_json(
        public / "reaction_probe_gate.json",
        {
            "status": "COMPLETE",
            "promotion": promotions,
            "AR2": ar2,
            "promoted_article_candidates": [
                {"candidate": "AR1", "horizon_minutes": int(x["horizon_minutes"]), "C": x["C"]}
                for x in passed
            ],
            "downstream": downstream,
        },
    )
    write_report(public, promotions, downstream)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "article_prediction_rows": int(len(prediction)),
                "fit_count": int(len(fits)),
                "passed_horizons": [int(x["horizon_minutes"]) for x in passed],
                "downstream_count": int(len(downstream)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    args = parser.parse_args()
    main(args.public, args.private)
