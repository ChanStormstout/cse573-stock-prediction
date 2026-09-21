#!/usr/bin/env python3
"""Independent integrity checks for chronological forward replay v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"
PRIVATE = ROOT / "work/stock-data/forward_replay_4h/v1"
PAPER_PUBLIC = ROOT / "outputs/stock_paper_methods_4h/v3"
PAPER_PRIVATE = ROOT / "work/stock-data/paper_methods_4h/v3"
INTEGRATED = ROOT / "outputs/stock_integrated_4h/prepared/articles.npz"
MODERN = ROOT / "work/stock-data/foundation_4h/v2/modern_embeddings.npz"
OLD = [f"{v}_{i}" for i in range(1, 7) for v in ("return", "range")] + ["history_age_hours", "return_mean", "return_std", "ny_hour"]
RECENT = [f"recent_{v}_{n}" for n in (5, 15, 30, 60) for v in ("return", "range", "rv", "missing")] + ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]
CS = (0.01, 0.1, 1.0)


# These classes intentionally duplicate the runner-side transform interface so
# joblib artifacts created by a script entrypoint can be independently loaded.
class PriceTransform:
    def transform(self, d, ii):
        x = d.iloc[ii][OLD + RECENT].to_numpy(float)
        return np.nan_to_num((x - self.mean) / self.scale)


class PriceTransformOld:
    def transform(self, d, ii):
        return np.nan_to_num((d.iloc[ii][OLD].to_numpy(float) - self.mean) / self.scale)


class ModernTransform:
    def _raw(self, d, ii, means):
        z = self.pca.transform(means[ii])
        z[d.iloc[ii]["has_original_news"].to_numpy() == 0] = 0
        return np.c_[z, np.log1p(d.iloc[ii]["news_count"].to_numpy(float)), d.iloc[ii]["has_news"].to_numpy(float)]

    def transform(self, d, ii, means):
        return np.c_[self.price.transform(d, ii), self.scaler.transform(self._raw(d, ii, means))]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_c(cv: pd.DataFrame, symbol: str, method: str, boundary: str) -> float:
    rows = cv[(cv.symbol == symbol) & (cv.method == method) & (cv.month < boundary)]
    if rows.empty:
        return 0.1
    scores = []
    for c in CS:
        q = rows[np.isclose(rows.C, c)]
        scores.append((-q.BA.mean(), q.Brier.mean(), c))
    return float(min(scores)[2])


def main() -> None:
    checks = {}
    manifest = json.loads((OUT / "training_evidence.json").read_text())
    predictions = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
    selections = pd.read_csv(OUT / "selections.csv")
    cv = pd.read_csv(OUT / "cv_metrics.csv")
    d = pd.read_pickle(PAPER_PRIVATE / "inputs.pkl").sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    paper = pd.read_csv(PAPER_PUBLIC / "predictions.csv", float_precision="round_trip").set_index("key")

    checks["canonical_rows"] = len(d) == 1607 and d.key.nunique() == 1607
    checks["evaluated_rows"] = len(predictions) == 1607 and int((predictions.phase != "warmup").sum()) == 1374
    checks["key_order"] = predictions.key.tolist() == d.key.tolist()
    source_paths = [PAPER_PRIVATE / "inputs.pkl", PAPER_PUBLIC / "predictions.csv", INTEGRATED, MODERN, HERE / "PRE_REGISTRATION.md"]
    actual_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    checks["source_hashes"] = actual_hashes == manifest["source_hashes"]
    evidence = manifest["evidence"]
    checks["fit_count"] = len(evidence) == 114 and manifest["new_model_fits"] == 114
    checks["chronology"] = all(pd.Timestamp(x["train_end"]) < pd.Timestamp(x["eval_cutoff_min"]) for x in evidence)
    checks["reload_tolerance_runner"] = manifest["max_reload_error"] <= 1e-12

    # Reconstruct frozen article membership and FinModernBERT window means.
    fin = np.load(INTEGRATED)
    modern = np.load(MODERN)
    checks["encoder_article_key_parity"] = np.array_equal(fin["keys"], modern["keys"])
    lookup = {k: i for i, k in enumerate(fin["keys"])}
    article_ids = [[lookup[k] for k in str(x).split("|") if k] for x in d.news_record_keys.fillna("")]
    emb = modern["embeddings"].astype(float)
    means = np.asarray([emb[ii].mean(0) if ii else np.zeros(emb.shape[1]) for ii in article_ids])

    # Recompute every registered C decision using only previously completed folds.
    selection_errors = []
    for row in selections.itertuples(index=False):
        boundary = "2018-09" if row.month == "final" else row.month
        expected = select_c(cv, row.symbol, row.method, boundary)
        if not np.isclose(expected, row.C):
            selection_errors.append({"symbol": row.symbol, "month": row.month, "method": row.method, "saved": row.C, "expected": expected})
    checks["past_only_C_selection"] = not selection_errors

    # Independently load each issued model and reproduce every issued probability.
    replay_errors = []
    max_error = 0.0
    for row in selections.itertuples(index=False):
        ev = np.flatnonzero((d.symbol == row.symbol) & ((d.month >= "2018-09") if row.month == "final" else (d.month == row.month)))
        path = PRIVATE / "models" / f"{row.symbol}_{row.month}_{row.method}_{row.C}.joblib"
        saved = joblib.load(path)
        if row.method == "PRICE_R1":
            p = saved["model"].predict_proba(saved["transform"].transform(d, ev))[:, 1]
        elif row.method == "TFIDF_SVM":
            p = saved["model"].predict_proba(d.iloc[ev].stem_body.fillna(""))[:, 1]
            no = d.iloc[ev].has_original_news.to_numpy() == 0
            p[no] = predictions.iloc[ev].PRICE_R1.to_numpy()[no]
        else:
            p = saved["model"].predict_proba(saved["transform"].transform(d, ev, means))[:, 1]
        got = predictions.iloc[ev][row.method].to_numpy(float)
        err = float(np.max(np.abs(p - got)))
        max_error = max(max_error, err)
        if err > 1e-12:
            replay_errors.append({"symbol": row.symbol, "month": row.month, "method": row.method, "error": err})
    checks["independent_model_replay"] = not replay_errors and max_error <= 1e-12

    # Existing freshly rerun paper branches must match exactly.
    evaluated = predictions.phase != "warmup"
    parity = {}
    for public, source in (("FULL_LR", "J0"), ("FINBERT_LR", "J2"), ("EVENT_META", "N1M")):
        err = float(np.max(np.abs(predictions.loc[evaluated, public].to_numpy() - d.loc[evaluated, "key"].map(paper[source]).to_numpy())))
        parity[public] = err
    checks["paper_branch_parity"] = max(parity.values()) <= 1e-12
    no_news = predictions.has_original_news == 0
    fallback_error = float(np.max(np.abs(predictions.loc[no_news, "TFIDF_SVM"] - predictions.loc[no_news, "PRICE_R1"])))
    checks["svm_no_news_exact_fallback"] = fallback_error <= 1e-12
    probability_columns = ["PRICE_R1", "FULL_LR", "TFIDF_SVM", "FINBERT_LR", "FINMODERN_LR", "EVENT_META"]
    checks["probability_bounds"] = bool(((predictions.loc[evaluated, probability_columns] >= 0).all().all()) and ((predictions.loc[evaluated, probability_columns] <= 1).all().all()))

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selection_errors": selection_errors,
        "replay_errors": replay_errors,
        "maximum_independent_probability_error": max_error,
        "paper_branch_max_errors": parity,
        "svm_no_news_fallback_error": fallback_error,
    }
    (OUT / "VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
