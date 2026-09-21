#!/usr/bin/env python3
"""Independent post-run checks for the FNSPID-augmented replay."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"
PRIVATE = ROOT / "work/stock-data/fnspid_augmented_4h/v1"
ORIGINAL = ROOT / "outputs/stock_forward_replay_4h/v1/predictions.csv"
CS = (0.01, 0.1, 1.0)
OLD = [f"{v}_{i}" for i in range(1, 7) for v in ("return", "range")] + ["history_age_hours", "return_mean", "return_std", "ny_hour"]
RECENT = [f"recent_{v}_{n}" for n in (5, 15, 30, 60) for v in ("return", "range", "rv", "missing")] + ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]
META = ["log_aug_articles", "log_aug_clusters", "log_aug_sources", "log_aug_newest_age", "log_aug_median_age", "aug_only_duplicates", "log_fnspid_groups"]


class PriceTransform:
    def transform(self, d, ii):
        return np.nan_to_num((d.iloc[ii][self.columns].to_numpy(float) - self.mean) / self.scale)


class SparseTransform:
    def transform(self, d, ii):
        return sparse.hstack([sparse.csr_matrix(self.price.transform(d, ii)), self.vectorizer.transform(d.iloc[ii].aug_stem_body.fillna(""))[:, self.keep]]).tocsr()


class SemanticTransform:
    def _raw(self, d, ii, means):
        z = self.pca.transform(means[ii]); z[d.iloc[ii].has_aug_news.to_numpy() == 0] = 0
        if self.kind == "event": extra = d.iloc[ii][META].to_numpy(float)
        else: extra = np.c_[np.log1p(d.iloc[ii].aug_articles.to_numpy(float)), d.iloc[ii].has_aug_news.to_numpy(float), np.log1p(d.iloc[ii].fnspid_added_groups.to_numpy(float))]
        return np.c_[z, extra]
    def transform(self, d, ii, means):
        return np.c_[self.price.transform(d, ii), self.scaler.transform(self._raw(d, ii, means))]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def signature(title):
    text = str(title).lower(); tokens = set(re.findall(r"[a-z0-9]+", text))
    groups = [r"\b(?:rais\w*|upgrad\w*|increas\w*|boost\w*)\b", r"\b(?:lower\w*|downgrad\w*|cut\w*|reduc\w*)\b", r"\b(?:maintain\w*|reiterat\w*)\b", r"\b(?:initiat\w*)\b", r"\b(?:deni\w*|deny\w*|not|no|refut\w*)\b", r"\b(?:accus\w*|alleg\w*)\b"]
    guard = (tuple(bool(re.search(x, text)) for x in groups), tuple(sorted(re.findall(r"\d+(?:[.,]\d+)*", text))), tuple(sorted(re.findall(r"\b(?:q[1-4]|first|second|third|fourth)\b", text))))
    return tokens, guard


def compatible(a, b):
    ta, ga = signature(a); tb, gb = signature(b)
    return ga == gb and bool(ta | tb) and len(ta & tb) / len(ta | tb) >= 0.8


def arrays(d):
    z = np.load(PRIVATE / "combined_embeddings.npz")
    keys = z["keys"].astype(str); lookup = {k: i for i, k in enumerate(keys)}
    fin = z["finbert"].astype(float); modern = z["modern"].astype(float)
    ids = [[lookup[k] for k in str(x).split("|") if k] for x in d.aug_news_record_keys.fillna("")]
    fin_mean = np.asarray([fin[ii].mean(0) if ii else np.zeros(fin.shape[1]) for ii in ids])
    modern_mean = np.asarray([modern[ii].mean(0) if ii else np.zeros(modern.shape[1]) for ii in ids])
    course = pd.read_pickle(ROOT / "work/stock-data/audit/news_index.pkl")
    course["article_key"] = course.archive + "::" + course.member
    titles = dict(zip(course.article_key, course.title.fillna("").astype(str)))
    ext = pd.read_parquet(PRIVATE / "fnspid_articles.parquet")
    titles.update(dict(zip(ext.article_key, ext.title.fillna("").astype(str))))
    events = []
    for row in d.itertuples(index=False):
        ks = [k for k in str(row.aug_news_record_keys).split("|") if k]
        clusters = []
        for k in [x for x in ks if not x.startswith("FNSPID::")]:
            target = next((c for c in clusters if all(compatible(titles[k], titles[j]) for j in c)), None)
            if target is None: clusters.append([k])
            else: target.append(k)
        clusters.extend([[k] for k in ks if k.startswith("FNSPID::")])
        events.append(np.mean([fin[[lookup[k] for k in c]].mean(0) for c in clusters], axis=0) if clusters else np.zeros(fin.shape[1]))
    return fin_mean, modern_mean, np.asarray(events)


def expected_c(cv, symbol, method, boundary):
    rows = cv[(cv.symbol == symbol) & (cv.method == method) & (cv.month < boundary)]
    if rows.empty: return 0.1
    return float(min(CS, key=lambda c: (-rows[np.isclose(rows.C, c)].BA.mean(), rows[np.isclose(rows.C, c)].Brier.mean(), c)))


def main():
    checks = {}
    d = pd.read_pickle(PRIVATE / "inputs.pkl").sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    for name, source in (("log_aug_articles", "aug_articles"), ("log_aug_clusters", "aug_clusters"), ("log_aug_sources", "aug_sources"), ("log_aug_newest_age", "aug_newest_age"), ("log_aug_median_age", "aug_median_age"), ("log_fnspid_groups", "fnspid_added_groups")):
        d[name] = np.log1p(d[source].to_numpy(float))
    p = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
    cv = pd.read_csv(OUT / "cv_metrics.csv")
    selections = pd.read_csv(OUT / "selections.csv")
    evidence = json.loads((OUT / "training_evidence.json").read_text())
    prep = json.loads((OUT / "preparation_evidence.json").read_text())
    checks["canonical_rows"] = len(d) == 1607 and d.key.nunique() == 1607 and p.key.tolist() == d.key.tolist()
    checks["evaluated_rows"] = int((p.phase != "warmup").sum()) == 1374
    checks["source_hashes"] = all(sha(ROOT / path) == value for path, value in evidence["source_hashes"].items())
    checks["fit_count"] = evidence["model_fits"] == 228 and len(evidence["evidence"]) == 228 and len(selections) == 84
    checks["chronology"] = all(pd.Timestamp(x["train_end"]) < pd.Timestamp(x["eval_cutoff_min"]) for x in evidence["evidence"])
    checks["runner_reload_tolerance"] = evidence["max_reload_error"] <= 1e-12
    checks["raw_fnspid_hashes_checked"] = prep["raw_record_hashes_verified"] == prep["representative_fnspid_groups"] == 1322

    external = pd.read_parquet(PRIVATE / "fnspid_articles.parquet").set_index("article_key")
    schedule = pd.read_csv(PRIVATE / "xnys_schedule.csv")
    session_index = dict(zip(schedule.session_date.astype(str), schedule.session_index.astype(int)))
    membership_errors = []
    for row in d.itertuples(index=False):
        current = session_index[str(row.day)]
        for key in [x for x in str(row.fnspid_record_keys).split("|") if x]:
            a = external.loc[key]
            if a.symbol != row.symbol or pd.Timestamp(a.available_at_utc) > pd.Timestamp(row.cutoff_utc) or not current - 2 <= int(a.available_session_index) <= current:
                membership_errors.append({"window": row.key, "article": key})
    checks["fnspid_cutoff_and_three_session_membership"] = not membership_errors
    checks["direct_only_article_keys"] = all(str(k).startswith("FNSPID::AAPL::") or str(k).startswith("FNSPID::AMZN::") for k in external.index)

    selection_errors = []
    for row in selections.itertuples(index=False):
        boundary = "2018-09" if row.month == "final" else row.month
        want = expected_c(cv, row.symbol, row.method, boundary)
        if not np.isclose(want, row.C): selection_errors.append({"symbol": row.symbol, "month": row.month, "method": row.method, "saved": row.C, "expected": want})
    checks["past_only_C_selection"] = not selection_errors

    fin_mean, modern_mean, event_mean = arrays(d)
    replay_errors = []; maximum = 0.0
    for row in selections.itertuples(index=False):
        ev = np.flatnonzero((d.symbol == row.symbol) & ((d.month >= "2018-09") if row.month == "final" else (d.month == row.month)))
        path = PRIVATE / "models" / f"{row.symbol}_{row.month}_{row.method}_{row.C}.joblib"
        saved = joblib.load(path)
        if row.method in ("PRICE_R1", "AUG_FULL_LR"):
            q = saved["model"].predict_proba(saved["transform"].transform(d, ev))[:, 1]
        elif row.method == "AUG_TFIDF_SVM":
            q = saved["model"].predict_proba(d.iloc[ev].aug_stem_body.fillna(""))[:, 1]
            no = d.iloc[ev].has_aug_news.to_numpy() == 0; q[no] = p.iloc[ev].PRICE_R1.to_numpy()[no]
        else:
            means = fin_mean if row.method == "AUG_FINBERT_LR" else modern_mean if row.method == "AUG_FINMODERN_LR" else event_mean
            q = saved["model"].predict_proba(saved["transform"].transform(d, ev, means))[:, 1]
        error = float(np.max(np.abs(q - p.iloc[ev][row.method].to_numpy(float))))
        maximum = max(maximum, error)
        if error > 1e-12: replay_errors.append({"symbol": row.symbol, "month": row.month, "method": row.method, "error": error})
    checks["independent_model_replay"] = not replay_errors and maximum <= 1e-12

    original = pd.read_csv(ORIGINAL, float_precision="round_trip").set_index("key")
    parity = {}
    for new, old in (("ORIG_PRICE_R1", "PRICE_R1"), ("ORIG_FULL_LR", "FULL_LR"), ("ORIG_TFIDF_SVM", "TFIDF_SVM"), ("ORIG_FINBERT_LR", "FINBERT_LR"), ("ORIG_FINMODERN_LR", "FINMODERN_LR"), ("ORIG_EVENT_META", "EVENT_META")):
        parity[new] = float(np.nanmax(np.abs(p[new].to_numpy(float) - p.key.map(original[old]).to_numpy(float))))
    checks["original_comparator_parity"] = max(parity.values()) <= 1e-12
    no_news = p.has_aug_news == 0
    fallback = float(np.max(np.abs(p.loc[no_news, "AUG_TFIDF_SVM"] - p.loc[no_news, "PRICE_R1"]))) if no_news.any() else 0.0
    checks["svm_no_news_exact_fallback"] = fallback <= 1e-12
    prob_cols = ["PRICE_R1", "AUG_FULL_LR", "AUG_TFIDF_SVM", "AUG_FINBERT_LR", "AUG_FINMODERN_LR", "AUG_EVENT_META"]
    evaluated = p.phase != "warmup"
    checks["probability_bounds"] = bool(((p.loc[evaluated, prob_cols] >= 0).all().all()) and ((p.loc[evaluated, prob_cols] <= 1).all().all()))
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "selection_errors": selection_errors, "membership_errors": membership_errors[:20], "replay_errors": replay_errors, "maximum_independent_probability_error": maximum, "original_comparator_max_errors": parity, "svm_no_news_fallback_error": fallback}
    (OUT / "VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
