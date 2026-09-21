#!/usr/bin/env python3
"""Chronological four-hour replay with course news plus direct FNSPID news."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"
PRIVATE = ROOT / "work/stock-data/fnspid_augmented_4h/v1"
MODEL_DIR = PRIVATE / "models"
ORIGINAL = ROOT / "outputs/stock_forward_replay_4h/v1/predictions.csv"
CS = (0.01, 0.1, 1.0)
MONTHS = [f"2018-{m:02d}" for m in range(3, 9)]
OLD = [f"{v}_{i}" for i in range(1, 7) for v in ("return", "range")] + ["history_age_hours", "return_mean", "return_std", "ny_hour"]
RECENT = [f"recent_{v}_{n}" for n in (5, 15, 30, 60) for v in ("return", "range", "rv", "missing")] + ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]
META = ["log_aug_articles", "log_aug_clusters", "log_aug_sources", "log_aug_newest_age", "log_aug_median_age", "aug_only_duplicates", "log_fnspid_groups"]
METHODS = ("PRICE_R1", "AUG_FULL_LR", "AUG_TFIDF_SVM", "AUG_FINBERT_LR", "AUG_FINMODERN_LR", "AUG_EVENT_META")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def metric(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); q = p >= 0.5
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise RuntimeError("invalid probabilities")
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, q)) if len(set(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, q)),
        "MCC": float(matthews_corrcoef(y, q)),
        "Brier": float(brier_score_loss(y, p)),
        "AUC": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "up_recall": float(q[y == 1].mean()) if (y == 1).any() else None,
        "down_recall": float((~q[y == 0]).mean()) if (y == 0).any() else None,
        "pred_up": float(q.mean()),
        "constant": bool(len(set(q)) == 1),
    }


def signature(title):
    text = str(title).lower(); tokens = set(re.findall(r"[a-z0-9]+", text))
    groups = [r"\b(?:rais\w*|upgrad\w*|increas\w*|boost\w*)\b", r"\b(?:lower\w*|downgrad\w*|cut\w*|reduc\w*)\b", r"\b(?:maintain\w*|reiterat\w*)\b", r"\b(?:initiat\w*)\b", r"\b(?:deni\w*|deny\w*|not|no|refut\w*)\b", r"\b(?:accus\w*|alleg\w*)\b"]
    guard = (tuple(bool(re.search(x, text)) for x in groups), tuple(sorted(re.findall(r"\d+(?:[.,]\d+)*", text))), tuple(sorted(re.findall(r"\b(?:q[1-4]|first|second|third|fourth)\b", text))))
    return tokens, guard


def compatible(a, b):
    ta, ga = signature(a); tb, gb = signature(b)
    return ga == gb and bool(ta | tb) and len(ta & tb) / len(ta | tb) >= 0.8


class PriceTransform:
    def __init__(self, columns): self.columns = columns
    def fit(self, d, ii):
        x = d.iloc[ii][self.columns].to_numpy(float)
        self.mean = np.nan_to_num(np.nanmean(x, axis=0))
        self.scale = np.nanstd(x, axis=0); self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        return self
    def transform(self, d, ii):
        return np.nan_to_num((d.iloc[ii][self.columns].to_numpy(float) - self.mean) / self.scale)


class SparseTransform:
    def fit(self, d, ii):
        self.price = PriceTransform(OLD).fit(d, ii)
        tr = d.iloc[ii]
        self.vectorizer = CountVectorizer(binary=True, min_df=3)
        matrix = self.vectorizer.fit_transform(tr.aug_stem_body.fillna(""))
        scores = chi2(matrix, tr.label)[0]
        terms = self.vectorizer.get_feature_names_out()
        self.keep = np.lexsort((terms, -np.nan_to_num(scores, nan=-np.inf)))[:500]
        return self
    def transform(self, d, ii):
        return sparse.hstack([sparse.csr_matrix(self.price.transform(d, ii)), self.vectorizer.transform(d.iloc[ii].aug_stem_body.fillna(""))[:, self.keep]]).tocsr()


class SemanticTransform:
    def __init__(self, kind): self.kind = kind
    def fit(self, d, ii, embeddings, article_ids, means):
        self.price = PriceTransform(OLD).fit(d, ii)
        train_articles = sorted({j for i in ii for j in article_ids[i]})
        if len(train_articles) < 16:
            raise RuntimeError("not enough training articles")
        self.pca = PCA(16, svd_solver="randomized", random_state=573).fit(embeddings[train_articles])
        self.scaler = StandardScaler().fit(self._raw(d, ii, means))
        return self
    def _raw(self, d, ii, means):
        z = self.pca.transform(means[ii])
        no = d.iloc[ii].has_aug_news.to_numpy() == 0
        z[no] = 0
        if self.kind == "event":
            extra = d.iloc[ii][META].to_numpy(float)
        else:
            extra = np.c_[np.log1p(d.iloc[ii].aug_articles.to_numpy(float)), d.iloc[ii].has_aug_news.to_numpy(float), np.log1p(d.iloc[ii].fnspid_added_groups.to_numpy(float))]
        return np.c_[z, extra]
    def transform(self, d, ii, means):
        return np.c_[self.price.transform(d, ii), self.scaler.transform(self._raw(d, ii, means))]


def choose_c(records, symbol, method):
    rows = [r for r in records if r["symbol"] == symbol and r["method"] == method]
    if not rows: return 0.1
    return min(CS, key=lambda c: (-np.mean([r["BA"] for r in rows if r["C"] == c]), np.mean([r["Brier"] for r in rows if r["C"] == c]), c))


def svm_model(c):
    base = make_pipeline(TfidfVectorizer(min_df=3, max_features=5000, sublinear_tf=True), LinearSVC(C=c, max_iter=10000, random_state=573))
    return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3), ensemble=True)


def fit_method(method, c, d, tr, ev, arrays):
    fin, modern, fin_ids, modern_ids, fin_mean, modern_mean, event_mean = arrays
    if method == "PRICE_R1":
        transform = PriceTransform(OLD + RECENT).fit(d, tr)
        model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, tol=1e-7, random_state=573)
        model.fit(transform.transform(d, tr), d.iloc[tr].label)
        raw = model.predict_proba(transform.transform(d, ev))[:, 1]
    elif method == "AUG_FULL_LR":
        transform = SparseTransform().fit(d, tr)
        model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, tol=1e-7, random_state=573)
        model.fit(transform.transform(d, tr), d.iloc[tr].label)
        raw = model.predict_proba(transform.transform(d, ev))[:, 1]
    elif method == "AUG_TFIDF_SVM":
        transform = None; model = svm_model(c)
        model.fit(d.iloc[tr].aug_stem_body.fillna(""), d.iloc[tr].label)
        raw = model.predict_proba(d.iloc[ev].aug_stem_body.fillna(""))[:, 1]
    else:
        if method == "AUG_FINBERT_LR": emb, ids, means, kind = fin, fin_ids, fin_mean, "article"
        elif method == "AUG_FINMODERN_LR": emb, ids, means, kind = modern, modern_ids, modern_mean, "article"
        elif method == "AUG_EVENT_META": emb, ids, means, kind = fin, fin_ids, event_mean, "event"
        else: raise ValueError(method)
        transform = SemanticTransform(kind).fit(d, tr, emb, ids, means)
        model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, tol=1e-7, random_state=573)
        model.fit(transform.transform(d, tr, means), d.iloc[tr].label)
        raw = model.predict_proba(transform.transform(d, ev, means))[:, 1]
    return transform, model, raw


def replay(saved, method, d, ev, arrays):
    fin, modern, _, _, fin_mean, modern_mean, event_mean = arrays
    if method in ("PRICE_R1", "AUG_FULL_LR"):
        return saved["model"].predict_proba(saved["transform"].transform(d, ev))[:, 1]
    if method == "AUG_TFIDF_SVM":
        return saved["model"].predict_proba(d.iloc[ev].aug_stem_body.fillna(""))[:, 1]
    means = fin_mean if method == "AUG_FINBERT_LR" else modern_mean if method == "AUG_FINMODERN_LR" else event_mean
    return saved["model"].predict_proba(saved["transform"].transform(d, ev, means))[:, 1]


def prepare_arrays(d):
    z = np.load(PRIVATE / "combined_embeddings.npz")
    keys = z["keys"].astype(str); lookup = {k: i for i, k in enumerate(keys)}
    fin = z["finbert"].astype(float); modern = z["modern"].astype(float)
    article_ids = [[lookup[k] for k in str(x).split("|") if k] for x in d.aug_news_record_keys.fillna("")]
    fin_mean = np.asarray([fin[ii].mean(0) if ii else np.zeros(fin.shape[1]) for ii in article_ids])
    modern_mean = np.asarray([modern[ii].mean(0) if ii else np.zeros(modern.shape[1]) for ii in article_ids])

    course = pd.read_pickle(ROOT / "work/stock-data/audit/news_index.pkl")
    course["article_key"] = course.archive + "::" + course.member
    titles = dict(zip(course.article_key, course.title.fillna("").astype(str)))
    ext = pd.read_parquet(PRIVATE / "fnspid_articles.parquet")
    titles.update(dict(zip(ext.article_key, ext.title.fillna("").astype(str))))
    event_mean = []
    for row, ii in zip(d.itertuples(index=False), article_ids):
        ks = [k for k in str(row.aug_news_record_keys).split("|") if k]
        original = [k for k in ks if not k.startswith("FNSPID::")]
        external = [k for k in ks if k.startswith("FNSPID::")]
        clusters = []
        for k in original:
            target = next((c for c in clusters if all(compatible(titles[k], titles[j]) for j in c)), None)
            if target is None: clusters.append([k])
            else: target.append(k)
        clusters.extend([[k] for k in external])
        event_mean.append(np.mean([fin[[lookup[k] for k in c]].mean(0) for c in clusters], axis=0) if clusters else np.zeros(fin.shape[1]))
    return fin, modern, article_ids, article_ids, fin_mean, modern_mean, np.asarray(event_mean)


def main():
    if MODEL_DIR.exists(): raise FileExistsError("models already exist; run is immutable")
    if not (PRIVATE / "inputs.pkl").exists(): raise FileNotFoundError("run prepare.py first")
    MODEL_DIR.mkdir(parents=True)
    d = pd.read_pickle(PRIVATE / "inputs.pkl").sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    for name, source in (("log_aug_articles", "aug_articles"), ("log_aug_clusters", "aug_clusters"), ("log_aug_sources", "aug_sources"), ("log_aug_newest_age", "aug_newest_age"), ("log_aug_median_age", "aug_median_age"), ("log_fnspid_groups", "fnspid_added_groups")):
        d[name] = np.log1p(d[source].to_numpy(float))
    arrays = prepare_arrays(d)
    original = pd.read_csv(ORIGINAL, float_precision="round_trip").set_index("key")
    if len(d) != 1607 or d.key.nunique() != 1607 or set(d.key) != set(original.index): raise RuntimeError("canonical key mismatch")

    cv, selections, evidence = [], [], []
    started = time.monotonic()
    with threadpool_limits(2):
        for symbol in ("AAPL", "AMZN"):
            for month in MONTHS + ["final"]:
                boundary = "2018-09" if month == "final" else month
                tr = np.flatnonzero((d.symbol == symbol) & (d.month < boundary))
                ev = np.flatnonzero((d.symbol == symbol) & ((d.month >= "2018-09") if month == "final" else (d.month == month)))
                if d.iloc[tr].end_utc.max() >= d.iloc[ev].cutoff_utc.min(): raise RuntimeError("chronology violation")
                for method in METHODS:
                    chosen = choose_c(cv, symbol, method)
                    candidates = (chosen,) if month == "final" else CS
                    for c in candidates:
                        tick = time.monotonic()
                        transform, model, raw = fit_method(method, c, d, tr, ev, arrays)
                        path = MODEL_DIR / f"{symbol}_{month}_{method}_{c}.joblib"
                        joblib.dump({"transform": transform, "model": model, "train_keys": d.iloc[tr].key.tolist()}, path, compress=3)
                        loaded = joblib.load(path); again = replay(loaded, method, d, ev, arrays)
                        error = float(np.max(np.abs(raw - again)))
                        if error > 1e-12: raise RuntimeError(f"reload mismatch {error}")
                        issued = raw.copy()
                        if method == "AUG_TFIDF_SVM":
                            no = d.iloc[ev].has_aug_news.to_numpy() == 0
                            issued[no] = d.iloc[ev].PRICE_R1.to_numpy()[no]
                        score = metric(d.iloc[ev].label, raw)
                        if month != "final": cv.append({"symbol": symbol, "month": month, "method": method, "C": c, **score})
                        if c == chosen: d.loc[ev, method] = issued
                        evidence.append({"symbol": symbol, "month": month, "method": method, "C": c, "train_n": len(tr), "eval_n": len(ev), "train_end": str(d.iloc[tr].end_utc.max()), "eval_cutoff_min": str(d.iloc[ev].cutoff_utc.min()), "reload_error": error, "model_sha256": sha(path), "seconds": time.monotonic() - tick})
                    selections.append({"symbol": symbol, "month": month, "method": method, "C": chosen, "selection_months": sorted({r["month"] for r in cv if r["symbol"] == symbol and r["method"] == method and r["month"] < boundary})})
                print(symbol, month, "complete", round(time.monotonic() - started, 1), flush=True)

    comparators = {"ORIG_PRICE_R1": "PRICE_R1", "ORIG_FULL_LR": "FULL_LR", "ORIG_TFIDF_SVM": "TFIDF_SVM", "ORIG_FINBERT_LR": "FINBERT_LR", "ORIG_FINMODERN_LR": "FINMODERN_LR", "ORIG_EVENT_META": "EVENT_META"}
    for new, old in comparators.items(): d[new] = d.key.map(original[old])
    if np.max(np.abs(d.loc[d.phase != "warmup", "PRICE_R1"] - d.loc[d.phase != "warmup", "ORIG_PRICE_R1"])) > 1e-12: raise RuntimeError("price parity failed")

    all_methods = list(METHODS) + list(comparators)
    rows, monthly, subgroups, transitions = [], [], [], []
    for (phase, symbol), g in d[d.phase != "warmup"].groupby(["phase", "symbol"]):
        for method in all_methods:
            rows.append({"phase": phase, "symbol": symbol, "method": method, **metric(g.label, g[method])})
            for mon, q in g.groupby("month"): monthly.append({"phase": phase, "symbol": symbol, "month": mon, "method": method, **metric(q.label, q[method])})
            for flag in ("has_original_news", "has_aug_news"):
                for value, q in g.groupby(flag): subgroups.append({"phase": phase, "symbol": symbol, "flag": flag, "value": int(value), "method": method, **metric(q.label, q[method])})
        for augmented, base in (("AUG_FULL_LR", "ORIG_FULL_LR"), ("AUG_TFIDF_SVM", "ORIG_TFIDF_SVM"), ("AUG_FINBERT_LR", "ORIG_FINBERT_LR"), ("AUG_FINMODERN_LR", "ORIG_FINMODERN_LR"), ("AUG_EVENT_META", "ORIG_EVENT_META")):
            a = (g[augmented] >= 0.5) == g.label; b = (g[base] >= 0.5) == g.label
            transitions.append({"phase": phase, "symbol": symbol, "method": augmented, "baseline": base, "repaired": int((a & ~b).sum()), "introduced": int((~a & b).sum()), "changed": int(((g[augmented] >= 0.5) != (g[base] >= 0.5)).sum())})
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_metrics.csv", index=False)
    pd.DataFrame(subgroups).to_csv(OUT / "subgroup_metrics.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUT / "transitions.csv", index=False)
    pd.DataFrame(cv).to_csv(OUT / "cv_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
    cols = ["key", "symbol", "day", "month", "phase", "label", "has_original_news", "has_aug_news", "fnspid_added_groups"] + all_methods
    d[cols].to_csv(OUT / "predictions.csv", index=False, float_format="%.17g")
    dump(OUT / "training_evidence.json", {"status": "COMPLETE_PENDING_VERIFICATION", "rows": len(d), "evaluated_rows": int((d.phase != "warmup").sum()), "model_fits": len(evidence), "evidence": evidence, "max_reload_error": max(x["reload_error"] for x in evidence), "seconds": time.monotonic() - started, "development_and_later_exposed": True, "fnspid_used": True, "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (PRIVATE / "inputs.pkl", PRIVATE / "combined_embeddings.npz", ORIGINAL, HERE / "PRE_REGISTRATION.md")}})
    print(pd.DataFrame(rows).query("phase in ['development','later']").to_string(index=False), flush=True)


if __name__ == "__main__": main()
