#!/usr/bin/env python3
"""Matched chronological replay for price, SVM, FinBERT and FinModernBERT."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"
PRIVATE = ROOT / "work/stock-data/forward_replay_4h/v1"
PAPER_PUBLIC = ROOT / "outputs/stock_paper_methods_4h/v3"
PAPER_PRIVATE = ROOT / "work/stock-data/paper_methods_4h/v3"
INTEGRATED = ROOT / "outputs/stock_integrated_4h/prepared/articles.npz"
MODERN = ROOT / "work/stock-data/foundation_4h/v2/modern_embeddings.npz"
CS = (0.01, 0.1, 1.0)
MONTHS = [f"2018-{m:02d}" for m in range(3, 9)]
OLD = [f"{v}_{i}" for i in range(1, 7) for v in ("return", "range")] + ["history_age_hours", "return_mean", "return_std", "ny_hour"]
RECENT = [f"recent_{v}_{n}" for n in (5, 15, 30, 60) for v in ("return", "range", "rv", "missing")] + ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def metric(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); q = p >= 0.5
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


class PriceTransform:
    def fit(self, d, ii):
        x = d.iloc[ii][OLD + RECENT].to_numpy(float)
        self.mean = np.nanmean(x, axis=0); self.mean = np.nan_to_num(self.mean)
        self.scale = np.nanstd(x, axis=0); self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        return self

    def transform(self, d, ii):
        x = d.iloc[ii][OLD + RECENT].to_numpy(float)
        return np.nan_to_num((x - self.mean) / self.scale)


class ModernTransform:
    def fit(self, d, ii, embeddings, article_ids, means):
        self.price = PriceTransformOld().fit(d, ii)
        train_articles = sorted({j for i in ii for j in article_ids[i]})
        self.pca = PCA(16, svd_solver="randomized", random_state=573).fit(embeddings[train_articles])
        z = self._raw(d, ii, means)
        self.scaler = StandardScaler().fit(z)
        return self

    def _raw(self, d, ii, means):
        z = self.pca.transform(means[ii])
        no = d.iloc[ii]["has_original_news"].to_numpy() == 0
        z[no] = 0
        return np.c_[z, np.log1p(d.iloc[ii]["news_count"].to_numpy(float)), d.iloc[ii]["has_news"].to_numpy(float)]

    def transform(self, d, ii, means):
        return np.c_[self.price.transform(d, ii), self.scaler.transform(self._raw(d, ii, means))]


class PriceTransformOld:
    def fit(self, d, ii):
        x = d.iloc[ii][OLD].to_numpy(float)
        self.mean = np.nanmean(x, axis=0); self.mean = np.nan_to_num(self.mean)
        self.scale = np.nanstd(x, axis=0); self.scale = np.where(self.scale > 1e-12, self.scale, 1.0)
        return self

    def transform(self, d, ii):
        return np.nan_to_num((d.iloc[ii][OLD].to_numpy(float) - self.mean) / self.scale)


def choose_c(records, symbol, method):
    rows = [r for r in records if r["symbol"] == symbol and r["method"] == method]
    if not rows:
        return 0.1
    return min(CS, key=lambda c: (
        -np.mean([r["BA"] for r in rows if r["C"] == c]),
        np.mean([r["Brier"] for r in rows if r["C"] == c]), c))


def svm_model(c):
    base = make_pipeline(TfidfVectorizer(min_df=3, max_features=5000, sublinear_tf=True),
                         LinearSVC(C=c, max_iter=10000, random_state=573))
    return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3), ensemble=True)


def fit_one(method, c, d, tr, ev, modern_embeddings, article_ids, modern_means):
    if method == "PRICE_R1":
        transform = PriceTransform().fit(d, tr)
        model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, tol=1e-7, random_state=573)
        model.fit(transform.transform(d, tr), d.iloc[tr].label)
        p = model.predict_proba(transform.transform(d, ev))[:, 1]
    elif method == "TFIDF_SVM":
        transform = None
        model = svm_model(c)
        model.fit(d.iloc[tr].stem_body.fillna(""), d.iloc[tr].label)
        p = model.predict_proba(d.iloc[ev].stem_body.fillna(""))[:, 1]
    elif method == "FINMODERN_LR":
        transform = ModernTransform().fit(d, tr, modern_embeddings, article_ids, modern_means)
        model = LogisticRegression(C=c, solver="liblinear", max_iter=3000, tol=1e-7, random_state=573)
        model.fit(transform.transform(d, tr, modern_means), d.iloc[tr].label)
        p = model.predict_proba(transform.transform(d, ev, modern_means))[:, 1]
    else:
        raise ValueError(method)
    return transform, model, p


def replay(saved, method, d, ev, modern_means):
    if method == "PRICE_R1":
        return saved["model"].predict_proba(saved["transform"].transform(d, ev))[:, 1]
    if method == "TFIDF_SVM":
        return saved["model"].predict_proba(d.iloc[ev].stem_body.fillna(""))[:, 1]
    return saved["model"].predict_proba(saved["transform"].transform(d, ev, modern_means))[:, 1]


def main():
    if OUT.exists() or PRIVATE.exists():
        raise FileExistsError("v1 exists; historical runs are immutable")
    OUT.mkdir(parents=True); (PRIVATE / "models").mkdir(parents=True)
    d = pd.read_pickle(PAPER_PRIVATE / "inputs.pkl").sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    paper = pd.read_csv(PAPER_PUBLIC / "predictions.csv", float_precision="round_trip").set_index("key")
    if len(d) != 1607 or d.key.nunique() != 1607:
        raise RuntimeError("canonical row mismatch")
    for col in ("J0", "J2", "N1M"):
        d[col] = d.key.map(paper[col])
    d["phase"] = np.where(d.month < "2018-03", "warmup", np.where(d.month < "2018-09", "train_forward_oof", np.where(d.month < "2018-11", "development", "later")))

    fin = np.load(INTEGRATED); keys = fin["keys"]
    modern = np.load(MODERN)
    if not np.array_equal(keys, modern["keys"]):
        raise RuntimeError("FinBERT/FinModern article keys differ")
    lookup = {k: i for i, k in enumerate(keys)}
    article_ids = [[lookup[k] for k in str(x).split("|") if k] for x in d.news_record_keys.fillna("")]
    modern_embeddings = modern["embeddings"].astype(float)
    modern_means = np.asarray([modern_embeddings[ii].mean(0) if ii else np.zeros(modern_embeddings.shape[1]) for ii in article_ids])

    methods = ("PRICE_R1", "TFIDF_SVM", "FINMODERN_LR")
    cv, selections, evidence = [], [], []
    start = time.monotonic()
    with threadpool_limits(2):
        for symbol in ("AAPL", "AMZN"):
            for month in MONTHS + ["final"]:
                boundary = "2018-09" if month == "final" else month
                tr = np.flatnonzero((d.symbol == symbol) & (d.month < boundary))
                ev = np.flatnonzero((d.symbol == symbol) & ((d.month >= "2018-09") if month == "final" else (d.month == month)))
                if d.iloc[tr].end_utc.max() >= d.iloc[ev].cutoff_utc.min():
                    raise RuntimeError("chronology violation")
                for method in methods:
                    chosen = choose_c(cv, symbol, method)
                    candidates = (chosen,) if month == "final" else CS
                    for c in candidates:
                        tick = time.monotonic()
                        transform, model, raw = fit_one(method, c, d, tr, ev, modern_embeddings, article_ids, modern_means)
                        path = PRIVATE / "models" / f"{symbol}_{month}_{method}_{c}.joblib"
                        joblib.dump({"transform": transform, "model": model, "train_keys": d.iloc[tr].key.tolist()}, path, compress=3)
                        loaded = joblib.load(path)
                        again = replay(loaded, method, d, ev, modern_means)
                        error = float(np.max(np.abs(raw - again)))
                        if error > 1e-12:
                            raise RuntimeError(f"reload mismatch {error}")
                        issued = raw.copy()
                        if method == "TFIDF_SVM":
                            no = d.iloc[ev].has_original_news.to_numpy() == 0
                            issued[no] = d.iloc[ev].PRICE_R1.to_numpy()[no] if "PRICE_R1" in d else np.nan
                            if np.isnan(issued).any():
                                # PRICE_R1 is fitted first for the same fold.
                                raise RuntimeError("missing price fallback")
                        score = metric(d.iloc[ev].label, raw)
                        if month != "final":
                            cv.append({"symbol": symbol, "month": month, "method": method, "C": c, **score})
                        if c == chosen:
                            d.loc[ev, method] = issued
                        evidence.append({"symbol": symbol, "month": month, "method": method, "C": c,
                                         "train_n": len(tr), "eval_n": len(ev), "train_end": str(d.iloc[tr].end_utc.max()),
                                         "eval_cutoff_min": str(d.iloc[ev].cutoff_utc.min()), "reload_error": error,
                                         "model_sha256": sha(path), "seconds": time.monotonic() - tick})
                    selections.append({"symbol": symbol, "month": month, "method": method, "C": chosen,
                                       "selection_months": sorted({x["month"] for x in cv if x["symbol"] == symbol and x["method"] == method and x["month"] < boundary})})
                print(symbol, month, "complete", round(time.monotonic() - start, 1), flush=True)

    d["FULL_LR"] = d["J0"]
    d["FINBERT_LR"] = d["J2"]
    d["EVENT_META"] = d["N1M"]
    methods_all = ["PRICE_R1", "FULL_LR", "TFIDF_SVM", "FINBERT_LR", "FINMODERN_LR", "EVENT_META"]
    rows, monthly = [], []
    for (phase, symbol), g in d[d.phase != "warmup"].groupby(["phase", "symbol"]):
        for method in methods_all:
            rows.append({"phase": phase, "symbol": symbol, "method": method, **metric(g.label, g[method])})
            for month, q in g.groupby("month"):
                monthly.append({"phase": phase, "symbol": symbol, "month": month, "method": method, **metric(q.label, q[method])})
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_metrics.csv", index=False)
    pd.DataFrame(cv).to_csv(OUT / "cv_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
    pred_cols = ["key", "symbol", "day", "month", "phase", "label", "has_original_news"] + methods_all
    d[pred_cols].to_csv(OUT / "predictions.csv", index=False, float_format="%.17g")
    manifest = {
        "status": "COMPLETE_PENDING_VERIFICATION",
        "rows": 1607,
        "evaluated_rows": int((d.phase != "warmup").sum()),
        "new_model_fits": len(evidence),
        "paper_method_fits_reused_from_fresh_v3_run": 266,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (PAPER_PRIVATE / "inputs.pkl", PAPER_PUBLIC / "predictions.csv", INTEGRATED, MODERN, HERE / "PRE_REGISTRATION.md")},
        "evidence": evidence,
        "max_reload_error": max(x["reload_error"] for x in evidence),
        "seconds": time.monotonic() - start,
        "development_and_later_exposed": True,
        "fnspid_used": False,
    }
    dump(OUT / "training_evidence.json", manifest)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

