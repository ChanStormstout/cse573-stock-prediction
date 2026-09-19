"""Independent Stage A2 verifier for frozen V9 NEWS-only evidence.

This file deliberately does not import the V9 runner or ``outputs/.../run.py``.
It reconstructs its inputs, selection, preprocessing and estimators from the
registered protocol, then compares only the frozen March--August candidate grid
and the issued monthly V9 predictions.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import sys
import zipfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS, CountVectorizer,
                                             TfidfVectorizer)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/stock-data"
BASE = ROOT / "outputs/stock_priorwork_repro"
V9 = BASE / "v9"
OUT = BASE / "v10"
PRIVATE = WORK / "priorwork_v10/models/news_only"
V9_COMMIT = "b7ee2af9829143707ec309e0ec6809935524324b"
SEED = 573
METHODS = [
    "PAPER_1G_L1LR", "PAPER_1G_LINSVM", "PAPER_2G_L1LR", "TFIDF_LR",
    "TFIDF_LINSVM", "TFIDF_RF", "TFIDF_ADABOOST", "TFIDF_KNN",
]
V9_FILES = [
    "GRID_OOF_4H.csv", "GRID_OOF_1D.csv", "METRICS_4H.csv",
    "METRICS_1D.csv", "PREDICTIONS_4H.csv", "PREDICTIONS_1D.csv",
    "MODEL_MANIFEST.json", "PREREGISTRATION_SNAPSHOT.md", "VERIFICATION.json",
]
METRIC_COLUMNS = [
    "n", "accuracy", "ba", "mcc", "precision", "recall", "f1",
    "up_recall", "down_recall", "pred_up", "true_up", "constant", "auc", "brier",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True) + "\n")


def pjson(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def phase(ts: pd.Timestamp) -> str:
    if ts < pd.Timestamp("2018-03-01", tz="UTC"):
        return "warmup"
    if ts < pd.Timestamp("2018-09-01", tz="UTC"):
        return "oof"
    if ts < pd.Timestamp("2018-11-01", tz="UTC"):
        return "development"
    return "later"


def method_spec(name: str) -> tuple[str, tuple[int, int], str]:
    if name.startswith("PAPER_1G"):
        return "count", (1, 1), "lr" if name.endswith("L1LR") else "svm"
    if name == "PAPER_2G_L1LR":
        return "count", (2, 2), "lr"
    tail = name.split("_", 1)[1].lower()
    return "tfidf", (1, 1), {
        "lr": "lr", "linsvm": "svm", "rf": "rf", "adaboost": "adaboost", "knn": "knn",
    }[tail]


def candidates(name: str) -> list[dict]:
    clf = method_spec(name)[2]
    if clf in ("lr", "svm"):
        return [{"C": x} for x in (0.01, 0.1, 1.0)]
    if clf == "rf":
        return [{"max_depth": d, "min_samples_leaf": leaf} for d in (8, None) for leaf in (1, 5)]
    if clf == "adaboost":
        return [{"learning_rate": rate, "n_estimators": n} for n in (50, 100) for rate in (0.05, 0.1)]
    if clf == "knn":
        return [{"n_neighbors": n, "weights": weight} for n in (5, 15, 31) for weight in ("uniform", "distance")]
    raise AssertionError(clf)


def march_default(name: str) -> dict:
    clf = method_spec(name)[2]
    if clf in ("lr", "svm"):
        return {"C": 0.1}
    if clf == "rf":
        return {"max_depth": 8, "min_samples_leaf": 1}
    if clf == "adaboost":
        return {"learning_rate": 0.05, "n_estimators": 50}
    return {"n_neighbors": 5, "weights": "uniform"}


def capacity_key(name: str, params: dict) -> tuple:
    """Registered lower-capacity preference, followed by canonical JSON."""
    clf = method_spec(name)[2]
    if clf in ("lr", "svm"):
        x = (0.01, 0.1, 1.0).index(float(params["C"]))
        return (x, pjson(params))
    if clf == "rf":
        return ((0 if params["max_depth"] == 8 else 1), (0 if params["min_samples_leaf"] == 5 else 1), pjson(params))
    if clf == "adaboost":
        return ((0 if params["n_estimators"] == 50 else 1), (0 if float(params["learning_rate"]) == 0.05 else 1), pjson(params))
    return ((31, 15, 5).index(int(params["n_neighbors"])), (0 if params["weights"] == "uniform" else 1), pjson(params))


def choose_independently(selection_input: pd.DataFrame, name: str) -> tuple[dict, list[dict]]:
    """Implement V10 registered chronological selection without V9 helpers."""
    if not selection_input.empty:
        assert selection_input["month"].max() <= "2018-08", "quarantined candidate reached selection"
    if selection_input.empty:
        return march_default(name), []
    rows = []
    decision = method_spec(name)[2] == "svm"
    for params, x in selection_input.groupby("params", sort=False):
        params_obj = json.loads(params)
        ba = float(x["ba"].mean())
        brier = None if decision else float(x["brier"].mean())
        rows.append({"params": params_obj, "ba": ba, "brier": brier, "capacity": capacity_key(name, params_obj)})
    # BA desc, probabilistic Brier asc, capacity, canonical JSON. Exact equality
    # is intentional; no tolerance-based pseudo ties are invented here.
    rows.sort(key=lambda x: (-x["ba"], 0 if decision else x["brier"], x["capacity"]))
    return rows[0]["params"], rows


def metrics(y: pd.Series, score: np.ndarray, decision_only: bool) -> dict:
    yv = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= 0 if decision_only else score >= 0.5).astype(int)
    both = len(np.unique(yv)) == 2
    out = {
        "n": int(len(yv)), "accuracy": float(accuracy_score(yv, pred)),
        "ba": float(balanced_accuracy_score(yv, pred)) if both else None,
        "mcc": float(matthews_corrcoef(yv, pred)) if both else None,
        "precision": float(precision_score(yv, pred, zero_division=0)),
        "recall": float(recall_score(yv, pred, zero_division=0)),
        "f1": float(f1_score(yv, pred, zero_division=0)),
        "up_recall": float(recall_score(yv, pred, pos_label=1, zero_division=0)),
        "down_recall": float(recall_score(yv, pred, pos_label=0, zero_division=0)),
        "pred_up": float(pred.mean()), "true_up": float(yv.mean()),
        "constant": bool(pred.min() == pred.max()),
        "auc": float(roc_auc_score(yv, score)) if both else None,
        "brier": None if decision_only else float(brier_score_loss(yv, score)),
    }
    return out


def make_estimator(name: str, params: dict):
    clf = method_spec(name)[2]
    if clf == "lr":
        return LogisticRegression(
            penalty="l1" if name.startswith("PAPER") else "l2", solver="liblinear",
            max_iter=3000, random_state=SEED, **params,
        )
    if clf == "svm":
        return LinearSVC(random_state=SEED, **params)
    if clf == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1, **params)
    if clf == "adaboost":
        return AdaBoostClassifier(random_state=SEED, **params)
    return KNeighborsClassifier(metric="cosine", **params)


def fit_independent(name: str, params: dict, train: pd.DataFrame, ev: pd.DataFrame):
    typ, ngrams, clf = method_spec(name)
    tr_text = train.stem_body.fillna("").astype(str)
    ev_text = ev.stem_body.fillna("").astype(str)
    vectorizer = (CountVectorizer(binary=True, ngram_range=ngrams, min_df=3)
                  if typ == "count" else
                  TfidfVectorizer(ngram_range=ngrams, min_df=3, max_features=10000, sublinear_tf=True))
    x_train = vectorizer.fit_transform(tr_text)
    x_ev = vectorizer.transform(ev_text)
    k = min(500, x_train.shape[1])
    selector = None
    if k:
        selector = SelectKBest(chi2, k=k).fit(x_train, train.label)
        x_train = selector.transform(x_train)
        x_ev = selector.transform(x_ev)
    model = make_estimator(name, params)
    model.fit(x_train, train.label)
    decision_only = clf == "svm"
    raw = model.decision_function(x_ev) if decision_only else model.predict_proba(x_ev)[:, 1]
    raw = np.asarray(raw, dtype=float)
    prior = float(train.label.mean())
    raw[ev.has_news.to_numpy(dtype=int) == 0] = prior - 0.5 if decision_only else prior
    bundle = {
        "vectorizer": vectorizer, "selector": selector, "classifier": model,
        "decision_only": decision_only, "training_class_prior": prior,
        "method": name, "issued_params": params,
    }
    return raw, decision_only, bundle


def predict_bundle(bundle: dict, ev: pd.DataFrame) -> np.ndarray:
    x = bundle["vectorizer"].transform(ev.stem_body.fillna("").astype(str))
    if bundle["selector"] is not None:
        x = bundle["selector"].transform(x)
    if bundle["decision_only"]:
        score = bundle["classifier"].decision_function(x)
    else:
        score = bundle["classifier"].predict_proba(x)[:, 1]
    score = np.asarray(score, dtype=float)
    score[ev.has_news.to_numpy(dtype=int) == 0] = (bundle["training_class_prior"] - 0.5
                                                    if bundle["decision_only"] else bundle["training_class_prior"])
    return score


def _clean_body_features(keys: list[str], raw: pd.DataFrame) -> dict[str, set[str]]:
    stemmer = EnglishStemmer()
    bad = re.compile(r"newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here", re.I)

    @lru_cache(maxsize=200000)
    def stem(token: str) -> str:
        return stemmer.stem(token)

    def tokens(text: str) -> set[str]:
        return {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS}

    out: dict[str, set[str]] = {}
    source = raw.loc[keys]
    for archive, rows in source.groupby("archive"):
        with zipfile.ZipFile(WORK / "raw/news" / archive) as z:
            for row in rows.itertuples():
                body = json.loads(z.read(row.member)).get("text", "")
                body = html.unescape(re.sub(r"<[^>]+>", " ", body))
                body = " ".join(line for line in body.splitlines() if not bad.search(line))
                body = re.sub(r"https?://\S+", " ", body)
                out[row.Index] = tokens(body) | tokens(str(row.title))
    return out


def canonical_candidates() -> dict[str, pd.DataFrame]:
    # The canonical V8 lineage uses this retained target-title association.
    sys.path.insert(0, str(ROOT / "outputs/stock_baseline"))
    from run_baseline import build  # canonical association source, not V9 code
    result = {}
    for symbol, prefix in (("AAPL", "APPLE"), ("AMZN", "AMAZON")):
        _, _, _, news = build(WORK, symbol, prefix)
        news = news.copy()
        news["key"] = news.archive + "::" + news.member
        result[symbol] = news.set_index("key", drop=False)
    return result


def reconstruct_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build 4h/daily inputs from canonical raw sources and prove cache parity."""
    four = pd.read_pickle(WORK / "nextgen_4h/price_v1/features.pkl").copy()
    four["news_window"] = "4H"
    four["split"] = four.start_utc.map(phase)
    raw = pd.read_pickle(WORK / "audit/news_index.pkl").copy()
    raw["key"] = raw.archive + "::" + raw.member
    raw = raw.set_index("key", drop=False)
    candidates_by_stock = canonical_candidates()

    keys4 = sorted({key for cells in four.news_record_keys.fillna("") for key in str(cells).split("|") if key})
    feats = _clean_body_features(keys4, raw)
    article_ok = 0
    body_ok = 0
    cutoff_ok = 0
    for row in four.itertuples():
        cand = candidates_by_stock[row.symbol]
        got = cand[(cand.available_utc > row.cutoff_utc - pd.Timedelta(hours=4)) &
                   (cand.available_utc <= row.cutoff_utc)].sort_values(["available_utc", "archive", "member"])
        got_keys = "|".join(got.index)
        want = str(row.news_record_keys or "")
        article_ok += got_keys == want
        stem = " ".join(sorted(set().union(*(feats[k] for k in want.split("|") if k)))) if want else ""
        body_ok += stem == str(row.stem_body or "")
        cutoff_ok += all(raw.loc[k, "available_utc"] <= row.cutoff_utc for k in want.split("|") if k)
    assert len(four) == 1607

    # Daily raw reconstruction mirrors the frozen V8 full-body transform; V8's
    # private cache is read only afterwards to prove source parity.
    daily_rows = []
    for symbol, prefix in (("AAPL", "APPLE"), ("AMZN", "AMAZON")):
        bars = pd.read_csv(WORK / f"raw/CHARTS/{prefix}1440.csv", header=None,
                           names=["date", "time", "open", "high", "low", "close", "activity"])
        bars["day"] = pd.to_datetime(bars.date, format="%Y.%m.%d").dt.strftime("%Y-%m-%d")
        bars = bars.set_index("day").sort_index()
        cal = pd.read_csv(WORK / "audit/xnys_schedule.csv", index_col=0)
        cal.index = pd.to_datetime(cal.index).strftime("%Y-%m-%d")
        cal[["open", "close"]] = cal[["open", "close"]].apply(pd.to_datetime, utc=True)
        dates = [d for d in cal.index if d in bars.index]
        article_table = candidates_by_stock[symbol]
        missing = [k for k in article_table.index if k not in feats]
        feats.update(_clean_body_features(missing, raw))
        for i, day in enumerate(dates):
            if i < 5:
                continue
            opening, close_time = cal.loc[day, ["open", "close"]]
            cutoff = opening - pd.Timedelta(minutes=5)
            history = bars.loc[dates[i - 5:i - 1]]
            r = np.log(history.close / history.open).to_numpy()
            base = {
                "symbol": symbol, "day": day, "start_utc": opening, "end_utc": close_time,
                "cutoff_utc": cutoff, "label": int(bars.loc[day, "close"] > bars.loc[day, "open"]),
                "split": phase(opening), "return_1": float(r[-1]), "return_2": float(r[-2:].sum()),
                "return_5": float(r.sum()),
                "range_1": float((history.high.iloc[-1] - history.low.iloc[-1]) / history.open.iloc[-1]),
                "return_mean": float(r.mean()), "return_std": float(np.sqrt((r * r).sum())),
                "history_age_hours": float((cutoff - cal.loc[dates[i - 1], "close"]).total_seconds() / 3600),
            }
            for window, left in (("DNEWS_OVERNIGHT", cal.loc[dates[i - 1], "close"]),
                                 ("DNEWS_24H", cutoff - pd.Timedelta(hours=24))):
                articles = article_table[(article_table.available_utc > left) &
                                         (article_table.available_utc <= cutoff)].sort_values(["available_utc", "archive", "member"])
                keys = articles.index.tolist()
                daily_rows.append({**base, "news_window": window,
                    "news_record_keys": "|".join(keys),
                    "stem_body": " ".join(sorted(set().union(*(feats[k] for k in keys)))) if keys else "",
                    "has_news": int(bool(keys)), "news_count": len(keys)})
    daily = pd.DataFrame(daily_rows)
    cache = pd.read_pickle(BASE / "v8/private_daily.pkl")
    daily_compare_columns = ["symbol", "day", "start_utc", "end_utc", "cutoff_utc", "label", "news_window", "news_record_keys", "stem_body", "has_news", "news_count"]
    left = daily.sort_values(["symbol", "news_window", "start_utc"])[daily_compare_columns].reset_index(drop=True)
    right = cache.sort_values(["symbol", "news_window", "start_utc"])[daily_compare_columns].reset_index(drop=True)
    daily_parity = bool(left.equals(right))
    audit = {
        "canonical_4h_rows": len(four), "article_key_parity": f"{article_ok}/{len(four)}",
        "stem_body_parity": f"{body_ok}/{len(four)}", "available_by_cutoff": f"{cutoff_ok}/{len(four)}",
        "daily_raw_rows": len(daily), "daily_cache_rows": len(cache),
        "daily_source_cache_parity": daily_parity,
    }
    return four, daily, audit


def slice_data(data: pd.DataFrame, stock: str, window: str, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = data[(data.symbol == stock) & (data.news_window == window)].sort_values("cutoff_utc")
    ev = group[group.start_utc.dt.strftime("%Y-%m") == month].copy()
    train = group[group.end_utc < ev.cutoff_utc.min()].copy()
    return train, ev


def source_binding() -> dict:
    current = {name: sha(V9 / name) for name in V9_FILES}
    commit = {}
    for name in V9_FILES:
        blob = subprocess.run(["git", "show", f"{V9_COMMIT}:outputs/stock_priorwork_repro/v9/{name}"],
                              cwd=ROOT, check=True, capture_output=True).stdout
        commit[name] = hashlib.sha256(blob).hexdigest()
    diff = subprocess.run(["git", "diff", "--exit-code", V9_COMMIT, "--", "outputs/stock_priorwork_repro/v9/"],
                          cwd=ROOT, capture_output=True, text=True)
    return {"v9_source_commit": V9_COMMIT, "git_path_identical": diff.returncode == 0,
            "git_diff_stdout": diff.stdout, "current_sha256": current,
            "commit_sha256": commit, "hash_identical": current == commit}


def selection_reconstruction(grid4: pd.DataFrame, grid1: pd.DataFrame, met4: pd.DataFrame, met1: pd.DataFrame):
    issued_records = []
    violations = []
    tie_records = []
    for horizon, grid, metrics_frame in (("4h", grid4, met4), ("1d", grid1, met1)):
        authorized = grid[grid.month <= "2018-08"].copy()
        assert authorized.month.max() <= "2018-08"
        for row in metrics_frame.itertuples(index=False):
            windows = ["4H"] if horizon == "4h" else [str(row.horizon).split(":", 1)[1]]
            window = windows[0]
            month = row.month
            if month == "2018-03":
                params = march_default(row.method)
                months_used = []
            elif month <= "2018-08":
                months_used = pd.period_range("2018-03", str(pd.Period(month) - 1), freq="M").astype(str).tolist()
                select_input = authorized[(authorized.stock == row.stock) & (authorized.horizon == row.horizon) &
                                          (authorized.method == row.method) & (authorized.month.isin(months_used))].copy()
                params, ranking = choose_independently(select_input, row.method)
                if len(ranking) > 1 and ranking[0]["ba"] == ranking[1]["ba"] and ranking[0]["brier"] == ranking[1]["brier"]:
                    tie_records.append({"horizon": horizon, "stock": row.stock, "method": row.method, "month": month, "ranking": ranking})
            else:
                months_used = pd.period_range("2018-03", "2018-08", freq="M").astype(str).tolist()
                select_input = authorized[(authorized.stock == row.stock) & (authorized.horizon == row.horizon) &
                                          (authorized.method == row.method) & (authorized.month.isin(months_used))].copy()
                params, _ = choose_independently(select_input, row.method)
            expected = pjson(params)
            actual = pjson(row.params)
            train_count = int(len(select_input)) if month != "2018-03" else 0
            issued_records.append({"stock": row.stock, "horizon": row.horizon, "method": row.method,
                "prediction_month": month, "issued_parameter_json": expected,
                "selection_months_used": json.dumps(months_used), "authorized_candidate_rows_used": train_count,
                "is_march_default": month == "2018-03", "is_sep_plus_frozen": month >= "2018-09",
                "v9_recorded_parameter_json": actual, "match": expected == actual})
    issued = pd.DataFrame(issued_records)
    for keys, group in issued[issued.is_sep_plus_frozen].groupby(["stock", "horizon", "method"]):
        if group.issued_parameter_json.nunique() != 1:
            violations.append({"group": keys, "parameters": group.issued_parameter_json.tolist()})
    four = issued[issued.horizon.eq("4h")].copy()
    daily = issued[issued.horizon.ne("4h")].copy()
    audit = {"4h_expected": 192, "1d_expected": 384, "4h_actual": len(four), "1d_actual": len(daily),
             "4h_mismatches": int((~four.match).sum()), "1d_mismatches": int((~daily.match).sum()),
             "sep_plus_freeze_violations": violations, "machine_precision_ties": tie_records,
             "status": "PASS" if four.match.all() and daily.match.all() and not violations and len(four) == 192 and len(daily) == 384 else "FAIL"}
    return four, daily, audit


def metric_errors(calculated: dict, frozen: pd.Series) -> dict:
    out = {}
    for key in METRIC_COLUMNS:
        a, b = calculated[key], frozen[key]
        if pd.isna(a) and pd.isna(b):
            out[key] = 0.0
        elif isinstance(a, (bool, np.bool_)):
            out[key] = 0.0 if bool(a) == bool(b) else 1.0
        else:
            out[key] = abs(float(a) - float(b))
    return out


def replay_grid(data: pd.DataFrame, grid: pd.DataFrame, horizon: str):
    authorized = grid[grid.month <= "2018-08"].copy()
    assert authorized.month.max() <= "2018-08"
    rows = []
    max_err = {key: 0.0 for key in METRIC_COLUMNS}
    for frozen in authorized.itertuples(index=False):
        window = "4H" if horizon == "4h" else frozen.horizon.split(":", 1)[1]
        train, ev = slice_data(data, frozen.stock, window, frozen.month)
        score, decision, _ = fit_independent(frozen.method, json.loads(frozen.params), train, ev)
        result = metrics(ev.label, score, decision)
        errors = metric_errors(result, pd.Series(frozen._asdict()))
        for key, value in errors.items():
            max_err[key] = max(max_err[key], float(value))
        rows.append({"stock": frozen.stock, "horizon": frozen.horizon, "method": frozen.method,
                     "month": frozen.month, "params": pjson(frozen.params), **result,
                     **{f"error_{k}": v for k, v in errors.items()}})
    replay = pd.DataFrame(rows)
    return replay, {"rows": len(replay), "max_metric_discrepancy": max_err,
                    "pass": len(replay) == (348 if horizon == "4h" else 696) and max(max_err.values()) <= 1e-10}


def issued_refit(data: pd.DataFrame, issued: pd.DataFrame, frozen_predictions: pd.DataFrame, horizon: str):
    public_manifest = []
    errors = {"probability": 0.0, "svm_score": 0.0, "direction_mismatches": 0, "model_count": 0}
    bundle_records = []
    for rec in issued.itertuples(index=False):
        window = "4H" if horizon == "4h" else rec.horizon.split(":", 1)[1]
        train, ev = slice_data(data, rec.stock, window, rec.prediction_month)
        score, decision, bundle = fit_independent(rec.method, json.loads(rec.issued_parameter_json), train, ev)
        expected = frozen_predictions[(frozen_predictions.symbol == rec.stock) &
                                      (frozen_predictions.horizon == rec.horizon) &
                                      (frozen_predictions.method == rec.method) &
                                      (frozen_predictions.month == rec.prediction_month)].copy()
        expected = expected.sort_values(["start_utc", "cutoff_utc"]).reset_index(drop=True)
        got = ev.sort_values(["start_utc", "cutoff_utc"]).reset_index(drop=True)
        assert len(expected) == len(got)
        assert expected[["symbol", "start_utc", "cutoff_utc", "label"]].equals(got[["symbol", "start_utc", "cutoff_utc", "label"]])
        err = float(np.max(np.abs(score - expected.p.to_numpy(dtype=float)))) if len(score) else 0.0
        if decision:
            errors["svm_score"] = max(errors["svm_score"], err)
        else:
            errors["probability"] = max(errors["probability"], err)
        pred = score >= (0 if decision else 0.5)
        frozen_pred = expected.p.to_numpy(dtype=float) >= (0 if decision else 0.5)
        mismatch = int(np.sum(pred != frozen_pred))
        errors["direction_mismatches"] += mismatch
        bundle.update({"stock": rec.stock, "horizon": rec.horizon, "prediction_month": rec.prediction_month,
                       "training_row_identifiers": [f"{x.symbol}|{x.start_utc}" for x in train.itertuples()],
                       "train_n": len(train), "max_train_outcome_time": str(train.end_utc.max()),
                       "prediction_n": len(ev), "v9_reproduction_error": err})
        filename = f"{horizon}_{window}_{rec.stock}_{rec.method}_{rec.prediction_month}.joblib"
        target = PRIVATE / filename
        joblib.dump(bundle, target)
        reloaded = joblib.load(target)
        reload_score = predict_bundle(reloaded, got)
        reload_error = float(np.max(np.abs(reload_score - score))) if len(score) else 0.0
        reload_mismatch = int(np.sum((reload_score >= (0 if decision else 0.5)) != pred))
        bundle["serialized_reload_error"] = reload_error
        bundle["serialized_reload_direction_mismatches"] = reload_mismatch
        # Rewrite the bundle with its complete metadata.
        joblib.dump(bundle, target)
        digest = sha(target)
        public_manifest.append({"logical_model_id": target.stem, "sha256": digest, "stock": rec.stock,
            "horizon": rec.horizon, "method": rec.method, "month": rec.prediction_month,
            "issued_params": json.loads(rec.issued_parameter_json), "train_n": len(train),
            "train_end": str(train.end_utc.max()), "prediction_n": len(ev),
            "v9_reproduction_error": err, "serialized_reload_error": reload_error,
            "direction_mismatch_count": mismatch, "serialized_reload_direction_mismatch_count": reload_mismatch})
        bundle_records.append((target, got, score, decision))
        errors["model_count"] += 1
        errors.setdefault("reload_error", 0.0)
        errors["reload_error"] = max(errors["reload_error"], reload_error)
        errors.setdefault("reload_direction_mismatches", 0)
        errors["reload_direction_mismatches"] += reload_mismatch
    return public_manifest, errors, bundle_records


def perturbation(data: pd.DataFrame, issued: pd.DataFrame) -> dict:
    results = []
    for name in ("PAPER_1G_L1LR", "TFIDF_LR"):
        rec = issued[(issued.method == name) & (issued.prediction_month == "2018-03")].iloc[0]
        train, ev = slice_data(data, rec.stock, "4H", rec.prediction_month)
        normal_score, _, normal = fit_independent(name, json.loads(rec.issued_parameter_json), train, ev)
        changed = data.copy()
        mask = changed.end_utc >= ev.cutoff_utc.min()
        changed.loc[mask, "stem_body"] = changed.loc[mask, "stem_body"].fillna("") + " FUTUREMUTATIONTOKEN_938475"
        changed_train, changed_ev = slice_data(changed, rec.stock, "4H", rec.prediction_month)
        altered_score, _, altered = fit_independent(name, json.loads(rec.issued_parameter_json), changed_train, changed_ev)
        vocab_equal = normal["vectorizer"].vocabulary_ == altered["vectorizer"].vocabulary_
        idf_equal = True
        if hasattr(normal["vectorizer"], "idf_"):
            idf_equal = bool(np.array_equal(normal["vectorizer"].idf_, altered["vectorizer"].idf_))
        selected_equal = ((normal["selector"] is None and altered["selector"] is None) or
                          np.array_equal(normal["selector"].get_support(indices=True), altered["selector"].get_support(indices=True)))
        coef_normal = normal["classifier"].coef_ if hasattr(normal["classifier"], "coef_") else None
        coef_altered = altered["classifier"].coef_ if hasattr(altered["classifier"], "coef_") else None
        coef_equal = bool(np.array_equal(coef_normal, coef_altered))
        same_train = [f"{x.symbol}|{x.start_utc}" for x in train.itertuples()] == [f"{x.symbol}|{x.start_utc}" for x in changed_train.itertuples()]
        results.append({"method": name, "stock": rec.stock, "month": rec.prediction_month,
                        "future_rows_mutated": int(mask.sum()), "training_rows_unchanged": same_train,
                        "vocabulary_unchanged": vocab_equal, "idf_unchanged": idf_equal,
                        "chi2_selected_terms_unchanged": selected_equal, "classifier_state_unchanged": coef_equal,
                        "prediction_max_error": float(np.max(np.abs(normal_score - altered_score))),
                        "pass": same_train and vocab_equal and idf_equal and selected_equal and coef_equal and np.array_equal(normal_score, altered_score)})
    return {"tests": results, "status": "PASS" if all(x["pass"] for x in results) else "FAIL"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    binding = source_binding()
    dump(OUT / "V9_SOURCE_BINDING.json", binding)
    grid4 = pd.read_csv(V9 / "GRID_OOF_4H.csv")
    grid1 = pd.read_csv(V9 / "GRID_OOF_1D.csv")
    met4 = pd.read_csv(V9 / "METRICS_4H.csv")
    met1 = pd.read_csv(V9 / "METRICS_1D.csv")
    pred4 = pd.read_csv(V9 / "PREDICTIONS_4H.csv", parse_dates=["start_utc", "cutoff_utc"])
    pred1 = pd.read_csv(V9 / "PREDICTIONS_1D.csv", parse_dates=["start_utc", "cutoff_utc"])
    bound = {"authorized_4h_rows": int((grid4.month <= "2018-08").sum()),
             "authorized_1d_rows": int((grid1.month <= "2018-08").sum()),
             "quarantined_4h_rows": int((grid4.month >= "2018-09").sum()),
             "quarantined_1d_rows": int((grid1.month >= "2018-09").sum())}
    four, daily, source_audit = reconstruct_inputs()
    four_issued, daily_issued, selection_audit = selection_reconstruction(grid4, grid1, met4, met1)
    four_issued.to_csv(OUT / "ISSUED_PARAMS_4H.csv", index=False)
    daily_issued.to_csv(OUT / "ISSUED_PARAMS_1D.csv", index=False)
    dump(OUT / "SELECTION_RECONSTRUCTION_AUDIT.json", selection_audit)
    if selection_audit["status"] != "PASS":
        final = {"status": "FAIL", "failure": "selection_reconstruction", "v9_source_git_binding": binding["git_path_identical"],
                 "v9_source_hash_binding": binding["hash_identical"], "authorized_grid_counts": bound,
                 "issued_selection_reconstruction": selection_audit, "canonical_4h_row_contract": source_audit}
        dump(OUT / "STAGE_A_FINAL_AUDIT.json", final)
        print("V10_V9_SELECTION_RECONSTRUCTION_FAILED")
        return
    replay4, replay4_audit = replay_grid(four, grid4, "4h")
    replay1, replay1_audit = replay_grid(daily, grid1, "1d")
    replay4.to_csv(OUT / "V9_GRID_REPLAY_4H.csv", index=False)
    replay1.to_csv(OUT / "V9_GRID_REPLAY_1D.csv", index=False)
    grid_audit = {"4h": replay4_audit, "1d": replay1_audit,
                  "status": "PASS" if replay4_audit["pass"] and replay1_audit["pass"] else "FAIL"}
    dump(OUT / "V9_GRID_REPLAY_AUDIT.json", grid_audit)
    manifests4, refit4, _ = issued_refit(four, four_issued, pred4, "4h")
    manifests1, refit1, _ = issued_refit(daily, daily_issued, pred1, "1d")
    manifest = manifests4 + manifests1
    dump(OUT / "NEWS_ONLY_MODEL_MANIFEST.json", manifest)
    refit_audit = {"4h": refit4, "1d": refit1,
                   "status": "PASS" if refit4["probability"] <= 1e-10 and refit1["probability"] <= 1e-10 and
                   refit4["svm_score"] <= 1e-10 and refit1["svm_score"] <= 1e-10 and
                   refit4["direction_mismatches"] == 0 and refit1["direction_mismatches"] == 0 else "FAIL"}
    dump(OUT / "V9_ISSUED_PREDICTION_REPLAY_AUDIT.json", refit_audit)
    pert = perturbation(four, four_issued)
    dump(OUT / "PREPROCESSING_PERTURBATION_TEST.json", pert)
    boundary_violations = []
    for issued, data in ((four_issued, four), (daily_issued, daily)):
        for rec in issued.itertuples(index=False):
            window = "4H" if rec.horizon == "4h" else rec.horizon.split(":", 1)[1]
            tr, ev = slice_data(data, rec.stock, window, rec.prediction_month)
            if not (tr.end_utc.max() < ev.cutoff_utc.min()):
                boundary_violations.append({"stock": rec.stock, "horizon": rec.horizon, "method": rec.method, "month": rec.prediction_month})
    passes = {
        "v9_source_git_binding": binding["git_path_identical"], "v9_source_hash_binding": binding["hash_identical"],
        "canonical_4h_row_contract": source_audit["canonical_4h_rows"] == 1607,
        "article_key_parity": source_audit["article_key_parity"] == "1607/1607",
        "stem_body_parity": source_audit["stem_body_parity"] == "1607/1607",
        "daily_source_reconstruction": source_audit["daily_source_cache_parity"],
        "authorized_grid_counts": bound["authorized_4h_rows"] == 348 and bound["authorized_1d_rows"] == 696,
        "quarantined_grid_counts": bound["quarantined_4h_rows"] == 348 and bound["quarantined_1d_rows"] == 696,
        "quarantine_selection_isolation": True,
        "candidate_grid_completeness": replay4_audit["rows"] == 348 and replay1_audit["rows"] == 696,
        "issued_selection_reconstruction": selection_audit["status"] == "PASS",
        "sep_plus_parameter_freeze": not selection_audit["sep_plus_freeze_violations"],
        "authorized_grid_independent_replay": grid_audit["status"] == "PASS",
        "issued_row_level_prediction_replay": refit_audit["status"] == "PASS",
        "serialized_model_count": len(manifest) == 576,
        "serialized_model_reload": refit4["reload_error"] <= 1e-12 and refit1["reload_error"] <= 1e-12 and refit4["reload_direction_mismatches"] == 0 and refit1["reload_direction_mismatches"] == 0,
        "future_text_perturbation": pert["status"] == "PASS",
        "training_boundary": not boundary_violations,
        "direction_mismatch_count": refit4["direction_mismatches"] == 0 and refit1["direction_mismatches"] == 0,
    }
    final = {"status": "PASS" if all(passes.values()) else "FAIL", "checks": passes,
             "v9_source_git_binding": binding, "canonical_4h_row_contract": source_audit,
             "authorized_grid_counts": bound, "quarantined_grid_counts": {"4h": bound["quarantined_4h_rows"], "1d": bound["quarantined_1d_rows"]},
             "issued_selection_reconstruction": selection_audit, "authorized_grid_independent_replay": grid_audit,
             "issued_row_level_prediction_replay": refit_audit, "serialized_model_count": len(manifest),
             "future_text_perturbation": pert, "training_boundary_violations": boundary_violations,
             "scope": {"dprice_run": False, "news_price_run": False, "method_family_selection_run": False}}
    dump(OUT / "STAGE_A_FINAL_AUDIT.json", final)
    report = "# V10 Stage A2 integrity report\n\n"
    report += f"Status: **{final['status']}**. This report evaluates only reproducibility and evidence integrity; it does not select or interpret a predictive winner.\n\n"
    report += f"- Authorized candidate rows: 4h {bound['authorized_4h_rows']}; 1d {bound['authorized_1d_rows']}.\n"
    report += f"- Quarantined Sep+ candidate rows: 4h {bound['quarantined_4h_rows']}; 1d {bound['quarantined_1d_rows']}.\n"
    report += f"- Issued-parameter mismatches: 4h {selection_audit['4h_mismatches']}; 1d {selection_audit['1d_mismatches']}.\n"
    report += f"- Independent grid replay rows: 4h {replay4_audit['rows']}; 1d {replay1_audit['rows']}.\n"
    report += f"- Reconstructed private issued models: {len(manifest)}; reload maximum errors are at most {max(refit4['reload_error'], refit1['reload_error']):.3g}.\n"
    report += f"- Future-text perturbation: {pert['status']}. Training-boundary violations: {len(boundary_violations)}.\n"
    (OUT / "STAGE_A_REPORT.md").write_text(report)
    print("V10_STAGE_A_COMPLETE_AWAITING_REVIEW" if final["status"] == "PASS" else "V10_V9_EVIDENCE_RECOVERY_FAILED")


if __name__ == "__main__":
    main()
