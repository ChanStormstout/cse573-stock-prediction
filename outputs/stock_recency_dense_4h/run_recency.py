"""Run cutoff-safe recency-weighted R1/F1/F2 comparisons."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    CS, HALF_LIVES, HALF_NAMES, INPUTS, INTEGRATED, MONTHS, OLD, PUBLIC, R1_COLS, RECENT,
    WORK, dump, load_data, metric_rows, schedule, scores, sha, standardize_fit, standardize_transform,
)


def load_embeddings():
    z = np.load(INTEGRATED / "articles.npz")
    keys = [str(x) for x in z["keys"]]
    vectors = z["embeddings"].astype(float)
    return {k: vectors[i] for i, k in enumerate(keys)}, sha(INTEGRATED / "articles.npz")


def row_embedding(row, lookup, dim=768):
    keys = [k for k in str(row.news_record_keys or "").split("|") if k in lookup]
    if not keys:
        return np.zeros(dim, dtype=float)
    return np.mean([lookup[k] for k in keys], axis=0)


def c_schedule():
    """Use the previously registered past-only schedules to isolate weighting."""
    choices = {}
    for symbol in ("AAPL", "AMZN"):
        choices[(symbol, "R1", "final")] = 0.1
        for month in MONTHS:
            choices[(symbol, "R1", month)] = 0.1
    sel_path = PUBLIC.parent / "stock_paper_methods_4h" / "v1" / "selection.json"
    raw = json.loads(sel_path.read_text())
    for row in raw["joint"]:
        if row["method"] == "J0":
            choices[(row["symbol"], "F1", row["month"])] = float(row["C"])
        if row["method"] == "J2":
            choices[(row["symbol"], "F2", row["month"])] = float(row["C"])
    for symbol in ("AAPL", "AMZN"):
        for m in MONTHS:
            choices.setdefault((symbol, "F1", m), 0.1)
            choices.setdefault((symbol, "F2", m), 0.1)
        choices.setdefault((symbol, "F1", "final"), 1.0)
        choices.setdefault((symbol, "F2", "final"), 1.0)
    return choices, sha(sel_path)


def age_weights(train: pd.DataFrame, boundary: pd.Timestamp, half_life, opens: np.ndarray):
    if train.empty:
        return np.array([], dtype=float), np.array([], dtype=int)
    end = train.end_utc.to_numpy(dtype="datetime64[ns]")
    b = np.datetime64(boundary.tz_convert("UTC").tz_localize(None), "ns")
    # A session is counted only when it opens strictly after the matured label and before the fold.
    ages = np.array([int(((opens > e) & (opens < b)).sum()) for e in end], dtype=int)
    if half_life is None:
        w = np.ones(len(train), dtype=float)
    else:
        w = np.power(2.0, -ages / float(half_life))
    return w, ages


def fit_text_transform(train: pd.DataFrame, method: str, lookup: dict[str, np.ndarray]):
    price_mean, price_scale = standardize_fit(train, OLD if method != "R1" else R1_COLS)
    vocab = None
    keep = None
    pca = None
    sem_scale = None
    if method == "F1":
        vocab = CountVectorizer(binary=True, min_df=3)
        matrix = vocab.fit_transform(train.stem_body.fillna("").astype(str))
        if matrix.shape[1]:
            stat = chi2(matrix, train.label.to_numpy())[0]
            names = vocab.get_feature_names_out()
            order = np.lexsort((names, -np.nan_to_num(stat, nan=-np.inf)))
            keep = order[: min(500, len(order))]
        else:
            keep = np.array([], dtype=int)
    if method == "F2":
        vecs = np.vstack([row_embedding(row, lookup) for _, row in train.iterrows()])
        pca = PCA(n_components=min(16, vecs.shape[0], vecs.shape[1]), svd_solver="randomized", random_state=573).fit(vecs)
        sem_scale = StandardScaler().fit(pca.transform(vecs))
    return {
        "price_cols": R1_COLS if method == "R1" else OLD,
        "price_mean": price_mean,
        "price_scale": price_scale,
        "vectorizer": vocab,
        "keep": keep,
        "pca": pca,
        "sem_scale": sem_scale,
    }


def transform(frame: pd.DataFrame, method: str, state, lookup):
    px = standardize_transform(frame, state["price_cols"], state["price_mean"], state["price_scale"])
    if method == "R1":
        return px
    if method == "F1":
        text = state["vectorizer"].transform(frame.stem_body.fillna("").astype(str))[:, state["keep"]]
        return sparse.hstack([sparse.csr_matrix(px), text], format="csr")
    vecs = np.vstack([row_embedding(row, lookup) for _, row in frame.iterrows()])
    sem = state["sem_scale"].transform(state["pca"].transform(vecs))
    return np.c_[px, sem]


def fit_branch(train, evaluate, method, C, half_life, boundary, lookup, opens, model_path: Path, fallback=None):
    if train.empty or evaluate.empty or train.end_utc.max() >= evaluate.cutoff_utc.min():
        raise AssertionError("invalid temporal split")
    state = fit_text_transform(train, method, lookup)
    x_train = transform(train, method, state, lookup)
    x_eval = transform(evaluate, method, state, lookup)
    w, ages = age_weights(train, boundary, half_life, opens)
    model = LogisticRegression(C=C, solver="liblinear", max_iter=3000, tol=1e-8, random_state=573)
    model.fit(x_train, train.label.to_numpy(dtype=int), sample_weight=w)
    p = model.predict_proba(x_eval)[:, 1]
    artifact = {"state": state, "model": model, "method": method, "C": C, "half_life": half_life}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    reloaded = joblib.load(model_path)
    p_reload = reloaded["model"].predict_proba(x_eval)[:, 1]
    np.testing.assert_allclose(p, p_reload, atol=1e-12, rtol=0)
    if method in ("F1", "F2") and fallback is not None:
        # The original paper pipeline returns the price-only fallback for windows with no accepted article.
        no_news = evaluate.has_original_news.to_numpy(dtype=int) == 0
        p[no_news] = fallback[no_news]
    return p, {
        "method": method, "C": C, "half_life": HALF_NAMES[half_life], "train_n": int(len(train)),
        "eval_n": int(len(evaluate)), "weight_sum": float(w.sum()), "weight_min": float(w.min()),
        "weight_max": float(w.max()), "age_min": int(ages.min()) if len(ages) else None,
        "age_max": int(ages.max()) if len(ages) else None, "model_sha256": sha(model_path),
        "reload_max_abs_error": float(np.max(np.abs(p_reload - (p if fallback is None else np.where(evaluate.has_original_news.to_numpy(dtype=int)==0, p_reload, p))))) if len(p) else 0.0,
    }


def boundary_for(month: str, evaluate: pd.DataFrame):
    return pd.Timestamp(evaluate.cutoff_utc.min())


def run(out: Path):
    start = time.monotonic()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    private = WORK / "recency_dense_4h" / out.name
    (private / "models").mkdir(parents=True)
    data = load_data()
    # Exact public reference predictions are used for the infinity branch.  This
    # prevents implementation details of a re-fit from being mistaken for a
    # recency effect and makes parity auditable against the original run.
    reference_path = PUBLIC.parent / "stock_goal60_4h" / "v1" / "predictions.csv"
    reference = pd.read_csv(reference_path).set_index("key")
    if not set(data.key).issubset(reference.index):
        raise AssertionError("official reference prediction keys changed")
    reference_sha = sha(reference_path)
    lookup, embedding_sha = load_embeddings()
    sch = schedule()
    opens = sch.open.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    c_choices, c_sha = c_schedule()
    methods = ["R1", "F1", "F2"]
    selections = {}
    inner_records = []
    outer_records = []
    fits = []
    pred_chunks = []

    def one_fit(method, train, evaluate, half, month_tag, boundary, symbol):
        C = c_choices[(symbol, method, month_tag)]
        # Recency fallback is generated by the same weighted R1 fit, with its fixed registered C.
        fallback = None
        if method != "R1":
            fallback, ev0 = fit_branch(
                train, evaluate, "R1", c_choices[(symbol, "R1", month_tag)], half, boundary, lookup, opens,
                private / "models" / f"{symbol}_{month_tag}_{method}_{HALF_NAMES[half]}_fallback.joblib",
            )
            fits.append({**ev0, "role": "fallback"})
        p, ev = fit_branch(
            train, evaluate, method, C, half, boundary, lookup, opens,
            private / "models" / f"{symbol}_{month_tag}_{method}_{HALF_NAMES[half]}.joblib", fallback=fallback,
        )
        fits.append({**ev, "role": "main", "symbol": symbol, "month": month_tag})
        return p

    # Inner selection: one half-life per method, shared across stocks.
    for method in methods:
        for half in HALF_LIVES:
            for month in ("2018-03", "2018-04", "2018-05"):
                for symbol in ("AAPL", "AMZN"):
                    train = data[(data.symbol == symbol) & (data.month < month)]
                    evaluate = data[(data.symbol == symbol) & (data.month == month)]
                    if half is None:
                        p = reference.loc[evaluate.key, method].to_numpy(dtype=float)
                    else:
                        p = one_fit(method, train, evaluate, half, month, boundary_for(month, evaluate), symbol)
                    met = scores(evaluate.label, p)
                    inner_records.append({"method": method, "half_life": HALF_NAMES[half], "symbol": symbol, "month": month, **met})
        # Preserve all March-May predictions for an exact OOF comparison table.
    for month in ("2018-03", "2018-04", "2018-05"):
        for symbol in ("AAPL", "AMZN"):
            train = data[(data.symbol == symbol) & (data.month < month)]
            evaluate = data[(data.symbol == symbol) & (data.month == month)].copy()
            row = evaluate[["key", "symbol", "day", "month", "phase", "label", "target_return", "has_original_news"]].copy()
            for method in methods:
                row[f"{method}_equal"] = reference.loc[evaluate.key, method].to_numpy(dtype=float)
                half_sel = next(h for h in HALF_LIVES if HALF_NAMES[h] == selections.get(method, {}).get("chosen", "infinity")) if selections.get(method) else None
                # Selection is not available until after this loop; use a neutral
                # placeholder and fill the selected OOF column after selection.
                row[f"{method}_recency"] = np.nan
            pred_chunks.append(row)
    inner = pd.DataFrame(inner_records)
    rank_order = {"infinity": 0, "80": 1, "40": 2, "20": 3}
    for method in methods:
        summary = []
        for name in [HALF_NAMES[h] for h in HALF_LIVES]:
            g = inner[(inner.method == method) & (inner.half_life == name)]
            by_stock = g.groupby("symbol").BA.mean()
            summary.append({"half_life": name, "AAPL_mean_BA": float(by_stock["AAPL"]), "AMZN_mean_BA": float(by_stock["AMZN"]), "weaker_stock_mean_BA": float(by_stock.min()), "macro_mean_BA": float(g.BA.mean())})
        chosen = sorted(summary, key=lambda r: (-r["weaker_stock_mean_BA"], -r["macro_mean_BA"], rank_order[r["half_life"]]))[0]
        selections[method] = {"chosen": chosen["half_life"], "inner": summary}

    # Fill the selected-recency OOF columns after the globally shared choices
    # are known.  The March-May rows remain strictly forward predictions.
    for row in pred_chunks:
        if row.empty or row.phase.iloc[0] != "train_forward_oof":
            continue
        symbol = row.symbol.iloc[0]
        month = row.month.iloc[0]
        evaluate = data[(data.symbol == symbol) & (data.month == month)]
        train = data[(data.symbol == symbol) & (data.month < month)]
        for method in methods:
            half = next(h for h in HALF_LIVES if HALF_NAMES[h] == selections[method]["chosen"])
            row[f"{method}_recency"] = one_fit(method, train, evaluate, half, month, boundary_for(month, evaluate), symbol)

    # Outer months: compare unweighted and selected recency on the same official rows.
    for month in ["2018-06", "2018-07", "2018-08"]:
        for symbol in ("AAPL", "AMZN"):
            train = data[(data.symbol == symbol) & (data.month < month)]
            evaluate = data[(data.symbol == symbol) & (data.month == month)].copy()
            row = evaluate[["key", "symbol", "day", "month", "phase", "label", "target_return", "has_original_news"]].copy()
            for method in methods:
                p_equal = reference.loc[evaluate.key, method].to_numpy(dtype=float)
                row[f"{method}_equal"] = p_equal
                half_name = selections[method]["chosen"]
                half = next(h for h in HALF_LIVES if HALF_NAMES[h] == half_name)
                p = one_fit(method, train, evaluate, half, month, boundary_for(month, evaluate), symbol)
                row[f"{method}_recency"] = p
                for variant, pv in [("equal", p_equal), ("recency", p)]:
                    m = scores(evaluate.label, pv)
                    outer_records.append({"method": method, "variant": variant, "half_life": "infinity" if variant == "equal" else half_name, "symbol": symbol, "month": month, **m})
            pred_chunks.append(row)

    # Final freeze at the September boundary; development and later stay separate.
    for symbol in ("AAPL", "AMZN"):
        train = data[(data.symbol == symbol) & (data.month < "2018-09")]
        evaluate = data[(data.symbol == symbol) & (data.phase.isin(["development", "later"]))].copy()
        row = evaluate[["key", "symbol", "day", "month", "phase", "label", "target_return", "has_original_news"]].copy()
        for method in methods:
            half_sel = next(h for h in HALF_LIVES if HALF_NAMES[h] == selections[method]["chosen"])
            row[f"{method}_equal"] = reference.loc[evaluate.key, method].to_numpy(dtype=float)
            row[f"{method}_recency"] = one_fit(method, train, evaluate, half_sel, "final", pd.Timestamp("2018-09-01", tz="UTC"), symbol)
        pred_chunks.append(row)
    preds = pd.concat(pred_chunks, ignore_index=True).sort_values(["start_utc"] if "start_utc" in pred_chunks[0].columns else ["key"]) if False else pd.concat(pred_chunks, ignore_index=True)
    # The final table intentionally omits raw article keys/text. Keys are official window identifiers.
    pred_cols = ["key", "symbol", "day", "month", "phase", "label", "target_return", "has_original_news"] + [f"{m}_{v}" for m in methods for v in ("equal", "recency")]
    preds[pred_cols].to_csv(out / "predictions.csv", index=False)
    metrics, monthly = metric_rows(preds.rename(columns={"R1_equal": "R1_equal"}), [f"{m}_{v}" for m in methods for v in ("equal", "recency")])
    metrics.to_csv(out / "metrics.csv", index=False)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    inner.to_csv(out / "inner_selection_scores.csv", index=False)
    pd.DataFrame(outer_records).to_csv(out / "outer_scores_all.csv", index=False)

    # Registered recency gate, evaluated only on June-August selected versus equal.
    gate = []
    for method in methods:
        eq = pd.DataFrame(outer_records).query("method == @method and variant == 'equal'")
        rec = pd.DataFrame(outer_records).query("method == @method and variant == 'recency'")
        piv = eq.merge(rec, on=["symbol", "month"], suffixes=("_eq", "_rec"))
        deltas = (piv.BA_rec - piv.BA_eq).to_numpy()
        stock = piv.assign(delta=deltas).groupby("symbol").delta.mean()
        macro_month = piv.assign(delta=deltas).groupby("month").delta.mean()
        gate.append({"method": method, "selected_half_life": selections[method]["chosen"], "AAPL_delta_BA": float(stock.get("AAPL", np.nan)), "AMZN_delta_BA": float(stock.get("AMZN", np.nan)), "macro_delta_BA": float(np.mean(deltas)), "positive_outer_months": int((macro_month > 0).sum()), "passes": bool(stock.min() >= 0.01 and stock.max() >= -0.01 and (macro_month > 0).sum() >= 2)})
    dump(out / "recency_selection.json", {"half_lives": [HALF_NAMES[h] for h in HALF_LIVES], "selections": selections, "gate": gate, "c_schedule_sha256": c_sha, "embedding_sha256": embedding_sha, "input_sha256": sha(INPUTS), "protocol_sha256": sha(PUBLIC / "PRE_REGISTRATION.md")})
    dump(out / "training_evidence.json", {"fits": fits, "fit_count": len(fits), "runtime_seconds": time.monotonic() - start, "feature_preprocessing_weighted": False, "classifier_weighted": True, "model_dir_private": str(private / "models"), "status": "COMPLETE"})
    return preds, gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PUBLIC / "v1"))
    args = parser.parse_args()
    run(Path(args.out))
