"""Build cutoff-safe recent-price features and run R0/R1/R2 plus S1/S2/S3."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import B, CS, INTEGRATED, PRICE0, W, assert_four_hour_rows, dump, metrics, sha

RECENT_BASE = []
for minutes in (5, 15, 30, 60):
    RECENT_BASE.extend(
        [f"recent_return_{minutes}", f"recent_range_{minutes}", f"recent_rv_{minutes}", f"recent_missing_{minutes}"]
    )
RECENT_BASE += ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]
R0 = list(PRICE0)
R1 = R0 + RECENT_BASE
R2 = list(RECENT_BASE)
VARIANTS = {"R0": R0, "R1": R1, "R2": R2}


@dataclass
class StandardizeMissing:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, columns: list[str]):
        values = frame[columns].to_numpy(dtype=float)
        self.mean = np.nanmean(values, axis=0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.scale = np.nanstd(values, axis=0)
        self.scale = np.where((self.scale > 1e-12) & np.isfinite(self.scale), self.scale, 1.0)
        return self

    def transform(self, frame: pd.DataFrame, columns: list[str]):
        values = frame[columns].to_numpy(dtype=float)
        values = (values - self.mean) / self.scale
        return np.where(np.isfinite(values), values, 0.0)


def load_bars(symbol: str):
    prefix = "APPLE" if symbol == "AAPL" else "AMAZON"
    path = W / f"raw/CHARTS/{prefix}5.csv"
    bars = pd.read_csv(
        path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "activity"],
    )
    bars.index = pd.to_datetime(bars.date + " " + bars.time, format="%Y.%m.%d %H:%M", utc=True)
    if bars.index.duplicated().any():
        raise ValueError(f"duplicate bars: {symbol}")
    return path, bars.sort_index()


def prepare_features():
    source = INTEGRATED / "prepared/data.pkl"
    data = pd.read_pickle(source).copy()
    assert_four_hour_rows(data)
    data["key"] = data.symbol + "|" + data.start_utc.astype(str)
    data["month"] = data.start_utc.dt.strftime("%Y-%m")
    data["day"] = data.start_utc.dt.strftime("%Y-%m-%d")
    schedule_path = W / "audit/xnys_schedule.csv"
    schedule = pd.read_csv(schedule_path)
    schedule["open"] = pd.to_datetime(schedule.open, utc=True)
    schedule["close"] = pd.to_datetime(schedule.close, utc=True)
    sessions = {row.open.date(): row for row in schedule.itertuples()}
    sources = [source, schedule_path]
    parity = 0
    audit = []
    for symbol, group in data.groupby("symbol"):
        path, bars = load_bars(symbol)
        sources.append(path)
        for index, row in group.iterrows():
            cutoff = row.cutoff_utc
            session = sessions.get(row.start_utc.date())
            if session is None:
                raise AssertionError("missing exchange session")
            # Independently recheck the target; target bars are never returned as features.
            target_index = pd.date_range(row.start_utc, row.end_utc - pd.Timedelta("5min"), freq="5min")
            target = bars.reindex(target_index)
            if len(target) != 48 or target[["open", "close"]].isna().any().any():
                raise AssertionError("incomplete target")
            if int(target.close.iloc[-1] > target.open.iloc[0]) != int(row.label):
                raise AssertionError("label mismatch")
            parity += 1

            usable = bars[(bars.index >= session.open) & (bars.index + pd.Timedelta("5min") <= cutoff)]
            for minutes in (5, 15, 30, 60):
                count = minutes // 5
                expected_end = cutoff
                expected_index = pd.date_range(expected_end - pd.Timedelta(minutes=minutes), expected_end - pd.Timedelta("5min"), freq="5min")
                window = usable.reindex(expected_index)
                missing = len(window) != count or window[["open", "high", "low", "close"]].isna().any().any()
                data.loc[index, f"recent_missing_{minutes}"] = int(missing)
                if missing:
                    data.loc[index, [f"recent_return_{minutes}", f"recent_range_{minutes}", f"recent_rv_{minutes}"]] = np.nan
                else:
                    data.loc[index, f"recent_return_{minutes}"] = np.log(window.close.iloc[-1] / window.open.iloc[0])
                    data.loc[index, f"recent_range_{minutes}"] = (window.high.max() - window.low.min()) / window.open.iloc[0]
                    intrabar = np.log(window.close.to_numpy() / window.open.to_numpy())
                    data.loc[index, f"recent_rv_{minutes}"] = float(np.sqrt(np.square(intrabar).sum()))
                    if not (window.index + pd.Timedelta("5min") <= cutoff).all():
                        raise AssertionError("future recent bar")

            open_bar_end = session.open + pd.Timedelta("5min")
            prior_sessions = schedule[schedule.close < session.open]
            gap_missing = open_bar_end > cutoff or prior_sessions.empty
            gap = np.nan
            if not gap_missing:
                prior_close_start = prior_sessions.iloc[-1].close - pd.Timedelta("5min")
                if session.open in bars.index and prior_close_start in bars.index:
                    gap = np.log(bars.loc[session.open, "open"] / bars.loc[prior_close_start, "close"])
                else:
                    gap_missing = True
            data.loc[index, "overnight_gap"] = gap
            data.loc[index, "overnight_gap_missing"] = int(gap_missing)
            data.loc[index, "minutes_from_open"] = (cutoff - session.open).total_seconds() / 60
            data.loc[index, "minutes_to_close"] = (session.close - cutoff).total_seconds() / 60
            audit.append(
                {
                    "key": row["key"],
                    "symbol": symbol,
                    "cutoff": cutoff.isoformat(),
                    "latest_used_bar_end": (usable.index.max() + pd.Timedelta("5min")).isoformat() if len(usable) else None,
                    "usable_current_session_bars": int(len(usable)),
                    "missing_5": int(data.loc[index, "recent_missing_5"]),
                    "missing_15": int(data.loc[index, "recent_missing_15"]),
                    "missing_30": int(data.loc[index, "recent_missing_30"]),
                    "missing_60": int(data.loc[index, "recent_missing_60"]),
                }
            )
    if parity != 1607:
        raise AssertionError("target parity incomplete")
    return data.sort_values(["start_utc", "symbol"]).reset_index(drop=True), pd.DataFrame(audit), sources


def fit_predict(train, evaluate, columns, C, name, model_dir, mode="independent"):
    if train.end_utc.max() >= evaluate.cutoff_utc.min():
        raise AssertionError("temporal leakage")
    transform = StandardizeMissing().fit(train, columns)
    x_train = transform.transform(train, columns)
    x_eval = transform.transform(evaluate, columns)
    stock_train = train.symbol.eq("AMZN").to_numpy(dtype=float)[:, None]
    stock_eval = evaluate.symbol.eq("AMZN").to_numpy(dtype=float)[:, None]
    if mode == "pooled":
        x_train = np.c_[x_train, stock_train]
        x_eval = np.c_[x_eval, stock_eval]
    elif mode == "partial":
        centered_train = stock_train - 0.5
        centered_eval = stock_eval - 0.5
        x_train = np.c_[x_train, stock_train, 0.5 * centered_train * x_train]
        x_eval = np.c_[x_eval, stock_eval, 0.5 * centered_eval * x_eval]
    model = LogisticRegression(
        C=C, solver="liblinear", max_iter=3000, random_state=573, tol=1e-8
    ).fit(x_train, train.label)
    if model.n_iter_.max() >= 3000:
        raise RuntimeError("logistic regression did not converge")
    artifact = {"columns": columns, "transform": transform, "model": model, "mode": mode}
    path = model_dir / f"{name}.joblib"
    joblib.dump(artifact, path)
    p = model.predict_proba(x_eval)[:, 1]
    loaded = joblib.load(path)
    check = loaded["transform"].transform(evaluate, columns)
    if mode == "pooled":
        check = np.c_[check, stock_eval]
    elif mode == "partial":
        check = np.c_[check, stock_eval, 0.5 * (stock_eval - 0.5) * check]
    np.testing.assert_allclose(p, loaded["model"].predict_proba(check)[:, 1], atol=1e-12, rtol=0)
    return p, {
        "name": name,
        "mode": mode,
        "C": C,
        "train_n": len(train),
        "eval_n": len(evaluate),
        "train_label_end_max": train.end_utc.max().isoformat(),
        "eval_cutoff_min": evaluate.cutoff_utc.min().isoformat(),
        "model_sha256": sha(path),
        "iterations": int(model.n_iter_.max()),
        "train_metrics": metrics(train.label, model.predict_proba(x_train)[:, 1]),
    }


def select_c(records, group):
    candidates = []
    for C in CS:
        rows = [row for row in records if row["group"] == group and row["C"] == C]
        candidates.append((-np.mean([r["BA"] for r in rows]), np.mean([r["Brier"] for r in rows]), C))
    return sorted(candidates)[0][-1]


def score_rows(predictions):
    rows = []
    methods = [c for c in predictions if c in {"R0", "R1", "R2", "S1", "S2", "S3"}]
    for (phase, symbol), group in predictions.groupby(["phase", "symbol"]):
        periods = [("all", group)] + [(month, x) for month, x in group.groupby("month")]
        if phase == "frozen":
            periods += [(split, x) for split, x in group.groupby("split")]
        for period, sample in periods:
            for method in methods:
                if sample[method].notna().all():
                    rows.append({"phase": phase, "symbol": symbol, "period": period, "method": method, **metrics(sample.label, sample[method])})
    return pd.DataFrame(rows)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    model_dir = out / "models"
    model_dir.mkdir()
    start = time.monotonic()
    data, audit, sources = prepare_features()
    data.to_pickle(out / "features.pkl")
    audit.to_csv(out / "feature_audit.csv", index=False)
    source_fingerprint = {str(path): sha(path) for path in sources + [B / "PROTOCOL.md", Path(__file__), B / "common.py"]}
    dump(out / "sources.json", source_fingerprint)
    cv_records, selections, fits, prediction_chunks = [], [], [], []

    # Independent R0/R1/R2 and S1=R1.
    for symbol, group in data.groupby("symbol"):
        for month in pd.period_range("2018-03", "2018-08", freq="M").astype(str):
            train = group[group.month < month]
            evaluate = group[group.month == month]
            row = evaluate[["key", "symbol", "day", "month", "split", "label", "has_news"]].copy()
            row["phase"] = "outer"
            for variant, columns in VARIANTS.items():
                prior = [r for r in cv_records if r["group"] == f"{symbol}|{variant}"]
                chosen = 0.1 if not prior else select_c(prior, f"{symbol}|{variant}")
                for C in CS:
                    p, evidence = fit_predict(train, evaluate, columns, C, f"{symbol}_{month}_{variant}_{C}", model_dir)
                    result = {"group": f"{symbol}|{variant}", "symbol": symbol, "month": month, "method": variant, "C": C, **metrics(evaluate.label, p)}
                    cv_records.append(result)
                    fits.append(evidence)
                selected_p, evidence = fit_predict(train, evaluate, columns, chosen, f"{symbol}_{month}_{variant}_selected", model_dir)
                fits.append(evidence)
                row[variant] = selected_p
                selections.append({"phase": "outer", "symbol": symbol, "month": month, "method": variant, "C": chosen, "selection_rule": "past forward months; current month excluded"})
            row["S1"] = row.R1
            prediction_chunks.append(row)
        train = group[group.month < "2018-09"]
        evaluate = group[group.month >= "2018-09"]
        row = evaluate[["key", "symbol", "day", "month", "split", "label", "has_news"]].copy()
        row["phase"] = "frozen"
        for variant, columns in VARIANTS.items():
            chosen = select_c(cv_records, f"{symbol}|{variant}")
            p, evidence = fit_predict(train, evaluate, columns, chosen, f"{symbol}_final_{variant}", model_dir)
            fits.append(evidence)
            row[variant] = p
            selections.append({"phase": "frozen", "symbol": symbol, "month": "final", "method": variant, "C": chosen, "selection_months": ["2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08"]})
        row["S1"] = row.R1
        prediction_chunks.append(row)

    predictions = pd.concat(prediction_chunks, ignore_index=True)

    # Shared models use exactly the same monthly rows and one global C.
    for method, mode in (("S2", "pooled"), ("S3", "partial")):
        shared_cv = []
        for month in pd.period_range("2018-03", "2018-08", freq="M").astype(str):
            train = data[data.month < month]
            evaluate = data[data.month == month]
            chosen = 0.1 if not shared_cv else select_c(shared_cv, method)
            for C in CS:
                p, evidence = fit_predict(train, evaluate, R1, C, f"{month}_{method}_{C}", model_dir, mode)
                fits.append(evidence)
                record = {"group": method, "symbol": "POOLED", "month": month, "method": method, "C": C, **metrics(evaluate.label, p)}
                cv_records.append(record)
                shared_cv.append(record)
            p, evidence = fit_predict(train, evaluate, R1, chosen, f"{month}_{method}_selected", model_dir, mode)
            fits.append(evidence)
            mapper = dict(zip(evaluate.key, p))
            mask = (predictions.phase == "outer") & (predictions.month == month)
            predictions.loc[mask, method] = predictions.loc[mask, "key"].map(mapper)
            selections.append({"phase": "outer", "symbol": "POOLED", "month": month, "method": method, "C": chosen, "selection_rule": "past pooled forward months; current month excluded"})
        chosen = select_c(shared_cv, method)
        train = data[data.month < "2018-09"]
        evaluate = data[data.month >= "2018-09"]
        p, evidence = fit_predict(train, evaluate, R1, chosen, f"final_{method}", model_dir, mode)
        fits.append(evidence)
        mapper = dict(zip(evaluate.key, p))
        mask = predictions.phase == "frozen"
        predictions.loc[mask, method] = predictions.loc[mask, "key"].map(mapper)
        selections.append({"phase": "frozen", "symbol": "POOLED", "month": "final", "method": method, "C": chosen, "selection_months": ["2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08"]})

    if predictions[["R0", "R1", "R2", "S1", "S2", "S3"]].isna().any().any():
        raise AssertionError("missing prediction")
    predictions = predictions.sort_values(["phase", "symbol", "key"])
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(cv_records).to_csv(out / "cv.csv", index=False)
    score_rows(predictions).to_csv(out / "metrics.csv", index=False)
    dump(out / "selection.json", selections)
    dump(out / "fits.json", fits)
    missing = {column: int(data[column].isna().sum()) for column in RECENT_BASE}
    dump(out / "feature_summary.json", {"columns": {name: cols for name, cols in VARIANTS.items()}, "missing": missing, "activity_used": False, "label_parity": 1607, "cutoff_check": "all recent bar ends <= cutoff"})
    for path, fingerprint in source_fingerprint.items():
        if sha(path) != fingerprint:
            raise AssertionError("source changed during run")
    dump(out / "status.json", {"status": "COMPLETE", "rows": 1607, "fits": len(fits), "seconds": time.monotonic() - start, "source_hashes_unchanged": True, "market_branch": "STOPPED_MISSING_QUALIFIED_DATA", "interpretation": "exploratory historical replay"})
    print(score_rows(predictions).query("phase == 'frozen' and period in ['validation','test']")[['symbol','period','method','BA','Brier']].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.out)
