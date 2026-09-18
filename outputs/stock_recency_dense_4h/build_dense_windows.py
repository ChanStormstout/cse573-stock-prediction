"""Construct regular-session four-hour windows at 30-minute starts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import OLD, RAW, RECENT, R1_COLS, WORK, dump, load_data, schedule, sha


def load_bars(symbol):
    name = "APPLE" if symbol == "AAPL" else "AMAZON"
    path = RAW / f"{name}5.csv"
    b = pd.read_csv(path, header=None, names=["date", "time", "open", "high", "low", "close", "activity"])
    b.index = pd.to_datetime(b.date + " " + b.time, format="%Y.%m.%d %H:%M", utc=True)
    return path, b.sort_index()


def feature_for(cutoff, session, sessions, bars):
    out = {c: np.nan for c in R1_COLS}
    # Six completed one-hour blocks immediately before the cutoff. A block
    # crossing a regular-session gap is treated as missing, never interpolated.
    returns, ranges = [], []
    for i in range(1, 7):
        end = cutoff - pd.Timedelta(hours=i - 1)
        begin = end - pd.Timedelta(hours=1)
        idx = pd.date_range(begin, end - pd.Timedelta("5min"), freq="5min")
        w = bars.reindex(idx)
        if len(w) == 12 and not w[["open", "high", "low", "close"]].isna().any().any():
            r = float(np.log(w.close.iloc[-1] / w.open.iloc[0]))
            rr = float((w.high.max() - w.low.min()) / w.open.iloc[0])
        else:
            r, rr = np.nan, np.nan
        out[f"return_{i}"] = r; out[f"range_{i}"] = rr
        returns.append(r); ranges.append(rr)
    valid = np.asarray(returns, dtype=float)[np.isfinite(returns)]
    out["history_age_hours"] = float((cutoff - session.open).total_seconds() / 3600.0)
    out["return_mean"] = float(np.nanmean(returns)) if np.isfinite(returns).any() else np.nan
    out["return_std"] = float(np.nanstd(returns)) if np.isfinite(returns).any() else np.nan
    out["ny_hour"] = float(cutoff.tz_convert("America/New_York").hour)
    for minutes in (5, 15, 30, 60):
        idx = pd.date_range(cutoff - pd.Timedelta(minutes=minutes), cutoff - pd.Timedelta("5min"), freq="5min")
        w = bars.reindex(idx)
        missing = len(w) != minutes // 5 or w[["open", "high", "low", "close"]].isna().any().any()
        out[f"recent_missing_{minutes}"] = int(missing)
        if missing:
            out[f"recent_return_{minutes}"] = np.nan; out[f"recent_range_{minutes}"] = np.nan; out[f"recent_rv_{minutes}"] = np.nan
        else:
            out[f"recent_return_{minutes}"] = float(np.log(w.close.iloc[-1] / w.open.iloc[0]))
            out[f"recent_range_{minutes}"] = float((w.high.max() - w.low.min()) / w.open.iloc[0])
            out[f"recent_rv_{minutes}"] = float(np.sqrt(np.square(np.log(w.close.to_numpy() / w.open.to_numpy())).sum()))
    prev = sessions[sessions.close < session.open]
    gap_missing = prev.empty or cutoff <= session.open + pd.Timedelta("5min")
    out["overnight_gap_missing"] = int(gap_missing)
    if not gap_missing:
        previous_close = prev.iloc[-1].close - pd.Timedelta("5min")
        if session.open in bars.index and previous_close in bars.index:
            out["overnight_gap"] = float(np.log(bars.loc[session.open, "open"] / bars.loc[previous_close, "close"]))
        else:
            out["overnight_gap"] = np.nan; out["overnight_gap_missing"] = 1
    out["minutes_from_open"] = float((cutoff - session.open).total_seconds() / 60.0)
    out["minutes_to_close"] = float((session.close - cutoff).total_seconds() / 60.0)
    return out


def build(out_private: Path, out_public: Path):
    out_private.mkdir(parents=True, exist_ok=False); out_public.mkdir(parents=True, exist_ok=False)
    official = load_data()
    sessions = schedule()
    session_by_date = {r.open.date(): r for r in sessions.itertuples()}
    rows = []
    for symbol in ("AAPL", "AMZN"):
        path, bars = load_bars(symbol)
        for s in sessions.itertuples():
            if s.open.year != 2018 or not (1 <= s.open.month <= 8):
                continue
            starts = pd.date_range(s.open, s.close - pd.Timedelta("4h"), freq="30min")
            for start in starts:
                target_idx = pd.date_range(start, start + pd.Timedelta("4h") - pd.Timedelta("5min"), freq="5min")
                target = bars.reindex(target_idx)
                if len(target) != 48 or target[["open", "high", "low", "close"]].isna().any().any():
                    continue
                cutoff = start - pd.Timedelta("5min")
                row = {"key": f"{symbol}|{start}", "symbol": symbol, "start_utc": start, "end_utc": start + pd.Timedelta("4h"), "cutoff_utc": cutoff, "day": str(start.date()), "month": start.strftime("%Y-%m"), "label": int(target.close.iloc[-1] > target.open.iloc[0]), "target_return": float(np.log(target.close.iloc[-1] / target.open.iloc[0])), "dense": 1}
                row.update(feature_for(cutoff, s, sessions, bars)); rows.append(row)
    dense = pd.DataFrame(rows).sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    # Official rows are a strict subset by symbol/start; do not duplicate them.
    official_key = set(official.key)
    dense["official"] = dense.key.isin(official_key).astype(int)
    dense["phase"] = np.select([dense.month < "2018-03", dense.month < "2018-09"], ["warmup", "train"], default="outside")
    dense["session_window_count"] = dense.groupby(["symbol", "day"])["key"].transform("count")
    dense["day_weight"] = 1.0 / dense.session_window_count
    dense["official_day_count"] = dense.groupby(["symbol", "day"]).official.transform("sum")
    dense["official_day_weight"] = np.where(dense.official > 0, 1.0 / dense.official_day_count, 0.0)
    if dense.empty or dense.key.duplicated().any(): raise AssertionError("dense construction failed")
    dense.to_pickle(out_private / "dense_windows.pkl")
    audit = dense.groupby(["symbol", "month", "official"], as_index=False).agg(windows=("key", "size"), days=("day", "nunique"), mean_day_weight=("day_weight", "mean"), min_cutoff=("cutoff_utc", "min"), max_end=("end_utc", "max"))
    audit.to_csv(out_public / "dense_window_audit.csv", index=False)
    dump(out_private / "build_manifest.json", {"rows": int(len(dense)), "official_rows": int(dense.official.sum()), "sources": {str(path): sha(path), "schedule": sha(Path(sessions.attrs.get("source", WORK / "audit/xnys_schedule.csv"))) if False else sha(WORK / "audit/xnys_schedule.csv")}, "feature_columns": R1_COLS})
    return dense


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v1"); p.add_argument("--private", default="work/stock-data/recency_dense_4h/v1")
    a = p.parse_args(); build(Path(a.private), Path(a.public))
