"""Build the corrected dense four-hour price-window pool.

The v3 builder stopped at August, which made post-August evaluation fall back
to canonical features.  v4 constructs the same cutoff-safe generator for every
available 2018-01 through 2019-02 session so training and evaluation can use a
single feature implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dense_windows import feature_for, load_bars
from common import R1_COLS, WORK, dump, load_data, schedule, sha


def build(out_private: Path, out_public: Path):
    if out_public.exists():
        if any(p.name not in {"PRE_REGISTRATION.md", "CORRECTIONS.md"} and p.name not in {"recency_refit_parity.csv", "recency_selection.json", "recency_training_evidence.json", "recency_predictions.csv", "recency_outer_scores.csv", "recency_metrics.csv", "recency_monthly_metrics.csv"} for p in out_public.iterdir()):
            raise FileExistsError("public run directory contains unexpected files")
    else:
        out_public.mkdir(parents=True)
    out_private.mkdir(parents=True, exist_ok=False)
    official = load_data(); sessions = schedule(); rows = []
    for symbol in ("AAPL", "AMZN"):
        path, bars = load_bars(symbol)
        for s in sessions.itertuples():
            if not ((s.open.year == 2018 and 1 <= s.open.month <= 12) or (s.open.year == 2019 and s.open.month <= 2)):
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
    dense["official"] = dense.key.isin(set(official.key)).astype(int)
    dense["phase"] = np.select([dense.month < "2018-03", dense.month < "2018-09", dense.month < "2018-11"], ["warmup", "train_forward_oof", "development"], default="later")
    dense["session_window_count"] = dense.groupby(["symbol", "day"])["key"].transform("count")
    dense["day_weight"] = 1.0 / dense.session_window_count
    dense["official_day_count"] = dense.groupby(["symbol", "day"]).official.transform("sum")
    dense["official_day_weight"] = np.where(dense.official > 0, 1.0 / dense.official_day_count, 0.0)
    if dense.key.duplicated().any() or not set(official.key).issubset(set(dense.key)):
        raise AssertionError("dense official coverage or uniqueness failed")
    dense.to_pickle(out_private / "dense_windows.pkl")
    audit = dense.groupby(["symbol", "month", "official"], as_index=False).agg(windows=("key", "size"), days=("day", "nunique"), mean_day_weight=("day_weight", "mean"), min_cutoff=("cutoff_utc", "min"), max_end=("end_utc", "max"))
    audit.to_csv(out_public / "dense_window_audit.csv", index=False)
    dump(out_private / "build_manifest.json", {"rows": int(len(dense)), "official_rows": int(dense.official.sum()), "feature_columns": R1_COLS, "official_source": str(WORK / "paper_methods_4h" / "v1" / "inputs.pkl")})
    return dense


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v4"); p.add_argument("--private", default="work/stock-data/recency_dense_4h/v4_dense")
    a = p.parse_args(); build(Path(a.private), Path(a.public))
