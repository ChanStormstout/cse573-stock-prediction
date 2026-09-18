"""Cutoff-safe helpers shared by the bounded recency/dense/audit run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "outputs" / "stock_recency_dense_4h"
WORK = ROOT / "work" / "stock-data"
INPUTS = WORK / "paper_methods_4h" / "v1" / "inputs.pkl"
INTEGRATED = ROOT / "outputs" / "stock_integrated_4h" / "prepared"
SCHEDULE = WORK / "audit" / "xnys_schedule.csv"
NEWS_INDEX = WORK / "audit" / "news_index.pkl"
RAW = WORK / "raw" / "CHARTS"
OLD = [f"{v}_{i}" for i in range(1, 7) for v in ("return", "range")] + [
    "history_age_hours", "return_mean", "return_std", "ny_hour"
]
RECENT = [
    f"recent_{v}_{n}" for n in (5, 15, 30, 60) for v in ("return", "range", "rv", "missing")
] + ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]
R1_COLS = OLD + RECENT
CS = (0.01, 0.1, 1.0)
MONTHS = [str(x) for x in pd.period_range("2018-03", "2018-08", freq="M")]
HALF_LIVES = (None, 80, 40, 20)
HALF_NAMES = {None: "infinity", 80: "80", 40: "40", 20: "20"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def load_data() -> pd.DataFrame:
    d = pd.read_pickle(INPUTS).copy()
    for c in ("start_utc", "end_utc", "cutoff_utc", "history_end"):
        if c in d:
            d[c] = pd.to_datetime(d[c], utc=True)
    d["key"] = d["symbol"] + "|" + d["start_utc"].astype(str)
    d["month"] = d["start_utc"].dt.strftime("%Y-%m")
    d["phase"] = np.select(
        [d.month < "2018-03", d.month < "2018-09", d.month < "2018-11"],
        ["warmup", "train_forward_oof", "development"],
        default="later",
    )
    d = d.sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    if len(d) != 1607 or d.key.duplicated().any():
        raise AssertionError("official input key count or uniqueness changed")
    if not ((d.end_utc - d.start_utc) == pd.Timedelta("4h")).all():
        raise AssertionError("official windows are not four hours")
    if not ((d.start_utc - d.cutoff_utc) == pd.Timedelta("5min")).all():
        raise AssertionError("cutoff changed")
    if not np.array_equal(d.label.to_numpy(dtype=int), (d.target_return.to_numpy(dtype=float) > 0).astype(int)):
        raise AssertionError("label is not target_return sign")
    return d


def schedule() -> pd.DataFrame:
    s = pd.read_csv(SCHEDULE)
    for c in ("open", "close"):
        s[c] = pd.to_datetime(s[c], utc=True)
    return s.sort_values("open").reset_index(drop=True)


def standardize_fit(frame: pd.DataFrame, cols: list[str]):
    x = frame[cols].to_numpy(dtype=float)
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return mean, scale


def standardize_transform(frame: pd.DataFrame, cols: list[str], mean, scale):
    x = frame[cols].to_numpy(dtype=float)
    z = (x - mean) / scale
    return np.where(np.isfinite(z), z, 0.0)


def scores(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    q = p >= 0.5
    if len(y) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("invalid metric input")
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, q)) if len(np.unique(y)) == 2 else None,
        "MCC": float(matthews_corrcoef(y, q)),
        "Brier": float(brier_score_loss(y, p)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "up_recall": float(q[y == 1].mean()) if (y == 1).any() else None,
        "down_recall": float((~q[y == 0]).mean()) if (y == 0).any() else None,
        "pred_up": float(q.mean()),
        "constant": bool(len(np.unique(q)) == 1),
    }


def metric_rows(predictions: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, monthly = [], []
    x = predictions[predictions.phase != "warmup"]
    for (phase, symbol), group in x.groupby(["phase", "symbol"]):
        for method in methods:
            rows.append({"phase": phase, "symbol": symbol, "method": method, **scores(group.label, group[method])})
            for month, sub in group.groupby("month"):
                monthly.append({"phase": phase, "symbol": symbol, "month": month, "method": method, **scores(sub.label, sub[method])})
    return pd.DataFrame(rows), pd.DataFrame(monthly)
