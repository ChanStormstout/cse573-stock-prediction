"""Load and validate the fixed experts and state for stock-specific gate v1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import STATE_COLUMNS, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = ROOT / "work" / "stock-data" / "paper_methods_4h" / "v1" / "inputs.pkl"
PREDICTIONS = ROOT / "outputs" / "stock_goal60_4h" / "v1" / "predictions.csv"
EVIDENCE = ROOT / "outputs" / "stock_goal60_4h" / "v1" / "training_evidence.json"


def phase_for(month):
    return np.select([month < "2018-03", month < "2018-09", month < "2018-11"], ["warmup", "train_forward_oof", "development"], default="later")


def load_inputs() -> tuple[pd.DataFrame, dict]:
    d = pd.read_pickle(INPUTS).copy()
    p = pd.read_csv(PREDICTIONS)
    for c in ("start_utc", "end_utc", "cutoff_utc"):
        d[c] = pd.to_datetime(d[c], utc=True)
    p["key"] = p["key"].astype(str)
    d["key"] = d.symbol + "|" + d.start_utc.astype(str)
    if len(d) != 1607 or d.key.duplicated().any() or p.key.duplicated().any():
        raise AssertionError("official keys are not unique")
    cols = ["key", "symbol", "month", "phase", "label", "has_original_news", "R1", "F1_new"]
    missing = set(cols) - set(p.columns)
    if missing:
        raise AssertionError(f"fixed expert columns missing: {sorted(missing)}")
    issued = p[cols].rename(columns={"symbol": "symbol_issued", "month": "month_issued", "phase": "phase_issued", "label": "label_issued", "has_original_news": "has_news_issued", "R1": "p_price_issued", "F1_new": "p_text_issued"})
    d = d.merge(issued, on="key", how="left", validate="one_to_one")
    d["phase"] = phase_for(d.month.astype(str))
    if not np.array_equal(d.label.to_numpy(int), d.label_issued.to_numpy(int)):
        raise AssertionError("prediction labels do not match canonical inputs")
    if not ((d.end_utc - d.start_utc) == pd.Timedelta("4h")).all() or not ((d.start_utc - d.cutoff_utc) == pd.Timedelta("5min")).all():
        raise AssertionError("four-hour/cutoff contract changed")
    d["p_price"] = pd.to_numeric(d.p_price_issued, errors="coerce")
    d["p_text"] = pd.to_numeric(d.p_text_issued, errors="coerce")
    d["has_news"] = pd.to_numeric(d.has_news_issued, errors="coerce").fillna(0).astype(int)
    d["log_news_count"] = np.log1p(pd.to_numeric(d.news_count, errors="coerce").fillna(0).to_numpy(float))
    d["abs_expert_gap"] = (d.p_text - d.p_price).abs()
    d["abs_price_centered"] = (d.p_price - 0.5).abs()
    if not np.isfinite(d.loc[d.p_price.notna(), "p_price"]).all() or not np.isfinite(d.loc[d.p_text.notna(), "p_text"]).all():
        raise AssertionError("non-finite fixed expert probability")
    if not np.array_equal(d.month.astype(str).to_numpy(), d.start_utc.dt.strftime("%Y-%m").to_numpy()):
        raise AssertionError("month key drift")
    for c in STATE_COLUMNS:
        if c not in d.columns:
            raise AssertionError(f"missing fixed state feature {c}")
    d["eligible_news"] = (d.has_news.eq(1) & d.p_price.notna() & d.p_text.notna()).astype(int)
    prov = {"inputs_sha256": sha(INPUTS), "predictions_sha256": sha(PREDICTIONS), "training_evidence_sha256": sha(EVIDENCE), "expert_file": str(PREDICTIONS.relative_to(ROOT)), "price_column": "R1", "text_column": "F1_new", "state_columns": STATE_COLUMNS}
    return d, prov


def validate_expert_evidence(frame: pd.DataFrame) -> dict:
    evidence = json.loads(EVIDENCE.read_text())
    fits = evidence.get("F1_new", {}).get("fits", [])
    by_month = {(x["symbol"], x["month"]): x for x in fits}
    checks = []
    for row in frame[frame.eligible_news.eq(1)].itertuples(index=False):
        if row.month < "2018-03":
            continue
        key = (row.symbol, row.month)
        if key in by_month:
            checks.append(pd.to_datetime(by_month[key]["train_end"], utc=True) < row.cutoff_utc)
    return {"fit_records": len(fits), "rows_checked": len(checks), "all_training_matured": bool(checks and all(checks)), "evidence_sha256": sha(EVIDENCE)}
