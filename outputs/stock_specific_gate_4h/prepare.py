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
R1_ISSUED = ROOT / "work" / "stock-data" / "nextgen_4h" / "price_v1" / "predictions.csv"
R1_EVIDENCE = ROOT / "work" / "stock-data" / "nextgen_4h" / "price_v1" / "fits.json"


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
    r1 = pd.read_csv(R1_ISSUED, float_precision="round_trip").rename(columns={"R1": "p_price_private"})
    r1["key"] = r1["key"].astype(str)
    issued_price = d[["key", "p_price"]].merge(r1, on="key", how="left", validate="one_to_one")
    if issued_price.p_price_private.isna().sum() != issued_price.p_price.isna().sum():
        raise AssertionError("R1 issued-probability missingness differs from canonical R1")
    paired = issued_price.p_price.notna()
    r1_parity = float(np.max(np.abs(
        issued_price.loc[paired, "p_price"].to_numpy(float)
        - issued_price.loc[paired, "p_price_private"].to_numpy(float)
    ))) if paired.any() else 0.0
    if r1_parity > 1e-15:
        raise AssertionError(f"R1 issued probability parity failure: {r1_parity}")
    if not np.array_equal(d.month.astype(str).to_numpy(), d.start_utc.dt.strftime("%Y-%m").to_numpy()):
        raise AssertionError("month key drift")
    for c in STATE_COLUMNS:
        if c not in d.columns:
            raise AssertionError(f"missing fixed state feature {c}")
    d["eligible_news"] = (d.has_news.eq(1) & d.p_price.notna() & d.p_text.notna()).astype(int)
    d["advantage"] = (
        (d["label"].to_numpy(float) - d["p_text"].to_numpy(float)) ** 2
        - (d["label"].to_numpy(float) - d["p_price"].to_numpy(float)) ** 2
    )
    if not np.isfinite(d.loc[d.eligible_news.eq(1), "advantage"]).all():
        raise AssertionError("controller-eligible advantage must be finite")
    prov = {
        "inputs_sha256": sha(INPUTS),
        "predictions_sha256": sha(PREDICTIONS),
        "training_evidence_sha256": sha(EVIDENCE),
        "experts": {
            "R1": {
                "issued_prediction_file": str(PREDICTIONS.relative_to(ROOT)),
                "issued_private_prediction_file": str(R1_ISSUED.relative_to(ROOT)),
                "private_prediction_sha256": sha(R1_ISSUED),
                "private_fit_evidence_file": str(R1_EVIDENCE.relative_to(ROOT)),
                "private_fit_evidence_sha256": sha(R1_EVIDENCE),
                "column": "R1",
                "private_column": "R1",
                "probability_parity_max_abs_error": r1_parity,
                "fit_evidence_key": "nextgen_4h/price_v1/fits.json",
            },
            "F1_new": {
                "issued_prediction_file": str(PREDICTIONS.relative_to(ROOT)),
                "column": "F1_new",
                "fit_evidence_key": "F1_new",
            },
        },
        "state_columns": STATE_COLUMNS,
    }
    return d, prov


def validate_expert_evidence(frame: pd.DataFrame) -> dict:
    evidence = json.loads(EVIDENCE.read_text())
    result = {"evidence_sha256": sha(EVIDENCE), "r1_evidence_sha256": sha(R1_EVIDENCE), "experts": {}}
    r1_fits = {x["name"]: x for x in json.loads(R1_EVIDENCE.read_text())}
    f1_fits = evidence.get("F1_new", {}).get("fits", [])
    f1_choices = {x["month"]: x["C"] for x in evidence.get("F1_new", {}).get("choices", [])}
    f1_indexed = {(x["symbol"], x["month"], x["C"]): x for x in f1_fits}
    audits = {"R1": {"checks": [], "freeze": [], "missing": []}, "F1_new": {"checks": [], "freeze": [], "missing": []}}
    for row in frame[frame.eligible_news.eq(1)].itertuples(index=False):
        if row.month < "2018-03":
            continue
        fit_month = row.month if row.month < "2018-09" else "final"
        r1_name = f"{row.symbol}_{fit_month}_R1_selected" if fit_month != "final" else f"{row.symbol}_final_R1"
        r1_fit = r1_fits.get(r1_name)
        f1_fit = f1_indexed.get((row.symbol, fit_month, f1_choices.get(fit_month)))
        for name, fit, end_key in (("R1", r1_fit, "train_label_end_max"), ("F1_new", f1_fit, "train_end")):
            if fit is None:
                audits[name]["missing"].append((row.symbol, fit_month))
                continue
            train_end = pd.to_datetime(fit[end_key], utc=True)
            audits[name]["checks"].append(train_end < row.cutoff_utc)
            if fit_month == "final":
                audits[name]["freeze"].append(train_end < pd.Timestamp("2018-09-01", tz="UTC"))
    for name, audit in audits.items():
        result["experts"][name] = {
            "fit_evidence_key": "nextgen_4h/price_v1/fits.json" if name == "R1" else "F1_new",
            "fit_records": int(len(r1_fits) if name == "R1" else len(f1_fits)),
            "rows_checked": int(len(audit["checks"])),
            "missing_selected_fit_records": int(len(audit["missing"])),
            "all_training_label_ends_before_row_cutoff": bool(audit["checks"] and all(audit["checks"]) and not audit["missing"]),
            "september_onward_frozen_august_model": bool(audit["freeze"] and all(audit["freeze"])),
        }
    result["all_expert_provenance_verified"] = bool(
        all(
            x["all_training_label_ends_before_row_cutoff"]
            and x["september_onward_frozen_august_model"]
            for x in result["experts"].values()
        )
    )
    return result
