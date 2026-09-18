"""Repaired dense comparison with an executable June--August gate.

The historical v4 run is never modified.  This runner reads its already
prepared cutoff-safe dense pool, writes a new v5 artifact, and filters BOTH
baseline and candidate tables to the registered June--August cells before
computing the advancement contrast.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_dense_v4 import fit
from common import R1_COLS, load_data, metric_rows, scores, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GATE_MONTHS = ["2018-06", "2018-07", "2018-08"]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def main(public_path: Path, private_path: Path, source_private: Path) -> None:
    if public_path.exists() and any(public_path.iterdir()):
        raise FileExistsError(f"new dense v5 directory must be empty: {public_path}")
    if private_path.exists():
        raise FileExistsError(f"new dense v5 private directory exists: {private_path}")
    public_path.mkdir(parents=True)
    (private_path / "models").mkdir(parents=True)
    start = time.monotonic()
    dense = pd.read_pickle(source_private / "dense_windows.pkl")
    official = load_data().set_index("key")
    dense = dense.copy()
    dense["month"] = dense["month"].astype(str)
    off = dense[dense.official == 1].set_index("key")
    keys = sorted(set(off.index) & set(official.index))
    parity_rows = []
    for col in R1_COLS:
        a = off.loc[keys, col].to_numpy(float)
        b = official.loc[keys, col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        dif = np.abs(a[both] - b[both])
        parity_rows.append({
            "column": col,
            "rows_compared": len(keys),
            "finite_pairs": int(both.sum()),
            "max_abs_diff": float(dif.max()) if len(dif) else None,
            "unequal_at_1e-12": int(np.sum(dif > 1e-12)) if len(dif) else 0,
            "dense_missing_only": int((~np.isfinite(a) & np.isfinite(b)).sum()),
            "official_missing_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
        })
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(public_path / "dense_feature_parity.csv", index=False)
    parity_pass = bool((parity["unequal_at_1e-12"] == 0).all() and (parity["dense_missing_only"] == 0).all() and (parity["official_missing_only"] == 0).all())

    fits = []
    chunks = []
    outer = []
    for month in [f"2018-{m:02d}" for m in range(3, 9)]:
        for symbol in ("AAPL", "AMZN"):
            ev = dense[(dense.symbol == symbol) & (dense.month == month) & (dense.official == 1)].copy()
            off_train = dense[(dense.symbol == symbol) & (dense.month < month) & (dense.official == 1)].copy()
            dense_train = dense[(dense.symbol == symbol) & (dense.month < month)].copy()
            if ev.empty:
                continue
            row = ev[["key", "symbol", "day", "month", "phase", "label", "target_return"]].copy().reset_index(drop=True)
            p0, e0 = fit(off_train, ev, np.ones(len(off_train)), f"{symbol}_{month}_D0_equal", private_path)
            p0d, e0d = fit(off_train, ev, off_train.official_day_weight.to_numpy(), f"{symbol}_{month}_D0_day", private_path)
            p1, e1 = fit(dense_train, ev, dense_train.day_weight.to_numpy(), f"{symbol}_{month}_D1", private_path)
            fits += [
                {**e0, "method": "D0_equal", "symbol": symbol, "month": month},
                {**e0d, "method": "D0_day", "symbol": symbol, "month": month},
                {**e1, "method": "D1", "symbol": symbol, "month": month},
            ]
            row["D0_equal"], row["D0_day"], row["D1"] = p0, p0d, p1
            chunks.append(row)
            for method, p in (("D0_equal", p0), ("D0_day", p0d), ("D1", p1)):
                outer.append({"method": method, "symbol": symbol, "month": month, **scores(ev.label, p)})

    for symbol in ("AAPL", "AMZN"):
        ev = dense[(dense.symbol == symbol) & (dense.month >= "2018-09") & (dense.official == 1)].copy()
        off_train = dense[(dense.symbol == symbol) & (dense.month < "2018-09") & (dense.official == 1)].copy()
        dense_train = dense[(dense.symbol == symbol) & (dense.month < "2018-09")].copy()
        row = ev[["key", "symbol", "day", "month", "phase", "label", "target_return"]].copy().reset_index(drop=True)
        p0, e0 = fit(off_train, ev, np.ones(len(off_train)), f"{symbol}_final_D0_equal", private_path)
        p0d, e0d = fit(off_train, ev, off_train.official_day_weight.to_numpy(), f"{symbol}_final_D0_day", private_path)
        p1, e1 = fit(dense_train, ev, dense_train.day_weight.to_numpy(), f"{symbol}_final_D1", private_path)
        fits += [
            {**e0, "method": "D0_equal", "symbol": symbol, "month": "final"},
            {**e0d, "method": "D0_day", "symbol": symbol, "month": "final"},
            {**e1, "method": "D1", "symbol": symbol, "month": "final"},
        ]
        row["D0_equal"], row["D0_day"], row["D1"] = p0, p0d, p1
        chunks.append(row)

    predictions = pd.concat(chunks, ignore_index=True)
    predictions.to_csv(public_path / "dense_predictions.csv", index=False)
    metrics, monthly = metric_rows(predictions, ["D0_equal", "D0_day", "D1"])
    metrics.to_csv(public_path / "dense_metrics.csv", index=False)
    monthly.to_csv(public_path / "dense_monthly_metrics.csv", index=False)
    outer_df = pd.DataFrame(outer)
    outer_df.to_csv(public_path / "dense_outer_scores.csv", index=False)

    # Critical repair: filter each side before the merge and assert exact cells.
    base_gate = outer_df[(outer_df.method == "D0_day") & outer_df.month.isin(GATE_MONTHS)].copy()
    candidate_gate = outer_df[(outer_df.method == "D1") & outer_df.month.isin(GATE_MONTHS)].copy()
    expected_cells = {(s, m) for s in ("AAPL", "AMZN") for m in GATE_MONTHS}
    assert set(zip(base_gate.symbol, base_gate.month)) == expected_cells
    assert set(zip(candidate_gate.symbol, candidate_gate.month)) == expected_cells
    gate_rows = base_gate.merge(candidate_gate, on=["symbol", "month"], suffixes=("_base", "_candidate"), validate="one_to_one")
    assert len(gate_rows) == 6
    gate_rows["delta_BA"] = gate_rows.BA_candidate - gate_rows.BA_base
    gate_rows["delta_Brier"] = gate_rows.Brier_candidate - gate_rows.Brier_base
    gate_rows.to_csv(public_path / "dense_gate_rows.csv", index=False)
    stock_delta = gate_rows.groupby("symbol").delta_BA.mean()
    month_delta = gate_rows.groupby("month").delta_BA.mean()
    gate = {
        "baseline": "D0_day",
        "candidate": "D1",
        "months": GATE_MONTHS,
        "filter_applied_before_merge": True,
        "expected_stock_month_cells": sorted([list(x) for x in expected_cells]),
        "n_rows": int(len(gate_rows)),
        "AAPL_delta_BA": float(stock_delta["AAPL"]),
        "AMZN_delta_BA": float(stock_delta["AMZN"]),
        "macro_delta_BA": float(gate_rows.delta_BA.mean()),
        "positive_outer_months": int((month_delta > 0).sum()),
        "passes": bool(stock_delta.min() >= 0.01 and stock_delta.max() >= -0.01 and int((month_delta > 0).sum()) >= 2),
    }
    summary = {
        "status": "COMPLETE",
        "feature_mode": "reconstructed_all_official_and_augmented",
        "canonical_parity_pass": parity_pass,
        "dense_gate_corrected": gate,
        "rows": int(len(dense)),
        "official_rows": int(dense.official.sum()),
        "runtime_seconds": time.monotonic() - start,
        "historical_v4_preserved": True,
    }
    write_json(public_path / "dense_gate_corrected.json", summary)
    write_json(public_path / "dense_training_evidence.json", {"fits": fits, "summary": summary, "fit_count": len(fits)})
    write_json(public_path / "protocol_fingerprint.json", {"source_dense_sha256": sha(source_private / "dense_windows.pkl"), "runner_sha256": sha(HERE / "run_dense_v5.py"), "gate_months": GATE_MONTHS})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=HERE / "v5")
    parser.add_argument("--private", type=Path, default=ROOT / "work/stock-data/recency_dense_4h/v5_dense")
    parser.add_argument("--source-private", type=Path, default=ROOT / "work/stock-data/recency_dense_4h/v4_dense")
    args = parser.parse_args()
    main(args.public, args.private, args.source_private)
