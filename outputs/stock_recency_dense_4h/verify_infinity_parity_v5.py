"""Recompute canonical infinity-weight parity on every Mar--Aug fold."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "v5"
WORK = ROOT / "work" / "stock-data"
MONTHS = [f"2018-{m:02d}" for m in range(3, 9)]


def load_v4():
    spec = importlib.util.spec_from_file_location("recency_v4_for_parity", HERE / "run_recency_v4.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load recency runner")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod); return mod


def main() -> None:
    v4 = load_v4(); paper = v4.load_paper_module()
    private = WORK / "recency_dense_4h" / "v5_infinity_parity"
    if private.exists():
        raise FileExistsError(private)
    private.mkdir(parents=True); (private / "canonical").mkdir(); (private / "models").mkdir()
    d, emb, ids, means, events, _ = paper.prepare(private / "canonical")
    d = d.reset_index(drop=True); choices, _ = v4.c_schedule()
    schedule = pd.read_csv(WORK / "audit" / "xnys_schedule.csv"); schedule["open"] = pd.to_datetime(schedule.open, utc=True); opens = schedule.open.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    rows = []
    for tag in MONTHS + ["final"]:
        cutoff = "2018-09" if tag == "final" else tag
        for symbol in ("AAPL", "AMZN"):
            train = np.flatnonzero((d.symbol == symbol) & (d.month < cutoff))
            ev = np.flatnonzero((d.symbol == symbol) & ((d.month >= "2018-09") if tag == "final" else (d.month == tag)))
            if not len(ev):
                continue
            boundary = pd.Timestamp("2018-09-01", tz="UTC") if tag == "final" else pd.Timestamp(d.iloc[ev].cutoff_utc.min()).tz_convert("UTC")
            c = choices[(symbol, "R1", tag)]
            r1, _ = v4.fit_r1(d.iloc[train], d.iloc[ev], c, None, boundary, opens, private / "models" / f"{symbol}_{tag}_R1.joblib")
            for method in ("R1", "F1", "F2"):
                if method == "R1":
                    p = r1
                else:
                    p_main, _ = v4.fit_text(paper, d, train, ev, method, choices[(symbol, method, tag)], None, boundary, opens, emb, ids, means, events, private / "models" / f"{symbol}_{tag}_{method}.joblib")
                    p = p_main.copy(); no = d.iloc[ev].has_original_news.to_numpy(dtype=int) == 0; p[no] = r1[no]
                ref = d.iloc[ev][method].to_numpy(float); valid = np.isfinite(ref)
                rows.append({"fold": tag, "symbol": symbol, "method": method, "n": int(valid.sum()), "max_probability_abs_diff": float(np.max(np.abs(p[valid] - ref[valid]))) if valid.any() else 0.0, "parity_pass": bool(np.allclose(p[valid], ref[valid], rtol=0, atol=1e-10))})
    out = pd.DataFrame(rows); out.to_csv(OUT / "recency_infinity_parity_v5.csv", index=False)
    result = {"status": "PASS" if len(out) == 42 and bool(out.parity_pass.all()) and set(out.fold) >= set(MONTHS) else "FAIL", "rows": int(len(out)), "folds": sorted(out.fold.unique().tolist()), "max_probability_abs_diff": float(out.max_probability_abs_diff.max()) if len(out) else None, "required_outer_folds": ["2018-06", "2018-07", "2018-08"], "private_models": int(len(list((private / "models").glob("*.joblib"))))}
    (OUT / "recency_infinity_parity_v5.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.parse_args(); main()
