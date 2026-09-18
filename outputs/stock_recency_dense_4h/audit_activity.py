"""Descriptive audit of the seventh raw-bar column; never uses it in a model."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW, PUBLIC, WORK, dump, sha


def run(public: Path, private: Path):
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    rows = []; source_hashes = {}
    for symbol, name in (("AAPL", "APPLE"), ("AMZN", "AMAZON")):
        path = RAW / f"{name}5.csv"; source_hashes[str(path)] = sha(path)
        b = pd.read_csv(path, header=None, names=["date", "time", "open", "high", "low", "close", "activity"])
        b["ret5"] = np.log(b.close / b.open); b["range5"] = (b.high - b.low) / b.open
        a = b.activity.astype(float); r = b.ret5.astype(float)
        rows.append({"symbol": symbol, "rows": int(len(b)), "dtype": str(b.activity.dtype), "nonnegative": bool((a >= 0).all()), "negative_count": int((a < 0).sum()), "zero_count": int((a == 0).sum()), "zero_fraction": float((a == 0).mean()), "min": float(a.min()), "max": float(a.max()), "median": float(a.median()), "q01": float(a.quantile(.01)), "q99": float(a.quantile(.99)), "unique_count": int(a.nunique()), "mean": float(a.mean()), "std": float(a.std()), "corr_abs_return": float(a.corr(r.abs())), "corr_range": float(a.corr(b.range5)), "likely_integer": bool(np.all(np.isclose(a, np.round(a)))), "source_sha256": source_hashes[str(path)]})
    pd.DataFrame(rows).to_csv(public / "activity_summary.csv", index=False)
    # Search only documentation/metadata, not raw data. This records where the
    # definition question was looked for and makes the unresolved status auditable.
    matches = []
    for root in (WORK / "audit", Path("docs"), Path("README.md")):
        if root.is_file(): files = [root]
        elif root.exists(): files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".json", ".csv"}]
        else: files = []
        for p in files:
            try: text = p.read_text(errors="ignore")
            except Exception: continue
            if re.search(r"activity|volume|tick.?count|成交量", text, re.I): matches.append(str(p))
    question = "Could you confirm what the seventh column named activity in APPLE5.csv and AMAZON5.csv represents (trade volume, tick count, or another vendor field), its units, and its adjustment/aggregation semantics?"
    lines = ["# Activity-column audit", "", "Status: **SEMANTICS_UNRESOLVED_NOT_USED**", "", "The raw five-minute files contain a seventh numeric column named `activity`. The repository documentation and audit metadata searched by this run do not provide an authoritative vendor definition. The column was not included in any recency, dense-window, or reaction experiment.", "", "## Descriptive statistics", "", pd.DataFrame(rows).to_markdown(index=False), "", "## Definition search", "", "Files containing activity/volume-related wording:"]
    lines.extend([f"- `{x}`" for x in sorted(set(matches))] or ["- None in the searched documentation/metadata."])
    lines += ["", "## Question for the professor/data provider", "", question, "", "The correlations in `activity_summary.csv` are descriptive associations only; they are not evidence that the column is volume or a usable predictive feature."]
    (public / "ACTIVITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    dump(public / "activity_audit.json", {"status": "SEMANTICS_UNRESOLVED_NOT_USED", "source_hashes": source_hashes, "searched_files": sorted(set(matches)), "question_for_professor": question})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v3"); p.add_argument("--private", default="work/stock-data/recency_dense_4h/v3/activity"); a = p.parse_args(); print(json.dumps(run(Path(a.public), Path(a.private)), indent=2))
