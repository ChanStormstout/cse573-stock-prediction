"""Render a concise report after an approved stock-gate run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(root: Path):
    metrics = pd.read_csv(root / "metrics.csv"); monthly = pd.read_csv(root / "monthly_metrics.csv"); adv = json.loads((root / "advancement.json").read_text());
    lines = ["# Stock-specific reliability gate v1", "", "This report is generated only after the explicitly approved run.", "", "## Overall metrics", "", metrics.to_markdown(index=False), "", "## Advancement contrasts", "", pd.DataFrame(adv).to_markdown(index=False), "", "## Outer monthly metrics", "", monthly[monthly.phase.eq("train_forward_oof") & monthly.month.isin(["2018-06", "2018-07", "2018-08"])].to_markdown(index=False), "", "The oracle ceiling is a hindsight diagnostic, not a model. Development and later rows are exposed exploratory historical backtests."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "COMPLETE", "metrics_rows": len(metrics), "monthly_rows": len(monthly)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "v1"); a = p.parse_args(); main(a.root)
