"""Create the plain-language report for the bounded recency/dense/audit run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from common import PUBLIC, dump, sha


def fmt(x):
    return "—" if pd.isna(x) else f"{100*x:.2f}%" if isinstance(x, (float, int)) and abs(x) <= 1 else str(x)


def report(public: Path):
    rec = pd.read_csv(public / "recency_metrics.csv")
    dense = pd.read_csv(public / "dense_metrics.csv")
    gate = json.loads((public / "recency_selection.json").read_text())["gate"]
    dsummary = json.loads((public / "dense_training_summary.json").read_text())
    reaction = json.loads((public / "article_reaction_summary.json").read_text())
    activity = json.loads((public / "activity_audit.json").read_text())
    lines = ["# Recency weighting, dense windows, and reaction audit — v3", "", "## Executive result", "", "This is a bounded four-hour experiment on the original AAPL/AMZN windows. The equal-weight R1/F1/F2 columns reproduce the previously saved official probabilities exactly; the new recency columns are the only changed prediction mechanism. No method passed the registered cross-month promotion line. The article branch is an audit only and no reaction predictor was trained.", "", "## What was actually run", "", "- Three real weighted logistic-regression pipelines: R1 recent price, F1 price plus binary full-text features, and F2 price plus frozen FinBERT PCA features.", "- Four exact half-lives: infinity, 80, 40, and 20 NYSE sessions. A single half-life was selected per input across both stocks using March–May forward OOF.", "- 30-minute-stride four-hour windows from raw regular-session five-minute bars. D0 equal, D0 day-normalized, and D1 augmented day-normalized were evaluated on official rows.", "- A descriptive article reaction audit at 30/60/120/240 minutes under same-session and trading-time definitions.", "- A descriptive audit of the seventh raw-bar `activity` column; it was excluded from every model.", "", "## Recency selection and outer gate", "", "| input | selected half-life | AAPL ΔBA | AMZN ΔBA | macro ΔBA | positive outer months | pass |", "|---|---:|---:|---:|---:|---:|---|"]
    for g in gate:
        lines.append(f"| {g['method']} | {g['selected_half_life']} | {100*g['AAPL_delta_BA']:.2f} pp | {100*g['AMZN_delta_BA']:.2f} pp | {100*g['macro_delta_BA']:.2f} pp | {g['positive_outer_months']}/3 | {'yes' if g['passes'] else 'no'} |")
    lines += ["", "The registered line required the weaker stock to gain at least 1 pp, no stock to lose more than 1 pp, and positive macro BA in at least two of June, July, and August. Recency did not satisfy it. F1 and F2 improved AAPL in this exposed outer interval but lost AMZN; R1 also failed the weaker-stock rule.", "", "## Development and later results", "", "| period | stock | method | BA | MCC | Brier |", "|---|---|---|---:|---:|---:|"]
    for _, r in rec[rec.phase.isin(["development", "later"])].iterrows():
        lines.append(f"| {r.phase} | {r.symbol} | {r.method} | {100*r.BA:.2f}% | {r.MCC:.3f} | {r.Brier:.4f} |")
    lines += ["", "The equal rows are the official historical reference. Development and later are already exposed and were not used to choose the half-life. A high later score is therefore descriptive evidence only.", "", "## Dense-window result", "", f"The dense builder produced {dsummary['rows']} unique training-window rows, including {dsummary['official_rows']} official rows. Relative to the day-normalized official control D0_day, D1 changed the June–August BA by {100*dsummary['dense_gate']['AAPL_delta_BA']:.2f} pp for AAPL and {100*dsummary['dense_gate']['AMZN_delta_BA']:.2f} pp for AMZN. This was below the registered 1 pp per-stock threshold, so the text extension was stopped. D2 was {dsummary['D2_status']} because the R1 recency gate failed.", "", "## Article reaction audit", "", f"The audit used only accepted article IDs already attached to official windows, retained raw occurrences and deterministic normalized online groups, and measured first complete bars after `available_utc`. It produced {reaction['rows']} article–horizon rows from {reaction['unique_articles']} accepted articles and {reaction['unique_groups']} normalized groups. The result is feasible for a future reaction experiment under the operational 100-group coverage check: {reaction['feasible_for_next_reaction_experiment']}. This is not evidence of causality, and no reaction model was trained.", "", "See `article_reaction_audit.csv` for stock, horizon, definition, period, coverage, sign balance, duplicate concentration, and return magnitude. The audit deliberately does not turn a post-availability return into a news causal effect.", "", "## Activity column", "", f"Status is `{activity['status']}`. The files contain a nonnegative integer-like seventh column, but no authoritative definition or units were found in the searched repository metadata. It remains unused. The question for the professor/data provider is recorded in `ACTIVITY_AUDIT.md`.", "", "## Interpretation", "", "1. Recency weighting can alter the model's learned prior, but the effect is not stable across the two stocks and exposed months.", "2. Dense windows add more highly overlapping samples. Their small positive shift does not meet the registered gate and cannot be treated as independent-data evidence.", "3. The reaction audit can tell us whether a future event-study branch has enough timestamped coverage; it cannot prove that the observed move was caused by the article.", "4. The remaining bottleneck is not solved by this run. A verified contemporaneous market-data source is still required before the market-state branch can be tested fairly.", "", "## Reproduction", "", "```text", "work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/run_recency.py --out outputs/stock_recency_dense_4h/v2", "work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/build_dense_windows.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3", "work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/run_dense.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3", "work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/audit_article_reactions.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3/reactions2", "work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/audit_activity.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3/activity", "```", "", "All public result files are aggregate metrics, official keys, and audit summaries. Raw news, cached embeddings, and model binaries remain under `work/`."]
    (public / "REPORT.md").write_text("\n".join(lines) + "\n")

    # Preselect case slots by key hash and category, then reveal outcomes. This
    # keeps the case list deterministic and prevents picking only attractive examples.
    p = pd.read_csv(public / "recency_predictions.csv")
    cases = []
    for stock in ("AAPL", "AMZN"):
        x = p[(p.symbol == stock) & (p.phase == "later")].copy()
        x["old_correct"] = (x.R1_equal >= .5) == x.label
        x["new_correct"] = (x.R1_recency >= .5) == x.label
        for category, mask in {
            "recency_fixed_error": (~x.old_correct) & x.new_correct,
            "recency_new_error": x.old_correct & (~x.new_correct),
            "both_wrong": (~x.old_correct) & (~x.new_correct),
            "both_right": x.old_correct & x.new_correct,
        }.items():
            sub = x[mask].copy(); sub["rank"] = sub.key.map(lambda k: hashlib.sha256(k.encode()).hexdigest())
            for _, r in sub.sort_values("rank").head(3).iterrows():
                cases.append({"stock": stock, "category": category, "key": r.key, "label": int(r.label), "equal_p": float(r.R1_equal), "recency_p": float(r.R1_recency), "has_original_news": int(r.has_original_news)})
    pd.DataFrame(cases).to_csv(public / "CASE_NOTES.csv", index=False)
    (public / "CASE_NOTES.md").write_text("# Fixed recency case slots\n\nCases were selected by a deterministic hash-ranked rule within four categories before reading their outcomes: fixed error, new error, both wrong, and both right. The table omits article text and URLs; it is a diagnostic sample, not a population estimate. See `CASE_NOTES.csv`.\n")
    dump(public / "protocol.json", {"pre_registration_sha256": sha(PUBLIC / "PRE_REGISTRATION.md"), "input_sha256": sha(Path("work/stock-data/paper_methods_4h/v1/inputs.pkl")), "reference_prediction_sha256": sha(PUBLIC.parent / "stock_goal60_4h" / "v1" / "predictions.csv"), "status": "COMPLETE", "all_evaluation_exposed": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v3"); a = p.parse_args(); report(Path(a.public))
