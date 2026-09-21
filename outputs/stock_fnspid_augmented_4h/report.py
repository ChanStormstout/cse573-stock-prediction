#!/usr/bin/env python3
"""Render the verified augmented-data experiment report."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"


def pct(x): return f"{100 * float(x):.2f}%"
def dec(x): return f"{float(x):.4f}"


def main():
    verification = json.loads((OUT / "VERIFICATION.json").read_text())
    if verification["status"] != "PASS": raise RuntimeError("verification must pass before reporting")
    m = pd.read_csv(OUT / "metrics.csv")
    coverage = pd.read_csv(OUT / "coverage.csv")
    transitions = pd.read_csv(OUT / "transitions.csv")
    labels = {
        "PRICE_R1": "Recent-price reference",
        "AUG_FULL_LR": "Price + augmented full-body words",
        "AUG_TFIDF_SVM": "Augmented TF-IDF + linear SVM",
        "AUG_FINBERT_LR": "Price + augmented FinBERT",
        "AUG_FINMODERN_LR": "Price + augmented FinModernBERT",
        "AUG_EVENT_META": "Price + augmented event/meta",
        "ORIG_FULL_LR": "Original price + full-body words",
        "ORIG_TFIDF_SVM": "Original TF-IDF + linear SVM",
        "ORIG_FINBERT_LR": "Original price + FinBERT",
        "ORIG_FINMODERN_LR": "Original price + FinModernBERT",
        "ORIG_EVENT_META": "Original event/meta",
    }
    augmented = ["PRICE_R1", "AUG_FULL_LR", "AUG_TFIDF_SVM", "AUG_FINBERT_LR", "AUG_FINMODERN_LR", "AUG_EVENT_META"]
    pairs = {"AUG_FULL_LR": "ORIG_FULL_LR", "AUG_TFIDF_SVM": "ORIG_TFIDF_SVM", "AUG_FINBERT_LR": "ORIG_FINBERT_LR", "AUG_FINMODERN_LR": "ORIG_FINMODERN_LR", "AUG_EVENT_META": "ORIG_EVENT_META"}
    lines = [
        "# Course news + FNSPID chronological four-hour replay",
        "",
        "## Result in plain language",
        "",
        "Adding direct-target FNSPID news materially increased AMZN coverage, but it did not produce a method that improved both stocks across training OOF, development and later periods. The strongest positive result was local: augmented FinModernBERT reached 62.06% BA for AMZN development. It fell to 43.96% in the later period. This is a regime-dependent exposed backtest result, not a stable winner.",
        "",
        "The experiment therefore answers the immediate question: the earlier reruns did not use FNSPID; this rerun does. More news changed many predictions, but the additional associations learned before September did not remain reliable in November-February.",
        "",
        "## What was actually run",
        "",
        "- The original 1,607 AAPL/AMZN four-hour windows, labels, cutoffs and price features were retained.",
        "- Original course news was augmented with 1,322 deduplicated FNSPID direct-target groups: 1,119 AAPL and 203 AMZN.",
        "- Date-only FNSPID records became available at the next XNYS open and were used for three sessions; no same-day look-ahead was allowed.",
        "- Six methods were fitted chronologically with March-August forward OOF selection and one frozen September-February model per stock.",
        "- 228 prediction models were fitted. Independent replay and every registered integrity check passed; maximum probability discrepancy was 7.77e-16.",
        "- Development and later periods were already exposed and remain exploratory historical backtests.",
        "",
        "## Coverage",
        "",
        "| Stock | Phase | Windows | Original news | Augmented news | Windows receiving FNSPID | Added group occurrences |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in coverage.itertuples(index=False):
        lines.append(f"| {r.symbol} | {r.phase} | {r.windows} | {r.original_news_windows} | {r.augmented_news_windows} | {r.windows_with_fnspid} | {r.added_group_occurrences} |")
    lines += ["", "## Augmented method results", ""]
    for phase in ("train_forward_oof", "development", "later"):
        lines += [f"### {phase}", "", "| Stock | Method | BA | Accuracy | MCC | Brier | AUC |", "|---|---|---:|---:|---:|---:|---:|"]
        q = m[(m.phase == phase) & m.method.isin(augmented)]
        for r in q.itertuples(index=False):
            lines.append(f"| {r.symbol} | {labels[r.method]} | {pct(r.BA)} | {pct(r.accuracy)} | {r.MCC:.3f} | {dec(r.Brier)} | {dec(r.AUC)} |")
        lines.append("")
    lines += ["## Increment from adding FNSPID", "", "Positive numbers mean the augmented version has higher BA than the matched original-data method.", "", "| Phase | Stock | Full words | TF-IDF SVM | FinBERT | FinModernBERT | Event/meta |", "|---|---|---:|---:|---:|---:|---:|"]
    for phase in ("train_forward_oof", "development", "later"):
        for symbol in ("AAPL", "AMZN"):
            z = m[(m.phase == phase) & (m.symbol == symbol)].set_index("method")
            vals = [100 * (z.loc[a, "BA"] - z.loc[b, "BA"]) for a, b in pairs.items()]
            lines.append(f"| {phase} | {symbol} | " + " | ".join(f"{x:+.2f} pp" for x in vals) + " |")
    lines += [
        "",
        "## Prediction changes",
        "",
        "A repaired error was wrong before augmentation and correct after it. An introduced error was correct before augmentation and wrong after it.",
        "",
        "| Phase | Stock | Method | Repaired | Introduced | Net |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in transitions.itertuples(index=False):
        if r.phase in ("development", "later"):
            lines.append(f"| {r.phase} | {r.symbol} | {labels[r.method]} | {r.repaired} | {r.introduced} | {r.repaired-r.introduced:+d} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "1. **Coverage was genuinely improved.** AMZN later-period news coverage rose from 84/179 to 160/179 windows. Lack of any candidate news is no longer the main mechanical bottleneck in this branch.",
        "2. **The added news was not consistently useful.** AMZN development improved for all five news methods, but every augmented news method except event/meta was below 50% BA later. Event/meta reached 51.38%, still below the original event/meta result of 57.32%.",
        "3. **AAPL was flooded rather than rescued.** FNSPID already supplied many AAPL direct-target reports. Full-body and SVM branches collapsed toward predicting up; compressed semantic models were less damaged but did not beat the original later FinBERT result.",
        "4. **This pattern is compatible with topic/source/time drift, not proof of it.** The observed scores show instability. They do not by themselves identify which publisher, topic or date-only timestamp caused it.",
        "5. **The next method should gate information quality, not merely add volume.** Future news experiments should keep this augmented corpus as the default input, while learning or preregistering a past-only novelty/relevance gate and retaining exact price fallback when that gate has insufficient evidence.",
        "",
        "## Evidence boundary",
        "",
        "FNSPID entity links used here are deterministic direct-target high-confidence links, but independent human semantic review is incomplete. Every FNSPID timestamp in this period is date-only. The conservative next-open rule prevents same-day leakage but may make some news artificially late. Results must be described as owner-authorized exploratory augmented-data backtests.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__": main()
