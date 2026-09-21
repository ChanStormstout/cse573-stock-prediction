#!/usr/bin/env python3
"""Create the human-readable forward replay report after verification passes."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"
NAMES = {
    "PRICE_R1": "Recent price LR",
    "FULL_LR": "Price + full-body words (LR)",
    "TFIDF_SVM": "Full-body TF-IDF + linear SVM",
    "FINBERT_LR": "Price + frozen FinBERT",
    "FINMODERN_LR": "Price + frozen FinModernBERT",
    "EVENT_META": "Event-group average + metadata",
}
PHASES = ["train_forward_oof", "development", "later"]


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def metric_table(df: pd.DataFrame, value: str) -> str:
    q = df.pivot(index="method", columns=["phase", "symbol"], values=value).reindex(NAMES)
    cols = [(p, s) for p in PHASES for s in ("AAPL", "AMZN")]
    q = q[cols]
    labels = ["Method"] + [f"{p} {s}" for p, s in cols]
    rows = [[NAMES[i]] + [pct(v) if value != "MCC" else f"{v:.3f}" for v in q.loc[i]] for i in q.index]
    return "| " + " | ".join(labels) + " |\n|" + "|".join(["---"] * len(labels)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)


def full_table(df: pd.DataFrame, phase: str) -> str:
    rows = []
    for r in df[df.phase == phase].sort_values(["symbol", "method"]).itertuples():
        rows.append([r.symbol, NAMES[r.method], str(r.n), pct(r.BA), pct(r.accuracy), f"{r.MCC:.3f}", f"{r.Brier:.4f}", f"{r.AUC:.3f}", pct(r.pred_up)])
    head = ["Stock", "Method", "N", "BA", "Accuracy", "MCC", "Brier", "AUC", "Predicted up"]
    return "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main() -> None:
    verification = json.loads((OUT / "VERIFICATION.json").read_text())
    if verification["status"] != "PASS":
        raise RuntimeError("reporting requires verifier PASS")
    metrics = pd.read_csv(OUT / "metrics.csv")
    selections = pd.read_csv(OUT / "selections.csv")
    finals = selections[selections.month == "final"][["symbol", "method", "C"]]
    final_c = ", ".join(f"{r.symbol} {NAMES[r.method]} C={r.C:g}" for r in finals.itertuples(index=False))
    report = f"""# Chronological forward replay v1

## Plain-language result

The methods were trained only on earlier data and then asked to predict later
four-hour windows. This replay does **not** reproduce the high random-fold
scores. No method is the stable winner across both stocks and all periods.

- The full-body word model is the clearest practical reference: it reaches
  57.94% BA for AAPL and 55.66% for AMZN in development, then 51.74% and
  54.36% in the later period.
- FinBERT is strongest for later AAPL (56.71%), while event-group aggregation
  is strongest for later AMZN (57.32%). Their earlier forward results do not
  support selecting them as stable winners.
- The TF-IDF SVM that looked strong under random folds fails in time order. It
  predicts every later AAPL window as up, so its BA is exactly 50%.
- FinModernBERT helps AMZN development relative to FinBERT, but it does not
  create a consistent two-stock gain.

Development and later results below are **EXPOSED EXPLORATORY HISTORICAL
BACKTESTS**. They were not used to choose a per-stock final method.

## Balanced accuracy overview

{metric_table(metrics, "BA")}

## Ordinary accuracy overview

{metric_table(metrics, "accuracy")}

## Training-period forward OOF details

{full_table(metrics, "train_forward_oof")}

## Development details (exposed)

{full_table(metrics, "development")}

## Later details (exposed)

{full_table(metrics, "later")}

## What was actually trained

- Fresh paper-method replay: 266 regularized logistic-regression fits for the
  full-body, FinBERT and article/event aggregation branches.
- This matched replay: 114 new fits for recent-price LR, chronologically
  calibrated linear SVM and FinModernBERT LR.
- FinBERT and FinModernBERT encoder vectors were frozen. Their downstream PCA,
  scaling and classifiers were fit again inside each past-only fold.
- Final frozen C choices were reconstructed from March-August forward results:
  {final_c}.

## Integrity evidence

The independent verifier passed all registered checks. It independently
reloaded every issued model and reproduced probabilities with maximum absolute
error {verification['maximum_independent_probability_error']:.3e}. The SVM
no-news fallback equals the price probability exactly. Canonical rows, source
hashes, chronology, past-only C selection and the three reused paper-branch
probabilities all match.

## Interpretation boundary

These results answer a different question from random ten-fold testing. Random
folds ask whether similar windows from the same historical pool can classify
one another. This replay asks whether relationships learned earlier survive in
later months. The large SVM gap is evidence of temporal instability, not a
training failure.

The FNSPID Amazon candidate corpus is excluded from this predictive replay.
Its entity-quality review is incomplete, and its date-only timestamps require
a separate preregistered experiment after that gate.
"""
    (OUT / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
