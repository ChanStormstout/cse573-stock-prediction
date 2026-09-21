# Chronological forward replay v1

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

| Method | train_forward_oof AAPL | train_forward_oof AMZN | development AAPL | development AMZN | later AAPL | later AMZN |
|---|---|---|---|---|---|---|
| Recent price LR | 52.68% | 52.18% | 56.21% | 51.11% | 53.87% | 50.85% |
| Price + full-body words (LR) | 50.30% | 53.27% | 57.94% | 55.66% | 51.74% | 54.36% |
| Full-body TF-IDF + linear SVM | 54.09% | 49.79% | 48.28% | 47.03% | 50.00% | 52.83% |
| Price + frozen FinBERT | 50.02% | 48.14% | 54.13% | 46.29% | 56.71% | 54.27% |
| Price + frozen FinModernBERT | 46.35% | 48.63% | 45.39% | 56.49% | 54.52% | 53.12% |
| Event-group average + metadata | 49.28% | 48.40% | 51.80% | 50.46% | 56.16% | 57.32% |

## Ordinary accuracy overview

| Method | train_forward_oof AAPL | train_forward_oof AMZN | development AAPL | development AMZN | later AAPL | later AMZN |
|---|---|---|---|---|---|---|
| Recent price LR | 51.96% | 53.14% | 53.97% | 48.41% | 53.37% | 51.40% |
| Price + full-body words (LR) | 52.48% | 53.14% | 55.56% | 57.14% | 51.12% | 54.19% |
| Full-body TF-IDF + linear SVM | 60.05% | 50.00% | 44.44% | 42.06% | 48.88% | 54.19% |
| Price + frozen FinBERT | 50.65% | 48.95% | 51.59% | 45.24% | 56.18% | 54.75% |
| Price + frozen FinModernBERT | 48.04% | 49.48% | 43.65% | 53.17% | 53.93% | 54.19% |
| Event-group average + metadata | 50.13% | 48.95% | 49.21% | 50.79% | 55.62% | 57.54% |

## Training-period forward OOF details

| Stock | Method | N | BA | Accuracy | MCC | Brier | AUC | Predicted up |
|---|---|---|---|---|---|---|---|---|
| AAPL | Event-group average + metadata | 383 | 49.28% | 50.13% | -0.014 | 0.2662 | 0.497 | 54.57% |
| AAPL | Price + frozen FinBERT | 383 | 50.02% | 50.65% | 0.000 | 0.2602 | 0.518 | 53.52% |
| AAPL | Price + frozen FinModernBERT | 383 | 46.35% | 48.04% | -0.073 | 0.2828 | 0.444 | 58.75% |
| AAPL | Price + full-body words (LR) | 383 | 50.30% | 52.48% | 0.006 | 0.2593 | 0.514 | 62.14% |
| AAPL | Recent price LR | 383 | 52.68% | 51.96% | 0.053 | 0.2577 | 0.541 | 46.48% |
| AAPL | Full-body TF-IDF + linear SVM | 383 | 54.09% | 60.05% | 0.109 | 0.2415 | 0.565 | 83.81% |
| AMZN | Event-group average + metadata | 382 | 48.40% | 48.95% | -0.032 | 0.2720 | 0.476 | 58.64% |
| AMZN | Price + frozen FinBERT | 382 | 48.14% | 48.95% | -0.038 | 0.2749 | 0.464 | 62.83% |
| AMZN | Price + frozen FinModernBERT | 382 | 48.63% | 49.48% | -0.028 | 0.2694 | 0.441 | 63.35% |
| AMZN | Price + full-body words (LR) | 382 | 53.27% | 53.14% | 0.065 | 0.2606 | 0.522 | 48.17% |
| AMZN | Recent price LR | 382 | 52.18% | 53.14% | 0.046 | 0.2562 | 0.516 | 65.45% |
| AMZN | Full-body TF-IDF + linear SVM | 382 | 49.79% | 50.00% | -0.004 | 0.2521 | 0.524 | 53.40% |

## Development details (exposed)

| Stock | Method | N | BA | Accuracy | MCC | Brier | AUC | Predicted up |
|---|---|---|---|---|---|---|---|---|
| AAPL | Event-group average + metadata | 126 | 51.80% | 49.21% | 0.047 | 0.2937 | 0.592 | 82.54% |
| AAPL | Price + frozen FinBERT | 126 | 54.13% | 51.59% | 0.107 | 0.2803 | 0.620 | 81.75% |
| AAPL | Price + frozen FinModernBERT | 126 | 45.39% | 43.65% | -0.103 | 0.2807 | 0.516 | 72.22% |
| AAPL | Price + full-body words (LR) | 126 | 57.94% | 55.56% | 0.195 | 0.2500 | 0.576 | 79.37% |
| AAPL | Recent price LR | 126 | 56.21% | 53.97% | 0.149 | 0.2441 | 0.644 | 77.78% |
| AAPL | Full-body TF-IDF + linear SVM | 126 | 48.28% | 44.44% | -0.138 | 0.2754 | 0.418 | 98.41% |
| AMZN | Event-group average + metadata | 126 | 50.46% | 50.79% | 0.009 | 0.2623 | 0.538 | 48.41% |
| AMZN | Price + frozen FinBERT | 126 | 46.29% | 45.24% | -0.073 | 0.2977 | 0.503 | 55.56% |
| AMZN | Price + frozen FinModernBERT | 126 | 56.49% | 53.17% | 0.132 | 0.2469 | 0.616 | 63.49% |
| AMZN | Price + full-body words (LR) | 126 | 55.66% | 57.14% | 0.112 | 0.2481 | 0.580 | 42.06% |
| AMZN | Recent price LR | 126 | 51.11% | 48.41% | 0.022 | 0.2719 | 0.555 | 61.90% |
| AMZN | Full-body TF-IDF + linear SVM | 126 | 47.03% | 42.06% | -0.065 | 0.2654 | 0.483 | 73.02% |

## Later details (exposed)

| Stock | Method | N | BA | Accuracy | MCC | Brier | AUC | Predicted up |
|---|---|---|---|---|---|---|---|---|
| AAPL | Event-group average + metadata | 178 | 56.16% | 55.62% | 0.141 | 0.2940 | 0.542 | 74.16% |
| AAPL | Price + frozen FinBERT | 178 | 56.71% | 56.18% | 0.152 | 0.2856 | 0.550 | 73.60% |
| AAPL | Price + frozen FinModernBERT | 178 | 54.52% | 53.93% | 0.105 | 0.2612 | 0.576 | 75.84% |
| AAPL | Price + full-body words (LR) | 178 | 51.74% | 51.12% | 0.042 | 0.2587 | 0.539 | 77.53% |
| AAPL | Recent price LR | 178 | 53.87% | 53.37% | 0.086 | 0.2558 | 0.530 | 71.91% |
| AAPL | Full-body TF-IDF + linear SVM | 178 | 50.00% | 48.88% | 0.000 | 0.2688 | 0.475 | 100.00% |
| AMZN | Event-group average + metadata | 179 | 57.32% | 57.54% | 0.147 | 0.2483 | 0.569 | 54.75% |
| AMZN | Price + frozen FinBERT | 179 | 54.27% | 54.75% | 0.087 | 0.2617 | 0.569 | 59.78% |
| AMZN | Price + frozen FinModernBERT | 179 | 53.12% | 54.19% | 0.069 | 0.2492 | 0.529 | 71.51% |
| AMZN | Price + full-body words (LR) | 179 | 54.36% | 54.19% | 0.087 | 0.2583 | 0.541 | 46.93% |
| AMZN | Recent price LR | 179 | 50.85% | 51.40% | 0.017 | 0.2722 | 0.520 | 60.89% |
| AMZN | Full-body TF-IDF + linear SVM | 179 | 52.83% | 54.19% | 0.067 | 0.2453 | 0.551 | 77.09% |

## What was actually trained

- Fresh paper-method replay: 266 regularized logistic-regression fits for the
  full-body, FinBERT and article/event aggregation branches.
- This matched replay: 114 new fits for recent-price LR, chronologically
  calibrated linear SVM and FinModernBERT LR.
- FinBERT and FinModernBERT encoder vectors were frozen. Their downstream PCA,
  scaling and classifiers were fit again inside each past-only fold.
- Final frozen C choices were reconstructed from March-August forward results:
  AAPL Recent price LR C=0.01, AAPL Full-body TF-IDF + linear SVM C=1, AAPL Price + frozen FinModernBERT C=0.1, AMZN Recent price LR C=0.1, AMZN Full-body TF-IDF + linear SVM C=1, AMZN Price + frozen FinModernBERT C=0.01.

## Integrity evidence

The independent verifier passed all registered checks. It independently
reloaded every issued model and reproduced probabilities with maximum absolute
error 3.886e-16. The SVM
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
