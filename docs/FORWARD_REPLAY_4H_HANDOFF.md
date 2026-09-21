# Four-hour chronological forward replay handoff

## Status

**COMPLETE / INDEPENDENT VERIFIER PASS.** The replay uses earlier months to
predict later four-hour windows. It does not use random folds, FNSPID text,
LLM output, per-stock post-result winner selection or a post-result ensemble.

Review entrypoints:

- [preregistration](../outputs/stock_forward_replay_4h/PRE_REGISTRATION.md)
- [plain-language report](../outputs/stock_forward_replay_4h/v1/REPORT.md)
- [metrics](../outputs/stock_forward_replay_4h/v1/metrics.csv)
- [monthly metrics](../outputs/stock_forward_replay_4h/v1/monthly_metrics.csv)
- [verification](../outputs/stock_forward_replay_4h/v1/VERIFICATION.json)

## What ran

The fresh paper-method replay fit 266 logistic-regression models for the
full-body, FinBERT and article/event aggregation branches. The matched replay
fit another 114 models for recent-price LR, chronologically calibrated TF-IDF
SVM and FinModernBERT LR. Frozen FinBERT and FinModernBERT vectors were reused,
while PCA, scaling and classifiers were refit separately within each past-only
fold.

Chronology is January-February initial history, March-August expanding forward
OOF, September-October development, and November-February later. The final
development and later model is trained through August; both reported periods
are exposed exploratory historical backtests.

## Main evidence

| Method | AAPL OOF | AMZN OOF | AAPL dev | AMZN dev | AAPL later | AMZN later |
|---|---:|---:|---:|---:|---:|---:|
| Recent price LR | 52.68% | 52.18% | 56.21% | 51.11% | 53.87% | 50.85% |
| Price + full-body words | 50.30% | 53.27% | 57.94% | 55.66% | 51.74% | 54.36% |
| TF-IDF + linear SVM | 54.09% | 49.79% | 48.28% | 47.03% | 50.00% | 52.83% |
| Price + FinBERT | 50.02% | 48.14% | 54.13% | 46.29% | 56.71% | 54.27% |
| Price + FinModernBERT | 46.35% | 48.63% | 45.39% | 56.49% | 54.52% | 53.12% |
| Event groups + metadata | 49.28% | 48.40% | 51.80% | 50.46% | 56.16% | 57.32% |

No method is a stable winner across the two stocks and three phases. The SVM
random-fold result does not transfer: it predicts every later AAPL window up.
FULL remains the clearest practical reference; later FinBERT and event scores
are useful observations but contradict their earlier forward performance.

## Integrity

The verifier checks 1,607 canonical rows and 1,374 evaluated rows, source
hashes, fold chronology, past-only C choices, exact no-news SVM fallback and
parity of the three reused paper branches. It reloads every issued new model;
the maximum probability discrepancy is `3.89e-16`.

## FNSPID boundary

The outcome-blind FNSPID candidate audit remains a separate input-coverage
lane. Its date-only availability policy and incomplete independent entity
review prevent it from entering this predictive replay.
