# FULL and target-paragraph semantic extension

## Interpretation

All results are exploratory random backtests on previously exposed outer splits. Random historical classification is not unseen-future forecasting. The owner redirected this lane after seeing prior baseline scores; no new design claims an unobserved outer test. Hyperparameters and finite fusion weights are chosen inside training partitions. Both stocks and all three seeds are retained.

## Traditional finite comparison

| Method | Stock | BA mean ± seed SD | Brier mean | Changed / repaired / introduced, mean |
|---|---|---:|---:|---:|
| TFIDF_LR | AAPL | 64.94% ± 1.20 | 0.2210 | 157.0 / 71.0 / 86.0 |
| TFIDF_SVM | AAPL | 69.49% ± 1.80 | 0.2056 | 145.7 / 80.7 / 65.0 |
| TFIDF_RF | AAPL | 59.86% ± 0.71 | 0.2332 | 217.3 / 83.7 / 133.7 |
| PAPER | AAPL | 67.25% ± 1.01 | 0.2272 | 101.7 / 49.0 / 52.7 |
| PRICE | AAPL | 55.84% ± 0.71 | 0.2468 | 299.3 / 105.0 / 194.3 |
| FULL | AAPL | 67.57% ± 1.21 | 0.2228 | 0.0 / 0.0 / 0.0 |
| FINBERT | AAPL | 55.01% ± 0.63 | 0.2463 | 302.7 / 102.3 / 200.3 |
| MODERN | AAPL | 53.65% ± 0.30 | 0.2502 | 296.7 / 94.7 / 202.0 |
| TFIDF_LR | AMZN | 60.11% ± 0.58 | 0.2365 | 208.3 / 106.0 / 102.3 |
| TFIDF_SVM | AMZN | 60.12% ± 0.49 | 0.2353 | 206.3 / 105.0 / 101.3 |
| TFIDF_RF | AMZN | 58.46% ± 0.73 | 0.2423 | 237.0 / 113.7 / 123.3 |
| PAPER | AMZN | 58.52% ± 1.14 | 0.2498 | 194.3 / 92.7 / 101.7 |
| PRICE | AMZN | 52.68% ± 1.06 | 0.2563 | 179.3 / 60.0 / 119.3 |
| FULL | AMZN | 60.28% ± 0.81 | 0.2524 | 0.0 / 0.0 / 0.0 |
| FINBERT | AMZN | 52.99% ± 0.60 | 0.2532 | 176.3 / 59.0 / 117.3 |
| MODERN | AMZN | 55.72% ± 1.58 | 0.2493 | 153.7 / 58.7 / 95.0 |

TFIDF methods here use text only; FULL combines binary full-body word features and past price features. Therefore text-only versus FULL is a practical comparator, not an isolated representation ablation. SVM probabilities use inner training-only sigmoid calibration. Seed SD describes split variation, not an independent confidence interval. Changed/repaired/introduced counts are relative to FULL.

## Target-paragraph semantic experiment

PENDING: do not infer a result from input preparation or encoding progress.

## Historical-case LLM

1,972 outputs preserved with fingerprints and processes suspended. Existing fold/time membership verification passed, but prompt coverage is incomplete. No partial-output selected-window predictive score is reported; full inference has not resumed.
