# FULL and target-paragraph semantic extension

## Interpretation

All results are exploratory random backtests on previously exposed outer splits. Random historical classification is not unseen-future forecasting. The owner redirected this lane after seeing prior baseline scores; no new design claims an unobserved outer test. Hyperparameters and finite fusion weights are chosen inside training partitions. Both stocks and all three seeds are retained.

## How to read the tables

BA averages the recognition rate for rising and falling windows; higher is better. Brier measures probability error; lower is better. FULL means past prices plus binary words from the accepted full news bodies. PAPER is the sparse word-only reference; PRICE uses recent prices only. FinBERT and FinModernBERT are financial-language encoders that turn text into numerical features. Neither encoder is finetuned in this experiment.

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

## FULL plus existing title-encoder controls

| Method | Stock | BA mean ± seed SD | Brier mean | Changed / repaired / introduced, mean |
|---|---|---:|---:|---:|
| FINBERT | AAPL | 67.20% ± 1.26 | 0.2185 | 26.0 / 11.7 / 14.3 |
| FINBERT | AMZN | 60.41% ± 0.91 | 0.2454 | 7.7 / 4.3 / 3.3 |
| MODERN | AAPL | 67.61% ± 0.80 | 0.2165 | 15.7 / 8.0 / 7.7 |
| MODERN | AMZN | 59.90% ± 1.03 | 0.2433 | 15.7 / 6.3 / 9.3 |

These finite mixtures use existing inner-selected title heads and choose only a mixing weight on inner validation. They help distinguish combination benefits from longer-text benefits, but are not a perfectly isolated input ablation: body-head C and weight are jointly selected and its evidence gate differs. No learned stacker is fitted.

## FULL plus frozen target-paragraph semantics

| Method | Stock | BA mean ± seed SD | Brier mean | Changed / repaired / introduced, mean |
|---|---|---:|---:|---:|
| FINBERT | AAPL | 67.19% ± 1.15 | 0.2181 | 12.7 / 5.0 / 7.7 |
| FINBERT | AMZN | 59.75% ± 0.84 | 0.2447 | 17.7 / 6.7 / 11.0 |
| MODERN | AAPL | 67.51% ± 1.00 | 0.2182 | 17.3 / 8.7 / 8.7 |
| MODERN | AMZN | 59.78% ± 0.97 | 0.2414 | 11.3 / 3.7 / 7.7 |

Encoder weights are frozen. Newly fitted objects are per-fold PCA/scalers, small logistic classifiers, and an inner-selected finite mixing weight. Weight zero is allowed; absent target evidence returns FULL exactly. This is not LLM finetuning, attention or a learned stacker. Paragraph matching uses explicit company names and remains unvalidated for semantic target role. Both encoders receive the same full selected paragraph content in common chunks.

## Findings and decision

Observed: neither full target-paragraph combination improves mean BA over FULL for either stock. FinBERT changes BA by -0.38/-0.53 percentage points (AAPL/AMZN); Modern changes it by -0.06/-0.50 points. Every body-branch day/block paired interval includes zero. This is failure to establish an increment in this finite design, not proof that every semantic model is useless.

Observed: Brier improves in both stocks for both body branches. Mean probability distance from 0.5 also decreases. The title-fusion controls improve Brier as well. A plausible explanation is confidence shrinkage through averaging; a matched training-selected calibration/shrinkage control would be needed to attribute this benefit to new semantic information. No such new predictive experiment was added after viewing results.

Training chose zero body weight in 18/30 AAPL FinBERT fits and 16/30 AAPL Modern fits, versus 10/30 and 7/30 AMZN fits. Evidence coverage is 99.25% for AAPL and 51.24% for AMZN. Corpus membership did not expand, so body encoding does not solve missing AMZN news. Lexical paragraph selection and simple pooling remain limitations; full semantic roles have not passed independent review.

Descriptive point-score leaders among the listed comparisons are text TF-IDF + SVM for AAPL (69.49%) and FULL + existing FinBERT titles for AMZN (60.41%). These are not assembled into a per-stock test-selected deployment policy. The latter is only +0.13 points over FULL, and split variation remains substantial. Retain FULL as the declared reference and SVM as a strong traditional comparator; do not claim the proposed body branch improves direction.

Completed: 1,620 traditional top-level fit calls and 1,200 body-head fit calls, with 180 + 120 selected checkpoints independently replayed. SVM has additional internal calibration fits. Body verifier replays all 9,642 probabilities with maximum error 4.44e-16 and zero no-evidence fallback mismatches. Encoder weights stayed frozen. Stop this finite extension here; attention, LoRA and renewed full LLM inference were not started.

## Historical-case LLM

1,972 outputs preserved with fingerprints and processes suspended. Existing fold/time membership verification passed, but prompt coverage is incomplete. No partial-output selected-window predictive score is reported; full inference has not resumed.
