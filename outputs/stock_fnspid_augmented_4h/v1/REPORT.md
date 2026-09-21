# Course news + FNSPID chronological four-hour replay

## Result in plain language

Adding direct-target FNSPID news materially increased AMZN coverage, but it did not produce a method that improved both stocks across training OOF, development and later periods. The strongest positive result was local: augmented FinModernBERT reached 62.06% BA for AMZN development. It fell to 43.96% in the later period. This is a regime-dependent exposed backtest result, not a stable winner.

The experiment therefore answers the immediate question: the earlier reruns did not use FNSPID; this rerun does. More news changed many predictions, but the additional associations learned before September did not remain reliable in November-February.

## What was actually run

- The original 1,607 AAPL/AMZN four-hour windows, labels, cutoffs and price features were retained.
- Original course news was augmented with 1,322 deduplicated FNSPID direct-target groups: 1,119 AAPL and 203 AMZN.
- Date-only FNSPID records became available at the next XNYS open and were used for three sessions; no same-day look-ahead was allowed.
- Six methods were fitted chronologically with March-August forward OOF selection and one frozen September-February model per stock.
- 228 prediction models were fitted. Independent replay and every registered integrity check passed; maximum probability discrepancy was 7.77e-16.
- Development and later periods were already exposed and remain exploratory historical backtests.

## Coverage

| Stock | Phase | Windows | Original news | Augmented news | Windows receiving FNSPID | Added group occurrences |
|---|---|---:|---:|---:|---:|---:|
| AAPL | development | 126 | 124 | 126 | 126 | 1171 |
| AAPL | later | 178 | 178 | 178 | 177 | 2201 |
| AAPL | train_forward_oof | 383 | 379 | 383 | 380 | 3202 |
| AAPL | warmup | 116 | 116 | 116 | 116 | 1977 |
| AMZN | development | 126 | 73 | 114 | 106 | 243 |
| AMZN | later | 179 | 84 | 160 | 147 | 419 |
| AMZN | train_forward_oof | 382 | 198 | 345 | 299 | 578 |
| AMZN | warmup | 117 | 60 | 110 | 102 | 332 |

## Augmented method results

### train_forward_oof

| Stock | Method | BA | Accuracy | MCC | Brier | AUC |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Recent-price reference | 52.68% | 51.96% | 0.053 | 0.2577 | 0.5412 |
| AAPL | Price + augmented full-body words | 50.67% | 52.22% | 0.013 | 0.2900 | 0.4977 |
| AAPL | Augmented TF-IDF + linear SVM | 51.65% | 59.01% | 0.057 | 0.2538 | 0.4829 |
| AAPL | Price + augmented FinBERT | 50.69% | 52.48% | 0.014 | 0.2619 | 0.5020 |
| AAPL | Price + augmented FinModernBERT | 45.53% | 46.74% | -0.088 | 0.2774 | 0.4464 |
| AAPL | Price + augmented event/meta | 50.69% | 52.48% | 0.014 | 0.2606 | 0.4969 |
| AMZN | Recent-price reference | 52.18% | 53.14% | 0.046 | 0.2562 | 0.5158 |
| AMZN | Price + augmented full-body words | 50.87% | 50.52% | 0.018 | 0.2567 | 0.5064 |
| AMZN | Augmented TF-IDF + linear SVM | 48.90% | 48.95% | -0.022 | 0.2535 | 0.4886 |
| AMZN | Price + augmented FinBERT | 49.39% | 49.48% | -0.012 | 0.2609 | 0.5021 |
| AMZN | Price + augmented FinModernBERT | 50.52% | 50.79% | 0.011 | 0.2902 | 0.4946 |
| AMZN | Price + augmented event/meta | 49.77% | 50.26% | -0.005 | 0.2679 | 0.5275 |

### development

| Stock | Method | BA | Accuracy | MCC | Brier | AUC |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Recent-price reference | 56.21% | 53.97% | 0.149 | 0.2441 | 0.6435 |
| AAPL | Price + augmented full-body words | 49.72% | 46.83% | -0.008 | 0.2994 | 0.5581 |
| AAPL | Augmented TF-IDF + linear SVM | 50.00% | 46.03% | 0.000 | 0.2683 | 0.4300 |
| AAPL | Price + augmented FinBERT | 57.71% | 54.76% | 0.225 | 0.2487 | 0.6463 |
| AAPL | Price + augmented FinModernBERT | 53.27% | 50.79% | 0.083 | 0.2526 | 0.6105 |
| AAPL | Price + augmented event/meta | 56.85% | 53.97% | 0.195 | 0.2506 | 0.6420 |
| AMZN | Recent-price reference | 51.11% | 48.41% | 0.022 | 0.2719 | 0.5553 |
| AMZN | Price + augmented full-body words | 57.14% | 57.14% | 0.139 | 0.2383 | 0.6032 |
| AMZN | Augmented TF-IDF + linear SVM | 55.47% | 58.73% | 0.113 | 0.2454 | 0.5611 |
| AMZN | Price + augmented FinBERT | 58.53% | 57.94% | 0.166 | 0.2436 | 0.6361 |
| AMZN | Price + augmented FinModernBERT | 62.06% | 59.52% | 0.239 | 0.2759 | 0.5953 |
| AMZN | Price + augmented event/meta | 56.40% | 57.14% | 0.125 | 0.2686 | 0.5773 |

### later

| Stock | Method | BA | Accuracy | MCC | Brier | AUC |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Recent-price reference | 53.87% | 53.37% | 0.086 | 0.2558 | 0.5301 |
| AAPL | Price + augmented full-body words | 49.47% | 48.88% | -0.012 | 0.2885 | 0.5649 |
| AAPL | Augmented TF-IDF + linear SVM | 50.00% | 48.88% | 0.000 | 0.2613 | 0.5772 |
| AAPL | Price + augmented FinBERT | 53.84% | 53.37% | 0.084 | 0.2583 | 0.5340 |
| AAPL | Price + augmented FinModernBERT | 53.84% | 53.37% | 0.084 | 0.2485 | 0.5616 |
| AAPL | Price + augmented event/meta | 51.64% | 51.12% | 0.037 | 0.2606 | 0.5252 |
| AMZN | Recent-price reference | 50.85% | 51.40% | 0.017 | 0.2722 | 0.5200 |
| AMZN | Price + augmented full-body words | 44.22% | 44.13% | -0.116 | 0.2702 | 0.4554 |
| AMZN | Augmented TF-IDF + linear SVM | 47.83% | 46.93% | -0.046 | 0.2658 | 0.4536 |
| AMZN | Price + augmented FinBERT | 48.61% | 49.16% | -0.028 | 0.2657 | 0.4748 |
| AMZN | Price + augmented FinModernBERT | 43.96% | 44.69% | -0.126 | 0.2871 | 0.4925 |
| AMZN | Price + augmented event/meta | 51.38% | 51.96% | 0.028 | 0.2939 | 0.5149 |

## Increment from adding FNSPID

Positive numbers mean the augmented version has higher BA than the matched original-data method.

| Phase | Stock | Full words | TF-IDF SVM | FinBERT | FinModernBERT | Event/meta |
|---|---|---:|---:|---:|---:|---:|
| train_forward_oof | AAPL | +0.36 pp | -2.44 pp | +0.67 pp | -0.81 pp | +1.41 pp |
| train_forward_oof | AMZN | -2.40 pp | -0.89 pp | +1.25 pp | +1.89 pp | +1.36 pp |
| development | AAPL | -8.22 pp | +1.72 pp | +3.58 pp | +7.89 pp | +5.05 pp |
| development | AMZN | +1.48 pp | +8.44 pp | +12.24 pp | +5.57 pp | +5.94 pp |
| later | AAPL | -2.27 pp | +0.00 pp | -2.87 pp | -0.68 pp | -4.52 pp |
| later | AMZN | -10.14 pp | -5.00 pp | -5.66 pp | -9.16 pp | -5.94 pp |

## Prediction changes

A repaired error was wrong before augmentation and correct after it. An introduced error was correct before augmentation and wrong after it.

| Phase | Stock | Method | Repaired | Introduced | Net |
|---|---|---|---:|---:|---:|
| development | AAPL | Price + augmented full-body words | 11 | 22 | -11 |
| development | AAPL | Augmented TF-IDF + linear SVM | 2 | 0 | +2 |
| development | AAPL | Price + augmented FinBERT | 13 | 9 | +4 |
| development | AAPL | Price + augmented FinModernBERT | 17 | 8 | +9 |
| development | AAPL | Price + augmented event/meta | 12 | 6 | +6 |
| development | AMZN | Price + augmented full-body words | 20 | 20 | +0 |
| development | AMZN | Augmented TF-IDF + linear SVM | 46 | 25 | +21 |
| development | AMZN | Price + augmented FinBERT | 33 | 17 | +16 |
| development | AMZN | Price + augmented FinModernBERT | 18 | 10 | +8 |
| development | AMZN | Price + augmented event/meta | 22 | 14 | +8 |
| later | AAPL | Price + augmented full-body words | 17 | 21 | -4 |
| later | AAPL | Augmented TF-IDF + linear SVM | 0 | 0 | +0 |
| later | AAPL | Price + augmented FinBERT | 17 | 22 | -5 |
| later | AAPL | Price + augmented FinModernBERT | 21 | 22 | -1 |
| later | AAPL | Price + augmented event/meta | 17 | 25 | -8 |
| later | AMZN | Price + augmented full-body words | 9 | 27 | -18 |
| later | AMZN | Augmented TF-IDF + linear SVM | 46 | 59 | -13 |
| later | AMZN | Price + augmented FinBERT | 23 | 33 | -10 |
| later | AMZN | Price + augmented FinModernBERT | 14 | 31 | -17 |
| later | AMZN | Price + augmented event/meta | 15 | 25 | -10 |

## Interpretation

1. **Coverage was genuinely improved.** AMZN later-period news coverage rose from 84/179 to 160/179 windows. Lack of any candidate news is no longer the main mechanical bottleneck in this branch.
2. **The added news was not consistently useful.** AMZN development improved for all five news methods, but every augmented news method except event/meta was below 50% BA later. Event/meta reached 51.38%, still below the original event/meta result of 57.32%.
3. **AAPL was flooded rather than rescued.** FNSPID already supplied many AAPL direct-target reports. Full-body and SVM branches collapsed toward predicting up; compressed semantic models were less damaged but did not beat the original later FinBERT result.
4. **This pattern is compatible with topic/source/time drift, not proof of it.** The observed scores show instability. They do not by themselves identify which publisher, topic or date-only timestamp caused it.
5. **The next method should gate information quality, not merely add volume.** Future news experiments should keep this augmented corpus as the default input, while learning or preregistering a past-only novelty/relevance gate and retaining exact price fallback when that gate has insufficient evidence.

## Evidence boundary

FNSPID entity links used here are deterministic direct-target high-confidence links, but independent human semantic review is incomplete. Every FNSPID timestamp in this period is date-only. The conservative next-open rule prevents same-day leakage but may make some news artificially late. Results must be described as owner-authorized exploratory augmented-data backtests.
