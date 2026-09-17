# FinBERT event adapter and gated four-hour correction

## Plain-language result

This run really trained three FinBERT variants on the provisional event task: A0 trained only new heads, A1 updated the top two FinBERT layers, and A2 trained rank-4 query/value LoRA adapters in those layers. Three fixed seeds and three chronological January--February folds were used. Training OOF selected **A1** before March/April extraction evaluation or any September-onward stock result was read.

The coverage-limited oracle diagnostic found only **15** event windows, **12** nonduplicate report groups, and **14** nonduplicate facts in the 765 stock-train OOF rows. On the single available April forward month it changed **0** directions and moved Brier from 0.26882 to 0.26910. This is not repeatable evidence that the fields improve four-hour prediction.

The event correction is strictly gated. Every row without a validated event was checked for bit-for-bit equality with F1. Residual structure and C were selected globally from March--August forward OOF only; development and later were not used.

## What was trained

- A0 trainable parameters: 9,228.
- A1 trainable parameters: 14,184,972.
- A2 trainable parameters: 33,804.
- The model predicts type-specific evidence for `rating` and `target_price` plus a five-way action for each type. Article event existence is the logical OR of selected type-specific evidence; FinBERT does not generate JSON or numbers.
- Official completed runs: 36; summed per-fit time 1.58 hours; maximum checkpoint reload difference 0.
- Modern FinBERT is being applied retrospectively to 2018 text. Model weights and cached tensors remain private and are not in Git.

## Extraction results on the provisional April check set

| Adapter | Symbol | Positive / predicted-positive articles | Event P/R/F1 | Type macro-F1 | Action macro-F1 | Evidence F1 | No-event FP |
|---|---|---:|---|---:|---:|---:|---:|
| A0 | AAPL | 17 / 0 | 0.00% / 0.00% / 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| A0 | ALL | 22 / 0 | 0.00% / 0.00% / 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| A0 | AMZN | 5 / 0 | 0.00% / 0.00% / 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| A1 | AAPL | 17 / 18 | 77.78% / 82.35% / 80.00% | 70.18% | 46.68% | 52.83% | 7.14% |
| A1 | ALL | 22 / 22 | 81.82% / 81.82% / 81.82% | 74.07% | 59.88% | 57.97% | 4.00% |
| A1 | AMZN | 5 / 4 | 100.00% / 80.00% / 88.89% | 83.33% | 33.33% | 75.00% | 0.00% |
| A2 | AAPL | 17 / 22 | 72.73% / 94.12% / 82.05% | 59.94% | 39.91% | 42.22% | 10.71% |
| A2 | ALL | 22 / 26 | 76.92% / 90.91% / 83.33% | 63.87% | 47.22% | 46.30% | 6.00% |
| A2 | AMZN | 5 / 4 | 100.00% / 80.00% / 88.89% | 75.00% | 30.00% | 66.67% | 0.00% |

The table above evaluates the hierarchy that FinBERT actually predicts. The next table uses the same complete-fact signature, including deterministically copied old/new values, for FinBERT and all prior systems.

| Extractor | Symbol | Exact full-fact TP/FP/FN | Precision | Recall | F1 | Predicted nonduplicate groups |
|---|---|---:|---:|---:|---:|---:|
| FinBERT A0 | AAPL | 0/0/17 | NA | 0.00% | 0.00% | 0 |
| FinBERT A0 | ALL | 0/0/23 | NA | 0.00% | 0.00% | 0 |
| FinBERT A0 | AMZN | 0/0/6 | NA | 0.00% | 0.00% | 0 |
| FinBERT A1 | AAPL | 14/6/3 | 70.00% | 82.35% | 75.68% | 18 |
| FinBERT A1 | ALL | 19/6/4 | 76.00% | 82.61% | 79.17% | 22 |
| FinBERT A1 | AMZN | 5/0/1 | 100.00% | 83.33% | 90.91% | 4 |
| FinBERT A2 | AAPL | 14/29/3 | 32.56% | 82.35% | 46.67% | 20 |
| FinBERT A2 | ALL | 19/29/4 | 39.58% | 82.61% | 53.52% | 24 |
| FinBERT A2 | AMZN | 5/0/1 | 100.00% | 83.33% | 90.91% | 4 |
| Qwen3-1.7B QLoRA | AAPL | 5/7/12 | 41.67% | 29.41% | 34.48% | unavailable |
| Qwen3-1.7B QLoRA | ALL | 6/8/17 | 42.86% | 26.09% | 32.43% | unavailable |
| Qwen3-1.7B QLoRA | AMZN | 1/1/5 | 50.00% | 16.67% | 25.00% | unavailable |
| Current deterministic rules (D2) | AAPL | 14/148/3 | 8.64% | 82.35% | 15.64% | 45 |
| Current deterministic rules (D2) | ALL | 18/180/5 | 9.09% | 78.26% | 16.29% | 73 |
| Current deterministic rules (D2) | AMZN | 4/32/2 | 11.11% | 66.67% | 19.05% | 28 |
| Qwen3.5-9B staged | AAPL | 4/10/13 | 28.57% | 23.53% | 25.81% | unavailable |
| Qwen3.5-9B staged | ALL | 9/15/14 | 37.50% | 39.13% | 38.30% | unavailable |
| Qwen3.5-9B staged | AMZN | 5/5/1 | 50.00% | 83.33% | 62.50% | unavailable |

Given provisional evidence that explicitly names the target company, the deterministic value copier exactly reproduced old/new fields for 60/60 rating facts and 52/53 target-price facts. The one recorded target-price failure needs a preceding sentence to bind the company to a later numeric sentence and is rejected by the strict deployed gate. These are regression diagnostics on a fully exposed provisional panel, not held-out parser estimates; they never select the FinBERT epoch, threshold, or adapter.

No result constitutes formal extraction acceptance because independent human review is still absent.

## Residual parameterization selected from stock-train OOF

- **D2 / rules**: training OOF chose `BASE` with C=None; event windows AAPL=340, AMZN=83; independent eligibility AAPL=True, AMZN=True.
- **D3 / A0_ensemble**: training OOF chose `BASE` with C=None; event windows AAPL=4, AMZN=0; independent eligibility AAPL=False, AMZN=False.
- **D4 / A1_ensemble**: training OOF chose `BASE` with C=None; event windows AAPL=160, AMZN=10; independent eligibility AAPL=True, AMZN=False.

| D4 residual | C | Train mean-monthly BA | Train mean-monthly Brier | Brier guardrail |
|---|---:|---:|---:|---:|
| BASE | NA | 52.14% | 0.2601 | pass |
| C0_shared | 0.01 | 51.14% | 0.2621 | pass |
| C1_independent | 0.01 | 51.27% | 0.2609 | pass |
| C2_partial_shared | 0.01 | 51.14% | 0.2621 | pass |

For D4, C=0.01 was the only non-base strength that passed the Brier guardrail. C1 caused the least damage, but its BA remained below BASE; C0 and C2 were nearly identical because AMZN had too few accepted event windows for a reliable stock-specific adjustment. The protocol therefore selected no residual correction. D2 had no non-base candidate inside the Brier guardrail, while D3 had too little coverage to change a direction.

C1 is disabled for any stock below 30 unique accepted events or 60 event windows. C2 retains shared event effects and adds one strongly shrunk AMZN event-window offset; selection is global, never per-stock after seeing later results.

## Four-hour comparison

BA, MCC, and Brier are shown together. Train OOF is March--August chronological forward prediction; development is September--October; later is November onward and already exposed.

| Method | AAPL train OOF BA / MCC / Brier | AAPL development | AAPL later | AMZN train OOF | AMZN development | AMZN later |
|---|---|---|---|---|---|---|
| F0 | 50.02% / +0.000 / 0.2650 | 50.46% / +0.013 / 0.2601 | 51.89% / +0.052 / 0.2694 | 51.87% / +0.039 / 0.2578 | 48.33% / -0.033 / 0.2710 | 49.79% / -0.004 / 0.2677 |
| R1 | 52.68% / +0.053 / 0.2577 | 56.21% / +0.149 / 0.2441 | 53.87% / +0.086 / 0.2558 | 52.18% / +0.046 / 0.2562 | 51.11% / +0.022 / 0.2719 | 50.85% / +0.017 / 0.2722 |
| F1 | 50.30% / +0.006 / 0.2593 | 57.94% / +0.195 / 0.2500 | 51.74% / +0.042 / 0.2587 | 53.27% / +0.065 / 0.2606 | 55.66% / +0.112 / 0.2481 | 54.36% / +0.087 / 0.2583 |
| F2 | 50.02% / +0.000 / 0.2602 | 54.13% / +0.107 / 0.2803 | 56.71% / +0.152 / 0.2856 | 48.14% / -0.038 / 0.2749 | 46.29% / -0.073 / 0.2977 | 54.27% / +0.087 / 0.2617 |
| F6 | 55.34% / +0.110 / 0.2452 | 50.35% / +0.015 / 0.2573 | 49.39% / -0.014 / 0.2529 | 55.14% / +0.110 / 0.2520 | 52.50% / +0.051 / 0.2652 | 51.16% / +0.024 / 0.2545 |
| D2 | 50.30% / +0.006 / 0.2593 | 57.94% / +0.195 / 0.2500 | 51.74% / +0.042 / 0.2587 | 53.27% / +0.065 / 0.2606 | 55.66% / +0.112 / 0.2481 | 54.36% / +0.087 / 0.2583 |
| D3 | 50.30% / +0.006 / 0.2593 | 57.94% / +0.195 / 0.2500 | 51.74% / +0.042 / 0.2587 | 53.27% / +0.065 / 0.2606 | 55.66% / +0.112 / 0.2481 | 54.36% / +0.087 / 0.2583 |
| D4 | 50.30% / +0.006 / 0.2593 | 57.94% / +0.195 / 0.2500 | 51.74% / +0.042 / 0.2587 | 53.27% / +0.065 / 0.2606 | 55.66% / +0.112 / 0.2481 | 54.36% / +0.087 / 0.2583 |

Method names: F0=price+title baseline, R1=recent price, F1=price+full text with R1 fallback, F2=price+frozen FinBERT, F6=previous training-selected fusion, D2=F1+rule facts, D3=F1+A0 facts, D4=F1+the training-selected adapter facts.

Constant-direction check: No whole stock-period-method slice made a constant direction prediction. `metrics.csv` also records each slice's predicted-up proportion.

## Event coverage and direction changes

| Method | Period / stock | Event / no-event windows | Changed right / wrong |
|---|---|---:|---:|
| D2 | train_forward_oof / AAPL | 340 / 43 | 0 / 0 |
| D2 | train_forward_oof / AMZN | 83 / 299 | 0 / 0 |
| D2 | development / AAPL | 102 / 24 | 0 / 0 |
| D2 | development / AMZN | 14 / 112 | 0 / 0 |
| D2 | later / AAPL | 153 / 25 | 0 / 0 |
| D2 | later / AMZN | 15 / 164 | 0 / 0 |
| D3 | train_forward_oof / AAPL | 4 / 379 | 0 / 0 |
| D3 | train_forward_oof / AMZN | 0 / 382 | 0 / 0 |
| D3 | development / AAPL | 0 / 126 | 0 / 0 |
| D3 | development / AMZN | 0 / 126 | 0 / 0 |
| D3 | later / AAPL | 0 / 178 | 0 / 0 |
| D3 | later / AMZN | 0 / 179 | 0 / 0 |
| D4 | train_forward_oof / AAPL | 160 / 223 | 0 / 0 |
| D4 | train_forward_oof / AMZN | 10 / 372 | 0 / 0 |
| D4 | development / AAPL | 86 / 40 | 0 / 0 |
| D4 | development / AMZN | 8 / 118 | 0 / 0 |
| D4 | later / AAPL | 115 / 63 | 0 / 0 |
| D4 | later / AMZN | 5 / 174 | 0 / 0 |

## Adapter seed variation

| Method | Period / stock | BA mean +/- SD | MCC mean +/- SD | Brier mean +/- SD |
|---|---|---:|---:|---:|
| D3 | development / AAPL | 57.94% +/- 0.00 pp | +0.195 +/- 0.000 | 0.2500 +/- 0.0000 |
| D3 | development / AMZN | 55.66% +/- 0.00 pp | +0.112 +/- 0.000 | 0.2481 +/- 0.0000 |
| D3 | later / AAPL | 51.74% +/- 0.00 pp | +0.042 +/- 0.000 | 0.2587 +/- 0.0000 |
| D3 | later / AMZN | 54.36% +/- 0.00 pp | +0.087 +/- 0.000 | 0.2583 +/- 0.0000 |
| D3 | train_forward_oof / AAPL | 50.30% +/- 0.00 pp | +0.006 +/- 0.000 | 0.2593 +/- 0.0000 |
| D3 | train_forward_oof / AMZN | 53.27% +/- 0.00 pp | +0.065 +/- 0.000 | 0.2606 +/- 0.0000 |
| D4 | development / AAPL | 57.94% +/- 0.00 pp | +0.195 +/- 0.000 | 0.2500 +/- 0.0000 |
| D4 | development / AMZN | 55.66% +/- 0.00 pp | +0.112 +/- 0.000 | 0.2481 +/- 0.0000 |
| D4 | later / AAPL | 51.74% +/- 0.00 pp | +0.042 +/- 0.000 | 0.2587 +/- 0.0000 |
| D4 | later / AMZN | 54.36% +/- 0.00 pp | +0.087 +/- 0.000 | 0.2583 +/- 0.0000 |
| D4 | train_forward_oof / AAPL | 50.30% +/- 0.00 pp | +0.006 +/- 0.000 | 0.2593 +/- 0.0000 |
| D4 | train_forward_oof / AMZN | 53.27% +/- 0.00 pp | +0.065 +/- 0.000 | 0.2606 +/- 0.0000 |

## Exposed later paired intervals

| Method | Stock | Block | BA-difference interval vs F1 | Brier-difference interval vs F1 |
|---|---|---:|---:|---:|
| D2 | AAPL | 1 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D2 | AAPL | 5 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D2 | AMZN | 1 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D2 | AMZN | 5 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D3 | AAPL | 1 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D3 | AAPL | 5 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D3 | AMZN | 1 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D3 | AMZN | 5 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D4 | AAPL | 1 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D4 | AAPL | 5 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D4 | AMZN | 1 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |
| D4 | AMZN | 5 day | [+0.00, +0.00] pp | [+0.0000, +0.0000] |

The interval table gives paired day and five-day block bootstrap intervals. The complete train-OOF, development, and later table is in `paired_intervals.csv`; repeated exploration means these are sensitivity summaries rather than unselected significance tests.

## How to read the winners

- **Highest exposed later AAPL BA:** F2 at 56.71% (Brier 0.2856).
- **Highest exposed later AMZN BA:** F1 at 54.36% (Brier 0.2583).
- **Highest later two-stock descriptive mean:** F2 at 55.49%. It did not participate in selection and later is exposed.
- **Cross-period stability diagnostic:** F1 has the highest worst BA across the four development/later stock cells (51.74%). This is a descriptive robustness summary, not a new selection rule.
- **Training-protocol choice:** extraction chose A1; residual choices are listed above. Those are the only choices that can be called protocol-selected in this run.

## Observations and explanations

Observed: the audit verified all 458 source mappings, but only 57 labeled article-target pairs enter any fixed window and only 15 labeled event windows overlap stock-train OOF. Observed: AMZN has only 13 positive training articles in the extraction set. Observed: no-event residual rows are exactly unchanged.

Interpretation: sparse event supervision limits both adapter reliability and the residual model's ability to learn a stable market response. A better extraction score can therefore coexist with no four-hour BA gain. This is an explanation consistent with the counts, not proof that analyst events contain no predictive information.

## Files

- `DATA_AUDIT.md`: label fields, source mapping, split and duplicate audit.
- `CASE_NOTES.md`: pre-fixed extraction and prediction cases.
- `metrics.csv`, `monthly_metrics.csv`, `subgroup_metrics.csv`: primary results.
- `predictions.csv`: all matched probabilities without raw article text.
- `paired_intervals.csv`, `seed_variation.csv`: uncertainty and seed sensitivity.
- `training_evidence.json`: device, trainable counts, checkpoint reload and hashes.
