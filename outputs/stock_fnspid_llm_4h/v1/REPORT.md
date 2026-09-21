# FNSPID-augmented LLM information-control experiment

## Plain-language result

L1-L4 were executed in the registered order. The OOF-only L5 choice is **L3 historical-case vote**. Development and later remain exposed exploratory backtests.

## Actual work

- L1/L2: 3,966 frozen-Qwen binary article scores across 1,322 added FNSPID articles.
- L4: 1,155 time-safe case-conditioned Qwen scores; 219 rows use exact fallback.
- L3: non-LLM similarity-weighted votes using strictly earlier same-stock cases.
- L1 and L2 refit their downstream models chronologically; LLM weights were frozen.
- Independent post-run verification is PASS.

## Results

### train_forward_oof

| Stock | Method | BA | Accuracy | MCC | Brier | AUC |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Recent-price reference | 52.68% | 51.96% | 0.053 | 0.2577 | 0.5412 |
| AAPL | Unfiltered augmented FinBERT | 50.69% | 52.48% | 0.014 | 0.2619 | 0.5020 |
| AAPL | Original-course FinBERT | 50.02% | 50.65% | 0.000 | 0.2602 | 0.5182 |
| AAPL | L1 LLM-filtered FinBERT | 54.82% | 56.66% | 0.097 | 0.2693 | 0.5545 |
| AAPL | L2 LLM fact-change residual | 51.41% | 50.91% | 0.028 | 0.2748 | 0.5034 |
| AAPL | L3 historical-case vote | 53.49% | 53.26% | 0.069 | 0.2574 | 0.5326 |
| AAPL | L4 historical cases + LLM | 52.62% | 53.26% | 0.052 | 0.2551 | 0.5338 |
| AAPL | L5 guarded system | 53.49% | 53.26% | 0.069 | 0.2574 | 0.5326 |
| AMZN | Recent-price reference | 52.18% | 53.14% | 0.046 | 0.2562 | 0.5158 |
| AMZN | Unfiltered augmented FinBERT | 49.39% | 49.48% | -0.012 | 0.2609 | 0.5021 |
| AMZN | Original-course FinBERT | 48.14% | 48.95% | -0.038 | 0.2749 | 0.4639 |
| AMZN | L1 LLM-filtered FinBERT | 52.21% | 53.14% | 0.046 | 0.2585 | 0.5343 |
| AMZN | L2 LLM fact-change residual | 51.13% | 52.09% | 0.024 | 0.2675 | 0.4875 |
| AMZN | L3 historical-case vote | 54.87% | 56.28% | 0.109 | 0.2532 | 0.5664 |
| AMZN | L4 historical cases + LLM | 50.39% | 51.83% | 0.009 | 0.2674 | 0.4777 |
| AMZN | L5 guarded system | 54.87% | 56.28% | 0.109 | 0.2532 | 0.5664 |

### development

| Stock | Method | BA | Accuracy | MCC | Brier | AUC |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Recent-price reference | 56.21% | 53.97% | 0.149 | 0.2441 | 0.6435 |
| AAPL | Unfiltered augmented FinBERT | 57.71% | 54.76% | 0.225 | 0.2487 | 0.6463 |
| AAPL | Original-course FinBERT | 54.13% | 51.59% | 0.107 | 0.2803 | 0.6197 |
| AAPL | L1 LLM-filtered FinBERT | 55.81% | 54.76% | 0.120 | 0.2550 | 0.6136 |
| AAPL | L2 LLM fact-change residual | 50.43% | 48.41% | 0.010 | 0.2567 | 0.5649 |
| AAPL | L3 historical-case vote | 58.77% | 57.14% | 0.191 | 0.2441 | 0.6313 |
| AAPL | L4 historical cases + LLM | 55.25% | 52.38% | 0.150 | 0.2579 | 0.5923 |
| AAPL | L5 guarded system | 58.77% | 57.14% | 0.191 | 0.2441 | 0.6313 |
| AMZN | Recent-price reference | 51.11% | 48.41% | 0.022 | 0.2719 | 0.5553 |
| AMZN | Unfiltered augmented FinBERT | 58.53% | 57.94% | 0.166 | 0.2436 | 0.6361 |
| AMZN | Original-course FinBERT | 46.29% | 45.24% | -0.073 | 0.2977 | 0.5030 |
| AMZN | L1 LLM-filtered FinBERT | 53.25% | 52.38% | 0.063 | 0.2478 | 0.5852 |
| AMZN | L2 LLM fact-change residual | 49.44% | 46.83% | -0.011 | 0.2662 | 0.5712 |
| AMZN | L3 historical-case vote | 54.92% | 50.79% | 0.102 | 0.2760 | 0.5762 |
| AMZN | L4 historical cases + LLM | 51.11% | 48.41% | 0.022 | 0.2719 | 0.5553 |
| AMZN | L5 guarded system | 54.92% | 50.79% | 0.102 | 0.2760 | 0.5762 |

### later

| Stock | Method | BA | Accuracy | MCC | Brier | AUC |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Recent-price reference | 53.87% | 53.37% | 0.086 | 0.2558 | 0.5301 |
| AAPL | Unfiltered augmented FinBERT | 53.84% | 53.37% | 0.084 | 0.2583 | 0.5340 |
| AAPL | Original-course FinBERT | 56.71% | 56.18% | 0.152 | 0.2856 | 0.5496 |
| AAPL | L1 LLM-filtered FinBERT | 53.37% | 52.81% | 0.077 | 0.2780 | 0.5583 |
| AAPL | L2 LLM fact-change residual | 48.21% | 48.31% | -0.036 | 0.2633 | 0.4992 |
| AAPL | L3 historical-case vote | 47.59% | 47.19% | -0.052 | 0.2597 | 0.5147 |
| AAPL | L4 historical cases + LLM | 48.49% | 48.31% | -0.031 | 0.2595 | 0.5059 |
| AAPL | L5 guarded system | 47.59% | 47.19% | -0.052 | 0.2597 | 0.5147 |
| AMZN | Recent-price reference | 50.85% | 51.40% | 0.017 | 0.2722 | 0.5200 |
| AMZN | Unfiltered augmented FinBERT | 48.61% | 49.16% | -0.028 | 0.2657 | 0.4748 |
| AMZN | Original-course FinBERT | 54.27% | 54.75% | 0.087 | 0.2617 | 0.5686 |
| AMZN | L1 LLM-filtered FinBERT | 47.66% | 48.04% | -0.047 | 0.2672 | 0.4685 |
| AMZN | L2 LLM fact-change residual | 49.62% | 50.28% | -0.008 | 0.2750 | 0.5104 |
| AMZN | L3 historical-case vote | 48.08% | 48.60% | -0.039 | 0.2815 | 0.5015 |
| AMZN | L4 historical cases + LLM | 50.85% | 51.40% | 0.017 | 0.2722 | 0.5200 |
| AMZN | L5 guarded system | 48.08% | 48.60% | -0.039 | 0.2815 | 0.5015 |

## L5 training-period gate

| Component | Macro ΔBA | AAPL ΔBA | AMZN ΔBA | Positive months | Max ΔBrier | Pass |
|---|---:|---:|---:|---:|---:|---:|
| L1 LLM-filtered FinBERT | +1.08 pp | +2.13 pp | +0.03 pp | 4/6 | +0.0117 | False |
| L2 LLM fact-change residual | -1.16 pp | -1.27 pp | -1.05 pp | 1/6 | +0.0171 | False |
| L3 historical-case vote | +1.75 pp | +0.81 pp | +2.69 pp | 4/6 | -0.0003 | True |
| L4 historical cases + LLM | -0.93 pp | -0.06 pp | -1.79 pp | 1/6 | +0.0112 | False |

## Coverage and evidence boundary

L1 retained added FNSPID evidence in 1081/1607 canonical windows. L2 strictly equals the price model in every row without accepted FNSPID evidence.

The LLM scores are frozen-model judgments, not independently reviewed gold labels. FNSPID entity links also remain without completed independent semantic review, and source timestamps are date-only. Scores therefore show how this fixed pipeline behaved, not that the LLM classifications are objectively correct or that the result will generalize to a fresh market period.
