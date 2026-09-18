# Recency weighting, dense windows, and reaction audit — v3

## Executive result

This is a bounded four-hour experiment on the original AAPL/AMZN windows. The equal-weight R1/F1/F2 columns reproduce the previously saved official probabilities exactly; the new recency columns are the only changed prediction mechanism. No method passed the registered cross-month promotion line. The article branch is an audit only and no reaction predictor was trained.

## What was actually run

- Three real weighted logistic-regression pipelines: R1 recent price, F1 price plus binary full-text features, and F2 price plus frozen FinBERT PCA features.
- Four exact half-lives: infinity, 80, 40, and 20 NYSE sessions. A single half-life was selected per input across both stocks using March–May forward OOF.
- 30-minute-stride four-hour windows from raw regular-session five-minute bars. D0 equal, D0 day-normalized, and D1 augmented day-normalized were evaluated on official rows.
- A descriptive article reaction audit at 30/60/120/240 minutes under same-session and trading-time definitions.
- A descriptive audit of the seventh raw-bar `activity` column; it was excluded from every model.

## Recency selection and outer gate

| input | selected half-life | AAPL ΔBA | AMZN ΔBA | macro ΔBA | positive outer months | pass |
|---|---:|---:|---:|---:|---:|---|
| R1 | 20 | 0.41 pp | -7.40 pp | -3.49 pp | 1/3 | no |
| F1 | 80 | 4.12 pp | -3.21 pp | 0.45 pp | 1/3 | no |
| F2 | 20 | 5.97 pp | -4.31 pp | 0.83 pp | 2/3 | no |

The registered line required the weaker stock to gain at least 1 pp, no stock to lose more than 1 pp, and positive macro BA in at least two of June, July, and August. Recency did not satisfy it. F1 and F2 improved AAPL in this exposed outer interval but lost AMZN; R1 also failed the weaker-stock rule.

## Development and later results

| period | stock | method | BA | MCC | Brier |
|---|---|---|---:|---:|---:|
| development | AAPL | R1_equal | 56.21% | 0.149 | 0.2441 |
| development | AAPL | R1_recency | 50.25% | 0.005 | 0.2635 |
| development | AAPL | F1_equal | 57.94% | 0.195 | 0.2500 |
| development | AAPL | F1_recency | 53.27% | 0.083 | 0.2499 |
| development | AAPL | F2_equal | 54.13% | 0.107 | 0.2803 |
| development | AAPL | F2_recency | 51.17% | 0.027 | 0.3187 |
| development | AMZN | R1_equal | 51.11% | 0.022 | 0.2719 |
| development | AMZN | R1_recency | 56.31% | 0.125 | 0.2482 |
| development | AMZN | F1_equal | 55.66% | 0.112 | 0.2481 |
| development | AMZN | F1_recency | 51.48% | 0.030 | 0.2420 |
| development | AMZN | F2_equal | 46.29% | -0.073 | 0.2977 |
| development | AMZN | F2_recency | 45.27% | -0.094 | 0.3092 |
| later | AAPL | R1_equal | 53.87% | 0.086 | 0.2558 |
| later | AAPL | R1_recency | 50.49% | 0.010 | 0.2813 |
| later | AAPL | F1_equal | 51.74% | 0.042 | 0.2587 |
| later | AAPL | F1_recency | 52.89% | 0.071 | 0.2544 |
| later | AAPL | F2_equal | 56.71% | 0.152 | 0.2856 |
| later | AAPL | F2_recency | 54.62% | 0.116 | 0.3240 |
| later | AMZN | R1_equal | 50.85% | 0.017 | 0.2722 |
| later | AMZN | R1_recency | 46.18% | -0.082 | 0.2806 |
| later | AMZN | F1_equal | 54.36% | 0.087 | 0.2583 |
| later | AMZN | F1_recency | 52.40% | 0.049 | 0.2545 |
| later | AMZN | F2_equal | 54.27% | 0.087 | 0.2617 |
| later | AMZN | F2_recency | 47.67% | -0.049 | 0.3032 |

The equal rows are the official historical reference. Development and later are already exposed and were not used to choose the half-life. A high later score is therefore descriptive evidence only.

## Dense-window result

The dense builder produced 2012 unique training-window rows, including 998 official rows. Relative to the day-normalized official control D0_day, D1 changed the June–August BA by 0.42 pp for AAPL and 0.78 pp for AMZN. This was below the registered 1 pp per-stock threshold, so the text extension was stopped. D2 was RUN_NOT_RUN_RECENCY_GATE_FAILED because the R1 recency gate failed.

## Article reaction audit

The audit used only accepted article IDs already attached to official windows, retained raw occurrences and deterministic normalized online groups, and measured first complete bars after `available_utc`. It produced 39440 article–horizon rows from 4930 accepted articles and 4930 normalized groups. The result is feasible for a future reaction experiment under the operational 100-group coverage check: True. This is not evidence of causality, and no reaction model was trained.

See `article_reaction_audit.csv` for stock, horizon, definition, period, coverage, sign balance, duplicate concentration, and return magnitude. The audit deliberately does not turn a post-availability return into a news causal effect.

## Activity column

Status is `SEMANTICS_UNRESOLVED_NOT_USED`. The files contain a nonnegative integer-like seventh column, but no authoritative definition or units were found in the searched repository metadata. It remains unused. The question for the professor/data provider is recorded in `ACTIVITY_AUDIT.md`.

## Interpretation

1. Recency weighting can alter the model's learned prior, but the effect is not stable across the two stocks and exposed months.
2. Dense windows add more highly overlapping samples. Their small positive shift does not meet the registered gate and cannot be treated as independent-data evidence.
3. The reaction audit can tell us whether a future event-study branch has enough timestamped coverage; it cannot prove that the observed move was caused by the article.
4. The remaining bottleneck is not solved by this run. A verified contemporaneous market-data source is still required before the market-state branch can be tested fairly.

## Reproduction

```text
work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/run_recency.py --out outputs/stock_recency_dense_4h/v2
work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/build_dense_windows.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3
work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/run_dense.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3
work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/audit_article_reactions.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3/reactions2
work/stock-data/finbert-env/bin/python outputs/stock_recency_dense_4h/audit_activity.py --public outputs/stock_recency_dense_4h/v3 --private work/stock-data/recency_dense_4h/v3/activity
```

All public result files are aggregate metrics, official keys, and audit summaries. Raw news, cached embeddings, and model binaries remain under `work/`.
