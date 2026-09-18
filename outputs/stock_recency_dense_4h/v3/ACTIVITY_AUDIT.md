# Activity-column audit

Status: **SEMANTICS_UNRESOLVED_NOT_USED**

The raw five-minute files contain a seventh numeric column named `activity`. The repository documentation and audit metadata searched by this run do not provide an authoritative vendor definition. The column was not included in any recency, dense-window, or reaction experiment.

## Descriptive statistics

| symbol   |   rows | dtype   | nonnegative   |   negative_count |   zero_count |   zero_fraction |   min |   max |   median |   q01 |   q99 |   unique_count |    mean |     std |   corr_abs_return |   corr_range | likely_integer   | source_sha256                                                    |
|:---------|-------:|:--------|:--------------|-----------------:|-------------:|----------------:|------:|------:|---------:|------:|------:|---------------:|--------:|--------:|------------------:|-------------:|:-----------------|:-----------------------------------------------------------------|
| AAPL     |  38634 | int64   | True          |                0 |            0 |               0 |     1 |  1663 |      702 |     2 |  1125 |           1237 | 607.472 | 335.708 |          0.305082 |     0.4416   | True             | 059a29b0b24f438bd2d6f1be4102974f9b10631fd69b6c75ac2076930b0381cf |
| AMZN     |  30283 | int64   | True          |                0 |            0 |               0 |     1 |  1093 |      309 |     3 |   815 |            955 | 318.29  | 209.751 |          0.454078 |     0.585213 | True             | ea5ec298fe39364fa3cb3d9dfdd46393c5490f46c9e2da7984f839953e380282 |

## Definition search

Files containing activity/volume-related wording:
- `/Users/victor/Documents/Codex/2026-09-14/wox/work/stock-data/audit/profile.json`
- `/Users/victor/Documents/Codex/2026-09-14/wox/work/stock-data/audit/receipt.json`

## Question for the professor/data provider

Could you confirm what the seventh column named activity in APPLE5.csv and AMAZON5.csv represents (trade volume, tick count, or another vendor field), its units, and its adjustment/aggregation semantics?

The correlations in `activity_summary.csv` are descriptive associations only; they are not evidence that the column is volume or a usable predictive feature.
