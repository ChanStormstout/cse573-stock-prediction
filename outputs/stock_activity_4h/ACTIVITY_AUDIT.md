# Activity v1 label-free audit

Status: **SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY**. No authoritative vendor definition was found locally. Descriptive correlations do not establish field meaning or predictive value.

## Raw field summary

| symbol   |   raw_rows | source_sha256                                                    | dtype   |   finite_count |   missing_count |   negative_count |   zero_count |   min |   max |    mean |   median |   q01 |   q05 |   q25 |   q75 |   q95 |   q99 |   unique_count |   corr_abs_return |   corr_high_low_range |
|:---------|-----------:|:-----------------------------------------------------------------|:--------|---------------:|----------------:|-----------------:|-------------:|------:|------:|--------:|---------:|------:|------:|------:|------:|------:|------:|---------------:|------------------:|----------------------:|
| AAPL     |      38634 | 059a29b0b24f438bd2d6f1be4102974f9b10631fd69b6c75ac2076930b0381cf | int64   |          38634 |               0 |                0 |            0 |     1 |  1663 | 607.472 |      702 |     2 |     8 |   471 |   850 |  1018 |  1125 |           1237 |          0.305082 |              0.4416   |
| AMZN     |      30283 | ea5ec298fe39364fa3cb3d9dfdd46393c5490f46c9e2da7984f839953e380282 | int64   |          30283 |               0 |                0 |            0 |     1 |  1093 | 318.29  |      309 |     3 |    11 |   162 |   466 |   679 |   815 |            955 |          0.454078 |              0.585213 |

## Cross-granularity check

| symbol   |   minutes |   complete_windows |   exact_sum_match_count |   exact_sum_match_fraction |   max_abs_sum_discrepancy |   mean_abs_sum_discrepancy |   mean_abs_error_vs_5m_mean | status                              |
|:---------|----------:|-------------------:|------------------------:|---------------------------:|--------------------------:|---------------------------:|----------------------------:|:------------------------------------|
| AAPL     |        15 |              12793 |                   12792 |                   0.999922 |                        44 |                 0.00343938 |                    1222.68  | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AAPL     |        30 |               6346 |                    6346 |                   1        |                         0 |                 0          |                    3080.13  | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AAPL     |        60 |               2994 |                    2994 |                   1        |                         0 |                 0          |                    6993.01  | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AMZN     |        15 |              10045 |                   10045 |                   1        |                         0 |                 0          |                     639.402 | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AMZN     |        30 |               4998 |                    4998 |                   1        |                         0 |                 0          |                    1606.1   | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AMZN     |        60 |               2359 |                    2359 |                   1        |                         0 |                 0          |                    3637.95  | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |

The raw 15/30/60 files exist, but their aggregation comparison is descriptive only and does not identify the opaque field as volume or any other semantic quantity.
