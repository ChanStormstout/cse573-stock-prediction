# Activity v1 label-free audit

Status: **SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY**. No authoritative vendor definition was found locally. Descriptive correlations do not establish field meaning or predictive value.

## Raw field summary

| symbol   |   raw_rows | source_sha256                                                    | dtype   |   finite_count |   missing_count |   negative_count |   zero_count |   min |   max |    mean |   median |   q01 |   q05 |   q25 |   q75 |   q95 |   q99 |   unique_count |   corr_abs_return |   corr_high_low_range |
|:---------|-----------:|:-----------------------------------------------------------------|:--------|---------------:|----------------:|-----------------:|-------------:|------:|------:|--------:|---------:|------:|------:|------:|------:|------:|------:|---------------:|------------------:|----------------------:|
| AAPL     |      38634 | 059a29b0b24f438bd2d6f1be4102974f9b10631fd69b6c75ac2076930b0381cf | int64   |          38634 |               0 |                0 |            0 |     1 |  1663 | 607.472 |      702 |     2 |     8 |   471 |   850 |  1018 |  1125 |           1237 |          0.305082 |              0.4416   |
| AMZN     |      30283 | ea5ec298fe39364fa3cb3d9dfdd46393c5490f46c9e2da7984f839953e380282 | int64   |          30283 |               0 |                0 |            0 |     1 |  1093 | 318.29  |      309 |     3 |    11 |   162 |   466 |   679 |   815 |            955 |          0.454078 |              0.585213 |

## Cross-granularity check

| symbol   |   minutes |   complete_windows |   mean_abs_error_vs_5m_mean |   mean_abs_error_vs_5m_sum | status                              |
|:---------|----------:|-------------------:|----------------------------:|---------------------------:|:------------------------------------|
| AAPL     |        15 |              12793 |                    1222.68  |                 0.00343938 | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AAPL     |        30 |               6346 |                    3080.13  |                 0          | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AAPL     |        60 |               2994 |                    6993.01  |                 0          | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AMZN     |        15 |              10045 |                     639.402 |                 0          | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AMZN     |        30 |               4998 |                    1606.1   |                 0          | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |
| AMZN     |        60 |               2359 |                    3637.95  |                 0          | DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF |

The raw 15/30/60 files exist, but their aggregation comparison is descriptive only and does not identify the opaque field as volume or any other semantic quantity.
