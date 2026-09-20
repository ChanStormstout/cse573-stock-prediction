# Mechanical alias contamination diagnostics

These source-only flags are not semantic labels or measured precision. The census contains **3,129** alias/CIK rows; **1,606** have at least one mechanical risk flag. The global top-1% edge-count threshold is **32,817**. `training_day_coverage` is a bounded source diagnostic: distinct 2018–2021 recorded dates divided by that stock's trading-day denominator and capped at one; the historical panel retains the stricter effective-session coverage calculation.

NDAQ's `nasdaq` alias produced **2,576,470** all-years matched edges, including **1,315,640** all-years currently high-confidence edges and **255,784** high-confidence edges during 2018–2021. Its bounded training-day coverage diagnostic is **1.000**, publisher count is **169**, and native-tag agreement is **0.0018**. In the 100-card sentinel sample, **3** cards match a listing/venue phrase and **15** match a narrow corporate-context phrase; overlap is possible. This confirms a material contamination risk but cannot establish precision.

## Highest mechanical-risk aliases

- `NDAQ` / `nasdaq`: 2,576,470 matches; `DUAL_ROLE_EXCHANGE_COMPANY_SENTINEL|EDGE_COUNT_TOP_1_PERCENT|EXCHANGE_OR_VENUE_NAME|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `AAPL` / `apple`: 271,000 matches; `COMMON_WORD_PRODUCT_OR_CATEGORY|EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `DXPE` / `index`: 936,956 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `MSTR` / `strategy`: 417,080 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `FIVE` / `cheap`: 227,886 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `MSFT` / `microsoft`: 190,673 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `JYNT` / `joint`: 124,437 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `TBBK` / `bancorp`: 123,851 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `MTCH` / `match`: 117,545 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `C` / `citigroup`: 116,239 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `TSLA` / `tesla`: 108,328 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `GOOG` / `alphabet`: 93,621 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `INTC` / `intel`: 84,354 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `BA` / `boeing`: 79,637 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`
- `VNCE` / `apparel`: 74,045 matches; `EDGE_COUNT_TOP_1_PERCENT|IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE|SINGLE_TOKEN_SEMANTIC_RISK`

Top risk aliases are recorded in `ALIAS_RISK_CENSUS.csv`. Risk ranking uses only alias form, source concentration, edge volume, and training-period coverage. No market outcome was accessed. The `nasdaq` alias remains pending independent review; no alias rule or complete relation ledger was changed in this stage.
