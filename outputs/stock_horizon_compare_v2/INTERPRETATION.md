# Interpretation of the horizon comparison

## What was compared

This independent, exploratory run keeps the original time safety rule and
compares price-only L2 logistic regression with price plus news-title TF-IDF
L2 logistic regression.  The C value is selected only using June--August
forward folds.  September--October and November--February are both already
exposed historical periods, not fresh tests.

| Target | Prediction cutoff | News lookback | Train / development / exposed rows per stock |
|---|---|---:|---:|
| 1 hour | five minutes before target hour | 4 hours | about 1,000 / 252 / 364--365 |
| 4 hours | five minutes before a candidate hour | 4 hours | 499 / 126 / 178--179 |
| next trading day | one minute after prior regular-session close | 72 hours | 162 / 42 / 61 |

The one-hour sample starts exactly reproduce the original experiment's 3,233
sample keys.  Four-hour windows begin at successive eligible hours and hence
overlap.  Their row count is not their number of independent market outcomes.
The daily task has especially small development and exposed samples.

## Result relevant to target choice

No horizon meets the predeclared practical interpretation rule: adding news
titles must improve balanced accuracy over the matched price model for both
stocks in both reported periods without a worse Brier score.

- **One hour:** AMZN news titles improve development balanced accuracy from
  48.87% to 51.58% and Brier from 0.2536 to 0.2523, but reverse in the exposed
  period (50.66% to 50.11%; Brier worsens).  AAPL is also inconsistent.
- **Four hours:** no stock-period pair has a robust news advantage.  AAPL's
  price-only exposed balanced accuracy is 52.92%, but titles lower it to
  51.89% and worsen Brier.  AMZN is below 50% in the exposed period for both.
- **Next day:** AMZN title results rise to 54.19% in the exposed period, but
  development is 43.51% and Brier worsens from 0.2507 to 0.2588.  AAPL shows
  the opposite reversal.  With 42 development and 61 exposed observations,
  these values are too variable to select a new main task.

Therefore this comparison does **not** justify changing the project main
target to the horizon with the highest retrospective score.  It supports
keeping the original one-hour task for continuity and sample size, while
optionally presenting the four-hour/daily outcomes as a robustness study about
time-scale mismatch.  A future target change would need a new pre-specified,
unseen time period before it can be claimed as a stable improvement.
