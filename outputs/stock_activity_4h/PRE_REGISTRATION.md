# Activity Incremental Experiment v1

Status: **PREREGISTERED_NOT_RUN**. This is a separate four-hour AAPL/AMZN
experiment. It does not modify or extend Phase B, news models, routers, or
market-regime models.

## Frozen comparison

- **A0:** exact canonical per-stock R1 recent-price feature family from
  `outputs/stock_nextgen_4h/recent_price.py`.
- **A1:** A0 plus exactly the eight `activity_*` fields below.
- **A1_matchedC:** A1 fields with the A0-selected C from the same fold. It is an
  attribution control, never a separately selected winner.

The seventh raw bar field remains **activity**: an opaque per-five-minute-bar
field with unresolved vendor semantics. This protocol does not call it volume,
tick count, order flow, liquidity, or transaction count.

```python
ACTIVITY_HORIZONS = (15, 60)
ACTIVITY_FEATURES = [
    "activity_logmean_15", "activity_logmean_60",
    "activity_rel_15", "activity_rel_60",
    "activity_missing_15", "activity_missing_60",
    "activity_rel_missing_15", "activity_rel_missing_60",
]
HISTORY_SESSIONS = 20
MIN_HISTORY_SESSIONS = 10
```

For a contiguous current-session completed-bar window, `activity_logmean_h` is
`log1p(mean(activity))`. A row-local relative feature subtracts the median of
the same h-minute, same-intraday-offset logmeans from up to 20 strictly prior
XNYS sessions; it is missing unless at least 10 valid references exist.

## Frozen learning protocol

Each stock is fitted separately. March--August 2018 are chronological forward
folds; September--October and November onward are exposed exploratory historical
backtests only. Training requires `train.end_utc.max() <
evaluate.cutoff_utc.min()`. Both A0 and A1 choose C in `(0.01, 0.1, 1.0)` from
previous forward months by mean BA, then mean Brier, then smaller C. The first
fold uses C=0.1. Models are liblinear logistic regression, seed 573,
`max_iter=3000`, `tol=1e-8`, with training-only StandardizeMissing.

The June--August promotion gate requires six stock-month cells, both stocks'
mean BA gain at least +1pp, positive macro BA in at least two months, each
stock's mean Brier worsening no more than .002, and no constant-direction
collapse. No interaction, extra horizon, external data, router, or news feature
is permitted in v1.

## Execution boundary

Preflight may audit raw data, build private features, reproduce A0, and run
synthetic/verifier checks. `run_activity.py` exits unless invoked with
`--approve-activity-run`. A1 and A1_matchedC have not been scored.
