# Data audit — market/return v1

## Main task input

- Source: `work/stock-data/paper_methods_4h/v1/inputs.pkl`
- Rows: 1,607 AAPL/AMZN four-hour windows.
- Target: `label == (target_return > 0)` for every row; `target_return` is a training target and is not in either feature matrix.
- Evaluated rows: 765 chronological March–August forward OOF, 252 September–October development, and 357 November-onward later. The 233 January–February rows are warmup for the first OOF folds.
- Features: the existing past-only `OLD + RECENT` price columns; the secondary `price_F1` variant appends the previously saved F1 probability. No future price or cutoff metadata is a model feature.
- Duplicate sample keys: none.
- Input SHA-256: `7d354e2553c338cd6a64e084e7cc4550c98fbce2dd5b3004c0829eef932d1d5b`.

## External market-state data

The registered audit requested SPY and QQQ one-minute regular-session bars for four fixed 2018 dates. Eight probes were issued through the Alpaca endpoint with `feed=iex`, `timeframe=1Min`, and 14:30–21:00 UTC bounds. The current environment has no Alpaca credentials, so every probe is recorded as `AUTH_REQUIRED_NO_CREDENTIALS` in `alpaca_audit.json`. No daily data, synthetic ETF series, or target-stock prices were substituted. This branch remains blocked until timestamped minute bars can be independently checked.

## Integrity decision

The internal input is sufficient for the continuous-return auxiliary experiment. The external market-state branch is not considered feasible from this environment. The absence of credentials is a data-access limitation, not evidence that market features would improve the model.
