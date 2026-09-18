# Market Context v1 — preregistered, not run

M0 is exact canonical R1. Mmeta is R1 plus `market_age_log1p_hours`,
`market_prior_session_fraction`, `market_window_missing`. M1 adds exactly
`spy_intrabar_return_60tm`, `spy_rv_60tm`, `qqq_intrabar_return_60tm`,
`qqq_rv_60tm`. SPY/QQQ use raw-adjusted SIP 5-minute bars, completed at least
one minute before cutoff, over the same 12 expected regular-session starts.

Mmeta/M1 use M0's per-stock selected C, with no new tuning. Real fitting is blocked
until both ETFs have >=95% valid contexts in every stock-month. Main gate is M1-M0
over six June--August cells: both mean BA deltas >=.01, >=2 positive macro months,
each Brier delta <=.002, and no constant outer prediction. Development/later are
exposed historical backtests. Future command (not run):
`python -m outputs.stock_context_4h.run_market --approve-market-context-run`.
