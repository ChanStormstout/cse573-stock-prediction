# Market state and continuous-return auxiliary supervision

This run follows the next project direction: first audit whether SPY/QQQ minute data
is actually available under the existing Alpaca connection, then test whether the
existing continuous four-hour return can enrich direction training.

```sh
work/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/audit_alpaca.py
work/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/run.py
work/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/report.py
work/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/verify.py
```

`run.py` uses CPU-friendly scikit-learn and PyTorch linear heads. It never reads the
future target return as an input. Raw data, fitted models and caches remain under
`work/stock-data/market_return_4h` and are excluded from Git; checked-in CSV/JSON
results are allowlisted explicitly.

The market branch is not declared complete when Alpaca authentication or minute-bar
quality is missing. The auxiliary branch is evaluated on the original task and does
not replace the four-hour direction target.
