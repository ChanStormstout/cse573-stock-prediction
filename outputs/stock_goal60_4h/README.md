# Four-hour new-information and dense-transfer experiment

Read `PRE_REGISTRATION.md` first. The sequential entrypoint is `run_all.py`; `replay_model.py` independently reloads a saved head. Detailed result interpretation will be in [v1/REPORT.md](v1/REPORT.md). All periods are exposed exploratory backtests.

Local prerequisites are the unchanged course dataset and private prior-run features/checkpoints. Raw data, article text, caches and weights are excluded from Git. Commands from repository root:

```sh
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/core.py
PYTHONPATH=work/stock-data/goal60_4h/runtime TABPFN_DISABLE_TELEMETRY=1 work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/tabular.py
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/encode_dense.py
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/text_models.py
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/text_models.py T_original T_A1
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/history.py prepare
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/history.py encode
work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/history_models.py
```

TabPFN uses the synthetic-only 2.5 checkpoint, one estimator, CPU, telemetry disabled. Its local training-sample conditioning is distinct from gradient training. Dense A1 uses existing January–February adapter weights, not newly fitted encoders. LR, boosting and bilinear heads are actually fitted in this round. Original and A1 token inputs are identical.

Alpaca fixed January 10, 2018 SPY/QQQ sample request returned HTTP401 without credentials. External-market branch is AUTH_REQUIRED and is excluded, not imputed or assumed available. No paid subscriptions or APIs are used.
