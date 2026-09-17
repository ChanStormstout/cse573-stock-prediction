# Paper-method four-hour follow-up

This directory contains the finite follow-up to the current AAPL/AMZN four-hour system. Read `PRE_REGISTRATION.md` before `v1/REPORT.md`.

The runnable entry point is `run.py`. Private fitted models and intermediate matrices are written under `work/stock-data/paper_methods_4h/`; raw news text and model binaries are not published.

Phase 1 completed: 266 LR fits, branch calibration, and full-set aggregation.
No mechanism met the preregistered advancement rule; Phase 2 was not run.

Use a fresh run id to preserve prior results:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_paper_methods_4h/run.py --run v2
```

The current report and verification entrypoints read v1:

```bash
work/stock-data/finbert-env/bin/python outputs/stock_paper_methods_4h/report.py
work/stock-data/finbert-env/bin/python outputs/stock_paper_methods_4h/verify.py
```

See [results](v1/REPORT.md), [metrics](v1/metrics.csv),
[selection](v1/selection.json) and [model manifest](v1/model_manifest.json).
