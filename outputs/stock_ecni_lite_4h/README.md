# ECNI-lite four-hour experiment

This directory implements a finite, gated residual experiment over the
canonical chronological AAPL/AMZN four-hour task.

It combines four previously separate information sources without retraining the
baseline text model:

1. dissemination metadata (article and approximate event-cluster counts);
2. provisional target-company rating/target-price updates;
3. past-only analogous-news reactions;
4. an existing saved LLM response to the same historical cases.

Run order:

```bash
work/stock-data/finbert-env/bin/python3 outputs/stock_ecni_lite_4h/test_synthetic.py
work/stock-data/finbert-env/bin/python3 outputs/stock_ecni_lite_4h/run.py
work/stock-data/finbert-env/bin/python3 outputs/stock_ecni_lite_4h/verify.py
work/stock-data/finbert-env/bin/python3 outputs/stock_ecni_lite_4h/report.py
```

The runner refuses to overwrite an existing run. Model binaries are private
runtime artifacts and must not be added to Git. Read `PRE_REGISTRATION.md`
before interpreting any result.
