# Four-hour price self-supervision pilot

Read PRE_REGISTRATION.md before results. Local prerequisites: original private course bars/calendar, the unchanged1607-window dataset, and goal60 P_own_lr probabilities. This is a small masked-reconstruction encoder, not full TS2Vec. LLM news-increment design is separate in docs/LLM_NEWS_INCREMENT_PLAN.md and has not been run.

```sh
work/stock-data/finbert-env/bin/python outputs/stock_ssl_4h/run.py
work/stock-data/finbert-env/bin/python outputs/stock_ssl_4h/report.py
work/stock-data/finbert-env/bin/python outputs/stock_ssl_4h/verify.py
work/stock-data/finbert-env/bin/python outputs/stock_ssl_4h/summarize.py
```

Private raw features, sequence indices, encoders, LR models and environments remain under work/stock-data/ssl_4h/v1, excluded from Git. Public artifacts describe exploratory history only. Matching completed runs resume without refitting; changed fingerprints are rejected.
