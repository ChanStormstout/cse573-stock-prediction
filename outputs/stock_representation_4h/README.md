# Four-hour representation diagnostics

This finite study preserves all previous experiments. Current evaluation periods
are exposed exploratory historical data. Random split scores are not forecasts
of performance on unseen future periods.

1. **Preprocessing:** restore only frequency, negation, numeric expressions or
   unstemmed words, against the existing calibrated TF-IDF SVM. Historical
   cleaning, article membership, splits and text-only C selection are preserved.
2. **Grouped SVM:** use existing association components in outer, inner and
   calibration splits. This is a separate robustness test, not a new main score.
3. **Aggregation order:** fixed word/price logistic baseline versus original
   vector mean, map-after-mean and map-before-mean. Reuse frozen FinBERT chunks;
   fit geometry only inside each training fold. No semantic encoder is trained.

Private inputs/model checkpoints are under
`work/stock-data/representation_4h/v1`. Public protocol and verified outputs are
under `v1`. Never restart an incomplete block automatically. Completed blocks
are reused only after hash validation.

## Commands

Use `work/stock-data/finbert-env/bin/python3`:

```sh
python outputs/stock_representation_4h/test_contracts.py
python outputs/stock_representation_4h/prepare.py
python outputs/stock_representation_4h/run.py preprocess
python outputs/stock_representation_4h/verify.py preprocess
python outputs/stock_representation_4h/run.py grouped
python outputs/stock_representation_4h/verify.py grouped
python outputs/stock_representation_4h/run.py aggregation
python outputs/stock_representation_4h/verify.py aggregation
python outputs/stock_representation_4h/report.py
```

Preparation is create-once and refuses overwriting. Verify performs no fitting.
The frozen budget screen controls whether an interaction proposal is justified;
it is not a significance test. TabPFN, price interactions, external data and
paused LLM work are not automatically launched.
