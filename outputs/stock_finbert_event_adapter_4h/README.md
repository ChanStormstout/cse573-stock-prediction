# FinBERT event adapter for the four-hour task

This directory contains the reproducible code for the finite AAPL/AMZN experiment registered in [`v1/PRE_REGISTRATION.md`](v1/PRE_REGISTRATION.md). Raw news, provisional labels, caches, environments, model checkpoints, and sentence-level probabilities stay under `work/stock-data` and are excluded from Git.

The official sequence is:

```bash
export PYTHONPATH=outputs/stock_finbert_event_adapter_4h
PY=work/stock-data/finbert-env/bin/python

$PY outputs/stock_finbert_event_adapter_4h/audit.py
$PY outputs/stock_finbert_event_adapter_4h/oracle.py
$PY outputs/stock_finbert_event_adapter_4h/value_audit.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY \
  outputs/stock_finbert_event_adapter_4h/input_audit.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY \
  outputs/stock_finbert_event_adapter_4h/train_adapter.py \
  --out work/stock-data/finbert_event_adapter_4h/v1/adapter_runs
$PY outputs/stock_finbert_event_adapter_4h/finalize_training.py \
  --run work/stock-data/finbert_event_adapter_4h/v1/adapter_runs
$PY outputs/stock_finbert_event_adapter_4h/evaluate_adapter.py \
  --adapter-run work/stock-data/finbert_event_adapter_4h/v1/adapter_runs \
  --out work/stock-data/finbert_event_adapter_4h/v1/exact_evaluation
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY \
  outputs/stock_finbert_event_adapter_4h/extract_corpus.py \
  --adapter-run work/stock-data/finbert_event_adapter_4h/v1/adapter_runs \
  --out work/stock-data/finbert_event_adapter_4h/v1/corpus_extraction
$PY outputs/stock_finbert_event_adapter_4h/downstream.py \
  --extraction work/stock-data/finbert_event_adapter_4h/v1/corpus_extraction \
  --out work/stock-data/finbert_event_adapter_4h/v1/downstream
$PY outputs/stock_finbert_event_adapter_4h/cases.py \
  --adapter-run work/stock-data/finbert_event_adapter_4h/v1/adapter_runs \
  --extraction work/stock-data/finbert_event_adapter_4h/v1/corpus_extraction \
  --downstream work/stock-data/finbert_event_adapter_4h/v1/downstream \
  --out work/stock-data/finbert_event_adapter_4h/v1/cases
$PY outputs/stock_finbert_event_adapter_4h/publish.py \
  --adapter-run work/stock-data/finbert_event_adapter_4h/v1/adapter_runs \
  --exact-eval work/stock-data/finbert_event_adapter_4h/v1/exact_evaluation \
  --extraction work/stock-data/finbert_event_adapter_4h/v1/corpus_extraction \
  --downstream work/stock-data/finbert_event_adapter_4h/v1/downstream
$PY outputs/stock_finbert_event_adapter_4h/verify.py
```

If training is interrupted, rerun `train_adapter.py` with `--resume`; completed fold and full-fit records are fingerprinted and skipped. Downstream stages refuse to overwrite an existing output directory.

`A0`, `A1`, and `A2` share identical sentences, folds, seeds, losses, and thresholds. Adapter and residual choices use training-time chronological predictions only. The check set, development period, and later period never choose a configuration. The public results remain exploratory because the labels are dual-model provisional and all stock evaluation periods have been exposed during project development.
