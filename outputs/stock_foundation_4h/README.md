# Finite foundation-model round

[Protocol](PRE_REGISTRATION.md) · [Report](v1/REPORT.md) · [Cases](v1/CASE_NOTES.md) · [Offline replay](v1/demo.html)

Uses existing private course data and the finbert-env environment. Official frozen model inference plus new LR training; no fine-tuning. Public results cannot fully recreate private inputs without the course dataset.

## Commands (repository root)

```sh
work/stock-data/finbert-env/bin/python -m pip install --target work/stock-data/foundation_4h/runtime --no-deps chronos-forecasting==2.2.2 einops==0.8.1 accelerate==1.12.0
work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/download_models.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/experiment.py aggregation
TOKENIZERS_PARALLELISM=false work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/encode_modern.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/experiment.py modern
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/chronos_experiment.py
work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/evaluate.py
work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/verify.py
work/stock-data/finbert-env/bin/python outputs/stock_foundation_4h/publish.py
```

Completed training refuses overwrite. For a new run, first create a new protocol and update OUT/PRIVATE in experiment.py to a new version; do not delete v1. Inference chunks resume only on matching keys, input, model, configuration and code fingerprints. Interrupted classifier stages reject existing weights and require a new run rather than silently mix artifacts. The one-time repair_cache.py already ran for v1 and is NOT part of clean-run reproduction: it preserves original object-key caches and converts keys to safe Unicode without changing predictions. Current inference saves Unicode from the outset.

Runtime: Python3.12, torch2.14.0, transformers4.57.6; additional pinned packages above are isolated from previous experiments. Model IDs/revisions/licenses and fingerprints are in inference JSONs. Contexts, raw source prices/text, weights, vectors and legacy-cache archives remain local. chronos_alignment.csv includes private raw anchor values and is deliberately excluded from Git; the published chronos_time_audit.csv contains only timing/coverage.

Open v1/demo.html directly, or serve the repository with `python3 -m http.server 8773 --bind 127.0.0.1`. New demo URL: http://127.0.0.1:8773/outputs/stock_foundation_4h/v1/demo.html . No network dependency or live inference. Browser checked both stocks, Modern/Chronos cards, text-only no-news fallback, label reveal and layout.
