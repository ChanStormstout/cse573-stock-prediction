# Small LLM four-hour component

Current status: bounded local Qwen diagnosis and actual QLoRA pilot. See [REPORT.md](REPORT.md) and [PROTOCOL.md](PROTOCOL.md). **Not a successful new four-hour forecasting model.**

## Components

- `prepare.py`, `extend_panel.py`: early-period raw evidence with exact spans; repaired AMZN dedup; all raw inputs remain local.
- `label_panel.py`: explicit assistant-provisional labels after reading inputs; no stock outcomes. Not independently verified training gold.
- `position_probe.py`: old-task option-order diagnostic.
- `common.py`, `infer.py`: same few-shot input for frozen and adapted Qwen, JSON / exact-evidence checks.
- `train_adapter.py`: actual MLX QLoRA, assistant-only token loss, consistent non-thinking template, frozen base hashes, gradient/update checks, saved checkpoints and reload verification.
- `rule_baseline.py`: pre-existing rules with the same selected input.
- `input_probe.py`: assistant gold evidence diagnostic only; not deployable retrieval.
- `syntax_audit.py`: post-hoc punctuation-only parsing diagnostic, original results unchanged.
- `current_context.py`, `prompt_context_probe.py`: deterministic historical-list filtering and matched context-by-prompt diagnostics; these are exposed development comparisons.
- `forecast.py`: quality checks, past-only ledger validation, <=20-feature fact aggregation / offset, exact original price+text fallback.
- `review_and_replay.py`: candidate-specific independent-review packet and blocked-integration replay. No fitted new predictor without quality acceptance.
- `verify_artifacts.py`, `test_pipeline.py`: provenance, spans, time boundaries and failure guards.

## Local commands

Two existing environments: `work/stock-data/structured-env/bin/python` for MLX, and `work/stock-data/finbert-env/bin/python` for pandas/scipy forecasting checks. Input files/labels/adapters are excluded from Git. A clone alone cannot rerun training.

```sh
work/stock-data/finbert-env/bin/python -m unittest discover -s outputs/stock_llm_4h -p 'test_*.py' -v
work/stock-data/finbert-env/bin/python outputs/stock_llm_4h/verify_artifacts.py
# Select a NEW output directory. Existing directories are refused.
work/stock-data/structured-env/bin/python outputs/stock_llm_4h/train_adapter.py --out outputs/stock_llm_4h/runs/reproduce_smoke --smoke
work/stock-data/structured-env/bin/python outputs/stock_llm_4h/train_adapter.py --out outputs/stock_llm_4h/runs/reproduce_balanced --balance-events
work/stock-data/structured-env/bin/python outputs/stock_llm_4h/infer.py --out outputs/stock_llm_4h/runs/reproduce_eval --adapter outputs/stock_llm_4h/runs/reproduce_balanced/selected
```

`infer.py` accepts `--split valid` for development-only evaluation; never inspect a new check split while choosing configurations. Historical pilot checks already exposed remain exploratory.

`review_and_replay.py` makes a local packet containing raw evidence and actual selected-candidate predictions. Reviewers should mark whether the event is present, object/action/value/evidence errors, missed events and uncertainty without looking at stock outcomes. Empty/assistant-only review cannot pass. Acceptance requires at least 30 independently reviewed positive event groups per type; negative-only reviews cannot satisfy that requirement. This 11-article check is insufficient even if every row is reviewed. It is preparation material, not sufficient acceptance.

## Scope not yet completed

No 300–600-example independently reviewed corpus; no full-corpus fact generation; no new B1–B3 four-hour model fitting; no independent quality pass. This pilot excludes guidance, withdrawals, personal valuation and full issuer relations. 8bit/thinking alternatives and additional training seeds have not run; they are not claimed as completed experiments.
