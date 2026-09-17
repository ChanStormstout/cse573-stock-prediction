# Four-hour context, recent-price and fusion experiment

## Completed result

Production run `runs/v1` is complete and verified. It contains all 1,607 fixed
four-hour windows, 200 fitted price LR models, frozen Qwen3.5-9B scores for
P0--P3, past-only calibration, F0--F6 fusion, monthly/subgroup metrics and fixed
cases. No single method beat F0 with the Brier guardrail in all four
stock-by-period cells. See [REPORT.md](REPORT.md),
[CASE_INTERPRETATIONS.md](CASE_INTERPRETATIONS.md) and
[FINAL_STATUS.md](FINAL_STATUS.md).

September 2018 onward is an exposed exploratory replay. The paragraph review is
assistant-only and does not satisfy the independent review gate.

This directory contains the reproducible public code and reports for the next
four-hour AAPL/AMZN experiment. Raw course news, cached model files, private
prompts, model binaries and full body spans stay under `work/stock-data` and are
not committed.

The checked-in `config.json` records the versioned production directories. For
a fresh version, edit those paths and run preparation through the quality gate:

```bash
python3 outputs/stock_nextgen_4h/run_all.py \
  --config outputs/stock_nextgen_4h/config.json \
  --start inventory --stop quality
```

Inspect the label-free cards and create the review record described in
`MANUAL_REVIEW.md`. A failed review requires another paragraph-input version.
After a passing review, continue with `--start price --stop verify`. This manual
gate is intentional; a fresh run cannot silently self-approve paragraph quality.

To resume an interrupted LLM stage, keep the same sealed inputs, model revision,
prompt and code, then run:

```bash
python3 outputs/stock_nextgen_4h/run_all.py \
  --config outputs/stock_nextgen_4h/config.json \
  --start llm --stop verify --resume-llm
```

Every producing stage refuses to overwrite an existing run directory. LLM-only
interruption recovery requires the same manifest and the explicit
`--resume-llm` flag.

Use the bundled Codex Python runtime recorded in `ENVIRONMENT.md`. LLM inference
uses the pinned local MLX model and does not call a paid API.

Read `PROTOCOL.md` before the results and `IMPLEMENTATION_NOTES.md` for the
forward-month wording clarification, identical-prompt batch normalization and
the superseded inventory audit.
