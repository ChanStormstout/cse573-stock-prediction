# Current interface: binary choice, not verbalized probability

CHOICE_PROTOCOL.md supersedes the output interface for the completed comparison.
The original JSON version remains a partial diagnostic: price609, news96, joint0.
Current `choice.py` reads UP/DOWN answer-token probabilities, normalizes the full
vocabulary in float32 then conditions on the two choices. This is uncalibrated
model token preference. It gives no rationale; do not invent one. `common.py`
still preserves the original JSON contract and input whitelist. `publish.py`
is its archived reporting path; use `publish_choice.py` for V2.

# Direct LLM four-hour forecasting

The LLM directly scores UP and DOWN; the program derives their conditional
probability and direction. It does not extract facts for another predictor. Frozen Qwen3.5-9B 4bit, three information variants:
price, news, price+news. Main four-hour target and all609 evaluation windows
remain unchanged. Read CHOICE_PROTOCOL.md and REPORT.md before interpreting results.

## Local reproduction

Raw course/news data, prepared packs, weights and full generated rationales are
private local assets, not redistributed. Requires existing pinned model from
stock_llm_model_compare. MLX requires Metal access; no model API or Sol needed.
Use a NEW ignored output directory each time:

```sh
work/stock-data/finbert-env/bin/python outputs/stock_llm_direct_4h/prepare.py --out work/stock-data/direct_4h/NEW_INPUTS
python3 -m unittest discover -s outputs/stock_llm_direct_4h -p 'test*.py'
work/stock-data/finbert-env/bin/python outputs/stock_llm_direct_4h/baseline.py --inputs work/stock-data/direct_4h/NEW_INPUTS --out work/stock-data/direct_4h/NEW_LR
work/stock-data/structured-env/bin/python outputs/stock_llm_direct_4h/run.py --choice --inputs work/stock-data/direct_4h/NEW_INPUTS --out work/stock-data/direct_4h/NEW_SMOKE --smoke
work/stock-data/structured-env/bin/python outputs/stock_llm_direct_4h/run.py --choice --inputs work/stock-data/direct_4h/NEW_INPUTS --out work/stock-data/direct_4h/NEW_RUN
work/stock-data/finbert-env/bin/python outputs/stock_llm_direct_4h/analyze.py --inputs work/stock-data/direct_4h/NEW_INPUTS --run work/stock-data/direct_4h/NEW_RUN --baseline work/stock-data/direct_4h/NEW_LR --out work/stock-data/direct_4h/NEW_ANALYSIS
work/stock-data/finbert-env/bin/python outputs/stock_llm_direct_4h/verify.py --inputs work/stock-data/direct_4h/NEW_INPUTS --run work/stock-data/direct_4h/NEW_RUN --baseline work/stock-data/direct_4h/NEW_LR --out work/stock-data/direct_4h/NEW_INTEGRITY.json
```

A stopped incomplete run may resume with `--resume` only when input, model,
protocol and code fingerprints and saved row prefixes match. Completed runs
cannot resume or be overwritten. Broken JSON lines are rejected, not silently
removed. Input truncation is refused. News selection/capping occurs in prepare,
with explicit source spans and counts; this is distinct from token truncation.

`choice.py` contains the current English prompt; `common.py` supplies whitelist
serialization and preserves the original JSON prompt.
`prepare.py` reconstructs six hourly histories from raw five-minute prices and
checks old labels/features. It keeps forecasting inputs separate from labels.
`baseline.py` really fits the LR controls; all vocab/scaling/C selection is past-only.
`run.py` does frozen inference and never loads outcome labels.
`analyze.py` joins outcomes after forecasting and records fixed paired cases.

## Interpretation

All market periods are exposed exploratory history. Modern LLM pretraining may
include these historical events; a request to ignore recalled outcomes cannot
certify absence of contamination. Relative old-model results are context, not a
clean architecture ablation: old models may see more/different news text. New
LR controls use the same selected title/excerpt text and past price source;
LLM sees structured per-article times while LR receives numeric time summaries.
Six per-bar numbers are rounded for language serialization, not model fitting.

Current p_up is conditional answer-token preference, not calibrated market
confidence; tied .5 predicts UP. V2 does not generate reasons or evidence IDs.
The original, partial JSON experiment used .5 fallback for invalid outputs and
retains its separate failure diagnostics. No trading strategy or causal effect
is claimed.

Four training examples, longer history, LoRA and output-repair comparisons are
not part of this primary run. Do not call this inference LLM weight training.

BATCH_ENGINE.md records the execution-only change to native batches of4. Eight
existing price outputs matched on probability/direction/validity, but six brief
explanations differed. The original JSON forecasts were restarted in a separate batch run, then
stopped after price609/news96. V2 is a separate full binary-choice run;
partial outputs are preserved and not mixed. This small check does not
prove full bitwise equivalence. OUTPUT_DIAGNOSTIC.md declares the additional
native-direction diagnostic for p=.5 / DOWN contradictions.

After all forecasts complete, publish only reviewed summaries:

```sh
python3 outputs/stock_llm_direct_4h/publish_choice.py --inputs work/stock-data/direct_4h/NEW_INPUTS --run work/stock-data/direct_4h/NEW_RUN --baseline work/stock-data/direct_4h/NEW_LR --analysis work/stock-data/direct_4h/NEW_ANALYSIS
```

Numerical prediction tables can be allowlisted after inspection; source excerpts,
raw generated text and weights must stay under ignored `work/` directories.
