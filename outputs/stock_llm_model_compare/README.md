# Frozen 9B, prompt and staged extraction comparison

This is an upstream **four-hour project** component experiment. It does not train
a stock predictor or produce a new BA result. All April examples are exposed
regression cases with provisional model-assisted labels, not independent gold.

## Reproduce with the owner's local data

```sh
work/stock-data/structured-env/bin/python outputs/stock_llm_model_compare/download_model.py
python3 -m unittest discover -s outputs/stock_llm_model_compare -p 'test*.py'
work/stock-data/structured-env/bin/python outputs/stock_llm_model_compare/run.py --out NEW_RUN --smoke
work/stock-data/structured-env/bin/python outputs/stock_llm_model_compare/run.py --out NEW_FULL_RUN
python3 outputs/stock_llm_model_compare/rules_v2.py --run NEW_FULL_RUN --out NEW_RULE_REPLAY
python3 outputs/stock_llm_model_compare/analyze.py --run NEW_FULL_RUN --rules-v2 NEW_RULE_REPLAY --out NEW_ANALYSIS
python3 outputs/stock_llm_model_compare/safety_repair.py --input NEW_RULE_REPLAY --out NEW_SAFETY_REPLAY
python3 outputs/stock_llm_model_compare/publish.py --run NEW_FULL_RUN --analysis NEW_ANALYSIS --safety NEW_SAFETY_REPLAY
```

Replace output placeholders with new ignored directories under `work/`; existing
directories are rejected. Download resumes pinned model files through the HF
client. Inference is offline, requires Metal access, and saves every call before
continuing. A failed inference run is preserved and not silently resumed.
The model and original news/labels are not distributed in this repository.

## What changes

- **original**: same compact prompt as the old frozen1.7B, native9B chat template.
- **prompt**: checklist and five synthetic few-shot examples; same full input.
- **staged**: select current target-event sentence IDs; classify actions and copy
  exact clauses; deterministic rules read USD amounts or rating strings.
  Ambiguous or unsupported values are rejected and failures recorded. At most
  two model calls per article, so compute budget differs from one-stage methods.
  This package uses its own stage prompts without the five few-shot examples;
  comparisons do not isolate decomposition from wording, examples or rules.

No LoRA or new weight training. No thinking, greedy decode,512 output tokens
per call,4096 total budget; truncation is rejected. Model, input, label and code
fingerprints plus full private outputs preserve the actual execution evidence.

## Limitations

The original frozen comparison uses a different model family, tokenizer and
parameter count together. It is not a pure model-size ablation. Four-bit
quantization, runtime and prompting may interact; neither a win nor a loss
alone identifies their separate causal contributions. Synthetic few-shot
examples do not add market outcome labels. The prompt-package comparison does
not isolate instruction wording from examples.

The gate can reject true events. Clause extraction can omit useful context.
Rules deliberately reject implicit currencies and ambiguous numeric pairings;
coverage loss is part of the measured result, not grounds to remove samples.
The unchanged article validator rejects an entire response if any event is
invalid, so report invalid outputs alongside TP/FP/FN. An accepted literal
quote still does not establish correct target, timing or semantic relation.

The official [Qwen3.5 card](https://huggingface.co/Qwen/Qwen3.5-9B) recommends
stochastic sampling with a presence penalty for general non-thinking tasks.
This experiment deliberately retains the old greedy policy for matched
comparisons; it does not test the model's recommended sampling configuration,
thinking mode, constrained JSON decoding or a larger output budget. Repetitions
and length stops are recorded, not silently repaired. Poor results here cannot
establish that every Qwen3.5 configuration fails.

See PROTOCOL.md, SMOKE_AMENDMENT.md and CASE_PROTOCOL.md for pre-run decisions,
the synthetic interface correction, and deterministic case selection.
RULE_AMENDMENT.md records one development-only delta-to-target parser repair;
it replays saved stage outputs without new model calls. Keep both rule versions.

SAFETY_FIX.md separately records the post-check rating-pair correctness fix.
Apply this guard to staged output before future use; original registered results
remain unchanged. This does not make the remaining extraction quality accepted.
