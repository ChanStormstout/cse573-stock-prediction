# Larger model, prompt and decomposed extraction — 2026-09-16

Registered before new-model predictions. Main forecast target remains four-hour
AAPL/AMZN direction. This round evaluates an upstream extractor, not stock BA.

## Fixed experiments

Use the sealed `annotation/student_v3` inputs: 87 March development and 122 April
check articles. Both panels are already exposed; April is a regression panel,
not a fresh holdout. Provisional teacher/assistant labels are not human gold.
Report original and enriched April separately and by stock. No new model or
prompt selection on April; run all registered variants, including failures.

1. Existing frozen Qwen3-1.7B compact prompt: saved reference, no rerun.
2. Frozen Qwen3.5-9B 4bit: exactly the same message content and output contract.
   The native model chat template is necessarily different. This comparison
   changes model family, size and tokenizer, not parameter count alone.
3. Same 9B with a clearer checklist and fixed synthetic few-shot examples.
   Examples distinguish static/current, new-set/raise, missing old value and
   ownership changes. This is a prompt-package comparison, not pure length.
4. Same 9B, two-stage extraction: identify current target-event evidence, then
   classify actions and select an exact source clause. Deterministic rules parse
   numbers/rating values from that clause. Never ask the model to invent numeric
   values. Reject ambiguous clauses and record component failures. All articles,
   including gate rejections, remain in the denominator.

No training or new adapter in this round. Use seed573, greedy decoding,
nonthinking mode, maximum512 output tokens per call. Maximum4096 total tokens,
refuse truncation. Staged method has up to two calls: explicitly report its
extra budget/runtime rather than claiming compute-matched superiority. A
synthetic smoke example may verify loader/template but must not tune prompts.
All variants fixed before development runs; no adaptive prompt search.

## Model and execution

`mlx-community/Qwen3.5-9B-4bit`, revision
`8b2b98c00a6b4d291155e4890773ca8f769aee53`.
Use existing MLX runtime; inspect native Qwen3.5 support. Download into ignored
local work directory. Record model/config hashes, input hashes, code snapshots,
messages, tokens, outputs, runtime and MLX peak memory. Preserve failed runs.
Restart only after fingerprint validation; no silent overwrite/reuse.

## Evaluation and stopping

Reuse original strict fact signatures and literal-evidence validator. Report
TP/FP/FN, precision/recall/F1, valid output rate, no-event false positives and
per-stock/cohort results. Invalid outputs count as missed facts and their count
is explicit; conditional FP does not capture all malformed hallucinations.
Case review includes changed correct/wrong, common wrong/correct and staged
rule/gate rejections; do not derive prevalence from handpicked cases.

No new unseen annotation panel is created here. Any later confirmation needs
separate event-group-held-out labels not used to design these prompts. More
reliable extraction still does not establish four-hour predictive improvement.
No automatic full-corpus promotion, paid API, or further prompt grid.
