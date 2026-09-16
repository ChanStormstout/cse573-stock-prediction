# L00–L03 small LLM implementation protocol
Registered before new inference/training on 2026-09-16.

- Main forecast remains AAPL/AMZN four-hour direction; original 1,607 windows unchanged.
- L00: old 40-case single-letter task, 12 fixed cases, three cyclic option orders. Reuse old provisional labels. This is a bias diagnostic, not independent accuracy.
- L01 pilot: deliberately limited analyst rating / target-price facts, including no supported current event, from January–February 2018. Other types remain unknown/out of scope, not negative financial news. Multi-fact JSON; no direction labels, future prices or issuer inference.
- Select chronological, event-group-separated real articles before reading outcomes. Assistant labels remain provisional. Inspect the selected text before writing labels. The pilot is not the planned 300–600-example fully reviewed corpus, and cannot license full-corpus expansion.
- Frozen Qwen3-1.7B 4bit revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e vs same model with QLoRA. Template non-thinking at both training and inference. Matched input and decoding. No arbitrary letter-only output for facts.
- First perform actual 24–48 micro-step smoke training with up to eight manually checked examples. Confirm base frozen, adapter updates, prompt masking, reload. This is engineering evidence, not predictive efficacy.
- If correct, pilot training last eight layers Q/V, rank8, scale16, dropout .05, LR1e-4, batch1, accumulation8, max2048, at most3 epochs. First seed573; further seeds only if provisional development field quality warrants expansion. Limited pilot checkpoint selection on development, never check set. Keep all results including deterioration.
- Independent reviewer gate: each used type >=30 independent nonduplicate checks and >=90% critical quality, plus coverage and recall. Absent reviewers => NOT_ACCEPTED. No full-corpus extraction or new four-hour fitted event branch before gate passes. Implement interface and validate refusal and exact price+text fallback instead.
- No paid APIs. Raw article/annotation/token/adapter artifacts local only. Code, protocol, aggregate metrics and sanitized manifests may be public.

## Targeted second pilot, registered from development failures before execution

The actual 22 unique training articles include only seven positive articles and fifteen no-event inputs, because many headline "raised/lowered" strings describe holdings/product upgrades. Epochs1–2 of the first pilot predict no events on all eight development examples (0/9 gold facts recovered). A second and final targeted variant keeps the model, seed, prompt, articles, LR, three epochs and update count fixed. Weight positive-article loss to 2/3 of the total expected article weight and negative loss to1/3; do not duplicate articles or count weighting as new supervision. No longer epochs or wider grid. Check results already exist for the frozen pilot; all quality claims remain provisional/exploratory. A second failure stops expansion.

## Input repair after diagnostic, before new inference

Gold-evidence diagnostic on seven TRAIN examples recovers2/8 facts versus0/8 with the selected context. This is not deployable evidence retrieval. Register a deterministic, implementable repair: cut candidate context at explicit other-broker/historical-analyst-list markers (preserve title and all preceding sentences), with no access to labels. Evaluate frozen model on this new input only, same schema/prompt/decoding. Existing two QLoRA pilots remain unchanged; do not claim adapted-on-new-input results or add more QLoRA versions after two failures. Preserve exact source spans and report any removed provisional gold evidence. New quality results remain exploratory; no downstream prediction before acceptance.

Acceptance requires at least30 independently reviewed POSITIVE event groups per type, to avoid an all-negative sample licensing an always-empty extractor. This conservative requirement is not an accuracy claim; current independent review remains0.

The user's follow-up asks whether context/prompt is responsible. Register a2x2 on the same eight already-exposed development inputs: original/current-only context x original few-shot/short prompt. Reuse the two unchanged original prompt cells, run16 new short-prompt generations, same model, schema and decoding. The short variant changes both prompt wording and removes chat examples; interpret as a prompt-package comparison, not proof of a single causal token factor. Do not use it to restart unlimited tuning.
