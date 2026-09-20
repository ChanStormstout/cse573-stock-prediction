# Independent GPT semantic review instructions

## Purpose

Review all 515 frozen entity-relation cards independently. Follow the exact shuffled order in `ENTITY_REVIEW_INDEX.jsonl`, or process `review_batches/BATCH_01.jsonl` through `BATCH_05.jsonl` in sequence. Each card appears exactly once across the five batches.

This is a blinded semantic review. The pack contains provenance and the original public source URL. It contains no prior semantic answer, model relation, gate outcome, article body, evidence excerpt, or stock return.

## Review procedure

For every row:

1. Read the reviewer-safe metadata, including the target company, matched alias, matched field, publisher, date, and native ticker.
2. Open `source_url` when `source_access_status` is `PUBLIC_URL_PRESENT`.
3. Use only information visible in the source article and reviewer-safe row.
4. Choose exactly one label from `GPT_REVIEW_LABEL_GUIDE.md`.
5. Use `UNCERTAIN` when the accessible evidence cannot support a reliable decision.
6. Set `source_checked` to `true` only if you actually inspected the source page. Otherwise set it to `false`.
7. Return one JSON object per line following `GPT_REVIEW_OUTPUT_SCHEMA.json`, in the frozen order. A short `review_note` is optional.

Do not search for or use stock prices, future returns, market reactions, or downstream outcomes. Do not infer what the relinker predicted. Do not reorder, omit, replace, or add cards.

## Blinding boundary

Do **not** open these files until all 515 labels have been finalized and saved:

- `outputs/stock_ecni_e1rv2/PRE_REGISTRATION.md`
- `outputs/stock_ecni_e1rv2/E1RV2_REPORT.md`
- `outputs/stock_ecni_e1rv2/E1RV2_FINAL_AUDIT.json`
- `outputs/stock_ecni_e1rv2/NDAQ_SENTINEL_AUDIT.json`
- `outputs/stock_ecni_e1rv2/ALIAS_RISK_CENSUS.csv`

Do not inspect other E1R-V2 reports or audit files during labeling. The independent review output should contain only the schema fields and the reviewer's own decisions.

## Files to use

- `ENTITY_REVIEW_INDEX.jsonl`: all 515 reviewer-safe rows in frozen display order.
- `review_batches/BATCH_01.jsonl` through `BATCH_05.jsonl`: the same rows divided into five batches of 103.
- `GPT_REVIEW_LABEL_GUIDE.md`: the eight allowed semantic labels.
- `GPT_REVIEW_OUTPUT_SCHEMA.json`: the required response format.
- `BLINDED_DISPLAY_ORDER.json`: the frozen card order and its integrity hash.
- `SOURCE_URL_MAPPING_AUDIT.json`: provenance and URL-mapping counts only; it contains no semantic answer.
