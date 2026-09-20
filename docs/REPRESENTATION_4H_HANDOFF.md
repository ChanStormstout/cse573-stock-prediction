# Representation diagnostics: complete

Starting SHA: `c8d4d8f73fa6d60cc03030aeddbbbb1dfb7519f6`.

All three registered stages executed and passed fit-free verification. Read
[findings](../outputs/stock_representation_4h/v1/FINDINGS.md),
[report](../outputs/stock_representation_4h/v1/REPORT.md),
[protocol](../outputs/stock_representation_4h/v1/protocol.json) and
[final audit](../outputs/stock_representation_4h/v1/FINAL_AUDIT.json).

## What happened

- Four exact one-operation lexical changes: no substantial reliable two-stock
  gain; all descriptive paired intervals against the strong SVM include zero.
- Strong SVM with group-disjoint outer/inner/calibration:49.23/51.17 BA,
  substantially below ordinary random69.55/61.14; AAPL up predictions97.72%.
- Mean-before-map vs map-before-mean: latter65.22/59.27 versus65.09/59.06,
  but matched no-semantic LR64.99/59.51. Frozen follow-up screen false.
- 5,400 saved model containers independently replayed; maximum error3.33e-16.
- 11,400 actual classifiers plus9,000 sigmoid calibrators; nine tests pass.
- 28,780 historical files unchanged; all legacy probabilities preserved.

## Boundaries and next decision

No interactions, TabPFN expansion, encoder run, LLM continuation or external ECNI
work occurred. Do not auto-start those. No stock-specific outer-score winner was
assembled. The original LLM processes remain paused with1,972 records unchanged.
All periods are exposed; group components are associations, not gold events.
Source-text case review is assistant analysis, not independent annotation.

A later proposal should address transfer to held-out news groups or missing
information rather than interpret a random-score peak as stable forecasting.
Keep the existing fixed SVM as the ordinary-random strong comparator and FULL as
the declared reference. Do not rewrite earlier results or expand the grid.

## Reproduction

Code and commands: [entrypoint](../outputs/stock_representation_4h/README.md).
Private immutable inputs/checkpoints: `work/stock-data/representation_4h/v1`.
The preparation path refuses overwriting; incomplete blocks are never silently
rerun. Verification/reporting are fit-free. Scientific runner fingerprints are
bound to the preregistration/input seal from before fitting.

Unrelated owner changes in `.gitignore` and `docs/EXPERIMENT_INDEX.md` must remain
outside this task's commit. Raw course text, caches, environments and weights
remain private.
