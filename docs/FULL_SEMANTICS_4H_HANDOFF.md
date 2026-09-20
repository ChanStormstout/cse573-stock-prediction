# FULL semantic extension — completed handoff

## Current state

**COMPLETE AND VERIFIED.** No new encoder or classifier process remains active.
The original Qwen inference and old continuation remain suspended at 1,972
saved outputs. Do not resume either automatically. Their preserved output hash is
c37cbc45187fd6cbd86a3144cd06e294963df509088471fc57664e817e8d8ae7.
No attention, LoRA, external entity/relation or market-context lane was started.

[Report](../outputs/stock_full_semantics_4h/v1/REPORT.md) contains all seed means,
standard deviations, Brier and changed/repaired/introduced counts. Public CSVs
contain each seed and each evaluated window. All splits are exposed exploratory
random backtests; this is not an unseen-future score.

## Completed and checked

- TF-IDF LR/SVM/RF: 1,620 top-level fit calls, 180 selected model containers;
  SVM has additional internal calibration fitting. Reload verification PASS,
  maximum probability error 2.22e-16.
- Full target paragraphs: 5,226 article-target pairs, 30,448 paragraphs and
  31,059 matched chunks per encoder. Exact source offset and word coverage PASS.
- Frozen FinBERT and Modern encoding: 1,100.83 and 1,395.37 seconds on MPS.
  Zero trainable encoder parameters or gradients.
- Paragraph heads: 1,200 fits, 120 selected models, all 9,642 probabilities
  replayed. Maximum error 4.44e-16; embedding reaggregation error zero and
  no-evidence fallback mismatches zero. No learned stacking coefficients.
- Four targeted contract tests passed. Descriptive one/five-day paired
  intervals are reported; every paragraph-branch BA interval includes zero.

## Findings and boundary

FULL BA AAPL/AMZN: 67.57% / 60.28%.
Text TF-IDF SVM: 69.49% / 60.12%.
FULL + FinBERT body: 67.19% / 59.75%.
FULL + Modern body: 67.51% / 59.78%.

Neither body branch improves mean BA. Both lower Brier and move average
probabilities toward 0.5. Confidence shrinkage is a possible explanation,
not a proven new semantic signal. Title-fusion controls are reported with
explicit C-selection/evidence-gate matching limitations. No per-stock
outer-score winner is assembled into a deployment method.

Keep FULL as the declared reference and SVM as a strong traditional comparator.
This finite extension stops here. A calibration control or improved paragraph
selection could be considered separately; do not silently expand this grid.
LLM reuse remains incomplete and has no selected-subset performance result.

## Reproduction and preservation

See outputs/stock_full_semantics_4h/README.md and the frozen protocols.
Private artifacts stay under work/stock-data/full_semantics_4h/v1. The trainer
uses an exclusive lock and validates/skips completed blocks; do not overwrite
models. Original predictive artifacts are preserved.

The fit-free token-search and array-loading runtime fixes are documented in
PREPARATION_RUNTIME_NOTE.json and ARRAY_LOADING_RUNTIME_NOTE.json, with initial
code/history retained in Git and initial private logs retained. The project log
records the execution sequence. Neither fix changed scientific parameters.

Preserve unrelated .gitignore and docs/EXPERIMENT_INDEX.md edits. Refresh/check
before any future task commit, and verify remote main after pushing. Do not
publish raw articles, caches, model weights or environments.
