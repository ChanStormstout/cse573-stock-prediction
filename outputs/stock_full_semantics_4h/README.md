# Full-text and target-paragraph semantic comparison

New course-data lane, authorized after the owner viewed existing random outer
scores. All comparisons are **exploratory random historical backtests**.
It does not resume external entity review, market context or relation-reader work.

- [Report](v1/REPORT.md): only independently replayed results.
- [Traditional protocol](v1/protocol.json): three finite text-only comparators.
- [Semantic protocol](v1/SEMANTIC_PROTOCOL.json): frozen target-paragraph encoders,
  small classification heads, finite FULL fusion including zero semantic weight.
- [LLM reuse audit](v1/LLM_REUSE_AUDIT.json): existing incomplete outputs retained,
  no selected-subset scoring and no automatic resumption.

Private artifacts live in `work/stock-data/full_semantics_4h/v1` and must not be
committed. The original random study remains intact. Its selected inner OOF
predictions are selection data, not training labels for a learned stacker.

## Actual execution order

Use `work/stock-data/finbert-env/bin/python3` for these scripts:

1. `classical.py`, then `verify_classical.py`.
2. `prepare_semantics.py`, then `verify_inputs.py`.
3. `encode_semantics.py FINBERT`, then `encode_semantics.py MODERN`.
4. `train_semantics.py`, then `verify_semantics.py`.
5. `report.py`.

`test_contracts.py` checks exact zero-weight/unknown-evidence fallback, target
name boundaries and calibration-fold text preprocessing. Existing completed
runs must never be overwritten. Preparation refuses an existing semantic
folder; training validates completed-block fingerprints and refuses incomplete
blocks. Inspect active processes before invoking any command again.

## What changed

Traditional methods use the same canonical stemmed full-body text and random
folds; TF-IDF vocabulary/IDF and model selection stay inside the training data.
SVM calibration refits the entire text pipeline in its calibration partitions.
These are text-only practical comparators, not price-matched ablations.

The semantic branch preserves every paragraph explicitly naming the target,
including negation, comparison words and numbers. Long paragraphs are split
into the same contiguous text chunks for both encoders. Source offsets and
hashes remain private. A lexical company-name match does **not** certify the
company's event role or independent semantic quality.

Encoder parameters are frozen. Each downstream training fold fits its own
article PCA, scaling and logistic head. Inner validation chooses head C and
one of five finite mixing weights. No learned fusion coefficients are fitted.
No reliable target paragraph means exact FULL fallback. No news therefore
retains the same previously fitted price fallback.
