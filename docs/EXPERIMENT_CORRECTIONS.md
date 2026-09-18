# Experiment Corrections

This append-only document records corrections to interpretation and
implementation. Historical output files are retained; corrections apply to
current-facing documentation and future comparisons.

## CORR-001 — F2 recency representation mismatch

- **Discovered:** The old recency implementation applied PCA approximately to
  window means and did not preserve canonical J2 metadata.
- **Affected results:** `stock_recency_dense_4h/v3` F2 recency delta.
- **Root cause:** Representation and sample weighting changed together.
- **Old claim:** F2 recency numbers were described as a pure recency test.
- **Correct interpretation before rerun:** The old F2 result was a confounded
  component probe.
- **Repair:** v4 uses canonical J2/article-level PCA, metadata, and a true
  infinity refit; v3 remains historical.
- **Corrected result:** v4 F2 outer delta is AAPL `-4.34pp`, AMZN `+1.16pp`,
  and it does not pass the gate.

## CORR-002 — Recency infinity was not a same-code refit

- **Discovered:** The old infinity branch reused saved probabilities.
- **Affected results:** all old recency parity claims.
- **Repair:** v4 fits all-one sample weights and compares probabilities.
- **Corrected result:** maximum difference is `1.665e-15`; v4 parity passes.

## CORR-003 — Dense gate used the wrong months

- **Discovered:** v3 computed the dense gate over March–August.
- **Repair:** v4 asserts exactly six June–August stock-month cells and uses a
  matched reconstructed control after feature parity failed.
- **Corrected result:** D1−D0_day is AAPL `+0.42pp`, AMZN `+0.78pp`; no upgrade.

## CORR-004 — Dense-window overlap wording

The old “non-overlapping” wording is incorrect. These are densely sampled,
overlapping four-hour windows with 30-minute start intervals. v4 and current
documentation use the corrected wording; v3 is preserved.

## CORR-005 — ModernBERT pooling mismatch

- **Discovered:** `encode_modern.py` included special tokens in mean pooling,
  while canonical FinBERT excludes them.
- **Affected results:** `stock_foundation_4h/v1` Modern and 8+8 comparisons.
- **Repair:** Stage 3 will use a shared non-special-token pooling helper and
  first reproduce canonical FinBERT before running ModernBERT.
- **Correct interpretation:** v1 is an unmatched frozen encoder probe and is
  not a clean encoder-family comparison.

## CORR-006 — TabPFN synthetic-only/configuration wording

The old goal60 README calls the checkpoint synthetic-only and uses
`n_estimators=1`, but provenance and installed package metadata have not yet
been independently established. Stage 4 will audit the checkpoint before any
corrected probe. Until then, call it a reduced-compute local TabPFN probe.

## CORR-007 — SSL is not TS2Vec

The masked-reconstruction Conv1d experiment is a small SSL pilot. Its negative
result cannot be written as “TS2Vec failed.” No rerun is authorized in this
task.

## CORR-008 — Historical analogy is not FinSeer

P1/P2/P3 use lexical/coarse-category retrieval and frozen Qwen. They are a
historical analogy RAG probe inspired by related work, not a FinSeer
reproduction. The safe claim is limited to this retrieval configuration.

## CORR-009 — Event Adapter is not Ding event embedding

A1/A2 adapt FinBERT for target evidence/action extraction. They do not
reproduce Ding structured event or knowledge-graph embeddings. The safe result
is provisional extraction quality versus narrow downstream event features.

## CORR-010 — Unconstrained Platt score remapping

The early `stock_nextgen_4h/calibrate.py` implementation can fit negative
slopes. A negative slope reverses ranking, so it is not simple monotone
probability calibration. Stage 5 audits all saved slopes; later constrained
`platt_shrunk` work remains the preferred calibration interpretation.

## CORR-011 — Dissemination gain was regularization-confounded

The local AMZN 57.32% event-aggregation result can be reproduced with strong
regularization without event clustering. No independent directional gain from
clustering was established; the result must not be attributed to deduplication
alone.

## CORR-012 — Qwen direct output is token preference

The binary UP/DOWN interface compares answer-token scores. It is a conditional
token preference, not a calibrated market probability. Brier is retained as a
diagnostic with this limitation.

## CORR-013 — Auxiliary return and cross-stock wording

The C2 experiment tests one shared linear direction-plus-return objective, not
whether return magnitude contains no signal. P_cross tests past-only peer
features and does not use future contemporaneous returns. These narrower claims
replace broad statements in current-facing documentation.

## CORR-014 — Historical and reselected F1/F2 controls

Canonical historical F1/F2 preserve their original selection logic for
reproducibility. F1_new/F2_new are reselected actual-system controls used in
later experiments. Reports must name which control they use; no silent swap is
allowed.

## CORR-015 — Recency weight formula and dense gate verification

- **Discovered:** The v4 results already used the registered session-age
  formula and the corrected June–August dense gate, but the public verifier
  checked only broad positivity/parity conditions.
- **Repair:** `age_weights` now asserts non-negative session ages, exact
  `2**(-session_age/half_life)` values, unit weights for infinity, and
  monotonic decrease with age. `verify_weight_audit_v4.py` audits all saved
  v4 fold boundaries without refitting models. `verify_v4.py` now checks the
  six gated stock-month cells and the matched reconstructed feature mode when
  canonical dense parity fails.
- **Evidence:** 17,140 private weight rows have zero formula error and zero
  monotonic/unit-weight violations; public v4 verification passes. This is a
  verification strengthening, not a new score search or a changed numeric
  result.

## CORR-016 — ModernBERT special-token pooling

- **Discovered:** v1 ModernBERT used attention-mask mean pooling that included
  special tokens, while the canonical FinBERT embeddings were documented as
  mean-pooling real non-special tokens.
- **Repair:** v2 re-encoded the same 5,078 titles with a shared
  `attention_mask=1` and `special_tokens_mask=0` pooling helper, then reran the
  same J2/PCA/LR protocol in a new output directory.
- **Evidence:** canonical FinBERT reproduction error is `1.72e-15`; v2
  Modern vectors differ from v1 (mean L2 `2.0673`). v2 remains a frozen
  encoder probe and does not validate the full Fin-ModernBERT paper system.
