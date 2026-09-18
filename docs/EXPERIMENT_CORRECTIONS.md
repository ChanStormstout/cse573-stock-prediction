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

## CORR-003 — Dense gate used the wrong months (historical statement superseded)

- **Historical statement:** the earlier audit reported that v4 computed the
  dense gate over exactly six June–August stock-month cells and treated
  D1−D0_day as AAPL `+0.42pp`, AMZN `+0.78pp`.
- **External review correction:** ISSUE-022 found that the executable gate still
  uses March–August even though it constructs June–August `gate_rows`. The
  earlier v4 PASS interpretation is therefore superseded pending a corrected
  rerun. The `+0.42/+0.78pp` values remain preserved historical output and must
  not be presented as the registered gate result.
- **Required repair:** audit the actual gate calculation path, make the
  June–August rule executable, then rerun in a new artifact directory.

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

## CORR-015 — Recency weight formula and dense gate verification (partly superseded)

- **Historical statement:** the v4 results were previously described as using
  the registered session-age formula and corrected June–August dense gate.
- **External review correction:** the recency weight formula evidence remains
  useful, but the dense-gate portion is superseded by ISSUE-022. The public
  verifier's six-cell checks do not prove that the executable advancement gate
  consumed those rows.
- **Repair:** `age_weights` now asserts non-negative session ages, exact
  `2**(-session_age/half_life)` values, unit weights for infinity, and
  monotonic decrease with age. `verify_weight_audit_v4.py` audits all saved
  v4 fold boundaries without refitting models. `verify_v4.py` now checks the
  six gated stock-month cells and the matched reconstructed feature mode when
  canonical dense parity fails.
- **Evidence:** 17,140 private weight rows have zero formula error and zero
  monotonic/unit-weight violations; the recency portion remains a verification
  strengthening. Dense gate evidence and its `+0.42/+0.78pp` deltas are
  retained only as historical diagnostics until the corrected rerun.

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

## CORR-017 — TabPFN checkpoint provenance and ensemble size

- **Discovered:** The old goal60 report described the default classifier
  checkpoint as synthetic-only without checking the bundled package metadata;
  the old code also set `n_estimators=1`.
- **Evidence:** TabPFN 6.3.0 metadata identifies the checkpoint as the
  `Prior-Labs/tabpfn_2_5` default classifier fine-tuned on real data, and the
  archive root is `real-large-samples-and-features`.
- **Repair:** A new v2 probe audited the local provenance and ran the same
  own/cross-stock chronological protocol with the library default
  `n_estimators=8`. The n=1 outputs remain a matched historical control.
- **Historical result wording (superseded):** the v2 report originally said
  that n=8 did not pass the cross-stock promotion rule. The v2 preregistration
  and runner did not save the exact goal60 formula as a machine-readable gate,
  so that wording is not evidence of a formal pass/fail decision.
- **Current safe claim:** n=8 showed no stable two-stock improvement under the
  fixed chronological comparison. This is a corrected frozen-prior
  configuration probe; neither synthetic-only performance nor the TabPFN
  paper's full pretraining system is established.

## CORR-018 — Stage 5 claim-only audit and calibration slopes

The claim-only audit read 420 saved calibration records without fitting or
changing predictions. The early unconstrained Platt manifest contains 32
negative slopes among 84 records, all in the AMZN branch. A negative slope
reverses score ordering, so those values are historical score remappings, not
simple monotone calibration. The later constrained `platt_shrunk`, temperature
and positive-slope manifests contain no negative slopes in the audit.

The same audit narrows several method labels: masked reconstruction is an SSL
pilot rather than TS2Vec; analogy is a lexical historical probe rather than
FinSeer; Event Adapter is a target-specific extraction probe rather than Ding
event/graph embedding; and Chronos is an endpoint-feature probe. It also
records that Qwen UP/DOWN values are token preferences, C2 tests one auxiliary
objective rather than all return signal, dissemination gains are
regularization-confounded, and historical F1/F2 controls must remain separate
from reselected F1_new/F2_new. No independent event-review gate is claimed.

## CORR-019 — Detailed reviewer addendum expands the dense/reaction blockers

The external reviewer addendum is more specific than the original coarse
ISSUE-022--ISSUE-027 list. Before any reaction code is changed, the detailed
requirements are tracked separately as ISSUE-028--ISSUE-040 in
`docs/CODEX_WORKING_SPEC.md`: censor unfinished five-minute bars and publish
per-row used-bar-end provenance (028--029); assert availability, reaction
start, and horizon maturity together (030); freeze the requested W0 baseline,
the W1/W2 horizon/aggregation definition, and the W3 residual/fallback
protocol (031--033); implement one machine-readable promotion gate covering
stock, month, Brier, coverage, and horizon completeness (034); fit AR1 numeric
and lexical transforms only on each past fold and use the registered AR2
target-context representation with source/coverage accounting (035--036);
verify time safety with used-bar assertions and future-price perturbation
replay (037--038); tie grouping and target-pair membership to the downstream
manifest (039); and use a protocol fingerprint plus a new output directory for
the repaired run (040).

These items supersede the affected dense/reaction PASS interpretation but do
not delete or rewrite the historical artifacts. No code has been repaired and
no rerun has been started under this correction; execution remains paused
until the complete detailed addendum is incorporated.

## CORR-020 — Phase A repaired dense/reaction checkpoint (2026-09-18)

The previous pause has now been resolved by new, separate artifacts; v4/v1
files were not overwritten. Dense v5 filters both baseline and candidate sides
to June--August before the six-cell merge. Its corrected deltas are AAPL
`+3.740pp`, AMZN `−0.079pp`, macro `+1.831pp`, and the two-stock gate fails.
The earlier dense `+0.42/+0.78pp` values remain superseded historical values.

Reaction v2 censors unfinished five-minute context bars, saves row-level used
bar-end provenance, uses the corrected W0--W3 protocol, and writes a complete
machine-readable promotion gate. Its 240m AR1 article gate passes, but the
downstream W1 changes AAPL from `48.70%` to `51.98%` and AMZN from `60.63%` to
`44.43%`; this is not a promoted four-hour improvement. AR2 is explicitly
`NOT_RUN_MODEL_BINARY_UNAVAILABLE`, and no independent human association gate
is claimed. `verify_v2.py`, dense v5 verification, and June--August infinity
parity verification all pass.

## CORR-022 — AR2 is optional for reaction downstream eligibility

The initial v3 preregistration wording incorrectly made AR2 availability a
precondition for W0--W3.  The reviewer addendum requires independent article-
level gates: a time-safe AR1 horizon that passes its full gate (including the
per-stock Brier guardrail, macro AUC, positive macro months, coverage and
non-constant checks) is eligible to supply reaction features even when the
target-context FinBERT binary for AR2 is unavailable.  AR2 remains explicitly
`NOT_RUN_MODEL_UNAVAILABLE`; cached title vectors are not substituted.  The
corrected v3 downstream run uses the registered W0=R1, W1=R1+coverage,
W2=R1+five reaction fields, and W3=F1+R1+the same five fields protocol.  The
first v3 article-only checkpoint is retained as a historical checkpoint; v2's
old residual W0--W3 results remain superseded.

The stock-specific reliability gate is implemented and preregistered, but its
predictive run is intentionally blocked until the user sends the exact
`APPROVE_GATE_RUN` approval. No G0--G3 result is available.

## CORR-023 — `8842d37` external review requires reaction v4 and Phase B preflight repair

The v3 reaction result is preserved but superseded pending a new v4 artifact.
v3 evaluated only 60m and 240m even though the corrected registered family is
30m, 60m, 120m, and 240m; it also represented unavailable pre-article context
as zero without model-visible validity flags. v4 must rerun every registered
horizon with completed-bar time safety, fold-local numeric preprocessing, and
explicit validity indicators. Its builder must additionally reject the known
non-Apple expansion of `AAPL` (American Association for Physician Leadership)
when no independent Apple evidence exists. This is a high-precision exclusion,
not a claim that entity ambiguity is solved generally.

The same review blocks Phase B execution until the registered advantage target,
exact R1 support fallback, identical train/evaluation imputation, directional
routing headroom, R1 provenance, and real-path static preflight checks are
repaired. No G0--G3 predictions, metrics, oracle, or exposed-period result is
generated during this correction.

## CORR-024 — reaction v4 completed the corrected article-level rerun; Phase B remains unexecuted

- **Historical statement:** reaction v3 was preserved as the corrected article
  probe even though it omitted 30m and 120m and encoded unavailable context as
  a numeric zero without model-visible flags.
- **Repair:** v4 used a new public/private artifact pair, all four registered
  30m/60m/120m/240m horizons, completed five-minute bars only, fold-local
  imputation/scaling, unscaled validity flags, and a narrow deterministic
  rejection of the known non-Apple AAPL acronym use. The verifier passed all
  registered time-safety and protocol checks.
- **Corrected result:** AR1 30m, 60m, 120m, and 240m each failed the complete
  June--August article-level gate. At 120m, AAPL/AMZN mean BA deltas were
  `+0.322pp/+2.570pp`, but AAPL missed the `+1pp` requirement and the Brier
  guardrail failed. At 240m, both BA deltas remained below `+1pp` and AMZN had
  constant-direction collapse. No candidate was promoted, so no W0--W3
  downstream artifact was generated.
- **Phase B boundary:** the stock-specific gate now passes its real-input
  preflight for advantage arithmetic, R1/F1_new provenance, exact support
  fallback, median preprocessing, directional headroom and nested fallback.
  It remains `PREREGISTERED_NOT_RUN`: no G0--G3 predictions, metrics, oracle
  ceilings, or exposed-period gate results exist.

## CORR-025 — final code-only verifier repairs before any Phase B approval

- **Oracle contract:** the future hindsight diagnostic now has a
  `ROUTING_ELIGIBLE` view that switches only rows with `eligible_news=1`, and a
  `FULL_SYSTEM_R1_ON_NO_NEWS` view that forces R1 on no-news rows. Both are
  labelled `HINDSIGHT DIAGNOSTIC ORACLE — NOT A MODEL` and remain excluded from
  fitting and selection.
- **Post-run audit:** `verify.py` now writes the existing preflight result when
  `v1` is absent, and otherwise independently reconstructs saved mappings,
  metrics, monthly metrics, advancement, chronology, preprocessing and
  coefficient structure without calling `run_gate.py`'s metric/advancement
  helpers.
- **Strengthened contracts:** preflight now regression-tests that 0.70 versus
  0.60 is not routing headroom, while 0.70 versus 0.40 is. Reaction v4's
  replay now changes bars whose *end* exceeds availability and separately tests
  an availability time of 10:02; `verify_v4.py` passes without rerunning the
  article models.
- **Boundary:** these are code-only repairs. Phase B remains
  `PREREGISTERED_NOT_RUN`; no G0--G3 prediction, metric, oracle, advancement,
  development or later artifact was generated.

## CORR-026 — approved Phase B run stopped at independent-verifier runtime failure

- **What happened:** after the owner approved one execution, the frozen
  `run_gate.py` command completed and preserved its `v1` artifact. The required
  immediate `verify.py` command then raised `NameError: clean is not defined`
  in its post-run branch before writing `v1/verification.json`.
- **Interpretation boundary:** this is an implementation failure, not evidence
  for or against G0--G3. The run is retained as `UNVERIFIED_STOPPED`; its
  metrics, oracle and advancement files must not be used to choose a mechanism
  conclusion or a new experiment.
- **Actions not taken:** no repair-and-rerun, report generation, post-result
  tuning, v2 creation or activity-column experiment was performed in this
  stage. A separate future authorization is required before repairing the
  verifier or using the preserved run further.

## CORR-027 — verifier-only recovery still leaves the Phase B run unverified

- **Scope preserved:** the only authorized code edit moved the existing
  `clean(value)` helper from `preflight_main` to module scope. No model,
  verification calculation, runner, saved prediction, metric, or advancement
  file was changed. SHA-256 records prove all eleven v1 files that predated
  recovery are byte-identical before repair, before verification, and after it.
  The recovery ledger initially omitted the final hexadecimal character of the
  `metrics.csv` SHA-256 and now records the corrected 64-character hash; this
  was a metadata transcription repair, not an artifact change.
- **Recovered result:** the existing post-run verifier now writes
  `v1/verification.json`, with 17 passing checks and one failed check:
  `advancement_independently_reconstructed`.
- **Exact stop reason:** the verifier reconstructs advancement at full floating
  point precision, whereas `advancement.json` was persisted by pandas
  `to_json` at approximately ten decimal digits. Its `1e-12` record comparison
  rejects that serialization difference. This is not repaired in place.
- **Boundary:** the single frozen controller run remains
  `UNVERIFIED_STOPPED_PENDING_EXTERNAL_REVIEW`. No runner rerun, report,
  tuning, v2 output, activity experiment, or result-artifact rewrite occurred.
