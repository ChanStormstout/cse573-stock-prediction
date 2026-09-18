# ChatGPT Review Handoff

This file is append-only. Each entry records a reviewable implementation
stage. It is not a substitute for detailed reports or the validity ledger.

---

# Handoff 2026-09-17 — Stage 0 audit bootstrap

## Git

Starting SHA: `e32785ded90646c7911e55ff931fd322358974c5`

Ending SHA: pending Stage 0 commit

Commit(s): existing local `e32785d` is preserved; remote tracking SHA is
`c0383ac` because GitHub DNS was unavailable.

## What changed

Created the persistent working specification, method-validity truth table,
append-only correction log, and this handoff file. No experiment code or
historical result was changed in Stage 0.

## Why

The current repository contains many useful probes, but several are narrower
than their paper/model names and a few comparison controls are materially
confounded. A repair-first audit prevents a new score search from hiding those
problems.

## Files changed

- `docs/CODEX_WORKING_SPEC.md`
- `docs/METHOD_VALIDITY_AUDIT.md`
- `docs/EXPERIMENT_CORRECTIONS.md`
- `docs/CHATGPT_REVIEW_HANDOFF.md`

## Commands actually executed

- `git status --short --branch`
- `git fetch` (failed: could not resolve github.com)
- `git log -n 5 --oneline --decorate`
- `git rev-parse HEAD`
- `git rev-parse origin/main`
- environment package-version probe using `work/stock-data/finbert-env/bin/python3`

## Results

The local branch was clean and one commit ahead of origin/main. GitHub DNS
resolution failed. Environment versions are numpy 2.3.5, pandas 2.2.3,
scikit-learn 1.9.1, torch 2.14.0, transformers 4.57.6; `tabpfn` is not
importable in this environment.

## Previous result that is now corrected/superseded

No numeric result is changed in Stage 0. The interpretation of the v3 F2
recency, v1 ModernBERT, TabPFN, SSL, analogy, Event Adapter, calibration and
dissemination claims is narrowed in the validity ledger and correction log.

## Verification

Stage 0 is documentation-only. Stage 1 must add canonical parity and weight
checks before a recency result is treated as clean.

## Known unresolved issues

ISSUE-001 through ISSUE-021 are listed in the working specification. Network
access is required for pushes and possibly official TabPFN provenance.

## Branches intentionally NOT run

No new reaction-supervised FinBERT, LLM prompt search, GNN, RL, Chronos tuning,
dense stride tuning, recency half-life search, or exposed-period score search.

## External-method interpretation changes

The ledger now labels SSL, analogy, Event Adapter and Chronos as probes or
simplifications rather than full paper reproductions. The TabPFN checkpoint is
unverified until Stage 4.

## What ChatGPT should review first

1. `docs/CODEX_WORKING_SPEC.md` issue list and stage gates.
2. `docs/METHOD_VALIDITY_AUDIT.md` fidelity and safe-claim rows.
3. `docs/EXPERIMENT_CORRECTIONS.md` before reading any old score table.

## Exact reproduction commands

No new experiment command was run in Stage 0. Start Stage 1 from the repository
root with the v4 recency command after its parity audit is complete.

## Stage 0 finalization

The Stage 0 documentation commit is `5eec28e` (`Document method audit and
repair plan`). A push was attempted after this commit and was blocked by the
same unresolved GitHub DNS error; `origin/main` remains the prior tracking
commit. The next stage proceeds locally without rewriting the existing v4 run.

## Stage 1–2 completion (local)

Stage 1 repaired the executable recency weighting guard and audited every
saved v4 fold boundary. `recency_weight_audit.json` reports 17,140 private
rows, zero formula error, zero monotonic violations, and zero infinity unit
weight violations. The existing v4 true-infinity refit parity remains
`1.665e-15`, so the recency numeric results remain valid as a matched probe;
their cross-stock advancement gate still fails.

Stage 2 strengthened the dense verifier. It now checks exactly six
June–August stock-month cells and requires the matched reconstructed feature
mode when canonical dense feature parity fails. The public v4 verification
status is PASS; D1 remains a small diagnostic and D2 remains stopped by gate.

No private weight audit rows or model binaries were added to Git. A push was
attempted after this local stage and again requires GitHub DNS/network access.

## Stage 3 completion (local)

The canonical FinBERT J2 path was rerun first and reproduced 1,374 non-warmup
probabilities with maximum absolute error `1.72e-15`. This satisfied the gate
for the ModernBERT comparison. v2 then encoded the same 5,078 titles with a
special-token-excluded mean pool on MPS (159 batches, 26.25 seconds, zero
trainable parameters) and reran the frozen PCA/LR protocol.

Modern v2 showed mixed exposed-period results and no stable two-stock
improvement; no dual-encoder fusion or further tuning was run. Because v2 did
not save the exact cross-stock promotion formula as a machine-readable gate,
this is not a formal pass/fail promotion claim. The safe claim is a repaired
frozen encoder probe, not a full paper reproduction or a proof that ModernBERT
is ineffective.

## Stage 4 completion (local)

The local TabPFN 6.3.0 runtime and checkpoint were audited before inference.
The bundled metadata and archive identify the default classifier as a
real-data-fine-tuned `Prior-Labs/tabpfn_2_5` checkpoint; the old
synthetic-only description is therefore corrected. A separate v2 run used the
same own/cross inputs and chronological splits with fixed `n_estimators=8`
(28 fits, no gradient training). Reload, finite-probability and time-order
checks passed, but the corrected probe showed no stable two-stock improvement.
Because v2 did not save the exact goal60 promotion formula as a machine-readable
gate, this is not a formal pass/fail promotion claim. No further TabPFN tuning
was run.

## Stage 5 completion (local)

Stage 5 was deliberately claim-only. `outputs/stock_method_validity_audit/v1`
records the exact public calibration manifests and the safe/unsafe wording
table for ISSUE-012 through ISSUE-021. The audit found 32 negative slopes in
the 84 saved early unconstrained Platt records (all AMZN); the later
constrained manifests contain no negative slopes. No fit, inference, model
selection, prediction, or exposed-period score was changed.

Current-facing course documentation now uses the repaired special-token-
excluded ModernBERT comparison and the verified real-data-fine-tuned TabPFN
description. It calls the SSL, analogy, Event Adapter and Chronos runs narrow
probes rather than full paper reproductions, treats Qwen direct scores as token
preferences, and keeps dissemination gains separate from regularization.
Historical F1/F2 controls are explicitly separated from F1_new/F2_new.

Stage 6 remains: run all public verifiers and repository checks, update the
project log, commit the audit, attempt the required push, and report the
network boundary if GitHub DNS is still unavailable.

## Stage 6 verification (local)

The final verification pass completed successfully:

- recency weight audit, recency/dense v4, ModernBERT pooling v2, TabPFN v2,
  and the claim audit all returned `PASS`;
- `scripts/refresh_repository.py` refreshed 378 selected result files and
  `scripts/check_repository.py` passed 836 tracked/selected files with 285
  Python sources parsed;
- `git diff --check` passed;
- `outputs/stock_method_validity_audit/v1/final_verification.json` records the
  evidence counts and limitations.

The task is complete locally pending the final commit and the required push
attempt. The remote branch will only be described as updated if a subsequent
network check verifies it.

## Final local commit and remote boundary

The audit payload was committed locally as `ea621d5` (`Complete method validity
audit and claim corrections`). After that commit, `refresh_repository.py` and
`check_repository.py` again passed. `git push origin main` was attempted and
failed with `Could not resolve host: github.com`; `origin/main` remains at
`e32785d`. The boundary note is recorded in a follow-up documentation commit;
GitHub has not been claimed as updated.

## Historical reviewer addendum pause (2026-09-18; superseded below)

The external reviewer reported six coarse blocking validity issues in the
earlier `e32785d` dense/reaction implementation. They are recorded as
ISSUE-022 through ISSUE-027 in `docs/CODEX_WORKING_SPEC.md`, with the detailed
requirements split into ISSUE-028 through ISSUE-040: dense executable gate
month range, unfinished-bar reaction context, the W0–W3 protocol, the complete
promotion gate, AR1/AR2 preprocessing and representation, reaction time-safety
verification, manifest auditability, and a new-run fingerprint.

At that point execution was paused pending the detailed addendum. No historical
output was deleted, reset, or overwritten. The prior verifier outputs remain
preserved as historical artifacts; the repair checkpoint below incorporates
the addendum and supplies the current v5/v2 evidence.

## Repair checkpoint before stock-specific gate experiment — 2026-09-18

Starting SHA for this repair was local `dd836d80c709cd98065249ab5cdde233bd7abdc1`.
All earlier dense/reaction files remain unchanged.

### Resolved issue IDs and evidence

- ISSUE-022/028: `stock_recency_dense_4h/v5` filters both gate sides before
  merging exactly six June--August stock-month cells. Corrected deltas are
  AAPL `+3.740pp`, AMZN `−0.079pp`, macro `+1.831pp`; the gate fails.
- ISSUE-023/029/030/037/038: reaction v2 uses completed bars only, saves row
  used-bar ends and reaction maturity, and passes future-price perturbation
  replay plus availability/start/end assertions.
- ISSUE-024/031/032/033: v2 freezes the F1 W0, single-horizon W1,
  multi-horizon W2 and strict two-article W3 fallback protocol.
- ISSUE-025/034: every horizon has machine-readable stock-month BA/Brier,
  coverage, row-count and horizon-completeness checks. 240m passes the article
  gate; 30/60/120m fail.
- ISSUE-026/035/036: AR1 preprocessing is fold-local. The required
  target-context FinBERT binary is unavailable, so AR2 is
  `NOT_RUN_MODEL_BINARY_UNAVAILABLE`; no substitute representation was used.
- ISSUE-039/040: target-pair keys, grouping counts, downstream manifests and a
  v2 preregistration fingerprint are saved in a new directory.

`verify_v5.py`, June--August infinity parity (`42` rows), and `verify_v2.py`
all pass. The v2 downstream result is not promoted: W0→W1 is AAPL
`48.70%→51.98%` and AMZN `60.63%→44.43%`. The old v4 `+0.42/+0.78pp` and v1
reaction 120m/240m claims are superseded for current validity statements.

### Phase B implementation, not execution

`outputs/stock_specific_gate_4h/` contains `PRE_REGISTRATION.md`,
`IMPLEMENTATION_PLAN.md`, `common.py`, `prepare.py`, `run_gate.py`,
`verify.py` and `report.py`. It freezes `R1` and `F1_new`, the ten state
features, weighted G0--G3 ridge controllers, no-news exact R1 fallback and
August-frozen exposed evaluation. Static contract tests pass. No G0--G3
prediction, metric, oracle or exposed result has been generated. The exact
later command is:

```bash
work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/run_gate.py \
  --approve-gate-run --output outputs/stock_specific_gate_4h/v1
```

## Addendum correction and reaction v3 corrected checkpoint — 2026-09-18

The earlier v2 downstream protocol and the first v3 preregistration wording
incorrectly made AR2 availability a prerequisite for W0--W3.  ISSUE-041 is
registered in the working spec.  AR1 now has an independent full article-level
gate, including per-stock Brier `<=0.002`, macro AUC, positive macro-month,
coverage, completeness and non-constant checks.  AR2 remains optional and is
`NOT_RUN_MODEL_UNAVAILABLE` because the target-context FinBERT binary is absent.

The corrected run is `outputs/stock_reaction_features_4h/v3/`.  AR1 240m gains
both stocks' June--August BA but fails macro AUC (`-0.570pp`); AR1 60m also
fails.  Therefore no article candidate is eligible and the corrected run stops
before W0--W3.  The exact eligible downstream protocol is frozen in
`PRE_REGISTRATION_v3.md`; no exposed score selects a candidate.  The previous
article-only run is preserved as `v3_article_audit_checkpoint/`, and v2 remains
a historical protocol-mismatch artifact.

The repository remains waiting for explicit `APPROVE_GATE_RUN` before that
command may run.

## External review repair completion — reaction v4 and Phase B preflight (2026-09-18)

- **Starting SHA:** `8842d37a58a9ac3fa00c3ec9a56df9e917fa4e62`.
- **Ending SHA (repair payload):** `a69044096cb4ac5421381da1aa12ddc46d6b4f95`
  (`Repair reaction v4 and gate preflight`).
- **Issues registered and resolved:** ISSUE-042 through ISSUE-050. These cover
  the complete 30/60/120/240m candidate family, missing price-context
  representation, the narrow AAPL acronym association rule, the controller
  advantage target, exact R1 fallback, consistent median preprocessing,
  direction-disagreement headroom, R1 provenance, and real-path preflight
  verification. Earlier v3 remains a preserved `SUPERSEDED_PENDING_REPAIR`
  artifact; no historical result was overwritten.

### Reaction v4 result

The new `outputs/stock_reaction_features_4h/v4/` article-level run passed its
time-safety, completed-bar, future-price-perturbation, fold-local
preprocessing, no-public-text, entity-audit, candidate-completeness and gate
recording verifier. It fits all four preregistered AR1 horizons (240 fits,
76,248 article prediction rows). AR2 is explicitly
`NOT_RUN_MODEL_UNAVAILABLE`; it was not replaced with cached title vectors.

| AR1 horizon | AAPL mean June--August BA delta vs AR0 | AMZN delta | Result |
|---|---:|---:|---|
| 30m | -0.996pp | -0.811pp | fail |
| 60m | -0.351pp | -0.561pp | fail |
| 120m | +0.322pp | +2.570pp | fail: AAPL misses +1pp and Brier guardrail |
| 240m | +0.336pp | +0.653pp | fail: both below +1pp; AMZN constant-direction collapse |

No AR1 candidate passed its full independent promotion gate. Per protocol,
W0--W3 was not run and no downstream performance selected a horizon.

### Stock-specific reliability gate preflight

`outputs/stock_specific_gate_4h/verify.py` passed in preflight-only mode on the
real 1,607 canonical keys. The public R1 column now has maximum absolute
probability difference `1.1102230246251565e-16` from the traced private
`nextgen_4h/price_v1` issued prediction source. R1 and F1_new each pass
chronological provenance checks for 1,036 evaluated rows; September onward is
verified as August-frozen. Independent advantage recomputation, state-column
allowlist, same-median train/evaluation transforms, exact R1 support fallback,
G1→G0, G3→G2, G3→G2→R1 and G3 nesting all pass.

This is still `PREREGISTERED_NOT_RUN`. No G0--G3 predictions, metrics, oracle
ceilings, advancement results, development/later gate results, or approval
command execution exist. The only permitted next predictive action is an
explicit owner message containing `APPROVE_GATE_RUN`.

## Final code-only verification repair — 2026-09-18

- **Starting SHA:** `6a0942337e6d5ec76fd8677076052c06e3b7bb4f`.
- **Ending SHA (repair payload):** `422717eda924a7013742a504fec1482159da6c08`
  (`Harden Phase B verification contracts`).
- **Issues resolved:** ISSUE-051 through ISSUE-055. This was a code-only
  checkpoint. It did not run `run_gate.py --approve-gate-run` or create a
  `stock_specific_gate_4h/v1` result directory.

### Reaction v4 stronger verifier

The strengthened verifier changes all bars whose **end** is later than article
availability, so it attacks a bar that began before cutoff but was unfinished
at cutoff. A separate 10:02 synthetic case proves that the 10:00--10:05 bar is
excluded. `verify_v4.py` passes with this check; the existing v4 article
models and predictions were not rerun because the saved corpus still satisfies
the completed-bar contract.

### Phase B verification state

The real-input preflight passes again, including explicit regression tests that
0.70 versus 0.60 has a probability difference but no BA-routing direction
disagreement, while 0.70 versus 0.40 does. It also tests that the hindsight
oracle forces R1 on no-news rows. `verify.py` now has an automatic post-run
branch: after an approved v1 exists, it independently reconstructs prediction
mapping, no-news and exact-R1 fallbacks, metrics, monthly metrics, advancement,
chronology, preprocessing, and coefficient structure without using the
runner's metric or advancement helper.

No G0--G3 prediction, metric, oracle ceiling, advancement, development, later,
or post-run verification artifact exists. The repository remains
`PREREGISTERED_NOT_RUN` pending an explicit owner approval.

## Approved Phase B execution stopped by verifier runtime error — 2026-09-18

- **Starting SHA:** `2906a1f0791490e04e10588644bc410fb5515467`.
- **Ending SHA (stop-record payload):** `5c36e38ef5df1c7ce2f4b1fc10d15bcf448bb32f`
  (`Record unverified Phase B stop`).
- **Approved run command:**
  `work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/run_gate.py --approve-gate-run --output outputs/stock_specific_gate_4h/v1`.

The one frozen Phase B command completed and its local artifact is preserved.
The immediate required command,
`work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/verify.py`,
then failed with `NameError: clean is not defined` in the post-run result
serialization branch. It did not write `v1/verification.json`.

This is a verifier implementation failure, so the saved G0--G3 scores are
`UNVERIFIED_STOPPED`, not a valid experiment result. No report was generated;
no repair-and-rerun, score interpretation, post-result tuning, v2 run, or
activity-column experiment was performed. The exact public failure record is
`outputs/stock_specific_gate_4h/v1/verification_failure.json`; raw predictions
and model artifacts remain local.

## Verifier-only recovery of preserved Phase B v1 — 2026-09-18

- **Starting SHA:** `a8bdfb204a7b99fb8933024c448d31bdfb431488`.
- **Ending SHA (recovery payload):** `6cb101297e27a58ef930d0b7ba617a98e3bf5fe7`
  (`Correct Phase B recovery hash evidence`).
- **Authorized scope:** `run_gate.py` was not run again. The only code repair
  moved the pre-existing `clean(value)` helper from `preflight_main` into module
  scope so the preserved run could reach the pre-existing post-run verifier.
  No model, predictor, result calculation, or existing v1 result file changed.
- **Integrity proof:** the corrected recovery ledger records matching SHA-256
  hashes for all eleven v1 artifacts existing before recovery, before and after
  verifier execution. Its first `metrics.csv` entry accidentally omitted one
  final hexadecimal character and was corrected as metadata only; the preserved
  artifact was not changed.
- **Verifier command and outcome:**
  `work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/verify.py`
  writes `v1/verification.json=FAIL`. Seventeen checks pass; the sole failed
  check is `advancement_independently_reconstructed`.
- **Exact stop reason:** the verifier reconstructs advancement with full
  floating-point precision, while saved `advancement.json` was serialized by
  pandas `to_json` at approximately ten decimal digits. The `1e-12` record
  comparison rejects the resulting serialization difference.
- **Boundary:** this preserved run is
  `UNVERIFIED_STOPPED_PENDING_EXTERNAL_REVIEW`. No report, scientific score
  interpretation, tuning, v2 run, activity-column experiment, or result-file
  rewrite occurred.

## Precision-only verifier repair of preserved Phase B v1 — 2026-09-18

- **Starting SHA:** `d85ee004edd7f0bf5128e8bb591ef656d85966a1`.
- **Patch:** only `verify.py` now passes independently reconstructed
  advancement records through `pandas.DataFrame.to_json(orient="records",
  double_precision=10)` before comparing them to the existing
  `advancement.json`. It saves every raw numeric full-precision value, saved
  JSON value and absolute difference. All other strict comparisons remain
  unchanged.
- **Preserved prior failure:** `v1/verification_precision_failure.json` records
  the prior 17-pass/one-fail state. Earlier `verification_failure.json`, recovery
  metadata and handoff history remain present.
- **Hash proof:** the ten predictive/result artifacts match before repair,
  after verification and after report generation. `run_gate.py` was not rerun;
  neither `advancement.json` nor any prediction/result artifact was rewritten.
- **Precision evidence:** `raw_max_abs_difference=4.843306398299996e-11` and
  `storage_normalized_match=true` in `v1/verification.json`.
- **Final verifier state:** `PASS` with 18/18 checks. The authorized report
  command consumed only preserved v1 files.
- **Interpretation boundary:** all three unchanged G1/G0, G2/G1 and G3/G2
  promotion contrasts fail. No post-result tuning or activity experiment
  occurred. Development/later remain exposed exploratory historical backtests.

## Activity v1 preregistration and preflight checkpoint — 2026-09-18

- **Starting SHA:** `11b4f29f806fc486c1519f1cfcdef02c6336924f`.
- **Ending SHA (preflight payload):** `b7ff8606cc6bfef9163c8e8ae3170b1b22d912a3`
  (`Add Activity v1 preregistration and preflight`).
- **Semantics:** `SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY`. No authoritative
  field/vendor definition was found in local course receipts, profile or chart
  metadata. Cross-granularity sum agreement is saved as a descriptive audit,
  not a claim that activity is volume.
- **Raw source hashes:** AAPL 5-minute
  `059a29b0b24f438bd2d6f1be4102974f9b10631fd69b6c75ac2076930b0381cf`;
  AMZN 5-minute
  `ea5ec298fe39364fa3cb3d9dfdd46393c5490f46c9e2da7984f839953e380282`.
- **Frozen fields and coverage:** exactly 15/60-minute current `logmean`,
  history-only relative level and four missing flags. AAPL current/relative
  valid fractions are 66.63%/64.51% at 15m and 33.37%/32.25% at 60m; AMZN is
  66.54%/64.30% and 33.33%/32.21%.
- **A0 parity:** exact threshold direction parity with both canonical sources;
  max absolute issued-probability error is `1.1102230246251565e-16` for each.
- **Safety:** all 13 preflight checks pass, including completed-bar cutoff,
  strictly preceding reference sessions, 20/10 history limits and a synthetic
  10:02 unfinished-bar exclusion. A1/A1_matchedC result artifacts do not exist.
- **Exact later command — not run:**
  `work/stock-data/finbert-env/bin/python3 outputs/stock_activity_4h/run_activity.py --approve-activity-run --output outputs/stock_activity_4h/v1`

## Activity v1 final code-only repair handoff — 2026-09-18

- **Starting remote SHA:** `c63a9397d7af9e64d577883f2dc872e669487a6b`.
- **Payload SHA before this handoff:** `e190386d41c570b9f8fb0dcbf8082c45655e0215`.
- **Issues:** ISSUE-058 through ISSUE-061 are resolved code-only/preflight; ISSUE-062
  adds a verifier-local post-run reconstruction of selection, selected-fit chronology,
  metrics, advancement and attribution control. None has generated an A1 score.
- **Independent raw reconstruction:** all eight prepared fields across 1,607 canonical
  keys have zero NaN-pattern mismatches and max finite absolute error `0.0`.
- **Sources and coverage:** every frozen source fingerprint matches `sources.json`;
  stock-by-month 15m/60m current/relative coverage is saved in
  `outputs/stock_activity_4h/activity_feature_coverage_by_month.csv`.
- **Cross-granularity audit:** records complete aligned windows, exact-sum count/fraction,
  and maximum/mean discrepancies; its status remains
  `DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF`.
- **A0 parity and preflight:** both canonical R1 sources have max probability error
  `1.1102230246251565e-16`; all preflight checks pass.
- **Boundary:** runner, post-run verifier and PASS-gated report are implemented. The
  post-run verifier and report have not been exercised on A1; no A1/A1_matchedC output
  artifact exists. Status is `PREREGISTERED_NOT_RUN` pending an explicit approval.
