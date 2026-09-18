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

## Reviewer addendum pause (2026-09-18)

The external reviewer reported six coarse blocking validity issues in the
earlier `e32785d` dense/reaction implementation. They are recorded as
ISSUE-022 through ISSUE-027 in `docs/CODEX_WORKING_SPEC.md`, with the detailed
requirements split into ISSUE-028 through ISSUE-040: dense executable gate
month range, unfinished-bar reaction context, the W0–W3 protocol, the complete
promotion gate, AR1/AR2 preprocessing and representation, reaction time-safety
verification, manifest auditability, and a new-run fingerprint.

Execution is paused pending the detailed addendum. No experiment or model
execution was started after this notice; no historical output was deleted,
reset, or overwritten. The prior verifier outputs remain preserved as
historical artifacts and are not treated as final validity evidence for the
affected paths until the addendum is incorporated and the repairs are rerun.
