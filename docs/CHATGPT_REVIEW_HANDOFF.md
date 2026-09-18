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
