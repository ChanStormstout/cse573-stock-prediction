# Method validity and claim audit (Stage 5)

## Scope

This is a claim-only audit. It reads saved public calibration manifests and
current-facing documentation; it does not fit a model, run inference, change
predictions, or select a method using development/later scores. The historical
outputs remain unchanged.

## Calibration finding

The early unconstrained Platt manifest contains **32 negative
slopes among 420 saved calibration records**. All 32 negative records
come from `outputs/stock_nextgen_4h/runs/v1/CALIBRATION_FITS.json` and are
AMZN records. A negative coefficient reverses the ordering of the raw score,
so this mapping is not a simple monotone calibration. The later
`platt_shrunk`/temperature/positive-slope manifests contain no negative
slopes in this audit and are the safer interpretation for probability quality.

This does not change any BA result. It narrows the claim: early Platt values
are historical score remappings, while the later constrained calibration is
the interpretable probability-calibration probe.

## Claim corrections

| issue | method | status | safe interpretation |
|---|---|---|---|
| ISSUE-012 | masked-reconstruction SSL | VALID_WITH_CAVEAT | small SSL pilot only; not TS2Vec |
| ISSUE-013 | historical analogy | VALID_WITH_CAVEAT | lexical/coarse historical analogy probe |
| ISSUE-014 | Event Adapter | VALID_WITH_CAVEAT | target evidence/action extraction probe |
| ISSUE-015 | Chronos-2 | VALID_WITH_CAVEAT | frozen endpoint-feature probe |
| ISSUE-016 | early unconstrained Platt | AUDITED | 32 of 84 saved early Platt fits have negative slopes; later constrained maps are the interpretable calibration |
| ISSUE-017 | dissemination/event clustering | VALID_WITH_CAVEAT | local gain is regularization-confounded |
| ISSUE-018 | Qwen direct UP/DOWN | VALID_WITH_CAVEAT | conditional token preference, not calibrated probability |
| ISSUE-019 | cross-stock past state | VALID | uses only peer state available before the cutoff |
| ISSUE-020 | continuous-return auxiliary | VALID_WITH_CAVEAT | one shared direction-plus-return objective lacked stable promotion |
| ISSUE-021 | F1/F2 controls | DOCUMENTED | historical controls are separate from reselected F1_new/F2_new |

See `METHOD_CLAIM_AUDIT.csv` for the corresponding unsafe wording to avoid.
The small masked-reconstruction encoder is not a TS2Vec reproduction; the
analogy run is not FinSeer; the Event Adapter is not a Ding graph/event
embedding; the Chronos result is an endpoint probe; Qwen direct scores are
token preferences; and continuous-return C2 does not show that return
magnitude lacks information.

## Verification artifacts

- `CALIBRATION_SLOPES.csv` contains the exact public slope records audited.
- `METHOD_CLAIM_AUDIT.csv` is the machine-readable safe/unsafe claim table.
- `verification.json` records counts and the no-fit scope.

No independent human event-review gate is claimed to have passed. All project
development/later numbers remain exposed historical backtests.
