# Reaction features v2 repair report

## Result in plain language

The repaired pipeline is time-safe and reproducible.  It found one article-
level candidate: AR1 at a 240-minute horizon.  When that signal was connected
to the original four-hour F1 model under the frozen W0--W3 protocol, it did
not improve both stocks.  Therefore it is not promoted as a project model.

## Article probe gate

| horizon | AAPL mean ΔBA | AMZN mean ΔBA | status |
|---:|---:|---:|---|
| 30m | +0.459 pp | −2.136 pp | FAIL |
| 60m | −0.451 pp | +0.758 pp | FAIL |
| 120m | +0.829 pp | +3.011 pp | FAIL (Brier condition) |
| 240m | +1.763 pp | +3.218 pp | PASS |

The gate required both stocks to have positive stable monthly gains, no
material Brier deterioration, complete outer months and sufficient coverage.

## Four-hour downstream

| stock | W0 fixed F1 | W1 240m | W3 strict fallback |
|---|---:|---:|---:|
| AAPL | 48.70% | 51.98% | 48.38% |
| AMZN | 60.63% | 44.43% | 55.95% |

W2 equals W1 because only one horizon passed.  W3 returns W0 exactly when
fewer than two valid reaction articles are available.

## Validity and limits

`verification_v2.json` passed checks for availability, reaction start and
maturity, completed-bar context, future-price perturbation invariance,
target-pair uniqueness, fold-local preprocessing, fit counts, promotion-gate
completeness and W3 fallback.  AR2 was not run because its required binary is
not available.  The association cards are assistant-generated deterministic
checks, not independent human review.  All periods remain exposed historical
backtests.

## Protocol status correction

This v2 directory is retained as a historical time-safety repair, but its
downstream W0--W3 implementation does **not** match the complete reviewer
addendum: it used the old F1 residual/fallback construction rather than the
registered W0=R1, W1=R1+coverage, W2=R1+five reaction fields, W3=F1+R1+the
same five fields protocol.  Its downstream numbers are therefore superseded
for current claims.  The corrected independent AR1/AR2 eligibility rule and
full gate are in the v3 preregistration; v3 stops before downstream because no
article candidate passes the full gate.
