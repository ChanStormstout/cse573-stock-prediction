# Dense four-hour v5 repair report

## Purpose

This is a new repaired artifact.  The historical v4 directory is preserved.
The specific blocker was that the old executable comparison merged March--
August rows before calculating the June--August advancement gate.

## What was run

The v5 runner used the existing cutoff-safe dense pool, reconstructed the
official and augmented features with one generator because raw dense-to-
canonical parity did not hold, and fit D0_equal, D0_day and D1 on each
March--August fold plus the final exposed period.  It performed 42 persisted
fits with reload checks.  The corrected gate filters both sides first and
asserts exactly six stock-month cells.

## Corrected gate

| quantity | value |
|---|---:|
| AAPL mean D1−D0_day BA | +3.740 pp |
| AMZN mean D1−D0_day BA | −0.079 pp |
| macro mean delta | +1.831 pp |
| positive outer month means | 2 / 3 |
| gate | **FAIL** |

The AAPL gain does not generalize to AMZN, so dense sampling is not promoted
as a two-stock improvement.  The old v4 `+0.42/+0.78` values remain available
only as superseded historical diagnostics.

## Evidence boundary

The raw canonical parity check is recorded as false; this is why the matched
reconstruction mode is explicit in `dense_gate_corrected.json`.  `verification_v5.json`
checks this handling, the June--August filter, six-cell merge, fit count and
historical preservation.  All development/later numbers are exposed
exploratory historical backtests.
