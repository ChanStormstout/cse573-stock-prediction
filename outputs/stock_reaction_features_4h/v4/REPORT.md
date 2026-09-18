# Reaction v4 report

## Scope

v4 reruns all preregistered 30/60/120/240 minute AR0/AR1 article candidates with time-safe completed-bar context. Missing continuous context is imputed fold-locally and has explicit unscaled validity flags. AR2 is not run because its registered target-context FinBERT binary is unavailable.

## Article gate
- AR1 30m: FAIL; AAPL mean ΔBA=-0.009963980826808772, AMZN mean ΔBA=-0.008109184794076574, macro AUC Δ=-0.022630484362563475.
- AR1 60m: FAIL; AAPL mean ΔBA=-0.0035127134945165763, AMZN mean ΔBA=-0.005610488037690727, macro AUC Δ=-0.018185214483780993.
- AR1 120m: FAIL; AAPL mean ΔBA=0.00322184159521105, AMZN mean ΔBA=0.0256978332342853, macro AUC Δ=0.0002364034873149993.
- AR1 240m: FAIL; AAPL mean ΔBA=0.003364951864470352, AMZN mean ΔBA=0.006533751493428885, macro AUC Δ=-0.0008527163758109227.

## Downstream status

- No AR1 horizon passed its full preregistered article gate; W0--W3 was not run.

All values are exposed exploratory historical backtests. v3 is preserved and superseded; no Phase B G0--G3 execution occurred.
