# TabPFN 2.5 provenance and ensemble-size audit

## What was corrected

The old report called the checkpoint “synthetic-only”. The local TabPFN 6.3.0
package metadata and checkpoint archive show that this default classifier is
the `Prior-Labs/tabpfn_2_5` default and is fine-tuned on real data. The old
experiment also used `n_estimators=1`, which is a reduced-compute probe. We
therefore created a new run with the same checkpoint and inputs but the
library default `n_estimators=8`. The old n=1 results remain unchanged as the
matched historical control.

This is still not a reproduction of TabPFN pretraining or a claim that a
larger ensemble adds independent data. It is a finite, frozen-prior
conditioning comparison.

## Actual run

- 1,607 identical AAPL/AMZN four-hour windows and the original cutoff.
- Own-price and past-only cross-stock price inputs, matching the v1 probe.
- 14 chronological fit/evaluation records per branch (March–August and the
  final September-onward freeze); no development/later selection.
- TabPFN 6.3.0 local runtime, CPU, random state 573, one preprocessing job,
  telemetry disabled, `n_estimators=8`.
- 28 fits total, 61.4 seconds own-price and 83.5 seconds cross-stock. No
  gradient training; each fit conditions the frozen prior on past rows.
- Checkpoint SHA-256:
  `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`.
- Fresh-model reload checks passed; maximum first-three-row probability
  difference was `1.79e-7`.

## BA / Brier comparison

| period | stock | v1 own n=1 | v2 own n=8 | v1 cross n=1 | v2 cross n=8 |
|---|---|---:|---:|---:|---:|
| development | AAPL | 55.25% / .2563 | 54.77% / .2584 | 53.17% / .2590 | 51.09% / .2623 |
| development | AMZN | 57.42% / .2572 | 53.25% / .2573 | 52.32% / .2509 | 48.98% / .2580 |
| later | AAPL | 45.65% / .2697 | 49.15% / .2695 | 50.17% / .2677 | 49.20% / .2661 |
| later | AMZN | 52.28% / .2509 | 54.37% / .2504 | 54.35% / .2498 | 50.14% / .2511 |

Training-period forward OOF BA for own n=8 was AAPL `53.85%` and AMZN
`48.54%`; cross n=8 was `52.04%` and `50.09%`. The n=8 ensemble does not
show a stable two-stock or cross-period improvement under this fixed
chronological comparison, so no combination or further TabPFN grid was run.
The v2 protocol did not save the exact goal60 promotion formula as a
machine-readable gate, so this report does **not** claim a formal promotion
pass/fail; the statement is descriptive.

## Safe conclusion

**Observed:** the previous synthetic-only label was incorrect; n=8 was run
with the verified real-data-fine-tuned default checkpoint and did not show a
stable two-stock improvement under the fixed comparison. No formal promotion
decision is claimed because the exact gate was not saved by v2.

**Not established:** that TabPFN is ineffective in general, that the real-data
fine-tuning is harmful, or that another checkpoint/feature representation
would improve this task. The experiment is a corrected local probe only.
