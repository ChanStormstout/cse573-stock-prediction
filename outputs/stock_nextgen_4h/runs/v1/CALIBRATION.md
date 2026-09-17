# LLM calibration

Calibration type was selected globally from March--August past-only forward predictions by Brier, then BA. Fits themselves remain stock-specific and past-only.

The private inference ledger preserves every observed batch score. For the controlled sequence P0→P1→P2→P3, a byte-for-byte identical prompt carries forward the preceding variant's normalized score; this prevents batch-composition numerical drift from being counted as content selection, added context or deduplication. `LLM_SCORES.csv` contains both `*_observed` and normalized `*_raw` values.

| Variant | Selected calibration | Mean OOF BA | Mean OOF Brier | Mean OOF ECE |
|---|---|---:|---:|---:|
| P0 | platt | 52.30% | 0.2480 | 0.1004 |
| P1 | platt | 49.92% | 0.2488 | 0.0955 |
| P2 | temperature | 50.35% | 0.2490 | 0.1123 |
| P3 | temperature | 49.81% | 0.2489 | 0.1035 |

Global paragraph choice after strict R1 fallback: **P0**. The choice used the registered P0 Brier guardrail and no later-period labels.

## No-news policies on training-period forward predictions

`policy_raw` keeps the calibrated model on news rows and uses the raw LLM prior on no-news rows. `policy_price` uses strict R1 fallback. `policy_learned` uses a past-only no-news Platt fit when enough examples exist, otherwise R1.

| Variant | Policy | Mean OOF BA | Mean OOF Brier | Mean OOF ECE |
|---|---|---:|---:|---:|
| P0 | policy_raw | 53.09% | 0.2469 | 0.1211 |
| P0 | policy_price | 53.53% | 0.2461 | 0.1034 |
| P0 | policy_learned | 52.28% | 0.2449 | 0.1078 |
| P1 | policy_raw | 50.72% | 0.2484 | 0.1152 |
| P1 | policy_price | 51.16% | 0.2476 | 0.1084 |
| P1 | policy_learned | 49.91% | 0.2464 | 0.1029 |
| P2 | policy_raw | 50.35% | 0.2491 | 0.1398 |
| P2 | policy_price | 50.79% | 0.2482 | 0.1218 |
| P2 | policy_learned | 49.54% | 0.2471 | 0.1159 |
| P3 | policy_raw | 49.81% | 0.2490 | 0.1297 |
| P3 | policy_price | 50.25% | 0.2482 | 0.1146 |
| P3 | policy_learned | 49.00% | 0.2470 | 0.1116 |
