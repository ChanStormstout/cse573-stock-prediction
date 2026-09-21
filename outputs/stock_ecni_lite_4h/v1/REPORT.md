# ECNI-lite four-hour combination

**Status:** verified exposed exploratory chronological historical backtest. All periods have already influenced project development.

## What was tested

The canonical price + full-body sparse probability was held fixed. Small gated residual corrections tested dissemination metadata, provisional company-event updates, past-only analogous reactions, and an existing saved LLM outcome-response score. When no eligible evidence exists, the output equals the baseline probability exactly.

Training-period selection chose **Price + full-body sparse baseline**. This choice used only the preregistered June-August advancement screen.

## Main results


### Training forward OOF

| Method | AAPL BA | AAPL Brier | AMZN BA | AMZN Brier |
|---|---:|---:|---:|---:|
| Price + full-body sparse baseline | 50.30% | 0.2593 | 53.27% | 0.2606 |
| Baseline + dissemination metadata | 52.96% | 0.2599 | 52.92% | 0.2629 |
| Baseline + provisional event updates | 48.29% | 0.2676 | 53.52% | 0.2603 |
| Baseline + past reaction analogies | 52.50% | 0.2641 | 53.27% | 0.2670 |
| Baseline + analogies + saved LLM response | 52.27% | 0.2607 | 53.02% | 0.2655 |
| Baseline + dissemination + updates + analogies | 52.37% | 0.2977 | 51.89% | 0.2738 |
| Baseline + all shared blocks | 54.48% | 0.2825 | 51.89% | 0.2696 |
| All blocks + shrunk stock contrast | 49.76% | 0.2817 | 52.09% | 0.2649 |

### Development (exposed)

| Method | AAPL BA | AAPL Brier | AMZN BA | AMZN Brier |
|---|---:|---:|---:|---:|
| Price + full-body sparse baseline | 57.94% | 0.2500 | 55.66% | 0.2481 |
| Baseline + dissemination metadata | 53.65% | 0.2562 | 51.95% | 0.2526 |
| Baseline + provisional event updates | 55.60% | 0.2500 | 55.66% | 0.2479 |
| Baseline + past reaction analogies | 56.57% | 0.2501 | 56.68% | 0.2477 |
| Baseline + analogies + saved LLM response | 57.05% | 0.2518 | 56.68% | 0.2478 |
| Baseline + dissemination + updates + analogies | 53.75% | 0.2602 | 52.41% | 0.2580 |
| Baseline + all shared blocks | 54.61% | 0.2611 | 52.41% | 0.2585 |
| All blocks + shrunk stock contrast | 53.40% | 0.2738 | 50.74% | 0.2519 |

### Later (exposed)

| Method | AAPL BA | AAPL Brier | AMZN BA | AMZN Brier |
|---|---:|---:|---:|---:|
| Price + full-body sparse baseline | 51.74% | 0.2587 | 54.36% | 0.2583 |
| Baseline + dissemination metadata | 50.77% | 0.2630 | 53.37% | 0.2547 |
| Baseline + provisional event updates | 48.00% | 0.2763 | 54.89% | 0.2574 |
| Baseline + past reaction analogies | 51.74% | 0.2578 | 54.89% | 0.2570 |
| Baseline + analogies + saved LLM response | 51.17% | 0.2580 | 54.89% | 0.2568 |
| Baseline + dissemination + updates + analogies | 45.77% | 0.3209 | 52.08% | 0.2498 |
| Baseline + all shared blocks | 46.37% | 0.3217 | 52.08% | 0.2498 |
| All blocks + shrunk stock contrast | 48.67% | 0.3358 | 50.66% | 0.2541 |

## Preregistered advancement screen

| Method | AAPL mean dBA | AMZN mean dBA | Positive macro months | Pass |
|---|---:|---:|---:|:---:|
| Baseline + dissemination metadata | +3.55 pp | -2.58 pp | 2/3 | no |
| Baseline + provisional event updates | -4.34 pp | +0.00 pp | 1/3 | no |
| Baseline + past reaction analogies | +1.91 pp | +0.00 pp | 2/3 | no |
| Baseline + analogies + saved LLM response | +1.12 pp | -0.42 pp | 1/3 | no |
| Baseline + dissemination + updates + analogies | +0.51 pp | -5.44 pp | 1/3 | no |
| Baseline + all shared blocks | +1.83 pp | -5.61 pp | 1/3 | no |
| All blocks + shrunk stock contrast | -2.98 pp | -4.30 pp | 1/3 | no |

## Evidence coverage

| Block | AAPL windows | AMZN windows |
|---|---:|---:|
| A | 165 | 36 |
| D | 797 | 415 |
| L | 161 | 36 |
| U | 480 | 30 |

## Interpretation

- Passing the advancement screen would mean a feature block improved both stocks during June-August under the frozen rule. It would still require a new untouched period for confirmation.
- Failing the screen means the combined representation did not earn promotion; exposed development or later scores cannot reverse that decision.
- Event facts are provisional model outputs rather than independently reviewed gold. Historical reactions are associations and do not establish causality.
- The saved LLM control reuses earlier outputs. No new LLM call was made.

## Verification

Fit-free reload/replay: **PASS**. Maximum probability error `0`; fallback errors `0`; verifier fit calls `0`.
