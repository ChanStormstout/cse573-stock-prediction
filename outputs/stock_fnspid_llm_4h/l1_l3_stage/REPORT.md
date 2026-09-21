# L1-L3 stage result: FNSPID information control

## Scope

This owner-requested checkpoint evaluates L1-L3 before L4 finishes. It uses the
same canonical 1,607 four-hour rows, augmented course + direct-target FNSPID
news, chronological expanding training, and past-only parameter selection from
the registered L1-L5 lane. L4 prompts and outputs were not changed after these
results were observed. All development and later periods are already exposed
exploratory historical backtests.

## Plain-language result

L1 filtering, L2 fact-change correction, and L3 historical-case voting do not
provide a stable improvement across time.

- **L1 quality-filtered FinBERT** improves AAPL in March-August forward OOF,
  but barely changes AMZN and violates the probability-error guardrail. Its
  gains do not persist into the later period.
- **L2 fact-change residual** is worse than the price reference in both stocks
  during forward OOF and later history.
- **L3 historical-case vote** passes the registered March-August screen and
  improves both stocks in September-October, but reverses sharply in
  November-February. The training-period gate therefore selected a rule that
  did not remain valid in the next exposed period.

## Balanced accuracy

| Period | Method | AAPL | AMZN |
|---|---|---:|---:|
| Mar-Aug forward OOF | Recent-price reference | 52.68% | 52.18% |
| Mar-Aug forward OOF | Unfiltered augmented FinBERT | 50.69% | 49.39% |
| Mar-Aug forward OOF | Original-course FinBERT | 50.02% | 48.14% |
| Mar-Aug forward OOF | L1 quality-filtered FinBERT | 54.82% | 52.21% |
| Mar-Aug forward OOF | L2 fact-change residual | 51.41% | 51.13% |
| Mar-Aug forward OOF | L3 historical-case vote | 53.49% | 54.87% |
| Sep-Oct development | Recent-price reference | 56.21% | 51.11% |
| Sep-Oct development | Unfiltered augmented FinBERT | 57.71% | 58.53% |
| Sep-Oct development | Original-course FinBERT | 54.13% | 46.29% |
| Sep-Oct development | L1 quality-filtered FinBERT | 55.81% | 53.25% |
| Sep-Oct development | L2 fact-change residual | 50.43% | 49.44% |
| Sep-Oct development | L3 historical-case vote | 58.77% | 54.92% |
| Nov-Feb later | Recent-price reference | 53.87% | 50.85% |
| Nov-Feb later | Unfiltered augmented FinBERT | 53.84% | 48.61% |
| Nov-Feb later | Original-course FinBERT | 56.71% | 54.27% |
| Nov-Feb later | L1 quality-filtered FinBERT | 53.37% | 47.66% |
| Nov-Feb later | L2 fact-change residual | 48.21% | 49.62% |
| Nov-Feb later | L3 historical-case vote | 47.59% | 48.08% |

## Registered screen

Against the recent-price reference on March-August forward OOF:

| Candidate | AAPL delta BA | AMZN delta BA | Macro delta BA | Positive months | Worst delta Brier | Pass |
|---|---:|---:|---:|---:|---:|---:|
| L1 | +2.13 pp | +0.03 pp | +1.08 pp | 4/6 | +0.0117 | No |
| L2 | -1.27 pp | -1.05 pp | -1.16 pp | 1/6 | +0.0171 | No |
| L3 | +0.81 pp | +2.69 pp | +1.75 pp | 4/6 | -0.0003 | Yes |

The stage-only guarded choice is therefore L3. This is a faithful training-only
decision, but it is not a good final method: later BA falls by 6.27 points for
AAPL and 2.77 points for AMZN versus price. The likely interpretation is that
the relationship between FinBERT similarity and subsequent returns changed
over time; this is an observation about instability, not proof of a causal
mechanism.

## Error changes relative to price

In September-October L3 repairs/introduces 12/8 AAPL errors and 12/9 AMZN
errors. In November-February it repairs/introduces only 6/17 AAPL errors and
13/18 AMZN errors. The mechanism continued to change decisions but no longer
knew reliably which changes were beneficial.

## Verification

Independent reload verification is PASS across canonical rows, source hashes,
chronology, article-score completeness, past-only C selection, exact L2
fallback, stage choice, and probability bounds. The maximum selected-model
probability replay error is `2.22e-16`.

This stage result was produced before L4 completion at the owner's request. It
does not authorize changing the already frozen L4 prompts or using exposed
development/later scores to tune L4.
