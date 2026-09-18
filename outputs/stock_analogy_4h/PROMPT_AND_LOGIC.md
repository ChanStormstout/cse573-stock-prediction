# What the LLM actually receives

This experiment implements prediction from historical analogues, not event annotation.
The exact system prompt is `SYSTEM` in [common.py](common.py); inference hashes the
full private messages. Source documents are treated as data, never instructions.

## Current window

- Target stock and cutoff, original target interval start/end.
- One eligible article's title and complete-sentence target excerpt (at most120 words).
- Publication time, availability time, publication age, collection delay.
- Six completed hourly returns and ranges, their mean/std, age and target time of day.

The current target opening price and return are absent. Old price bars can be stale,
especially at the market open; age is shown. No volume field is used.

## Historical cases (at most three)

P2 and P3 receive exactly the same case IDs, historical cutoffs and intervals, article
evidence, time metadata, price context and retrieval scores. P3 alone additionally
receives each historical case's program-calculated four-hour simple return in percent
and UP/DOWN label. Those outcomes are complete before the current cutoff and come
only from earlier training months. Cases are selected before examining their labels.

P0 receives only the current window. P1 has no LLM; it computes:

`p_up = (1 + sum(similarity_i * historical_up_i)) / (2 + sum(similarity_i))`

The symmetric pseudocounts shrink a handful of noisy cases toward0.5. They do not
create an empirical confidence interval or imply calibrated probability.

## Why use binary token scoring?

Previous free-form JSON forecasts had direction/number inconsistencies. This run asks
for UP or DOWN and computes their conditional probability from actual output logits.
It does not parse a generated narrative or treat a self-reported percentage as calibrated.
The LLM is prompted to compare important differences but does not emit reasoning.
Public case explanations come from assistant review, not hidden model reasoning.

## How to read the four comparisons

- P3 vs P2 isolates adding historical outcomes to the same retrieved text/context.
- P2 vs P0 measures retrieved historical context without observed outcomes.
- P3 vs P1 compares LLM analogy use with a simple vote, but P3 also receives current
  inputs; a difference is not pure proof of LLM reasoning in isolation.
- P0/P3 vs F0/F1/F2/R1 are system comparisons with different input representations.

No automatic model fusion or per-stock winner selection is performed. When retrieval
is empty, P2/P3 are exactly P0; P1 is exactly R1. When current evidence is absent,
all four are exactly R1. Coverage is part of the answer, not a reason to drop windows.

## Limits that the experiment cannot erase

Matching headline categories does not certify target roles, actors, units, quarters,
novelty or expectations. Grouping is approximate; overlapping market periods and
same-quarter reporting remain correlated. The library is small and fixed through
August, while conditions later change. A modern pretrained LLM may have seen the
historical events. A historical price move is not a causal response attributed to
the selected article. Better retrieval is a distinct future experiment, not a claim
that this finite run exhausts retrieval-augmented forecasting.
