# FNSPID-augmented LLM information-control experiment

## Status and question

This protocol was saved after the unfiltered augmented v1 results were known
and before any new LLM score or downstream fit in this lane. All results are
exploratory. The question is whether a frozen small LLM can control or interpret
the added FNSPID information better than unfiltered augmentation.

## Data and chronology

- Preserve the canonical 1,607 AAPL/AMZN four-hour rows, labels, cutoffs and
  price features from `stock_fnspid_augmented_4h/v1`.
- Use the union of canonical course news and direct-target FNSPID news under the
  same next-XNYS-open availability and three-session lookback.
- January-February is warmup; March-August uses expanding monthly forward OOF;
  September-October is exposed development; November-February is exposed later.
- Every retrieval candidate and historical outcome in a prompt must precede the
  query cutoff. September onward uses a case bank ending August 2018.
- The frozen LLM is `mlx-community/Qwen3.5-9B-4bit` at revision
  `8b2b98c00a6b4d291155e4890773ca8f769aee53`. It is not fine-tuned.

## L1: LLM quality filter

Score each deduplicated FNSPID article once using target-company evidence and
earlier same-stock titles. A binary prompt contrasts a specific target-company
fact with recap, holdings template, ranking, repeat, unrelated or uncertain
content. Keep an added article when `p(KEEP) >= 0.60`. Canonical course news is
retained. Fit the same price + frozen FinBERT PCA16 classifier as the unfiltered
augmented counterpart with C in `{0.01, 0.1, 1}` selected from earlier OOF
months.

## L2: LLM fact-change residual

Two additional binary scores estimate whether the article adds a new/changed
fact relative to retrieved prior reports and whether its explicit change is
positive versus negative for the target. Aggregate at most ten fixed features
per window: accepted count, quality mean/max, novelty mean/max, positive-minus-
negative mass, positive mass, negative mass, repeat mass and FNSPID count.

Fit a strict residual correction:

`logit(p_final) = logit(p_price) + g * beta'z`.

`g=0` when no article passes L1; those probabilities must equal the price model
bit-for-bit. Select C from `{0.01, 0.1, 1}` using earlier OOF months.

## L3: non-LLM historical-case vote

Retrieve same-stock prior windows by cosine similarity of augmented FinBERT
window means. Compare `k` in `{3, 5, 10}` and correction alpha in
`{0, 0.25, 0.5}` using earlier OOF months. The case vote is similarity-weighted
and never uses the query outcome. This is the required comparator for L4.

## L4: historical cases plus LLM

For each evaluated row, show the frozen LLM the query's accepted/current news,
three prior same-stock cases, their already observed four-hour directions and
brief news titles. Obtain a binary UP/DOWN score. Combine it with the price base
using alpha in `{0, 0.25, 0.5}` selected from earlier OOF months. No query label
appears in a prompt.

## L5: guarded complete system

Each L1-L4 component is screened on March-August OOF only. It passes only if:

- macro mean BA gain over PRICE_R1 is at least +1 percentage point;
- each stock's mean BA delta is positive;
- macro BA delta is positive in at least four of six months;
- each stock's Brier increase is at most 0.002;
- neither stock collapses to a constant direction.

If no component passes, L5 equals PRICE_R1 exactly. If one passes, L5 uses that
component. If several pass, choose by higher macro OOF BA, lower macro Brier,
then the fixed order L1, L2, L3, L4. Development/later performance never
selects a component. No per-stock exposed-period winner is allowed.

## Reporting and stop rules

Report L1-L5 with PRICE_R1, unfiltered augmented FinBERT and original-course
FinBERT. Report BA, accuracy, MCC, Brier, AUC, coverage, monthly metrics and
changed/repaired/introduced errors. LLM output is a model-derived feature, not
human gold. Stop expansion after this finite matrix; do not add prompt variants,
thresholds, retrieval models, RL or attention based on observed scores.
