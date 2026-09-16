# Development panel amendment, before expanded student fitting

The first 60 March candidates received `none` in both teacher passes. The broad
candidate regex often selected changes in fund holdings and historical broker
lists. This panel can measure false positives but cannot select an extractor for
recall. Preserve all 60 and their original labels. Do not change the 100 April
check inputs or use their answers to select this amendment.

Add every remaining March candidate group whose headline mentions its target
company and matches `price.target|target.price|upgrad|downgrad|reiterat|reaffirm|rating|\\bPT\\b`,
excluding headlines matching `Holder|Holding|Stake|Position|Sentiment|Earning.*Coverage`.
These are candidates, not positive labels. This deterministic content-only query
returns 38 AAPL candidates and no AMZN candidates. Annotate twice with unchanged
teacher instructions, adjudicate disagreements and audit agreements, then apply
the same event grouping. Preserve negatives and uncertainty. Record remaining
AMZN coverage limitations; do not claim per-company recall can be established
when the development panel has no AMZN positives.

Training remains January-February. Development is March; check is April. Earliest
extractor freeze remains May 1. The student is Qwen3-1.7B 4bit, rank8 on the last
8 q/v attention layers, seed573, 3 epochs, LR1e-4, completion-only loss and 2048
maximum tokens. Use the previously registered event-balanced loss (2/3 positive,
1/3 negative total weight) to address the prior empty-output collapse; no grid.
Select epoch on development facts only. Frozen and adapted use the same compact
prompt and deterministic decoding. All labels remain model-provisional.

## Separate April challenge supplement

The original April panel contains one AAPL and five AMZN positive candidates after dual review
(and one uncertain article). Keep its results separate. Apply the same previously
specified headline selector to the remaining April groups, yielding 39 AAPL and
zero AMZN candidates. These form an enriched extraction challenge, not an estimate
of natural prevalence and not an independent human gold set. Freeze the student
contract and training choices above before collecting these additional labels;
do not choose checkpoints, prompts or hyperparameters on this challenge. Report
original-check and enriched-challenge scores separately. The small AMZN positive check count remains an explicit limitation; do not
imply broad extraction acceptance from AAPL results. The two Credit Suisse
AMZN articles share an event and are not independent examples.
