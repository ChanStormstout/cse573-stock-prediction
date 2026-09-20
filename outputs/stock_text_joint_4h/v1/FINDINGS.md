# Findings in plain language

**All finite models completed and passed verification. These are exposed exploratory random historical backtests, not unseen-future scores.**

## What was actually trained

A1/A2/A3 learned from the same original articles after restoring ordered words,
repeated words, negation and numeric expressions. A1 is unigram LR, A2 adds
within-sentence bigrams, and A3 uses calibrated SVM on A2's representation.
B1/B2 jointly trained the original FULL price/word features with frozen
FinBERT target-company paragraph vectors, using PCA16 versus all 768 dimensions.
There was no encoder finetuning or LLM inference.

The run made 4,043 top-level fit calls, representing 5,243 classifier fits
(including the three classifiers inside each SVM calibration call), plus 1,800
sigmoid calibrators. Thirty-seven outer B configurations selected rho=0 and
reused FULL instead of fitting redundant classifiers. These counts describe
computation, not independent samples. Classifier fitting/predicting/checkpoint
serialization took 3,394.6 seconds; end-to-end training took 3,663.4 seconds.

## Main comparison: three-seed mean BA

| Method | AAPL | AMZN |
|---|---:|---:|
| Fixed reference: price + original full-text words (FULL) | 67.57% | 60.28% |
| Existing TF-IDF + calibrated SVM | 69.49% | 60.12% |
| Existing SVM + identical no-news PRICE fallback | **69.55%** | **61.14%** |
| A1: restored raw unigrams + LR | 61.71% | 61.06% |
| A2: restored unigrams/bigrams + LR | 59.21% | 60.26% |
| A3: restored unigrams/bigrams + SVM | 63.28% | 61.16% |
| B1: FULL + paragraph semantics, PCA16 | 67.59% | 59.96% |
| B2: FULL + paragraph semantics, 768 dimensions | 65.98% | 60.40% |
| FULL probability shrinkage | 67.57% | 60.28% |

### 1. Restoring information did not improve both stocks

A3 gains 0.88 percentage points over FULL for AMZN, but loses 4.29 points on
AAPL. Against the **fallback-matched** old SVM, its AMZN gain is only 0.023
points while AAPL loses 6.27 points. Adding bigrams to raw LR makes AAPL worse,
not better. This is evidence against the current complete preprocessing package;
it does not prove that negation or numbers are intrinsically unhelpful.

Several aspects changed together: stemming, stopword removal, repetition,
numbers, vocabulary competition and fallback-aware selection. The restored
view also keeps more auxiliary companies and repeated holdings/template text.
The inspected cases support this as a plausible source of noise, not a measured
causal explanation of the score decline. Conservative segmentation also splits
abbreviations; attached monetary/percentage tokens are modeled, while standalone
punctuation/currency symbols are not independent features. No claim of complete
financial relation understanding is made.

### 2. Direct semantic joint training did not produce a useful direction gain

B1 changes AAPL by +0.02 points and AMZN by -0.32 points versus FULL. B2 changes
AAPL by -1.59 points and AMZN by +0.13 points. B2 has worse Brier for both stocks.
Removing PCA16 therefore did not solve the current bottleneck. This does not
prove that all semantic representations are useless: only this frozen input,
aggregation, linear head and finite regularization design were tested.

### 3. Simple confidence shrinkage explains an important alternative

| Model | AAPL Brier | AMZN Brier |
|---|---:|---:|
| FULL | 0.2228 | 0.2524 |
| FULL + paragraph FinBERT probability fusion | 0.2181 | 0.2447 |
| FULL + paragraph Modern probability fusion | 0.2182 | 0.2414 |
| **FULL shrinkage only** | **0.2133** | **0.2382** |
| Existing SVM + matched fallback | 0.2054 | 0.2345 |

Shrinkage preserves every direction, has no new news information, and improves
Brier more than either prior paragraph fusion on both stocks. Consequently,
those fusion Brier gains alone cannot establish useful new semantics. This is a
successful alternative control, not a proof that every component of the old
fusion improvement has been causally decomposed.

### 4. AMZN has a large unchangeable error block in this news-only extension

AMZN has 389/804 no-news windows. FULL makes a mean 172.67 errors there across
seeds, out of 318.33 total errors: approximately **54.2% of its errors**. Every
new A/B method deliberately preserves the same PRICE decisions on these rows.
Three further windows have news but no qualifying target paragraph, so B1/B2
cannot change those either. These are error counts, not BA contributions.

On the 412 target-paragraph AMZN windows, A3 repairs a mean 39.67 FULL errors
and introduces 32.00, whereas B1 repairs 4.67 and introduces 7.33. Coverage and
reliability of corrections are distinct constraints; more text is not itself a
solution to either.

### 5. What to retain

Keep FULL as the declared main reference and the **existing SVM with matched
PRICE fallback** as the strong complete-pipeline comparator. This uses one rule
for both stocks and involves **no new SVM training** in the matched control.
Its +1.98/+0.86 point gains over FULL are descriptive point estimates; all
reported day/group paired intervals include zero. Do not label it a confirmed
stable improvement or a fresh holdout winner.

No new A/B mechanism beats FULL and the strong SVM on both stocks. A3's tiny
AMZN advantage over the matched control does not justify its large AAPL loss.
No per-stock test-selected combination was created.

## Next step, proposed only

First isolate the raw-word representation changes with a small matched study:
keep the classifier, fallback, vocabulary budget and folds fixed, and separate
binary versus frequency weighting from stemming/stopword choices while retaining
negation. This would test why the raw representation lost AAPL performance.
The current paired intervals do not establish a stable SVM improvement; an
association-group split is also a useful separate robustness check for that
fixed comparator. Neither study has been run in this task.

Do not expand to more encoders, attention, LoRA or LLM routing on the basis of
these results. The full finite task stops here. Raw source evidence remains
private; case review is assistant analysis, not independent human gold.
