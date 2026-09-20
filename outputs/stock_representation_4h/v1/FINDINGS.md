# Findings in plain language

**All three finite stages executed and passed fit-free prediction verification.**
These are exposed exploratory historical backtests. Ordinary random and
association-group random results are separate protocols, not interchangeable
estimates of future performance. No outer score selected a stock-specific winner.

## 1. Single preprocessing changes: no large reliable two-stock improvement

Three-seed mean BA, unchanged ordinary random partitions:

| Method | AAPL | AMZN |
|---|---:|---:|
| Original price + full-text word reference (FULL) | 67.57% | 60.28% |
| Fixed TF-IDF + calibrated SVM, identical PRICE fallback | 69.55% | 61.14% |
| Restore word frequency only | 70.13% | 60.61% |
| Preserve no / not / nor only | 69.59% | 61.36% |
| Add numeric expressions only | 68.69% | 61.59% |
| Disable stemming only | 69.09% | 60.35% |

Frequency's AAPL gain is driven by seed 574 (+2.77pp); seeds 573/575 lose
approximately 0.52pp each. It reduces AMZN BA in every seed. Therefore 70.13%
is an observed mean, not evidence of a robust improvement.

Negation improves the means by only 0.05pp /0.22pp. Numeric expressions help
AMZN by 0.45pp but hurt AAPL by 0.86pp. **Every one-day and five-day paired interval
for these preprocessing changes against the fixed SVM includes zero.** These
are descriptive intervals after exploration, not untouched significance tests.

The numeric branch devotes an average 1,284/5,000 AAPL and 974/5,000 AMZN
vocabulary entries to numbers. It changes vocabulary competition as well as
providing numerical tokens. The branch does not identify whose number it is,
what it measures, or whether it describes a new financial event.

These isolated effects do not explain the entire earlier raw-text loss:
that earlier package also changed cleaning, restored other stopwords and added
bigrams. The present ablations do not identify all interactions among those
changes . We did not sum their deltas as a causal decomposition or search more
combinations after seeing the results.

## 2. The strong random SVM is not robust to association-group isolation

Same three seeds; group-disjoint outer, inner and calibration partitions:

| Method | AAPL BA | AMZN BA |
|---|---:|---:|
| Price-only | 55.84% | 51.45% |
| Price + original full-text words (FULL) | 52.39% | 51.74% |
| TF-IDF + calibrated SVM, PRICE fallback | 49.23% | 51.17% |

AAPL SVM predicts up for 97.72% of all windows on average. Ordinary random SVM
predicts up for 58.41%. Its grouped failure includes a pronounced one-sided
prediction tendency, not just a small decline in its mean score.

**Observed:** scores change substantially under association-group isolation.
**Plausible interpretation:** shared/related articles and temporal associations
make ordinary random interpolation easier; the model does not transfer well
when those groups are held out together.
**Not proved:** an exact numerical leakage contribution, that the original
labels were wrong, or that all historical work is invalid. Group composition,
training distribution and the required group-disjoint calibration also change.
The 203 components are association groups, not independently certified events.

## 3. Mapping each chunk before averaging: small order effect, no useful pipeline win

All rows use the same word/price logistic-regression design, fixed C grid,
article membership and frozen FinBERT chunk cache. Geometry is fitted only from
the respective training partition. No new encoder inference or finetuning.

| Matched method | AAPL BA | AMZN BA | AAPL Brier | AMZN Brier |
|---|---:|---:|---:|---:|
| Word + price LR, no semantic block (SET_BASE) | 64.99% | 59.51% | .22118 | .23943 |
| Add original 768-dimensional mean | 65.42% | 59.41% | .21955 | .23899 |
| Average, then nonlinear mapping | 65.09% | 59.06% | .22085 | .24081 |
| Nonlinear mapping per chunk, then average | 65.22% | 59.27% | .22082 | .24049 |

Mapping before averaging gains 0.13pp /0.21pp against mapping afterward, but
changes the matched no-semantic reference by +0.23pp /-0.23pp. AMZN Brier also
worsens against that reference. It remains below the fixed strong SVM for both
stocks. **The preregistered follow-up budget screen is false.**

We therefore do not add price-state interactions, TabPFN, attention, another
encoder or a larger grid in this run. This is a bounded result about the tested
normalized 256-feature random mapping and linear head; it does not refute all
nonlinear distribution representations. Token pooling/paragraph selection had
already discarded some information before this experiment; the mapping cannot
recover that, nor create news for the 389 AMZN no-news windows.

## 4. Source-checked cases

Cases follow the registered hash-first panel; they were not selected for the
largest score improvement. Raw evidence and margin terms remain private.

- **AAPL frequency repaired,2018-02-16,51c4af5cdca2cf95a3190235:** the window
  contains device-crash coverage and repeated reports about headquarters glass
  collisions. Probability changes .5770→.4842, matching the down label. Bug/fix
  terms contribute negatively to the new average uncalibrated SVM margin.
  This illustrates repetition changing model input weight, not proof that either
  event caused the subsequent return.
- **AAPL frequency harmed,2019-01-31,fcf8e9b4266f9133a21f2bed:** several holdings
  reports repeat institutional investor names and background positions. Those
  names are among the strongest negative margin terms; probability changes
  .5014→.4537 despite an up label. Restoring counts also amplifies boilerplate.
- **AAPL numeric harmed,2018-05-08,b9880b404e762805301b7e3c:** source paragraphs
  show that prominent numeric features include a fund's portfolio allocation and
  an ETF's return, not uniformly a new Apple operating result. Probability
  changes .5123→.4785 despite an up label; selected C stays 1 in both models.
  Numerical presence alone does not establish correct financial attribution.
- **AMZN numeric repaired,2018-08-14,b125d1733f4cbe8a5cbe4250:** the article compares
  large technology companies' banking initiatives. Probability changes
  .4863→.5132, matching up, but selected C also changes .01→.1. This is not proof
  that a specific number supplied the correction.
- **AMZN negation repaired,2018-11-13,fdf738d71d1f83cd954bc682:** probability
  changes .5251→.3759, matching down. The added words do not dominate its top
  margin terms, and selected C changes 1→.1. The fitted-system change includes
  the normal inner-selection response; it cannot be called learned negation
  reasoning from this case alone.
- **AAPL negation harmed,2018-10-01,bdfc8acbbc69a32b5a880a56:** a marginal
  .49994→.50405 movement crosses the fixed threshold and creates an error.
  Small representation changes can change counts without substantial evidence.

Uncalibrated linear margin contributions are mathematical model contributions,
not probability attributions or causal market explanations. Shared correct/wrong
and semantic cases are retained in case_index.csv/CASE_NOTES.md; not every card
has received manual source review or independent human annotation.

## 5. Actual computation and decision

- 5,400 real predictive model-container fits and independent reloads.
- 9,000 internal SVM classifier fits +2,400 LR fits =11,400 classifiers;
  additionally 9,000 sigmoid calibration fits. These are computations, not
  independent samples.
- Preprocessing/grouped replay errors 0; aggregation maximum 3.33e-16.
- No new FinBERT/Modern encoder execution or training, no LLM continuation.
- Original article membership, four-hour labels, windows and historical models
  retained. Per-stage verification, source audit and final preservation evidence
  accompany this report.

Keep FULL as the declared reference and the old SVM + PRICE fallback as the
strong ordinary-random comparator. Negation is a small inexpensive candidate,
not an established improvement. Do not assemble an AAPL-frequency/AMZN-numeric
winner from exposed results. The major lesson is that performance under related
news held out together is still weak; more lexical detail or a different
aggregation order has not solved that.
