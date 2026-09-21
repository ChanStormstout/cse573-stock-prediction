# FNSPID-augmented chronological four-hour replay v1

## Objective

Rerun the principal four-hour methods after augmenting the original course-news
windows with FNSPID historical news. This is one end-to-end augmented-data
experiment, not a sequence of post-result feature tweaks.

## Fixed data policy

- Preserve all 1,607 canonical AAPL/AMZN windows, labels and price features.
- Retain every original course-news article selected by the canonical pipeline.
- Add only FNSPID `DIRECT_TARGET_HIGH_CONFIDENCE` records independently relinked
  to AAPL or AMZN. Multi-company and indirect strata are excluded from v1.
- Collapse FNSPID records by URL, normalized title, near-title fingerprint, or
  stable record identity in that order.
- Every relevant FNSPID timestamp is date-only. An article is first available
  at the next XNYS regular-session open strictly after its recorded date.
- A window receives deduplicated FNSPID groups available by its cutoff during
  the current and preceding two XNYS sessions. Exact normalized-title matches
  to an original course article already in that window are removed.
- Sparse models use title and full available body. Frozen semantic encoders use
  the article title to match the original article-embedding protocol.

The relinker has not passed independent human semantic review. This run is an
owner-authorized exploratory augmented-data backtest, not a validated data
quality claim.

## Frozen chronology

- January-February 2018: initial training history.
- March-August 2018: expanding monthly forward OOF folds.
- September-October 2018: exposed development, predicted by the model frozen
  after August.
- November 2018-February 2019: exposed later backtest, predicted by the same
  frozen model.

Candidate `C` values are `0.01, 0.1, 1`. Each issued value is selected using
completed earlier forward months only by higher mean BA, lower mean Brier and
then smaller C. March defaults to `0.1`. All vocabulary, scaling, PCA and SVM
calibration are fit inside earlier data only.

## Fixed method matrix

1. `PRICE_R1`: unchanged cutoff-safe price reference.
2. `AUG_FULL_LR`: price plus binary full-body words from course + FNSPID.
3. `AUG_TFIDF_SVM`: augmented TF-IDF text with exact price fallback only when
   neither source provides news.
4. `AUG_FINBERT_LR`: price plus frozen FinBERT article means.
5. `AUG_FINMODERN_LR`: the same design using corrected FinModernBERT v2 pooling.
6. `AUG_EVENT_META`: FinBERT event-group means plus article/event/source/age
   metadata and price.

The original-data probabilities from `stock_forward_replay_4h/v1` are fixed
comparators. No LLM, per-stock post-result winner, development/later tuning or
post-result ensemble is allowed in v1.

## Reporting

Report input coverage, number of added deduplicated groups, BA, accuracy, MCC,
Brier, AUC, direction balance and monthly results. Compare augmented and
original versions on the same window keys. Development and later remain exposed
exploratory historical backtests.
