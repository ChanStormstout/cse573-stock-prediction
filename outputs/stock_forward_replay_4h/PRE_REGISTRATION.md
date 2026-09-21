# Chronological forward replay v1

## Question

Re-run the main existing four-hour methods under a genuine past-to-future
setting. Random-fold scores are not used or mixed into this experiment.

## Frozen chronology

- January--February 2018: initial training history; no fabricated OOF score.
- March--August 2018: six expanding monthly forward folds.
- September--October 2018: development, predicted by a model frozen after
  August.
- November 2018--February 2019: later exposed historical backtest, predicted by
  the same frozen model.

For every fold, vocabulary, scaling, PCA, SVM calibration and classifier fitting
use only earlier rows. Candidate `C` values are `0.01, 0.1, 1`. The issued C for
a month is chosen only from already completed earlier forward months by higher
mean BA, lower mean Brier and then smaller C. The March default is `0.1`.

## Fixed method matrix

1. `PRICE_R1`: old and recent cutoff-safe price features, L2 LR.
2. `FULL_LR`: price plus binary full-body word features (the freshly rerun J0
   branch from `stock_paper_methods_4h/v3`).
3. `TFIDF_SVM`: TF-IDF full-body text with calibrated linear SVM; exact
   `PRICE_R1` fallback when the original course window has no news.
4. `FINBERT_LR`: frozen FinBERT article means, training-fold PCA16 and LR (the
   freshly rerun J2 branch).
5. `FINMODERN_LR`: identical article membership, pooling, PCA16, price columns,
   metadata and LR; only the frozen encoder vector source changes.
6. `EVENT_META`: existing near-duplicate event aggregation plus metadata (the
   freshly rerun N1M branch).

No per-stock method winner is chosen from development or later results. No LLM,
attention, RL, FNSPID text or post-result ensemble is added.

## FNSPID boundary

The new FNSPID candidate audit is not used for prediction in this replay. Its
120-card independent entity review is incomplete. A later FNSPID predictive
experiment requires a separate protocol after the quality gate.

## Reporting

Report BA, accuracy, MCC, Brier, AUC, up/down recall and prediction balance by
stock and phase, plus monthly results. Development and later are explicitly
exposed exploratory historical backtests.

