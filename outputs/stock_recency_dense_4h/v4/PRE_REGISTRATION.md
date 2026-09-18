# v4 pre-registration: corrected recency and dense controls

This run is a correction and extension of `stock_recency_dense_4h/v3`.  The
v3 directory is historical and is not overwritten.  All development and later
periods are already exposed exploratory backtests.

## Recency

The canonical paper-method transforms are used.  `J0` is the F1 text model and
`J2` is the F2 semantic model.  For F2, PCA(16) is fit on training article
vectors (not window means), followed by per-window mean vectors and the
registered `log1p(news_count), has_news` metadata.  The infinity branch is a
real unweighted refit, rather than a copy of a saved probability file.

The registered half-lives are infinity, 80, 40 and 20 XNYS sessions.  The
March--May chronological folds select one global half-life per method by the
weaker-stock BA, then macro BA, then the simpler (longer) half-life.  The
June--August outer comparison is the only recency gate.

Two text policies are reported separately:

* `branch_recency`: weighted text classifier, with the equal-weight R1 refit
  as the no-news fallback;
* `system_recency`: weighted text classifier and weighted R1 fallback.

The primary contrast is `branch_recency` versus the infinity refit.  The
historical equal-weight predictions are compared fold by fold with the true
infinity refit.  A maximum probability difference greater than `1e-10` stops
the recency branch before it is treated as a valid improvement.

## Dense controls

The dense-window gate is exactly June--August (six month rows).  Before any
augmentation result is interpreted, the reconstructed R1 columns are compared
column by column with the canonical official inputs.  If they differ, all
official evaluation rows and augmented training rows use the same reconstructed
generator; the matched reconstructed equal/day-normalized controls are then
the authority.  The old v3 March--August gate is retained only as historical
context.

## Frozen choices

No new LLM/FinBERT fine-tuning, GNN, RL, Chronos or external data is introduced
in this run.  The target remains the AAPL/AMZN four-hour direction task and the
original cutoff and labels.
