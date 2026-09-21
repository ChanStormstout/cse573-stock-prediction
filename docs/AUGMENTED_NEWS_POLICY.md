# Default news corpus for subsequent experiments

Owner direction on 2026-09-20: subsequent news-based experiments use the
union of the canonical course news and the time-safe FNSPID direct-target
news prepared by `stock_fnspid_augmented_4h`.

## Required behavior

- Preserve the canonical 1,607 four-hour windows, labels, cutoffs and price
  features unless a separately named task changes the target.
- Keep all canonical course-news membership.
- Add only FNSPID records linked as `DIRECT_TARGET_HIGH_CONFIDENCE` for the
  corresponding stock in the default branch.
- Treat date-only FNSPID records as available at the next XNYS regular-session
  open. Do not move them earlier.
- Deduplicate exact course/FNSPID title matches and FNSPID repeat groups before
  aggregation.
- Fit vocabulary, scaling, PCA, calibration and model parameters using only the
  permitted training history for each fold.
- Preserve price-only and original-course-news variants as ablations. They are
  no longer the default input for a new news method.
- Report coverage and performance separately. More covered windows do not by
  themselves establish more useful predictive information.

## Evidence boundary

This policy records the input default, not a quality or performance claim.
The FNSPID entity relinker has no completed independent human semantic review,
and the relevant timestamps are date-only. Results using this corpus remain
exploratory historical backtests until those limits and a fresh future period
are resolved.
