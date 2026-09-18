# v3 to v4 corrections

The following v3 issues are corrected without changing the historical v3
artifacts:

1. The v3 F2 recency transform fit PCA on window means and omitted the
   canonical F2 metadata. v4 uses the `stock_paper_methods_4h/run.py`
   `Transform('J2')` implementation, whose PCA is fit on training article
   vectors and whose window representation includes `log1p(news_count)` and
   `has_news`.
2. The v3 infinity row reused saved probabilities. v4 fits an all-one-weight
   model and records per-fold/final parity against the historical reference.
3. v3 used a weighted R1 fallback for text recency by construction. v4 reports
   the registered branch fallback (equal R1) and the separate system fallback
   (weighted R1).
4. The v3 dense gate included March--August. v4 requires exactly the six
   June--August rows.
5. The original preregistration's phrase “non-overlapping windows” was wrong:
   the data contain densely sampled, overlapping four-hour windows with
   30-minute starts. v4 uses the corrected wording.

If a true infinity refit does not reproduce the historical reference within
the registered `1e-10` probability tolerance, the recency branch is marked
stopped and no half-life result is used as evidence of improvement.
