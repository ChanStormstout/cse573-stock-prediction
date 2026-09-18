# Target-association audit

The reaction corpus is built from conservative target evidence: ticker/legal
name, a company name in the title, or a target-centered body sentence.  Product
aliases alone do not qualify.  The counts below are sampled deterministic review
cards, not an independent human validation set.

| symbol | tier | sampled cards | distinct sites | interpretation |
|---|---:|---:|---:|---|
| AAPL | 0 | 50 | 1,626 | ticker/legal-name evidence |
| AAPL | 1 | 5 | 1 | title company-name evidence |
| AAPL | 2 | 0 | 0 | target-centered body evidence |
| AMZN | 0 | 50 | 702 | ticker/legal-name evidence |
| AMZN | 1 | 50 | 41 | title company-name evidence |
| AMZN | 2 | 50 | 257 | target-centered body evidence |

All 85,402 canonical `target_pair_key` values are unique.  A target pair is
not proof that the article contains a new, market-relevant company event.  In
particular, rankings, holdings reports, multi-company summaries and near
duplicates can still be associated with a target.  Those limitations are why
the article-level gate is required before any W0--W3 downstream use.

No independent reviewer has certified these associations.  The audit therefore
supports time-safe exploratory measurements only; it does not support a claim
of high-precision production news filtering.
