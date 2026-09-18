# Reaction features v3 preregistration (reviewer protocol)

This protocol incorporates the complete reviewer addendum.  The earlier v2
directory is retained as a historical protocol-mismatch diagnostic.  The main
task remains AAPL/AMZN four-hour direction, with March--May inner selection and
June--August outer evaluation; September onward is exposed history.

## Article-target pairs and time

The full raw news corpus is converted to `(article_key,target_symbol)` pairs
with conservative ticker/legal-name, title-name, or target-centered body
evidence.  Product aliases alone do not qualify.  Availability is
`max(valid published_utc,crawled_utc)`.  Exact/normalized duplicate groups use
the earliest available canonical record.  Independent human gold review is
not claimed.

Reactions use regular-session five-minute bars.  The first bar starts at or
after availability; every bar is contiguous, non-missing, and ends by the
same session close.  Price context uses only completed bars whose end is at or
before availability and records per-row provenance.  Reaction directions are
`1[return > 0]`; near-zero returns are retained.

## Feasibility and article model

Both stocks must have at least 800 unique Jan--August groups with valid 60m
and 240m reactions, 100 dates, and six months with at least 50 groups.  Only
60m and 240m enter predictive models; 30m/120m remain audit statistics.

AR0 is completed pre-article price context. AR1 is AR0 plus binary lexical
features with fold-local vocabulary, chi-square top-500 selection and L2
logistic regression. AR2 is AR0 plus PCA16 from the pinned frozen target-context
FinBERT masked-mean representation. C is selected only from `{0.01,0.1,1.0}`
on March--May chronological folds. Training article reaction labels must be
fully mature before each evaluation availability, with stock-day weights
`1/N`.

The available environment lacks the required target-context FinBERT binary.
Therefore AR2 is explicitly `NOT_RUN_MODEL_UNAVAILABLE`; no article vectors
are substituted. AR2 unavailability does not by itself block an independently
passing AR1 candidate.

## Article promotion gate

AR1 and AR2 are independent preregistered candidates. For each horizon, an AR1
candidate is eligible when all of the following hold on June--August: both
stocks have mean BA gain at least 1 percentage point versus AR0; each stock has
all three months, at least 30 rows per stock-month and coverage at least 0.8;
each stock's maximum monthly Brier deterioration is at most 0.002; macro AUC
improves versus AR0; macro BA delta is positive in at least two of three months;
and there is no constant-direction collapse. These quantities are written to a
machine-readable gate.

AR2 has the stricter registered comparison: both stocks must gain at least 1
percentage point versus AR0, neither may lose more than 1 point versus AR1,
macro AUC must improve versus AR0, macro BA must be positive in at least two
months, and no constant-direction collapse may occur. AR2 is recorded as
`NOT_RUN_MODEL_UNAVAILABLE`; cached title vectors are not substituted.

If both horizons pass, choose one only by weaker-stock BA gain, macro AUC and
shorter-horizon tie-break. Do not combine horizons or use exposed periods.

## Conditional official-window protocol

Only after at least one AR1 or AR2 horizon passes, use the original official
windows and strictly past-trained article predictions over the fixed
24-clock-hour lookback. A passing AR1 horizon supplies W2/W3 when AR2 is
unavailable. If neither candidate family passes, stop before W0--W3; exposed
downstream scores never select between candidate families.

* **W0:** R1 only.
* **W1:** R1 plus `log1p(article_count)`, newest article age and a reaction-
  article-present flag (coverage control).
* **W2:** R1 plus exactly `reaction_mean`, signed score with largest absolute
  magnitude, `log1p(reaction_article_count)`, newest reaction age and a
  reaction-present flag.
* **W3:** saved past-only F1 probability plus R1 and the same five reaction
  features.

All downstream heads are L2 logistic regression with C selected from the fixed
three candidates on March--May. W0 is R1; W1 is R1 plus only the three coverage
fields; W2 is R1 plus exactly the five reaction fields; W3 is saved F1 plus R1
plus those same five fields. The old v2 F1 residual W0--W3 experiment is
superseded.
