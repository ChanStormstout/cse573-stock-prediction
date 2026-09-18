# Reaction v4 preregistration

This non-overwriting repair replaces neither v2 nor v3. It is written before
v4 article-model fitting. The task is article-level same-session reaction
direction, used only as a preregistered eligibility gate for the frozen
four-hour W0--W3 protocol.

## Fixed source and time contracts

- Source: v4 canonical `(article_key, target_symbol)` groups built from the raw
  course corpus. Article text, article keys, model files, and review cards stay
  private.
- Availability is `max(valid published_utc, crawled_utc)`.
- A reaction starts at the first five-minute bar start at or after availability,
  uses contiguous bars, and ends no later than the session close.
- Every pre-article context bar ends at or before availability. For each of
  5/15/30/60 minutes, unavailable return and realized volatility are `NaN` and
  the explicit `valid=0` indicator remains in the model input.
- The registered horizon family is exactly **30, 60, 120, and 240 minutes**.
  No horizon is removed after exposure.
- The known `American Association for Physician Leadership` AAPL expansion is
  rejected only when it is the apparent AAPL evidence and independent Apple
  evidence is absent. This is a high-precision exclusion, not complete entity
  linking.

## Article models and past-only fitting

AR0 uses the continuous numeric context plus `minutes_from_open`. AR1 adds a
past-fold CountVectorizer and chi-square-selected lexical context. Both use
`C in {0.01, 0.1, 1.0}`, article weighting of `1/(stock x session-day count)`,
and the same March--May inner forward selection. Continuous numeric values use
median imputation then standardization fit on the training fold only; the four
validity indicators are binary features and are not standardized. AR2 remains
`NOT_RUN_MODEL_UNAVAILABLE`: no cached title embedding may substitute for its
registered target-context FinBERT representation.

## Article candidate gate

For each AR1 horizon independently, June--August outer predictions must have
all of: both stock mean BA gains at least +1pp versus AR0; per-stock maximum
Brier delta at most +0.002; at least two positive stock months; positive macro
AUC delta; at least two positive macro-BA months; all six cells with at least
30 rows and 80% coverage; horizon completeness; and no constant-direction
collapse. The decision is made only from this article gate.

If no AR1/AR2 candidate passes, v4 stops before W0--W3. If an AR1 horizon
passes, it alone may enter the already frozen W0=R1, W1=R1+coverage,
W2=R1+five reaction features, W3=F1+R1+same reaction-features protocol. AR2
unavailability does not disqualify an independently passing AR1 horizon.

All scores remain exposed exploratory historical backtests. This protocol does
not authorize Phase B G0--G3 predictive execution.
