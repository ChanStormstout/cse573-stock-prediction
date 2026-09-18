# Stock-specific reliability gate v1 — preregistration

This file is written before any G0--G3 result is calculated.  It is a finite
test of whether the same observable prediction-time context should receive a
different price/text reliability response for AAPL and AMZN.  It is not a new
base model, a hindsight router, or a search for a score above 60%.

## Fixed experts and provenance

The two issued probabilities are frozen from the current matched pipeline:

* **P (price):** `R1` from `outputs/stock_goal60_4h/v1/predictions.csv`;
* **T (full text):** `F1_new` from the same file.

The file's `training_evidence.json` is the provenance source.  March--August
rows must be chronological forward predictions.  September onward uses the
already August-trained frozen expert rows.  No gate model is allowed to fit or
refit either expert.  The older `F1` column, adaptive routers, F2, reaction,
ModernBERT and other model families are excluded.  This differs from prior
work because it compares reliability-controller slopes for two fixed experts,
rather than introducing stock-specific base predictors.

## Fixed state and target

The only state variables are, in this order:

`recent_return_15`, `recent_return_60`, `recent_rv_15`, `recent_rv_60`,
`minutes_from_open`, `overnight_gap`, `overnight_gap_missing`,
`log1p(news_count)`, `abs(p_text-p_price)`, `abs(p_price-0.5)`.

Rows with original news and finite expert probabilities define the controller
target after their prediction was issued:

`advantage = (label - p_text)^2 - (label - p_price)^2`.

This is a supervised reliability target, not a causal attribution.  No-news
rows stay in final metrics but never receive a synthetic text contribution.

## Controller models

All four variants use the same fixed mapping with `tau=0.05`:

`w_text = 1/(1+exp(clip(d_hat/tau,-60,60)))`; `p_final = (1-w_text)*p_price + w_text*p_text`.

For no-news rows, `w_text=0` and `p_final=p_price` exactly.

* **G0:** one weighted global mean advantage;
* **G1:** one weighted static mean per stock, otherwise G0 when stock support is
  insufficient;
* **G2:** weighted ridge `intercept + AMZN indicator + shared state slopes`;
* **G3:** G2 plus centered stock interactions, `+0.5` AMZN and `-0.5` AAPL.

The ridge penalties are fixed, without a grid: G2 `[0,10,10,...,10]`; G3
`[0,10,10,...,10,50,...,50]`.  A `1e-8` diagonal jitter is allowed only for
conditioning and is recorded.  G3 must reduce exactly to G2 when interaction
coefficients are zero.

Rows within each stock×trading-day have total training weight one
(`1/N`).  State imputation and standardization use past training rows only.

## Chronological replay and fallback

For evaluation month M, controller training uses only rows from months before M
whose target end is before the first evaluation cutoff.  March through August
are separate forward folds.  September onward uses one frozen controller fit
on eligible March--August rows and never updates with exposed labels.

Before any controller is used, require at least 60 eligible news rows overall
and 20 unique stock-days. G0--G3 return exact R1 until this holds: this is an
explicit force-price mode, not a synthetic `d_hat=0` mixture. G1/G3
require at least 20 eligible rows for each stock; G1 falls back to G0 and G3
falls back to G2 when that stock-specific support is absent.  Every fallback is
recorded.

## Evaluation and fixed advancement rules

Report AAPL and AMZN separately for each month, plus development and later as
exposed exploratory historical backtests.  Report BA, MCC, Brier, AUC, recalls,
prediction-up rate, text-weight summaries, changed directions, repaired errors
and newly introduced errors.

The three prespecified contrasts are G1−G0, G2−G1 and G3−G2.  A contrast is
supported only when the weaker stock's June--August mean BA improves at least
1 percentage point; neither stock loses over 1 point; macro BA is positive in at
least two of three months; neither stock's mean Brier worsens over 0.002; no
stock is effectively constant; and each stock has at least ten eligible rows
where the two experts have different **0.5-threshold directions**. Probability
differences that leave both experts on the same side of 0.5 are recorded for
diagnosis but do not satisfy this BA-routing headroom condition. These are
engineering evidence conditions, not
significance claims.

The label-using per-row perfect switch between P and T is reported only as
`HINDSIGHT DIAGNOSTIC ORACLE — NOT A MODEL`; it is never used for fitting or
selection.  If no contrast passes, stop this mechanism and do not expand into a
neural mixture of experts.
