# Reaction features v3 corrected report

## Plain-language conclusion

This run repaired the eligibility rule as well as the time boundary.  AR1 and
AR2 were treated as independent article-level candidates.  The 240-minute AR1
candidate improved both stocks' June--August balanced accuracy relative to
AR0, but its macro AUC was lower than AR0.  The 60-minute candidate also failed
the full gate.  Because no article candidate passed **all** preregistered
conditions, the official W0--W3 downstream phase was correctly stopped.  AR2
was not run because the target-context FinBERT binary was unavailable.

This is a valid negative result about the registered reaction mechanism; it is
not evidence that every possible news representation fails.

## What actually ran

- 85,402 canonical article-target groups were prepared from the full corpus.
- AR0 used completed pre-article price context.
- AR1 added fold-local binary lexical features, chi-square top-500 selection and
  L2 logistic regression.
- Only 60m and 240m entered predictive article models; 30m/120m were not used
  for promotion.
- Numeric imputation, scaling, lexical vocabulary and feature selection were
  fit separately on each past fold.  There were 120 article fits and every
  saved model was reloaded with zero prediction difference.
- Article scores for exposed months were not generated because the gate failed;
  no official-window labels were used to rescue the candidate.

## Article-level gate (June--August)

| horizon | AAPL mean ΔBA | AMZN mean ΔBA | AAPL max ΔBrier | AMZN max ΔBrier | macro ΔAUC | positive macro months | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 60m | −0.451pp | +0.758pp | +0.000446 | +0.000989 | −0.007469 | 1/3 | FAIL |
| 240m | +1.763pp | +3.218pp | +0.000060 | +0.000710 | −0.005700 | 3/3 | FAIL |

The 240m candidate satisfies the per-stock BA, Brier, coverage, completeness
and positive-stock-month checks, but fails the required macro-AUC condition.
The 60m candidate fails the AAPL BA and monthly conditions as well as macro AUC.
The machine-readable details are in `promotion_gate.json` and the six-cell
files `promotion_h60m_monthly.csv` and `promotion_h240m_monthly.csv`.

## AR2 and downstream status

AR2 is `NOT_RUN_MODEL_UNAVAILABLE`; no cached title embeddings were substituted.
The official status is `NOT_RUN_NO_PROMOTED_ARTICLE_HORIZON`.  Consequently
there are no v3 W0/W1/W2/W3 predictive scores or downstream promotion claims.
The registered downstream protocol, if a future article candidate passes, is
W0=R1; W1=R1 plus coverage metadata; W2=R1 plus exactly five reaction fields;
W3=saved F1 plus R1 plus those same five fields.

The earlier v2 downstream table used a different residual/fallback protocol.
It remains preserved in `../v2` for history but is superseded and cannot be
combined with this v3 article gate.

## Validity boundary

`verify_v3.py` passes completed-bar cutoff, maturity, target-pair uniqueness,
future-price perturbation, fold-local preprocessing, model reload, AR2 explicit
stop, gate recording and no-downstream-on-failure checks.  The source
association cards and provisional event labels were generated with assistant
review; independent human association review is not claimed.  All June--August,
development and later values are exposed historical backtests.
