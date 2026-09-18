# Reaction features v2 preregistration

This protocol is frozen before the repaired v2 corpus is built.  The original
v1 directory and its scores remain historical artifacts.  The task is still
the AAPL/AMZN four-hour direction problem; March--August is training-period
forward evaluation and June--August is the registered outer period.  No
development or later score is used to choose a model.

## Information boundary and article-target unit

The source is the complete raw news index.  One row is
`(article_key, target_symbol)` and a dual-target article creates two rows.
Availability is `max(valid published_utc, crawled_utc)`.  A target association
requires a ticker/legal-name match, a conservative title-name match, or a
body sentence match.  Duplicate groups are formed from normalized full text,
exact text, then normalized title; the earliest available group row is the
canonical row.  `target_pair_key` is persisted and is the join key for all
downstream manifests.  Independent human gold review is not claimed.

## Time-safe reactions and context

The reaction starts at the first five-minute bar whose **start** is at or
after article availability.  A 30/60/120/240 minute reaction is valid only if
all bars are non-missing, contiguous, within the same regular session, and
the final bar **end** is before the session close.  Its end timestamp and
maturity flag are saved.

Price context uses only bars whose bar end is at or before availability.  The
last completed contiguous suffix is used; each row records the maximum used
bar end and a completeness flag.  No future-price context is allowed.

## Article models

AR0 uses completed pre-event numeric context.  AR1 adds a binary CountVectorizer
representation with training-fold-only vocabulary and chi-square top-500
selection; the numeric median imputer and standard scaler are also fit inside
each past fold.  Articles are weighted by `1 / count(symbol, session_day)`.
March--May are chronological inner folds and June--August are outer folds;
each training reaction label must mature before the earliest evaluation
article availability.  C is selected from `{0.01, 0.1, 1.0}` using only inner
fold BA.  AR2 is registered as a target-context FinBERT representation, but
the required binary is not available in this environment; the repaired run
must record `NOT_RUN_MODEL_BINARY_UNAVAILABLE`, not substitute article vectors.

## Promotion gate

For each horizon, AR1 is compared with AR0 on every stock and every outer
month.  A horizon passes only if both stocks have all three outer months, at
least 30 valid rows per stock-month, coverage at least 0.8, and horizon
completeness; each stock must have mean monthly ΔBA >= 0.01, maximum monthly
ΔBrier <= 0.002, and positive ΔBA in at least two of three months.  The public
`promotion_gate.json` and monthly files contain every quantity used by this
formula.

## Frozen downstream W0--W3 protocol

The base is the saved four-hour `F1` expert from
`outputs/stock_goal60_4h/v1/predictions.csv`, with the original official
windows and cutoff.  W0 is this probability unchanged.  For each promoted
horizon, article AR1 predictions available in the four hours ending at the
window cutoff are aggregated as mean prediction and `log1p(article_count)`;
missing articles use mean `0.5` and count zero.  W1 fits a fixed `C=0.1`
logistic residual model using the base logit plus one promoted horizon's two
aggregates.  W2 uses the same model with all promoted horizons' aggregates.
W3 uses the first promoted horizon plus its aggregate standard deviation and
is allowed to adjust only when that horizon has at least two articles; it
returns **exactly W0** otherwise.  W1/W2/W3 train only on March--May and are
evaluated on June--August.  If no horizon passes, no downstream predictive
fit is run and a machine-readable `NOT_RUN_NO_PROMOTED_HORIZON` status is
written.

## Reproducibility and stopping

The v2 public directory is new, the private corpus contains raw text and
models, and `PRE_REGISTRATION_v2.md` plus source hashes are recorded.  The
verifier must assert target-pair uniqueness, maturity and context time safety,
future-price perturbation invariance for pre-context, manifest membership,
preprocessing fit counts, and complete promotion-gate fields.  The run stops
at the registered gate; no extra horizon/grid is searched after failure.
