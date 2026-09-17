# Four-hour next-generation experiment protocol

Registered before this run's outcome scoring. All September 2018 onward labels
have already been exposed by earlier work, so every result is an exploratory
historical replay. This protocol controls the present search; it does not turn
the old periods into an untouched test set.

## Fixed task and rows

- Predict AAPL and AMZN direction over the original four-hour regular-session
  interval. `UP` means final close exceeds interval open.
- Keep all 1,607 rows: AAPL 499 train / 126 validation / 178 later; AMZN
  499 / 126 / 179. No-news and low-confidence rows remain.
- A row may use only completed price bars and news available at its cutoff.
- March--August 2018 expanding forward folds select hyperparameters. Models for
  September onward fit only rows before September. Results after August are
  reported as exposed validation/later replays.
- Report BA, MCC, Brier, ECE, predicted-up rate, coverage, stock and month.

## P0--P3 news inputs

The six article keys and their recency ordering are inherited from the sealed
direct-LLM input pack. Labels never affect article or passage selection.

- **P0:** the existing title plus character excerpts. Existing evaluation
  scores are reused byte-for-byte; missing training scores are generated.
- **P1:** whole target-company sentences/paragraphs, under the token count of
  that row's P0 prompt. A unit is included only when it contains the explicit
  target aliases `Apple/AAPL` or `Amazon/Amazon.com/AMZN`. A unit is never cut.
- **P2:** the same eligible articles and ordering, with whole target units up to
  a fixed 6,000 prompt-token ceiling (6,144 including the one-token answer).
- **P3:** P2 after online duplicate grouping. At a cutoff, grouping sees only
  articles already available. Exact normalized hashes or normalized-title
  token Jaccard at least 0.65 form a group. The newest available representative
  is supplied; source and group counts are retained.

No target unit means an empty article evidence set; the system never substitutes
an unrelated introduction. HTML/navigation boilerplate is removed before
paragraph splitting. Exact raw spans, sentence boundaries, omissions and
cluster membership stay in private artifacts.

Input quality is checked on a deterministic training-only sample before P1--P3
inference. It covers both stocks, no/few/many news, multi-company items,
background language, ratings, earnings, duplicates and collection delay. The
engineering gate requires exact source spans, complete sentence boundaries,
explicit target aliases for every selected unit, zero irrelevant fallback and
fewer observed fragments than P0. This is assistant review, not independent
semantic verification.

## LLM scoring and calibration

Use the pinned local Qwen3.5-9B 4-bit files and revision from the earlier direct
forecast experiment, non-thinking chat template, seed 573, temperature zero and
the same one-token `UP`/`DOWN` conditional log-probability interface. The system
prompt is unchanged. Inputs cannot be edited after a variant begins inference.

For every paragraph variant report raw conditional scores and three calibration
options: Platt logistic regression, one positive temperature fitted on logits,
and identity. Starting with April, each monthly prediction is calibrated using
only earlier labelled rows; the frozen September-onward calibrator uses rows
ending before September. Method selection uses March--August forward predictions,
lowest Brier then highest BA, and never later labels. Calibration may be called
an improvement in probability quality only; it is not assumed to improve BA.

No-news policies are all reported: raw LLM prior, strict recent-price fallback,
and a past-only missing-news calibrator. The missing-news calibrator is fitted
only with at least 30 earlier no-news rows containing both labels; otherwise it
falls back to recent price.

## Recent price features R0--R2

Five-minute timestamps denote bar starts. A bar is usable only if start plus
five minutes is no later than cutoff.

- **R0:** the existing six completed-hour returns/ranges, history age, return
  mean/std and New York hour.
- **R1:** R0 plus cutoff-available 5/15/30/60-minute log returns, high-low
  ranges and realized volatility, missing indicators, overnight gap when the
  current session open bar is complete, minutes from open/to close, and age of
  latest complete hour.
- **R2:** recent features only.

Each recent window requires contiguous bars from the current regular session;
unavailable values are marked missing and imputed to zero after training-only
standardization. No target opening price or target-period bar is used. The
seventh chart field is named `activity`; because course material does not define
it, no volume, VWAP or order-flow feature is constructed.

All price variants use L2 logistic regression, `C in {0.01, 0.1, 1}`, fitted
separately by stock unless stated. Selection maximizes mean monthly forward BA,
then minimizes Brier, then chooses smaller C.

## Sharing S1--S3

- **S1:** independent R1 models, identical grid for each stock.
- **S2:** one pooled R1 model plus stock identity and shared slopes.
- **S3:** pooled shared R1, stock identity and stock-by-feature deviations.
  Deviations are multiplied by 0.5 before fitting, giving them stronger
  effective L2 shrinkage than shared coefficients.

All three use identical rows, features and C grid. There is no post-hoc
stock-specific winner rule.

## External market branch

M0 is R1. M1/M2/M3 require reliable cutoff-aligned minute SPY, QQQ and industry
ETF data respectively. A branch is stopped if local data are absent and no free,
reproducible 2018 intraday source with adequate timestamps, adjustment semantics
and redistribution terms is found. No daily data may enter this hourly task.
PCA/HMM are allowed only after a simple factor shows forward gain. A two-stock
GNN is out of scope.

## Fusion F0--F7

All learned weights use training-period out-of-fold predictions. News components
strictly return R1 on no-news rows. Candidate weights are the non-negative 0.25
simplex; the chosen candidate maximizes mean monthly BA among candidates whose
Brier is no more than 0.002 worse than the matched F0. Ties use lower Brier,
fewer non-zero components and lexical weight order.

- F0 title baseline; F1 body; F2 FinBERT; F3 body+FinBERT;
- F4 body+selected calibrated paragraph LLM;
- F5 body+FinBERT+LLM; F6 adds selected recent-price/shared price score;
- F7 adds a market factor only if the market-data gate passes.

Every single component remains in the result table. P0--P3 selection uses the
same training OOF rule: Brier within 0.002 of P0, then BA, Brier and smaller
context in order. This creates one shared rule for both stocks; it cannot choose
one paragraph method for Apple and another for Amazon after seeing evaluation.

## Cases, stopping and integrity

Case slots are selected deterministically before reading their outcomes: for
each stock and exposed split, the lowest SHA-256 key in baseline-wrong/new-right,
baseline-right/new-wrong, both-wrong and both-right, plus changed P0/P-selected,
R0/R1 and news/no-news strata when available. Notes first record actual inputs,
then reveal outcomes. Input, selection, LLM, market and final-prediction errors
are separate categories.

- Two targeted versions without forward gain stop that mechanism.
- No qualified minute factor data stops M1--M3, PCA, HMM and graph work.
- Failed paragraph quality stops large LLM inference.
- QLoRA is not run unless paragraph quality improves and event-group separated
  supervision demonstrates stable semantic errors. Direct return-label fine
  tuning is not run.
- Existing files are immutable. Each run gets a new directory, source hashes,
  prediction ledger and status record; cache mismatch stops execution.
