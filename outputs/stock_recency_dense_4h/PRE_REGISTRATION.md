# Recency weighting, dense-window, and reaction audit — v1

Date: 2026-09-17

## Scope

This run keeps the CSE 573 primary task fixed: AAPL and AMZN, four-hour direction, the original 1,607 official windows, and the original information cutoff five minutes before the target window. Development and later periods are exposed historical backtests. They are not used to select a method.

The run tests three bounded questions:

1. Does giving more weight to recent training labels help a frozen R1, F1, or F2 pipeline across June–August?
2. Does adding non-overlapping 30-minute-start four-hour price windows improve the price-only control without changing the official evaluation rows?
3. Do the available articles show a measurable, sufficiently covered post-availability reaction at 30/60/120/240 minutes?

No new LLM, neural network, graph, RL, or external market-data component is introduced here. Raw news text, article keys, cached features, and model binaries remain private under `work/`.

## Part A: recency-weighted labels

Half-lives are exactly `{infinity, 80, 40, 20}` NYSE sessions. A training row whose label is mature before an evaluation boundary receives `2**(-age_sessions/half_life)`, with infinity equal to one. Feature vocabulary, missing-value transformation, text vectorization/PCA, and feature selection are fit without weights; only the final logistic classifier receives `sample_weight`.

The three inputs are the existing R1 (recent price), F1 (price plus binary chi-square text), and F2 (price plus frozen FinBERT PCA text), with the existing no-news policy. Historical past-only C schedules are reused where available so this experiment isolates label weighting.

March–May are inner forward months. One half-life is selected globally across both stocks by weaker-stock mean balanced accuracy, then macro mean, then the simple order infinity, 80, 40, 20. June–August are outer evaluation. A recency method passes the engineering gate only when the weaker stock gains at least one percentage point, neither stock loses more than one point, and macro BA improves in at least two of the three outer months. Brier is reported and is not used as a veto. A final Jan–Aug fit reports development and later separately; no later result selects a method.

## Part B: dense windows

From raw regular-session five-minute bars, create four-hour windows at 30-minute strides, exactly 48 completed bars, with no interpolation or after-hours bars. Training weights are normalized to sum to one per stock-day. D0 is the official equal-weight control, D0_day is the official day-normalized control, and D1 is the augmented day-normalized set. D2 (D1 plus selected recency) is run only if the R1 recency gate passes. Evaluation remains on the official rows.

Text extension is conditional on D1 passing its registered gate and an exact reconstruction check; otherwise it is explicitly stopped and marked not run.

## Part C: article reaction audit

This is an audit only; it does not train a reaction model. Article-to-stock association reuses the existing accepted article IDs from official windows. Raw occurrences and deterministic online groups are both counted. For each first complete regular-session bar after `available_utc`, reaction returns are measured at 30, 60, 120, and 240 minutes under SAME_SESSION and TRADING_TIME. No partial bar, synthetic overnight bar, or future article is used.

The audit reports coverage, crossing-session counts, sign balance, absolute-return distribution, duplicate concentration, dates, sources, and Jan–Aug versus Sep-onward periods. It does not call a reaction a causal effect.

## Part D: activity audit

The seventh raw-bar column is audited descriptively. Because the repository does not contain an authoritative vendor definition, it is not used as a model feature. The status remains `SEMANTICS_UNRESOLVED_NOT_USED` unless an authoritative source is found.

## Verification and stopping

The run checks original keys and labels, cutoff integrity, no future features, weight sums and monotonicity, model reload, metrics reconstruction, and future-bar perturbation. If the quality or engineering gate fails, the corresponding extension stops and its negative result is retained. Results are exploratory historical evidence, not a claim of new-period generalization.
