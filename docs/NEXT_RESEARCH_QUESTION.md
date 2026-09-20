# Next research question: ECNI

## Question

When does financial news provide predictive information about future stock direction beyond information already available from price history, market context, and previously observed news?

The candidate mechanism is **Evidence-Grounded Conditional News Innovation (ECNI)**. It keeps four channels separate: factual innovation, dissemination change, dense current-text semantics, and price/market context. This lane is outcome blind. It does not train or score a stock-direction model.

## Candidate tasks

The primary candidate is a daily fixed-window task. At regular-market open minus five minutes, predict same-session open-to-close direction using only information available by the cutoff. Every stock-day remains, including no-news days. Exact intraday timestamps are required; date-only records are assigned to a conservative prior-day cutoff or excluded according to a rule frozen before modeling.

The secondary task is event conditioned: for articles with reliable exact publish times, measure a preregistered future response horizon. It does not replace the fixed-window task.

## Conceptual target

The target is conditional information `I(Y; N | P, M, H)`, where `Y` is future direction, `N` current news, `P` prior stock prices, `M` market/sector context, and `H` previously available news. A deterministic representation cannot create Shannon mutual information beyond its source. ECNI instead tests whether a structured representation makes task-relevant information more usable to a restricted predictor.

The held-out usable-information statistic is:

`G_bits = (NLL_baseline - NLL_augmented) / (n * ln(2))`.

Positive values mean better average held-out probability assignment on the same rows. They are not an unbiased estimate of true market mutual information.

## Authorization boundary

Before prediction, the project must have an acquired source manifest, timestamp and entity audits, duplicate groups, a deterministic panel and company split, frozen time cutoffs, a locked future period, reader-label protocol, encoder token-budget contract, and preregistered comparisons. No return label may influence dataset, stock, split, or reader-example selection.
