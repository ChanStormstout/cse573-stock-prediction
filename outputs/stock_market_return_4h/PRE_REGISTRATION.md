# Market state and continuous-return auxiliary experiment — v1

Registered before fitting this run. The main task remains AAPL/AMZN four-hour
direction on the unchanged 1,607 windows. Development and later periods have been
exposed and are exploratory historical backtests.

## Part A: market-data feasibility audit

Probe SPY and QQQ one-minute Alpaca bars on four fixed 2018 dates (2018-01-10,
2018-04-27, 2018-09-04, 2018-12-03), regular-session bounds, with the feed and
adjustment semantics recorded. No credentials are stored or printed. If access is
unauthenticated or bars cannot be verified, stop the market branch. Do not replace
minute data with daily data, synthetic ETF prices, or the target-stock bars.

## Part B: richer target supervision

The target direction remains `y=1[target_return>0]`; `target_return` is a training
target only. Features use only the original past price columns, with missing values
imputed and scaling fitted inside each chronological training fold. No future return
is present in an input matrix.

Methods:

- `C0`: classification-only L2 logistic regression.
- `C1`: ridge regression of the continuous four-hour return; convert its output to
  direction only for BA/MCC. This is a diagnostic, not a replacement task.
- `C2`: shared linear projection (8 latent units) with a logistic direction head and
  a Huber return head. The objective is BCE plus `lambda * Huber(return)`. Lambda is
  selected only from earlier forward-OOF months from `{0, .1, .5, 1}`; no development
  or later labels choose it. Three fixed seeds (573, 574, 575) are averaged.

The primary input is the raw R1 price feature set (six completed hourly returns and
ranges, recent 5/15/30/60-minute features, time and missingness). A secondary frozen
score input appends the already-saved F1 probability, and is labelled `+F1`; it does
not retrain FinBERT or claim to be a new text representation. This checks whether
auxiliary return supervision can use an existing text signal without changing the
text pipeline.

For every March–August forward month, train on strictly earlier rows for the same
stock and evaluate the month. For development and later, freeze the lambda selected
from all available forward-OOF records through August, fit using January–August
training rows, and evaluate each exposed period separately. Report BA, MCC, Brier,
continuous-return MAE/RMSE/correlation, predicted-up rate, monthly values, and
paired 1/5-day bootstrap intervals. No per-stock later winner selection.

Promotion line (engineering exploration, not statistical significance): both stocks'
June–August mean BA must gain at least 1 percentage point over C0, no stock may lose
more than 1 point, and mean Brier may not worsen by more than .002. Otherwise stop
before residual-market/text combinations or neural expansion.

## Integrity checks

Save the exact feature names, row keys, time boundaries, target hashes, lambda/seed
choices, model reload errors and source fingerprints. Test target permutation does
not enter features, future-row perturbations cannot alter earlier fits, and C2
`lambda=0` still has a separate classification head. This is a linear multi-task
pilot, not a full neural architecture or causal market model.
