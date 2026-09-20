# V10 classical lane final summary

Random-split prior-work scores are not directly comparable because they allow temporally mixed train/evaluation samples. V9 replaced that protocol with chronological issued NEWS-only models and full March--August grids. Stage B1 independently verified a seven-feature DPRICE baseline and froze three NEWS methods per setting using only March--August issued evidence. Stage B2 then executed every frozen NEWS+PRICE branch without a new search.

## Findings

- Four-hour methods descriptively strictly dominating historical F1 across all four stock/phase cells: none.
- Daily methods with positive system-level BA deltas versus DPRICE in all four cells: none.
- Frozen 24-hour KNN all-four positive pattern: False.
- Matched PAPER_2G 24-hour consistently stronger than overnight: False.

These are descriptive exposed historical evaluations. Joint-versus-price comparisons may also change classifier family, so they do not isolate a causal news contribution. The evidence covers only AAPL and AMZN, is not prospective financial validation, and does not establish state of the art. No B2 branch was promoted after seeing results.

CLASSICAL_LANE_FROZEN
