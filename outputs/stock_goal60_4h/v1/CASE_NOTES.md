# Fixed cases

Keys were selected by SHA256 within stock/phase/original-news strata before new predictions. These are prediction diagnostics, not independent event labels. See cases.json for exact probabilities. No-news T branches must equal new price fallback; H1/H2 may change only when qualified past news exists. A changed or corrected prediction does not establish a causal market mechanism.

- AAPL|2018-10-25 14:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.
- AAPL|2018-09-21 14:30:00+00:00: F0 correct; A1 correct; history+reaction wrong.
- AAPL|2018-09-04 14:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.
- AAPL|2018-10-04 14:30:00+00:00: F0 wrong; A1 wrong; history+reaction correct.
- AAPL|2018-11-16 15:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.
- AAPL|2018-12-26 16:30:00+00:00: F0 wrong; A1 correct; history+reaction correct.
- AAPL|2018-08-21 13:30:00+00:00: F0 correct; A1 correct; history+reaction wrong.
- AAPL|2018-04-24 14:30:00+00:00: F0 wrong; A1 wrong; history+reaction correct.
- AAPL|2018-07-10 13:30:00+00:00: F0 correct; A1 correct; history+reaction correct.
- AAPL|2018-04-06 13:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.
- AMZN|2018-09-13 14:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.
- AMZN|2018-10-01 15:30:00+00:00: F0 correct; A1 correct; history+reaction correct.
- AMZN|2018-10-02 13:30:00+00:00: F0 wrong; A1 correct; history+reaction correct.
- AMZN|2018-10-10 14:30:00+00:00: F0 correct; A1 wrong; history+reaction correct.
- AMZN|2019-01-14 14:30:00+00:00: F0 correct; A1 correct; history+reaction correct.
- AMZN|2018-11-09 16:30:00+00:00: F0 correct; A1 correct; history+reaction correct.
- AMZN|2018-11-12 15:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.
- AMZN|2018-11-30 14:30:00+00:00: F0 wrong; A1 correct; history+reaction correct.
- AMZN|2018-03-09 14:30:00+00:00: F0 correct; A1 correct; history+reaction wrong.
- AMZN|2018-04-05 15:30:00+00:00: F0 correct; A1 correct; history+reaction correct.
- AMZN|2018-07-16 14:30:00+00:00: F0 correct; A1 correct; history+reaction correct.
- AMZN|2018-04-02 15:30:00+00:00: F0 wrong; A1 wrong; history+reaction wrong.

## Input inspection of fixed AMZN no-current-news cases

- **2019-01-14 14:30 UTC:** nine older accepted articles. The latest titles include repeated stock-ranking/market-cap commentary, a multi-company recommendations roundup, and an institutional-holdings report. This supports a limited observation: historical coverage contains recurring commentary as well as potentially relevant company information. It does not establish a newly disclosed event or explain the four-hour outcome causally.
- **2018-11-09 16:30 UTC:** six older accepted articles. Titles include multi-company analyst commentary, a competitor-focused retail article, and two similar reports of Amazon retaking a market-cap ranking. These entered under the unchanged company-title filter. A filled window is therefore not equivalent to six independent new Amazon events.

This inspection checked titles and availability metadata of five most recent older articles per fixed case, not independent human gold or full-text extraction quality. Exact new-model probabilities and correctness are in cases.json. Historical-state aggregation deliberately did not introduce a new semantic event filter in this round, preserving the intended H0/H1/H2 comparison.
