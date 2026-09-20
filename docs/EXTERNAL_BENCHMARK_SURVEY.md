# External benchmark survey for the ECNI lane

Audit date: 2026-09-19. Evidence labels are `CONFIRMED_FROM_PRIMARY_SOURCE`, `CONFIRMED_FROM_SAMPLE_RECORD`, `INFERRED`, and `UNKNOWN`.

## Recommendation

**Primary new data foundation: FNSPID, conditionally, for a daily panel.** It is the only currently downloadable candidate with large stock/year coverage and paired raw daily prices. The public first-row sample also exposes exact UTC timestamps for some records. It is not ready yet: 90/100 inspected rows were midnight placeholders, all 100 lacked article/body and summary content, and the sample is concentrated on one ticker. A stratified, file-level audit must pass before acquisition or prediction. If the larger files do not provide adequate text and point-in-time timestamps, the primary decision reopens without outcome inspection.

**Standard external benchmark: CMIN-US.** Preserve its official 110-stock, 2018--2021 task and report it separately as `CMIN_OFFICIAL`. Its repository contains per-ticker news and daily prices; an inspected AAPL record has an exact timestamp, title, summary, URL and ticker. A separately named `CMIN_POINT_IN_TIME_REAUDITED` lane may repair ticker and timing rules but is not an official leaderboard comparison.

**Reader/event auxiliary supervision: EDT.** EDT supplies 9,721 token-level event records and an 11-event taxonomy. Its approximately 303k trading articles contain minute-level publish timestamps and full text, but official documentation warns that automatic ticker assignment can be wrong. The released trading labels are not continuous raw minute-price data.

FinMultiTime is promising in scope but not presently the primary choice: its paper reports U.S. news and daily prices through 2025, while the inspected official-author Hugging Face revision exposes only image files plus a description CSV. Raw U.S. news, price and financial-table components could not be sampled from that revision. StockNet is a useful legacy benchmark but covers only 88 stocks in 2014--2016 and uses tweets rather than news. SEC EDGAR is an auxiliary factual source, not a prediction benchmark.

## Dataset findings

| Dataset | Confirmed scope | Text/time | Price | Current role |
|---|---|---|---|---|
| Course data | Sample/archives: 78,055 records; AAPL-dominant, 2018--2019 | Full text, title, published and crawl timestamps; source-time anomalies already documented | AAPL/AMZN 5-minute bars | Historical diagnostic only |
| CMIN-US | Primary repository: 110 stocks, 2018--2021 | Sample: title, summary, URL, exact timestamp; full body absent from schema | Raw daily OHLCV | Standard benchmark |
| FNSPID | Primary repo/card: 15.7m news, 29.7m prices, 4,775 tickers, 1999--2023 | 100-row sample: title/URL/ticker; 90 midnight placeholders; bodies/summaries all null | Download advertises per-stock daily history | Conditional primary foundation |
| FinMultiTime | Paper: U.S./China, 2009--2025, 112.6 GB | Paper reports minute-level news and full-article collection; inspected public revision lacks news files | Paper reports daily Yahoo OHLCV; files unavailable in inspected revision | Blocked candidate |
| EDT | Official repo/paper: 9,721 labeled + about 303k trading articles, 2020-03 to 2021-05 | Full text, title, adjusted minute timestamp | Start/end/high/low labels; continuous minute bars not verified | Reader supervision |
| StockNet | Official repo: 88 stocks, nine sectors, 2014--2016 | Exact tweet `created_at`; tweet text | Raw Yahoo daily OHLCV | Legacy external benchmark |
| SEC EDGAR | SEC APIs and archives | Filing/accession metadata; availability must use filing acceptance/public time | No market-price panel | Auxiliary facts |

## Primary-source references

- [CMIN-US repository](https://github.com/BigRoddy/CMIN-Dataset) and [ACL paper](https://aclanthology.org/2023.acl-long.673/)
- [FNSPID repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset), [dataset](https://huggingface.co/datasets/Zihan1004/FNSPID), and [paper](https://arxiv.org/abs/2402.06698)
- [FinMultiTime paper](https://arxiv.org/abs/2506.05019) and [author dataset revision](https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting)
- [EDT repository](https://github.com/Zhihan1996/TradeTheEvent) and [ACL paper](https://aclanthology.org/2021.findings-acl.186/)
- [StockNet repository](https://github.com/yumoxu/stocknet-dataset) and [ACL paper](https://aclanthology.org/P18-1183/)
- [SEC submissions API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

## Access and license boundary

CMIN-US and StockNet repositories declare MIT licenses. FNSPID's repository license file is CC BY-NC 4.0, while its later README says research and commercial rights are released; this conflict and the underlying publishers' text rights require clarification before redistribution. EDT data is linked through Google Drive and the GitHub repository does not expose a standard license in its metadata. The inspected FinMultiTime Hugging Face dataset has no confirmed license tag. Therefore raw copyrighted news text remains private and is never committed.
