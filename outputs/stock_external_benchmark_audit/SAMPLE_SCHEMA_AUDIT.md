# Sample schema audit

## Inspected records

- Course: all 78,055 local JSON metadata records were enumerated. Fields include `uuid`, title, text, URL, language, published, crawled, entities, thread and thread order. This does not establish their semantic meaning.
- CMIN-US: a 200,000-byte bounded AAPL news sample and complete AAPL daily-price file from commit `3f571a5`. News columns are `date,time,ticker,name,title,summary,link`; prices are daily raw OHLCV.
- FNSPID: Hugging Face dataset-server first 100 rows at revision `bf9189c`. Fields are Date, Article_title, Stock_symbol, Url, Publisher, Author, Article and four summaries. All bodies/summaries were null in this sample.
- EDT: official data README at commit `51befed`; the Drive payload was not downloaded. The documented trading record has title, text, pub_time and a label dictionary with automatically assigned ticker plus future price labels.
- StockNet: one AAPL tweet-day and complete AAPL raw-price file at commit `330708b`. Tweets contain tokenized text, UTC `created_at`, and user ID; prices are daily OHLCV.
- FinMultiTime: paper tables and the official-author Hugging Face file inventory at revision `72d4e97`; raw U.S. text/price samples were unavailable there.

No sampled copyrighted body text is committed. Sample bytes and hashes remain in the ignored working area.
