# Entity and ticker association audit

- **Course:** 78,021/78,055 records contain AAPL and 18,118 contain AMZN under the earlier raw audit; AMZN records overlap the Apple-oriented collection. Association is query/package based rather than an independently verified target label.
- **CMIN-US:** files are stored under ticker and rows contain ticker/company fields. The inspected AAPL sample includes broad multi-company market articles, so directory membership is not proof that AAPL is the event object. `CMIN_POINT_IN_TIME_REAUDITED` must retain an explicit target-evidence status.
- **FNSPID:** records carry `Stock_symbol`; sample titles include multi-stock roundup articles. The mapping procedure and false-association rate remain UNKNOWN.
- **FinMultiTime:** the paper describes ticker-frequency weighting and retaining a daily representative narrative. Raw mappings are unavailable, so target quality is UNKNOWN.
- **EDT:** its own README warns that the greedy automatic ticker recognizer assigns random tickers to some non-company articles; the claimed 98% applies to company-specific articles, not the full benchmark.
- **StockNet:** ticker-directory/cashtag collection is transparent but can include multi-ticker tweets.
- **SEC:** CIK plus accession is the strongest identity anchor, but issuer filings can discuss counterparties; statement-level target/object roles are still required.

No file or ticker directory is treated as proof of event-object identity.
