# External dataset next step (plan only)

| Dataset | Coverage | Compatibility | Risks / burden | Role |
|---|---|---|---|---|
| Current CSE573 dataset | AAPL/AMZN 2018--19 course news and charts | Canonical 4h and preregistered 1d | Provider timestamps, repetition, two stocks, short span | **Primary benchmark** |
| FNSPID | Financial news; release/version needs source audit | 1d possible; 4h requires reliable intraday timestamps/prices | Licence, ingestion/publication times, entities, storage | Robustness candidate |
| StockNet | Tweets/news plus sequences | Published daily convention, not automatic 4h compatibility | Terms, timestamps/labels, cross-dataset leakage | External validation candidate |

No external data are downloaded, trained, or scored. First conduct a
source/licence/timestamp audit. Course data remains the primary benchmark.
