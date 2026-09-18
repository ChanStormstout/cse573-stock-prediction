# Dataset and representation audit

- Unchanged four-hour task: 1,607 unique AAPL/AMZN windows; 233 warmup, 765 March–August forward OOF, 252 development, 357 later. Raw 48-five-minute-bar labels independently agree for all windows. No outcome or flat-return filtering was added.
- Every stock fit uses realized label ends strictly earlier than the first evaluation cutoff. Cross-stock state is reconstructed from raw bars, never joined using another stock's future label or existence of its four-hour row.
- Original and all three A1 encoders use exactly the same token hash: 5,226 article–target pairs, 10,201 protected candidate units, 20 title fallbacks. These are representation inputs, not new event annotations. Existing A1 checkpoints were trained on January–February provisional labels and only used for March-onward OOF predictions.
- Historical universe preserves English/body-length/nonnegative-lag/company-title/no-physician filters and earliest normalized-body/title deduplication. 18,599 distinct article records are used across current and historical buckets; 13,521 extra titles were encoded. The interval is current plus previous two scheduled trading sessions, including intervening nights.
- AMZN original-empty windows filled by history: training OOF182/184, development53/53, later95/95. Warmup54/57. This is availability, not event quality or predictive success.
- Historical reaction uses only completed scheduled bars after availability and before cutoff; any required missing bar means unknown. Cross-night returns include the gap and carry an explicit overnight flag. Six fixed slow-reference reconstructions agree within1e-10.
- All existing periods have been exposed. No independent new holdout, additional gold labels, or independent human extraction review was created.
- Alpaca SPY/QQQ sample for January10,2018 returned401 without credentials. No external-market features are included. Raw articles, features, course files and model binaries are not redistributed.
