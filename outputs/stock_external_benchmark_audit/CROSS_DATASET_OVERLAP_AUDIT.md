# Cross-dataset overlap audit

The only locally comparable news samples were the complete course archive and a bounded 394-row CMIN-US AAPL sample. Exact URL overlap was 0. Normalized-title overlap was 3. Within the course archive, 287 URL repetitions and 20,752 normalized-title repetitions were observed among 78,055 records. In the FNSPID first 100 rows, URLs were unique but only 88 titles were unique.

These counts are sample diagnostics, not full cross-corpus estimates. Full acquisition must group records using, in order: canonical URL, normalized full-text hash, normalized title, body near-duplicate similarity, then company/date semantic similarity. Grouping must use only information available at each cutoff when used in prediction.

Dataset names do not establish independence. CMIN and FNSPID both include syndicated financial publishers; FinMultiTime explicitly builds on the FNSPID Nasdaq scraping approach. A future overlap manifest must be frozen before reader or prediction splits.
