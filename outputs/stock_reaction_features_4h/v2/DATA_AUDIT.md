# Reaction v2 data audit

- Source: the complete 78,055-row news index, not the old accepted subset.
- Candidate units: 89,958 `(article_key,target_symbol)` pairs; 85,402
  canonical duplicate groups.
- Jan--August feasibility: AAPL 40,764 groups / 11,288 valid 60-minute and
  5,490 valid 240-minute reactions; AMZN 10,721 / 3,137 / 1,410.  Both pass
  the registered feasibility gate.
- Availability is `max(valid published time, crawled time)`.  The target pair
  key is retained through every private and downstream join.
- Pre-event context uses only completed five-minute bars and records the last
  used bar end.  Reaction bars start at or after availability and mature only
  when contiguous and before the session close.
- Deterministic tier-stratified association cards are private.  No independent
  human gold review is claimed.

The public CSV files contain aggregate audit data and predictions, not article
body text or private feature matrices.
