# Reaction v3 corrected data audit

## Scope

This run uses the full time-safe reaction corpus for AAPL/AMZN four-hour
exploration.  It does not claim that the provisional event labels are human
gold.  The source label audit is the existing FinBERT event-adapter audit and
is repeated here so the article-level and downstream claims are not confused.

| item | value |
|---|---:|
| provisional accepted article rows | 458 |
| train / development / check | 249 / 87 / 122 |
| train positive articles / facts | 60 / 75 |
| train positive articles (AAPL / AMZN) | 47 / 13 |
| check positive articles (AAPL / AMZN) | 17 / 5 |
| event facts in that panel | 118 |
| independent human gold review | **not completed** |

The available fields are `kind`, `action`, `old`, `new`, `unit`, and
`evidence_ids`.  No target-company or event field is invented when it is
absent.  This v3 reaction experiment does not use those provisional event
labels to create the article reaction target; its article target is the
strictly later 60- or 240-minute signed return.

## Full reaction corpus

The raw-to-canonical counts are: 78,055 raw rows, 70,732 English rows, 70,601
rows with sufficient English text, 89,958 candidate article-target pairs and
85,402 canonical groups.  In January--August there are 51,485 canonical groups;
40,764 AAPL and 10,721 AMZN groups.  Both stocks have 168 session dates and
pass the registered minimum-data feasibility check.  Valid 60-minute reactions
are 11,288 (AAPL) and 3,137 (AMZN); valid 240-minute reactions are 5,490 and
1,410 respectively.  Thirty- and 120-minute reactions are audit-only.

Availability is `max(valid published time, crawled time)`.  Duplicate groups
use the earliest available canonical record.  The target-pair audit and its
sample counts are in [TARGET_ASSOCIATION_AUDIT.md](TARGET_ASSOCIATION_AUDIT.md).

## Time and split audit

Article models select C only on March--May chronological folds.  June--August
is the article gate.  September onward is an exposed historical period and
never selects a horizon or gate.  Training reaction labels are required to be
mature before the first evaluation article availability.  Price context uses
only completed five-minute bars whose end is at or before article availability;
the v3 verifier also replays a future-price perturbation.

The v3 article predictions contain no raw title/body text.  The 458-row event
panel and the 85,402-group reaction corpus are separate evidence sources; the
former is provisional extraction supervision, not an independent association
gold set.
