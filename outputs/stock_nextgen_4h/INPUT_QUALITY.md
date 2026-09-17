# Paragraph input quality

Training-only mechanical audit checked 32,106 window-passage occurrences (repeated articles can appear in adjacent windows): exact raw-span rate 1.000, explicit-target rate 1.000, complete-unit rate 1.000.

In the 16 fixed blind cards, P0 contained 106 character-cut fragments and P1 contained 0. P1 exceeded its matched P0 prompt budget in 0 rows. P3 removed 168 repeated training-period reports.

Assistant review status: **PASS_ENGINEERING_GATE**. This was not an independent review. Background/stale facts can remain when they explicitly concern the target company.

P3 keeps report counts in its sealed window input and independent-source counts in `DEDUP_AUDIT.csv`; raw member identities and article text remain private. Counts are window-cluster occurrences, so a report can recur in adjacent prediction windows.

| Stock | Variant | Rows | Mean tokens | Max tokens | Mean passages | Zero-passage rows |
|---|---|---:|---:|---:|---:|---:|
| AAPL | P1 | 803 | 2702.6 | 3314 | 8.26 | 6 |
| AAPL | P2 | 803 | 4988.1 | 6000 | 24.20 | 6 |
| AAPL | P3 | 803 | 4907.6 | 6000 | 23.67 | 6 |
| AMZN | P1 | 804 | 1225.9 | 2890 | 1.43 | 422 |
| AMZN | P2 | 804 | 1569.6 | 5871 | 4.40 | 402 |
| AMZN | P3 | 804 | 1520.4 | 4939 | 4.00 | 405 |
