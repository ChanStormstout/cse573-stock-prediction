# Timestamp audit

| Dataset | Finding | Evidence status | Consequence |
|---|---|---|---|
| Course | Published and crawl timestamps exist; 14.93% previously observed with crawl before published | CONFIRMED_FROM_SAMPLE_RECORD | Keep both; do not infer first disclosure |
| CMIN-US | AAPL sample uses second-resolution datetimes, e.g. `2021-12-31 21:39:55` | CONFIRMED_FROM_SAMPLE_RECORD | Timezone is not encoded and remains UNKNOWN; open-minus-5m is unsafe until resolved |
| FNSPID | 10/100 sample rows have non-midnight UTC times and 90/100 are exactly midnight | CONFIRMED_FROM_SAMPLE_RECORD | Daily task can conservatively use only information assigned by a frozen date rule; strict intraday is unsupported without a broader audit |
| FinMultiTime | Paper calls news minute-level, but raw timestamp fields were unavailable | CONFIRMED_FROM_PRIMARY_SOURCE / UNKNOWN schema | Block point-in-time use |
| EDT | Official README says adjusted minute-level publish time | CONFIRMED_FROM_PRIMARY_SOURCE | Suitable for event-time study after payload validation |
| StockNet | Sample has exact UTC tweet timestamps | CONFIRMED_FROM_SAMPLE_RECORD | Point-in-time tweet task feasible within 2014--2016 |
| SEC | Filing metadata can retain accepted/filing timestamps | CONFIRMED_FROM_PRIMARY_SOURCE | Use public availability, never period end |

For date-only records, no time is invented. The candidate conservative rule is eligibility at the next regular-market open after the recorded calendar date, subject to source timezone validation.
