# ECNI Stage E1R-V2 report

## Status

**`AWAITING_INDEPENDENT_ENTITY_REVIEW`**. Historical E1R remains preserved. No independent semantic labels were supplied, so semantic precision and the registered gate are not measurable yet.

## Mechanical findings

The complete alias census contains 3,129 alias/CIK rows. NDAQ's `nasdaq` alias has 2,576,470 all-years matches, 1,315,640 all-years high-confidence edges, and 255,784 2018–2021 high-confidence edges. Its historical E1R panel coverage was 1.0. The dedicated 100-card sentinel contains 3 mechanical listing/venue phrase matches and 15 narrow corporate-context matches. These are contamination diagnostics, not semantic labels.

The private blinded entity pack contains 515 unique cards: 330 panel-base cards covering all 66 companies, 100 NDAQ sentinel cards, and 100 alias-risk-enriched cards before overlap. No copyrighted card text is committed; the public manifest records the private hash and card IDs.

## Panel and repair boundary

No alias rule was changed, complete relinking was not rerun, and no E1RV2 replacement panel was created before independent labels. The historical panel remains 66 companies with 44/11/11 company splits, pending semantic validation. A failed alias must trigger a rule-level repair and full relinking; individual rows cannot be deleted.

## SEC channel

Filer classes are `{'DOMESTIC_REGISTRANT': 54, 'FOREIGN_PRIVATE_ISSUER': 12}`. Training-period form totals are `{'8-K': 2903, '10-Q': 655, '10-K': 237, '6-K': 1343, '20-F': 55}` and 5,193 outcome-blind SEC filing candidates were retained. Foreign-private issuers use 6-K/20-F coverage rather than being judged by domestic forms alone. Per-company form counts range from 19 to 238, so the frozen mechanical criterion flags severe coverage inequality; missingness must remain explicit.

## Reader boundary

Reader-pair construction and reader training are blocked. The next required action is independent blinded semantic labeling, including the frozen NDAQ sentinel. Returns/outcomes inspected: **0**. Predictive models fitted: **0**.
