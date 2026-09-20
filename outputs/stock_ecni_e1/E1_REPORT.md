# ECNI Stage E1 report

## Executive decision

**Status: `CONDITIONAL_DATA_REDESIGN_REQUIRED`.** FNSPID is not qualified as the primary news panel. It is assigned `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS`; the official price corpus is useful, while news can only enter a later `ARCH_C` design after target and timing checks. No named panel or company split is frozen.

## Audited evidence

The sample manifest was frozen before interpretation and contains 7,711 records: 5,711 from `All_external` and 2,000 from the Nasdaq file, spanning 17 years, 358 source labels, and 55 ticker tags. The private sample hash is `acc5310f0cf6f8bba0a95f99a4fb34d98c24a95bf115179308019597c5b8a3a6`.

| File | Records | Intraday | Date-only | Body | Title | Summary | Publisher | URL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All_external | 5,711 | 4.06% | 95.94% | 18.53% | 100% | 0% | 95.31% | 99.98% |
| Nasdaq | 2,000 | 0% | 100% | 100% | 100% | 100% | 0% | 100% |
| Combined | 7,711 | 3.01% | 96.99% | 39.66% | 100% | 25.94% | 70.59% | 99.99% |

Date-only records become available at the next applicable regular-market open after their recorded calendar date. They are never placed at the same-day pre-open cutoff.

The provisional deterministic entity audit examined 600 frozen records: 54 direct-target, 263 multi-company including target, 15 indirect/competitor, and 268 with no target evidence. The association rate is 52.83%; its 95% Wilson interval is 48.83%–56.80%. This is a lexical audit, not human gold, and fails the preregistered 80% lower-bound gate.

Within the frozen sample, exact normalized titles produce 198 repeated groups containing 521 records (maximum group size 12); 44 groups span multiple source labels. The bounded near-title comparison finds 2,285 similarity edges at 0.90. Exact URL repetition is zero because sampled record IDs/URLs were deduplicated before sampling. These are dissemination candidates, not collapsed events.

The official price archive SHA-256 matches the frozen remote hash. It contains 7,693 symbol files; 3,172 pass the strict 2018–2023 availability/integrity screen. AAPL is among them. This establishes broad price feasibility without computing returns or directions.

## Why the panel is not frozen

The bounded byte-range sample is adequate for quality rates but cannot recover complete stock/day news counts for 2018–2021. Therefore high/medium/low coverage ranks cannot be computed. The current SEC and Nasdaq identity snapshots also lack a versioned sector/industry taxonomy and do not establish historical ticker membership. A named 66-stock or fallback panel would violate the preregistration. `FROZEN_STOCK_PANEL.csv` and `COMPANY_SPLITS.json` record this stopped state rather than invented assignments.

The proposed 2018–2021 / 2022 / 2023 time partition remains frozen conditionally. Existence/coverage was audited, but 2023 outcomes were not inspected or summarized.

## Auxiliary-source decisions

The SEC point-in-time pilot did not run because one company per represented frozen sector could not be selected. The evidence-ledger JSON schema and query contract are implemented for future NEWS/SEC evidence. CMIN-US official dates were recovered from the correct ACL 2023 paper (long paper 679): train through 2021-04-30, development May–August 2021, test September–December 2021. EDT remains `BLOCKED_PENDING_ACCESS_OR_TERMS`.

## Required redesign

Acquire a complete metadata index for the news corpus and a versioned issuer/sector mapping. Then rerun the registered panel rule outcome-blind. If those inputs cannot be obtained, reconsider CMIN-US for the named benchmark and use SEC plus separately licensed point-in-time news for the ECNI reader lane.

## Scientific boundary

- Predictive stock models fitted: **0**
- BA/MCC/Brier calculated: **0**
- Return-based stock selection: **0**
- Locked 2023 direction labels inspected or summarized: **0**
- Reader trained: **0**
