# ECNI Stage E1 preregistration

Frozen before new acquisition at repository SHA `287649ec82a72bee72982f2bf311f99acfc82d2e` on 2026-09-19.

## Boundary

E1 is metadata, availability, identity, licensing and point-in-time evidence qualification. Stock-direction predictors, reader training, BA, MCC, Brier, trading returns, label-balance inspection and return-conditioned stock/year selection are prohibited. Existing V6--V10, F0/F1/F2, Market Context and relation-reader artifacts are protected.

## Sources and acquisition hierarchy

Authoritative sources are official `Zihan1004/FNSPID` at its acquired immutable Hugging Face revision, `BigRoddy/CMIN-Dataset`, `Zhihan1996/TradeTheEvent`, SEC EDGAR APIs/archives, and stable company/sector identity sources recorded in the manifest. Community mirrors may diagnose schema only and cannot become authority without byte equivalence.

Use: dataset-server/streaming/query; deterministic range reads; smaller official news file; then the large Nasdaq file only if required. Official `Stock_price/full_history.zip` is authorized. Raw bodies remain private and are not committed.

## Deterministic sampling

Seed is integer `57320260919`. Before interpretation, records are assigned `SHA256(source_file || canonical_record_id || seed)`. Within available file × year × publisher/source strata, take the lowest hashes, targeting at least 2,000 total records and at least 20 records per populated selected stratum where capacity permits. Entity review uses the lowest 600 hashes balanced across file, year and symbol as far as the accessible data permits. Sample IDs are frozen in `FNSPID_STRATIFIED_SAMPLE_MANIFEST.json` before quality summaries.

## FNSPID gates

- `ROLE_A_FULL_ECNI_SOURCE`: bodies/evidence non-null on at least 70% of sampled records, at least 80% timestamps classifiable as exact or conservative date-only, and sampled `DIRECT_TARGET + MULTI_COMPANY_INCLUDES_TARGET` Wilson lower 95% bound at least 0.80.
- `ROLE_B_PANEL_DISSEMINATION_SOURCE`: title/URL/publisher metadata non-null at least 90%, at least 80% timestamps classifiable, but Role A body threshold fails.
- `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS`: price coverage qualifies but news metadata/timing or entity evidence fails Role B.
- `ROLE_D_REJECT_PRIMARY`: price corpus or stable identity/common panel requirements fail.

Timestamp classes are `EXACT_INTRADAY_USABLE`, `DATE_ONLY_CONSERVATIVE`, `AMBIGUOUS`, `INVALID`. Exact time needs parsed time plus source/timezone evidence. Midnight without such evidence is date-only, not automatically invalid. Date-only evidence becomes available at the next applicable regular-market open after the recorded calendar date.

Price eligibility requires unique monotone dates, valid finite OHLC with `low <= min(open,close) <= max(open,close) <= high`, and at least 95% of common 2018--2023 sessions present. Corporate actions are audited through adjusted-close/split fields when available, not inferred from returns.

## Panel and splits

Training metadata period is 2018-01-01 through 2021-12-31. Target panel is 11 sectors × 6 stocks, with two each from within-sector high/medium/low news-coverage terciles. Eligibility requires verified issuer identity, sector, price coverage and acceptable timestamp metadata. Ties: greater common-session coverage, fewer missing cells, greater identity confidence, then lexicographic `CIK|ticker`. If infeasible, use the largest balanced panel with at least five sectors and four stocks per sector.

Company split uses `SHA256(permanent_company_id || "ECNI_E1_COMPANY_SPLIT")`, stratified by sector and coverage stratum: approximately 2/3 train, 1/6 development-unseen and 1/6 locked-unseen. Candidate time split is 2018--2021 train/forward validation, 2022 development and 2023 locked test. It may change only for coverage/timestamp availability, never outcomes. Locked-period direction labels will not be calculated or summarized.

## Daily task and audits

Future cutoff is XNYS regular open minus five minutes; target is same-session open-to-close binary direction. No-news days remain. `EXACT_INTRADAY_USABLE` enters only if published by cutoff; date-only uses the next-open rule; ambiguous/invalid is excluded from primary text.

Duplicate candidates preserve dissemination: exact URL, normalized-title hash, title similarity at least 0.90, then body similarity at least 0.90 where legal/private bodies exist. They are not collapsed in E1.

The SEC pilot chooses one frozen-panel company per sector using the smallest hash of permanent ID, with bounded 8-K/10-Q/10-K metadata. Period end and public availability remain distinct.

## E1 decision

`PASS_READY_FOR_READER_PILOT` requires the broader audit, price corpus, deterministic named panel and company/time splits, outcome-uninspected 2023 lock, tested evidence-ledger schema, architecture and human-label protocol, and zero predictive fits/scores. Otherwise emit `CONDITIONAL_DATA_REDESIGN_REQUIRED` or `FAIL_PRIMARY_DATA_REJECTED` using only these gates.
