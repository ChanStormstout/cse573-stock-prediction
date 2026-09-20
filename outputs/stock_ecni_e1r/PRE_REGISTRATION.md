# ECNI Stage E1R preregistration

Frozen on 2026-09-19 at reviewed starting SHA `eed3d3461f4b5176f76dc2ed471d51db73fb04f7`, before complete-corpus acquisition or coverage inspection. E1's native FNSPID decision remains immutable: `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS`.

## Outcome-blind boundary

E1R is a data-redesign and provenance stage. It will not construct or inspect stock directions, 2022/2023 UP/DOWN labels, returns for selection, BA/MCC/Brier, trading results, reader models, or stock predictors. Prices are restricted to availability, missingness, calendar integrity, and corporate-action metadata.

## Frozen sources

- FNSPID official Hugging Face dataset, revision `bf9189c41527198897d1af3e17b1a0095279fc45`: `Stock_news/All_external.csv` SHA-256 `5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3`, `Stock_news/nasdaq_exteral_data.csv` SHA-256 `1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da`, and `Stock_price/full_history.zip` SHA-256 `03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91`.
- SEC EDGAR submissions/company-ticker data acquired from official SEC endpoints with acquisition timestamps and hashes.
- Historical SIC from SEC submissions/filing metadata available by the relevant information date.
- Fama–French 12 industry definitions from Kenneth French's published SIC ranges; exact fetched version/hash will be recorded.

Raw news bodies remain private. Full news CSVs may be streamed/chunked locally and removed after safe metadata/relink artifacts are built. Public outputs contain hashes, spans/field identifiers, counts, and no copyrighted bodies.

## Complete metadata census

Process every parseable row in both official news files. Preserve: source file, stable record hash, parsed recorded time, year, publisher/source, native tag as `NATIVE_CANDIDATE_TAG`, title hash and normalized-title hash, normalized URL hash, body presence/length/hash, summary presence, duplicate fingerprints, timestamp class, and independently resolved company edges. Count total/parsed/failed rows per file and year. Native ticker is never company ground truth.

## Timestamp policy

Classes remain `EXACT_INTRADAY_USABLE`, `DATE_ONLY_CONSERVATIVE`, `AMBIGUOUS`, and `INVALID`. An exact time needs a parsed non-midnight value and explicit source/timezone evidence. A date-only record becomes available at the next applicable regular-market open after the recorded calendar date, never same-day pre-open. Exact-time and date-only censuses remain separate.

## Issuer identity and safe aliases

Permanent key is SEC CIK. Build issuer names, former names, ticker/exchange facts, effective information dates, and normalized aliases from SEC evidence. Company-name aliases must retain at least one distinctive token and cannot be only a stopword/common finance term. Bare tickers are prohibited evidence when alphabetic or common-word ambiguous, including all one-letter tickers and examples `C`, `D`, `CAT`, `IT`, `ALL`. A ticker is evidence only in `$TICKER`, `NASDAQ:TICKER`, `NYSE:TICKER`, or an equivalent explicit exchange-symbol syntax. Ambiguous aliases are recorded and never automatically matched.

## Relinking rules

Relink from title/body/metadata without prices. A high-confidence edge requires either: (a) an unambiguous normalized issuer/former-name alias in title/body; or (b) explicit ticker syntax. One article may have zero, one, or several edges. One high-confidence company yields `DIRECT_TARGET_HIGH_CONFIDENCE`; multiple independently evidenced companies yield `MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE`. Competitor-only, ambiguous, and no-evidence cases are excluded from coverage. Native-tag agreement is diagnostic only. Store matched alias, field, character offsets where privately computable, rule, timestamp class, and hashes, never body text.

## Relink audit

Seed `57320260919`. Before interpretation freeze the 600 lowest `SHA256(record_id|company_id|relation_stratum|seed)` candidates balanced as feasible across file, year, publisher, native agreement/disagreement, single/multi edges, ambiguous short symbols, and body/no-body. Report deterministic rule support separately from native association. Without independent human gold, do not call it measured precision; create a locked review manifest.

## Duplicate/dissemination groups

Create candidates from normalized URL, normalized title, body hash, and bounded near-title/body fingerprints. Retain all articles. Group summaries contain first/latest observation, count, publisher inputs, company IDs, and time spread. Future online construction must use only records observed by each cutoff; full-corpus groups are descriptive manifests, never retroactively available features.

## Historical industry taxonomy

Use SEC SIC facts whose public information date falls in 2018–2021, then apply the fixed published Fama–French 12 SIC ranges. A company is stable only if all observed training-period mapped industries agree. Missing SIC is `INDUSTRY_UNKNOWN`; differing mapped industries are `INDUSTRY_UNSTABLE`. Both are excluded from primary panel. No current GICS label substitutes for historical SIC.

## Coverage and panel algorithm

Coverage uses only 2018–2021 high-confidence relinked edges and eligible timestamps. Compute trading days, days with any/exact/date-only news, articles, duplicate groups, publishers, body/title-only counts and timing fractions. Price eligibility inherits E1's strict complete 2018–2023 integrity rule.

Represent every Fama–French industry with at least six valid companies, selecting two per within-industry coverage tercile (`HIGH`, `MEDIUM`, `LOW`). Tier boundaries use ranks within eligible industry, with deterministic `(coverage_fraction, CIK)` ordering. Selection priority is price completeness, stable CIK identity, stable industry, timestamp eligibility, longest dated news span, then smallest `SHA256(CIK|ECNI_E1R_PANEL|57320260919)`. If six are unavailable, allow four with at least one from each nonempty tier and fill remaining positions by the same priority. Primary PASS requires at least eight industries and at least four selected stocks per industry. The rule is not relaxed after coverage inspection.

Company split uses `SHA256(CIK|ECNI_E1R_COMPANY_SPLIT|57320260919)`, stratified by industry and coverage tier, targeting 2/3 train, 1/6 development-unseen, 1/6 locked-unseen. Time split is 2018–2021 train/forward validation, 2022 development, 2023 locked test if existence permits; locked outcomes remain uninspected.

## Task populations and roles

Freeze `DAILY_FIXED_WINDOW` at XNYS open minus five minutes with exact evidence by cutoff plus conservative date-only evidence, retaining no-news days. Freeze `EXACT_TIME_EVENT_CONDITIONED` eligibility counts only; no response horizon is selected.

Derived role gates:
- `RELINKED_ROLE_A_FULL_ECNI`: deterministic high-confidence relations plus at least 70% body-bearing retained edges, at least 80% timestamp eligibility, and PASS panel.
- `RELINKED_ROLE_B_DISSEMINATION_AND_DENSE_TEXT`: PASS panel, at least 80% timestamp eligibility, titles on at least 90% retained edges, but Role A body gate fails.
- `RELINKED_ROLE_C_DISSEMINATION_ONLY`: entity/timing coverage supports dissemination analysis but Role B fails.
- `RELINKED_ROLE_D_REJECT`: complete census/relinking cannot support a defensible balanced panel.

Final architecture is selected only from entity evidence, timestamps, body/coverage, SEC factual coverage, and access/license conditions. If panel PASS, run a bounded SEC metadata pilot on the smallest deterministic CIK hash per represented industry. Reader training remains prohibited; only outcome-blind candidate-pair counts may be produced.

## Stop and status gates

`PASS_PANEL_AND_ARCHITECTURE_FROZEN` requires full metadata census, independent relinking, high-confidence coverage, historical industry ledger, balanced named panel, company/time splits, exact/date-only separation, derived role, SEC pilot, frozen architecture, and zero predictive work. If a balanced panel cannot be frozen, emit `CONDITIONAL_NEWS_SOURCE_REDESIGN_REQUIRED`. If entity relinking itself is not viable, emit `FAIL_FNSPID_RELINKING_INSUFFICIENT`. Stop after E1R regardless of status.
