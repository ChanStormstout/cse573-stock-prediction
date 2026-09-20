# ECNI Stage E1R-V2 preregistration

Frozen before inspecting or generating semantic review cards. Historical E1R at `0f3719e7ef3b386c3e8e7df3780e056af16c1eaf` remains immutable. This stage is outcome-blind and creates a new `outputs/stock_ecni_e1rv2/` lineage.

## Prohibited access and computation

No returns, UP/DOWN labels, BA, MCC, Brier, 2022/2023 predictive outcomes, reader training, FinBERT/Fin-ModernBERT/Qwen training, or stock-predictor fitting. No semantic rule may be selected using market outcomes. Copyrighted article text and private review cards remain outside Git.

## Semantic relations

Every reviewed article-company candidate receives exactly one label: `DIRECT_TARGET`, `MULTI_COMPANY_DIRECT`, `INDIRECT_COMPETITOR_OR_COUNTERPARTY`, `VENUE_OR_MARKET_NAME_ONLY`, `PUBLISHER_OR_SOURCE_NAME_ONLY`, `INCIDENTAL_MENTION`, `NO_TARGET_EVIDENCE`, or `UNCERTAIN`.

`DIRECT_TARGET` and `MULTI_COMPANY_DIRECT` require that the resolved company itself is an actor, object, or subject of a substantive reported event or statement. Listing or trading venue references alone are not company-event evidence. Automated relinker output is never treated as semantic gold.

## Alias-risk definitions

An alias is mechanically flagged when it may also denote an exchange, index, market, industry/common noun, geographic entity, product/category, another common organization, common surname/word, or publisher/source name. Additional diagnostics flag extreme article/day coverage, strong publisher concentration, exchange/listing boilerplate, edge-count peer outliers, and overlap with registered exchange/site names. One-alias-to-one-CIK uniqueness is insufficient evidence of semantic safety.

The sentinel alias `nasdaq` for CIK 1120193 / NDAQ is always `DUAL_ROLE_EXCHANGE_COMPANY` risk. This preregistration does not decide its semantic precision.

## Deterministic entity-review sampling

Seed string: `ECNI_E1RV2_ENTITY_REVIEW|57320260919`.

1. Panel base set: for each of the 66 historical E1R selected companies, select up to five high-confidence edges using deterministic SHA-256 rank while greedily covering, when available, title/body match, native agreement/disagreement, and single/multi-company relation strata. If five exist, exactly five are selected.
2. NDAQ sentinel: select at least 100 NDAQ high-confidence edges. Allocate deterministically across year, source file, title/body field, and native agreement/disagreement strata, then fill by SHA-256 rank.
3. Alias-risk enrichment: select at least 100 edges from the highest mechanically flagged aliases, including NDAQ. Rank aliases before article inspection by risk-flag count, training coverage, edge count, and deterministic alias/CIK hash. Sample candidates by deterministic hash within alias and diagnostic strata.
4. Union and deduplicate by `(record_id_hash, company_id)`. The manifest records overlap, stratum shortfalls, hashes, and counts. Private cards include target, title, evidence context/span, alias, field, native ticker, publisher, and date, with no outcomes.

The locked set is suitable for blinded independent labeling. Codex may create cards and diagnostics but any Codex label is non-independent and cannot satisfy the precision gate.

## Frozen semantic precision gate

Accepted labels are only `DIRECT_TARGET` and `MULTI_COMPANY_DIRECT` for edges currently used as high confidence.

- Overall selected-panel point precision must be at least 0.90.
- Overall selected-panel 95% Wilson lower bound must be at least 0.85.
- No selected company with at least five reviewed examples may have precision below 0.60.
- The NDAQ sentinel must independently have precision at least 0.90; otherwise the ambiguous `nasdaq` alias must be repaired before panel use.

No threshold may be relaxed after labels are observed. Without independent labels, status is `AWAITING_INDEPENDENT_ENTITY_REVIEW` and no semantic precision is reported.

## Frozen rule-level repair and panel reconstruction

If independent review later fails an alias, repair the alias rule rather than deleting articles. Dual-role exchange/company aliases require explicit corporate context; bare exchange/market names are forbidden as issuer evidence; ambiguous aliases require full legal/company context; explicit `$TICKER` or `EXCHANGE:TICKER` remains strong evidence. A repair triggers complete-corpus relinking, never row edits.

If repaired coverage changes, rerun the exact historical E1R industry × coverage-tier × deterministic selection algorithm. The repaired panel is named `FROZEN_STOCK_PANEL_E1RV2.csv`; E1R files remain unchanged. Recompute the frozen per-industry 4/1/1 company split using the same deterministic hash procedure. No current member is manually retained.

Before independent labels, no repaired relinking or panel rebuild is authorized merely by mechanical flags. E1RV2 records the historical 66-company panel and 44/11/11 split as awaiting semantic validation.

## Foreign-filer SEC procedure

Classify each selected CIK from SEC submissions metadata available without market outcomes as `DOMESTIC_REGISTRANT`, `FOREIGN_PRIVATE_ISSUER`, or `OTHER_OR_UNCERTAIN`. Domestic factual forms are 8-K, 10-Q, and 10-K. Foreign-private factual forms include 6-K and 20-F as appropriate. Counts use only filings accepted in 2018–2021 and preserve CIK, accession, form, acceptance time, period end, document reference, industry, company split, and filer class. Missing forms remain explicit missingness, never evidence of no facts.

## Reader-pair preparation boundary

Semantic validation or a rule-level repair is required before constructing final news-based current/predecessor pairs. With no independent labels, E1R-V2 may prepare only SEC-backed pair candidates and a deferred news-pair specification; it must not train a reader. Any bounded review pack keeps near-duplicate families together and assigns no market-direction label.

## Final decision rule

Without independent semantic labels, final status is `AWAITING_INDEPENDENT_ENTITY_REVIEW`. Other statuses require legitimate independent labels and application of the frozen gates. In all cases reader and predictive training remain unauthorized.
