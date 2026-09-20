# ECNI final data architecture

- Original native FNSPID role remains `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS`.
- Derived relinked role: `RELINKED_ROLE_B_DISSEMINATION_AND_DENSE_TEXT`.
- Architecture: `ARCH_B_SEC_FACTS_PLUS_RELINKED_NEWS_DISSEMINATION`.
- Relinked edge body fraction: 66.6718%; timestamp-eligible fraction: 100.0000%.
- Balanced panel gate: `True`.

News facts are usable only when supported by an explicit high-confidence issuer edge and private evidence reference. SEC acceptance time is authoritative for filing availability. Missing factual evidence remains unknown. This choice uses only entity, timing, text, coverage, SEC and access evidence; no prediction result exists.
