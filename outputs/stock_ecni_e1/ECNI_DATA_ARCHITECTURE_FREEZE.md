# ECNI data architecture freeze

## Decision: ARCH_C, conditional

Use FNSPID news bodies only when a record has usable evidence, a legally stored private reference, a usable availability class, and verified target association. Otherwise factual innovation `U` remains missing/unknown. Never infer a fact from a ticker tag alone.

The bounded sample contains complete bodies in 100% of sampled Nasdaq-file rows but only 18.53% of `All_external` rows. Nearly all records are date-only, and the provisional target-association lower confidence bound is 48.83%, below the preregistered 80% gate. FNSPID therefore receives `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS` for this E1 decision. Its price archive remains useful; its sampled titles and duplicate structure remain diagnostic, but it is not a qualified primary panel/dissemination source.

SEC is the intended high-confidence factual source for filing events. The SEC pilot did not run because the deterministic panel prerequisite failed. This is a data-architecture decision based only on availability, timestamps, identity evidence, and licensing; no stock outcomes were used.
