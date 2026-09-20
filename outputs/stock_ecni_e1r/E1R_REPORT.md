# ECNI Stage E1R report

## Decision

**`PASS_PANEL_AND_ARCHITECTURE_FROZEN`**. E1's native association rejection remains unchanged. The derived complete-corpus view is `RELINKED_ROLE_B_DISSEMINATION_AND_DENSE_TEXT` and the final architecture is `ARCH_B_SEC_FACTS_PLUS_RELINKED_NEWS_DISSEMINATION`.

## Complete census and relinking

The two official files contributed 28,606,813 processed rows with 0 parse failures. Native ticker was diagnostic only. Independent distinctive SEC issuer aliases or explicit ticker syntax created 7,377,941 high-confidence article-company edges. This is deterministic evidence-rule support, not independently measured human precision.

Relinked edges have 66.67% body availability and 100.00% exact-or-conservative timestamp eligibility. Exact-time and date-only records remain separate. The complete 10,848,303-edge ledger and 31,457,434-group duplicate ledger are locally retained and hash-addressed in `LOCAL_CORPUS_ARTIFACTS_MANIFEST.json`; they are not committed because they exceed repository artifact limits.

## Panel

The frozen selection contains 66 companies across {'Money': 6, 'Hlth': 6, 'BusEq': 6, 'Durbl': 6, 'Chems': 6, 'Enrgy': 6, 'Utils': 6, 'NoDur': 6, 'Manuf': 6, 'Shops': 6, 'Telcm': 6} historical Fama–French industries. The registered requirement is at least eight industries with four companies each; panel gate = **True**. Company split status is `FROZEN` with `{'TRAIN_COMPANIES': 44, 'DEV_UNSEEN_COMPANIES': 11, 'LOCKED_UNSEEN_COMPANIES': 11}`; each represented industry contributes 4 train, 1 development-unseen, and 1 locked-unseen company. Time split status is `FROZEN`. 2023 outcome labels were not inspected or summarized.

## SEC and reader boundary

The SEC pilot contains 66 filing metadata objects. Reader input counts are recorded, but no reader was trained and no reader score exists. CMIN-US remains a separate standard benchmark.

## Outcome-blind confirmation

- Stock predictive models fitted: **0**
- BA/MCC/Brier computed: **0**
- Return-based stock selection: **0**
- Reader models trained: **0**
- Locked 2023 outcomes inspected: **0**
