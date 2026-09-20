# Evidence-Grounded Conditional News Innovation specification

## Structured statement

Each statement stores `target_company`, `actor`, `object`, `event_type`, `action`, `metric`, `old_value`, `new_value`, `unit`, `event_period`, `assertion_status`, `available_time`, `source`, `evidence_span`, `predecessor_id`, and `reader_uncertainty`.

`assertion_status` is one of `reported_fact_or_claim`, `plan`, `opinion`, `hypothetical`, `allegation`, `denial`, or `unknown`. It describes what the source asserts, not metaphysical truth.

## Factual innovation U

A predecessor is comparable only when company/relevant entity, event family, metric, period and unit/currency are compatible where applicable. U contains `new_event_indicator`, `action_transition`, `normalized_numeric_delta`, `period_transition`, `assertion_transition`, `contradiction_or_denial`, `time_since_predecessor`, `predecessor_similarity`, `comparison_valid`, and `reader_confidence`. Invalid comparisons have an explicit missing/not-comparable state; no delta is fabricated.

## Dissemination D

D contains cutoff-safe `article_count_so_far`, `independent_source_count_so_far`, `time_since_first_observation`, `time_since_latest_observation`, `source_entropy`, `republication_ratio`, and `arrival_rate`. Repetition can update D but cannot change the factual state by itself.

## Dense channel E

Use the identical eligible article pool, cutoff, target context and token budget for E0 sparse F1 text, E1 FinBERT, E2 Fin-ModernBERT, and E3 a small Qwen reader representation only after reader quality passes. Additional text is an additional-data comparison, not an encoder comparison.

## Future predictor

Inputs are price P + market/sector M + dense semantics E + factual innovation U + dissemination D. Begin with regularized logistic regression and one capacity-matched small nonlinear control. Candidate partial pooling is `theta_company = theta_shared + theta_sector + small_company_deviation`, with L2 shrinkage. No predictor is fitted in this audit.

## SEC auxiliary design

Use SEC submissions/company-facts endpoints and accession-aware filing archives. Retain CIK, accession, form, filing/acceptance availability time and version for 8-K, 10-Q, 10-K and relevant exhibits. Keep `period_end` separate from `public_available_time`. SEC can improve issuer identity and filing facts; it does not supply general news dissemination, market prices, analyst actions, media timestamps, or prove market novelty.
