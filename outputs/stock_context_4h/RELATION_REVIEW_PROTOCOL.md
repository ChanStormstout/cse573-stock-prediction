# Blind relation review protocol

This review covers exactly the 28 frozen `locked_check` cards. Reviewers see
only `pair_id`, `target`, and the CURRENT/PAST sentence IDs and text. They must
not see Qwen output, sampling stratum, title similarity, family metadata, prices,
outcomes, or reactions.

For every card, record an empty-template field set:

```json
{"pair_id":"","fine_relation":"repeat_same_fact | new_or_changed_fact | explicit_correction_or_denial | background_or_recap | different_actor_object_or_period | unrelated_or_insufficient | unknown","evidence_sufficient":false,"reviewer_uncertain":false,"brief_reason":""}
```

Coarse mapping is frozen:

- `REPEAT_BACKGROUND`: `repeat_same_fact`, `background_or_recap`
- `FACTUAL_INCREMENT`: `new_or_changed_fact`, `explicit_correction_or_denial`
- `NOT_SAFELY_COMPARABLE`: `different_actor_object_or_period`,
  `unrelated_or_insufficient`, `unknown`

All 28 cards are the denominator. Qwen's one mechanically invalid output counts
as incorrect. The fixed mechanical baseline maps `near_duplicate_candidate` to
`REPEAT_BACKGROUND` and every other stratum to `NOT_SAFELY_COMPARABLE`; it never
predicts `FACTUAL_INCREMENT`.

The gate is coarse accuracy >= 0.75, at least 24/28 sufficient-evidence cards,
and recall >= 0.60 for every gold coarse class with at least five cards. Report
FACTUAL_INCREMENT precision and require >= 0.75 before using it as a downstream
positive feature when gold support permits interpretation. Also report seven-way
accuracy, confusion matrices, and per-target results. These are pilot estimates.
