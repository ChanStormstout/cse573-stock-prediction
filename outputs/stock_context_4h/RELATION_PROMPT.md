# Frozen relation-reader prompt

Compare only the supplied evidence for `TARGET`. Do not predict stock direction,
market reaction, importance, priced-in status, or first disclosure. Do not merge
different actors, institutions, objects, or reporting periods. A maintained rating
and a changed target price are separate facts. Each non-null old/new value must
occur literally in cited evidence. If support is insufficient, use `unknown`.

`pair_relation` must be **exactly one** of:

- `repeat_same_fact`
- `new_or_changed_fact`
- `explicit_correction_or_denial`
- `background_or_recap`
- `different_actor_object_or_period`
- `unrelated_or_insufficient`
- `unknown`

Return JSON only with this schema:

```json
{
  "pair_relation": "one enum above",
  "current_evidence_ids": [],
  "past_evidence_ids": [],
  "changes": [{
    "actor": null,
    "object": null,
    "old_value": null,
    "new_value": null,
    "unit": null,
    "period": null
  }]
}
```
