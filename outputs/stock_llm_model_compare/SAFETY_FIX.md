# Post-check parser safety correction

During fixed case review, Q0032 exposed a real value-role defect: the finite
rating dictionary recognizes Positive but not Mixed. For `to Mixed from Positive`
the single-match fallback incorrectly made Positive the new rating.

The safety replay removes such accepted events when both from/to appear but
the existing parser cannot produce a complete pair. It never guesses an unknown
rating and does not expand the dictionary using the check answer. A synthetic
regression test ensures the input prediction is preserved and the bad field is
removed. Original v1 and development-v2 scores remain the registered results.

This is explicitly **post-check**, not independently validated improvement.
It addresses a correctness defect, not a new prompt/parameter search. Existing
rules still miss articles and some grammatical forms; this patch does not claim
complete parsing. No model rerun or forecast training occurs.

The first safety replay (`safe_v3`) looked for from/to anywhere in the clause.
We tightened it in `safe_v4` to a grammatical pair of rating-like words, adding a
synthetic regression test for a rating clause that also mentions a numeric price
pair. Both replays change the same two actual articles: Q0024 and Q0032 each had
a wrong Positive-as-new-rating fact. Q0024's correct overweight-maintain fact is
retained. There was no observed loss of that correct fact; the numeric-pair test
is preventive. Preserve both replays and the first code snapshot.
