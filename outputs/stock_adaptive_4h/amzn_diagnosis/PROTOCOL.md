# AMZN case study protocol — registered before new case selection/read

Scope: original FOUR-hour task, unchanged 1607 rows. Main diagnostic population is all 305 AMZN frozen predictions (Sep–Oct development126; Nov–Feb exposed later179). These are exposed exploratory periods, not an independent test. Do not retrain or choose a new model on these outcomes.

Questions:
1. Is AMZN always below50%, or do specific branches exceed it? Recompute matched BA/MCC/Brier/class recall and monthly/news-presence slices.
2. Where does usable branch information disappear: old body vs title; old body vs old integrated; title vs latest zero-bias gate?
3. Are failures associated with missing news, stale/background items, wrong target, conflicting implications, duplicate reports, or price/base-model changes? Association and illustrative cases are not causal proof.
4. Why is the AMZN news correction disabled? Trace the saved training-only C/Brier decisions; do not force-enable a candidate using later scores.

Population calculations first:
- Align all methods by the exact sample key; preserve every window including no-news and small returns.
- Report all dates and both exposed periods, up/down recalls, predicted-up fraction and constant baseline. November–February has only60 common trading days, overlapping four-hour labels; not179 independent market events.
- For every comparison count correct-to-wrong and wrong-to-correct on exactly the same rows. Decompose total BA differences additively by news presence and true class using full-population class denominators. Do not take an unweighted average of subgroup BAs.
- Saved-candidate CV means and selected C/zero fallback are explanatory engineering records, not evidence the guardrail improves accuracy.

Case panel (up to24 unique windows):
- Three matched comparisons: old body minus title; old body minus old integrated; title minus zero-bias gate.
- For each, development/later × four outcomes (both correct, both wrong, first wins, second wins), choose one deterministically using SHA256(seed573 + key + comparison), preferring previously unused dates and windows. Explicitly record empty strata or repeats. Never choose by how persuasive the article sounds.
- Inspect a second representation of selected cases with outcomes/probabilities hidden first. Record all available titles; read up to two distinct article bodies per news-bearing window selected by latest availability then key, suppress exact normalized-title duplicates. Save character limits; state excerpt rather than full-article review when truncated. No-news windows remain cases.
- Only after content notes are fixed, reveal model probabilities and target return to connect input issues with changed decisions. Existing familiarity with this dataset means this is not truly blinded independent annotation.
- Labels: target relation (direct/comparison/incidental/unclear), temporal role (new disclosure/background/mixed/unknown), polarity/conflict, duplicates, what the predictor can/cannot observe, interpretation confidence. Allow unknown; positive factual news need not imply four-hour positive returns.
- Optional saved-model perturbation only: remove/swap text while preserving price/row to inspect which branch depends on news. If run, label out-of-distribution sensitivity, never a causal simulation or new performance candidate.

Deliver: reproducible analysis script, complete population tables, deterministic panel, pre-outcome content notes, case interpretations, mechanism status matrix, recent-primary-paper suitability matrix, final Chinese report and project log. No claims of all-case independent review; no changes to labels/old models/selection thresholds.
