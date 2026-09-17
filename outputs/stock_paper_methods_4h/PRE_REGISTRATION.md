# Four-hour paper-method follow-up: pre-registration

Saved before the new Phase 1 results were computed. All September 2018 and later periods are already exposed exploratory replay.

## Fixed task and time protocol

- Stocks: AAPL and AMZN.
- Target: direction from the target four-hour interval open to its final five-minute close.
- Fixed universe: the existing 1,607 windows; cutoff is five minutes before target start.
- Training selection: chronological March--August forward predictions only. Development (September--October) and later (November onward) are reported separately and never averaged for selection.
- Every scaler, text vocabulary, PCA, calibrator, classifier and cluster uses only information available before its evaluation cutoff. Article embeddings remain frozen ProsusAI/finbert revision `4556d13015211d73dccd3fdd39d39232506f3e43`.
- C candidates are exactly `0.01, 0.1, 1`. No stock-specific winner is chosen after reading development/later results.

## Phase 1A: F1/F2 calibration

Calibrate each branch independently from its own past forward-OOF probabilities.

- `identity`: unchanged probability.
- `temperature`: positive scalar temperature.
- `platt_shrunk`: positive slope and intercept fitted by log loss with a fixed penalty of 10 toward identity `(slope=1, intercept=0)`.
- March has no earlier OOF calibration ledger and therefore remains identity. April--August use only earlier OOF months. The frozen September-onward calibrator uses March--August OOF.
- Selection minimizes mean stock-month Brier over April--August, then maximizes BA. A candidate is eligible only when macro Brier and each stock's mean Brier are no worse than identity by more than 0.002.
- BA, MCC, Brier, ROC-AUC, class recalls and predicted-up rate are reported. Brier-only improvement is called calibration improvement, not directional improvement.

## Phase 1B: recent price inside matched text models

- `J0`: old six-hour price features plus the existing full-body bag-of-words representation.
- `J1`: the same body representation plus old price and cutoff-safe 5/15/30/60-minute features.
- `J2`: old price plus frozen FinBERT article mean, training-only PCA16 and the existing count/coverage fields.
- `J3`: the same FinBERT representation plus the recent price features.
- Transformations are refitted inside every chronological fold. For no-original-news rows, all J methods strictly use the already sealed R1 probability so that missing-news policy is matched.
- C selection uses earlier forward months only; the final C uses all March--August OOF.

## Phase 1C: full-set dissemination/event aggregation

Use every article key in the original F2 window, rather than the at-most-six direct-LLM pack.

- `N0`: equal article mean.
- `N0M`: equal article mean plus metadata.
- `N1`: cluster articles first, average within event, then give events equal weight.
- `N1M`: event mean plus exactly the same metadata columns as N0M.
- All N methods use old price features, frozen article embeddings, training-only PCA16 and the same C candidates. No-news rows strictly use R1.
- Metadata columns: log article count, log event-cluster count, log distinct source count, log age of newest received article, log median received age, and whether every article belongs to a duplicated cluster.

The fixed online clustering rule processes articles by available time. Exact/normalized hashes merge. Otherwise titles merge only at token Jaccard >= 0.80 and when directional-action tokens, numeric tokens and quarter/year tokens do not conflict. Clustering sees only articles already available at the prediction cutoff. A fixed representative is the newest available article; no outcome label enters clustering.

## Advancement rule

A mechanism advances to a more complex follow-up only if, against its information-matched baseline over at least three complete forward months:

- at least two months have positive BA difference;
- macro mean BA improves by at least 1 percentage point;
- neither stock loses more than 1 percentage point in mean BA;
- neither stock's mean Brier worsens by more than 0.002.

Calibration may be retained for probability quality without satisfying the directional rule. If no Phase 1 mechanism advances, FinModernBERT, Chronos-2, attention, TabPFN and graph models are not run in this experiment.

## Reporting boundary

F0 remains the price+title baseline; F1 remains the current full-text method; F2 remains the modern semantic baseline. New results are exploratory historical replay. Independent event-extraction review remains incomplete.

## Implementation clarification before fitting

Calibration is fit only on each stock's past original-news branch probabilities; no-news R1 outputs remain exactly unchanged. Calibration scheme choice is global across stocks and uses earlier calibrated OOF months; September-onward uses a frozen training-only choice and calibrator. Positive monotone maps preserve within-slice AUC; pooled cross-month AUC can change when monthly maps differ, and is not by itself evidence of better ranking.

To reproduce the existing J0/J2 controls, C selection uses each branch's raw (before no-news fallback) past monthly BA/Brier, as in the existing integrated run. This identical selection rule applies to all joint/aggregation candidates; complete-system outcomes and advancement include the matched R1 fallback. The first two training months remain warmup with no pretended OOF predictions; all 1,607 windows are retained and the common evaluable set is 1,374.

`has_usable_text` means at least one nonempty title actually represented by the frozen article cache. `has_qualified_event` is the existing A1 gate where available, unknown in warmup; it does not assert independent extraction acceptance. Clusters are recomputed from the complete eligible window, in arrival order, requiring all-pairs title similarity and matching direction/numeric/period guards. No future member is used. This is a conservative near-republication proxy, not independently validated semantic event identity.

Final development/later models are frozen at the original August boundary for matched comparison. March--August selectors are replayed chronologically; this is not monthly retraining on exposed development/later labels.
