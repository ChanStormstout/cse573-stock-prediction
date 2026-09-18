# V2 amendment: input-only retrieval inspection, before any metric computation

The twelve predetermined v1 input cards reveal unsupported analogies: a wage
announcement retrieved market-cap stories; an Apple/Microsoft comparison retrieved
insider selling; a short source header supplied no substantive evidence. Abort v1,
preserve inputs, its generated token scores and code snapshot privately, and record
cost without computing v1 prediction metrics. No prediction/label-conditioned cases
were used to decide this amendment. V1 is an aborted engineering run, not a hidden
negative evaluated result.

For v2 keep all four methods, original windows, model, time rules and statistical
protocol. Change only input/retrieval quality before new inference:

1. First complete target passage with at least 25 words, capped at 120 whole words
   by sentence; header-only articles have no eligible excerpt.
2. Determine conservative event/action signature from TITLE ONLY: target-price
   raise/lower, rating upgrade/downgrade, earnings beat/miss/preview, legal, labor,
   acquisition, holdings increase/decrease, product change. Unknown is ineligible
   for historical retrieval (P0 may still use its excerpt). These are heuristics,
   not formal event extraction or evidence that the title action targets the stock.
3. Historical signature must exactly match; omit unknown subtypes. Remove company,
   ticker and generic market/share terms and English stopwords from lexical hashing.
4. Same maximum3, cosine>=.08 and fixed score; no threshold search. Empty analogue
   set returns P2/P3 exactly to P0 and P1 to R1. Full-window coverage is reported.

This is a deliberately conservative feasibility experiment. It does not implement
an LLM retriever, formal entity/actor matching, multi-news reasoning or a full-text
semantic retriever. Remaining mixed-target and stale background errors are audited
as limitations, not silently relabeled. Stop after v2, do not keep tuning retrieval
based on development/later predictions. Fixed cases are selected before v2 inference.
