# Fixed input cards — before prediction reveal

Twelve slots were selected by key hash within stock × phase × has-analogue, before
v2 inference. Assistant inspection only; not independent human gold. No current or
historical four-hour outcome was displayed during this input review.

| Query key | Input observation | Analogy assessment |
|---|---|---|
| AAPL 2018-10-04 14:30 UTC | Hardware-security allegation plus denials; title taxonomy cannot represent this relation | No retrieval; unknown retained instead of forcing a misleading analogue |
| AAPL 2018-09-04 14:30 UTC | Target-price increase; histories also increase Apple targets, with different brokers and recommendations | Best direct action match in this small panel; different broker, target level and news age still matter |
| AAPL 2018-11-16 15:30 UTC | Recently disclosed prior-quarter institutional sale; first substantial paragraph preserves the filing context | No retrieval; correct to avoid claiming the transaction itself happened now |
| AAPL 2018-11-08 16:30 UTC | Negative experience/review of a new tablet; histories are earlier launches/previews | Product topic matches, but action and sentiment role differ; not equivalent events |
| AAPL 2018-04-25 14:30 UTC | Static/reiterated target, not an explicit increase or decrease | No retrieval under unknown-action rule |
| AAPL 2018-07-10 13:30 UTC | Competitor tablet launch affecting Apple indirectly; histories are Apple's own tablet announcements | Entity-role mismatch remains despite same stock and product topic |
| AMZN 2018-10-02 13:30 UTC | Minimum-wage increase | No training analogue; v1 market-cap false match is removed |
| AMZN 2018-09-24 14:30 UTC | Amazon voice-assistant/speaker changes; history is Apple's competing speaker launch | Same sector but opposite affected-entity roles; weak analogy |
| AMZN 2018-11-12 15:30 UTC | Amazon/Apple retail cooperation, earnings only background | Title signature unknown; no retrieval rather than forcing an earnings match |
| AMZN 2019-01-30 16:30 UTC | Earnings-calendar preview; history is another multi-company earnings preview | Comparable scheduling context, but different companies dominate and event may occur outside target horizon |
| AMZN 2018-03-09 14:30 UTC | No eligible current paragraph | Exact price-model fallback required |
| AMZN 2018-06-13 15:30 UTC | Institution-holdings title, but excerpt is aggregate shareholder statistics | Template similarity, not institution-specific new fact; distinct-looking records can describe the same quarter |

## Implications fixed before scoring

- V2 improves gross topic/action filtering but does not establish accurate event
  identity, actor roles, novelty, or sufficient market-state matching.
- Very low AMZN later coverage prevents a broad claim about the proposed approach.
- The supplied six completed hourly bars may be stale overnight; their age is shown,
  but this is not a rich pre-event market-state representation.
- The LLM must distinguish case differences. A high retrieval score is not a verified
  fact, and historical returns cannot be interpreted as causal event responses.
- No third retrieval version or tuning from development/later errors in this run.
  Keep all four arms and report cases after inference, including unhelpful analogues.
