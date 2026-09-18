# Reaction features v1 preregistration

This file is written before any predictive score is computed.  The main task
remains the original AAPL/AMZN four-hour direction problem.  Existing
development and later periods are exposed exploratory history.

## Article-target pairs

The source is the complete raw news corpus, not only the 4,930 article IDs
already accepted by the window builder.  The unit is
`(article_key,target_symbol)`; a dual-target article may create two units.
Availability is `max(valid published_utc, crawled_utc)`, English text is
required, and the article must contain sufficient text.  Evidence tiers are:

* Tier 0: target ticker or legal company name;
* Tier 1: target company name in the title with conservative disambiguation;
* Tier 2: target company name in a body sentence plus adjacent context.

Product-only aliases are excluded.  Pair duplicates use normalized full text,
then exact text, then a normalized-title fallback.  The earliest available
record is canonical; counts of suppressed records and sources are retained.

The primary reaction is same-session, regular-session realized return from the
first five-minute bar whose start is at or after availability, for 30/60/120/
240 minutes.  An incomplete or non-contiguous horizon is unknown; no overnight
return is used as a same-session reaction.

## Feasibility gate

For each stock, the Jan--Aug 2018 corpus must contain at least 800 unique
target-article groups with valid 60-minute and 240-minute reactions, at least
100 session dates, and at least six months with 50 or more groups.  The gate
must pass for both stocks.  If it fails, this run stops before predictive
training and only reports the audit.

## If the gate passes

The frozen ProsusAI/FinBERT encoder is used only as an article representation;
no encoder fine-tuning is performed.  AR0 is price context, AR1 is a lexical
context model, and AR2 is PCA(16) over frozen target-context embeddings.  For
each evaluation article, training articles must have a mature reaction label
before the evaluation article's availability.  January--February are warmup,
March--May are inner selection, and June--August are outer evaluation.  Article
weights are normalized by the number of articles per symbol-day.  A horizon is
promoted only if its predeclared outer rule passes; otherwise no W0--W3
downstream experiment is run.

No new LLM or FinBERT fine-tuning, GNN, RL, Chronos or external paid data is
part of this run.
