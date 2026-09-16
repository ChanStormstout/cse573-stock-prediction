# Input-only audit during execution

Before joining LLM forecasts to market outcomes, selected the lowest SHA256 key
in each training-only stock × eligible-news-count stratum (0,1–6,>6). Six windows
were read. This is a deterministic input audit, not a performance case study;
original input packs and prompt remain unchanged. Full snippets stay local in
work/stock-data/direct_4h/input_audit.json.

- AAPL 2018-08-21 open: no news, latest historical bar ~17.92h old.
- AMZN 2018-03-09 open: no news, same overnight age. No-news windows preserved.
- AAPL 2018-07-10 open: five articles, four primarily older-quarter institutional
  holdings plus a Microsoft competitive product story delayed ~18.9h. These are
  not five fresh directional catalysts. Publication and delay metadata help but
  do not establish the true first disclosure time.
- AAPL 2018-04-06 open: ten eligible/six included, overlapping promotional alerts,
  holdings reports and ~37h-old supply-chain/capital-return commentary. One
  multi-company title mentions Apple while much selected opening text concerns
  Merck. First target mention may be only the repeated headline; this simple
  input rule does not guarantee the core company paragraph is present.
- AMZN 2018-07-16 10:30: one article about prior gains and Prime Day, published
  ~14.4h before cutoff. Latest complete historical bar ~66.92h old after weekend
  and the one-hour completion rule; not six consecutive wall-clock hours.
- AMZN 2018-02-02 11:30: eight eligible/six included. Contains earnings evidence,
  but also promotional text, broad-market compilation and another company's
  nearby target-price sentence. This is a target-attribution stress case.

Literal character spans preserve provenance but may cut words/sentences and
separate a number from its company. The first200+up-to400 budget can omit useful
context. This is a limitation of this fixed primary pack, not proof of inherent
LLM inability. No outcome-selected article replacement or retrospective source
repair was made. A later input experiment should compare whole-sentence/company
paragraph selection and larger coverage using matched LR inputs and fixed rules.

Evaluation coverage, before reading model scores:
AAPL development:124/126 have news,64 capped; later178/178 have news,151 capped.
AMZN development:73/126 have news,0 capped; later84/179 have news,0 capped.
AAPL later included1,031 of2,465 article occurrences; AMZN included all143.
Occurrences are per-window counts, not unique independent articles.

## Completed-hour history freshness (all609 evaluation windows)

Raw-feature inspection confirms AAPL101/304 and AMZN101/305 windows have their
latest hourly bar ending55min before cutoff. The other203/304 and204/305 end
more than16h before cutoff (up to90.92h with non-trading gaps). At09:25 and10:25
there is no completed same-day09:30–10:30 bar yet; only at11:25 can that bar be
used. This follows the inherited complete-hour design and is not an alignment
or future-leakage error. Available partial-hour five-minute history is omitted.
A useful NEXT separate information experiment could add the most recent fully
available5min bars/partial-hour return to BOTH LR and LLM, keeping target prices
unknown and the original four-hour labels/windows. It is not implemented in this
primary comparison, and the freshness counts do not prove a causal contribution
to prediction error.
