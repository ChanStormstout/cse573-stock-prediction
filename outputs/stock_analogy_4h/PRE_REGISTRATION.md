# Historical news/outcome analogy — v1

Registered before new inference or scores. AAPL/AMZN four-hour direction; unchanged
1,607 windows (233 warmup; 765 March–August forward; 252 development; 357 later).
All evaluation periods are exposed exploratory historical backtests.

## Fixed design

- Local pinned Qwen3.5-9B 4-bit, frozen, thinking off, deterministic UP/DOWN
  conditional token probabilities. These are uncalibrated preferences, not self-
  reported probabilities. No fine-tuning, no paid APIs, no browser annotation.
- Use existing complete target-company passage packs P1. Select the most recently
  available article with a target passage; retain its first passage as complete
  sentences up to 120 words (never cut a sentence; oversized first sentence means
  no eligible excerpt). Fixed compact context, not all articles/full-text encoding.
- Same input for current and historical cases: title, evidence excerpt, publication
  age, collection delay, six completed hourly returns/ranges, their age, time of day.
- Retrieve only same-stock original-task windows from strictly earlier months,
  capped at August 2018 for development/later. Target outcomes must end strictly
  before query cutoff. Each historical outcome is the original four-hour close/open
  return, not four hours after publication. All 48 five-minute bars must exist.
- Stateless nonnegative word unigram/bigram hashing cosine (16,384 features), no
  learned vocabulary/IDF. Coarse title/excerpt event categories are deterministic
  retrieval hints, not validated event extraction. Score .75 lexical cosine + .15
  category equality + .10 exp(-price-state distance). Price distance uses fixed
  units: previous six-hour mean return /1 percentage point, std /1pp, intraday
  hour /6.5h, log1p publication age /5. Lexical cosine must be >=.08.
- Three nearest historical cases at most; require same coarse category; collapse
  same-stock reports by online normalized-title Jaccard>=.65 or identical excerpt,
  preserve earliest case per group, exclude query's group, one case per day. No
  outcome-aware retrieval or outcome balancing. Categories/duplicate heuristics
  may be imperfect; inspect predetermined cases and report that limitation.
- P0: current news/price only. P1: similarity-weighted historical label vote with
  symmetric pseudocount 1 per class. P2: P0 plus retrieved historical inputs only.
  P3: same P2 inputs plus those cases' realized four-hour return/direction.
- No usable current excerpt: all four use the exact saved R1 probability. If no
  eligible analogues: P2/P3 exactly P0; P1 exactly R1. No-news coverage is explicitly
  reported; this experiment does not invent events or solve missing news coverage.
- No hyperparameter/threshold/fusion search. Fixed threshold .5; same rule both
  stocks. No selecting later winners. No stochastic seed sweep for deterministic
  frozen inference. Do not call this retraining the LLM.

## Evaluation and gates

Report each stock's forward OOF, development, later BA/MCC/Brier, monthly,
coverage/constant predictions, P3–P0/P1/P2 transitions and 1/5-day paired block
intervals (500 bootstrap resamples, seed573; exploratory, not selection-adjusted).
Include existing F0/F1/F2/R1 on identical keys. Preselect up to 12 case windows
using key hash within stock x phase x has-analogue before inference; examine
evidence/retrieval first, reveal predictions afterward. Supplement error counts
without selecting attractive stories. No automatic fusion or promotion; a future
step would require each stock's June–August mean monthly BA gain>=1pp vs P0 and
R1 with Brier deterioration<=.002, positive BA in >=2/3 months for each stock.

## Integrity

Private source text/prompts/mappings/model binaries stay in work/. Save input,
protocol, code and pinned model hashes; reject mismatched resume and never overwrite
historical runs. Test future cases rejected, query labels not used for retrieval,
train-only libraries, group/day uniqueness, original labels, exact fallback,
probability reconstruction and metric reconstruction. Raw source has timestamp
limitations; modern model pretraining may overlap historical events. Evidence is
retrospective, not independent human quality review or news causal effects.
