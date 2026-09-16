# Frozen LLM direct four-hour prediction — registered before inference

Scope: original 1,607 AAPL/AMZN four-hour samples; prepare all, predict every
609 non-training sample (252 Sep–Oct development,357 Nov–Feb exposed evaluation).
No new labels, horizon, exclusion or extraction-quality filter. This is a frozen
LLM forecast experiment, not extraction and not weight training. Existing periods
and model choices are exposed; modern pretrained model may remember old events.
A prompt cannot eliminate pretraining contamination.

## Fixed comparisons

Qwen3.5-9B 4bit at existing pinned revision, local MLX, nonthinking, greedy,
seed573,192 output tokens, max input+output6144. Three variants: price, news,
joint. No examples in primary comparisons. No new prompt based on market scores.
Only synthetic smoke may fix an interface bug; document any such amendment.
All calls keep full inputs/output and cost. Invalid forecast -> p=.5 (UP on tie),
flag failure; evaluate all windows plus valid-only diagnostics. No silent retries.
Evidence citation failure is separate from probability validity. No probability
calibration claimed. Prompt template and all keys exclude actual target prices,
returns, labels and split. Explicit task = target interval close above its open;
future opening price is unknown; cutoff is start minus5min.

Prices: six most recent fully completed regular-session hourly bars, exact
log-return*100 and range/open*100 used by legacy model, oldest first, timestamped;
mean/std in same units, history age and interval session time. Rebuild from5min
bars and calendar and check parity. Do not invent volume or overnight bars.
News: exact legacy accepted record set in (cutoff−4h,cutoff]. Sort newest
availability then key; max6 articles. Title first240 characters; source excerpts
at most600 characters = initial200 plus up to400 around first literal target
company mention. Exact spans, no LLM paraphrase; excerpt may omit key context.
Keep publication/availability times and ages, omitted counts/characters. No
semantic dedup beyond the original record set. Same pack used for news/joint.
No unseen market/index data. Other baseline access differences clearly labelled.

Matched LR controls: price, packed-news, joint with same selected news snippets;
L2, C in .01/.1/1, June/July/Aug past-only folds, mean BA then Brier then smaller C.
Fit transformations inside each fold; freeze Jan–Aug fit for Sep onward. News
numeric ages/counts and text supplied identically; LR can ignore structure and
still is not an identical architecture. Legacy title/body/FinBERT/integrated are
context references, not identical-input baselines. No choosing winner on test.

## Reporting

Per-stock, split and month BA/MCC/Brier, direction proportions, failure and
coverage. Paired whole trading-day resampling, 1-/5-day blocks (500 replicates),
for joint vs matched LR joint and old title baseline. Descriptive intervals only.
Fixed cases: joint vs matched joint, stock × dev/eval × both correct/both wrong/
LLM-only correct/LR-only correct; SHA(key) first per nonempty cell; preserve
unavailable cells. Inspect supplied inputs, brief rationale and errors. Reasons
are not verified internal reasoning and correct prediction is not factual proof.

## Optional next steps

Four training-only examples and longer historical context remain separate future
contrasts after primary three finish. No fine-tuning this run. Do not automatically
select high-scoring prompt or replace the existing project system.
