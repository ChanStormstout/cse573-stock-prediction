# CSE 573: Four-hour stock direction from prices and news

## Abstract

We compare price and news representations for AAPL and AMZN using 1,607 fixed four-hour windows. A chronological training protocol controls vocabulary, PCA, classifier selection and probability calibration. Full-text sparse features provide a useful reference improvement over a title baseline, while frozen FinBERT, recent-price features, news aggregation and event adaptation expose different tradeoffs between direction accuracy and probability quality. Event-task adaptation improves provisional extraction F1 to 79.17%, but does not establish incremental four-hour prediction value. Positive temperature scaling reduces overconfidence while preserving decisions. All evaluation periods have been exposed during exploration, so results are historical replay rather than independent deployment evidence.

## 1. Task and data

For each stock, predict whether the close at the end of a four-hour target exceeds its opening price. The information cutoff is five minutes BEFORE target start; the future actual open is never an input. News enters only after its recorded availability. Completed price bars and the original target labels are preserved.

The fixed universe contains 1,607 windows: 998 original training windows, 252 September–October development windows and 357 November-onward later windows. Of training, 233 January–February windows initialize the model; 765 March–August windows receive chronological forward predictions. Overlapping windows and repeated news reduce effective independence. AAPL dominates the supplied news corpus, and AMZN has much weaker news/event coverage.

## 2. Methods

F0 combines historical prices with title TF-IDF and regularized logistic regression. It is the project's operational classical baseline, not a claim of full replication of a published system. F1 replaces titles with full-body binary word features, with training-only chi-square selection. F2 uses frozen FinBERT article representations, training-only PCA16, article averaging and the same classifier family. With no original news, all these systems use the identical recent-price R1 fallback.

R1 adds cutoff-safe 5/15/30/60-minute returns, ranges, volatility and missingness. Joint experiments insert these into the text classifier rather than averaging two predictions. News aggregation compares article means against conservative near-republication groups, with matched count/source/age metadata. Positive temperature scales logits without changing the 0.5 decision; its scalar is fitted only to earlier OOF probabilities of that exact branch.

The event adapter predicts target-specific evidence, rating/target-price type and action. Numbers are copied and checked by code. Frozen heads, top-two-layer adaptation and rank-4 q/v LoRA were compared with three seeds. A1 was selected on extraction-training forward folds; labels remain dual-model provisional labels with assistant adjudication, not independently reviewed human gold.

## 3. Evaluation

We report balanced accuracy (mean up/down recall), MCC, Brier probability error, AUC, class recalls, monthly and coverage results. Transformations and hyperparameters use past data. Development and later results remain separate. Final models freeze at August; no later-label tuning or per-stock hindsight winner assignment is permitted.

The resource gate requires at least three forward months, positive macro BA differences in at least two, mean BA gain of one percentage point, no stock losing more than one point, and Brier worsening no more than 0.002 per stock and overall. This controls exploration, not statistical significance. Day/block paired intervals reflect serial dependence imperfectly and do not eliminate repeated-exploration bias.

## 4. Main results

| 方法 | AAPL开发 BA / Brier | AAPL后续 | AMZN开发 | AMZN后续 |
|---|---|---|---|---|
| F0 title baseline | 50.46% / 0.2601 | 51.89% / 0.2694 | 48.33% / 0.2710 | 49.79% / 0.2677 |
| R1 recent price | 56.21% / 0.2441 | 53.87% / 0.2558 | 51.11% / 0.2719 | 50.85% / 0.2722 |
| F1 full text | 57.94% / 0.2500 | 51.74% / 0.2587 | 55.66% / 0.2481 | 54.36% / 0.2583 |
| F2 frozen FinBERT | 54.13% / 0.2803 | 56.71% / 0.2856 | 46.29% / 0.2977 | 54.27% / 0.2617 |
| F2 + temperature | 54.13% / 0.2473 | 56.71% / 0.2503 | 46.29% / 0.2560 | 54.27% / 0.2462 |
| R1 + FinBERT | 55.86% / 0.2492 | 56.71% / 0.2589 | 52.69% / 0.3039 | 53.79% / 0.2669 |
| R1 + FinBERT + temperature | 55.86% / 0.2478 | 56.71% / 0.2494 | 52.69% / 0.2560 | 53.79% / 0.2462 |
| Article mean + metadata | 52.54% / 0.2764 | 56.74% / 0.2831 | 47.59% / 0.3029 | 54.74% / 0.2639 |
| Event mean + metadata | 51.80% / 0.2937 | 56.16% / 0.2940 | 50.46% / 0.2623 | 57.32% / 0.2483 |

F1 is the most defensible simple full-text main method across the original four stock/period cells, but it does not improve every cell versus F0. F2 has better AAPL later BA but poor raw probability quality. The calibrated joint model preserves direction while reducing Brier; it still fails the training-period AMZN guardrail. Later-period improvements do not override that decision.

## 5. Event understanding is distinct from prediction

On 122 provisional April check articles (23 positive facts), exact fact F1 was 16.29% for current rules, 32.43% for Qwen1.7B QLoRA, 38.30% for staged Qwen9B, and 79.17% for A1. AAPL/AMZN A1 fact F1 was 75.68%/90.91%; AMZN has only six check facts. The coverage-limited oracle mapped to only 15 stock-training OOF event windows. Shared, separate and partially shared downstream corrections did not beat zero correction; the final probabilities therefore remained exactly F1. Extraction improvement is demonstrated against provisional labels, not formal independent acceptance or a market forecast improvement.

## 6. Case study and ablation insight

AMZN event aggregation reaches 57.32% later BA versus the matched article+metadata 54.74%. However, swapping only current aggregation vectors under fixed models does not change a single AMZN direction. Event-trained versus article-trained models also selected different regularization strengths, C=0.01 versus C=1. Thus training and selection changes, rather than direct deletion of duplicate current news, explain the computational path to the changed decisions. A subsequent fixed-C experiment resolved the later AMZN direction contrast: article and event models at C=0.01 make exactly the same decisions (57.32% BA), whereas article C=1 reaches54.74%. The observed direction gain therefore does not require event aggregation; stronger regularization alone reproduces it in this period. This does not establish a stable cross-period accuracy gain.

Long-term opinion and competition news can become a corrected bearish call even when two distinct articles remain two groups. Conversely earnings-preview and partnership articles can be changed from a correct bullish call to an incorrect bearish one. Evidence content, model contribution and realized market direction must be discussed separately.

## 7. Limitations and conclusion

All periods were exposed during exploration; modern pretrained models applied to 2018 data may carry later pretraining knowledge. Availability timestamps are source metadata, not externally authenticated first disclosure. Near-republication grouping uses titles and cannot guarantee full-body event identity. Small, correlated samples and scarce AMZN events constrain statistical power. No trading-profit claim follows from these direction metrics.

Our contribution is a reproducible, controlled pipeline and component analysis: richer text sometimes helps; better event understanding does not necessarily help four-hour prediction; calibration improves confidence without inventing direction information; aggregation gains require retraining-aware attribution. We retain F0, F1 and F2 as the principal comparison, with the adapter and finite combinations as transparent ablations. Further models were not automatically launched when the training gate failed.

## Reproduction and presentation

- Latest experiment: [report](../outputs/stock_combination_4h/v1/REPORT.md), [protocol](../outputs/stock_combination_4h/PRE_REGISTRATION.md).
- Full core comparisons: [Phase 1](../outputs/stock_paper_methods_4h/v1/REPORT.md), [event adapter](../outputs/stock_finbert_event_adapter_4h/v1/REPORT.md).
- Offline replay: [demo](../outputs/stock_combination_4h/v1/demo.html). It displays saved historical predictions, not new live forecasts.
- Suggested demo: choose AAPL later; compare F2 and temperature (same direction, different confidence); choose AMZN later and a no-news window (strict fallback); reveal label; show aggregation success and failure in the report.
- Raw data and weights remain local; public source and model manifests alone do not permit full retraining without the course data.

This is the report main draft. Group names/contributions and the instructor's final page/template/submission requirements must be filled from the authoritative course instructions; no submission has been made.


## Additional finite foundation-model experiments

A separately authorized preregistered round ran 208 new LR fits and frozen inference with Fin-ModernBERT and Chronos-2. Modern used exactly the same titles, mean pooling and training-only PCA16 as F2. It reached AAPL development/later BA45.51%/54.52% and AMZN54.82%/56.03%; its train-month mean BA change was−0.06pp, with AAPL−3.56pp and AMZN+3.44pp. It did not pass the joint gate.

Chronos-2 forecast open/close from512 completed scheduled five-minute trading bars, with target indices aligned to the unchanged four-hour label. Median differences are not probabilities; a past-only LR maps three forecast features to probabilities. Its AAPL development/later BA was50.00%/50.55%, AMZN51.67%/47.01%. The raw median direction later was48.12%/49.31%. A same-history-length LR was evaluated separately. Chronos did not pass the gate, and neither branch received additional fusion or fine-tuning.

These are retrospective component comparisons: Modern's listed financial pretraining includes FNSPID, and historical pretraining overlap cannot be excluded for either foundation model. They do not justify selecting different models for different stocks based on exposed outcomes. See the [complete new report](../outputs/stock_foundation_4h/v1/REPORT.md), [case analysis](../outputs/stock_foundation_4h/v1/CASE_NOTES.md) and [new replay](../outputs/stock_foundation_4h/v1/demo.html).
