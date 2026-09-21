# Grouping financial news: presenter notes

16 main slides and 3 appendix slides. Suggested speaking time: 10–12 minutes.

## Slide 1: Grouping financial news for stock prediction

Opening: Our implemented method represents financial news with frozen FinBERT, groups near-duplicate titles, retains coverage/arrival metadata, and trains a small price-plus-news classifier. This deck replaces the former relative-fact-change proposal as the main presentation. The two methods are distinct. Sources: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py ; https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md

## Slide 2: The prediction task

Canonical task: 1,607 total windows including warmup; 765 training forward predictions, 252 development, 357 later. Cutoff is five minutes before the four-hour target. Only complete available price history and admitted news are inputs. All evaluation periods are exposed exploratory historical backtests. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_combination_4h/v1/REPORT.md

## Slide 3: Classical baselines and the modern comparison

Baseline roles: constant-direction BA is analytically50% when both labels are present, not a new random simulation. R1 is saved price-only LR. Primary classical news baseline F0 is deterministic TF-IDF title word features plus16price features and LR. F1 is binary full-text word features, train-only chi-square selection, price features and LR. FinBERT is an additional modern representation comparator and no-group ablation reference. All displayed historical scores are from the same four-hour canonical report. Do not substitute random-split SVM/RF results into this chronological table. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md ; https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 4: Modern comparison: FinBERT with article averaging

The semantic reference uses frozen ProsusAI/FinBERT title embeddings, train-article PCA16, 16 old price features and basic news metadata in a regularized logistic classifier. No admitted news falls back to saved R1. The matched grouping control later has the same six metadata fields. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py ; https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md . FinBERT background: https://arxiv.org/abs/1908.10063

## Slide 5: Why repeated reports can dominate the input

Illustrative teaching example, not six observed source articles. Five compatible near-duplicate Apple target-price-cut reports and one distinct product report. The example is designed to explain article weighting, not predict a market return. Equal group weighting is a modeling hypothesis, not verified event importance.

## Slide 6: Our method: group content and retain reporting patterns

This is the implemented N1M structure, described without internal IDs. Accepted input is titles, not whole-body event extraction. FinBERT title vectors are averaged within complete-link compatible near-title groups, then across groups; PCA16 fits training articles only. Six metadata columns and 16 OLD price columns enter regularized LR. No news returns R1. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 7: How we form a group

Exact implementation: title tokens are lowercase alphanumeric sets; Jaccard >= 0.8; guards require equal raise/lower/maintain/initiate/deny/accuse patterns, numerical tokens and quarter expressions. Each new title must be compatible with every group member. Process admitted articles available by the cutoff. This is near-report grouping, not validated event coreference. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 8: Two-step averaging changes the balance

Illustration follows the six-report example. Every title first has a vector. Average the five cut-report vectors into c1 and the single product vector into c2, then compute (c1+c2)/2. Before PCA this gives each group half the aggregate weight and each of the five reports 1/10. This does not mean the classifier assigns equal causal importance to the two stories.

## Slide 9: Metadata preserves the reporting pattern

N0M and N1M use exactly these six fields. Actual inputs apply log1p to the five count/age values; only_duplicates is binary and means every group contains more than one article. Illustrative six-article arrival ages 10,20,30,50,60,70 minutes have newest age10 and median40; five duplicates plus one singleton gives false. Availability is the source-defined arrival time, not certified first market disclosure. Sources count recognizable sites, not independent evidence. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 10: How news and prices reach the forecast

N1M uses 16 OLD price variables (six return/range pairs, history age, mean return, return std, NY hour), PCA16 of group means, six metadata columns. Price normalization and semantic/metadata scaling fit training only. LogisticRegression liblinear C grid .01,.1,1, max_iter3000 tol1e-7 seed573; no news replaces the joint probability with R1. Do not conflate the predictor with generative LLM or state that recent extra R1 columns are in the N1M joint training. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 11: Training uses earlier months to predict later months

March–August monthly issued C choices use previous CV records; March uses default .1. Final training before September chooses C from preceding monthly CV; final evaluation September onward. Code asserts training label end before evaluation cutoff. Inner raw classifier metrics select C and R1 fallback applies to issued no-news rows. This distinction is recorded rather than implying a joint fallback selection objective. Sources: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py ; https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md

## Slide 12: Matched comparisons isolate the mechanism

Modern reference F2, N0M, N1, N1M all use the frozen title representation and OLD price features, canonical windows, training selection framework and R1 no-news fallback. Basic news metadata differs from six-field expanded metadata. Main grouping attribution comparison is N1M vs N0M; different selected C values remain a source of combined representation/regularization effects. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py ; https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md

## Slide 13: Classical baselines, modern reference and our method

All measured cells are copied from stock_paper_methods_4h/v1/metrics.csv, methods R1,F0,F1,F2,N1M, same canonical windows and chronological periods. The constant-direction row is an analytic BA reference, not a newly run estimator. Both classes occur in each evaluation cell. Main classical baseline is F0 title word features + prices + LR; FinBERT is a modern comparator. Development September–October2018, later November2018–February2019. Existing evaluated periods are exposed. https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md

## Slide 14: What the result does and does not explain

N1M vs N0M mean March–August monthly BA AAPL -.64pp AMZN+.12pp macro-.26pp, failed original promotion. Later AMZN repaired8 introduced4. Article training C1 vs group training C.01. Fixed-model replacement of article/group current vectors changed no directions; therefore no proof duplicate removal caused the gain. BA paired one-day interval [-.08,5.48]pp and five-day[-.24,6.14]pp cross0. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md ; https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_combination_4h/v1/REPORT.md

## Slide 15: The course contribution is how we organize information

Course alignment: semantic representation, similarity-based document grouping, source/time metadata mining and classical supervised prediction. This method does not establish an entity-event knowledge graph and does not require LLM generation or extraction labels. No claim that the instructor has endorsed this particular mechanism. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 16: Current method and next decision

This presentation is a mechanism explanation of the existing implemented N1M system. It does not run new training or authorize restart of paused experiments. The full current-vs-prior fact-change reader is optional future work rather than the claimed implemented mechanism. Group membership is heuristic, and entity review is not claimed. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_combination_4h/v1/REPORT.md

## Slide 17: Appendix: exact representation and model

Technical detail from run.py. Title Jaccard .8, complete-link compatibility, group and article means. PCA learned on distinct training articles, no groups receive a learned attention weight. log1p metadata features. 16 OLD price features. Fixed regularization candidate grid .01,.1,1. No-news fallback is saved R1 rather than zeroed joint output. https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py

## Slide 18: Appendix: a repaired error and an introduced error

Saved case descriptions are from combination report and do not disclose raw full article text. AMZN Jan8 before .606 after .499 actual down; Jan31 before .621 after .493 actual up. Both have distinct reports; Jan8 has2articles2groups and Jan31 two distinct articles. These are rounded saved probabilities, not new inference. They illustrate retraining changes, not causal news impact. https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_combination_4h/v1/REPORT.md

## Slide 19: Appendix: sources and scope

FinBERT: https://arxiv.org/abs/1908.10063 . Implementation: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/run.py . Results: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_paper_methods_4h/v1/REPORT.md . Attribution: https://github.com/ChanStormstout/cse573-stock-prediction/blob/06e98a766a566e8692ec41d2231f01cc611d3061/outputs/stock_combination_4h/v1/REPORT.md . Internal IDs: reference F2, matched N0M, grouping-only N1, complete N1M. No new model fit or verification resumes during this deck edit. All historical slide versions preserved.
