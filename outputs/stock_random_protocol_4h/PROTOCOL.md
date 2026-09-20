# Random-protocol four-hour study v1

Registered before this lane's first estimator fit. Owner authorization: implement
the staged plan, 2026-09-20. All course periods have been exposed. This is NOT a
new holdout, an exact paper reproduction, or a deployment estimate. Historical
experiments and the blocked external ECNI/relation lanes remain untouched.

## Fixed task and primary comparisons

Use all 1,607 canonical AAPL/AMZN windows (803/804), including warmup windows,
four-hour open-to-close labels, cutoff five minutes before target start. Never
import saved R1/F1/F2 probabilities, fitted PCA, supervised reader checkpoints,
or old prediction-conditioned selections into this experiment.

Five core methods: PAPER (canonical stemmed full-body binary unigrams, min_df=3,
train-only chi2 top 500, L1 LR); PRICE (canonical R1 36 price features, L2 LR);
FULL (canonical 16 legacy price features plus the same binary chi2 body features,
L2 LR); FINBERT and MODERN (canonical 16 legacy prices, title embeddings,
train-only unique-article PCA16, log news count and has-news, L2 LR). These are
new fits of registered structures, not the old chronological checkpoint results.
FINBERT is the fixed primary modern comparator. MODERN is a strong comparator,
never a post-test per-stock substitution. FULL/FINBERT/MODERN have exact PRICE
fallback on no-news rows; PAPER retains its own classifier prediction.

FinBERT and Modern reuse only verified immutable frozen article vectors with
matched special-token-excluded pooling. Modern must use corrected foundation v2.
Vocabulary, chi2, all imputation/scaling, PCA and classifiers fit training rows
only. Pretraining corpus overlap with 2018 remains an explicit limitation.

## Outer/inner protocol

Seeds 573, 574, 575. Ten outer stratified folds per stock; assignments generated
and hashed before fitting. Three inner stratified folds within each outer train.
Methods share splits. No outer metric is used to choose method, parameter,
prompt, threshold or architecture. C candidates .01/.1/1; select descending
inner mean BA, ascending mean Brier, smaller C. Price selects its own C; text
selection includes exact price fallback from that inner training partition.
Threshold .5. liblinear tol=1e-7, max_iter=3000. PCA randomized seed573.
No selection by exposed development/later or best random seed.

Selected inner OOF predictions are not automatically honest meta-training
scores: selecting C on those labels creates selection reuse. Before fitting
a correction, produce additional cross-fitted probabilities with C selected
inside each meta-training partition. Outer validation labels and all their
derived historical-return context must be excluded from every supervised
training/selection step, including prompt exemplars.

Run grouped diagnostics using connected components of shared article IDs,
near-duplicate title links (Jaccard >= .8 with identical numeric/action guards),
same-session stock windows and cross-stock same sessions. Keep entire components
together. Report feasibility and component sizes before choosing fold count;
use 10 folds if each stock/class occurs in >=10 groups, otherwise the largest
feasible count >=3; otherwise NOT_FEASIBLE. Inner splits likewise preserve groups.
Group construction uses no returns. A group is a conservative dependence block,
not an independently validated financial event.

Chronological checks remain separate: monthly Mar-Aug forward OOF and frozen
Jan-Aug fits for Sep-Oct development and Nov-Feb later; fit-time labels must
have matured. Never compare different protocols as matched causal deltas.

## News-context pipeline, frozen before outer scores

Base -> retrieval -> evidence -> bounded residual. Implement independently of
the external ECNI and relation-reader lanes. Full target-aware source passages,
not caches of supervised reader predictions. No full-text coverage expansion.

Analogy: same stock, preceding 30 calendar days, at most 3 distinct past news
groups. Fixed hashing text representation and cosine similarity; remove company
boilerplate, require score >=.1, exclude shared/near-duplicate current news.
Rank only on text, ties by latest available then stable key. Historical targets
must finish strictly before current cutoff. In random CV, history labels must
also belong to the allowed training partition; no held-out targets may be
smuggled into prompts or vote features. The current/old selected-model
probabilities must never enter prompts. Compare fixed current-only LLM,
smoothed similarity vote, and identical LLM plus matured analogies. Record all
case IDs, availability/end times, excluded-label IDs and prompt hashes.
Frozen local Qwen3.5-9B 4bit revision 8b2b98c00a6b4d291155e4890773ca8f769aee53;
no paid API, no model finetuning. Prompt selection is not result-driven.

Fact-change: current canonical news vs earlier same-stock records in 30 days,
at most 3 title-similar candidates. Rating/target-price only. Require exact
evidence spans, issuer, action, actor (unknown permitted), units, current vs
background. Numeric changes computed in code, never guessed. Compare repeat,
further change, correction/denial, different-actor opinion, background, unknown.
No claim of first market disclosure or priced-in status. If actor/entity/unit
cannot be established, do not assert a numeric transition across articles.
Pilot fixed by input hashes, independent of direction labels; assistant labels
are provisional. Complete code even if coverage blocks predictive use. Require
>=30 distinct accepted pairs and >=60 covered windows per stock for a standalone
stock correction; otherwise shared only, or exact base fallback. A failed
quality pilot is reported, not relaxed after seeing prediction results.

P1 metadata, P2 current facts, P3 relative changes share articles and gates;
at most 12 features. Shared correction across stocks, no unrestricted intercept:
logit(p)=logit(base)+g*alpha*tanh(w'z). Alpha 0/.1/.25/.5; L2 C .01/.1.
Scalers and w fit meta-training only; select shared settings by average stock
BA subject to each stock Brier worsening <=.002, tie smaller alpha then C.
If no nonzero option passes guardrail, alpha0. Zero gate returns exact original
floating-point probability. Run both encoder bases without test-picked switching.
Combined analogy/fact correction is eligible only if inner matched deltas for
each mechanism are positive on both stocks and Brier guardrails hold. All
decisions recorded before outer predictions. Independent human quality review
is not claimed.

## Execution and reporting boundary

Stage A prepares splits, source binding and fit-free leakage tests. Stage B
executes five baselines; selected weights and outer predictions sealed locally,
no public scores until registered mechanisms have completed or documented a
predefined feasibility stop. Stage C prepares fold-safe analogy and outcome-blind
fact pilot; estimate exact inference count/runtime before batch execution, reuse
only byte-identical model/prompt caches. Stage D executes correction and grouped/
chronological checks. Stage E verifies before reporting and slides updates.
More expensive optional RF/SVM/KNN/AdaBoost/Chronos remain extensions, not a
test-score-driven replacement of the main method. Any extension gets a dated
finite config before its own execution; do not silently widen this core grid.

Store every fit/selection, folds and training keys, sources/code/model hashes,
runtime, checkpoints and reload errors. Independent verifier never fits models.
Test future-input refusal, held-out outcome refusal, source mismatch, exact
fallback, saved model replay and disjoint/exhaustive folds. Corruptions must fail.
Report BA/MCC/Brier/accuracy, seed mean/std, month/stock, news coverage, constants,
changed/repaired/introduced counts and paired 1-day/5-day uncertainty. No claim
of fresh significance after repeated exploration. No fabricated scores.
