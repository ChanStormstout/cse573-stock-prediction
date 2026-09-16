# Integrated pipeline v1 — registered before execution

Objective: complete a coordinated price/title/full-body/FinBERT prediction pipeline against a classical news baseline. Exploratory historical replay: all periods have previously influenced design. No claim of untouched final test.

Task: original 3233 one-hour directional windows, all original news IDs, no deletions. Source: verified E09 train/development and E12 historical test inputs. Jan–Aug2018 training; monthly forward ledger March–August; outer diagnostic June–August; frozen Sept2018–Feb2019 replay, separately September–October and November onward.

Branches (all L2 logistic regression, C=.01/.1/1, seed573):
- price: original16 numeric features.
- title: price plus deterministic TF-IDF 500 uni/bigram title terms, min_df2 (primary classical baseline).
- body: price plus binary stemmed body unigrams, min_df3, training-only chi-square top500 with lexical tie-breaking. Uses existing audited E09/E12 body preprocessing. Not a complete paper reproduction.
- semantic: price, news count/availability and frozen FinBERT title article vectors projected to16 dimensions by PCA fitted on unique training articles, then equal article mean. No encoder finetuning. Cache identity/title/model revision checked before reuse.

For each monthly training cutoff, choose each branch C from earlier monthly forward scores; March defaults C=.1. June selection uses March–May, July March–June, August March–July. No evaluation-month label participates in selection or transforms. Final Jan–Aug fit selects C from March–Aug fold metrics. Every fit requires max training label end < min evaluation cutoff.

Fusion: nonnegative weights on four branch probabilities, step .25, sum1 (35 candidates including endpoints). All no-news rows return price probability exactly. Select using earlier chronological out-of-fold branch probabilities, where each monthly branch C was itself chosen from earlier folds. Require mean monthly Brier <= title baseline+.002, then maximize mean monthly BA, tie lower Brier, then fewer active components, then lexicographic weights. If no candidate meets guardrail, return the unchanged title baseline, explicitly marked fallback (no forced new method). Final weights selected only from March–Aug OOF. Also report the best >=2-component candidate under same guardrail, even if full selection collapses to a single model.

Ablations: rerun the same past-only weight selection with semantic/body/title respectively forbidden. Preserve sample coverage and no-news rule. Not ad hoc test-time weight removal. Uniform four-branch mixture is a fixed reference. No extra calibration grid, technical indicator search, manual keyword removal or independent-review-gated body extraction.

Evaluation: BA, MCC, Brier, accuracy, class recalls, predicted-up proportion, coverage and per-month/per-stock metrics. Shared-date paired bootstrap 10000 draws, 5-day primary blocks with1/10-day sensitivity, descriptive not multiple-search-adjusted. Case panel fixed hash sampling of repaired/introduced/common errors/common correct. Save weights, C selection provenance, individual branch models/transforms, input/cache hashes, OOF and frozen probabilities, and execution cost. No seed selection. Output directory must not exist.
