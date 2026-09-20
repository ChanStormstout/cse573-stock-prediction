# GPT Pro mechanism completion — active finite execution

Reference commit: `e686c49305a7771ff9298568b98587f4af44a7c0`.

The owner resumed after a discussion of chronological versus random evaluation. This lane completes the already requested random-protocol mechanisms; it does **not** substitute these scores for future-period forecasts or the proposed faculty-slide news-fact-change system.

## Frozen scope

- Strict real-adjacency unigram/bigram calibrated SVM, plus an optional finite NB-weighted bigram SVM.
- Word + price sum-kernel SVC; full linear mean, map-after-mean, map-before-mean, and a three-price-state interaction.
- Independent compact-table LR / shallow RF / local fixed TabPFN, with and without semantic columns. This challenger is **not conditional** on the set-aggregation result.
- PCA16 regularized residual with freshly nested cross-fitted FULL/PRICE base probabilities. Includes unchanged-base selection candidate. Does not use `selection_oof_NOT_META_TRAIN.csv` to fit a correction.

No encoder training/inference or new LLM calls. No TabSTAR, learned retrieval, attention, LoRA or RL in this finite run; these were later-stage items in the original Pro advice.

## Files and commands

Public code/results: `outputs/stock_pro_completion_4h/`, `v1/`.
Private artifacts: `work/stock-data/pro_completion_4h/v1/`.

```sh
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/test_contracts.py
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/run.py lexical
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/verify.py lexical
```

Repeat run/verify for `kernel`, `table`, `residual`, then run `report.py` only after all four verification files report PASS.

Completed blocks are hash-checked and skipped, never re-fitted. An incomplete block stops a restart and must be inspected before any recovery. Original predictions, caches, raw text and models remain protected.

## Preflight history

Five synthetic tests pass, including exact no-evidence residual fallback, offset serialization, tensor-product/summed-kernel algebra, NB inference immutability and cross-process TabPFN conditioning/save/reload.

The old local TabPFN package requires its existing sklearn1.7 runtime; the main environment is sklearn1.9. TabPFN therefore runs in a separate process using the previous local runtime, with telemetry disabled. No dependency upgrade or network model call was made. This is TabPFN6.3.0 using a TabPFN2.5 default checkpoint fine-tuned on real data, not a synthetic-only checkpoint.

A verifier local-module-name typo was corrected before any real fit; the original seal and a pre-execution repair record are preserved. Predictive inputs and protocol were unchanged.

## Execution scheduling

To avoid serializing independent lanes, `run_stage.py` reuses the unchanged fitting functions in the sealed `run.py`; its main body differs only in using a per-stage lock. The first wrapper import resolved an unrelated legacy module named `run` and failed before initialization/fitting. It was repaired using an explicit file import before those stages began. Startup tracebacks remain in private logs. The waiting duplicate lexical launcher was stopped before obtaining the lock; the original lexical training process continued unchanged. The final audit compares wrapper/main bodies mechanically.

Use one executor per stage. Do not launch a normal and per-stage runner for the same unfinished stage concurrently.

## Kernel runtime equivalence repair

The first kernel block spent excessive time in sparse libsvm linear-kernel operations. It was interrupted before any kernel prediction/result/model checkpoint was written. The empty partial directory and original log remain private; its uncheckpointed fit count is unavailable and must not be included as an exact completed-fit total.

`kernel_fast.py` keeps the same libsvm linear SVC objective, C grid and features, uses dense BLAS storage for fitting, and computes the binary linear decision from coefficients. Three-C synthetic sparse/dense and coefficient-form comparisons differ by at most about3e-15. `run_kernel_fast.py kernel` imports the same frozen prediction functions with this equivalent estimator implementation. Final audit also compares saved real-model fast margins to libsvm's original dense decision routine. No scores were used to choose this runtime change.
