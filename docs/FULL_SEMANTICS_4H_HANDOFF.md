# FULL semantic extension handoff (2026-09-20)

## Authority and status

Owner redirected work through the side conversation: pause original analogy,
retain completed outputs, audit reuse, then run finite TF-IDF comparisons and
FULL plus target-paragraph semantics. Original random and grouped outer scores
have been disclosed. All new same-split comparisons are exploratory random
backtests, not fresh independent tests. No outer score chooses a per-stock winner.

Current pushed checkpoint before completed semantic results: 93c481d11a1b726bda5915552983a58f690bc875.
Traditional tests and public source protocols are already pushed.

## Preserved old work

Original inference PID 2333 and continuation PID 3030 are SIGSTOP-suspended.
1,972 complete output lines are retained. See PAUSE_AUDIT.json and the original
private inference seal. Output SHA256:
c37cbc45187fd6cbd86a3144cd06e294963df509088471fc57664e817e8d8ae7.
Old live progress files are copied under private pause_live_before and now show
PAUSED_OWNER_REDIRECT. Never resume the old automatic continuation without a new
explicit decision; it implements the old downstream order. No completed LLM
output has been rerun or discarded, and no partial-window score was reported.

## Completed new work

- Three TF-IDF text-only comparisons, same 3 seeds / 10 outer / 3 inner splits.
- 1,620 top-level estimator fit calls, 180 selected checkpoints. SVM calibration
  fits three separate text pipelines inside each call.
- Independent no-fit replay PASS; max probability difference 2.22e-16.
- Four targeted contract tests passed; classical descriptive day/block intervals.
- Raw-source semantic input verification PASS: 5,226 article-target pairs,
  30,448 paragraphs, 31,059 common chunks, zero omitted selected words.
- FinBERT complete frozen encoding: 1,100.83 seconds, MPS, no gradients.
- Fit-free semantic preflight PASS: all 240 checks, min 263 training article pairs.
- FinBERT downstream training complete: 600 fits, 60 outer checkpoints.
- Existing-title fusion controls registered and calculated separately; their
  limitations relative to the jointly selected body head are explicit.

## Active work: inspect before starting anything

Private root: work/stock-data/full_semantics_4h/v1.
Public root: outputs/stock_full_semantics_4h/v1.

The Python orchestrator (PID 6935 at this update; exec session 85002) is waiting
for Modern encoding (PID 8177; inspect live rather than assuming these are current).
It then runs train_semantics.py, verify_semantics.py, report.py with private logs.
FinBERT heads were run separately while Modern encoded. The trainer uses an
exclusive process lock and validates/skips completed blocks, so the pending
full invocation must not refit the 60 completed FinBERT blocks. Expected final
semantic total: 1,200 fits, 120 selected models, 9,642 outer rows.

The fit-free preparation had a slow linear token-boundary search and was stopped
before any encoder output. Original code is in f7fbae1; original log and empty
folder preserved. The replacement preserves every selected word and the common
512-token cap. See PREPARATION_RUNTIME_NOTE.json.

A fit-free preflight identified repeated NpzFile decompression. Arrays now load
once with verified exact equality; no scientific parameter changed and no
semantic classifier had fitted yet. See ARRAY_LOADING_RUNTIME_NOTE.json.
Do not change frozen predictive code or budgets in response to results.

## Remaining completion work

1. Confirm Modern encoding and the orchestrated training/verification/report finish.
2. Do not interpret body scores unless SEMANTIC_VERIFICATION.json is PASS.
3. Re-run paired_intervals.py after body results exist to add their descriptive
   intervals; no model training occurs in that script.
4. Inspect final tables, training-selected weights, both stocks and three seeds;
   report BA/Brier and repaired/introduced errors versus FULL. Distinguish
   descriptive random-backtest gains from future-period stability.
5. Update CURRENT_STATUS, PROJECT_LOG and this handoff with actual results.
6. Inspect/allowlist new compact JSON/CSV result files. Raw paragraphs, vectors,
   model checkpoints and environment stay private.
7. Refresh/check repository and git diff --check, commit only task files, push
   and verify remote main. Preserve unrelated .gitignore and EXPERIMENT_INDEX
   edits byte-for-byte around refresh; force-add only inspected allowlisted data.
8. Do not automatically resume LLM, add attention/LoRA, or touch external lanes.

## Important caveats

The paragraph selector checks explicit company names, not independently reviewed
semantic event roles. Pronoun-only surrounding paragraphs are not guaranteed.
The encoders are frozen and the small logistic heads are newly trained. Finite
mixing weights are selected inside outer-training data; no learned meta-model
uses selection_oof_NOT_META_TRAIN.csv. Title-fusion controls retain old selected
head C, whereas body C and weight are jointly selected, so their difference is
not a perfectly isolated causal effect of longer text.
