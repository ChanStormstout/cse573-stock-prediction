# Ordered text / joint semantics — completed handoff

**COMPLETE / VERIFIED.** Start from [plain-language findings](../outputs/stock_text_joint_4h/v1/FINDINGS.md), [report](../outputs/stock_text_joint_4h/v1/REPORT.md), [case reviews](../outputs/stock_text_joint_4h/v1/CASE_REVIEW.md) and `FINAL_STATUS.json`.

## What completed

Five finite methods on the original 1,607 windows, three seeds, ten outer folds and three inner folds. A1/A2/A3 restore raw text with unigram LR / unigram-bigram LR / calibrated SVM. B1/B2 append frozen FinBERT target-paragraph vectors to original FULL, PCA16 / 768 dimensions. Three C values and three rho values only. No new encoding or LLM calls.

- 5,243 classifier fits, 1,800 SVM sigmoid calibrators; 4,043 top-level fit calls.
- 60 complete blocks; 4,821 outer window/seed rows.
- 4,043 checkpoint containers replayed without fitting; max probability error 6.66e-16.
- 1,171,503 inner prediction rows independently checked; 263,307 rho-zero probabilities replayed.
- All 1,800 SVM calibration training memberships and 300 selected pipeline vocabularies/IDFs verified.
- Independent raw reread: 5,078 articles, exact old-stem parity for 1,607 windows.
- Five contract/corruption tests PASS. 24,230 protected artifacts and 42 prior public result hashes unchanged.

## Results and decision

Three-seed mean BA AAPL/AMZN:

| Method | AAPL | AMZN |
|---|---:|---:|
| FULL reference | 67.57% | 60.28% |
| Existing SVM + matched PRICE fallback | 69.55% | 61.14% |
| A1 | 61.71% | 61.06% |
| A2 | 59.21% | 60.26% |
| A3 | 63.28% | 61.16% |
| B1 PCA16 | 67.59% | 59.96% |
| B2 uncompressed | 65.98% | 60.40% |

No A/B method beats FULL and strong SVM on both stocks. Keep FULL as declared reference and matched-fallback SVM as the strong complete-pipeline comparator. The control reuses existing predictions and does not retrain SVM. Its paired intervals include zero. These are exposed exploratory random historical backtests; do not mix chronological scores or claim stable future gains.

Probability-only shrinkage improves Brier to .2133/.2382 with exactly unchanged directions, better than previous paragraph fusions. AMZN no-news windows account for approximately 54.2% of FULL's errors. New raw text can amplify repeated/mixed-company material; this is a case-supported hypothesis, not a proven causal explanation.

## Preservation / execution boundary

Private weights, original source text, reconstructed tokens and case evidence: `work/stock-data/text_joint_4h/v1`. Read README commands and protocol before reuse. Training resumes only hash-matched complete blocks, rejects incomplete directories, and never overwrites models. The verifier is fit-prohibited; it reuses its own hash-bound per-block verification fragments, not scientific authority from the runner. Report-only one-class BA/MCC clarification is recorded in REPORTING_NOTE.json; initial derived reports are retained privately. Main predictions and stock/seed metrics did not change.

LLM inference and old continuation remain suspended at 1,972 records, hash `c37cbc45187fd6cbd86a3144cd06e294963df509088471fc57664e817e8d8ae7`. Do not resume. No advanced branch is authorized by this completed task. Any future isolation of stemming/frequency or grouped robustness needs a separate finite protocol.

Preserve pre-existing user edits to .gitignore and docs/EXPERIMENT_INDEX.md. Only this task's index entry is included in its commit.

Test evidence import note: finalize.py uses an explicit file-path import so the parent lane's same-named test_contracts.py cannot shadow this lane. TEST_RECORD_CORRECTION.json retains the prior-record fingerprint; all five current tests passed again. No scientific output changed.
