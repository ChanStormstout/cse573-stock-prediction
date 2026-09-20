# Analogy implementation detail frozen before downstream evaluation

This elaborates the analogy branch; it does not alter the five baseline fits,
the registered splits, or the proposed learned fact-change residual.

The first analogy correction is deliberately a finite shared scalar:
`sigmoid(logit(base) + alpha*tanh(logit(signal)))`, alpha in 0/.1/.25/.5.
Signal is respectively fixed current-only LLM probability, smoothed similarity
vote, or LLM probability with past cases. This is an ablation of historical
retrieval and LLM comparison, NOT the complete fact-change method.

No learned meta-classifier is fitted to the selected baseline OOF records.
Those records are used only for inner hyperparameter selection. One alpha per
encoder/variant/outer-fold is shared across stocks; select inner mean stock BA
with each stock Brier worsening <=.002, tie smaller alpha. Alpha0 is always a
candidate. The selected correction is then applied once to sealed outer rows.
No-event/no-case probabilities return the exact baseline float. UP/DOWN token
probabilities are conditional choices, not calibrated confidence; retain choice
mass and unconstrained first token in the private inference audit.

Fold-local historical outcomes increase inference cost: 48,210 query evaluations
deduplicate to 7,851 exact prompts. This is not 48,210 independent market samples.
Preflight estimated about four hours from an older 1.85 sec/prompt rate; actual
runtime is recorded, not promised. Exact-prompt reuse is allowed; replacing
test-partition exclusions with old contaminated prompts is not a new holdout
test. The owner permits non-independent historical diagnostics, but the core
comparison retains its registered fresh-training interpretation.

The model and prompts are frozen. The initial inference launch failed during
import, before model load/calls, because MLX's environment lacks scipy. The
inference utility now depends on standard-library IO instead of importing the
sklearn training module. Original traceback is preserved privately; no model,
prompt, output, split or feature was changed by that runtime repair.
