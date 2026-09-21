# ECNI-lite four-hour combination: frozen finite protocol

This is an exposed exploratory chronological historical backtest. It does not
alter the canonical 1,607 AAPL/AMZN windows or any prior run.

## Question

Can cutoff-safe dissemination metadata, provisional event-update facts and
past-only analogous-event reactions add a stable residual correction to the
canonical price + full-body sparse model (`F1`)?

## Inputs

- Canonical `paper_methods_4h/v1/inputs.pkl` rows and labels.
- Canonical chronological `F1`/`R1` OOF and frozen Sep+ probabilities.
- Existing article/cluster/source/age metadata.
- Existing A1 event-adapter corpus facts. Facts are provisional model outputs,
  not independently reviewed gold. Only facts whose record is in the current
  window and whose availability is no later than the cutoff are eligible.
- Existing analogy-v2 past-only retrieval. Historical outcome features use
  only cases whose target interval ended before the query cutoff. Sep+ uses the
  case library frozen at August.
- Existing P2/P3 frozen Qwen scores are reused only as a direct-LLM outcome
  response control. They are not relabeled as pairwise comparability scores.

## Fixed methods

- `BASE`: canonical F1.
- `D`: BASE + dissemination block.
- `U`: BASE + provisional event-update block.
- `A`: BASE + simple past-reaction analogy block.
- `AL`: BASE + analogy block + saved P3-vs-P2 LLM outcome-response block.
- `DUA`: BASE + D + U + A.
- `DUAL`: BASE + D + U + A + LLM control.
- `DUAL_PARTIAL`: DUAL plus one shared, shrunk stock-contrast correction.

Every block is gated. If every eligible block is absent, final probability must
equal BASE bit-for-bit. No-news rows therefore remain the canonical R1 fallback
already embedded in F1.

## Model and selection

The final logit is the fixed BASE logit plus L2-regularized linear residual
blocks. Candidate C is exactly `{0.01, 0.1, 1}`. March has no earlier correction
OOF and returns BASE. April-August select C from candidate predictions in
strictly earlier OOF months, then train only on earlier correction OOF rows.
The final Sep+ C is selected from March-August past-only candidate records; the
correction is fit on March-August OOF and frozen for all Sep+ rows.

No development/later outcome selects a method, stock-specific winner, C,
feature or gate. `DUAL_PARTIAL` uses training active-count shrinkage only.

## Evaluation and gate

Report train-forward OOF, development and later separately, by stock: BA, MCC,
Brier, Accuracy, AUC, predicted-up rate, coverage, corrected/broken decisions.
The advancement screen uses June-August only and requires, versus BASE:

- mean BA gain >= 1 percentage point for both stocks;
- positive macro BA delta in at least 2 of 3 months;
- per-stock mean Brier worsening <= 0.002;
- no constant-direction collapse.

If no method passes, the selected system remains BASE. Later results cannot
override that decision.

## Interpretation limits

Event facts are provisional; dataset-first observation is not market-first
disclosure; article groups are approximate; historical reactions are
associations rather than causal effects. The experiment can validate incremental
features under this data protocol, not market causality or deployment.
