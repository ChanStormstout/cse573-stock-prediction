# Codex Working Specification

## Task

Repair and audit the experimental foundation of the ASU CSE 573 AAPL/AMZN
four-hour direction project. Preserve the existing 1,607 official windows and
all historical runs, repair only materially invalid or confounded implementations,
correct claims about paper-inspired methods, and leave an external reviewer a
reproducible method-validity trail. This task ends before any new
reaction-supervised FinBERT, LLM prompt search, GNN, RL, Chronos tuning, or
exposed-period score search.

## Starting Git SHA

- Current local HEAD at task start: `e32785ded90646c7911e55ff931fd322358974c5`
- Remote-tracking `origin/main`: `c0383aca45022ec4a444509d445f4a02ab60d6e5`
- Branch: `main`, ahead of the remote by one local commit (`e32785d`).
- `git fetch` was attempted before changes and failed because the environment
  could not resolve `github.com`. The local e32785d work is preserved.
- Working tree at task start: clean.

## What I believe the user/reviewer is asking for

The reviewer is asking for a repair-first audit, not another broad model
search. We must prove canonical parity where possible, separate matched
component tests from paper reproductions, rerun only invalid/confounded
experiments, correct current-facing language, maintain an issue ledger, and
push each stage when network access permits. All results remain exploratory
historical backtests because September onward has already been exposed.

## Scientific invariants

- fixed 1,607 official four-hour windows;
- stocks are AAPL and AMZN;
- UP iff target final close is greater than target opening price;
- prediction cutoff is target start minus five minutes;
- January–February warmup, March–August forward historical training evaluation,
  September–October development, November onward later;
- development/later/June–August values are `EXPOSED EXPLORATORY HISTORICAL BACKTEST`;
- keep all windows, including hard and no-news windows;
- no future labels or feature timestamps beyond cutoff;
- no threshold or per-stock architecture selection on development/later;
- Balanced Accuracy is the primary directional metric; also report MCC and Brier;
- preserve one-hour history separately;
- raw data, article text, caches, model weights, and private environments stay
  outside Git.

## Known issues that must be repaired

| ID | Severity | Issue | Why it matters | Initial status |
|---|---|---|---|---|
| ISSUE-001 | HIGH | F2 recency uses window-mean PCA rather than canonical article-level J2 and omits metadata | F2 recency deltas are confounded by a representation change | NEEDS_RERUN |
| ISSUE-002 | HIGH | Recency infinity uses saved probabilities instead of a true all-one-weight refit | Reference parity does not prove same-code parity | NEEDS_RERUN |
| ISSUE-003 | MEDIUM | F1/F2 recency also changes the no-news R1 fallback | Text weighting and fallback effects cannot be separated | REPAIR |
| ISSUE-004 | HIGH | Dense advancement gate includes March–August rather than June–August | Reported dense deltas do not match preregistered gate | RECOMPUTE |
| ISSUE-005 | HIGH | Dense reconstructed R1 features are not proven identical to canonical inputs | Mixed train/evaluation generators can create an input confound | REPAIR |
| ISSUE-006 | LOW | Dense preregistration incorrectly says non-overlapping | Misstates the dependence structure of windows | DOCUMENT |
| ISSUE-007 | MEDIUM | Weight verification checks positivity but not formula/monotonicity | A wrong age/half-life implementation could pass | REPAIR |
| ISSUE-008 | MEDIUM-HIGH | ModernBERT pooling includes special tokens while canonical FinBERT excludes them | Encoder and pooling both change in the comparison | NEEDS_MATCHED_RERUN |
| ISSUE-009 | MEDIUM | FinBERT8 + Modern8 inherits the Modern pooling mismatch | Fusion comparison is not encoder-matched | CONDITIONAL_RERUN |
| ISSUE-010 | HIGH | TabPFN is described as synthetic-only without verified checkpoint provenance | The method label may be false | AUDIT |
| ISSUE-011 | MEDIUM | TabPFN uses `n_estimators=1` | This is a reduced-compute probe, not standard configuration | DOCUMENT/AUDIT |
| ISSUE-012 | MEDIUM | Masked reconstruction SSL is not TS2Vec | A pilot cannot support a claim about TS2Vec | CLAIM-ONLY |
| ISSUE-013 | MEDIUM | Historical analogy is not FinSeer reproduction | Lexical RAG probe cannot refute FinSeer | CLAIM-ONLY |
| ISSUE-014 | MEDIUM | Event Adapter is not Ding event/graph embedding | Extraction heads are a narrower task-adaptation probe | CLAIM-ONLY |
| ISSUE-015 | LOW | Chronos result is an endpoint-feature probe, not a full capability test | Prevents overclaiming a foundation model failure | CLAIM-ONLY |
| ISSUE-016 | MEDIUM | Early unconstrained Platt slopes can be negative | Ranking may be inverted, so it is not simple monotone calibration | CLAIM-ONLY |
| ISSUE-017 | MEDIUM | Dissemination/event-clustering local AMZN gain is confounded by regularization | Cannot attribute local gain to clustering | CLAIM-ONLY |
| ISSUE-018 | MEDIUM | Qwen direct UP/DOWN score is token preference, not calibrated probability | Brier interpretation must be qualified | CLAIM-ONLY |
| ISSUE-019 | LOW | Cross-stock past-state experiment needs explicit validity wording | Avoids implying future contemporaneous peer data | CLAIM-ONLY |
| ISSUE-020 | MEDIUM | Continuous-return auxiliary result cannot show return magnitude has no signal | C2 failure only applies to the tested shared objective | CLAIM-ONLY |
| ISSUE-021 | HIGH | Historical F1/F2 controls and reselected F1_new/F2_new are not clearly separated | Silent control swapping invalidates comparisons | DOCUMENT |

## Planned execution stages

- [x] Stage 0 — record starting state and create persistent working spec,
  validity ledger, correction log, and ChatGPT handoff.
- [ ] Stage 1 — canonical parity and recency repairs (ISSUE-001/002/003/007).
- [ ] Stage 2 — dense gate and feature parity repair (ISSUE-004/005/006).
- [ ] Stage 3 — matched FinBERT/Fin-ModernBERT pooling probe (ISSUE-008/009).
- [ ] Stage 4 — TabPFN checkpoint/configuration audit and conditional corrected
  probe (ISSUE-010/011).
- [ ] Stage 5 — claim-only wording and calibration/dedup interpretation audit
  (ISSUE-012 through ISSUE-021).
- [ ] Stage 6 — global verification, final logs, handoff, commit and push.

## Stop conditions

- If a true infinity refit does not reproduce canonical probabilities within
  `1e-10`, stop that method's weighted recency result.
- If canonical FinBERT cannot reproduce its existing matched design/predictions,
  stop the ModernBERT comparison until the shared pipeline is fixed.
- If TabPFN checkpoint provenance cannot be established, do not run or label a
  corrected synthetic-only probe; mark it `BLOCKED_UNVERIFIED_CHECKPOINT`.
- Do not start a new reaction-supervised FinBERT branch or any new LLM/GNN/RL/
  Chronos/tuning branch in this task.
- Do not use September onward to choose a method or stock-specific winner.
- If a new material issue appears, document a new ISSUE ID here and in the
  validity ledger before changing code.

## Experiments explicitly NOT authorized in this task

New reaction-supervised FinBERT, fine-tuning on returns, new Qwen prompts,
GNNs, RL, Chronos tuning, dense-stride tuning, new recency half-lives,
ModernBERT architecture search, and any search for a score above 60% on exposed
periods.

## Current stage

Stage 0 documentation bootstrap completed. Stage 1 is next. The local v4
recency/reaction implementation from commit e32785d is preserved; this audit
will not overwrite it. Remote push is pending network/DNS recovery.

## Deviations / new problems discovered while executing

- The requested remote fetch was attempted but GitHub DNS resolution failed;
  no remote state beyond the existing tracking ref can be asserted.
- The current environment has NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.9.1,
  torch 2.14.0, transformers 4.57.6, and no importable `tabpfn` package in the
  project environment. This makes the TabPFN audit conditional.
- The public v4 outputs already passed their own parity/gate checks; this task
  audits additional historical method claims without rewriting v3/v4 files.

## Exact commands executed

```text
git status --short --branch
git fetch                         # failed: could not resolve github.com
git log -n 5 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/main
work/stock-data/finbert-env/bin/python3 - <<'PY' ... importlib.metadata ... PY
```
