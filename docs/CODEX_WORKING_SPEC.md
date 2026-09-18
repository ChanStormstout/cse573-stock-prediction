# Codex Working Specification

## Task

Repair and audit the experimental foundation of the ASU CSE 573 AAPL/AMZN
four-hour direction project. Preserve the existing 1,607 official windows and
all historical runs, repair only materially invalid or confounded implementations,
correct claims about paper-inspired methods, and leave an external reviewer a
reproducible method-validity trail. Dense v5 is the accepted Phase A dense
artifact. Reaction v3 is retained as a historical artifact but is superseded
pending the v4 repair below. Phase B is implemented and preregistered but
remains unexecuted until explicit `APPROVE_GATE_RUN` after its preflight
repairs pass.

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
| ISSUE-022 | BLOCKING | Dense v4 still computes its executable advancement gate over March–August even though it constructs June–August `gate_rows` | The published dense PASS/stop decision is not the registered June–August gate | RESOLVED_PHASE_A |
| ISSUE-023 | BLOCKING | Reaction article price context can include an unfinished five-minute bar | Reaction features may read price movement after the article availability cutoff | RESOLVED_PHASE_A |
| ISSUE-024 | BLOCKING | Reaction W0–W3 does not match the originally requested downstream protocol | The reported reaction downstream comparison is not the requested matched test | RESOLVED_PHASE_A |
| ISSUE-025 | BLOCKING | Reaction promotion gate is incomplete | A reaction branch may have been promoted or stopped without all registered conditions | RESOLVED_PHASE_A |
| ISSUE-026 | BLOCKING | AR1 numeric/text preprocessing and AR2 representation need correction | Reaction and representation results are not yet valid evidence for their intended mechanisms | RESOLVED_PHASE_A |
| ISSUE-027 | BLOCKING | Reaction verification does not assert the critical time-safety contracts | A PASS status does not currently prove no future price context entered the reaction features | RESOLVED_PHASE_A |
| ISSUE-028 | BLOCKING | Reaction price context selects bars by start time but can include the bar that was still forming at article availability | The feature can include post-availability price movement | RESOLVED_PHASE_A |
| ISSUE-029 | BLOCKING | Reaction features do not publish a per-row maximum used-bar end time and completion flag | The cutoff contract cannot be audited row by row | RESOLVED_PHASE_A |
| ISSUE-030 | HIGH | Article availability, same-session reaction start, and horizon maturity are not asserted together in the downstream input manifest | A reaction label or context can be aligned to a different information boundary | RESOLVED_PHASE_A |
| ISSUE-031 | BLOCKING | W0 baseline construction is not frozen against the originally requested base model, split, and feature source | W0–W3 deltas may compare a different baseline than the protocol specifies | RESOLVED_PHASE_A |
| ISSUE-032 | BLOCKING | W1/W2 horizon selection and aggregation do not have an explicit registered definition and coverage rule | A downstream gain/loss cannot be attributed to the intended reaction input | RESOLVED_PHASE_A |
| ISSUE-033 | BLOCKING | W3 residual adjustment and strict fallback condition are not recorded as a complete protocol | Windows with insufficient valid reactions may be changed or retained inconsistently | RESOLVED_PHASE_A |
| ISSUE-034 | BLOCKING | Reaction promotion needs per-stock, per-month, Brier, coverage, and horizon-completeness checks in one machine-readable gate | Article-level promotion can occur without satisfying the downstream guardrails | RESOLVED_PHASE_A |
| ISSUE-035 | BLOCKING | AR1 numeric scaling and lexical preprocessing must be fit only on the past fold and applied identically to evaluation rows | Text and numeric effects can be confounded or leak future distribution information | RESOLVED_PHASE_A |
| ISSUE-036 | BLOCKING | AR2 must use the registered target-context representation with explicit source provenance and coverage accounting | Precomputed or mismatched vectors cannot support the intended AR2 claim | RESOLVED_NOT_RUN_MODEL_BINARY_UNAVAILABLE |
| ISSUE-037 | BLOCKING | Reaction verification lacks assertions that every feature's used-bar end is at or before the article cutoff | Existing PASS does not establish time safety | RESOLVED_PHASE_A |
| ISSUE-038 | HIGH | Reaction verification lacks a future-price perturbation/replay test for context and labels | A hidden future dependency may survive ordinary key and range checks | RESOLVED_PHASE_A |
| ISSUE-039 | HIGH | Reaction grouping, duplicate suppression, and target-pair membership are not tied to the downstream manifest with auditable counts | The downstream sample may not be the same registered article/event unit | RESOLVED_PHASE_A |
| ISSUE-040 | HIGH | The corrected reaction run needs a frozen protocol fingerprint and a new output directory rather than in-place repair | Historical reaction results could be silently mixed with repaired results | RESOLVED_PHASE_A |
| ISSUE-041 | BLOCKING | Reaction downstream eligibility was incorrectly described as requiring AR2, even though AR1 has its own complete article-level promotion gate | A valid AR1 article candidate could be discarded solely because the optional AR2 binary is unavailable | RESOLVED_BEFORE_V3_CORRECTED |
| ISSUE-042 | BLOCKING | Reaction v3 ran only 60m/240m although the preregistered corrected article gate applies to 30m/60m/120m/240m | The exposed 120m candidate was silently excluded and the candidate family was incomplete | RESOLVED_V4_ALL_HORIZONS_GATE_FAIL |
| ISSUE-043 | BLOCKING | Reaction v3 encodes unavailable pre-article price context as numeric zero and does not give AR0/AR1 validity flags | The model cannot distinguish a true zero return/realized volatility from missing context | RESOLVED_V4_NAN_PLUS_UNSCALED_FLAGS |
| ISSUE-044 | HIGH | The target builder treats every standalone AAPL token as Apple ticker evidence, including the known American Association for Physician Leadership acronym expansion | AAPL target association can include a known non-Apple entity | RESOLVED_V4_NARROW_REJECTION |
| ISSUE-045 | BLOCKING | Stock-specific controller preparation omits the registered `advantage` target used by the runner | Phase B cannot execute the stated supervised controller protocol | RESOLVED_PREFLIGHT_NOT_RUN |
| ISSUE-046 | BLOCKING | The controller's advertised R1 support fallback sets `d_hat=0`, which maps to a 50/50 R1/F1 mixture | Rows labelled as R1 fallback are not exact R1 predictions | RESOLVED_PREFLIGHT_NOT_RUN |
| ISSUE-047 | BLOCKING | Controller evaluation fills missing state with the training mean although fitting used the training median before scaling | Training and evaluation preprocessing are not the same registered transform | RESOLVED_PREFLIGHT_NOT_RUN |
| ISSUE-048 | HIGH | Controller routing headroom counts any probability difference rather than prediction-direction disagreement | The BA advancement requirement can be satisfied by rows a convex mixture cannot directionally repair | RESOLVED_PREFLIGHT_NOT_RUN |
| ISSUE-049 | BLOCKING | Controller evidence validates F1_new substantially but does not establish the R1 issued-probability chronological provenance | One of the two frozen experts is not independently evidenced | RESOLVED_PREFLIGHT_NOT_RUN |
| ISSUE-050 | BLOCKING | Static verification does not exercise the actual preparation path or the registered fallback/imputation/headroom contracts | Existing PASS could miss the concrete Phase B failures above | RESOLVED_PREFLIGHT_NOT_RUN |
| ISSUE-051 | HIGH | The hindsight routing oracle can switch to F1_new on no-news rows although the registered system requires exact R1 there | A diagnostic oracle can manufacture routing headroom that the controller may never use | RESOLVED_CODE_ONLY_ORACLE_CONTRACT |
| ISSUE-052 | BLOCKING | `verify.py` verifies only preflight contracts and cannot independently reconstruct a future approved controller run | A future G0--G3 report would lack an independent result audit | RESOLVED_CODE_ONLY_POSTRUN_VERIFIER |
| ISSUE-053 | MEDIUM | The preflight has no direct regression example proving BA headroom requires direction disagreement rather than any probability difference | A later edit could silently regress the registered advancement diagnostic | RESOLVED_CODE_ONLY_REGRESSION_TEST |
| ISSUE-054 | HIGH | Reaction v4 perturbation replay changes bars by start time and does not explicitly mutate a still-forming bar whose end is after availability | The replay does not directly attack the historical unfinished-bar failure mode | RESOLVED_V4_STRONGER_VERIFIER_PASS |
| ISSUE-055 | LOW | The stock-specific implementation plan still says no command has executed after real-input preflight completed | Documentation understates executed verification and can blur the predictive boundary | RESOLVED_DOCUMENTATION |
| ISSUE-056 | BLOCKING | The approved run's post-run verifier raises `NameError: clean is not defined` before writing `v1/verification.json` | The sole G0--G3 run is not independently verified and must not be interpreted or used for tuning | STOPPED_UNVERIFIED_RUN_PRESERVED |
| ISSUE-057 | BLOCKING | Verifier-only recovery writes `v1/verification.json=FAIL`: the independent advancement record differs from `advancement.json` at more than `1e-12` because the latter was persisted with approximately ten decimal digits | The one frozen run remains unverified unless external review authorizes a resolution; the runner and saved result files must not be changed | STOPPED_UNVERIFIED_RUN_PRESERVED |

### Addendum correction for ISSUE-041

AR1 and AR2 are independent article-level candidates.  AR1 may supply the
official-window reaction features when it passes its complete preregistered
June--August gate (including the per-stock Brier guardrail, macro AUC, monthly
direction and coverage checks).  AR2 is separately compared with its stricter
gate and is recorded as unavailable when the target-context FinBERT binary is
missing.  AR2 unavailability does not stop a valid AR1 downstream run.  The
corrected v3 run must use W0=R1, W1=R1 plus coverage metadata, W2=R1 plus the
five reaction features, and W3=F1 plus R1 plus those same five features.  The
earlier v2 downstream protocol remains a historical mismatch.

## Planned execution stages

- [x] Stage 0 — record starting state and create persistent working spec,
  validity ledger, correction log, and ChatGPT handoff.
- [x] Stage 1 — canonical parity and recency repairs (ISSUE-001/002/003/007).
- [x] Stage 2 — dense gate and feature parity repair (ISSUE-004/005/006).
- [x] Stage 3 — matched FinBERT/Fin-ModernBERT pooling probe (ISSUE-008/009).
- [x] Stage 4 — TabPFN checkpoint/configuration audit and conditional corrected
  probe (ISSUE-010/011).
- [x] Stage 5 — claim-only wording and calibration/dedup interpretation audit
  (ISSUE-012 through ISSUE-021).
- [x] Stage 6 — global verification, final logs, handoff, commit and push.
- [x] Stage 7 — incorporate reviewer addendum and repair/re-run blocked dense and
  reaction artifacts (ISSUE-022 through ISSUE-040).
- [x] Stage 8A — implement and preregister the stock-specific reliability gate;
  predictive execution remains gated by explicit `APPROVE_GATE_RUN`.
- [x] Stage 8B — repair Reaction v4 and Phase B preflight contracts
  (ISSUE-042 through ISSUE-050); do not generate Phase B predictions.
- [x] Stage 8C — final code-only verification repairs (ISSUE-051 through
  ISSUE-055); preflight and strengthened reaction v4 verifier passed without
  a predictive controller replay.
- [x] Stage 9 — execute the single approved Phase B run and invoke its
  independent verifier. ISSUE-056 stopped interpretation because the verifier
  failed before writing `v1/verification.json`; no repair, report or tuning is
  authorized in this stage.
- [x] Stage 9R — verifier-only recovery moved the pre-existing `clean()` helper
  into module scope and ran only the existing verifier. ISSUE-057 leaves one
  advancement reconstruction check false; stop without runner or result edits.

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

Stages 0–2 are complete locally. The local v4 recency/reaction implementation
from commit e32785d is preserved; the audit added formula/gate assertions and
public verification only, without overwriting historical runs. Remote push is
pending network/DNS recovery.

Stage 3 is also complete: canonical FinBERT reproduction passed at
`1.72e-15`; ModernBERT v2 used special-token-excluded pooling and showed no
stable two-stock improvement. Because v2 did not save the exact promotion
formula as a machine-readable gate, no formal promotion pass/fail claim is
made and no dual-encoder fusion was run.

Stage 4 is complete. The local TabPFN 6.3.0 metadata/checkpoint audit passed;
the default classifier is real-data-fine-tuned, so the old synthetic-only
wording is corrected. A fixed n=8 own/cross probe ran successfully but showed
no stable two-stock improvement. Because v2 did not save the exact goal60
formula as a machine-readable gate, no formal promotion pass/fail claim is
made; no further TabPFN grid was authorized.

Stage 5 is complete. The claim-only audit found 32 negative slopes in the 84
saved early unconstrained Platt records, all from the AMZN branch. The later
constrained calibration manifests have no negative slopes. Current-facing
wording now distinguishes small probes from full paper reproductions, treats
the dissemination result as regularization-confounded, treats Qwen scores as
token preferences, and separates historical F1/F2 controls from F1_new/F2_new.
No prediction, model, or exposed-period score was changed in this stage.

Stage 6 is complete locally. All five public verifiers, `refresh_repository.py`,
`check_repository.py`, and `git diff --check` passed. The final evidence is in
`outputs/stock_method_validity_audit/v1/final_verification.json`. A push is
still attempted after the final commit; if GitHub DNS remains unavailable,
the local commit and remote-tracking boundary are reported explicitly.

## External review checkpoint after `8842d37` (2026-09-18)

The external reviewer accepts dense v5 with its existing `VALID_WITH_CAVEAT`
status. It requires a new, non-overwriting reaction v4: all four registered
30/60/120/240 minute horizons; NaN plus explicit validity flags for missing
pre-article price context; and high-precision rejection of the known AAPL
association acronym. v3 is `SUPERSEDED_PENDING_REPAIR` until that verifier
passes. The same review found Phase B defects: the missing `advantage` target,
false R1 support fallback, mismatched imputation, probability rather than
direction headroom, incomplete R1 provenance, and insufficient static tests.
`run_gate.py --approve-gate-run` remained prohibited during repair. The
preflight now passes but Phase B remains `PREREGISTERED_NOT_RUN` until a later
explicit approval; repair wrote no G0--G3 metrics, predictions, oracle
ceilings, or exposed-period gate results.

## Reaction v4 and Phase B preflight checkpoint (2026-09-18)

`stock_reaction_features_4h/v4` rebuilt the full source corpus in a new
directory. It rejected five occurrences of the known non-Apple AAPL acronym
under the intentionally narrow rule, used completed-bar context with NaN for
missing returns/realized volatility, and supplied unscaled validity indicators
to AR0/AR1. The all-horizon verifier passed. All AR1 candidates failed their
independent June--August gates: 30m has AAPL/AMZN mean BA deltas
`-0.996pp/-0.811pp`; 60m `-0.351pp/-0.561pp`; 120m `+0.322pp/+2.570pp`
but violates AAPL's +1pp and Brier requirements; 240m `+0.336pp/+0.653pp`
and has AMZN constant-direction collapse. Therefore no W0--W3 downstream run
was permitted. AR2 remains `NOT_RUN_MODEL_UNAVAILABLE`.

The Phase B preflight reads the actual 1,607 keys. It now independently proves
advantage arithmetic, R1 parity to the real `nextgen_4h/price_v1` output,
R1/F1_new chronological fit provenance, exact R1 and nested fallback behavior,
identical median-based train/evaluation transforms, direction-disagreement
headroom, finite weights, and G3 nesting. It is a preflight only, not a
controller result.

## Historical reviewer-addendum pause record (2026-09-17; superseded)

The external reviewer identified new blocking validity issues in the
implementation that was based on the earlier `e32785d` state. These findings
supersede the earlier local PASS interpretation for the affected dense and
reaction artifacts; they do not delete or invalidate the historical files.
The exact repair instructions/addendum have not yet been incorporated here.

At that checkpoint execution was **PAUSED_PENDING_REVIEWER_ADDENDUM**. That
state is superseded by the Phase A repair checkpoint below. The coarse blocking
areas are ISSUE-022 through ISSUE-027; the detailed reaction requirements are
split into ISSUE-028 through ISSUE-040 above: completed-bar censoring and row
provenance, availability/horizon alignment, the exact W0/W1/W2/W3 protocol and
fallback, the full promotion gate, AR1 preprocessing, AR2 provenance/coverage,
time-safety and future-perturbation verification, grouping auditability, and a
new-run fingerprint.

No command was running when this pause was received. All previous work,
historical runs, local commits, and audit artifacts remain preserved; no reset,
discard, or overwrite was performed. The subsequent Phase A section records
the incorporated addendum and new output directories.

## Deviations / new problems discovered while executing

- The requested remote fetch was attempted but GitHub DNS resolution failed;
  no remote state beyond the existing tracking ref can be asserted.
- The current environment has NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.9.1,
  torch 2.14.0, transformers 4.57.6, and no importable `tabpfn` package in the
  project environment. This makes the TabPFN audit conditional.
- The public v4 outputs already passed their own parity/gate checks; this task
  audits additional historical method claims without rewriting v3/v4 files.
- The v4 recency weight audit covered 17,140 private fold rows: formula error
  was zero and there were no monotonic or infinity-unit-weight violations.
- The historical dense verifier artifact reports exactly six June–August
  stock-month cells and requires the matched reconstructed feature mode after
  parity failed; ISSUE-022 now supersedes that interpretation because the
  executable advancement gate was found to use March–August. A corrected
  June–August calculation is pending.
- The v2 ModernBERT run used 5,078 matched titles, 159 MPS batches, and zero
  trainable encoder parameters. Its old-v1 versus v2 article-vector mean L2
  difference is 2.0673.
- The TabPFN audit verified package 6.3.0, checkpoint SHA
  `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`, and
  the real-data-fine-tuned provenance. The corrected n=8 probe completed 28
  fits with maximum reload error `1.79e-7`.

## Exact commands executed

```text
git status --short --branch
git fetch                         # failed: could not resolve github.com
git log -n 5 --oneline --decorate
git rev-parse HEAD
git rev-parse origin/main
work/stock-data/finbert-env/bin/python3 - <<'PY' ... importlib.metadata ... PY
```


## Current authoritative checkpoint — Phase A repaired, Phase B preregistered

The repair checkpoint began from local `dd836d80c709cd98065249ab5cdde233bd7abdc1`
and preserves every earlier artifact.  New Phase A artifacts are:

- dense `outputs/stock_recency_dense_4h/v5`: the executable June--August
  gate now filters both sides before the six-cell merge; the gate fails because
  AAPL is `+3.740pp` while AMZN is `-0.079pp`;
- reaction `outputs/stock_reaction_features_4h/v2`: completed-bar context,
  row-level used-bar provenance, corrected AR1 preprocessing, explicit AR2
  `NOT_RUN_MODEL_BINARY_UNAVAILABLE`, and the full promotion gate; 240m passes
  the article gate, but its W1 downstream score is AAPL `51.98%` and AMZN
  `44.43%` versus W0 `48.70%` and `60.63%`.

`verify_v5.py`, `verify_infinity_parity_v5.py`, and `verify_v2.py` pass.  The
raw dense canonical parity failure is explicitly handled by the matched
reconstruction mode and is not silently called a parity pass.  The old v4
dense/reaction numbers remain historical and are superseded for current claims.

Phase B `outputs/stock_specific_gate_4h/` is implemented and preregistered,
with static contract tests passing.  It uses fixed `R1` and `F1_new` experts,
ten fixed state variables, G0--G3 weighted-ridge controllers, exact no-news
R1 fallback, and no exposed-label updates.  **No G0--G3 predictive run has
been executed.**

## Current reaction v3 corrected checkpoint — 2026-09-18

The first v3 article-only run is preserved as
`outputs/stock_reaction_features_4h/v3_article_audit_checkpoint/`.  The
corrected v3 run applies ISSUE-041: AR1 and AR2 have independent article-level
promotion gates.  AR1 240m has AAPL/AMZN June--August mean BA deltas
`+1.763pp/+3.218pp` and satisfies the per-stock Brier guardrail, but its macro
AUC delta is `-0.570pp`; AR1 60m also fails.  AR2 is
`NOT_RUN_MODEL_UNAVAILABLE`.  Thus no candidate passes the full article gate,
and v3 intentionally produces no official W0--W3 predictions.  The v2
downstream numbers remain historical protocol-mismatch evidence only.
