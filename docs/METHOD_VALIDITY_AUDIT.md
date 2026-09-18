# Method Validity Audit

This is a persistent truth table, not a leaderboard. September onward and all
other exposed periods are exploratory historical backtests. `VALID` means the
implementation supports the stated narrow claim; it does not mean independent
generalization has been proven.

| ID | Project method | External paper/model | Actual implementation | Fidelity | Code status | Result status | Safe claim | Unsafe claim | Repair status |
|---|---|---|---|---|---|---|---|---|---|
| F0 | price + title baseline | none | four-hour canonical price features plus title features, chronological LR | DIRECT_BASELINE | `outputs/stock_integrated_4h/run.py` (F0/title branch); result `outputs/stock_integrated_4h/runs/v1/` | VALID | course four-hour title baseline is a direct project reference | F0 is state of the art | unchanged |
| F1 | price + full text | none / TF-IDF tradition | sparse full-news features plus price LR, past-only C selection | DIRECT_BASELINE | `outputs/stock_paper_methods_4h/run.py` J0 | VALID_WITH_CAVEAT | fixed full-text classical comparison | every wording is semantically understood | unchanged |
| F2 canonical FinBERT | price + frozen FinBERT | ProsusAI/FinBERT | frozen article vectors, canonical J2 PCA16/article mean/metadata, price LR | FOUNDATION_MODEL_PROBE | `outputs/stock_paper_methods_4h/run.py` J2 | VALID_WITH_CAVEAT | frozen FinBERT representation probe | reproduced a stock-prediction paper or deployment | unchanged |
| R1 | recent price fallback | none | recent five-minute/price feature LR with canonical fallback | DIRECT_BASELINE | `outputs/stock_nextgen_4h` and matched controls | VALID | tested recent-price baseline | recent price is universally predictive | unchanged |
| recency R1 | half-life weighted price | recency inspiration | same LR with infinity/80/40/20 XNYS sample weights | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/run_recency_v4.py` | VALID_WITH_CAVEAT | tested recency weighting failed the preregistered cross-stock gate | recency method is invalid in general | Stage 1 complete; formula audit PASS |
| recency F1 | half-life weighted sparse text | recency inspiration | weighted J0 text classifier, separate equal/weighted R1 fallbacks | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/run_recency_v4.py` | VALID_WITH_CAVEAT | v4 tested canonical J0 with explicit fallback modes | paper recency model was reproduced | Stage 1 complete; formula audit PASS |
| recency F2 | half-life weighted FinBERT | recency inspiration | v4 J2 canonical transform plus sample weights | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/run_recency_v4.py` | VALID_WITH_CAVEAT | v4 corrected F2 recency probe did not pass gate | old v3 F2 delta was recency-only | Stage 1 complete; formula audit PASS |
| dense price windows (v4 historical) | 30-minute stride dense bars | dense sampling idea | 48 contiguous five-minute bars, reconstructed matched control when parity fails | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/build_dense_windows_v4.py`, `run_dense_v4.py` | SUPERSEDED | historical v4 artifact retained; its executable March–August gate is superseded by repaired v5 | six-cell June–August comparison is currently valid or dense sampling was ruled out | superseded by v5 (ISSUE-022/028) |
| Fin-ModernBERT | modern financial encoder | Fin-ModernBERT | v2 frozen title encoder, special-token-excluded pooling, PCA/LR downstream | FOUNDATION_MODEL_PROBE | `outputs/stock_foundation_4h/encode_modern_v2.py`, `outputs/stock_foundation_4h/experiment_v2.py` | VALID_WITH_CAVEAT | matched frozen encoder probe has mixed exposed-period results; it is a descriptive comparison, not a machine-recorded promotion failure | reproduced Fin-ModernBERT stock forecasting or proved it ineffective | Stage 3 complete; promotion wording downgraded; no fusion |
| FinBERT + Modern 8+8 | dual representation | ensemble idea | v1 PCA8 concatenation with old pooling | MATCHED_COMPONENT_TEST | `outputs/stock_foundation_4h/experiment.py` | NEEDS_RERUN | historical v1 fusion remains unmatched; no v2 fusion was authorized after Modern gate failed | ModernBERT fusion is validated | Stage 3 stopped after single-encoder probe |
| Chronos-2 | time-series foundation model | Chronos-2 | frozen endpoint/quantile features, prediction_length 49, matched 512-bar LR | FOUNDATION_MODEL_PROBE | `outputs/stock_foundation_4h/chronos_experiment.py` | VALID_WITH_CAVEAT | tested frozen Chronos endpoint features lacked stable gain | Chronos-2 is ineffective for stocks | claim correction only |
| TabPFN 2.5 | tabular foundation model | TabPFN 2.5 | v2 local conditional classifier, verified default real-data-finetuned checkpoint, `n_estimators=8`; v1 n=1 retained as control | FOUNDATION_MODEL_PROBE | `outputs/stock_tabpfn_4h/v2/run_tabpfn_v2.py` | VALID_WITH_CAVEAT | corrected frozen-prior configuration probe showed no stable two-stock improvement under its fixed chronological comparison; no formal promotion pass/fail is claimed because v2 did not save the exact gate formula | synthetic-only/full standard TabPFN failed; reproduced TabPFN pretraining | Stage 4 complete; promotion wording downgraded; no further grid |
| masked-reconstruction SSL | price SSL pilot | TS2Vec related work | small Conv1d masked reconstruction, frozen LR representation | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_ssl_4h` | VALID_WITH_CAVEAT | this pilot did not give stable gains | TS2Vec failed | claim correction only |
| Event Adapter A1/A2 | task-adapted event extraction | Ding event/graph work related | FinBERT evidence/action heads, top layers or q/v LoRA | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_finbert_event_adapter_4h` | VALID_WITH_CAVEAT | provisional extraction improved, narrow event downstream did not | Ding event embedding was reproduced/failed | claim correction only |
| historical analogy P1/P2/P3 | historical-news analogy | FinSeer related motivation | HashingVectorizer/coarse category, max three cases, frozen Qwen | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_analogy_4h/v2` | VALID_WITH_CAVEAT | conservative lexical analogy with outcomes lacked stable increment | FinSeer was reproduced/failed | claim correction only |
| Qwen direct forecast | direct LLM direction | none | frozen Qwen3.5-9B conditional UP/DOWN token preference | FOUNDATION_MODEL_PROBE | `outputs/stock_llm_direct_4h` | VALID_WITH_CAVEAT | token preference probe had poor calibration/stability | output is true probability | claim correction only |
| historical-news H0/H1/H2 | past-news state | none | prior article/event state and price reaction variants | MATCHED_COMPONENT_TEST | `outputs/stock_goal60_4h/history*.py` | VALID_WITH_CAVEAT | historical state inputs did not show stable gain | H1 solves AMZN missing information | wording audit |
| cross-stock P_cross | peer-price state | cross-asset feature idea | past-only peer return/volatility features at cutoff | MATCHED_COMPONENT_TEST | `outputs/stock_goal60_4h/core.py::cross_features`, `price(cross=True)` | VALID | tested two-stock past-state features lacked stable gain | future peer returns were used | wording audit |
| continuous-return C1/C2 | auxiliary return objective | multi-task learning inspiration | Ridge return diagnostic and shared direction+return linear head | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_market_return_4h/run.py` | VALID_WITH_CAVEAT | tested shared auxiliary objective lacked stable promotion | return magnitude contains no signal | claim correction only |
| dissemination N0/N1/N0M/N1M | event aggregation | dissemination/event clustering motivation | article/event aggregation and metadata, LR | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_paper_methods_4h`, `stock_combination_4h` | VALID_WITH_CAVEAT | local gain was confounded by regularization | clustering independently caused 57.32% | claim correction only |
| calibration | probability calibration | Platt/temperature calibration | several unconstrained and later constrained variants | MATCHED_COMPONENT_TEST | `outputs/stock_paper_methods_4h`, `stock_combination_4h`, `stock_nextgen_4h/calibrate.py` | VALID_WITH_CAVEAT | later constrained calibration is the interpretable probe | every historical Platt map is monotone calibration | claim correction only |
| probability fusion | model probability mixture | ensemble calibration | past-OOF nonnegative weights and temperature/fusion controls | MATCHED_COMPONENT_TEST | `outputs/stock_combination_4h` | VALID_WITH_CAVEAT | tested fusion did not provide stable direction gain | best exposed-period mixture is deployable | wording unchanged |

## Global temporal contract

All claims above are conditional on the project contract: AAPL/AMZN, 1,607
four-hour windows, cutoff `start - 5 minutes`, Jan–Feb warmup, March–August
forward historical training evaluation, September–October development, later
from November onward, and no future labels. Where a row says `NEEDS_RERUN`,
the old artifact remains a historical diagnostic and is not overwritten.

## Evidence interpretation

`FOUNDATION_MODEL_PROBE` means a frozen model endpoint was tested with a chosen
downstream head; it is not a reproduction of a paper's full training system.
`PAPER_INSPIRED_SIMPLIFICATION` explicitly records a related idea with missing
components. A negative result for a simplified probe cannot be generalized to
the cited paper or model family.

## Phase A repaired artifacts (authoritative current status)

The old dense v4 row remains a preserved historical artifact and is
`SUPERSEDED_PENDING_RERUN`. The repaired v5 calculation is the current
evidence:

| method | implementation / result | status | safe claim |
|---|---|---|---|
| dense v5 | `outputs/stock_recency_dense_4h/run_dense_v5.py`, `outputs/stock_recency_dense_4h/v5/` | `VALID_WITH_CAVEAT` | executable June--August six-cell gate is correctly filtered; AAPL +3.740pp, AMZN −0.079pp, so no two-stock promotion |
| reaction v2 | `outputs/stock_reaction_features_4h/run_reaction_probe_v2.py`, `outputs/stock_reaction_features_4h/v2/` | `SUPERSEDED` | historical time-safety repair retained, but its W0--W3 downstream protocol did not match the complete reviewer addendum |
| reaction v3 corrected | `outputs/stock_reaction_features_4h/run_reaction_probe_v3.py`, `outputs/stock_reaction_features_4h/v3/` | `SUPERSEDED_PENDING_REPAIR` | historical v3 used 60m/240m only and encoded unavailable context as zero without model-visible flags; preserve it, but rely on neither its gate nor stop result until v4 runs all registered horizons |
| reaction v4 corrected | `outputs/stock_reaction_features_4h/build_reaction_dataset_v4.py`, `run_reaction_probe_v4.py`, `verify_v4.py`, `outputs/stock_reaction_features_4h/v4/` | `VALID_WITH_CAVEAT` | the v4 implementation passed time-safety/preprocessing/entity-rule verification; all four registered AR1 candidates were evaluated and each failed its complete June--August gate, so no W0--W3 downstream score exists | lexical reaction modeling is generally ineffective or article reactions cannot help four-hour direction |
| reaction AR2 | `outputs/stock_reaction_features_4h/v4/` | `NOT_RUN_MODEL_UNAVAILABLE` | required target-context FinBERT binary was unavailable; no substitute vectors were used; this does not block a separately passing AR1 candidate |
| stock-specific reliability gate | `outputs/stock_specific_gate_4h/` | `UNVERIFIED_STOPPED` | the one approved frozen run is preserved, but its post-run verifier raised `NameError: clean is not defined` before writing a verification artifact; no score is interpreted, reported as a method result, or used for tuning | repaired preflight means the issued G0--G3 numbers are scientifically valid |

The v5 raw dense feature parity check is false and is explicitly recorded;
both control and augmented rows use the reconstructed generator. This is a
matched component comparison, not a claim of canonical feature parity. All
June--August, development and later figures are exposed exploratory historical
backtests.
