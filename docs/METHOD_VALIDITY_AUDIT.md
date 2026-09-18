# Method Validity Audit

This is a persistent truth table, not a leaderboard. September onward and all
other exposed periods are exploratory historical backtests. `VALID` means the
implementation supports the stated narrow claim; it does not mean independent
generalization has been proven.

| ID | Project method | External paper/model | Actual implementation | Fidelity | Code status | Result status | Safe claim | Unsafe claim | Repair status |
|---|---|---|---|---|---|---|---|---|---|
| F0 | price + title baseline | none | canonical price features plus title features, chronological LR | DIRECT_BASELINE | `outputs/stock_baseline/run_baseline.py` | VALID | course title baseline is a direct project reference | F0 is state of the art | unchanged |
| F1 | price + full text | none / TF-IDF tradition | sparse full-news features plus price LR, past-only C selection | DIRECT_BASELINE | `outputs/stock_paper_methods_4h/run.py` J0 | VALID_WITH_CAVEAT | fixed full-text classical comparison | every wording is semantically understood | unchanged |
| F2 canonical FinBERT | price + frozen FinBERT | ProsusAI/FinBERT | frozen article vectors, canonical J2 PCA16/article mean/metadata, price LR | FOUNDATION_MODEL_PROBE | `outputs/stock_paper_methods_4h/run.py` J2 | VALID_WITH_CAVEAT | frozen FinBERT representation probe | reproduced a stock-prediction paper or deployment | unchanged |
| R1 | recent price fallback | none | recent five-minute/price feature LR with canonical fallback | DIRECT_BASELINE | `outputs/stock_nextgen_4h` and matched controls | VALID | tested recent-price baseline | recent price is universally predictive | unchanged |
| recency R1 | half-life weighted price | recency inspiration | same LR with infinity/80/40/20 XNYS sample weights | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/run_recency_v4.py` | VALID_WITH_CAVEAT | tested recency weighting failed the preregistered cross-stock gate | recency method is invalid in general | Stage 1 finalize |
| recency F1 | half-life weighted sparse text | recency inspiration | weighted J0 text classifier, separate equal/weighted R1 fallbacks | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/run_recency_v4.py` | VALID_WITH_CAVEAT | v4 tested canonical J0 with explicit fallback modes | paper recency model was reproduced | Stage 1 finalize |
| recency F2 | half-life weighted FinBERT | recency inspiration | v4 J2 canonical transform plus sample weights | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/run_recency_v4.py` | VALID_WITH_CAVEAT | v4 corrected F2 recency probe did not pass gate | old v3 F2 delta was recency-only | Stage 1 finalize |
| dense price windows | 30-minute stride dense bars | dense sampling idea | 48 contiguous five-minute bars, reconstructed matched control when parity fails | MATCHED_COMPONENT_TEST | `outputs/stock_recency_dense_4h/build_dense_windows_v4.py`, `run_dense_v4.py` | VALID_WITH_CAVEAT | small matched dense-window comparison | windows are independent/non-overlapping | Stage 2 wording/parity |
| Fin-ModernBERT | modern financial encoder | Fin-ModernBERT | frozen title encoder, PCA/LR downstream | FOUNDATION_MODEL_PROBE | `outputs/stock_foundation_4h/encode_modern.py`, `experiment.py` | NEEDS_RERUN | historical v1 is an unmatched frozen encoder probe | reproduced Fin-ModernBERT stock forecasting | Stage 3 shared pooling |
| FinBERT + Modern 8+8 | dual representation | ensemble idea | PCA8 concatenation with price LR | MATCHED_COMPONENT_TEST | `outputs/stock_foundation_4h/experiment.py` | NEEDS_RERUN | dual-encoder probe only once pooling is matched | ModernBERT fusion is validated | Stage 3 conditional |
| Chronos-2 | time-series foundation model | Chronos-2 | frozen endpoint/quantile features, prediction_length 49, matched 512-bar LR | FOUNDATION_MODEL_PROBE | `outputs/stock_foundation_4h/chronos_experiment.py` | VALID_WITH_CAVEAT | tested frozen Chronos endpoint features lacked stable gain | Chronos-2 is ineffective for stocks | claim correction only |
| TabPFN 2.5 | tabular foundation model | TabPFN 2.5 | local conditional classifier, `n_estimators=1`, checkpoint provenance not yet verified | FOUNDATION_MODEL_PROBE | `outputs/stock_goal60_4h/tabular.py` | VALID_WITH_CAVEAT | reduced-compute local TabPFN probe | synthetic-only/full standard TabPFN failed | Stage 4 audit |
| masked-reconstruction SSL | price SSL pilot | TS2Vec related work | small Conv1d masked reconstruction, frozen LR representation | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_ssl_4h` | VALID_WITH_CAVEAT | this pilot did not give stable gains | TS2Vec failed | claim correction only |
| Event Adapter A1/A2 | task-adapted event extraction | Ding event/graph work related | FinBERT evidence/action heads, top layers or q/v LoRA | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_finbert_event_adapter_4h` | VALID_WITH_CAVEAT | provisional extraction improved, narrow event downstream did not | Ding event embedding was reproduced/failed | claim correction only |
| historical analogy P1/P2/P3 | historical-news analogy | FinSeer related motivation | HashingVectorizer/coarse category, max three cases, frozen Qwen | PAPER_INSPIRED_SIMPLIFICATION | `outputs/stock_analogy_4h/v2` | VALID_WITH_CAVEAT | conservative lexical analogy with outcomes lacked stable increment | FinSeer was reproduced/failed | claim correction only |
| Qwen direct forecast | direct LLM direction | none | frozen Qwen3.5-9B conditional UP/DOWN token preference | FOUNDATION_MODEL_PROBE | `outputs/stock_llm_direct_4h` | VALID_WITH_CAVEAT | token preference probe had poor calibration/stability | output is true probability | claim correction only |
| historical-news H0/H1/H2 | past-news state | none | prior article/event state and price reaction variants | MATCHED_COMPONENT_TEST | `outputs/stock_goal60_4h/history*.py` | VALID_WITH_CAVEAT | historical state inputs did not show stable gain | H1 solves AMZN missing information | wording audit |
| cross-stock P_cross | peer-price state | cross-asset feature idea | past-only peer return/volatility features at cutoff | MATCHED_COMPONENT_TEST | `outputs/stock_goal60_4h/history_models.py` | VALID | tested two-stock past-state features lacked stable gain | future peer returns were used | wording audit |
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
