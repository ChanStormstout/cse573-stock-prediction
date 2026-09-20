# Finite completion of GPT Pro recommendations

This lane follows the owner's explicit resume after the representation study. It is a **random-split exploratory historical experiment**, not a future-period evaluation and not the proposed faculty-slide relative-news-fact correction.

## Mechanism map

| Question | Matched test |
|---|---|
| Did the old word bag discard useful phrases? | UNI vs BIGRAM, identical vocabulary cap, C grid and sigmoid calibration; optional NB_BIGRAM |
| Does nonlinear mapping before averaging preserve a useful news distribution? | K_AFTER vs K_SET, same RFF, article weights, SVC and calibration |
| Could semantics matter only in a price state? | K_SET vs K_INTERACTION; no main-effect gate blocks execution |
| Is a pretrained small-table learner useful independently? | LR/RF/TabPFN on the same compact table, each with/without semantics |
| Can semantics safely make a small correction? | RBASE vs R16, nested cross-fitted offsets, OFF candidate, exact fallback |

K_WP is the new same-solver word/price control. It is not silently equated with the old calibrated LinearSVC. The explicit concatenation implements a sum kernel; tensor-product features implement the semantic-kernel × state-kernel interaction. This is not full SimpleMKL optimization.

The table holds up to128 train-selected phrase/word columns,16 price columns,2 metadata columns and optionally32 train-PCA semantic columns. Fixed TabPFN2.5 conditions on labeled task examples. It has no new task-gradient finetuning; its existing default checkpoint includes real-data finetuning, so it is not called synthetic-only.

## Reproduction

Use the existing primary runtime. The TabPFN worker isolates the prior local sklearn1.7 runtime; other models use the primary sklearn1.9 environment. No new dependency installation, paid API or encoder calls are needed.

```sh
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/test_contracts.py
# prepare.py is first-run only and refuses existing output directories.
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/prepare.py
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/run.py lexical
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/verify.py lexical
```

Repeat run/verify for kernel, table and residual. Alternatively `run_stage.py` uses per-stage locking and the exact frozen fitting functions, allowing independent stages to run concurrently. Do not start two executors for the same stage.

```sh
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/report.py
work/stock-data/finbert-env/bin/python3 outputs/stock_pro_completion_4h/final_audit.py
```

Report requires all four no-fit verifiers to pass. Private raw text, feature caches, training membership bundles, conditioning examples and all weights remain outside Git. A fresh checkout therefore needs the owner's authorized local course data and original caches; published result CSVs are readable without those assets.

## Evidence boundaries

These periods have been exposed repeatedly. Random folds allow shared events and adjacent market periods across training and evaluation. Prior association-group tests were substantially weaker; these results must not be advertised as prospective trading accuracy. Read historical chronological reports separately.

TabSTAR, learned retrieval, new LLM runs, attention and LoRA are later-stage suggestions, not silently counted as completed here. No human extraction-quality acceptance is claimed.

Primary mechanism sources: [Baselines and Bigrams](https://aclanthology.org/P12-2018/), [Support Measure Machines](https://papers.nips.cc/paper_files/paper/2012/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper.pdf), [Random Features](https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf), [SimpleMKL](https://jmlr.org/papers/v9/rakotomamonjy08a.html), [TabPFN official implementation](https://github.com/PriorLabs/TabPFN). These are component inspirations, not claims of complete paper reproduction.
