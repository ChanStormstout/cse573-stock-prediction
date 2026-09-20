# CSE 573 股票方向预测：实验与决策日志

## 2026-09-19 — ECNI Stage E1 external-data qualification

- **Frozen sample:** preregistered and pushed the E1 protocol before acquisition,
  then froze 7,711 deterministic FNSPID records across both official news files,
  17 years, 358 source labels and 55 ticker tags before interpreting quality.
- **Observed:** 96.99% of sampled timestamps are date-only; combined body
  availability is 39.66%. The provisional 600-record target-association rate is
  52.83% (95% Wilson interval 48.83%–56.80%). Duplicate diagnostics retain 198
  exact normalized-title groups and 2,285 near-title edges as dissemination
  evidence without collapsing articles.
- **Price:** the complete official archive matched its frozen SHA-256. It contains
  7,693 symbol files; 3,172 meet the strict 2018–2023 availability/integrity
  screen. No return or direction statistic was computed.
- **Decision:** `CONDITIONAL_DATA_REDESIGN_REQUIRED`. FNSPID is
  `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS`; a conditional hybrid `ARCH_C`
  keeps facts unknown unless body, time and target checks pass. Exact per-stock
  training coverage and historical sector identity were unavailable, so the
  named panel and company split remain explicitly unfrozen rather than guessed.
- **Boundary:** zero stock-prediction fits, zero BA/MCC/Brier, zero return-based
  selection, zero reader training, and no inspection or summary of locked 2023
  outcomes. The next prerequisite is a metadata-complete news index plus a
  versioned issuer/sector mapping.

## 2026-09-19 — External benchmark audit and ECNI data foundation

- **Scope:** opened a new outcome-blind research lane after freezing V10. No stock-direction estimator, reader model, FinBERT/Fin-ModernBERT prediction, BA, MCC, Brier or return-based selection ran.
- **Evidence:** inspected primary repositories/papers and bounded samples for course data, CMIN-US, FNSPID, FinMultiTime, EDT, StockNet and SEC EDGAR. Recorded source revisions and sample hashes without publishing copyrighted article bodies.
- **Decision:** FNSPID is a conditional primary daily-panel candidate, subject to a larger point-in-time/text audit; CMIN-US remains the standard official benchmark; EDT is auxiliary event-reader supervision. FinMultiTime is blocked because the inspected author dataset revision exposed images but not raw U.S. news/prices.
- **Protocol:** froze candidate ECNI statement, factual-innovation and dissemination schemas; an outcome-blind balanced sector/coverage panel and unseen-company split; candidate time locks; reader-label controls; fixed-token encoder comparisons; low-capacity prediction and usable-information evaluation specifications.
- **Boundary:** named panel members, final time split and predictive execution remain unauthorized until full metadata acquisition and audits satisfy the manifest blockers.

## 2026-09-19 — V10 Stage B2 frozen NEWS+PRICE execution and classical-lane freeze

- **Pre-result repair boundary:** the original B2 checkpoint `4e2428e` failed at the canonical daily merge before any fit. The guarded many-to-one repair was committed at `8cde819`; its post-commit preflight exposed only a dynamic file-set ledger problem. The final static-ledger/schema repair was committed at `5bbd7d5`. The original 160-file ledger, 576 NEWS model hashes, 24 DPRICE model hashes, and explicit inputs all matched; the V2 fit-free preflight passed with zero joint models.
- **Execution:** the single authorized frozen run executed 18 branches and 108 monthly models with zero candidate-grid fits. It produced 1,827 four-hour and 1,248 daily predictions. No scientific parameter, feature, window, classifier, or comparison rule changed after the first fit.
- **Independent verification:** all 108 models were independently refit and all 3,075 rows replayed. Row-key and direction mismatches were zero; maximum probability error was `1.1102230246251565e-16`; manifest mismatches and computed fallback overrides were zero. Future-text and future-price perturbation tests passed, and protected artifacts remained unchanged. `STAGE_B2_FINAL_AUDIT.json` is PASS.
- **Scientific result:** no four-hour NEWS+PRICE method had positive BA increment versus R1 in all four stock/phase cells, and none strictly dominated historical F1 across those four cells. No daily method had positive BA increment versus DPRICE in all four cells; the frozen 24-hour KNN pattern and matched PAPER_2G 24-hour-over-overnight pattern both failed. These development/later values are exposed historical backtests and no branch was promoted after inspection.
- **Outcome:** the classical lane is frozen complete. The durable outputs are `STAGE_B2_REPORT.md` and `CLASSICAL_LANE_FINAL_SUMMARY.md`; private model binaries remain outside Git.

## 2026-09-19 — V10 Stage A2 independent V9 NEWS-only reconstruction

- **Purpose:** establish whether the frozen V9 full-grid NEWS-only evidence can be independently reproduced before any Stage B work. The audit did not rank exposed September-and-later candidates or interpret predictive winners.
- **Input contract:** nine frozen V9 evidence files were byte-identical to commit `b7ee2af9829143707ec309e0ec6809935524324b`. Independent raw reconstruction reproduced all 1,607 canonical four-hour article memberships and full-body `stem_body` fields; the daily raw reconstruction had 1,072 rows and exactly matched the retained private daily cache.
- **Selection and replay:** an independent implementation reconstructed 192 four-hour and 384 daily issued parameters from physically isolated March--August grid rows, with zero mismatches and no September-plus freeze violation. It replayed 348 four-hour plus 696 daily authorized candidate rows. The largest metric discrepancy was `1.11e-16`, entirely floating-point rounding.
- **Model evidence:** 576 V10-reconstructed private model bundles reproduced V9 row-level probabilities/scores at no more than `4.44e-16`, with zero direction mismatches; every reload produced an exact score match. Future-text perturbation passed for PAPER_1G_L1LR and TFIDF_LR, and all 576 training boundaries were strict.
- **Boundary:** `STAGE_A_FINAL_AUDIT.json` is PASS. No DPRICE, NEWS+PRICE, method-family selection, Market Context, or relation-reader action ran. V6/V7/V8/V9, F0/F1/F2 and existing artifacts remain unmodified. This authorizes only external review of Stage A, not automatic Stage B execution.

## 2026-09-19 — V10 Stage A2 final verifier-only repair

- **Repair:** the historical audit's literal quarantine Boolean was replaced by a computed record-level isolation audit. It checked all 576 decisions, their expected month histories, and 8,874 referenced authorized grid rows; zero September-plus candidate rows were referenced.
- **Final-artifact check:** the verifier read, hashed and reloaded the final 576 manifest-hashed joblib files without fitting or writing any model. All manifest hashes matched; maximum probability and LinearSVM score errors versus frozen V9 predictions were `1.11e-16` and `4.44e-16`, respectively, with zero direction mismatches.
- **Integrity:** every private model file had identical before/after bytes. All 32 tracked V6--V9 artifacts were also identical before/after. `STAGE_A_FINAL_AUDIT.json` is preserved; `STAGE_A_FINAL_AUDIT_V2.json` is PASS.
- **Boundary:** no candidate-grid rerun, refit, DPRICE, NEWS+PRICE, method-family selection, Market Context, or relation-reader work occurred. This is an external-review checkpoint; it does not start Stage B.

## 2026-09-19 — V10 Stage B1 DPRICE and frozen NEWS-only family selection

- **DPRICE source contract:** raw teacher daily charts and the XNYS schedule yielded 536 distinct stock-session targets: 70 warmup, 258 March--August OOF, 84 development and 124 later. Every row used exactly five completed prior sessions and only `DRET_1`, `DRET_2`, `DRET_5`, `RANGE_1`, `RV_5`, `MEAN_5`, and `HISTORY_AGE_HOURS`.
- **Chronological experiment:** the authorized grid contains 36 rows only (two stocks × six OOF months × three C values). It produced 24 issued DPRICE models. September--February each use the C frozen from March--August evidence for their stock; the training rows expand only with already-ended targets.
- **Verification:** the independent verifier rebuilt daily labels/features, recomputed aggregate metrics, reconstructed the C chronology, and reloaded all final manifest-hashed files. Its maximum probability difference was `8.33e-17`, with zero hash or direction mismatches. Pre-B1 Stage A, V6--V9, canonical F0/F1/F2, Market Context and relation-reader tracked files were unchanged.
- **Method freeze:** all eight text methods were ranked separately for 4h, daily overnight and daily 24-hour using only V9 issued March--August OOF predictions. The top three per window and their already-frozen September-plus parameters are stored for a future B2; no joint model was fitted or evaluated.

## 2026-09-19 — V7 repair stopped before daily prediction

The repair addendum was committed before any v7 work. A stronger v6 verifier
passes canonical row, prediction mapping, timing and training-boundary checks,
with an explicit limitation that v6 does not retain fitted models for an
independent probability replay. Canonical 4h article membership is 1,607/1,607
under the retained reconstruction, but that reconstruction is the prohibited
title matcher. There is no retained all-universe accepted full-text association
generator, so v7 daily fitting and scoring were correctly not run.

## 2026-09-19 — Prior-work clean reproduction and one-day study

Created and committed preregistration before fitting. v1--v5 are preserved
software-abort artifacts; v6 cleanly separates the two daily news windows and
passes its contract verifier. The run uses train-fold-only sparse vocabulary,
IDF and chi-square with chronological selection. It does not alter F0/F1/F2,
Market Context or the relation lane. Results are exploratory historical
backtests, not a live or independent final test.

## Context increment: outcome-blind news relation pilot (2026-09-18)

## Context increment: market production-path repair (2026-09-18)

- **Problem:** the prior synthetic test used `p0` rather than raw R1 features;
  nine listed corruptions shared one generic mutation, so it could not establish
  the production-path contracts.
- **Change:** created a shared R1/Mmeta/M1 core, raw-bar feature builder,
  chronological C selection, independent fit-free verifier, PASS-gated reporter,
  and versioned fault-specific synthetic fixture.
- **Observation:** clean synthetic `prepare → run → verify → report` passed;
  all fourteen distinct mutations were rejected by their named expected checks.
- **Boundary:** no real SPY/QQQ bars were acquired, no real Mmeta/M1 score was
  generated, and the Qwen relation lane was not changed.

## Context increment: final market integrity freeze (2026-09-18)

- Connected the future real entrypoint without running it.
- Strengthened source hashes, R1 parity binding, verifier independence and
  synthetic corruption coverage in `audit_v4`.
- No real source was acquired or scored.

## Context increment: real verifier path repair (2026-09-18)

- Repaired only canonical-path, canonical-session and final-C verifier logic.
- Audit v5 clean synthetic run passed; the final-C fault failed its dedicated
  check. No market prediction was run.

## Context increment: canonical R1 row binding (2026-09-18)

- Added a real-mode independent row/window contract and a cutoff-only fault.
- Audit v6 passes cleanly; cutoff shift fails the dedicated contract.

## Context increment: authenticated market-source preflight (2026-09-19)

- The permitted Alpaca SIP/raw 5-minute access probe found no supported
  credential pair and recorded `AUTH_REQUIRED` with zero attempts.
- Stopped before acquisition, source materialization, coverage, or model work.

- Full title-and-body association and target-local family isolation produced 64
  pairs, with no family crossing pilot and locked-check splits.
- The frozen Qwen3.5-9B pass received association-centered article sentences only:
  no prices, returns, labels, reactions, or future articles.
- All 64 outputs were written; 63 passed mechanical JSON/evidence/time validation.
  The invalid output is retained without repair or rerun.
- These are provisional model relation labels, not human gold or prediction scores.

维护日期：2026-09-15。用途：给组员解释我们尝试过什么、为什么改变、观察到了什么，以及下一步决定的依据。目标是完成课程项目与有说服力的实验，不以发表新算法为前提。

## 使用约定

- 本文件是主日志。后续实验继续追加编号，不覆盖失败或无提升的结果；纠正旧结论时注明新证据。
- 每次记录：问题 → 已有观察 → 假设 → 改动与保持不变的部分 → 实验结果 → insight → 下一步。
- **观察**是数据或实际运行结果；**解释/假设**是可能原因；**决定**是行动选择。三者分开写。
- 训练集用于拟合；验证集用于有限选参及开发比较；最终测试集用于方法冻结后的最终评估。验证集已经参与选参，不能冒充无偏最终成绩。
- 新闻窗口重叠、相邻小时相关；小时样本数不等于独立实验次数。
- 文中已有数据审计与基线属于回填记录，来源是先前保存的文件；E03 协议在本轮 FinBERT 推理和结果产生前写入。

## 实验目录

**2026-09-15 范围核对：** 用户询问当前工作是否属于项目要求。已重新核对课件第 3–5、20–21 页及股票 guidelines。传统与现代方法比较、时间对齐、系统演示和报告属于项目主线；E05–E07 是团队选择的改进探索，不能表述为老师指定必做。暂停新增复杂模型实验，优先整理已有结果、形成主模型与原型、对齐课程交付。详见 [要求与范围核对](PROJECT_SCOPE_CHECK.md)。

**本轮执行快照：** E05 每股六组正文表示实验、E06 每股四组去重实验已完成，独立检查通过；E07 两个版本各对同一批前 20 条进行了本地推理并因质量不足停止，没有全量推理。最终测试未评估。详细诊断备注与 E05–E07 完整分析报告仍待整理；本条是实际状态快照，不替代完整实验条目。

| 编号 | 尝试 | 状态 | 核心问题 |
|---|---|---|---|
| E00 | 原始数据审计 | 完成，回填 | 数据是否足以支持小时预测？ |
| E01 | 时间安全的样本构造 | 完成，回填 | 每次预测真正能看到哪些信息？ |
| E02 | 价格与 TF-IDF 基线 | 完成，回填 | 简单模型能达到什么水平？ |
| E03-A | 训练期新闻标题抽查 | 完成 | 标题命中公司是否等于有效信息？ |
| E03-B | 冻结 FinBERT 与数量对照 | 完成，含一次记录定位修复 | 金融情绪表示是否比词频更有用？ |
| E04 | 论文方法、时效与语义向量 | 完成 | 有哪些性能瓶颈？ |
| E05 | 标题、等长正文开头、公司相关句 | 对照完成；质量标签部分完成 | 公司相关句是否更有效？ |
| E06 | 近似转载与数量对照 | 对照完成 | 去重是否改善预测？ |
| E07 | 规则事件与三版本地模型试验 | 规则对照完成；LLM 未过质量条件 | 结构化事件是否有帮助？ |
| E08-B | 相同信息量 LR/MLP/GRU | 完成；网络各三个种子 | 递归时间模型是否有额外收益？ |

## E00：先检查数据，而不是先训练模型

### 为什么做

股票任务最容易出现的假提升是时间对齐错误：模型读到了目标价格发生后的新闻，或者把完整小时的高低收盘价当作小时开始时就知道的变量。先确认数据的单位、时间和覆盖范围。

### 做了什么与观察

- 全量解析 78,055 条新闻，读取两只股票的 12 个价格文件；新闻英文 70,732 条。
- 11,651 条新闻抓取时间早于发布时间，占 14.93%；其原因不能从文件确定，不能推断老师故意埋坑或认为无所谓。
- 8,917 条新闻抓取比发布晚超过 24 小时。发布时间与可用时间不是同一个概念。
- 全文提到 AMZN 的 18,118 条全部同时提到 AAPL；标题里 AAPL/Apple 31,299 条，AMZN/Amazon 2,442 条。支持“语料偏 AAPL”的推断，但不能证明原始采集策略。
- 用连续 12 根 5 分钟线核对小时 OHLC：AAPL 2,082 个完整窗口、AMZN 2,101 个窗口全部精确一致，支持小时标签表示区间起点。

### Insight 与决定

1. **数据内部一致不代表来源语义全部确认。** 小时起点有证据，但实际交易场所、延长交易时段、第七列含义仍未确认。
2. **先沿用 AAPL、AMZN，不为追求分数挑“稳定股票”。** 波动小不等于涨跌方向容易预测；更换股票会增加数据与选择偏差问题。
3. 选小时作为主要周期，因为能构造更多样本且有跨频率证据；日频作为备用，暂不反复切换。

证据：[原始审计报告](stock_data_audit.md)、[原始文件来源与哈希](stock_baseline/source_manifest.json)。

## E01：把一个样本的“时间合同”写清楚

### 具体任务

例如纽约时间 10:25 做预测，判断 10:30–11:30 这一个完整小时的收盘价是否高于开盘价。10:30 的开盘价是定义目标所需的数据，不进入预测输入。

### 实施规则与原因

- 每个目标小时必须位于 XNYS 常规交易时段，并具有连续 12 根 5 分钟线；避免不完整区间当作完整小时。
- 输入最近 6 个已结束的完整常规小时，可能跨日；以实际结束时间早于预测截止点为准。
- 注意：提前 5 分钟预测意味着紧邻目标的上一小时还未结束，会被跳过。此规则保守，但损失最新信息，是后续可单独检验的设计限制。
- 新闻使用 `(截止时刻−4小时, 截止时刻]`，可用时刻取发布时间与抓取时间较晚值；隔离负延迟。
- 标题匹配公司、英文、正文至少 100 字符，正文和标题按最早可用记录去重；保留无新闻窗口。
- 平盘目标排除，因此评价范围只涵盖非平盘小时，未来演示必须说明这一范围。

### 数据变化

| 股票 | 完整常规目标小时 | 最终训练 | 验证 | 测试留出 |
|---|---:|---:|---:|---:|
| AAPL | 1,627 | 996 | 252 | 364 |
| AMZN | 1,627 | 1,004 | 252 | 365 |

从完整目标小时到最终样本，排除了平盘和历史不足；相比初步候选清单，本版还要求历史价格完整。训练 2018 年 1–8 月，验证 9–10 月，测试 11 月至 2019 年 2 月 1 日。

**Insight：清洗不是单纯减少数据量，而是限定结果到底回答什么问题。** 之后每个特征实验都使用相同样本和时间划分，避免把样本变化误当成模型改善。

## E02：建立传统模型参照

### 问题与逻辑

在引入深度模型前，需要知道“总猜上涨”“只看历史价格”“只看新闻用词”能做到什么。复杂模型必须和这些参照比较，不能只展示一个准确率。

### 方法

- 多数类：按照训练期多数方向预测。
- 价格 LR：6 个历史小时的收益、振幅、收益均值/标准差、历史距截止点时长、纽约时刻。
- TF-IDF + LR：把标题转成词和双词组合的加权频次，再用逻辑回归预测方向。
- 价格 + TF-IDF + LR：组合两类信息。
- 每种 LR 比较预设 C = 0.01 / 0.1 / 1，按验证 balanced accuracy 选，平手时选 Brier 更低者。词表和标准化只在训练集拟合。

### 观察

| 模型 | AAPL 平衡准确率 | AMZN 平衡准确率 |
|---|---:|---:|
| 多数类 | 50.00% | 50.00% |
| 价格 | 51.59% | 50.03% |
| TF-IDF | 50.00% | 51.12% |
| 价格 + TF-IDF | 51.20% | 52.29% |

平衡准确率是上涨召回率和下跌召回率的平均；多数类参照为 50%。MCC 衡量两类预测与实际的整体相关，0 接近无相关；Brier 衡量预测概率的误差，越低越好。

- AAPL 标题模型一直预测上涨；组合未超过价格。价格模型 9 月约 56.8%，10 月约 47.5%，不稳定。
- AMZN 组合有小幅提升，但概率误差 Brier 比多数类差。
- 验证新闻覆盖率：AAPL 99.2%，AMZN 66.7%。两家公司可用文本条件不同。
- 时间边界、训练集缩放/词表、目标列隔离等自动检查通过；没有评估最终测试集。

### Insight、假设与下一步

**观察支持的结论：** 当前结果接近简单参照，不能声称稳定有效。

**待检验假设：** TF-IDF 特征稀疏，对“盈利提高”“成本下降”等不同词表达的相似含义难以共享统计信息；金融领域预训练模型可能提供更紧凑的情绪特征。也可能根本没有足够的一小时预测信号，换模型不会解决。

**决定：** 下一轮只替换文本表示，保持价格特征、样本、窗口和划分不变。额外加入新闻数量对照，区分语义贡献与新闻活跃度。

证据：[基线说明](stock_baseline/README.md)、[全部指标](stock_baseline/results/results.json)、[运行代码](stock_baseline/run_baseline.py)。

## E03-A：训练期标题抽查

### 设计

对每只股票训练期每个月抽取 3 条实际进入样本的新闻，共 48 条；固定随机种子 573。由助手逐条阅读标题并作诊断标记，**不是组员人工标注，也不是经过独立复核的金标准**。每月等额抽样，不能把比例直接推广到整包新闻。

### 观察与案例

- 有“Apple Shareholder … Cut Its Holding …; … Another Company …”这类长持仓标题，兼有不同公司的好坏信息。整体标题情绪未必对应目标公司。
- 有“Amazon's stock jumps above $1,800 for the first time”等已发生价格的描述；并不因为它是金融新闻就能预测未来。
- 有“10 things you need to know before the opening bell”多公司摘要，提及股票代码的程度与实际主体相关性不同。
- 抽样中可见 15 小时以上抓取滞后，甚至 38.55 小时；我们的 4 小时窗口限制的是可用时间，不保证事件本身新鲜。
- 相近标题出现在不同来源、附带不同后缀，精确去重不能清除全部近似转载。

### 抽样统计

48 条中，助手标记了 27 条多公司标题、10 条持仓模板、13 条描述已发生价格走势的标题；类别可重叠，不能相加。11 条抓取滞后超过 12 小时，1 条超过 24 小时。48 条均在后续修复后重新确认实际进入训练窗口。

### Insight 与决定

**实体相关性、事件新鲜度和语义表示是三个不同问题。** FinBERT 只能提供文本情绪表示，不会自动解决前两个问题。为了知道改善来自哪里，本轮不根据这些案例修改清洗规则；先完成同样本表示实验，下一轮可单独比较公司相关句和新闻滞后过滤。

证据：[48 条标题与审阅标记](stock_finbert/news_review.csv)。

## E03-B：冻结 FinBERT 特征实验

### 在结果出现前登记的逻辑

FinBERT 对每条标题输出正面、负面、中性概率。对同一 4 小时窗口求三类概率均值，并取正负差的标准差描述分歧，再交给逻辑回归。FinBERT 不直接输出股价涨跌，也不在本组价格数据上微调。

比较三组：

1. **价格 + 新闻数量对照**：价格、log(1+新闻数)、是否有新闻。检查单纯新闻活跃度。
2. **FinBERT 文本**：三类均值、情绪分歧、数量与是否有新闻。
3. **价格 + FinBERT**：组合以上变量，与 E02 价格、TF-IDF 以及本轮数量对照比较。

无新闻时情绪变量设 0，同时显式提供是否有新闻，避免与真正的中性新闻混淆。C 网格沿用 E02，不因看到结果扩大搜索。最终测试行不进入本轮 FinBERT 推理。

### 模型来源与历史边界

采用 `ProsusAI/finbert`，固定 revision `4556d13015211d73dccd3fdd39d39232506f3e43`。作者说明该模型基于 BERT，在金融语料上继续训练并用 Financial PhraseBank 做情绪微调。[模型卡](https://huggingface.co/ProsusAI/finbert) / [作者仓库](https://github.com/ProsusAI/finBERT)。

这是一项将后来的预训练模型用于 2018 年数据的回顾性特征实验，不是证明 2018 年当时可以部署的交易回测。未完成对全部预训练语料与本组新闻重合的审计。情绪分类能力不等于价格预测能力。

### 执行与结果

已完成本地冻结模型推理和逻辑回归训练。下面只使用修复记录定位之后的有效结果。

#### 先记录一次发现问题与修复的过程

- 初版错误地把 UUID 当成唯一记录键。补做逐条新闻窗口检查时发现同一个 UUID 存在不同月份、不同标题版本。
- 示例：同一 UUID 的 1 月标题是 “Why Apple’s new campus and Amazon’s HQ2 probably won’t be neighbors - MarketWatch”，2 月版本略有改写；按 UUID 取最早记录会替换掉基线真正使用的版本。
- 共有 2 条标题版本被错取，影响 AAPL 4 个训练窗口与 4 个验证窗口、AMZN 4 个训练窗口；并非所有窗口都有问题。
- 修复：使用 `archive::member` 作为唯一记录键。按 E02 原规则重建每个新闻窗口，并断言标题拼接、UUID 序列和新闻数量全部与原基线一致，再重新推理。
- 初版结果保留在 `stock_finbert/results_uuid_invalid/`，有明确作废标记，不用于结论；原始数据和 E02 基线文件未改变。
- 修复后方向分类指标未变，但概率出现小幅变化。这不能成为保留错误实现的理由。
- **实现 insight：业务 ID 不一定等于一条不可变观察。跨阶段传递特征应使用准确的记录版本，而不是只用可能复用的 ID。**

#### 数据变化与验证

- 实际编码 5,783 条不同记录；模型最大长度 256 token，本次截断标题 0 条。冻结模型在本机 MPS 上运行，没有把新闻发给在线推理服务。
- 样本数、标签、划分、新闻窗口与 E02 完全一致：AAPL 996 训练 / 252 验证；AMZN 1,004 / 252。未对测试行提取特征或评估。
- 有效结果保留记录键，能够追溯到原始月度归档。
- 检查：基线文件哈希未变；开发样本顺序一致；逐条新闻在时间窗口内；各窗口均值与分歧聚合正确；概率和为 1；标准化只用训练集；模型输入不包含结果列。

#### 结果：验证集平衡准确率

| 模型 | AAPL | AMZN |
|---|---:|---:|
| E02 价格 | 51.59% | 50.03% |
| E02 价格 + TF-IDF | 51.20% | 52.29% |
| E03 价格 + 新闻数量 | 51.11% | 51.20% |
| E03 FinBERT 文本 | 48.07% | 49.12% |
| E03 价格 + FinBERT | 50.38% | 51.20% |

#### 观察与解释边界

1. **语义模型没有超过 TF-IDF 组合。** AAPL 价格 + FinBERT 为 50.38%，AMZN 为 51.20%，都低于各自 E02 价格 + TF-IDF。当前只有这一种 FinBERT 聚合与小网格，不能推广成“所有语义模型都无效”。
2. **AMZN 的整体分数不能证明语义增益。** 加 FinBERT 与只加数量均为 51.20%；两者逐小时预测并不相同。FinBERT 组合的 Brier 约 0.2531，数量对照约 0.2520，概率误差也未改善。
3. **月份表现会抵消。** AMZN 价格 + FinBERT 的 9 月平衡准确率约 55.46%，10 月约 46.47%；数量对照分别约 50.56%、49.71%。这支持“不稳定”的描述，不能说明 9 月上涨由情绪因果驱动。
4. **小样本下的几个百分点很不确定。** 每只股票验证集只有 42 个交易日、252 小时。按交易日整体重采样 1,000 次，保留同日小时相关性，得到如下描述性区间：

| 股票 | FinBERT 组合减去价格的 BA 差值，95% 区间 | FinBERT 组合减去数量对照的 BA 差值，95% 区间 |
|---|---:|---:|
| AAPL | -5.58 至 +2.84 个百分点 | -5.02 至 +3.19 个百分点 |
| AMZN | -2.15 至 +4.64 个百分点 | -3.58 至 +3.17 个百分点 |

这些区间包含 0。它们没有纠正验证集选参偏差，也没有完整建模跨日相关性，不能当作最终显著性检验。

#### 本轮 insight

- **金融情绪与下一小时股价方向之间还隔着一层映射。** 标题“股票已经上涨”可能是正面情绪，却只是在描述过去；即使 FinBERT 情绪判断正确，预测仍可能失败。此处是机制解释的假设，不是本轮因果验证。
- **复杂特征需要简单对照。** 若没有新闻数量对照，很容易把 AMZN 相比纯价格约 1 个百分点的差异归功于语义理解。
- **平均情绪会丢失对象和事件信息。** 多公司标题与长滞后是已观察到的问题；它们是否解释低分，需要单独实验，不能仅凭几个案例下结论。
- **没有提升也是可写进课程报告的结果。** 我们已经具备传统模型、冻结预训练模型、数量对照、月份分析和可复核的失败诊断；项目价值来自清晰实验与诚实解释，不需要把 50% 左右包装成成功。

#### 下一步决定（尚未执行）

先完成一项数据层对照，再考虑更大模型：固定样本与模型，单独检查去除“发布到抓取超过 4 小时”的新闻是否改变结果；保留因此变成无新闻的样本。阈值对应当前预测新闻窗口，是待检验假设，不是已确认最优值。另一个独立方向是提取目标公司相关句，避免多公司整体情绪混合；两项改动不合并，才能分辨贡献。

继续保留同一最终测试集；不因本轮结果差而挑股票、扩大参数搜索或提前看测试分数。

证据：[有效指标](stock_finbert/results/results.json)、[记录键修复审计](stock_finbert/uuid_fix_audit.json)、[自动检查输出](stock_finbert/check_result.txt)、[本轮复现说明](stock_finbert/README.md)。

协议：[protocol.json](stock_finbert/protocol.json)。代码：[run_finbert.py](stock_finbert/run_finbert.py)。

---

## 后续记录模板

### E__：名称 / 日期 / 状态

- **问题：** 上一步有什么未解释的观察？
- **假设：** 为什么此改动可能有效？什么结果会反驳它？
- **改动：** 特征、数据、模型或评估哪里变了？
- **保持不变：** 控制哪些因素，使结果可比较？
- **数据变化：** 总量、各划分、覆盖率、缺失和剔除原因。
- **协议：** 参数、模型版本、随机种子、评估指标、测试集使用情况。
- **实际结果：** 指标表、月份/股票差异、错误案例及证据文件。
- **Insight：** 观察支持什么，不支持什么？可能原因有哪些？
- **决定：** 保留、放弃或继续验证什么；为什么。

## 阶段评估：改进是否足够、为什么需要 case study（2026-09-15）

### 判断

当前足以构成阶段性实验，尚不足以直接作为最终项目完成。课件第 3 页要求同时包含传统 DM/ML 与现代 AI/深度学习特征和算法；LR、TF-IDF 与冻结 FinBERT 已覆盖这项方法组合。课件还要求演示原型、完整报告与代码、成员贡献说明，不能仅凭模型数量判断完成度或保证分数。

应将 E03 称为“改进尝试”：有效验证结果没有证明预测性能提升。课程实验不必每一步都涨分，但需要解释失败、做有针对性的验证并完成最终评价。

### 当前缺口

1. 48 条标题抽查解释了语料里有什么，但尚未连到“这一预测时刻的输入 → 情绪特征 → 模型概率 → 实际方向”。目前不能断言旧新闻或多公司标题是失败主因。
2. 冻结情绪均值表示可能丢失目标公司、事件类型和时间顺序；这是设计限制，其影响仍待实验。
3. AMZN 整体数量对照与 FinBERT 组合分数相同，月份表现不同；还缺少逐样本分析来说明哪些预测变好、哪些变坏。
4. 验证集参与选参，尚无最终测试成绩；现阶段没有稳定提升证据。

### 下一项建议：案例驱动诊断（计划，尚未完成）

以价格 + 新闻数量对照与价格 + FinBERT 的验证预测为参照，分别检查：两者都对、数量对照对而 FinBERT 错、数量对照错而 FinBERT 对、两者都错。每只股票每类最多固定种子抽 2 个不同交易日窗口，最多 16 个案例；不足则如实保留数量，不强行凑齐或只挑有故事的案例。案例选择属于诊断抽样，不估计总体错误比例。

每例保存：预测起止/截止时间、所有相关新闻的准确记录键、发布与抓取时间、新闻数量、主要标题及正文相关段落、FinBERT 情绪概率、两个股票模型的上涨概率、实际小时收益和方向。解释时区分情绪分类错误、目标对象混合、历史事件/滞后、聚合抵消、以及情绪看似正确但价格反向。最后一类不能简单解释成模型“理解错误”，也不能推断新闻导致价格变化。

这些案例用来提出假设，不能证明某类问题普遍存在或因果成立。根据最有支持的诊断，只执行一个数据层改动，并在全验证集做量化对照。比较语义贡献时还应检查相同 C 下的数量对照，避免把正则化选择差异全部归因于语义特征。

### 合理的完成标准

保留已有基线与 FinBERT；补一轮系统案例分析、一项有依据的控制实验和解释；冻结方案后统一做最终测试；完成可演示流程、报告和组员贡献。若控制实验仍无提升，保留负结果并解释适用边界，不继续无限换模型或搜索到高分为止。

## E04：回到论文、扩大案例、检验性能瓶颈（2026-09-15，已完成）

### 为什么改进方向需要重新核对

用户指出论文落实不清楚、16 个案例覆盖不足，并要求深入分析性能。这一轮以此前原始调研清单为准核对实现，扩大案例并实际运行有限控制实验。此前“最多 16 例”计划由下面的新范围替代，历史计划保留。

### 论文与代码之间的差距

- 指定论文 Alostad & Davulcu：此前只有 LR 和时间聚合思路，尚未实现二值词袋 + Chi-square + L1；E04 已补这一组合及价格融合，但标题、预处理和时间验证协议与原论文不同，不能叫完整复现。目标公司正文句提取与 Twitter 突增仍未实现。
- FinBERT：E03 已用情绪头；E04 增加实际最后隐层语义表示与训练期 PCA16。
- StockNet：仅借鉴价格与文本联合输入。E04 固定时间权重也不等于其注意力/变分结构。
- CausalStock：未实现 LLM 五维打分、因果图或功能因果模型。之前只借鉴降噪动机。

### 案例覆盖变化与实际审阅范围

| 层次 | 覆盖 | 完成内容 |
|---|---:|---|
| 全量诊断 | 504 个验证窗口 | 两只股票、月份、时段、新闻数量/滞后、涨跌幅、正确/错误组合 |
| 分层案例卡 | 96 个窗口 | 每只股票每月 24 个，完整输入可追溯；程序生成 |
| 助手逐例审阅 | 24 个窗口 | 实际查看概率、结果、最新最多 3 条标题 |
| 正文进一步核对 | 8 个案例 / 7 条不同记录 | 查看原文摘录以确认对象与信息遗漏 |

不把自动卡片称为 96 例人工深读；也没有组员独立复核。案例中翻转方向共 50/504：24 个改善、26 个恶化。全部统计才能描述总体，诊断抽样不能估计错误原因比例。

### 从案例得到的具体 insight

1. **输入对象错误比“模型够不够大”更值得先查。** AMZN-104 标题是分析师升级/降级汇总，FinBERT 给负面；正文 Amazon 段却是维持 Buy 并上调目标价。
2. **同一文本可对不同公司含义不同。** 苹果/谷歌追赶 Amazon 的标题，在 AAPL 与 AMZN 窗口得到相同情绪特征；当前表示没有区分谁领先、谁落后。
3. **精确去重不够。** 高通指控事件多个相近标题、旧苹果上涨报道不同后缀均进入同窗。
4. **情绪判断与价格判断应分开。** “Meh”标题疑似情绪误判，但对应股票预测却改对；不能从预测对错反推情绪是否正确。
5. **微小涨跌与旧新闻不能被直接当作主因。** 小于 0.1% 的目标仅 89/504；AMZN 旧新闻较多切片反而表现较好。切片有关联，不能断言因果。

### 实验设计与变更理由

固定原样本、标签、窗口和测试留出；将 C 选择改为训练期内 6/7/8 月三次向前验证，候选仍为 0.01/0.1/1。同协议重跑价格参照，所以 E04 与旧 E03 的选参后分数不能直接混比。词表、Chi-square、PCA 和缩放都在每个训练折内拟合。

预先登记 9 组方法；初步结果检查发现论文式组合也改变了正则化，随后明确登记追加价格 L1 对照，最终 10 组。该追加发生在初步结果之后，不冒充预注册。

### 数据变化

- 样本数始终不变：AAPL 996 训练 / 252 验证；AMZN 1,004 / 252。
- 从已结束 5 分钟线补充近端价格后，最新价格信息距截止点的中位数由 55 分钟变为 0；这依赖线结束时可获得的假设，源端传播延迟未知。
- 过滤抓取滞后 >4h 的新闻后，AAPL 验证新闻覆盖 99.2%→96.8%，AMZN 66.7%→39.7%；因此噪声减少与信息损失同时发生。
- 原窗口训练新闻“出现次数”（包括跨窗口重复使用）AAPL 13,167→9,642，AMZN 1,080→558；不是唯一文章数。

### 有效验证结果：平衡准确率

| 方法 | AAPL | AMZN |
|---|---:|---:|
| 价格 L2 | 51.20% | 48.87% |
| 价格 L1 对照 | 51.90% | 49.26% |
| 加最新可用价格 | 51.21% | 50.86% |
| 价格 + 新闻数量 | 49.17% | 49.26% |
| 价格 + 情绪 | 49.99% | 50.84% |
| 过滤滞后 + 情绪 | 48.77% | 50.87% |
| 时间加权情绪 | 48.74% | 49.26% |
| 价格 + 语义向量 | 50.13% | 47.16% |
| 二值词袋 + 卡方 + L1 | 49.75% | 47.64% |
| 价格 + 二值词袋 + 卡方 + L1 | 52.72% | 50.03% |

### 怎么解释改善和失败

- AAPL 论文式组合 52.72%，相对价格 L1 51.90% 仅约 +0.82 个百分点；按日重采样 95% 描述性区间约为 -3.13 至 +4.93 个百分点，仍包含 0。
- 它的 9/10 月 BA 分别 53.24% / 52.67%，比此前波动小，但这是开发观察，未经最终测试验证。
- AAPL 组合中只有 3 个非零文本词：bought、marketwatch、sold；AMZN 只剩 apple。存在来源/格式依赖的可能，不能归功于充分理解事件。
- 语义向量、时间加权没有跨股票稳定提升；更新价格信息也未带来一致改善。更丰富或更新的输入并不自动包含稳定预测信号。
- 参数只在训练期选择后，部分原先好看的验证结果下降。泛化稳定性是需要解决的问题；继续在 9–10 月调参会让这些月份逐渐变成训练反馈。

### 下一步的性能路线（待执行）

**E05：目标公司相关正文。** 保持原文章和样本集合，提取有原文证据的目标公司句/段与必要上下文，与标题做同模型对照。训练期制定规则，检查多公司对象与上下文，找不到时回退并标记。首先检验输入是否更准确，再检验预测是否改善。

**E06：事件去重与表示。** 时间顺序处理近似转载，保留独立来源数；提取主体、事件、对象、回顾/预告及原文中有证据的实际/预期差异。不要同时更换多个部分，逐项消融。

**评估完成条件：** 相同任务和覆盖下优于恰当基线；跨月稳定；概率误差不明显恶化；控制实验支持对应改动；方法冻结后最终测试仍成立。当前没有任何路径能保证显著提升。

### 证据与复现

[论文落实与完整性能分析](stock_improvement/ANALYSIS.md)；[扩展案例审阅](stock_diagnostics/QUALITATIVE_REVIEW.md)；[96 张案例卡](stock_diagnostics/CASES_96.md)；[E04 全部指标](stock_improvement/results/results.json)；[检查通过输出](stock_improvement/check_result.txt)。

自动检查确认样本与标签一致、测试未评估、最新价格在截止点之前结束、保存概率可复现、PCA/缩放仅拟合训练、C 来自训练期三折。统计区间未校正多模型选择或全部跨日相关性。

## 方案更新：完整改进与探索路线（2026-09-15，规划已完成；新实验未执行）

根据用户对未用论文方法与案例改进的追问，形成 [完整改进与探索方案 v1](IMPROVEMENT_PLAN.md)。这是本组建议的实施范围，不是课件新增要求。

- 主线：E05 目标公司相关内容 → E06 在线近似转载归并 → E07 有证据的结构化新闻评分小规模试验。
- 条件扩展：E07 全量、最多一个轻量注意力/GRU；完整 StockNet 与 CausalStock 因果图不进入当前主线。Twitter 突增须有真实历史消息序列，不以未知时点累计值代替。
- 两层评价：先验证内容/去重/证据的组件质量，再做相同样本与预算下的预测对照；组件改善不等于股价预测改善。
- E05 新增同长度正文开头对照，区分输入更多与选对目标信息。E06 保留来源数量，区分事件重复计权与传播强度。E07 加规则事件特征对照，区分字段设计与 LLM 贡献。
- 新的训练期质量集计划为 160 个文章—目标股票对；现有 504 窗口/96 卡/24 审阅保留，优先扩展到所有 50 个方向翻转窗口并减少重复事件阅读。
- 沿用训练期向前验证，承认旧开发数据已影响方案设计；最终测试仍保留，冻结后统一运行预声明的比较。
- 每阶段写明继续/停止条件、成本和适用范围。新实验目前均未启动，本文不把方案当成结果。

## E05：正文内容选择对照（2026-09-15，已完成预测实验）

**问题与动机：** 部分标题遗漏目标公司评级事实，或混合竞争对手评价。假设目标公司正文可能优于标题；同时需要排除“只是增加了文字”的解释。

**实际改动：** 对相同训练/验证文章做原标题、等长正文开头、最多三句目标相关内容 × 稀疏词/FinBERT 六组对照。抽取保留字符位置，最大 254 内容 token，缺失回退原标题。沿用 E04 训练期 6/7/8 月向前选参协议。没有改变原窗口、样本或测试范围。

**数据与检查：** 5,951 个公司—文章对，AAPL 5,459、AMZN 492；标题回退率分别 0.99%、1.63%。E05 质量样本 160 对；40 个锁定样本的展示目标摘录已由助手阅读并记录诊断，其他 E05 抽取标签尚未完成，不是 160 对人工金标准。精确原文跨度、文本长度、时间边界、训练变换和已保存预测复核通过。

**观察：** AAPL 正文开头稀疏模型验证 BA 56.31%，原标题 52.72%；但训练期向前 BA 分别 51.97% 与 56.51%，不能凭单次开发高分晋级。正文开头相对原标题的日分块差值区间为 [-1.98,+9.28] 个百分点。目标相关句稀疏模型 AAPL BA 50%，系数全部被正则化为零。目标句 FinBERT AAPL 51.93%、AMZN 50.82%，也未建立一致优势。

**Insight：** 公司名称命中不是主事件识别。持仓标题有清楚的买卖变化，正文排序却容易选中历史涨幅、待发财报和 EPS 背景。回退很少不等于抽取正确。

**决定：** 当前目标句提取器不作为已验证改进；正文开头保留为候选观察。六组完整结果、局限和案例见 [探索报告](EXPLORATION_REPORT.md)。

## E06：近似转载归并（2026-09-15，已完成预测实验）

**问题与假设：** 多个来源的近似标题可能重复计权。若按近似转载组平均更好，仍需区分内容去重与新闻数量信息的作用。

**实际改动：** 仅向过去 24 小时匹配，标题相似度 0.95，保护数字和方向词。初版 100 对候选已做助手标题级诊断；第 87 对包含新增 BX，随后在预测实验前加入股票代码集合一致保护，保留初版与新版文件。不是全文人工去重准确率评估。新版重新抽出的质量候选不冒充已全部阅读。

**控制：** 每股表示按 E05 训练期向前验证选择，AAPL 为原标题稀疏词、AMZN 为正文开头情绪。比较文章等权、文章等权加数量、簇等权、簇等权加数量。代表文章必须原本就在当前窗口内；没有用未来转载改写过去。

**观察：** AAPL 文章等权 BA 52.72%，簇等权 52.73%，变化很小。AMZN 对应 50.04% 和 50.44%；但加入额外数量后文章对照为 52.00%，簇对照为 50.82%。没有稳定去重收益。

**Insight/决定：** 转载量可能携带传播信息，减少重复与保留数量必须分开检查。近似文本簇也不等于真实事件。保留结果，不据此升级模型复杂度。

## E07：结构化信息试验与用户恢复探索（2026-09-15）

### A. 已有多字段小样本试验

v1/v2 都在同一批前 20 条触发质量停止；没有完成 100 条或全量运行。v1 严格 schema 合格 0/20；独立逐字引用匹配 15/20，原先带 schema 前提的引用指标为 0，两个口径不可混淆。v2 改为选证据编号，schema 合格 11/20、schema 合格且编号可用 10/20，仍出现持仓变动被误判成业绩指引等问题。

### B. 本轮为何继续及范围

用户明确要求继续原先探索，恢复 E07 的有限验证；此前的暂停属于当时收敛决定。本轮预先登记“规则事件预测＋简化 LLM 事件/证据选择”，不直接跑完整 CausalStock、注意力或 GRU。沿用同一个本地固定 Qwen 模型，没有新增付费 API 或外部新闻推理。

### C. E07-R 规则事件特征

**假设：** 当前主事件类型（例如持仓、评级）比整体情绪更能区分信息。规则先读取含目标公司的标题分句，固定九类；将窗口内各类文章比例加入 LR。它不直接规定某种事件必涨/必跌。原价格、新闻数量、样本和选参预算保持一致。

| 方法 | AAPL 验证 BA | AMZN 验证 BA |
|---|---:|---:|
| 价格＋新闻数量 | 49.17% | 49.26% |
| 价格＋FinBERT＋数量 | 49.99% | 50.84% |
| 价格＋数量＋规则事件 | 48.71% | 51.21% |
| 价格＋FinBERT＋数量＋规则事件 | 49.11% | 51.20% |

相对 FinBERT 数量对照，AAPL 38 个方向变化（18 改对、20 改错），AMZN 23 个变化（12 改对、11 改错）。共 504 个窗口，30 改对、31 改错。新增事件特征各配对日分块区间均跨零；没有跨股一致收益。全部预测、月份、Brier、参数网格均已保存。

### D. E07 v3 小模型诊断

**改动：** 输出缩到主事件类型与一个原文证据。用单字母选项限制生成，再由程序构造 JSON 和回填证据；移除重要性、影响周期等不易核实字段。输入仍是标题＋原目标摘录；本轮没有声称修好了 E05 正文抽取。

**验收安排：** 原试验 20 条诊断样本＋20 条新增检查，均为训练期质量开发样本。助手在模型输出前记录事件及允许证据；有歧义时预先允许两类。它不是组员独立金标准，不能用一致率代表全语料准确率。

**实际结果：** 新增 20 条上，LLM 事件一致率 10%、事件与证据共同符合 10%；规则事件一致率 55%。既有 20 条分别为 0%、0%、75%。模型总计选择 earnings 37 次、product_business 3 次。格式 100% 由约束代码保证，不代表语义改善。任务简化后，与 v2 的指标不可直接作提升比较。

**实现排查：** 人工构造分数验证限制器只保留九个选项且保留正确最大值；对 4/6/12 三条读取原模型分数并取消输出限制，仍得到 A（earnings）。当前证据支持模型/提示组合表现不可靠，不能据此断言所有 LLM 都无用或归因于具体训练/量化问题。

**病例 insight：** 规则新检查中的错误多为过宽的 other，包括收购、ETF/行业行情和标题泛化但正文含具体事实的文章。另外发现德语标题夹杂链接摘要，原元数据语言分类未完全可靠。继续增加关键词只能修补局部覆盖；若继续，要先建立经组员复核的目标主事件与证据标签，并保留新的检查样本。

### E. 当前决定与交付

本轮规则模型和小模型均不晋级全量结构化处理。候选 90% 新检查一致性门槛未达到，独立组员标签也不存在；不进入注意力/GRU。最终测试没有评价。保留传统基线＋FinBERT 主比较，所有负结果可用于报告的消融与错误分析。

已完成 [完整探索报告](EXPLORATION_REPORT.md)、[40 例事件案例](stock_events/CASES_40.md)、[本轮复现说明](stock_events/README.md)、[独立检查结果](stock_events/check_result.txt)。主日志和结果文件保留先前的暂停、修复与失败历史，不把探索重新描述为老师要求。

## E08-B：同信息量的时间模型对照（2026-09-15，已完成）

### 为什么继续、与之前停止条件的关系

用户在核对论文方法后明确要求执行时间建模比较。E07 的结构化事件提取没有通过质量门槛，本轮不依赖它；直接使用既有 E03 标题 FinBERT 与价格特征，检验一个不同假设：按时间读取同样的信息，是否比扁平输入更有效？这是 StockNet 启发的组件实验，没有复现其变分隐变量、辅助目标、原始 Twitter 数据或完整模型。

### 具体输入与控制

每个当前样本读取当前截止快照＋此前五个目标时段已结束的快照。快照包含原 16 个价格变量和 6 个新闻情绪/数量变量，追加时间间隔和有效标记，形成最多 6×24 维。旧快照从原开发样本索引选取，不包含它的真实涨跌或目标收益作为特征。

同样输入比较扁平 LR、普通 MLP、GRU；另保留当前快照 LR 检查历史信息增量。MLP 877 参数，GRU 825 参数，避免只把“有非线性”当成时间模型优势。每股原样本保留，最早 6 行用补零与长度掩码；最老快照距当前中位数 24 小时、最大 96–97 小时。因此不能将六个快照描述为六个连续自然小时，也不能说当前快照模型没有历史价格。

### 预先登记的训练安排

神经网络固定种子 573/574/575，一层 GRU hidden=8、MLP hidden=6、40 epochs、AdamW lr=0.003、batch=64、clip=1。各比较三种 weight decay；LR 比较三个 C。用训练期 6/7/8 月向前验证，神经网络在所有月份及种子的平均 BA 后按 Brier 选参数。开发验证 9–10 月不参与选种子或配置。本轮共 120 次神经网络拟合、40 次 LR 拟合；主训练脚本 CPU 实测约 26 秒，另做检查和一次确定性重训。

### 结果与变化

| 方法 | AAPL 验证 BA | AMZN 验证 BA | AAPL Brier | AMZN Brier |
|---|---:|---:|---:|---:|
| 当前快照 LR | 49.99% | 50.84% | 0.2599 | 0.2555 |
| 六快照扁平 LR | 50.79% | 53.58% | 0.2551 | 0.2598 |
| 六快照 MLP | 51.90% ± 1.33 | 50.80% ± 2.49 | 0.2969 | 0.3059 |
| 六快照 GRU | 49.39% ± 1.07 | 52.63% ± 1.16 | 0.3208 | 0.2964 |

“±”为三个种子 BA 的样本标准差（百分点），不是置信区间。主结果不是挑最好种子或集成结果。

AMZN 六快照 LR 在 9 月为 63.17%、10 月为 44.01%；AAPL 对应 57.78%、45.36%。整体的历史信息增幅没有跨月稳定性。GRU 两股均未超过同信息量 LR，Brier 也都更差。神经网络训练 BA 约 73%–79%，验证接近 50%，符合泛化不足/过拟合的表现；样本少、重叠输入或时间变化是待区分解释，不是已证明原因。

联合同交易日重采样两股，序列 LR−当前 LR、GRU−序列 LR、GRU−MLP 的描述性区间都跨零。反序有效快照明显改变 GRU 概率，说明其依赖输入排列；由于这属于输入分布扰动且保留了时间间隔特征，不能解释成顺序机制已被证实有效。

### 案例与检查

保存全部 504 个验证窗口比较和 16 张程序诊断卡，固定种子 573，覆盖改对、改错、共同正确和共同错误。该种子下 AAPL 43 改对/44 改错，AMZN 44 改对/43 改错；这不是选择最佳种子，也不是正文人工因果解释。

独立检查通过：源/代码哈希、样本与标签一致、历史快照和新闻可用时间、训练部分缩放和参数选择、每个模型权重重现预测、GRU 掩码忽略填充、反序结果复现、原当前 LR 对照复现、代表 GRU 相同种子重训复现。最终测试未评价，没有新增外部模型推理。

### 决定

GRU 暂不晋级；保留所有结果，不自动扩大网络或网格。若另开一轮，训练期内早停或加强正则化是针对已观察训练/验证差距的有限候选，但需要独立协议与同预算对照，不能事后在开发月份挑最优轮次。本轮已完成用户授权的时间模型对比。

证据：[E08 完整报告](stock_temporal/REPORT.md)、[16 张诊断卡](stock_temporal/CASES_16.md)、[代码与复现说明](stock_temporal/README.md)、[检查](stock_temporal/check_result.txt)。

## D09：根因分析、微调与图方法设计（2026-09-15，诊断已完成，新模型未运行）

### 用户问题与本轮范围

用户要求结合已有结果、课堂知识和论文分析表现不足的原因，讨论完整系统、FinBERT 微调、方法组合及数学/图方法。本轮核对原论文与 Canvas 课程主题，并运行只读取既有开发数据的诊断；没有把方案当作已经实施，没有训练新模型，没有评价最终测试。

### 新观察

- 训练窗口 AAPL 996、AMZN 1,004，均覆盖同样的 168 个交易日；文章记录分别 4,640、375，新闻覆盖 99.30%、56.67%。新闻记录多不等于涨跌监督多。
- 跨窗口重复使用的训练记录为 3,767、306，其中 2,987、240 同时对应过涨与跌，比例 79.29%、78.43%。这是正常的时间依赖，不是原窗口标签错误；将每个窗口的标签复制给文章会丢掉不同时间与价格背景。
- 两股同期目标收益相关：训练 0.570、验证 0.735。截止前最后已完成合格小时的其他股票收益与目标收益相关：训练约 0.041/0.054、验证 −0.002/−0.007。这不是因果或条件预测能力检验，且没有覆盖全部 5 分钟滞后；同期相关不能直接用作可预测性证据。
- AMZN 9→10 月上涨比例 55.26%→44.20%、新闻覆盖 76.32%→58.70%、绝对收益中位数 0.280%→0.591%。它们与性能变化并存，但不能据此断言具体成因。
- 同协议价格 LR 与标题 FinBERT LR 的错误相关两股均约 0.72；序列 LR 与 GRU 约 0.31，后者改对/改错仍几乎相抵。互不相同不等于可以成功融合；本轮没有融合成绩。

### Insight 与对之前方法的修正

最有依据的解释是有限监督、输入与目标时段对应不足、时间泛化问题共同存在；尚未确定单一根因或数据预测上限。已有 GRU/MLP 的训练 BA 高而验证接近 50%，说明要检查泛化；但原训练固定 40 轮、无早停，不应把它当成已经穷尽神经模型。

此前目标公司片段和 E07 提取组件仍有具体质量问题。改准语义不等于改准股价：新闻可能已在预测目标小时之前反映，或与长期预期有关。这是需要事件时点诊断验证的假设，不能通过事后删困难样本“证明”。最新 5 分钟价格已经在 E04 试过，不能再次包装为未尝试的新修复。

### 设计决定（尚未执行）

1. 有限补齐更贴近老师论文的新闻分支；保留原协议差异。没有历史 Twitter 消息流，不能称完整复现。
2. 比较独立、共享、部分共享 LR，用正则约束每股修正；神经模型采用训练期内部早停。目标是减少估计波动，不保证提升。
3. 值得做一次窗口级 FinBERT 小 LoRA：同输入、同聚合、同预测头的冻结对照，按整个窗口计算损失，三种子、训练期选参。我们此前没有再微调 FinBERT。
4. 图优先用于有证据和时点的实体—事件关系，先覆盖评级/持仓两类，必须先有独立内容复核；比较同一事实的平面特征和关系特征，不能把画图当算法改进。两股票节点的关系网络优先级较低。
5. 单项有价值才做四格消融或训练期样本外预测融合。不同时堆完整 StockNet、CausalStock、多股 GNN 与全参数微调；继续探索不等于这些都成为课程必做项。

### 交付与检查

[完整分析和数学设计](ROOT_CAUSE_AND_NEXT_DESIGN.md)、[新诊断说明](stock_signal_audit/README.md)、[复现脚本](stock_signal_audit/audit.py)、[统计及源哈希](stock_signal_audit/audit.json)。脚本实际检查通过开发范围、原标签和数量、预测对齐、同一基线复现、过去价格截止约束。新设计受既有开发结果影响，未来报告必须公开；最终测试仍封存。


## E09–E11：Review 修正与新一轮实际实验（2026-09-15）

### 授权、顺序与保留原则

用户要求据根因分析实际逐步实验，并提供side-chat code review。先保存9份旧源码与204个历史产物哈希，修正恢复/缓存/准备阶段保护；随后登记E09进行统一基线、全文、共享、早停实验；E10独立协议微调。在看过E09结果后另登记E09-C校准和E11正则化对照，明确它们是自适应开发探索。

本轮使用原AAPL996/252、AMZN1004/252训练/开发窗口，不删无新闻时段、不改任务、不看最终测试。首个E09准备因混合时间字符串退出，在任何新训练之前发生；现场保存在attempt_01_input_parse，修正解析后重新准备。

### Review 五项实际修正

早期同验证集选C/报分流程标为历史开发并防覆盖；E09用训练期三向前折重训全部基线。结构化pilot恢复前持续检查首20条质量、输入/提示词/配置指纹，旧失败试验实际被阻止。缓存纳入内容、上游概率、模型revision、编码/聚合、代码和payload指纹，旧缓存不自动追认。时间模型从prepare开始保护整个实验目录，训练前核验指纹。固定40轮问题通过新的训练期内部早停实验处理。5个保护单元测试、实际入口拒绝检查通过；历史204个哈希保持不变。

### E09：进一步借鉴论文新闻分支

标题/全文采用同一词干化、停用词、二值词、训练折卡方500、L1 LR＋价格。AAPL标题52.72%→全文59.44%，+6.72个百分点；AMZN49.77%→47.59%，−2.18个百分点。AAPL全文九月60.65%/十月57.94%，但训练BA79.93%、训练期CV51.12%低于标题55.60%。AAPL正文Brier.2868偏高。更明显的单股开发提升不等于跨股稳定泛化；全部504窗口比较AAPL60改对44改错，AMZN33改对38改错。

32张分层程序卡＋8例重点摘录复核显示：作者、免责声明、其他公司、旧财务背景都会贡献预测；正确预测也可能依赖这些内容。非零词少于5训练交易日的比例并不占多数，不能宣称根因已证明是极少数事件记忆。

### E09：共享与训练控制

独立/共享/部分共享BA依次49.58/51.66、50.06/52.18、50.78/51.25（AAPL/AMZN，百分数），没有明显稳定增幅。部分共享与两节点图平滑正则数学等价，已在报告推导；这不等于GNN或因果图。

早停每个训练前缀内留最后20交易日，按BCE选轮数后完整前缀重训。原GRU BA49.39/52.63变为49.94/50.62，Brier.3208/.2964降为.2571/.2571；更小GRU48.90/50.61。训练控制修正改善了概率损失，但没有解决方向泛化；MLP与所有种子也保存。

### E09-C：概率校准的成功和失败

看到全文模型高Brier后另行设计。对六月/七月/八月生成嵌套选C的训练期样本外logit，优化正温度，应用到E09完整训练模型。AAPL全文T6.954，Brier.2868→.2470、BA59.44%不变；AMZN全文T.414，Brier.2730→.3214、BA47.59%不变。标题也报告：AAPL.2507→.2496，AMZN.2656→.3261。12个样本外模型重训、温度和预测重现通过；校准无法保证跨时间迁移。

### E10：窗口级FinBERT LoRA，实际完成

同输入冻结与LoRA两臂，每窗最多16条目标公司前缀标题；两层Q/V、rank4，24,576个适配参数＋37个头参数。整窗BCE监督，训练期内部早停、三种子，不把窗口标签复制给文章。AAPL训练新闻出现次数保留82.71%、开发92.64%，AMZN100%；全504开发窗口保留。

- AAPL：冻结BA 50.51%±1.23个百分点，LoRA 50.38%±1.20个百分点；Brier 0.2522→0.2522。
- AMZN：冻结BA 50.99%±1.21个百分点，LoRA 50.87%±1.24个百分点；Brier 0.2513→0.2513。

零LoRA重现、非零适配梯度/更新、冻结基座逐张量不变、输入时间与训练投影、六份保存模型重现预测均通过。固定PCA投影、最后两层与单一配置是能力限制，不能宣称穷尽微调。配对差和逐月结果见E10报告。

### E11：只改L1/L2

这是看到E09过拟合和来源词影响后登记的小对照，保持词表/特征/价格输入一致、相同三C预算。AAPL全文L2 BA54.74%低于L1 59.44%，AMZN46.07%低于47.59%；标题L2也未超过L1。没有继续扩大C网格。检查确认特征集合相同、选参与缩放来自训练、所有权重可复现。

### 本轮决定

保留全文L1及校准为AAPL有希望的候选，报告AMZN失败；统一基线、早停、LoRA、共享、L2作为完整探索证据。停止根据当前开发集继续扩网格。最终测试仍未评价，候选/比较规则冻结后再统一测试。实体—事件图仍受结构化抽取质量门槛约束，没有扩为复杂GNN。

[本轮总报告](EXPERIMENT_PROGRESS_20260915.md)、[Review修正](stock_review_fixes/README.md)、[E09](stock_robust/REPORT.md)、[校准](stock_calibration/README.md)、[E10](stock_finetune/REPORT.md)、[E11](stock_text_regularization/README.md)、[案例](stock_robust/QUALITATIVE_REVIEW.md)。

## E12：用户授权后的最终测试（2026-09-15，已完成）

### 为什么做、怎样固定

用户明确要求在测试集上尝试全文L1和L2。首先锁定已保存的Jan–Aug模型、训练期选择的C、词表、卡方选择、阈值和数据处理，不加入Sep–Oct重训。加入相同正则的标题对照、价格对照；固定的全文L1温度校准作为补充，两股均报告。方案在首次测试预测前登记，代码/模型/数据哈希保存于E12协议。

从原始价格、新闻重新准备，精确重现此前2504个训练/开发输入后，评估2018-11-01至2019-02-01：AAPL364、AMZN365窗口，两股各62交易日；2月仅2月1日，各股5窗口。未修改过滤规则、未删无新闻窗口，也没有测试集调参。

### 实际结果

| 股票 | 方法 | 开发BA | 最终测试BA | 对应标题测试BA |
|---|---|---:|---:|---:|
| AAPL | 全文L1 | 59.44% | 49.24% | 46.33% |
| AAPL | 全文L2 | 54.74% | 48.06% | 45.46% |
| AMZN | 全文L1 | 47.59% | 50.92% | 51.19% |
| AMZN | 全文L2 | 46.07% | 55.04% | 50.39% |

两股等权平均BA：固定全文L1为50.08%，固定全文L2为51.55%。没有把AAPL的L1与AMZN的L2事后拼接成一个预先声明的系统。

### Insight 与数据变化

- AAPL开发集全文L1的59.44%没有保持；测试49.24%虽比标题高2.91个百分点，仍低于恒定方向50%的BA参照。不能把开发时看到的绝对高分当作最终泛化能力。
- AMZN全文L2在本次测试较好：11月52.98%、12月50.88%、1月59.76%，三个完整月份都超过匹配标题。相对标题整体+4.65个百分点，全部365窗口中52个改对、35个改错；全文L1相对标题40个改对、41个改错。
- AAPL全文L1相对标题86个改对、66个改错，全文L2为85/73。方向变化来自整个联合模型，不能直接解释为每个新闻语义判断被修复。
- AMZN L2相对标题的单日成对95%区间为[+0.04,+9.76]个百分点，五日移动块为[−1.30,+8.96]，跨零；相对全文L1的两种区间为正。两股平均主要对照区间均跨零。证据支持较好的单股测试结果，尚不能保证跨时段、跨股票稳定提升。
- 概率质量没有同步改善：AMZN全文L2 Brier .2624，比标题L2 .2536差。AAPL全文L1原始.3392，经原固定温度降至.2567，BA不变；AMZN全文L1校准从.2708变为.3222，仍失败。没有用测试集重拟合温度或临时给L2增加校准。
- AAPL测试364窗口全部有新闻；AMZN195有新闻、170无新闻。AMZN全文L2有/无新闻BA54.77%/54.66%；新增文本改变训练得到的价格系数，因此无新闻窗口也能改变。划分间上涨比例、覆盖和收益幅度变化已保存为描述统计，不据此认定单一根因。

### 检查与决定

全部729窗口的时间、新闻键/词集合、历史价格和原样本对齐通过。10份保存模型使用原训练期数据与配置独立重训，系数、截距、特征集合重现；所有测试预测、整体/月度指标、固定温度方向不变检查通过。历史204个产物及上一轮171个新实验文件未变化。

保留这组完整最终测试结果：AAPL开发高分未保持，AMZN L2出现较好的单股结果，方向与概率质量需要分开陈述。当前保留时段已经被评估；后续据此修改的方法不能再把同一时段称作未见过的最终测试。此次没有进一步搜索参数。

[最终测试完整报告](stock_final_test/REPORT.md)、[固定协议](stock_final_test/protocol.json)、[分数表](stock_final_test/results/test_scores.csv)、[全部729窗口比较](stock_final_test/results/all_729_comparisons.csv)、[复现检查](stock_final_test/check_result.txt)。

## D13：E12 后的方法盘点与 side chat 方案核对（2026-09-15，未运行新实验）

用户询问尚未尝试的方法，并提供只读根因审查和后续设计。本轮核对已有协议、代码及 E12 分数，确认定期更新训练、对象/动作/数值变化事件事实、保留绝对证据强度的时效聚合尚未完成；简单时间过滤/加权、公司句抽取、事件类别、早停及真实 LoRA 已做，不能重新包装为全新方法。

接受“优先检验时间稳定性和输入信息”的主线，不将单词分布变化或新闻缺陷直接当作全部下降的因果解释。用户提供的 3,233 样本及逐词统计记为 side chat 诊断，本轮未重跑其全部审查。当前 E12 保留期已暴露，之后的改进在该时期属于历史回测。

方案细化：先在同一全文 LR 内比较冻结、按月扩展、六个月滚动；L1/L2 和两股均完整报告。事件先限定评级/目标价与业绩指引，检查对象、旧新值、指标/单位/期间及披露证据，组员复核与助手开发标签分开。质量/覆盖合格后运行 B0–B3，再决定是否需要图学习。

数学 insight：分母 λ+总权重使普通加权平均乘以总权重/(λ+总权重)。一条权重 1/16 的旧事件从 1 收缩为约 0.0588；100 条相同旧报道仍可累积到约 0.862。因此它缓解弱证据归一化后被放大的问题，但不保证消除重复旧闻，也不保证学习后的概率更好；还需明确事件计量单位、时间代理和特征尺度。完整推导、对照、未尝试方法及停止条件见[下一轮设计](NEXT_EXPLORATION_AFTER_E12.md)。

本轮只完成核对与文档维护，没有新增训练、测试预测或外部模型推理，没有改动已完成实验产物。

## E13–E15：按授权实现更新策略、事件事实与浅层提升树（2026-09-15）

### 目标、评估边界与执行顺序

用户要求将 D13 建议实际实现并实验。本轮先运行冻结/每月扩展/六个月滚动训练，再实现三版事件抽取、逐条检查，最后进行限缩字段 B0–B3 与一个浅层提升树对照。所有新结果均是原 E12 测试已暴露后的历史回测，不重新称为独立最终测试。每次只使用预测月之前已实现的标签，完整保留原 3,233 样本；不改一小时目标、不删除无新闻窗口。

### E13：定期更新没有稳定优势

保持全文词干二值词、训练折卡方 500、L1/L2 LR；每个更新训练范围内最后三个完整月向前选三个 C，重新拟合词表/缩放/筛选。72 份月度模型记录与全部预测保存，其中 24 份为原冻结模型副本。

原保留期 BA（冻结 → 扩展 → 滚动）：

| 股票/模型 | 冻结 | 扩展 | 滚动六个月 |
|---|---:|---:|---:|
| AAPL L1 | 49.24% | 46.08% | 52.53% |
| AAPL L2 | 48.06% | 47.46% | 48.12% |
| AMZN L1 | 50.92% | 49.21% | 50.36% |
| AMZN L2 | 55.04% | 48.19% | 48.74% |

AAPL 滚动 L1 +3.29 个百分点，但五日块区间约 [−6.03,+8.54]。深入核验发现 12 月、1 月 C=0.01，系数与截距全零，始终输出 0.5；各月 BA 都为 50%。这不能解释为跨时段稳定理解词语，概率损失下降的一部分来自抑制过度自信。完整 11/12/1 月 BA 为 51.88%/50%/50%；2 月只有一天，单独报告。不能用总体合并 BA 代替跨月稳定性证据。

初次启动因 common 模块同名循环导入，在协议登记和训练之前退出；修正加载方式后执行。原 E12 预测精确重现，四个代表更新模型独立重训重现。模型更新的样本量、词表和所选正则同时变化，本轮不声称分离其单独因果贡献。

### E14：事件抽取发现新的具体失败机制

v1 接受案例中完整字段助手检查 18/37 合格，仍混入历史旁支事件、丢失动作。v2 为 25/31，进一步发现标题的 upgrade/downgrade 不一定是评级变化：Apple 例子的正文明确更正“仍 overweight，只下调目标价”；Amazon 例子实际是目标价 1,850→2,500，并维持 overweight。两版均未进入预测。

v3 要求正文支持明确旧评级→新评级；否则弃权，修正部分机构名范围。新 30 个 AAPL 记录中完整字段 27/30，但目标价机构字段 16/18，仍未达到该类型 90% 门槛；完整事件系统和机构关系图没有宣布通过。不同版本检查集不同，且存在相关转载，不能将三个比率当同集准确率提升。

另行登记 E14_LIMITED_PROJECTION：去除机构字段及其存在标记，仅使用公司/类型/动作/数值或评级变化。实际输入字段在 v3 的 30 个检查记录上均有证据支持；这是助手复核，独立组员验收仍未完成，AMZN 没有新质量样本。失败完整 schema 状态保留，没有恢复 v1/v2 或把它们作为可靠输入。此限缩实验不能代表完整事件方法。

业绩指引的公司收入区间抽取已实现，但 AAPL 训练仅 8 条、AMZN 0 条，没有足够新质量样本，因此保存诊断而不进入主预测。完整首次披露时间仍未知，用文章发布时间作为明确标记的年龄代理。

### E14 限缩版 B0–B3 的实际表现

统一每月扩展训练；B0 价格，B1 加事件数量/类别，B2 加 11 个动作/数值/已知标记，B3 用发布时间代理年龄、H=4 小时、λ=1 及冻结规则 q 做收缩聚合并加入年龄/延迟。B3 有 18 个事件/时效特征，事件列不减均值以保留无事件为零；每臂同三个 C 和过去选参。

| 输入 | AAPL 原保留期 BA | AMZN 原保留期 BA |
|---|---:|---:|
| B0 | 46.19% | 50.11% |
| B1 | 47.35% | 50.11% |
| B2 | 47.20% | 48.74% |
| B3 | 45.65% | 48.19% |

没有观察到事实或收缩时效的增益。覆盖是明确限制：AAPL 训练 162/996 窗口、75 篇接受事件；AMZN 仅 8/1004 窗口、2 篇文章，原保留期只有 4/365 窗口、1 篇文章。新公式的数学收缩性质并未转换为预测改善；不能据此否定所有事件表示。

### E15：一个小型非线性对照

比较同样每月扩展训练的 LR 和浅层 HistGradientBoosting，分别用价格、价格＋新闻数。树深 2、最多 4 叶、每叶至少 30 样本、100 次提升、学习率 .05，关闭随机早停，三个正则候选。价格 LR→树：AAPL 46.19%→48.67%，AMZN 50.11%→47.69%；价格＋新闻数 LR→树：AAPL 47.00%→49.18%，AMZN 50.39%→47.68%。有限非线性没有带来跨股一致收益，不扩大模型搜索。

### 检查、交付与停止决定

168 份保存月度模型/副本重现预测，8 个代表模型独立重训；全部样本事件成员/可用时间、190 条抽取证据及数值配对、无事件零值/有限特征检查通过。两个数值实验的相同价格 LR 逐窗口概率相同。204 个早期产物、171 个上一轮实验文件、23 个 E12 产物哈希未变化。各阶段独立协议和目录，失败版本保留，没有付费推理。

这轮没有找到稳定、明显改善，按原停止逻辑暂不自动扩大注意力、更多微调或 GNN。机构和首次披露质量、组员独立复核仍是完整事件系统的未完成部分；课程报告可使用本轮有对照的负结果与适用边界。更大的探索需要新的具体证据或数据，新的独立泛化确认需要未用于设计的时期。

[本轮完整报告](EXPERIMENT_PROGRESS_E13_E15.md)、[E13](stock_adaptive/REPORT.md)、[E14](stock_event_facts_v3/REPORT.md)、[案例面板](stock_event_facts_v3/CASES.md)、[组员复核表](stock_event_facts_v3/HUMAN_REVIEW_TEMPLATE.csv)、[E15](stock_small_boost/REPORT.md)、[核验](E13_E15_VERIFICATION.json)。

## 2026-09-15 — M01–M09 全面机制计划实施

### 目的和协议

按用户批准方案，保持AAPL/AMZN、原一小时方向、全部3,233分类窗口；E00–E15保留。先检验容量/样本、模型更新、到达时点和语义融合，再按固定跨月门槛决定高级机制。增加独立日内RL任务。所有新分数统一称探索性历史回测，没有新独立测试期。未使用付费数据/API；本地计算完成，未提交Sol。

### 实际完成、观察与逻辑

| 阶段 | 实施和观察 | Insight / 解释边界 |
|---|---|---|
| M01 | 新建 `stock_comprehensive` 固定配置CLI、独立目录、来源/输入/模型哈希、样本键、逐概率/逐月结果、保存权重、阶段耗时、失败目录保留；新原始输入失配拒绝测试通过 | 实验可核对与重现；事后封存不冒充独立预注册 |
| M02 | 108个容量折拟合＋54个日期学习曲线拟合＋12个最终模型；k25/100/500 × L1/L2 × 3C。AAPL学习曲线向前BA 52.26/53.80/51.77%，AMZN52.01/52.16/53.29% | 减少容量改善部分Brier，但BA不稳定；增加同源样本不是两股一致的单调改善。100%三次相同，不是独立重复数据 |
| M03 | 冻结、只更新系数、更新变换、全重选；48个月度权重/副本。AAPL已暴露回测BA48.06/47.20/47.46/47.46%；AMZN55.04/49.30/50.10/48.19% | 不能只归因于模型旧了；独立复现的扩展参照与E13逐概率一致 |
| M04 | 新闻前60分钟、后15/30/60/120分钟；文章与在线72小时近似标题组；盘中/盘外、迟到、缺线覆盖分列 | 不跨缺线/正常时段，不回填首次披露；描述关联，不归因新闻、不重新挑目标周期 |
| M05 | 扫描78,055原始记录。有效AMZN正文命中且标题未命中14,694条；AAPL36,048条；保留直接/比较/不确定候选复核材料 | 原过滤确实限制覆盖，但原始条数含转载和顺带提及，不自动纳入训练。公开免费来源仅可行性候选，未取得合格新数据/保留期 |
| M06 | 修复机构长串/截断、评级与目标价混淆、正文纠错、历史列表、单位/期间；补规范化证据偏移、未知值、证据图、按组开发/检查、漏抽审查表 | 原已知3个错误机构长串不再输出，但不是全事件正确证明。检查集AAPL评级11/目标价18/指引3，AMZN目标价1；数量和独立复核均不足，正式门槛仍未过 |
| M07 | offset事件修正器、3C选择函数、门槛与无事件原概率精确回退已实现并测试；未训练。训练期暂定接受组AAPL69/AMZN2，事件窗口156/8 | AMZN除质量外还不满足样本量门槛。不能将代码或合成测试当预测收益 |
| M08 | 7,861篇标题重新进行冻结FinBERT编码；PCA8/16/32、三C、价格/文本单模型、训练OOF融合权重；8个最终单模型 | AAPL权重0，融合=价格。AMZN权重0.75、PCA8：回测BA50.66→52.27%、Brier0.2580→0.2501，但训练跨月门槛未过；不能称稳定改善 |
| M09 | 两股三个种子，各实际PPO训练20,000步；平盘保留/缺线整日排除；连续成交价格、日终清仓、5/10/20bps；空仓/日内多头/阈值/同信息单步策略 | 主成本10bps下所有PPO均亏损：AAPL约−17.95%至−32.86%，AMZN约−23.75%至−27.86%；空仓0。RL没有补出缺失信号，不能替代分类BA |

### 高级机制决定

两股等权的六月/七月/八月门槛在看结果前写入配置：≥2月正增量、平均BA≥+1pp、Brier恶化≤0.002。k25平均提升主要来自八月，六月/七月为负；k100 L2两月正但均值仅+0.66pp；融合仅一月正。全部未过。因此注意力、新LoRA未启动，高级GPU训练预算使用0小时。完整StockNet继续后置；Graph只完成证据关系存储，未训练消息传递。

### 核验与可重现证据

- 68个最终分类/月度模型重新加载重现逐概率；24个RL月度价格LR保存时也检查重载；6个PPO重新加载重现全部日内路径，确认为各20,000步。
- 全3,233原样本截止点与22,876新闻成员关系检查；RL历史价格与原分类特征相同，日内收益连乘等于首开盘至末收盘比值。
- 缓存内容/模型版本/损坏拒绝，质量停止在20及重启21条仍拒绝，无事件精确回退，交易空仓/反转/清仓解析例子通过。
- 1,017个E00–E15历史产物哈希未变化。旧实验不重写。
- 全面结果、逐月、共同交易日单日/五日块配对区间、176张固定选择案例卡；区间条件于已选模型，不修正反复探索。
- 离线demo已浏览器核查股票切换、概率、截止时间和新闻显示；另提供实际权重推理CLI。

### 保留的失败与修正

M04第一次因DataFrame索引/列同名在计算前停止，失败目录保留，修复为M04_v2。报告第一次因缺tabulate停止，安装后生成，失败记录保留。M07初版诊断计数把所有时期事件组计入训练规模，修复为M07_v2（69/2），原计数保留说明；门槛仍未过，未发生事件模型训练。未据此重复搜索涨分。

### 交付和停止

工程、低成本实验和RL已完成；方法层尚未达到稳定提升。正式事件/Graph等待独立复核和足量检查组；未找获合格新时期，不宣称新泛化确认。保持预算，不扩大网格。组员贡献/最终教师rubric/排版与提交需要团队核对，未在Canvas提交。

[交付入口](stock_comprehensive/README.md) · [完整报告](stock_comprehensive/runs/v1/report/REPORT.md) · [离线demo](stock_comprehensive/runs/v1/report/demo.html) · [课程报告材料](stock_comprehensive/COURSE_REPORT_MATERIALS.md) · [核验](stock_comprehensive/runs/v1/verification.json)


## 2026-09-15 — GPT-6 Pro 独立复核委托

通过 Chrome UI 向 GPT-6 Pro 提交完整原始新闻和价格、代码与实验结果、课程材料、阅读索引、文件哈希清单及详细审查 prompt。对话：https://chatgpt.com/c/6aa9cf2c-c984-83e8-969e-f19b009285ff 。页面已确认五个附件和正文成功提交，并显示“Pro 思考中”。此记录仅证明提交成功，不证明复核完成或任何新结论成立。附件范围及省略项见 gpt6_pro_review/START_HERE.md，提交指纹见 gpt6_pro_review/SUBMISSION_RECEIPT.json。

## 2026-09-15 — H01 预测周期对照（已实际训练）

**动机：**用户指出一小时是工程选择，而非已证实最适合新闻的周期；比较1h、4h和日频，区分更容易的目标与新闻带来的增量。

**实际改变：**新增 outputs/stock_horizons 独立实验目录。1h保持原定义，4h为每小时起点的连续四个盘中小时，day为当日常规开盘到收盘（含短日），不跨夜。所有预测提前五分钟，历史六个已完成小时与过去四小时新闻固定。同一截止点三种任务的输入逐项一致。各周期独立排除缺线、平盘和历史不足；全部排除原因保存。

**执行：**每股每周期价格/价格＋数量/价格＋数量＋标题TFIDF三组LR，固定三个C，以六月—八月向前验证选择。共162次候选折拟合、18个最终模型；本地训练与bootstrap约118秒。原1h全部3233样本、标签、16项价格特征和新闻标题一致。全部模型重载概率一致、训练标签结束早于预测截止、输入哈希不变。每次独立目录且拒绝覆盖。完成单日/五日块的配对区间和共同开盘日期面板。

**结果：**价格＋数量＋文本的原测试BA：AAPL 1h/4h/day =46.65%/53.02%/45.35%；AMZN=52.86%/51.21%/48.98%。AAPL 4h价格自身为52.92%，文本增量仅约0.10个百分点。AMZN 4h文本九十月49.72%，原测试51.21%，未跨时期显示一致优势。共同开盘AMZN 4h文本原测试58.29%（58天），但开发42.45%，且原测试数量对照已57.93%，不能挑高分单独宣传。

**数据变化：**日频训练每股167天，原测试61天；4h训练每股499行，相邻目标重叠，不能当独立样本。AMZN日频原测试新闻可用率39.34%，AAPL为100%。六个全窗口文本模型原测试Brier均高于固定0.5的0.25。

**解释和决定：**未发现统一稳定的更长周期；保留1h主线、4h辅助比较，不宣称独立最佳周期。日频本轮使用固定4小时新闻窗，只代表这个输入设计，不能排除其他回看周期；以后改变新闻窗要另立实验。当前全部是已暴露历史回测。本轮未执行Pro提出的三个新机制；它们与这次周期请求分开。报告：outputs/stock_horizons/REPORT.md；协议、数据、模型、逐月指标、排除原因和指纹均在 runs/v1。


## 2026-09-15 — 四小时任务 Pro 后续委托

已在原 GPT-6 Pro 对话提交最新 stock_horizons 全目录（不含 bytecode）、当前项目日志、side chat 原文、详细四小时审查 prompt 与清单。请求核验时点构成、价格新鲜度、方向偏向、新闻对象及数量/校准混杂，阅读论文方法，扩展案例，输出一个优先实验及至多两个条件后续，并主动反驳自己的方案。页面确认“Pro 思考中”，尚未收到本轮结果；不视为新实验完成。委托及指纹：outputs/gpt6_pro_review/four_hour_followup/。

## 2026-09-15 — 执行 GPT-6 Pro 四小时独立审计后的价格机制方案

来源：同一 ChatGPT 对话的最新回复及其 `CSE573_FOUR_HOUR_INDEPENDENT_AUDIT.zip`，证据副本保存于 `stock_four_hour_v2/evidence/`。所有结果继续标记为已暴露历史的探索回测，未覆盖任何旧实验。

**为什么做：** 检验先前小时聚合遗漏的收盘/开盘与截止前五分钟信息是否解释四小时表现，同时通过O/M/S/R拆分价格值、可用性元数据和陈旧信息；避免仅增加维度就归功于新信息。

**实际实施与训练：** 172次LR拟合（嵌套向前选参、四小时四组对照、相同起点的一小时/四小时重训），再补68次五分钟可用延迟敏感性，共240次。所有权重、预测、样本键、特征来源、选参月份、模型哈希保存。按训练行数换算平均损失lambda；两组延迟运行约12秒和11秒，均为本地小型CPU模型。

**观察：** 六月至八月平均BA，AAPL O 52.21%→R 48.83%，AMZN O 57.79%→R 56.75%；R相对O均只有1/3月非负，两股均未通过工程继续门槛。M分别54.09%/58.87%，R还不如仅增加可用性元数据。五分钟延迟下R−O为−2.61/−2.01个百分点，结论不变。九月以后冻结回放也没有给R稳定支持。

**Insight与局限：** 截止前增加价格信息在这组固定容量/正则化对照中没有稳定收益；不能据此证明市场完全不可预测。M较好提示时间/可用性本身值得作为解释变量，但冻结回放未持续，不能追着单段高分继续扩网格。BA按月/类别平均，不等同于总改对数量：AMZN外层修正16个、引入15个错误，但月均BA仍下降。冻结回放AAPL修正24个/引入28个，AMZN修正15个/引入20个。保留全部案例与每象限8个的固定128行案例面板，尚未逐篇人工解释。

**核验：** 重现原18模型推断（误差≤1e-10）；原始五分钟线独立核对1607个四小时标签；240个训练收敛、保存模型重新预测误差≤1.12e-16；1607个新旧特征来源逐项检查没有越过asof/cutoff。12项边界/机制测试通过。新增10000次共享交易日块配对区间，5日主分析及1/10日敏感性，仅描述性而非选择校正后的显著性。

**后续分支：** 已修复训练词频并列确定性，但没有重训文本模型。已准备40篇无未来标签的独立正文复核材料，并实现/测试校准、固定基础offset、无新闻严格回退、独立文章PCA和每次恢复都重验的质量闸门；完整文本训练编排尚未接入，正文实验未运行。独立复核缺失、正文语义增量未证明，时间分箱按条件保持未运行。未运行非RTH报价实验，因报价语义未认证。不把这些未完成项写成成功实验。

交付：`stock_four_hour_v2/REPORT.md`、`STATUS.json`、`verification.json`、`runs/v1/`、`runs/lag5/`、`review/blind_review_40.csv`、`environment.lock.txt`。


## 2026-09-15 — 完整组合流水线与统一新闻 baseline

用户要求把已有机制协调成完整系统，并实际比较结果。本轮固定一小时3233窗口，152次LR拟合（训练期前向参数选择、四分支、冻结最终模型），使用过去OOF选择35种非负组合，Brier约束与无新闻价格回退；追加按相同规则选权重的消融和四分支等权参照。未使用测试成绩选择权重。FinBERT复用已校验冻结文章缓存，在唯一训练文章上重新PCA16；不宣称微调。

AAPL：BA 46.20% → 45.04%（-1.16个百分点）；Brier 0.2748 → 0.2705。

AMZN：BA 50.93% → 51.21%（+0.28个百分点）；Brier 0.2571 → 0.2543。

AAPL最终50%价格+25%全文+25%FinBERT，AMZN75%价格+25%标题。开发期AAPL49.9962%→48.8155%，AMZN49.6346%→49.6346%。组合仍无稳定方向增量；Brier小幅改善不等于BA提升。AMZN后续全文单模型54.77%，但不得按回放成绩事后改成最终候选。

第一轮float32语义重载检查最大概率差1.91e-8而中断，保留runs/v1；v2仅改float64和1e-12误差/方向一致检查，非看分数调参。最终系统去掉标签后独立推断一致（误差<1e-16）；5项单元测试，另核验全部参数选择、模型哈希、时间边界和组合回退。所有历史数据已暴露，结果属探索回放。详见stock_integrated/REPORT.md、PROTOCOL.md、runs/v2/。


## 2026-09-15 — 纠正目标：四小时完整组合实验

用户再次确认四小时为主目标。此前一小时组合未满足目标，本次独立新目录执行完整四小时流水线，不迁移一小时参数/权重。152次LR拟合，1607窗口，原始48根五分钟线逐一核对标签；正文重新聚合5078文章，4篇补冻结FinBERT推断。训练期选择各分支参数、融合权重、无新闻回退、同协议消融；保存全部候选和运行模型。

AAPL：BA 51.89%→55.69%（+3.80个百分点）；Brier 0.2694→0.2630。

AMZN：BA 48.92%→48.67%（-0.25个百分点）；Brier 0.2772→0.2610。

开发期两股组合较baseline提高，但后续AMZN收益未保持。AAPL最终权重[.25,0,.5,.25]，AMZN[.25,.25,.5,0]（价格/标题/全文/FinBERT）。AAPL后续去FinBERT51.35%、去全文53.47%，完整55.69%；改进区间仍跨零，不宣称稳定泛化。后续AAPL单语义56.71%和AMZN单全文53.20%仅作为分支结果，不回看择优替代冻结系统。

核验：5项单元测试；152次拟合来源/模型哈希、时间边界、C及权重选择重算通过；609个冻结窗口端到端无标签重载误差<1e-16；全部1607行均为4小时、信息提前5分钟。10000次共享日期块bootstrap、完整案例/逐月表保存。预处理一次缓存性能中断保留，不改变统计协议。

主交付：stock_integrated_4h/REPORT.md；runs/v1/；prepared/；PROTOCOL.md。后续默认主任务四小时，不擅自切回一小时。

## 2026-09-15 — 四小时组合失败诊断、论文映射与训练证据

本轮扫描1607输入窗口和609开发/后续预测，分新闻覆盖/月份/股票检查；审阅16诊断窗口中的29篇新闻标题与正文节选，补足上轮仅生成case panel的不足。不是全部5078篇人工审阅，也非独立质量金标准。新证据：AMZN后续仅84/179有新闻（72篇独立新闻），有新闻时全文BA55.43%而融合47.24%；全文正确被融合改错20次、反向修正12次。价格与标题后续概率相关AAPL .984、AMZN .938，四分支都含价格，简单融合不是四个独立专家。AAPL全文训练84.70%→后续51.74%；完整模型78.09%预测上涨、下跌召回仅27.47%。

实际重训核验：PID3969从输入重新拟合AAPL语义与AMZN全文，499行/167日，34/516维，优化器7/6步，重现原模型预测最大差0。FinBERT基座冻结，实际训练的是PCA和LR；原152拟合=144前向候选+8最终模型。本轮2次是验证refit，不作为新性能候选。

方案首选：价格底座只建模一次+新闻增量；利用过去OOF误差训练低容量条件门控；以公司—事件—证据—时间关系改善新闻输入，并保留高覆盖原文回退。先单列校准/近期误差更新/静态与条件门控对照，避免重复之前月度重训和严格事件低覆盖。完整事件/graph仍需独立质量复核。相关论文重新核对ADE、Ding2016、MASTER、DoubleAdapt、StockNet、FinBERT及指定论文；没有宣称新门控已训练。

交付：stock_integrated_4h/diagnosis/FINDINGS_AND_NEXT_PIPELINE.md、CASE_CARDS.md、各诊断CSV、training_process.log、training_verification.json。

## 2026-09-15 — 完善四小时自适应新闻增量方案（计划，未训练）

用户要求在已发现的问题基础上给出完整方案。本轮重读现有四小时协议、实际训练代码、覆盖诊断、训练复核和独立审阅状态，并核对动态专家选择、知识增强事件表示、FinBERT 与时间验证的原始来源。未改动已保存数据、模型或预测，未启动新实验。

完善后的主线：共享一个价格底座，正文100词/冻结FinBERT PCA16只学习新闻增量；静态组合包含“零修正”，再有限尝试预测分支损失的小型条件选择器。价格/修正/gate/校准都使用分层前向账本，禁止把训练内结果当作后层监督；保留暖启动、样本不足与异常上下文的明确回退。概率校准单列，冻结回放与使用成熟标签的在线更新分开比较。

新增针对性控制：相同offset/样本/变换下允许或锁定价格修正；旧500词模型仅作历史参照。AMZN原始覆盖与过滤损耗单独审计，扩展候选新闻时另建输入分支；关系抽取在独立检查通过后再进入相同文本预算的对照和最终流水线。现有40篇独立审阅仍待完成，不宣布通过。免费新数据调查限时，新保留期先锁定至少60个共同交易日。

方案包含 N00—N06 实验矩阵、有限参数、时间成熟规则、BA/Brier/类别召回与配对日期块评价、最多64窗口案例面板、代码接口、关键测试、停止条件、报告和离线demo。披露冻结FinBERT是事后可获得的历史表示工具，监督时间检查不能认证预训练语料无重叠。

交付：`outputs/stock_integrated_4h/next_plan/PLAN.md`。状态为方案完成；新残差/条件gate尚未实现或训练，没有新增表现结论。

## 2026-09-15 — 执行四小时共享价格、新闻增量与有约束选择器

**为什么做：** 旧融合的各分支重复包含价格，AMZN覆盖不足，后续赢家不同且不能事后挑选。用户要求实际执行N00—N06方案，尤其实现“证据不足不调整”、独立覆盖审计、抽取质量与预测收益分别验收。

**实际改动与训练：** 在 `stock_adaptive_4h/` 建立完整前向训练→冻结推断→指标/案例→demo链。价格/标题baseline重现；正文100词与冻结FinBERT/PCA16对固定价格logit学习增量；包含零修正的静态组合；Ridge条件损失选择器向静态收缩，历史/分股新闻样本不足或异常上下文回退；校准单列。保留允许重新调整价格的联合对照。分层OOF监督全部只用此前成熟标签。

核心第一版有效运行 `runs/v3` 完成233次实际拟合。后续AAPL gate BA51.52%、Brier .3041，AMZN46.20%、.2677；标题baseline分别51.89%/.2694、48.92%/.2772。第一版gate相对静态后续AAPL修复0例、引入2错，没有证明条件选择增益。

**新insight与针对性第二版：** AAPL过去OOF实际上涨58.64%、价格平均概率49.91%，后续实际上涨48.88%、价格平均概率58.66%。第一版新闻修正截距正文+.594/语义+.417，后续100%/98.31%修正向上；说明新闻分支会吸收过去价格校准/类别偏差。另登记ZERO_BIAS_PROTOCOL后仅把新闻截距固定0，完整重训231次；AAPL正文BA51.57%→54.09%、Brier .3076→.2801，概率质量仍差于标题baseline；AMZN无收益。静态/条件选择最终均回到价格，BA51.79%/46.20%；两版预设完整系统继续条件均不通过，最终保留标题baseline。没有按后续54.09%事后选赢家，也未继续扩大网格。

**在线机制：** 两版均执行固定专家、过去20个完成交易日误差与训练先验收缩的更新，给价格/标题提供同信息截距更新对照。每版408次小截距拟合，另408次纯未来标签扰动核验。零截距版在线后续BA53.49%/46.20%，仍无两股稳定改善。报告保存共同日期10000次配对区间及1/5/10日块敏感性，非探索选择校正后的显著性。

**覆盖与抽取：** 扫描78055原始索引、读取原窗口时点范围内14105篇正文。先读宽规则16候选，发现比较对象和他司评级误标，修复并新增回归测试；再读严格规则8篇AMZN候选（部分重复）。最终AMZN训练覆盖258→286/499、开发73→73/126、后续84→99/179，新增候选分别35/2/21篇；只是潜在覆盖，新文章没有进入模型。评级/目标价分离、对象/单位/期间/冲突/历史/证据位置/未知字段实现；65行独立检查0行完成，质量未过，不训练关系或Graph。原40篇正文复核仍待完成。未建立合格免费新增小时数据或新保留期。

**案例实际阅读：** 固定64卡，实际读16窗口、23篇节选；明确区分正确来自价格回退还是新闻修正，观察到历史持仓、指控与否认、跨公司、重复事件及正面事实但四小时下跌。详见CASE_REVIEW.md；不声称所有卡已人工审阅。

**核验与交付：** 两版共464次主拟合，保存权重与fits/OOF/候选/哈希。27项边界回归测试通过；各自独立核对1607标签、609无标签冻结推断，重载概率误差0，选参/权重重算、时间/缓存检查、未来标签扰动和无新闻精确回退通过。FinBERT未微调；本地CPU核心训练约9秒/6秒，不含准备和核验，无需Sol。浏览器实际验收本地demo，切换AAPL/AMZN、无新闻回退与独立显示结果均通过。

**失败保留：** v1共享内存改写上游概率导致中断，已修复并测试；v2元数据NumPy类型字符串问题修复为v3，概率重现一致。覆盖时间解析、宽规则、报告相对路径和demo旧同名模块导入问题修复并保留记录。没有使用失败运行成绩或覆盖旧实验。

**结论/下一步边界：** 完整系统已实现且可复现，但没有稳定两股增益。抽取与覆盖缺口只有部分代码/候选层面改善，须独立复核后再按同信息协议比较关系输入。方法门槛失败因此不进入新高级神经探索。总报告 `stock_adaptive_4h/REPORT.md`；可复现命令 `README.md`；质量表 `coverage/v3/independent_check.csv`；状态 `STATUS.json`。课程报告可以据这些有对照的失败与限制形成完整结果，不将工程可靠性当作涨分。

## 2026-09-16 — AMZN机制审计、先设计后阅读的案例诊断、近期论文

**用户问题：** 之前机制是否全部尝试、AMZN为何总低于50%、先构思case study再做、近两年论文能否提供大幅提升方法。

**执行边界：** 本轮无新训练、无改标签/数据集/选定系统。核对旧模型、预测及代码；先写 `stock_adaptive_4h/amzn_diagnosis/PROTOCOL.md`，再选择病例。对AMZN305行（开发126/后续179）重算类别召回、新闻覆盖、逐月BA/Brier及按完整类别分母相加的配对BA分解。

**实际案例：** 三组比较×两个时期×四种对错结果，固定SHA256选取24窗口、24不同日期；13有新闻、11无新闻。阅读全部标题及20篇正文节选，每篇最多5500字符；先隐藏结果记录内容笔记并SHA封存，再揭示收益/概率。不是独立盲审，不能从按对错分层的面板估计全体失败原因比例。保存旧全文模型24例线性贡献并重现概率误差<1e-12；没有把正确预测等同正确语义理解。

**关键发现：** AMZN旧全文53.20%、旧语义52.22%，不是一直低于50%。最新gate46.20%在全部305行与价格的最大浮点差1.11e-16、方向完全相同，因为所有新闻C候选训练Brier未过门槛。零截距正文C=.01训练月均BA57.53%，Brier比基础恶化.003847，超过.002被禁用；提示需检验分支校准/有限收缩，而非看后续结果放松约束。旧全文后续11月44.44%、12月58.95%、1月52.21%，概率Brier.2608仍差于常数.25，没有稳定优于随机证据。

**纠正旧描述范围：** 上次日志“全文正确被融合改错20次、反向12次”属于全部179行；有新闻84行实际是12对7，无新闻95行8对5。无新闻强制回退可以丢掉全文联合训练的价格部分；C11全文52.87%正确，回退价格45.04%错误，文本贡献严格为0。C04反向情况也保留。全文相对标题的BA增量主要在有新闻部分，但相关性不是因果证明。

**模型理解核验：** C15 EPS超预期、营收/指引不及预期，全文对、融合错；最大词贡献却是direct/momentum/year等，不能声称模型已学会预期差。C14政府支持否认芯片指控后仍下跌2.30%；C12盘后财报预告落在目标结束之后；C21迟到13—14天的报道存在但模型仍对。输入缺陷、语义错误、预测错误分开记录。

**实现完整性：** N00—N03/N05核心已跑，两版464主拟合；N04只有覆盖/抽取代码与候选，65独立复核仍0，关系输入未训练。新注意力/可学习投影LoRA/GNN未进入；此前容量、GRU、LoRA、PPO主要是1h，未统统移植4h。发现原计划N00“过去OOF单分支”漏列；先补登记ADDENDUM后用保存3—8月OOF选择，AAPL正文、AMZN价格，后续51.74%/46.20%。AMZN旧正文OOF50.10%，不能提前知道后来53.20%。补齐不改变原选择。addendum_v1因pickle的__main__.Transform解析错误中断，v2修正绑定；无重新拟合。

**论文与下一步（未新训练）：** 核对2025/2026的GS-Fuse、StockMem、RETuning、CAMEF、FactorGCL及情绪迁移论文。优先事件新增信息/预期差及低维内容感知效用gate；现有元信息损失gate已做，不重新包装。StockMem删日收益±1%内样本，RETuning三分类F1相对提高不等于我们的BA，GS-Fuse宏观资产回归不等同两股方向。没有可承诺大幅涨分的直接证据。建议先有限分支校准/收缩，再验收关系信息，最后有条件内容gate。

**交付：** `stock_adaptive_4h/amzn_diagnosis/REPORT.md`、`MECHANISM_STATUS.md`、`CASE_INTERPRETATIONS.md`、`RECENT_PAPERS.md`、analysis_v1原始表/封存笔记、addendum_v2模型贡献与补充对照；原实验文件哈希核验不变。

## 2026-09-16 — 继续向GPT 6 Pro讨论当前四小时诊断

按用户授权整理机制、实验、案例和论文。打包1,161项当前代码/权重/预测/输入/价格/案例证据，ZIP 88,352,332字节，保留manifest；准备讨论提纲和合订简报。原对话6aa9cf2c-c984-83e8-969e-f19b009285ff在Chrome显示6 Pro。附件选择受文件URL权限限制，未更改权限；系统窗口操作遇到用户正在交互后中断。随后改用同一对话直接发送9,194字符中文讨论正文，包含全部24例摘要、训练期候选、两股结果与论文，并明确新ZIP未上传。页面已确认正文提交、输入框为空、停止回答按钮与Pro思考中。没有把附件准备当上传成功，也没有将Pro尚未给出的观点写成结论。记录：gpt6_pro_review/adaptive_4h_followup/SUBMISSION_RECEIPT.json、DELIVERY_STATUS.md。

## 2026-09-16 — GitHub 版本管理与审阅入口

用户授权用 GitHub 管理并推送项目。本轮保持历史实验代码、权重和分数不变，新增根 README、当前四小时状态、ChatGPT 审阅提示词、实验索引、复现层级、更新流程和 AGENTS.md。代码/Markdown/环境记录通过忽略规则纳入版本管理，17 个关键结果文件采用显式清单，1,929 个本地模型/缓存文件记录大小与 SHA-256。原始新闻、课程文件、环境和大模型资产不上传；克隆仓库不等于具备全部重训数据。新增标准库仓库检查和 GitHub Actions，检查 Python 语法、结果哈希、文件范围与常见凭据模式，不冒充训练验证。可见性尚无回复，按私有方式创建以完成已授权推送；未公开课程资料。远端同步状态以 Git HEAD 核验为准，不等于 ChatGPT 已读取。

## 2026-09-16 — 小型LLM适配与四小时接入方案（未训练）

根据用户side chat建议，重读Qwen试验/四小时状态，确认旧40条中37条earnings且它为第一选项；位置偏差待诊断，不认定为原因。核对M5、24GiB、MLX0.32.2/mlx-lm0.31.3及Metal可用；未运行新微调。发现本机ChatDataset未显式传non-thinking参数，计划在项目loader统一训练/推理模板。新增SMALL_LLM_4H_PLAN：修输入→冻结小模型→早期分组监督数据→1.7B QLoRA→证据核验→固定价格＋文本底座的事实增量。分离抽取质量和四小时BA/Brier；无事件精确回退、嵌套过去监督、独立复核与样本不足停止。保留预算/候选/真实耗时测量，不保证涨分。本轮仅方案和环境核验，不修改历史实验。

## 2026-09-16 — 执行小型LLM诊断、M5 QLoRA与接入保护

实际运行12案例×3选项顺序，12案例类别均变。阅读43真实文章—公司输入的296片段，形成助手暂定标注：22训练、8开发、11检查，另2同事件诊断；未提供目标四小时收益/标签。原监督仅7个正事件训练文章，AMZN无正例，不冒充300–600合格样本。实现多事实JSON、精确证据位置、标签指纹、统一非thinking模板和assistant-only loss。

M5本地完成32步smoke与两版各66步QLoRA，合计164 micro-steps、22次更新，458752可训练参数；峰值约2.9GB。验证基座哈希不变、adapter改变、重载误差0。普通微调开发9事实匹配0；正例加权匹配1，但仍混入历史证据，检查期3个正事实匹配0。没有使用Sol，也未宣布微调成功或LLM路线无效。

训练期人工证据输入7例从0/8到2/8；另登记历史列表截断，移除16篇中的58句，无暂定gold证据被删，冻结模型开发1/9。按用户后续问题再执行固定8开发案例的prompt/context二乘二：原上下文＋原prompt0/9，原上下文＋短prompt0/9，截断上下文＋原prompt1/9，截断上下文＋短prompt0/9。简化版还改变了few-shot配置，因此是提示方案比较，不能证明纯长度因果。输入与提示确有影响，但未解决抽取质量。

实现质量门槛、低维事实/无截距修正接口、过去OOF检查与精确回退。独立复核为0，类型正例数量不足，按质量停止条件未执行全量新闻推理或新的四小时模型训练。原1607窗口保留；1488条已有预测（含609开发/后续）回退后原价格＋标题概率不变。17项测试和296个原文span核验通过，不是新的BA结果。

保留失败：prepare_v1过早按文章去重丢AMZN→v2修复；检查补3篇盘外评级形成v3，明确非随机总体；smoke_v1在bfloat16哈希转换前失败，无训练，v2修复；integration_v2复核包绑定实际候选输出，避免把助手gold正确性当模型质量。代码/汇总结果入Git，原文/标注数据/权重留本地。报告stock_llm_4h/REPORT.md；300–600独立标注、8bit/thinking、更多种子和完整B0—B3回测均未完成。

## 2026-09-16 — 扩大 GPT 标注与 M5 QLoRA（执行中）

**动机：** 用户要求继续标注和微调，检验此前22条训练样本不足、输入/提示设计和空输出问题。主预测目标保持四小时。登录恢复后，通过ChatGPT网页实际提交两轮GPT 6 High标注；原文、标签和adapter留本地。

**数据准备与纠错：** 初始300训练/60开发/100检查候选，按1—2月/3月/4月切分，抽取器最早5月1日冻结。原开发集60条全部为无事件，发现宽关键词把基金增减持当评级候选；保留这60条，按预先登记的更具体标题查询补38条3月候选。原4月有6篇正例（其中两篇为同一个Credit Suisse事件），另按同一内容查询补39条检查挑战，单列结果。最初补充说明误写只有1篇正例，已更正为1篇AAPL、5篇AMZN；旧说明快照留存于本地，不把模型一致性当正确率。

**实际标注：** 537候选×两轮=1074份输出；字段一致517/537，证据ID一致416/537。逐项审阅分歧、固定10%同意项抽查和额外AMZN样本，共183条助手暂定决议。25条不确定、4条冲突不转成负例，50条近似事件重复排除。最后249训练（60正例，AMZN13）、87开发（19正例，均AAPL）、122检查（22正例，其中AMZN5）。537条的4546段原文位置和日期/分组约束核验通过；同一事件的完整语义独立性仍不能保证。

**训练：** 第一版实际QLoRA正在M5上运行，249篇、固定3轮、rank8/最后8层q-v、458752可训练参数、LR1e-4、累积8。第一轮开发20事实匹配0，第二轮3；loss下降不等于任务掌握。精确核对目标函数：按完成长度平均后，空/非空分支那个token的负例总权重16.6、正例4.1482，约4倍，尽管正文章权重已占2/3。登记第二个有限对照：额外对事件存在分支施加类别均衡监督，保持其余设置不变；完成精确梯度/提示遮罩测试。后续结论以下方完成记录为准。

**范围：** 所有标签MODEL_PROVISIONAL，独立组员复核未完成。未把标注/抽取F1当四小时BA，原历史预测不改写。入口 `stock_llm_annotation/README.md`；协议/开发补充/损失对照分别登记。

## 2026-09-16 — 扩大标注与两版M5 QLoRA完成记录

**已完成：** 上节执行中的两版均已结束。普通版699.7秒、辅助版730.7秒，各747个micro-step、96次更新，458752可训练参数，MLX峰值2.753GB（非整机内存）。相同249篇训练、顺序、seed、模型和预算；非零梯度、adapter改变、基座不变、重载logits误差0均已验证。开发集分别选epoch2、epoch3，没有用四月检查输出选模型。

**实际对照：** 122篇检查共23个暂定事实。规则TP8/FP29；相同简短无few-shot prompt的冻结Qwen TP0、仅9/122格式和字面证据有效；普通QLoRA TP6/FP8/FN17、120/122有效；辅助版TP1/FP1/FN22、122/122有效。微调改善了输出约定，却未得到足够语义质量。辅助分支监督没有解决失败，不再追加同机制网格。完整分股票、原面板/挑战结果在METRICS.csv；不把富集面板当自然分布，不把两轮GPT一致当金标准。

**案例与context：** 阅读两个adapter全部7条匹配及9条错配事实的证据，确认历史券商列表误当当前、共识误当动作、新设目标价误当上调、抽出另一个机构数字、擅填旧值等。辅助版修正1个旧值错配却失去普通版6个正确事实。537输入最长939 prompt tokens，无截断，训练与MLX推理前缀全部一致。说明不能把本轮失败归结为context长度；输入内容与prompt设计仍未穷尽。

**检查与边界：** 20项新单元测试、17项既有组件测试、MLX精确梯度/遮罩检查通过；537条/4546原文span、训练/开发封存、预算/数据匹配、开发选epoch重算、权重重载核验通过。两轮标注和183条助手决议仍为MODEL_PROVISIONAL，独立人工复核未完成。完整事件分支因实际抽取质量不足和原质量门槛未过而不晋级；未运行全量新闻抽取或新的四小时下游训练，没有新的BA/Brier，历史结果保持原样。

**交付：** stock_llm_annotation/REPORT.md、CASE_NOTES.md、README.md、三份协议、代码及汇总证据。原文、教师输出、标签、模型及逐例输出留work目录；公开仓库仅发布代码和已检查的汇总。后续若继续，优先限定动作的证据分类/数值规则，或另立固定few-shot与约束输出对照；这些尚未执行，不能宣称已改善预测。


## 2026-09-16 — 换9B、改prompt、分步抽取实际完成

**动机与协议：** 用户要求依次换模型、改prompt、拆小任务。保留1.7B历史实验；固定Qwen3.5-9B 4bit revision 8b2b98c00a6b4d291155e4890773ca8f769aee53，在封存student_v3的87开发＋122检查输入上做三组。原消息内容保持；第二组检查清单＋5个合成示例；第三组选择当前目标事件句→分类动作及原文子句→程序读值。第三组无few-shot且最多两次调用，是整套方案对照，不把效果单归因于分步。所有标签仍为暂定，四月检查已暴露。

**实际执行：** M5/24GB、MLX0.32.2/mlx-lm0.31.3，本地705次调用，三组分别11.0/13.3/11.1分钟，总约35.4分钟，累计MLX峰值6.24GB。greedy、nonthinking、每次512输出token、总预算4096；所有原消息实际小于旧2048预算，无输入截断。没有新训练、付费API或Sol运行，没有新四小时BA/Brier。保留输入/代码/模型哈希、完整私有调用与失败记录。synthetic smoke先发现枚举占位符照抄，真实数据前登记修订；sandbox Metal失败的smoke_v1保留，非模型能力失败。

**结果：** 检查23事实，旧规则TP8/FP29、旧冻结0/18、旧微调6/8；新9B原提示0/2、新prompt 3/11、分步9/15（FN14，F1 38.3%）。原/新提示分别85/122、96/122整篇格式与字面校验通过，分步最终118/122；最后一项不代表所有中间输出正确。开发20事实，三组TP1/3/4。仅凭开发D0014修复“by $2 to $168”规则，TP从4变5；重放不新增调用，检查结果不变。

**案例与原因：** 固定21篇及额外匹配/定向诊断共29篇输入已审阅。原/新提示有把1750正确数值写成不含逗号而被拒绝的情况，不能说0严格匹配代表完全不理解。直接放宽整篇校验会使逐事件FP达到67/126，因此不能简单删除校验。分步仍把静态评级当维持动作、历史列表当当前事件，标题单独选取会漏旧值；“from a hold rating to a sell rating”被程序误拒绝。检查正例未整篇丢于gate，后续动作/规则也有损失。AAPL分步F1 25.8%低于旧微调34.5%；AMZN62.5%高于25%，仅6事实且有同事件重复，不宣称普遍提高。

**事后正确性修复：** Q0032暴露词表不认识Mixed，误把Positive当新评级。独立安全重放拒绝不完整评级对子，Q0024与Q0032各删一错字段，保留Q0024正确overweight；TP9/FP13/FN14，F1 40.0%。safe_v3/v4均保留，后者增加防止数字对子误触发的合成测试，实际结果相同。此修复受检查启发，单列而不替换注册实验，不恢复漏抽事实。独立人工验收未通过，不晋级全量预测。

**交付与下一步：** stock_llm_model_compare包含协议、代码、指标、质量统计、运行/版本证据、固定案例及安全修复记录。先补可用合成句验证的解析器语法/格式等价，再检验逐事实当前性与标题正文证据合并；须有限新协议与新的检查材料，不能继续把这批迭代当独立验证。原历史分数/权重不改写。

**最终检查：** 新增16项、原标注20项、原组件17项测试通过，共53项。原组件测试须使用finbert-env；系统Python缺numpy、structured-env缺scipy的启动失败已定位为选错环境，换回已声明的环境后通过。仓库178个Python源文件语法检查通过；42份历史选定结果哈希全部保持不变。


## 2026-09-16 — LLM直接预测四小时（执行中）

用户要求从抽取转向直接预测，并确认先价格、新闻、联合三组。已登记stock_llm_direct_4h/PROTOCOL.md与精确英文prompt；保持1607原样本，准备全部输入，对609非训练窗口运行冻结9B。截止为区间起点前5分钟，预测区间开盘至四小时结束方向；未来开盘价不输入。重建六段已完成交易小时并核对原价格特征/标签，新闻保留原候选但固定最近6篇、标题240字符＋原文600字符片段，记录省略和原文位置；不使用抽取成功门槛。

已完成60次价格/新闻/联合LR拟合，C仅六月至八月训练期向前选择，保存模型与重载概率一致证据。冻结LLM合成smoke三组通过；14项白名单、未来时间拒绝、数值/方向一致性测试通过。正式三组最大输入1044/2601/3194tokens，均未触发6144总预算。正在M5进行实际全窗口推理，未取得完整成绩；不宣称提高。无微调、无历史示例。本次LLM语义可受预训练历史记忆影响；所有市场时期已暴露。

**执行中接口诊断：** 前42个串行价格输出有11个p=.5但DOWN，尚未连接LLM预测与真实标签。原协议失败回退保留；另登记明确方向与原始数值概率诊断，不把两个不一致输出包装成同一校准系统。为控制耗时，登记批处理工程检查；8个先前价格输出的概率/方向/有效性全部一致，原始解释仅2个逐字相同。保留160条串行输出（552.9秒）后停止，三组在batch_v1全部重新运行，不拼接结果。批处理基准与串行曾同时占GPU，不能用该基准时间估算独立加速比例。

**V2输出接口变更（未按结果选prompt）：** JSON价格组609完成，新闻96后停止，联合0；不声称JSON版完整。未连接LLM输出与真实市场答案时已观察价格概率大量固定.5、DOWN与概率矛盾。因此另登记CHOICE_PROTOCOL：同样输入/模型，去掉示例数值和理由要求，只回答UP/DOWN，程序读取两token真实log概率并条件归一化，三组全部在choice_v1重新做。两token均为单token（3024/21854）。合成smoke_v1发现低精度全词表log概率导致UP+DOWN质量略大于1，正式V2前用float32全词表重归一化修复，smoke_v2质量<1且相对偏好一致。未训练LLM权重，未以行情分数作选择；V2概率不是已校准的市场概率。

## 2026-09-16 — LLM直接四小时预测完成记录

**实际运行：** V2价格、新闻、联合各609，共1,827窗口全部完成；实际推理耗时1111.9、1425.1、2136.2秒，合计约77.9分钟。固定9B/4bit、nonthinking、seed573、本地MLX；无LLM权重训练。另60次LR拟合（训练期六月至八月选C）及六个模型重载核验完成。原JSON部分实验与合成smoke保留，额外成本见ENGINE_HISTORY，不混入V2成绩。

**成绩：** 后续AAPL/AMZN BA：价格47.09/48.84，新闻55.65/52.54，联合50.08/51.21%；原标题baseline51.89/48.92%。新闻Brier0.3684/0.3354，联合0.3576/0.3077，都不如恒定.5的.25。新闻AAPL开发49.54%；AMZN十二月58.65%、一月44.47%，没有跨时期稳定胜出。旧FinBERT与全文及相同输入LR全部保留，不事后选赢家。

**观察与解释：** 价格版方向与历史均值符号高度一致；新闻版AMZN95个无新闻窗口全判下。苹果新闻→联合18个变化，4改对14改错，BA降5.57个百分点；亚马逊的67个变化中60来自无新闻窗口，有新闻的7个变化仅2改对5改错。新闻相对标题baseline的后续BA差虽为+3.76/+3.62个百分点，描述性五日块区间均跨0。新闻区间为看到成绩后补充的分析；联合区间原已计划，均不作独立显著性证据。

**案例与输入：** 固定16例全部实际输入已审阅，覆盖两股、两时期、四种对错格。旧行情回顾、长期观点、当前风险、重复报道和其他公司内容并存；字符剪裁可能漏掉核心信息。六个完整小时在早盘常停于前日，盘中也距截止55分钟。单例正确不能验证因果、推理或99%信心；V2没有生成理由，不编造模型思考过程。补最新五分钟信息、改善完整句段、过去数据上的校准/融合只是下一轮候选，本轮没有执行。

**验证与交付：** 1,607输入未来边界、1,827预测提示白名单/真实log分数重算、模型与代码哈希、60次过去拟合及模型重载记录通过。17项新约束测试通过，51份旧结果哈希保持不变。输出REPORT、CASE_NOTES、INPUT_AUDIT、逐窗口/逐月指标、配对区间和运行成本，原文/权重仍私有。全部时期已暴露，现代LLM预训练记忆也无法排除；无新独立泛化声明。

**发布前检查：** 仓库刷新后70份允许公开结果哈希一致，390个文件/189个Python源文件检查通过；17项本轮测试再次通过，git diff空白检查通过。仅本轮代码、汇总与维护文档进入提交，历史结果保留。

## 2026-09-17 — 四小时完整段落、近期价格、共享模型与9B融合完成

**为什么做：** 直接LLM实验仍使用每篇600字符片段，旧六小时行情在盘中最多落后55分钟、早盘还可能止于前一日；既有融合的新闻覆盖与股票参数处理也不统一。用户要求在固定AAPL／AMZN、1,607个四小时窗口上实际比较完整目标段落、最近5–60分钟价格、市场因子可行性、独立／共享模型及统一融合，并保留所有失败版本。

**数据清单与停止分支：** 重新盘点AAPL 38,634行、AMZN 30,283行五分钟OHLC及78,055条新闻。课程文件未定义第七列，故只称`activity`，没有用于volume、VWAP或订单流。原始新闻有11,651条“抓取早于标注发布时间”的异常，派生输入使用两者较晚值作为保守可用时间。AMZN标签全部也带AAPL，确认覆盖不对称。本地无SPY／QQQ／XLK／XLY的2018分钟线，也无宏观和盈利consensus／actual。Alpha Vantage历史分钟月度属于premium；Polygon长期分钟访问绑定付费；Twelve Data未确认足够免费2018深度和可复现权限。按预注册停止M1–M3、F7、market PCA、HMM与Graph，没有用日频数据或伪造市场特征。

**段落输入P0–P3：** P0精确复用旧标题＋600字符输入；P1保持相同文章和token上限，改成完整目标公司单位；P2放宽到6,000 tokens；P3只用截止前已到达文章做在线近似事件去重并保留报道／来源数。机械检查覆盖32,106个训练期窗口—段落出现次数，原始span、目标公司与完整单位命中率均100%。固定16张盲卡中P0有106处字符断句，P1为0；P1没有超出匹配预算；P3删除168个重复报道出现。助手结论只通过工程门槛，独立复核仍为0。

覆盖差异很大：AAPL的803行中P1/P2/P3均只有6行零段落；AMZN的804行分别有422/402/405行零段落。全1,607行的目标段落出现数，AAPL由P1的6,629增至P2的19,429、P3的19,008；AMZN只有1,150、3,538、3,219。训练期机械审计另有P1/P2/P3纳入4,717/13,983/13,406个单位，对应候选16,032/16,032/15,341个；窗口间重复同文会重复计数，不能把这些数字当成独立监督量。

**近期价格与参数共享训练：** 从截止前已完成五分钟bar构造5/15/30/60分钟收益、振幅、实现波动、隔夜缺口、日内位置与完整小时年龄。1,607个四小时标签均用48根五分钟bar重算一致；无bar越过cutoff。R0旧六完整小时、R1加入近期、R2仅近期；S2完全共享、S3共享底座＋股票交互均用March–August前向折和同一三个C。实际完成200次LR拟合并逐个重载复现，约6.13秒。

| 价格方法BA | AAPL开发 | AAPL后续 | AMZN开发 | AMZN后续 |
|---|---:|---:|---:|---:|
| R0 | 49.59% | 51.79% | 50.09% | 46.20% |
| R1 | 56.21% | 53.87% | 51.11% | 50.85% |
| R2 | 55.48% | 51.12% | 51.86% | 50.76% |
| S2 | 58.16% | 49.27% | 60.30% | 53.62% |
| S3 | 62.83% | 48.04% | 60.76% | 50.46% |

R1四个单元BA均超过R0并高于50%，但AMZN两段Brier反而更差；共同交易日1日与5日块区间都跨0。S2/S3在开发期很高，AAPL后续回落至49.27%/48.04%，显示共享参数在旧时期学到的关系不稳定，不能按后续为各股选择赢家。

**冻结9B全量推理：** 固定Qwen3.5-9B 4-bit revision、nonthinking、seed573与UP/DOWN token-logprob接口；prompt不含收益标签。P0共1,607行，其中609行复用旧封存输出、998行新生成；P1/P2/P3各新生成1,607行，总新分数5,819。各阶段生成耗时3,173.8／6,112.3／8,220.9／10,922.0秒，总进程约7.91小时；峰值MLX内存7.79／7.79／8.66／8.66GB，全部在M5本地完成，无付费API、Sol或权重微调。

同一prompt因batch组成不同出现小数值漂移，先保留所有观测分数，再按封存的相邻顺序只对逐字节相同输入复用前一版本归一化分数：P0→P1 395行、P1→P2 493行、P2→P3 1,391行。这个处理不读标签；最大观测差分别0.1225、0.1036、0.0912。原分数和归一化分数均公开，避免把batch漂移当内容效果。

**过去月份校准和无新闻策略：** raw、Platt和单参数temperature均只用过去月份拟合；校准类型由March–August全局Brier、再BA选定。P0/P1选择Platt，P2/P3选择temperature。严格R1无新闻回退后的训练期平均BA/Brier：P0 53.53%/.2461，P1 51.16%/.2476，P2 50.79%/.2482，P3 50.25%/.2482；按P0 Brier护栏选P0。校准显著压低过度自信，但后续P0四单元BA为44.45/47.62/47.40/51.66%，没有稳定方向能力。

完整段落的实际方向变化同时有修复和引入：相对P0，P1四单元改对／改错为8/2、15/15、4/3、3/1；P2为10/7、38/33、22/10、15/19；P3为9/7、38/35、21/10、15/19。输入质量改善和股票预测改善分开验收；去重没有形成一致增量。

**F0–F6统一融合：** F0是价格＋标题TF-IDF主baseline，无新闻严格回退R1；F1全文词特征，F2冻结FinBERT下游模型。非负0.25步长权重只由March–August OOF、同一Brier护栏选择。F3退化为body=1；F4/F5为body=.75＋LLM=.25；F6为LLM=.50＋R1=.50。没有根据September之后的结果改权重。

| 方法BA/Brier | AAPL开发 | AAPL后续 | AMZN开发 | AMZN后续 |
|---|---:|---:|---:|---:|
| F0 | 50.46%/.2601 | 51.89%/.2694 | 48.33%/.2710 | 49.79%/.2677 |
| F1 | 57.94%/.2500 | 51.74%/.2587 | 55.66%/.2481 | 54.36%/.2583 |
| F2 | 54.13%/.2803 | 56.71%/.2856 | 46.29%/.2977 | 54.27%/.2617 |
| F4/F5 | 52.92%/.2544 | 51.27%/.2557 | 52.50%/.2515 | 53.07%/.2530 |
| F6 | 50.35%/.2573 | 49.39%/.2529 | 52.50%/.2652 | 51.16%/.2545 |

F1是四个单元BA都高于50%的最清楚统一候选，相对F0却分别为+7.48、−0.15、+7.33、+4.57个百分点；仍不是四段都提升。F4/F5也四段高于50%，但没有四段都超过F0。F6训练期平均BA/Brier最好，到了AAPL后续BA49.39%。最终没有一个方法同时跨两股、两时期超过F0且守住Brier。

**案例与诊断：** 预先固定44个股票×时期×类别槽位，5个槽位无合格窗口而不替换，得到37个唯一案例。全部先记录输入再附标签；覆盖baseline错→新对、baseline对→新错、共同对／错、新闻／无新闻、P0→P1/P2/P3和近期价格改变。AAPL营收预警案例显示正文LLM可修复标题模型；AMZN偏弱指引案例即使保留了关键预期差仍错误看涨；Walmart竞争案例则LLM单支正确、融合错误。无新闻案例精确等于R1，证明没有继续使用LLM缺失先验。完整释义见`stock_nextgen_4h/CASE_INTERPRETATIONS.md`；它是助手审阅，不是总体错误比例估计或因果分析。

**工程核验与可复现交付：** 6,428条prompt截止、1,607个标签、200个价格模型重载、112个校准时间检查、338个严格无新闻回退、融合权重有限性和28组公开指标重算通过。生产代码／协议哈希封存，结果包括逐窗口、OOF、逐月、分组、校准、权重网格、转移、配对1日／5日块区间及执行成本。原新闻、prompt、权重和私有案例留在`work/`，公开CSV不含文本／URL／私有记录键。

**最终判断：** 工程目标已完成；方法层没有稳定统一胜者。最适合课程报告的主线是：F0作为主baseline，F1作为清楚的全文改进，R1作为行情新鲜度修正；LLM、共享学习和融合用于有控制的消融与失败分析。所有September 2018之后时期已暴露，下一次确认必须冻结方法并收集未参与设计的新时期，不能继续用同一后续标签挑股别赢家。

## 2026-09-17 — FinBERT事件适配与严格门控四小时修正

**为什么做：** 先前Qwen抽取常把历史分析师列表当当前事件、混淆评级维持与目标价改变，也没有完成“事件理解改善能否增加四小时方向信息”的公平闭环。本轮固定AAPL／AMZN四小时任务与既有窗口，先登记协议，再分别验收抽取与股票预测；development和later均保持已暴露回放，不能参与模型选择。

**标签与输入审计：** 重验458篇暂定文章及原文映射，train／development／check为249／87／122。训练正例AAPL 47、AMZN 13，共75个事实；实际字段仅有kind、action、old、new、unit、evidence_ids。全部body hash和span通过。密封重复链接没有两个端点同时进入接受面板；另做同股14日内字符n-gram保守扫描，发现1对近似重复且未跨split或前向折。标签仍是双GPT＋助手裁决，不是独立人工gold。

**预先字段效用诊断：** 将暂定事实映射到765个股票训练OOF窗口，仅15个事件窗口、12个非重复报道组、14个非重复事实；只有April一个可向前评价月份。基础与oracle BA同为47.37%，方向改变0次，Brier 0.26882→0.26910。因此没有“字段本身在多个训练月份可重复增益”的证据，按停止规则不扩大adapter网格。

**实际训练：** ProsusAI/finbert revision `4556d130...`，M5 MPS本地运行；A0冻结encoder只训9,228参数，A1解冻顶部两层共14,184,972参数，A2在顶部两层q/v加rank-4 LoRA共33,804参数。三个配置使用相同完整候选句＋编号目标公司上下文、三个时间前向折、seeds 573／574／575、损失与阈值网格。完成27个前向折和9个完整拟合，共36次正式训练；每次拟合时间合计5,700.39秒，当前恢复进程墙钟6,076.45秒。所有梯度有限并裁剪，checkpoint重载最大概率差0。废弃邻句上下文与缓存实现检查保留私有目录，未进入选择。

训练期综合分A0／A1／A2为0.0075／0.5202／0.3959，均选阈值0.35；协议选择A1、固定7 epoch。A0几乎全判no-event；A2召回较高但更易误报。没有依据April、development或later改变选择。

**抽取结果：** April暂定check的完整事实签名两股F1：规则16.29%、Qwen3-1.7B QLoRA 32.43%、Qwen3.5-9B分步38.30%、A0 0%、A1 79.17%、A2 53.52%。A1为TP19／FP6／FN4；AAPL F1 75.68%，AMZN 90.91%，但AMZN只有6个事实。层级指标中A1事件P/R/F1均为81.82%，type macro-F1 74.07%、action macro-F1 59.88%、evidence F1 57.97%。完整事实改善成立于暂定面板，独立质量验收仍未通过。

**数值与对象修复：** 目标价程序改成按目标公司与动作距离匹配旧／新值，新增多公司句回归测试。显式目标证据上的暂定回归诊断为评级60/60、目标价52/53；唯一失败需要前句绑定公司、后句给数字，部署门槛拒绝。这个解析器面板已暴露，统计不是独立泛化成绩。

**全语料与四小时残差：** 对1,641个候选文章—股票对、6,692个模型单元实际完成A0／A1各三seed共6次推理，约1,631秒。A1 ensemble输出456篇、912个候选事实。残差输入最多16维，严格`g=0`逐位等于F1；比较C0共享、C1独立、C2共享＋单个强收缩AMZN事件偏差，C为0.01／0.1／1。D4训练覆盖AAPL 160、AMZN 10个事件窗口，非重复事件334／5；AMZN未过30事件＋60窗口门槛。

训练OOF为D2、D3、D4全部选择BASE。D4在唯一通过Brier护栏的C=0.01下，C0／C1／C2月均BA为51.14%／51.27%／51.14%，均低于BASE 52.14%；C1损失最小但仍没有增量。D2所有非BASE均超出Brier护栏；D3覆盖不足。最终D2／D3／D4在全部1,374个可比较窗口严格等于F1，改对／改错均为0，1日／5日配对差值区间均为0。

**案例：** 预先固定的N0400历史／当前混淆、N0415评级维持＋目标价上调、N0417多公司对象三例都被A1正确修复；重复报道窗口保留一个两报道事件簇；无事件窗口D1／D2／D3／D4概率逐位相等。因为训练协议选择BASE，预设“抽取正确但预测改错”和“抽取错误导致改错”两类窗口不存在，按协议报告 unavailable，没有替换案例。

**最终判断：** 观察上，事件理解可以明显改善；观察上，这些事件事实没有给当前四小时方向提供训练期可验证增量。解释上，事件与市场四小时反应的监督太少且不稳定，尤其AMZN；这与计数一致，但不能证明分析师事件永远无预测价值。当前后续单股最高仍是AAPL F2 56.71%、AMZN F1 54.36%；F2后续两股描述均值55.49%但Brier较差且未参与选择。跨development／later四个单元最差BA最高者是F1（最低51.74%），所以F1仍是统一稳定候选；本轮唯一协议选择是A1抽取＋BASE残差。

**交付与核验：** `stock_finbert_event_adapter_4h/v1/`包含REPORT、DATA_AUDIT、CASE_NOTES、protocol、训练证据、逐窗口概率、BA／MCC／Brier、逐月／分组、覆盖、种子波动与配对区间。30个公开文件通过协议检查：1,374预测行、60主指标行、36唯一训练记录、无事件严格回退、源码／标签／缓存／checkpoint指纹一致；原文、暂定标签、sentence概率、缓存和权重保留在`work/`，不进入Git。

## 2026-09-17 — Phase 1文本校准、近期价格联合与完整新闻聚合

**动机与协议：** 用户批准参考GPT Pro建议继续执行。保留启动前已有的stock_paper_methods_4h预注册草稿，在新分数前补清：校准只作用有新闻分支，无新闻R1精确不变；C按旧原始分支前向分数选择以匹配对照；January–February只作warmup；September之后冻结。全体1,607窗口保留，765训练OOF、252开发、357后续参与评价；后两期已暴露。

**实际训练：** 复用5,078文章冻结FinBERT向量，重新拟合J0/J1全文、J2/J3语义、N0M/N1/N1M聚合模型。三个C、六个前向月、两股，含最终模型共266次LR拟合。端到端约11.2秒，所有模型保存私有目录并重载，最大误差3.33e-16。J0/J2复现旧F1/F2最大误差8.33e-17/1.67e-15。没有新的FinBERT或生成模型训练。

**观察：** J1对J0训练月均BA+0.76个百分点，但AMZN−2.16；J3对J2+1.59个百分点，两股Brier分别恶化0.0119/0.0240。N1对N0 BA−0.014个百分点；N1M对N0M−0.258个百分点。四机制均未过晋级线。N1M的AMZN后续BA57.32%是局部好结果，不能据此改选AMZN赢家或启动注意力。

**校准：** 正温度保持所有方向，F2四格Brier改善，AAPL/AMZN后续0.2856/0.2617→0.2503/0.2462；F1的AMZN开发Brier反而变差。只用过去OOF的全局规则为两分支选择正斜率Platt；后续AAPL F1变成全上涨BA50%，F2 BA49.32%。拟合的正截距失效是合理解释，不是已证明的唯一原因；没有事后改选温度。部分AMZN温度达到预设边界99.48，明确报告概率收缩而非语义进步。

**聚类：** F2全部新闻集合在截止前按到达顺序使用固定标题Jaccard及动作/数字/季度守卫分组。共271种多报道成员组合，338窗口768维均值改变；N0M/N1M使用相同六个元信息。按成员SHA固定抽查8簇均为来源后缀/标点变体；这是事后描述，非独立语义验收，正文冲突仍可能遗漏。原始向量、聚合向量和成员表不公开。

**结论与停止：** 概率尺度解释了一部分F2误差，但训练期选择的校准器同样跨期失效；近期价格联合与去重没有满足统一方向/概率门槛。按已保存协议不执行FinModernBERT、Chronos-2、注意力、TabPFN、Graph，不声称它们无效。课程报告保留F0/F1/F2与A1主线，新增温度概率消融和失败诊断。

**交付与检查：** 精简目录含入口、预注册、REPORT、114主指标行、逐月/三类覆盖、全1,607行预测（warmup概率空）、校准参数、选择记录、1/5日配对区间和模型manifest。核验标签/时间/过去选参/标题缓存revision/266模型重载/旧对照/无新闻回退/温度方向保持及聚类冲突守卫。没有修改历史实验，未宣布独立抽取验收通过。


## 2026-09-17：有限组合、AMZN聚合归因与课程交付

动机：验证近期价格与FinBERT的方向增益能否由自身温度改善概率，并解释AMZN事件聚合57.32%的来源。先保存stock_combination_4h/PRE_REGISTRATION.md，历史结果不覆盖；所有时期已暴露。

实际工作：复用已保存分类器，建立28个过去OOF温度记录，其中22次优化，其余identity；完成8次固定模型/替换向量推理。温度优化计算约1.30秒，不称为新FinBERT训练。生成四组指标、逐月结果、配对区间、按日贡献、8槽位6个不同窗口案例及609窗口离线回放。

观察：新组合后续AAPL/AMZN BA56.71%/53.79%，Brier0.2494/0.2462；相对F2温度的训练月均BA+1.59个百分点，但AMZN Brier恶化0.0220，故不晋级。AMZN N1M对N0M后续BA+2.58个百分点，改对8改错4，最大三天占净增量68.4%；BA区间跨0。固定模型换聚合向量不改变AMZN方向；最终C从1变为0.01，训练表示和正则选择尚未分开。

解释边界：温度修复置信度不改变分类；不能声称去重因果提升，更不能根据后续期选择股票专属赢家。案例核对标题和元信息，不冒充完整正文或独立人工审核。

交付：docs/COURSE_REPORT_4H.md已串联任务、方法、对照、案例和局限；成员信息及最终教师格式待补。demo为保存预测回放，非实时推理。独立重算54行BA/MCC/Brier，核验1607键、过去校准、无新闻严格回退、方向不变、609条干预端点、按日贡献与哈希；浏览器验证换股票、无新闻、标签揭示和换时期复位。停止扩大机制，保留所有失败结果。


## 2026-09-17：固定C归因与两种冻结基础模型

动机：用户授权将Fin-ModernBERT和Chronos-2纳入最后有限探索，不再以上一轮晋级失败为进入前提；同时完成聚合×正则四格，避免错误解释AMZN57.32%。新协议在模型结果前保存，旧记录保留。

实际：聚合56、匹配旧F2 38、Modern38、Chronos与同512历史LR76，共208次LR拟合。Modern冻结149,014,272参数，在M5 MPS编码5,078条原F2标题30.82秒；Chronos冻结119,477,664参数，CPU推理1,607窗口34.71秒。后者输入512根regular-session open/close，缺失不压缩，信息截止前才可用，目标开盘/末收盘分别第1/48或2/49步，跨任务cross_learning关闭。3个预测特征的概率由训练期LR学习，不把分位数当上涨概率。

结果：AMZN文章与事件C=.01后续方向逐个相同，均57.32%；C=1文章54.74%、事件54.21%。较强正则即可重现此前局部增益，去重没有该后续方向的独立贡献；强正则训练期BA并未稳定提高。Modern AAPL开发/后续45.51/54.52%，AMZN54.82/56.03%；Chronos LR分别50.00/50.55%、51.67/47.01%。Modern训练月均BA−0.06pp、Brier+0.0090；Chronos对等历史LR BA−0.64pp、Brier−0.0361。两者均未过门槛，不运行融合或继续加网格。AMZN好结果不用于股票专属赢家选择。

案例：按hash固定16个类别槽位；标题核对发现多年观点、混合正负信息、HTML撇号重复仍存在。Chronos正确案例中也有原始中位差向下、LR却预测上涨，不能把LR成功称作原始价格预测成功。Modern无新闻窗口严格R1；Chronos是独立价格分支，仍使用价格而不是新闻回退。

工程：缺少accelerate导致首次启动失败，独立runtime安装后解决；本地Pandas object键阻止NumPy默认安全恢复，迁移52个缓存为Unicode，原缓存与执行源码归档，数值逐元素不变，恢复复测通过。72行指标复算、208权重哈希/重载、旧F2误差1.73e-15、Modern/Chronos重载差0、禁止gap bar扰动、cache拒绝通过。原始价格anchor不进入公开时间审计。报告、案例、新demo与课程主稿增补已完成；全部暴露回测、现代模型预训练重叠可能、独立抽取验收未完成的限制保留。


## 2026-09-17：有限跨资产、A1稠密与历史状态探索

**为什么做：**按照GPT Pro建议，让新机制能作用于共同错误和AMZN无新闻窗口，而非继续按已暴露成绩挑专家；先保存stock_goal60_4h/PRE_REGISTRATION.md。主任务、1,607窗口和四小时标签不变，选参使用过去实际发出概率，关注两股较弱者的月均BA，Brier单独诊断。

**实际执行：**456次LR、28次浅层树、84次最终rank2交互头＋84次内部早停拟合；28次固定synthetic TabPFN条件化及新实例重载。原始FinBERT与三份既有A1冻结编码相同10,201单元；扩展历史编码13,521篇额外标题。比较六组价格、同输入原始/A1×线性/交互、标题8+8、H0/H1/H2和新F1/F2系统对照。没有新编码器训练、额外标签、RL或Graph。

**观察：**跨股LR外层两股56.88/57.06%，后续54.84/49.20%；A1稠密外层52.47/62.89%，后续54.34/46.65%。AMZN后续95个原无新闻窗口都可用历史新闻填补，但H1/H2后续48.72/48.67%。匹配价格后的8+8较弱股仅+0.21pp。全部未过预设晋级线，因此没有最终融合。F2_new后续55.14/56.79%，但AMZN开发49.17%，不宣称稳定改进。

**解释与边界：**覆盖增加不是新增独立事件，固定案例仍见转载、排名回顾和多公司内容；抽取任务适配也不等于市场方向适配。不能由这些失败证明不存在信号，不能用单股或单月60%替代目标。所有时期为已暴露回测；独立事件人工验收未完成。

**修复与验收：**前期运行私有存档；完善缓存指纹、Python兼容性、CSV round-trip和float64严格回退；发现8+8误用了R1而对照用OLD后，修为相同价格输入并重新执行，不保留混杂版本的晋级结论。222指标独立复算误差<=1.12e-16，596模型/条件化包哈希一致，原始标签、时间限制、重载、未来扰动、历史反应慢速参考核验通过。完整逐月/分股/覆盖/种子/1与5日块区间和案例已保存。

**外部数据：**Alpaca SPY/QQQ固定2018日期请求401、无凭据，用户尚未注册。认证后才能确认免费历史权限、feed和完整性；本轮不包含市场数据分支，不承诺它能涨分。下一步不扩大当前网格。


## 2026-09-17：小型价格自监督和LLM增量设计

**授权与动机：**用户同意尝试现有五分钟行情的自监督学习，并询问LLM能否解决新闻空缺/共同错误。先保存stock_ssl_4h/PRE_REGISTRATION.md，LLM部分仅形成设计，没有擅自提交网页标注任务。

**实际执行：**固定48根已完成交易时段五分钟线，跨夜明确标记，缺线不插值；B、展平PCA16、随机冻结卷积、掩码重建后冻结卷积四组。使用相同LR、过去月份全局C选择、三种子。1,988参数编码器共21次早停＋21次重新拟合，266次LR；CPU预训练/编码约103秒。最终每种子2,172片段、169日期，重叠片段不是独立样本。初次不同推理批大小的float32舍入导致严格重载断言中止，原运行已归档；改为相同批大小核对，模型/网格不变。

**观察：**SSL外层月均AAPL/AMZN54.72/57.29%，对照B55.04/60.87%。比随机编码的较弱股提高2.21pp，但没有胜过B；后续49.24/50.66%，没有稳定方向增量。RAW后续54.81/50.07%。最终重建验证误差约0.2036，零值对照0.2370，说明历史结构学习不等于四小时预测收益。AMZN固定95无新闻窗口修复旧F2错误10个、新增6个，净+4；不以这一后续子集创建路由。

**验收与决定：**24指标独立复算，21编码器/266头哈希、独立最终重载概率零误差，原键/标签/输入时点、前向选参、未来扰动、cache拒绝和精确回退通过。保存三种子、逐月、覆盖、1/5日块区间和固定案例。停止本轮，不扩大模型或声称所有自监督方法失败。

**LLM后续设计：**程序检索截止前同目标事件，LLM只判断重复、背景、动作/数字变化、冲突及未知，程序验证证据、对象、机构、期间和数值。是否已被市场消化不作为事实标签；GPT审阅不等于独立人工gold。价格特征和训练本地执行，网页标注尚未启动。


## 2026-09-17：历史新闻案例＋已实现收益的冻结LLM对照

**为什么做：**检验历史相似新闻与当时完整四小时结果能否帮助LLM预测当前窗口，区别于单篇预测及单纯加入旧新闻。先保存协议，按输入卡修正一次检索，不用开发/后续分数设计。

[报告](stock_analogy_4h/v2/REPORT.md)、[案例](stock_analogy_4h/v2/CASE_NOTES.md)。实际完成1393次冻结Qwen3.5-9B本地推理，约42.9分钟；没有新LLM微调。P0当前新闻＋价格；P1相似案例投票；P2历史案例无结果；P3同样案例加已实现四小时收益。

| 方法 | AAPL开发 | AMZN开发 | AAPL后续 | AMZN后续 |
|---|---:|---:|---:|---:|
| F0 | 50.46% | 48.33% | 51.89% | 49.79% |
| R1 | 56.21% | 51.11% | 53.87% | 50.85% |
| F1 | 57.94% | 55.66% | 51.74% | 54.36% |
| F2 | 54.13% | 46.29% | 56.71% | 54.27% |
| P0 | 54.28% | 56.40% | 51.21% | 56.15% |
| P1 | 57.68% | 51.11% | 51.57% | 51.38% |
| P2 | 55.02% | 56.40% | 50.63% | 56.68% |
| P3 | 54.28% | 57.42% | 50.66% | 56.68% |

- AAPL后续：加入历史收益比相同案例不带收益改对2、改错2；AMZN后续：加入历史收益比相同案例不带收益改对0、改错0。
- 未通过预注册探索线，不新增融合或按后续成绩挑股票赢家。
- 检索严格限于更早训练月份，September之后冻结August案例库。所有1,607窗口保留，1,374个OOF/开发/后续窗口评价。
- V1因输入卡显示工资/市值、产品/减持错配中止，保留44次调用及源码；未计算V1 BA。V2收紧标题事件动作和未知拒绝，仍有主体/竞争对手、模板背景混淆。
- AMZN开发/后续各仅2个窗口触发案例；95个原无新闻后续窗口精确使用R1。因此结果不证明已解决新闻缺失，也不能排除更好的语义检索。
- 时间、未来相似度干预、标签翻转检索不变、缓存拒绝、精确回退、指标独立复算和模型重载通过。原文/prompt/权重留本地，独立抽取验收未完成；所有时期仍是暴露回测。

**解释边界：**匹配对象/动作和市场状态尚不完善、训练库小且时间跨度有限，是有输入证据的限制；并非已证明所有失败都由它们导致。区分覆盖、匹配质量、预测增量；模型不生成解释，案例解释来自助手。
## 2026-09-17：市场状态审计与连续四小时收益辅助监督

**为什么做：**前一轮诊断指出，继续堆叠新闻编码器或历史相似新闻，无法改变大量价格与文本模型共同错误；下一条低成本、可解释的路径是检查市场状态数据，并把真实四小时收益作为辅助监督。主任务仍固定为AAPL／AMZN四小时方向、原1,607个窗口；没有改变标签、预测周期或历史结果。

**市场数据审计：**按预注册的四个固定日期（2018-01-10、04-27、09-04、12-03）和SPY／QQQ regular-session一分钟Alpaca请求运行了8个探针。当前环境没有API凭据，全部记录为`AUTH_REQUIRED_NO_CREDENTIALS`，市场分支按规则停止；没有用日频、合成ETF或目标股票代替。结果见`outputs/stock_market_return_4h/v1/alpaca_audit.json`。获得认证数据后才可继续该分支，当前不声称市场因子能提升。

**实际训练：**读取现有真实`target_return`，在每个过去月份内部拟合中位数填充和标准化，完成：C0价格分类LR、C1连续收益Ridge诊断、C2八维共享线性投影＋方向BCE＋收益Huber双头。C2使用lambda `{0,.1,.5,1}`和固定seeds 573／574／575；另做一组追加已冻结F1概率的`price_F1`对照。共360个候选／选定拟合记录，生成15,114个方法—窗口预测行；C2每个种子做了内存checkpoint重载核验，最大概率/收益差低于1e-12。没有把`target_return`或未来时间字段放进输入。

**只用过去选择：**跨月forward OOF（March–August）选择了`price/C0=0.1`、`price/C1=1.0`、`price/C2=0.0`，以及`price_F1/C0=0.1`、`price_F1/C1=0.01`、`price_F1/C2=0.5`。September–October development和November之后later分别报告，未参与选择。

**结果：**raw price分支C2在OOF、development、later的AAPL／AMZN BA分别为52.59／52.62%、52.38／53.06%、51.47／50.26%；同一分支C0分别为53.87／52.49%、55.10／53.15%、52.64／50.21%。`price_F1`分支C2的later BA为52.04／49.73%，C0为51.52／51.38%；连续收益辅助没有带来两股跨时期稳定增量。按协议真正的June–August筛选线（逐股至少+1个百分点、平均Brier不恶化超过0.002），四个候选全部未通过：raw C2的AAPL／AMZN变化+0.76／−0.61pp，平均BA仅+0.07pp，Brier+0.0008；price+F1 C2为+0.83／−1.51pp，平均BA−0.34pp，Brier+0.0020。C1两股方向BA均下降。报告中同时给出MCC、Brier、return MAE/RMSE/correlation、逐阶段和日期块配对区间。

**判断与停止：**这是实际训练和完整对照，不是只写方案。观察上，连续收益能在个别时期提供回报相关性，但当前输入没有把它稳定转化为方向增益；raw分支的OOF选择把辅助权重压到0，是停止扩大双任务网格的直接证据。解释上，仍不能证明收益监督永远无效，主要结论是这份数据和特征不足以支持更复杂组合。下一步优先级保持：若用户配置可验证的SPY／QQQ分钟数据，再做市场状态分支；否则不继续扩大RL、Graph、LLM或高容量搜索。

**交付与核验：**`stock_market_return_4h/v1/REPORT.md`、`METRICS_SUMMARY.md`、`metrics.csv`、`cv_and_metrics.csv`、`predictions.csv`、`training_evidence.json`、`promotion_gate.csv`、`paired_block_intervals.csv`、`choices.json`、`alpaca_audit.json`和`verification.json`已生成；`verification.py`通过键、概率范围、标签符号、窗口计数、特征排除、时间顺序、checkpoint重载和注册参数检查。原始输入、环境和模型文件仍留在`work/`，不进Git。
# 2026-09-17 — Recency weighting, dense windows, and article reaction audit

**为什么做：**在市场状态数据仍缺少可验证 SPY/QQQ 分钟输入、连续收益辅助监督未通过晋级线之后，执行一轮有限、可复现的历史信息实验。目标是区分“近期样本应否更重要”“增加日内窗口是否有增量”和“新闻在可用时间后是否有足够可测反应”，不继续堆叠 LLM 或神经模型。

**实际执行：**在新目录 `outputs/stock_recency_dense_4h/v3` 保存预注册协议、R1/F1/F2 四种精确半衰期（infinity/80/40/20 NYSE sessions）的加权逻辑回归、30 分钟 stride 的 48 根五分钟稠密窗口、文章 30/60/120/240 分钟反应审计，以及 activity 列审计。等权 R1/F1/F2 直接与已保存官方概率逐窗口对齐，最大误差为 0；加权训练和模型重载在 M5 本地环境完成。原文、embedding、模型和逐文章反应行保留在 `work/`。

**观察到的结果：**recency 选择为 R1=20、F1=80、F2=20，但 June–August 的 AAPL/AMZN BA 增量分别为 R1 +0.41/−7.40pp、F1 +4.12/−3.21pp、F2 +5.97/−4.31pp，均未通过“弱股至少 +1pp、无股损失超过 1pp、三个月至少两个月为正”的工程线。稠密 D1 相对官方日归一化 D0_day 为 AAPL +0.42pp、AMZN +0.78pp，低于每股 +1pp，文本扩展和 D2 按协议停止。文章审计覆盖 4,930 篇 accepted article、4,930 个 normalized groups 和 39,440 个 article×horizon rows；这证明可做后续 event-study，但不构成因果或预测提升。activity 为非负整数型描述字段，但定义/单位未在仓库元数据中确认，保持 `SEMANTICS_UNRESOLVED_NOT_USED`。

**核验与决定：**通过官方 keys/labels、等权概率精确复现、概率与指标重算、权重有限、dense 唯一性、未来 target-bar 扰动不改变 cutoff 特征、反应计数和 activity 未使用检查。结论是近期加权和稠密窗口没有稳定增量；后续优先级仍是获得可验证的 contemporaneous market-state 数据，不能把本轮局部改进包装成新泛化。
# 2026-09-17：v4 recency/dense 修正与全语料 reaction probe

## 为什么做

上一轮 v3 同时存在三个需要拆开的风险：F2 recency 的变换与 canonical J2 不完全一致；infinity 结果复用了旧概率而不是真正 refit；稠密窗口 gate 把 March–August 与 June–August 混在一起。新闻反应审计也只看了已接受的 4,930 个文章 ID，不能回答完整语料是否有足够覆盖。因此本轮先做可复现修正，再做全语料可行性审计和冻结 reaction probe，不继续堆叠新模型。

## 实际执行

1. 用 `outputs/stock_recency_dense_4h/run_recency_v4.py` 按预登记的 infinity/80/40/20 session half-life 运行 R1/F1/F2。F2 使用 canonical J2（文章向量 PCA16、窗口均值、`log1p(news_count)`、`has_news`）。infinity 为 all-one sample-weight refit。
2. 用 `build_dense_windows_v4.py` 和 `run_dense_v4.py` 生成 30 分钟起点的重叠四小时窗口，并先做官方/重建特征 parity。parity 不通过后，官方控制与增广数据统一使用重建发生器；gate 固定 June–August 六个月份行。
3. 从完整 78,055 条 raw news index 构造 `(article_key,target_symbol)` pairs，按时间、英文、同 session 反应和规范化重复组建 reaction dataset。gate 通过后运行 AR0/AR1，并在有限的既有冻结向量覆盖上运行 AR2；只对通过 horizon 做 W0–W3 下游对照。
4. 修复下游 predictions 输出在 metrics 循环中重复追加的实现问题；补充 `verify_v4.py`、`verify_v1.py`，检查 parity、6 行 gate、公开文件无正文、下游 key 唯一等条件。

## 结果与 insight

- infinity refit 与历史 reference 的 24 条 fold×stock×method 记录最大概率误差 `1.67e-15`，因此 parity 通过。
- recency 外层增量（AAPL/AMZN）为 R1 `+2.80/-7.40pp`、F1 `+4.57/-0.38pp`、F2 `-4.34/+1.16pp`，均未通过“每股至少 +1pp、无股损失超过 1pp、三个月至少两个月为正”的 gate。加权近期新闻改变了预测，但没有稳定跨股收益。
- dense D1 相对 matched D0_day 为 AAPL `+0.42pp`、AMZN `+0.78pp`，低于门槛；D2 停止。这个结果不能与 v3 的错误 March–August gate 直接比较。
- 完整语料 reaction gate：78,055 raw、89,958 candidate pairs、85,402 canonical groups；AAPL/AMZN 均有足够 60m/240m same-session reaction。AR1 文章级 120m/240m 通过，但接入四小时窗口后 AAPL W1-120 为 51.66% 对 W0 48.70%，AMZN 为 45.51% 对 W0 60.63%，没有两股稳定下游增量。
- “文章反应可预测”与“四小时方向可改善”是不同问题。当前 evidence level 只支持前者的有限 probe，不支持将 reaction 机制作为主模型。

## 限制与下一步

- 官方/重建 dense parity 的差异仍需在未来获得输入生成器定义后进一步追溯；v4 已用 matched reconstructed control 避免把差异当作收益。
- AR2 因模型二进制缺失只覆盖约 10.36% reaction rows；没有把它写成完整语义结果。独立组员事件/目标关联复核仍未完成。
- 按停止规则，本轮不继续扩大 reaction horizon、recency 网格或 D2；保留全部结果供课程报告说明“为什么机制没有稳定提高”。

## 2026-09-17：Stage 5 方法主张与校准审计

**为什么做：**Stage 0–4 已修复并验证 recency/dense、ModernBERT pooling
和 TabPFN provenance。剩余风险主要是把小型 paper-inspired probe 说成
完整论文复现，或把校准／去重的局部结果解释过宽。因此本轮只做 claim-only
审计，不追逐新分数。

**实际执行：**先固定 `outputs/stock_method_validity_audit/v1/PRE_REGISTRATION.md`，
再运行 `audit_claims.py` 读取四份公开校准清单，生成 420 条 slope 记录、
安全／不安全主张表和机器核验 JSON。没有重新训练、推理、标签选择、开发／
后续期调参或改写历史输出。

**发现：**早期未约束 Platt 的 84 条记录中 32 条为负斜率，全部来自 AMZN；
负斜率会反转分数排序，不能笼统称作单调概率校准。后续 constrained
`platt_shrunk`、temperature 和正斜率清单没有负斜率。该结果只改变解释，
不改变 BA/Brier 数值。

**主张收窄：**SSL 是 masked-reconstruction pilot 而不是 TS2Vec；analogy
是词法历史案例 probe 而不是 FinSeer；Event Adapter 是目标证据／动作抽取
probe 而不是 Ding event/graph embedding；Chronos 是 frozen endpoint probe；
Qwen 直接输出是 token preference；C2 只检验一个连续收益辅助目标；事件聚合
增益与正则化混杂；历史 F1/F2 与 F1_new/F2_new 已明确分开。独立人工事件
复核仍未通过或声称通过。

**交付：**`outputs/stock_method_validity_audit/v1/REPORT.md`、
`CALIBRATION_SLOPES.csv`、`METHOD_CLAIM_AUDIT.csv`、`verification.json` 和
`manifest.json`；当前课程报告、状态页、工作规范、修正日志和 ChatGPT handoff
均已更新。下一步只剩 Stage 6 全量 verifier、仓库刷新／检查、提交和 push。

## 2026-09-17：Stage 6 最终核验（本地完成）

五个公开 verifier（recency weight、recency/dense v4、ModernBERT v2、TabPFN
v2、claim audit）均返回 PASS；`refresh_repository.py` 刷新 378 个选定结果，
`check_repository.py` 检查 836 个文件并解析 285 个 Python source，
`git diff --check` 也通过。最终证据写入
`outputs/stock_method_validity_audit/v1/final_verification.json`。本轮没有
修改原始数据、模型二进制或历史运行。commit 后按要求尝试 push；remote
是否更新需由实际网络响应确认。

最终本地 commit 为 `ea621d5`（`Complete method validity audit and claim
corrections`）。commit 后再次运行刷新／检查仍为 PASS；push 因
`Could not resolve host: github.com` 失败，`origin/main` 仍是
`e32785d`。因此交付状态是本地完成、远端未验证更新。

## 2026-09-18：外部审查阻塞项与执行暂停

外部审查指出，较早 `e32785d` 实现中的 dense/reaction 路径仍有六类
阻塞问题：dense v4 的实际 gate 仍可能使用 March–August；reaction article
price context 可能包含未完成五分钟 bar；W0–W3 与原协议不匹配；reaction
promotion gate 不完整；AR1 数值／文本预处理和 AR2 表示需要修正；以及
reaction verifier 没有核验关键时间安全约束。上述内容已作为 ISSUE-022 至
ISSUE-027 写入 `docs/CODEX_WORKING_SPEC.md`。

按审查要求暂停执行。没有重置、删除或覆盖既有结果，也没有启动新的
ModernBERT、TabPFN、reaction extension 或其他模型实验。当前等待详细
reviewer addendum；在其纳入新的 repair protocol 之前，不将此前 affected
artifact 的 PASS 当作最终有效性结论。

# 2026-09-18：checkpoint evidence corrections and detailed blocker register

本次只更新文档和证据边界，没有运行模型、reaction 代码或新的实验。外部
审查要求已落实：dense v4 的可执行月份 gate 在 `METHOD_VALIDITY_AUDIT.md`
中降为 `NEEDS_RERUN`／`SUPERSEDED_PENDING_RERUN`；CORR-003/CORR-015 保留
历史 `+0.42/+0.78pp` 但明确 superseded；CURRENT_STATUS 的 v4 历史段落加入
reaction 120m/240m/W0–W3 predictive/gate 警告。

ModernBERT v2 与 TabPFN v2 的 current-facing 文案改为“未显示稳定的两股提升”，
并明确它们没有保存 exact promotion formula，因此不作正式 pass/fail 晋级声称；
没有重跑 encoder 或 TabPFN。详细 reaction addendum 要求拆分为
ISSUE-028—ISSUE-040，覆盖完成 bar、逐行 provenance、时间对齐、W0–W3、完整
promotion gate、AR1/AR2、future perturbation、grouping manifest 和新运行指纹。
该段是历史暂停记录；已由本节 2026-09-18 Phase A repair checkpoint supersede。

## 2026-09-18：Phase A dense/reaction repair and Phase B preregistration

从本地 checkpoint `dd836d80c709cd98065249ab5cdde233bd7abdc1` 继续，未改写
任何历史目录。新 dense v5 修复了 June--August gate 的实际过滤：6 个
stock-month cells 中 AAPL D1−D0_day 为 `+3.740pp`，AMZN 为 `−0.079pp`，
两股 gate FAIL。原 v4 `+0.42/+0.78pp` 仅保留为 superseded historical
diagnostic。June--August infinity parity 新增 42 行，最大概率误差
`1.67e-15`，通过。

新 reaction v2 从完整语料重建 89,958 个候选和 85,402 个 canonical groups。
完成 bar、逐行 used-bar end、reaction maturity、fold-local AR1 preprocessing、
target-pair manifest、future-price perturbation replay 和完整 promotion gate
均通过。240m AR1 article gate PASS；但 downstream W0→W1 AAPL
`48.70%→51.98%`、AMZN `60.63%→44.43%`，未作为四小时改进推广。AR2 因
缺少 target-context FinBERT binary 明确 `NOT_RUN_MODEL_BINARY_UNAVAILABLE`。
`verify_v2.py`、`verify_v5.py` 和 infinity parity verifier 均 PASS。

随后实现并静态测试 `outputs/stock_specific_gate_4h/` Phase B：固定 R1 与
F1_new、十个状态变量、G0--G3、加权 ridge、exact no-news 回退和 August
freeze；没有执行 `run_gate.py`，没有生成任何 G0--G3 结果。等待明确
`APPROVE_GATE_RUN`。

## 2026-09-18：reaction v3 addendum correction and full-gate rerun

The reviewer clarified that AR1 and AR2 have independent article-level
eligibility.  AR2 unavailability must not block a separately passing AR1.  This
was registered as ISSUE-041 and added to `PRE_REGISTRATION_v3.md` before the
rerun.  The first v3 article-only checkpoint is preserved at
`v3_article_audit_checkpoint/`; the corrected run is a new `v3/` artifact.

The corrected v3 run used 120 fold-local AR0/AR1 fits.  The 240-minute AR1
candidate improved June--August mean BA by AAPL `+1.763pp` and AMZN `+3.218pp`
and stayed within the per-stock Brier guardrail, but macro AUC changed by
`-0.570pp`, so the full gate failed.  The 60-minute candidate also failed.
AR2 is explicitly `NOT_RUN_MODEL_UNAVAILABLE`.  Because no candidate passed,
the exact W0=R1, W1=coverage, W2=five reaction fields, W3=F1+R1+five fields
downstream protocol was not run.  v2's old W0--W3 numbers remain a historical
protocol mismatch and are superseded for current claims.

`verify_v3.py` passed completed-bar and maturity safety, future-price replay,
target-pair uniqueness, fold-local preprocessing, model reload, gate recording
and the no-downstream-on-failure rule.  Independent human association review is
still not claimed.  Phase B G0--G3 remains preregistered but unexecuted pending
the explicit `APPROVE_GATE_RUN` message.

## 2026-09-18：reaction v4 complete candidate family and Phase B preflight repair

**Why:** the external review of `8842d37` found that v3 did not execute every
registered article horizon, represented missing pre-article price context as a
true zero without exposing validity flags to the model, and retained a known
ambiguous AAPL acronym. It also identified six implementation defects that
would make the preregistered stock-specific gate unsafe to execute.

**What changed:** v4 is a new, non-overwriting reaction artifact. It runs
30/60/120/240-minute AR0/AR1 candidates with completed-bar context; unavailable
returns and realized volatility are `NaN`, with explicit unscaled validity
features and fold-local imputation/scaling. The builder applies only a narrow
high-precision rejection for *American Association for Physician Leadership*
when independent Apple evidence is absent, recording five private review cards.
The Phase B preparation now creates the registered advantage target, traces the
real R1 private source with parity to the public column, implements exact R1
support fallback, applies median-based transforms identically at train/eval,
and measures routing headroom by direction disagreement. Its verifier exercises
the real 1,607-row preparation path and all nested fallback contracts.

**Observed result:** the v4 verifier passed. All four independent AR1 gates
failed: 30m mean BA changes were AAPL/AMZN `-0.996pp/-0.811pp`; 60m
`-0.351pp/-0.561pp`; 120m `+0.322pp/+2.570pp` but misses AAPL's BA/Brier
requirements; 240m `+0.336pp/+0.653pp` and AMZN collapses to one direction.
AR2 is `NOT_RUN_MODEL_UNAVAILABLE`. As preregistered, W0--W3 did not run.

**Boundary:** dense v5 remains current. v3 remains historical/superseded.
Phase B preflight passed, but it is not a predictive experiment: no G0--G3
prediction, metric, oracle or exposed-period result was generated. The project
waits for an explicit `APPROVE_GATE_RUN` before that separate execution.

## 2026-09-18：最终 code-only verification repair（无 Phase B 预测）

**Why:** the follow-up external review found four final evidence gaps: oracle
could illegally select text on no-news rows; the promised post-run independent
verifier had not been implemented; headroom lacked an explicit regression case;
and reaction replay did not mutate an unfinished bar that began before cutoff.

**What changed:** the future oracle now reports a routing-eligible perfect
switch separately from a full-system diagnostic that forces R1 for no-news
rows. `verify.py` keeps its preflight mode and adds a future post-run mode that
rebuilds saved predictions, metrics, monthly metrics, advancement, chronology,
preprocessing and coefficient checks without using runner metric helpers.
Preflight now checks .70/.60 versus .70/.40 direction semantics. Reaction v4
replay mutates every bar whose end time exceeds availability and includes a
10:02 synthetic unfinished-bar test.

**Observed result:** strengthened `verify_v4.py` and real-input Phase B
preflight both PASS. This did not retrain AR0/AR1 and did not execute the
controller. `outputs/stock_specific_gate_4h/v1/` remains absent: no G0--G3
predictions, metrics, oracle ceiling or advancement results exist.

## 2026-09-18：批准后的 Phase B 单次运行在 verifier 失败处停止

**Authorization and command:** the owner explicitly supplied
`APPROVE_GATE_RUN`. The one allowed command was executed exactly as registered:
`work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/run_gate.py --approve-gate-run --output outputs/stock_specific_gate_4h/v1`.

**Failure:** the immediately required independent command,
`work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/verify.py`,
raised `NameError: clean is not defined` in the post-run branch before creating
`v1/verification.json`. The run directory is preserved, but its output is
`UNVERIFIED_STOPPED`.

**Decision:** no scientific interpretation, report, post-result protocol
repair, score search, v2 run, or activity-column experiment was performed.
The precise failure record is public; raw predictions and model artifacts stay
local under the repository artifact rules. A future instruction must explicitly
authorize any verifier repair or additional action.

## 2026-09-18：仅 verifier recovery 仍因 advancement 精度审计停止

**Authorized scope:** 未重跑模型或 `run_gate.py`。唯一代码修改是把既有的
`clean(value)` helper 从 `preflight_main` 移到 module scope，使保存的 v1
结果能进入既有 post-run verifier。

**Integrity evidence:** 恢复前、修复后且验证前、以及 verifier 执行后的
十一份既存 v1 artifact 的 SHA-256 完全一致。初始 recovery 记录中的
`metrics.csv` SHA-256 曾遗漏最后一个十六进制字符，已作为 metadata transcription
修正为完整 64 字符值，非 artifact 改写。新增的仅是 recovery 元数据和
`verification.json`；预测、metrics、monthly metrics、advancement、系数和
训练证据均未改写。

**Observed verifier result:** `verification.json` 记录 17 项 PASS 和 1 项
FAIL：`advancement_independently_reconstructed`。独立重算保留完整浮点精度，
而保存的 `advancement.json` 经 pandas `to_json` 写入时约保留十位小数，和
verifier 的 `1e-12` 比较阈值不一致。因此 v1 是
`UNVERIFIED_STOPPED_PENDING_EXTERNAL_REVIEW`，不是可报告的模型结果。

**Actions not taken:** 没有 runner rerun、报告、调参、v2、activity-column
实验，或为消除此差异改写任何既有结果文件。

## 2026-09-18：Phase B verifier precision-only repair PASS（既有 v1）

**Authorized repair:** 外部审查仅允许 verifier 将独立重算的 advancement
records 经与 frozen runner 相同的 pandas JSON 十位小数表示后再比较。没有改变
任何 prediction、metric、fallback 或 expert-parity 的严格核验阈值，也没有运行
`run_gate.py`。

**Integrity and verification:** 十个既存预测/结果 artifact 在修复前、verifier
后和 report 后的 SHA-256 一致。保存的 full-precision 与 JSON value 的最大绝对
差为 `4.843306398299996e-11`；同样 storage normalization 后 records 严格匹配。
`verify.py` 的 18 项 post-run checks 全部 PASS，`REPORT.md` 只读取既有 v1
结果生成。

**Result boundary:** G1/G0、G2/G1、G3/G2 三个原注册 contrast 都未通过
promotion gate；因此不存在可晋级的 stock-specific reliability controller。
development 和 later 只保留为已暴露的探索性历史回测。没有 post-result tuning、
v2 或 activity experiment。

## 2026-09-18：Activity Incremental Experiment v1 preregistration and preflight

**Question:** 未定义语义的第七根五分钟字段 `activity` 是否能在 canonical
R1 之外增加四小时方向信息？本阶段只审计、冻结和验证 A0；没有计算 A1 或
A1_matchedC 的预测成绩。

**Audit:** AAPL 为 38,634 行、AMZN 为 30,283 行，均为无缺失、无零、无负值
的整数 activity。未找到权威定义，所以状态为
`SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY`。跨时间粒度的可加性现象仅作描述，
不把字段称为 volume。

**Preflight:** 新分支严格使用 15/60 分钟、八个冻结 feature、最多 20 个且
至少 10 个历史 session 的同时间点 references。13 项 verifier checks PASS：
1,607 canonical keys、完成 bar cutoff、历史 session、10:02 synthetic case、
A0/R1 parity 和 no-A1-result 都通过。A0 与 goal60 及 nextgen 保存 R1 的最大
误差均为 `1.1102230246251565e-16`，方向完全相同。

**Boundary:** `run_activity.py` 只有带 `--approve-activity-run` 才能执行。
v1 的 predictions、metrics、monthly metrics 和 advancement 均不存在；等待
外部审批，不运行任何 candidate 或 Phase B 变体。

## 2026-09-18：Activity v1 code-only final repair

外部 review 要求在任何 A1 运行前补齐 future runner、post-run verifier 和 report。
新 verifier 对 1,607 keys 的全部八个 activity features 直接从 raw five-minute
bars 和 schedule 重构；每个字段 NaN mismatch 为 0、最大有限误差为 0。它还验证
全部 frozen source hashes、严格 20-session membership、按月 coverage 和合成边界
cases。Activity v1 仍为 `PREREGISTERED_NOT_RUN`；没有执行 A1。

补充 ISSUE-062：future post-run verifier 现从 `predictions.csv` 本地重算
monthly/phase metrics、C selection、训练时序、A1/A1_matchedC 六格 June--August
contrasts，并与 runner 输出逐项核对。它尚未在 A1 结果上执行；该条是代码合同，
不是预测证据。

## 2026-09-18：Activity v1 runtime-only repair

外部审查发现未来 runner 的 frozen A0 `A0_C` 键错误、verifier 的 runner-global
月份依赖和 bool/float 选择检查，以及报告期标签问题。ISSUE-063--068 已以代码修复：
统一 selected evidence 的 `C` 字段、使用 verifier-local months、执行真实的 synthetic
selection/self-test、记录 training months 和协议哈希，并让 future verifier 重载每一个
selected model 后逐概率核验。未执行 A1，也未生成 candidate 模型或分数。

## 2026-09-18：Activity v1 approved single run and verifier recovery

已批准的 runner 仅执行一次。第一次 verifier 因 runner pickle 内的
`__main__.StandardizeMissing` 名称失败；恢复前保存了 10 个预测/结果文件和 64 个
模型文件的 SHA-256。A1/A1_matchedC 所有 selected/frozen 模型保持匹配；A0 私有
模型哈希改变，符合失败 verifier 调用 `reproduce_a0()` 的覆盖风险。verifier-only
兼容 loader 不改写模型，逐一重建 candidate scaler 和概率后 PASS。原始结果及
candidate 模型最终哈希仍与恢复前账本一致。A1 对预注册双股票 gate 失败：AAPL
六月--八月 BA 平均仅 +0.237pp，AMZN -3.662pp；因此没有 promotion。

## 2026-09-18：Context increment initial audit

新 lane 不运行真实 Mmeta/M1、news 或 joint scoring。Alpaca 文档端点的 bounded
SPY/QQQ raw SIP probe 在无本地授权凭据时正确停止为 `AUTH_REQUIRED`。原始 archive
metadata 全量扫描完成；`thread`/`ord_in_thread`/highlights 均保持语义未决，不作为
金融事件或摘要特征。64 个无结果标签 pair（四层各16）完成 lexical comparator；没有
可验证 frozen Qwen revision，因此 LLM 0 calls、`QUALITY_UNVERIFIED`。Synthetic
fixture 的独立 subprocess save→reload→report 与未来未完成 bar fault 通过。

## 2026-09-19: V10 Stage B1 final independent replay verifier

**Scope:** verifier-only. The frozen Stage B1 DPRICE predictions, candidate
grid, issued parameters, metrics, method selections, B2 configuration, and all
24 private model files were hashed before replay and remained byte-identical.
No Stage B2 or NEWS+PRICE model was run.

**Independent replay:** verifier-local code reconstructed 536 daily rows from
the raw teacher daily bars and XNYS schedule (70 warmup, 258 OOF, 84
development, 124 later). It independently fit all 36 authorized March--August
candidate models; the maximum discrepancy from the frozen grid was
`1.1102230246251565e-16`, with zero training-boundary violations. The replayed
grid reconstructed all 24 issued C decisions with zero mismatches and zero
September-onward freeze violations.

**Issued models and integrity:** all 24 issued models were independently refit
and all 466 row-level predictions replayed. The independent-refit versus frozen
maximum probability error was `8.326672684688674e-17`, with zero row-key or
direction mismatches. Independently refit predictions matched the final
manifest-hashed serialized models exactly; manifest mismatches were zero and
private bytes were unchanged. The V9 row-level March--August evidence reproduced
the three frozen top-three method sets, and all 18 B2 branches retained their
verified September-onward parameters. `STAGE_B1_FINAL_AUDIT_V2.json` is PASS.
