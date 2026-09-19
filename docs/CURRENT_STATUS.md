# 当前权威状态（2026-09-18，Phase A 已修复；Phase B 仅预注册）

> **Context relation-reader pilot:** The repaired full-universe target association
> produced 64 target-aware pairs with zero cross-split family leakage. The exact
> frozen Qwen checkpoint passed 13/13 file hashes; its single outcome-blind pass
> produced all 64 records, 63 mechanically valid. This is
> `PROVISIONAL_MODEL_RELATIONS_NOT_GOLD`; no stock-direction or market-context
> candidate was fitted or scored.

> **最新权威状态 — Phase B v1 已验证：**外部审查仅授权修复 verifier 对
> `advancement.json` 的存储精度比较。没有重跑 `run_gate.py`；十个既存预测/
> 结果 artifact 的 SHA-256 在修复前、verifier 后和报告后完全一致。修复后的
> verifier 为 `PASS`（18/18 checks），其 raw advancement 最大序列化差异为
> `4.843306398299996e-11`；同样经 pandas `to_json(double_precision=10)`
> 归一化后与保存文件严格一致。`G1 vs G0`、`G2 vs G1`、`G3 vs G2` 均未通过
> 原注册的 promotion gate。详见
> [验证结果](../outputs/stock_specific_gate_4h/v1/verification.json) 与
> [报告](../outputs/stock_specific_gate_4h/v1/REPORT.md)。development/later
> 仅为 `EXPOSED EXPLORATORY HISTORICAL BACKTEST`，没有用于选择方法。

> 下方关于 Phase B “仅预注册／未运行”及 `UNVERIFIED_STOPPED` 的段落是历史
> 记录，已被上述 verifier-only precision repair 的 PASS 状态取代；历史证据仍
> 保留在 v1 目录和审查交接中。

> **Activity v1 preflight:** 新的独立 activity branch 是
> `PREREGISTERED_NOT_RUN`。第七列无权威 vendor 定义，保持
> `SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY`；15/60 分钟、20 个历史 session 和
> 8 个冻结特征均已构建。A0 对两个保存的 canonical R1 来源最大概率误差均为
> `1.1102230246251565e-16`、方向完全一致；13 项 preflight checks PASS，且
> 尚不存在 A1/A1_matchedC v1 结果。只有精确 `APPROVE_ACTIVITY_RUN` 可解锁。
> Code-only repair 另加入 raw-bar 独立八字段重构（最大误差 0）、feature source
> hash 校验、按月 coverage、future post-run verifier 和 PASS-gated report；它们
> 均未运行 A1。
> ISSUE-063--068 又修复了 future runner 的 frozen-C schema、verifier-local
> calendar/selection、独立模型重载、协议 fingerprint 和报告期标签；preflight
> self-tests 通过，状态仍为 `PREREGISTERED_NOT_RUN`。

> **Activity v1 approved run:** runner 已严格执行一次，verifier runtime recovery
> 后通过全部独立检查。预注册 A1 gate 为 FAIL（AAPL June--August +0.237pp、AMZN
> -3.662pp），所以该 opaque activity construction 没有稳定双股票增量证据。
> development/later 仅是 `EXPOSED EXPLORATORY HISTORICAL BACKTEST`。

> **Context-increment lane:** `outputs/stock_context_4h/` has completed its initial
> access, metadata and synthetic-contract audit. SPY/QQQ intake is `AUTH_REQUIRED`
> (no credentials were printed or requested); no real Mmeta/M1/news/joint candidate
> was fitted or scored. See `docs/CONTEXT_INCREMENT_WORKING_SPEC.md`.

> **Market Context v1 repair:** the previous synthetic corruption matrix is
> `SYNTHETIC_V1_SUPERSEDED_PENDING_FAULT_SPECIFIC_REPAIR`; it used a simplified
> `p0` model path and repeated a generic mutation. The replacement shared core
> now uses raw canonical R1 features, completed raw market bars, chronological
> C inheritance, and a fit-free independent verifier. Its synthetic E2E and
> fourteen fault-specific checks pass in `audit_v3`; this is not real market
> scoring. Real Mmeta/M1 remains blocked on authenticated frozen Alpaca SIP data
> and explicit execution approval.

> **当前外部审查状态：**dense v5 仍是当前有效的稠密窗口 artifact。reaction
> v3 已保留且为 `SUPERSEDED_PENDING_REPAIR`；新的 v4 已用全部
> 30/60/120/240 分钟候选、`NaN + valid` 缺失值表示和高精度 AAPL 缩写排除
> 重新执行，并通过时间安全核验。四个 AR1 候选均未通过完整文章 gate，因此
> 没有 W0–W3 下游预测。stock-specific gate 的 repair preflight 已通过，但仍为
> `PREREGISTERED_NOT_RUN`；在明确 `APPROVE_GATE_RUN` 前不得执行或生成
> G0–G3 预测结果。

> **最终 code-only 审查修复：**reaction v4 的 future-price replay 现会
> 攻击任何结束时间晚于文章可用时刻的 bar，并以 10:02 的合成案例证明
> 10:00--10:05 未完成 bar 不会进入特征；强化 verifier 通过。Phase B 的
> `verify.py` 现同时包含批准后独立重算 prediction mapping、指标、advancement、
> chronology、preprocessing 和 coefficient structure 的分支。当前实际运行的仍
> 只有 preflight，且通过；没有 `v1` 预测、G0--G3 metrics、oracle 或 advancement
> artifact。

> **Phase B execution boundary (2026-09-18):** the owner approved exactly one
> frozen run, which completed at `outputs/stock_specific_gate_4h/v1/`. Its
> immediate independent verifier then stopped with `NameError: clean is not
> defined` before producing `v1/verification.json`. The saved run is preserved
> as `UNVERIFIED_STOPPED`; its scores are not a scientific result, no report was
> generated, and no post-result repair or tuning has been performed. See
> `v1/verification_failure.json` for the exact failure record.

> **Verifier-only recovery boundary:** the authorized scope repair moved only
> the pre-existing `clean()` helper. The recovery record corrects one missing
> final hexadecimal character in its initial `metrics.csv` SHA-256 entry; the
> corrected 64-character hashes for all eleven v1 files are identical before
> repair, immediately before verification, and after verification. The recovered verifier wrote
> `v1/verification.json=FAIL`: 17 checks pass, but independent advancement
> reconstruction rejects values at `1e-12` because `advancement.json` was
> serialized at approximately ten decimal digits. The runner and saved
> predictive CSVs remain unchanged; no report, tuning, rerun, or activity
> experiment occurred. Status remains
> `UNVERIFIED_STOPPED_PENDING_EXTERNAL_REVIEW`.

本次从 `dd836d80c709cd98065249ab5cdde233bd7abdc1` 继续，历史运行和原始
数据均保留。外部审查提出的 ISSUE-022—ISSUE-040 已在新的 v5/v2 目录中
修复并核验：

- dense v5 的 June–August gate 确实在合并前过滤两边，6 个单元完整；AAPL
  平均增量 `+3.740pp`，AMZN `−0.079pp`，因此没有两股稳定提升。
- reaction v2 只使用完成的五分钟 bar，并记录每行 used-bar end；它保留为
  历史 time-safety 修复，但其 W0--W3 下游协议与完整 addendum 不一致，
  所以下游数字已 superseded。
- reaction v3 按修正后的独立候选规则重跑了 AR0/AR1：240m AR1 的两股
  June--August BA 增量为 AAPL `+1.763pp`、AMZN `+3.218pp`，但宏 AUC
  增量为 `−0.570pp`，完整文章 gate 失败；60m 也失败。因此 v3 正确停止
  在 W0--W3 之前。AR2 记录为 `NOT_RUN_MODEL_UNAVAILABLE`，没有用不匹配
  的向量替代，也没有宣称独立人工复核通过。

Phase B `outputs/stock_specific_gate_4h/` 已写好协议、实现和静态契约测试，
固定 R1/F1_new、十个状态变量、G0–G3 和精确 no-news 回退；**尚未运行
`run_gate.py`，没有任何 G0–G3 结果。** 只有收到明确的
`APPROVE_GATE_RUN` 后才允许执行。

# 历史状态（2026-09-17）

## 当前审计进度：Stage 1–6 repair-first 审计已在本地完成

本轮 repair-first 审计没有重新追逐 exposed period 分数。v4 recency 的
session-age 权重已通过 17,140 个私有 fold 行的公式、单调性和 infinity
等权检查（公式最大误差 0，违规数 0）；公开 parity 仍为
`1.67e-15`。早期 dense verifier artifact 报告 gate 只含 June–August 的六个
stock-month cells，并确认 canonical parity 失败时训练与评价都使用同一
`reconstructed_all_official_and_augmented` 特征模式；外部审查随后发现可执行
gate 仍使用 March–August，因此该 gate/PASS 解读已 superseded，等待 time-safe
rerun。数值结果未改写；详见 [recency weight audit](../outputs/stock_recency_dense_4h/v4/recency_weight_audit.json)
和 [v4 verification](../outputs/stock_recency_dense_4h/v4/verification.json)。

Stage 3 的 FinBERT/Fin-ModernBERT 公平 pooling 对照已完成：canonical
FinBERT 重现误差为 `1.72e-15`，v2 ModernBERT 排除了 special tokens，
但没有显示稳定的两股提升；v2 没有保存原晋级公式的 machine-readable gate，
因此不作正式 pass/fail 晋级判定，也没有继续做融合。Stage 4 的 TabPFN 来源审计和
Stage 5 的 claim-only 审计也已完成；旧 Modern、TabPFN、SSL、analogy、
Event Adapter、Chronos 和校准结果继续按方法真值表的窄声明解释。

详见 [ModernBERT v2 报告](../outputs/stock_foundation_4h/v2/REPORT.md) 和
[v2 verification](../outputs/stock_foundation_4h/v2/verification.json)。

TabPFN 的旧“synthetic-only”描述也已纠正：本地 TabPFN 6.3.0 metadata 与
checkpoint archive 表明 default classifier 是 real-data fine-tuned。固定
`n_estimators=8` 的 own/cross probe 已实际运行，但没有显示稳定的两股提升；
v2 没有保存 exact goal60 gate，因此不作正式 pass/fail 晋级判定，也不再扩大
TabPFN 网格。详见 [TabPFN v2 report](../outputs/stock_tabpfn_4h/v2/REPORT.md)
和 [provenance audit](../outputs/stock_tabpfn_4h/v2/provenance.json)。

Stage 5 的[方法主张审计](../outputs/stock_method_validity_audit/v1/REPORT.md)
只读取保存的公开校准清单和文档，没有重新训练或推理。早期未约束 Platt
记录中 84 条有 32 条负斜率（全部是 AMZN）；后续受约束清单没有负斜率。
因此早期数值只能称为历史 score remapping，不能笼统称作单调校准。该轮
同时把 SSL/analogy/Event Adapter/Chronos 统一改成窄 probe 说法，并明确
事件聚合增益与正则化混杂、Qwen 直接输出是 token preference，以及
F1/F2 历史控制和 F1_new/F2_new 的区别。

最终五个公开 verifier、`refresh_repository.py`、`check_repository.py` 和
`git diff --check` 均通过；汇总见
[final verification](../outputs/stock_method_validity_audit/v1/final_verification.json)。
本地 commit 后仍会尝试 push；GitHub 是否更新以网络可验证结果为准。

本轮审计主体 commit 为 `ea621d5`；边界记录随后单独提交。主体 commit 后
仓库刷新与检查再次通过；
`git push origin main` 因 `Could not resolve host: github.com` 失败，当前
`origin/main` 仍为 `e32785d`，所以不能把 GitHub 说成已更新。

## 历史暂停记录（2026-09-18；已由上方 Phase A checkpoint supersede）

外部审查发现 dense v4 的实际月份 gate、reaction 的未完成五分钟 bar、
W0–W3 协议、promotion gate、AR1/AR2 预处理／表示和 reaction 时间安全
核验仍有阻塞问题。这些问题已登记为粗粒度 `ISSUE-022`—`ISSUE-027`，并已
拆成详细的 `ISSUE-028`—`ISSUE-040`；历史结果
全部保留，但受影响 artifact 的 PASS 只能视为此前本地检查结果。

当时状态为 `PAUSED_PENDING_REVIEWER_ADDENDUM`。该记录保留以说明审查边界；
详细 addendum 已纳入，新的 repair protocol 和独立输出目录见上方。

## 最新完成：v4 recency/dense 修正与全语料新闻 reaction probe（2026-09-17）

[v4 recency/dense 报告](../outputs/stock_recency_dense_4h/v4/REPORT.md)、[v4 数据审计](../outputs/stock_recency_dense_4h/v4/DATA_AUDIT.md)、[reaction 报告](../outputs/stock_reaction_features_4h/v1/REPORT.md)已经实际运行完成。该轮保留 AAPL/AMZN 四小时任务、原始窗口和 cutoff，v3 未改写。

- v4 使用 canonical J0/J2；F2 PCA 在训练文章向量上拟合并保留窗口元信息。infinity 分支是真正全样本等权重重训，与历史 reference 的最大逐概率误差 `1.67e-15`，低于 `1e-10`。
- recency 只在 2018-03—05 选择全局半衰期，2018-06—08 做 gate；R1/F1/F2 均选 20，但外层增量分别为 AAPL/AMZN `+2.80pp/-7.40pp`、`+4.57pp/-0.38pp`、`-4.34pp/+1.16pp`，均未通过。稠密窗口修正为 6 个月份行，D1 相对重建 D0_day 为 `+0.42pp/+0.78pp`，也未通过，D2 按协议停止。
- 官方与重建 dense 输入逐列 parity 不一致，所以 v4 的官方控制、训练和增广统一使用 reconstructed generator；没有混合两套特征。原 v3 “non-overlapping”措辞也已在 v4 纠正为 30 分钟起点的重叠四小时窗口。
- 全部 78,055 条原始新闻索引参与 reaction coverage 审计，得到 89,958 个 article×target 候选、85,402 个 canonical groups；AAPL/AMZN 都通过预登记可行性 gate。冻结 AR1 在 120m/240m 的文章级外层 gate 通过，但下游 W0–W3 没有两股稳定提升：AAPL W0/W1-120 为 48.70%/51.66%，AMZN 为 60.63%/45.51%。
- ⚠️ **历史结果已 superseded：**外部审查发现 dense v4 的可执行月份 gate、reaction 的完成 bar／时间安全、W0–W3 协议和 promotion gate 均需修复；因此本节的 `+0.42/+0.78pp` dense 数值以及 `120m/240m`、W0–W3 predictive/gate 表述只能作为保留的 exploratory artifact，不能作为当前有效结论。dense v5 与 reaction v3 已完成 time-safe rerun；reaction v2 的旧下游协议仍标为 superseded。
- 当前没有宣布独立人工事件关联验收通过；FinBERT 二进制不可重新加载，AR2 只是已有私有向量的约 10.36% coverage probe。没有新增微调、GNN、RL、Chronos 或付费数据。

新的公开核验：[recency v4 verification](../outputs/stock_recency_dense_4h/v4/verification.json)、[reaction v1 verification](../outputs/stock_reaction_features_4h/v1/verification.json)。所有 development/later/Jun-Aug 结果仍是已暴露的探索性历史回测。

## 最新完成：recency、稠密窗口与新闻反应审计（stock_recency_dense_4h/v3）

[运行报告](../outputs/stock_recency_dense_4h/v3/REPORT.md)、[数据审计](../outputs/stock_recency_dense_4h/v3/DATA_AUDIT.md)、[核验](../outputs/stock_recency_dense_4h/v3/verification.json)。本轮固定原始 AAPL/AMZN 四小时任务和 1,607 个官方窗口，等权 R1/F1/F2 逐窗口复现原保存概率（最大误差 0），再只改变训练标签权重。

- R1/F1/F2 在 infinity/80/40/20 NYSE session 半衰期中分别选择 20/80/20；June–August 增量为 R1 AAPL +0.41pp、AMZN −7.40pp，F1 +4.12/−3.21pp，F2 +5.97/−4.31pp，均未过弱股 +1pp、无股损失 >1pp、三个月至少两个月为正的预注册线。
- 30 分钟 stride、48 根完成五分钟线的稠密窗口共 2,012 行（含 998 个 Jan–Aug 官方行）；D1 相对官方日归一化 D0_day 的外层增量为 AAPL +0.42pp、AMZN +0.78pp，低于门槛，因此文本扩展和 D2 按协议停止。
- 文章反应审计使用已接受文章 ID，覆盖 4,930 篇文章、4,930 个 normalized groups 和 39,440 个 article×horizon 行，按 30/60/120/240 分钟、same-session/trading-time 分开统计；它是描述性 event-study 输入审计，不是因果或预测模型。
- activity 第七列为非负整数型，但未找到权威定义和单位，状态保持 `SEMANTICS_UNRESOLVED_NOT_USED`，没有进入任何模型。
- 所有 development/later 数值都是已经暴露的历史回测；后续优先级仍是获得可验证的 contemporaneous market-state 数据，不依据本轮局部提升挑选方法。

## 最新完成：市场状态审计与连续四小时收益辅助监督（stock_market_return_4h/v1）

[运行报告](../outputs/stock_market_return_4h/v1/REPORT.md)、[指标表](../outputs/stock_market_return_4h/v1/metrics.csv)、[协议](../outputs/stock_market_return_4h/PRE_REGISTRATION.md)。按GPT Pro诊断先做了两条低成本验证：固定日期的SPY/QQQ一分钟Alpaca可行性审计，以及在原1,607个AAPL/AMZN四小时窗口上加入真实`target_return`的辅助监督。主任务和时间切分没有改变，development/later仍是已暴露历史回测。

- Alpaca审计的8个SPY/QQQ探针全部因当前环境没有凭据记录为`AUTH_REQUIRED_NO_CREDENTIALS`，按协议停止；没有用日频或合成ETF替代。认证数据可验证后再重开市场分支。
- 实际完成360个候选/选定拟合记录：C0价格分类LR、C1连续收益Ridge、C2八维共享线性投影＋方向BCE＋收益Huber；C2使用3个固定seed，另有追加既有F1概率的对照。生成15,114个预测行，checkpoint重载、特征排除和时间顺序检查通过。
- 过去OOF选择的C2为raw-price `lambda=0.0`、price+F1 `lambda=0.5`。raw-price C2在later AAPL/AMZN BA为51.47%/50.26%，price+F1为52.04%/49.73%；注册的June–August晋级线四个候选全部未通过，未扩展残差市场、RL或更大网络。
- 这轮结果说明连续收益监督本身没有提供稳定方向增量；它没有证明市场状态或更好的外部数据一定无效。全部新结果与验证记录见运行目录；原始数据、环境和私有模型仍在`work/`。

## 最新完成：历史相似新闻与已实现收益的LLM对照

[报告](../outputs/stock_analogy_4h/v2/REPORT.md)、[案例](../outputs/stock_analogy_4h/v2/CASE_NOTES.md)。实际完成1393次冻结Qwen3.5-9B本地推理，约42.9分钟；没有新LLM微调。P0当前新闻＋价格；P1相似案例投票；P2历史案例无结果；P3同样案例加已实现四小时收益。

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

## 最新完成：小型价格自监督试验

[报告](../outputs/stock_ssl_4h/v1/REPORT.md)、[案例](../outputs/stock_ssl_4h/v1/CASE_NOTES.md)。按用户同意实际运行：过去48根五分钟线，原始PCA/随机冻结编码/掩码重建后冻结三组与价格B对照。不是完整TS2Vec复现，也没有新LLM标注。

- 1,988参数卷积编码器，21次内部早停＋21次最终预训练，266次LR拟合；CPU约103秒完成预训练与编码。最终每种子2,172重叠片段、169端点日期。重建损失约0.2036，零值对照0.2370，但不等于方向可预测。
- SSL后续AAPL/AMZN BA49.24%/50.66%，B为53.19%/50.26%；没有取得全窗口稳定提升。SSL对随机编码器的训练外层比较通过探索线，但没有胜过既有B，不升级系统、不扩网格。
- AMZN原无新闻95窗口，SSL修复旧F2的10个错误、新增6个，净+4；这一局部改善不用于事后创建路由。
- 24指标独立复算、21编码器/266头哈希、最终两股三方法重载概率误差0、时间边界/缓存拒绝/未来扰动/精确回退通过。所有时期仍已暴露。
- [LLM新闻增量组件](LLM_NEWS_INCREMENT_PLAN.md)目前是设计：对比截止前同一对象事件的前后事实，标出重复/背景/变化/未知；未向ChatGPT网页发送任务，未声称能判断“市场已消化”。

## 最新完成：跨股票价格、A1稠密迁移与历史新闻状态

[完整报告](../outputs/stock_goal60_4h/v1/REPORT.md)、[案例](../outputs/stock_goal60_4h/v1/CASE_NOTES.md)、[回放](../outputs/stock_goal60_4h/v1/demo.html)。固定1,607个四小时窗口；六月至八月作外层训练期评价，开发/后续分开报告，所有时期已暴露。

- 实际完成456次LR拟合、28次浅层树拟合、84次交互头最终训练＋84次内部早停训练；TabPFN做28次过去样本条件化（不是基础模型梯度训练）。原始FinBERT及三个既有A1检查点读取相同10,201个保护候选chunk；没有新增编码器微调。
- **没有机制通过预注册晋级线，最终不融合，不宣称两股稳定60%。** 跨股LR外层月均56.88%/57.06%，后续54.84%/49.20%；跨股TabPFN后续50.17%/54.35%。
- A1稠密LR外层52.47%/62.89%，后续54.34%/46.65%；原始同输入LR后续52.12%/48.36%。A1抽取适配没有变成稳定的股票方向增量；交互未改善此结论。
- AMZN开发/后续原无当期新闻的53/95个窗口全部找到过去两日的合格新闻，但H1后续AMZN48.72%，H2 48.67%。固定案例含近似转载、市值回顾和多公司综述；这是信息内容的观察，不是市场反应的因果证明。
- 新实际系统选参F2_new后续55.14%/56.79%，AMZN开发49.17%；不能按后续结果选股票赢家。F0/F1/F2仍保留课堂参照角色。
- 222指标独立复算、596模型/条件化包哈希、精确无新闻回退、未来价格扰动、时间边界和历史反应慢速复算通过。修复float32回退舍入与8+8价格输入不匹配；前版本私有存档，不纳入最终结果。
- Alpaca固定2018年SPY/QQQ取样返回401，用户尚无账户；市场数据分支等待认证后核验，不阻塞以上已完成实验，也未使用付费服务。

## 最新完成：固定正则归因、Fin-ModernBERT与Chronos-2

[本轮报告](../outputs/stock_foundation_4h/v1/REPORT.md)、[案例](../outputs/stock_foundation_4h/v1/CASE_NOTES.md)、[新demo](../outputs/stock_foundation_4h/v1/demo.html)。按用户新增授权实际执行208次LR拟合；冻结Fin-ModernBERT在M5 MPS编码5,078标题（30.82秒），冻结Chronos-2在CPU处理1,607窗口（34.71秒），并非微调基础模型。

- **归因已补齐：** AMZN后续文章等权C=0.01与事件等权C=0.01均为57.32%，逐窗口方向完全相同。文章C=1为54.74%，事件C=1为54.21%；原局部增益不需要去重就能重现。但强正则训练期方向没有一致增量，不能作为稳定新赢家。
- **Modern：** AAPL开发/后续BA45.51%/54.52%；AMZN54.82%/56.03%。相对F2训练月均AAPL−3.56pp、AMZN+3.44pp，宏平均−0.06pp且Brier恶化，未通过统一门槛。
- **Chronos＋LR：** AAPL开发/后续50.00%/50.55%；AMZN51.67%/47.01%。原始中位差后续48.12%/49.31%；未显示稳定方向优势。相同512历史LR也单独报告，未用更长输入冒充模型优势。
- 两个新模型均未通过门槛，本轮没有融合、进一步微调或扩大搜索。这里F2及Modern均读取标题，不是全文编码器实验。
- 72行指标独立复算、208权重哈希/重载、过去选参、时间间隔、禁止gap bar扰动、cache失配检查通过。修复本地object-key缓存安全重载，原件已保留且预测不变。
- F0/F1/F2仍为课堂主线；两新模型为有限探索。全部现有时期暴露，预训练历史重叠无法排除；不宣布新泛化或独立人工验收。

## 最新完成：有限组合验证、AMZN归因与课程交付

[新报告](../outputs/stock_combination_4h/v1/REPORT.md)、[课程报告主稿](COURSE_REPORT_4H.md)和[离线回放](../outputs/stock_combination_4h/v1/demo.html)已完成。实际优化22个过去OOF温度参数（28个记录含identity），复用既有分类器，完成8次冻结模型输入替换推理；本轮没有重训FinBERT。

- J3＋自身温度的后续BA为AAPL 56.71%、AMZN 53.79%，Brier为0.2494/0.2462。温度不改变方向。
- 相对F2＋温度，训练月均BA增加1.59个百分点，但AMZN训练Brier恶化0.0220，未通过预注册护栏；不扩大模型搜索。
- AMZN事件聚合57.32%相对匹配文章聚合54.74%增加2.58个百分点。固定模型替换当前向量不改变方向；两个训练分支C=1/0.01，故训练表示与选参效应仍混杂，不能把收益直接归因于去重。BA的1日/5日块区间均跨0。
- 主报告继续用F0作baseline、F1作统一全文方法、F2作现代语义对照。报告为课程主稿，成员贡献和教师最终版式待补；demo是保存预测回放，不是实时系统。
- 独立复算54行指标通过，浏览器回放交互通过。全部时期仍是已暴露历史回测，事件独立人工验收仍未完成。

## 最新完成：校准、近期价格联合与完整新闻聚合

[本轮报告](../outputs/stock_paper_methods_4h/v1/REPORT.md)：完成266次LR拟合、F1/F2独立校准与全文章集合的近似转载组聚合。J0/J2重现旧F1/F2至浮点误差。保留1,607窗口，其中233用于初始训练，765训练OOF、252开发、357后续；September之后冻结August末模型。

- F2正温度保持方向，四格Brier都降低；后续AAPL 0.2856→0.2503、AMZN 0.2617→0.2462。但训练期规则选中Platt，AAPL后续BA反而降至49.32%，不能事后改选温度赢家。
- J3 R1＋FinBERT训练月均BA提高1.59个百分点，但两股Brier分别恶化0.0119/0.0240，超过0.002护栏。J1 R1＋全文在AMZN训练BA下降2.16个百分点。
- N1M事件聚合＋元信息的AMZN后续BA57.32%、Brier0.2483，但相对匹配文章＋元信息的训练宏平均BA下降0.26个百分点，不能据此晋级。
- 四个机制均未通过预注册晋级线，FinModernBERT、Chronos-2、注意力、TabPFN和Graph本轮未运行，不代表这些模型被证明无效。

F0继续作为主baseline，F1作为稳妥统一全文方法，F2作为现代语义对照。新增温度作为概率质量消融。完整指标、逐月/覆盖、转移、配对区间、校准参数和模型证据已保存。独立事件质量验收仍未完成。

## 最新完成：FinBERT事件适配与严格门控四小时修正

[完整报告](../outputs/stock_finbert_event_adapter_4h/v1/REPORT.md)已实际完成预注册的数据审计、36次FinBERT训练、April暂定检查集评价、6次全四小时语料推理、D2/D3/D4残差对照和固定案例。训练期January–February前向OOF从A0／A1／A2中选择**A1顶部两层解冻**；没有使用April check、September–October development或later选择模型。M5 MPS上36次正式拟合累计5,700秒，checkpoint重载最大概率差为0。

| April暂定检查集完整事实签名 | AAPL F1 | AMZN F1 | 两股F1 |
|---|---:|---:|---:|
| 当前确定性规则 | 15.64% | 19.05% | 16.29% |
| Qwen3-1.7B QLoRA | 34.48% | 25.00% | 32.43% |
| Qwen3.5-9B分步抽取 | 25.81% | 62.50% | 38.30% |
| FinBERT A0冻结编码器 | 0.00% | 0.00% | 0.00% |
| **FinBERT A1顶部两层解冻** | **75.68%** | **90.91%** | **79.17%** |
| FinBERT A2 q/v LoRA | 46.67% | 90.91% | 53.52% |

这是对双GPT＋助手裁决暂定标签的结果，AMZN只有6个check事实，独立人工验收仍为0。A1修复了预先固定的历史／当前事件、评级维持＋目标价上调和多公司目标对象三类案例，但不能据此宣布正式抽取质量已经通过。

事件字段效用诊断在765个股票训练OOF窗口中只找到15个事件窗口、12个非重复报道组和14个非重复事实；唯一可向前评价的April月份没有改变任何方向，Brier由0.26882变为0.26910。部署式实验中，D4虽然覆盖AAPL 160、AMZN 10个训练事件窗口，但共享、独立和部分共享修正均未胜过BASE；协议最终让D2／D3／D4逐窗口严格等于F1。AMZN只有5个非重复A1事件、10个事件窗口，未达到30／60的独立模型门槛。

| 四小时方法 | AAPL开发 BA | AAPL后续 BA | AMZN开发 BA | AMZN后续 BA |
|---|---:|---:|---:|---:|
| F0价格＋标题baseline | 50.46% | 51.89% | 48.33% | 49.79% |
| R1近期价格 | 56.21% | 53.87% | 51.11% | 50.85% |
| **F1价格＋全文** | **57.94%** | 51.74% | **55.66%** | **54.36%** |
| F2价格＋冻结FinBERT | 54.13% | **56.71%** | 46.29% | 54.27% |
| F6原训练期融合 | 50.35% | 49.39% | 52.50% | 51.16% |
| D4 F1＋A1事件修正 | 57.94% | 51.74% | 55.66% | 54.36% |

后续时期单股最高是AAPL的F2 56.71%和AMZN的F1 54.36%；这是已暴露回放，不能据此指定股票专属模型。F2后续两股描述性均值55.49%，但概率误差较高且没有参与选择。跨development／later四个单元的最差BA最高者仍是F1（最低51.74%），因此当前最稳妥的统一项目方法仍是F1；本轮训练期协议选择的是“A1负责抽取、股票残差选择BASE”。

## 最新完成：完整段落、近期价格、共享模型与9B融合

[新一轮完整报告](../outputs/stock_nextgen_4h/REPORT.md)已经实际完成固定1,607个四小时窗口上的P0–P3、R0–R2、S2/S3和F0–F6。冻结Qwen3.5-9B在M5上生成5,819个新分数并复用609个已封存P0分数，推理约7.91小时，峰值MLX内存8.66GB；没有微调。另实际完成200次价格LR拟合、过去月份校准和低容量融合训练。公开结果通过逐标签、时间截止、缓存、恢复、权重重载和概率重算检查。

| 方法 | AAPL开发 BA/Brier | AAPL后续 BA/Brier | AMZN开发 BA/Brier | AMZN后续 BA/Brier |
|---|---:|---:|---:|---:|
| F0 价格＋标题baseline | 50.46%/.2601 | 51.89%/.2694 | 48.33%/.2710 | 49.79%/.2677 |
| F1 价格＋全文词特征 | **57.94%**/.2500 | 51.74%/.2587 | 55.66%/.2481 | **54.36%**/.2583 |
| F2 价格＋FinBERT | 54.13%/.2803 | **56.71%**/.2856 | 46.29%/.2977 | 54.27%/.2617 |
| F4 全文75%＋段落LLM25% | 52.92%/.2544 | 51.27%/.2557 | 52.50%/.2515 | 53.07%/.2530 |
| F6 LLM50%＋近期价格50% | 50.35%/.2573 | 49.39%/.2529 | 52.50%/.2652 | 51.16%/.2545 |
| R1 旧价格＋最近5–60分钟 | 56.21%/.2441 | 53.87%/.2558 | 51.11%/.2719 | 50.85%/.2722 |

结论是**没有一个方法同时在两股、两个暴露时期都超过F0并满足Brier护栏**。F1在四个单元的BA都高于50%，但AAPL后续比F0低0.15个百分点；R1四个单元都高于50%且BA均超过旧价格R0，但AMZN概率误差变差。它们是目前最合理的统一候选，不是新的独立泛化证明。

完整目标公司段落把训练期盲审的字符断句从106降到0，但训练期规则仍选择P0旧片段。AMZN的覆盖瓶颈仍很明显：804个窗口中，P1/P2/P3分别有422/402/405个没有合格段落，AAPL都只有6个。更多上下文和在线去重产生改对也产生改错，不能把更完整的输入直接等同于更高BA。

本地缺少SPY、QQQ、XLK、XLY的合格2018分钟线，也没有宏观／盈利预期差数据；免费来源调查没有确认符合历史深度、对齐、调整和可复现条件的无付费数据。因此F7、market PCA、HMM和Graph按预注册停止，没有用日频数据冒充。详见[数据盘点](../outputs/stock_nextgen_4h/DATA_INVENTORY.md)、[案例解释](../outputs/stock_nextgen_4h/CASE_INTERPRETATIONS.md)和[最终状态](../outputs/stock_nextgen_4h/FINAL_STATUS.md)。

## 最新完成：LLM直接预测四小时

[完整报告](../outputs/stock_llm_direct_4h/REPORT.md)：冻结Qwen3.5-9B直接预测，价格/新闻/联合各609窗口，共1,827个输出；另实际训练60次匹配LR。最终三组推理约77.9分钟，额外接口试验成本单列。没有新LLM微调。

| 后续时期BA | AAPL | AMZN |
|---|---:|---:|
| 原价格＋标题baseline | 51.89% | 48.92% |
| LLM看价格 | 47.09% | 48.84% |
| LLM看新闻 | 55.65% | 52.54% |
| LLM看价格＋新闻 | 50.08% | 51.21% |

新闻版在此段方向成绩最好，但Brier0.3684/0.3354较差，开发/逐月不稳定，配对差值区间跨0；不能称稳定改进。联合输入未胜过新闻输入。相同输入LR、旧全文/FinBERT也在完整报告中，未事后挑股票专属赢家。

原JSON接口完成价格609、新闻96后停止，因未接行情答案时发现数值/方向矛盾；新登记二选一接口读取真实UP/DOWN答案token相对概率。这是未校准偏好，不是模型自报概率。原失败与成本均保留。

[16例固定案例](../outputs/stock_llm_direct_4h/CASE_NOTES.md)与[输入检查](../outputs/stock_llm_direct_4h/INPUT_AUDIT.md)已完成：旧行情/长周期观点、重复报道、字符截取和价格历史陈旧仍是限制。AMZN95个无新闻窗口在新闻版全部判下。全部1,607窗口边界及1,827输入/输出、60次过去选参拟合核验通过，51份历史结果未变。

## 前序：换9B、改提示、分步抽取三组已完成

[本轮报告](../outputs/stock_llm_model_compare/REPORT.md)：固定Qwen3.5-9B 4bit，在同一87开发＋122检查文章上实际完成705次模型调用，M5约35.4分钟、累计MLX峰值6.24GB。**本轮为冻结推理，没有新微调或四小时预测训练。**

检查23条暂定事实：原提示TP0/FP2，新提示与示例TP3/FP11，分步抽取＋规则TP9/FP15；旧1.7B微调TP6/FP8。分步版召回更高，但AAPL F1从旧微调34.5%降至25.8%，AMZN从25.0%升至62.5%（仅6条事实，存在重复事件），没有两股稳定胜出。开发单项数值修正不改变检查成绩；事后安全修复拒绝两条旧/新评级误配，TP9/FP13，另列而不覆盖主结果。

完成21篇固定案例及额外诊断，共29篇输入审阅。模型仍混淆静态与新动作、历史与当前；程序也存在格式拒绝和语法覆盖缺口。独立验收未通过，保留原四小时输出，不把抽取F1作为预测BA。原新闻、模型与逐例输出留本地。

## 前序：扩大标注与两版QLoRA已完成

[完整报告](../outputs/stock_llm_annotation/REPORT.md)：GPT 6 High实际生成537候选×两轮标注，助手完成183条分歧/抽查决议；排除不确定、冲突和近似重复后为249训练、87开发、122检查。标签仍为模型暂定，独立人工验收未通过。

M5本地实际完成两版QLoRA，合计1,494个micro-step、192次优化更新，约24分钟，MLX峰值约2.75GB。相同检查集23个事实：规则匹配8个（29个错抽），冻结Qwen匹配0个，普通QLoRA匹配6个（8个错抽），额外事件存在监督匹配1个（1个错抽）。格式合规明显改善，但语义仍不可靠。检查集不参与checkpoint选择；所有成绩为抽取指标，**没有新增四小时BA/Brier**。

全部537输入无截断、训练/推理模板一致；逐项案例显示当前/历史、共识/动作、新设/上调、旧值与事件对象仍会混淆。本轮已完成预定两版对照，按质量条件停止下游扩展，原四小时结果保留。代码、汇总、训练证据和复现入口已整理；原文、标注及权重不进入公开仓库。

## 任务与评价边界

- 两股、四小时方向、原定 1,607 个窗口，信息截止点为目标起点前 5 分钟。
- 后续评价 AAPL 178 个、AMZN 179 个窗口；开发和后续时期均已暴露。
- BA 为两类召回率平均值，Brier 越低越好。表内数值不是新时期泛化证明。

| 后续时期模型 | AAPL BA | AMZN BA |
|---|---:|---:|
| 价格＋标题 baseline | 51.89% | 48.92% |
| 价格对照 | 51.79% | 46.20% |
| 旧价格＋全文 | 51.74% | 53.20% |
| 旧价格＋FinBERT | 56.71% | 52.22% |
| 旧完整组合 | 55.69% | 48.67% |
| 第一版条件选择器 | 51.52% | 46.20% |
| 零截距版条件选择器 | 51.79% | 46.20% |

来源：[当前比较](../outputs/stock_adaptive_4h/COMPARISON.csv)、[旧完整流水线](../outputs/stock_integrated_4h/runs/v1/metrics.csv)、[详细总报告](../outputs/stock_adaptive_4h/REPORT.md)。更高的单段成绩不等于可提前选择、跨月稳定的赢家。

## 最近确认的失败机制

1. AMZN 最新 gate 与价格模型在全部 305 个开发/后续窗口方向相同：新闻候选未通过训练期 Brier 约束，因此被关闭；不是已证明每条新闻毫无信息。
2. 无新闻强制回退也会移除旧联合模型学到的不同价格系数；不能把无新闻窗口上的差异归给新闻理解。
3. 自由新闻截距会吸收过去类别/校准偏差，随后失效。锁定截距改善了部分结果，但没有得到两股稳定提升。
4. 目标对象、历史背景、预期差和稀疏覆盖仍是输入问题；抽取正确与预测有效需要分开验收。

[24 例诊断](../outputs/stock_adaptive_4h/amzn_diagnosis/REPORT.md)包含预先登记的案例选择、内容先读、结果后揭示和线性贡献核验。助手审阅不等于独立盲审。

## 已做与未做

当前自适应四小时两版完成 464 次主拟合，另有在线截距更新；原报告保存训练和验证证据。所有数值来自已保存实验，本次 Git 整理没有重新训练。

完整事件关系、Graph、新文章注意力和可学习投影 LoRA 未完成当前四小时实验。65 行独立事件检查仍未完成，不能绕过门槛。[机制清单](../outputs/stock_adaptive_4h/amzn_diagnosis/MECHANISM_STATUS.md)为详细依据。

下一步候选是有限分支校准/收缩、经质量检查的新增事件信息、条件允许时内容感知选择器。这些是待验证方案，不保证提高表现。GPT Pro 的最新讨论未作为已验证结论。

## 新登记方案：小型LLM事实抽取与适配

[完整设计](SMALL_LLM_4H_PLAN.md)：冻结Qwen与同源QLoRA做抽取质量和四小时下游对照，保留价格＋文本底座。本机M5/24GiB已完成有限pilot，完整四小时LLM分支尚未验收。

## 小型LLM实际执行结果

[报告](../outputs/stock_llm_4h/REPORT.md)：旧单字母任务存在明显选项顺序敏感性。43个助手暂定样本中22个非重复训练样本；M5完成32步工程训练及两版各66步QLoRA，验证adapter更新、基座不变、重载一致，峰值约2.9GB。开发9个事实中冻结匹配0个、普通微调0个、正例加权1个；历史列表截断的冻结对照匹配1个。均不足以通过质量检查。独立复核0，未运行全量抽取或新的四小时事件预测。17项边界测试和回退核验通过，不能当作预测收益。另有用户要求的固定prompt/context对照，见该报告。
