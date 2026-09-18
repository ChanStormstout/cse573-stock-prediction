# 全语料新闻反应特征 v1

运行起点仓库提交：`c0383aca45022ec4a444509d445f4a02ab60d6e5`。本报告对应的新目录和代码在该提交之后生成；最终推送提交由项目日志记录。

## 结论先说

这轮解决了“只看 4,930 篇已接受文章”的覆盖审计问题：我们从完整的 78,055 条原始新闻索引建立了 `(article_key, target_symbol)` 级别的数据集，两个股票都通过了预先登记的可行性 gate。随后做了冻结 reaction probe，并把通过 gate 的 horizon 接入四小时方向修正。

**文章级 reaction probe 在 120 分钟和 240 分钟通过了预登记外层线，但这个局部信号没有转化成两股都更好的四小时方向预测。** 下游 W0–W3 结果反而显示 AMZN 基线很强时，直接加入反应均值会明显损失 BA。因此停止继续扩展 horizon/grid；这不是“新闻反应已经解决方向预测”的证据。

## 数据审计与可行性

| 阶段 | 数量 |
|---|---:|
| 原始索引行 | 78,055 |
| 英文 | 70,732 |
| 足够长文本 | 70,601 |
| 候选 article×target 对 | 89,958 |
| 全时段 canonical groups | 85,402 |
| 被重复/近似报道压掉的候选 | 4,556 |
| 2018-01—08 canonical 对 | 51,485 |

可行性 gate（两股都要求至少 800 个唯一目标文章组、至少 100 个日期、至少 6 个月每月 50 组）通过：

| 股票 | 唯一目标文章组 | 60m 有效反应 | 240m 有效反应 | 日期 | 月份≥50 | gate |
|---|---:|---:|---:|---:|---:|---|
| AAPL | 40,764 | 11,288 | 5,490 | 168 | 8 | PASS |
| AMZN | 10,721 | 3,137 | 1,410 | 168 | 8 | PASS |

可用时间是 `max(valid published_utc, crawled_utc)`；反应只在同一 regular session 内计算，跨缺失区间或不完整 horizon 记为 unknown。一个文章可以合法地产生 AAPL 和 AMZN 两个 target pair。Tier 0 关联占主导，尤其 AAPL；这只是规则关联统计，不是独立人工 gold。独立组员复核仍未完成。

## 实际训练/推理

- AR0：价格反应前置上下文。
- AR1：价格上下文＋词特征。
- AR2：价格上下文＋冻结 FinBERT 目标段落向量的 PCA(16)。
- 每个 horizon 独立选择 C；训练文章的 reaction label 必须在评估文章 available time 之前成熟。没有进行 FinBERT 或 LLM 微调。
- 当前主机缺少可重新加载的 ProsusAI/FinBERT 模型二进制；AR2 使用已有的私有冻结文章向量，只覆盖 1,608 行、约 10.36% 的 reaction rows。因此 AR2 是 coverage-limited probe，不是完整 AR2 证明。

## 文章级外层结果

AR1 相对 AR0 的 BA 变化：

| horizon | AAPL ΔBA | AMZN ΔBA | 宏平均 | 预登记 gate |
|---:|---:|---:|---:|---|
| 30m | +1.38pp | −0.38pp | +0.50pp | 否 |
| 60m | +2.87pp | −1.01pp | +0.93pp | 否 |
| 120m | +1.65pp | +1.60pp | +1.62pp | 是 |
| 240m | +2.69pp | +5.40pp | +4.04pp | 是 |

这里的“通过”只表示文章级 reaction probe 达到预先登记的两股外层比较线；不表示四小时方向模型已经改善。

## 四小时下游 W0–W3

下游只使用 2018-06—08 的 192 个 AAPL 与 192 个 AMZN 官方外层窗口，F1 作为基础概率；W0 是基础模型，W1 是单个通过的 reaction horizon，W2 合并两个 horizon，W3 在至少两个文章反应可用时才修正。

| 股票 | 方法 | BA | Brier | MCC |
|---|---|---:|---:|---:|
| AAPL | W0 | 48.70% | 0.2533 | −0.025 |
| AAPL | W1-120m | 51.66% | 0.2422 | 0.049 |
| AAPL | W1-240m | 51.66% | 0.2422 | 0.049 |
| AAPL | W2 | 51.26% | 0.2417 | 0.037 |
| AAPL | W3 | 49.60% | 0.2431 | −0.010 |
| AMZN | W0 | 60.63% | 0.2441 | 0.211 |
| AMZN | W1-120m | 45.51% | 0.2591 | −0.101 |
| AMZN | W1-240m | 45.96% | 0.2590 | −0.090 |
| AMZN | W2 | 45.96% | 0.2590 | −0.090 |
| AMZN | W3 | 55.50% | 0.2509 | 0.110 |

W3 对 AMZN 比 W1 好，但仍低于 W0；AAPL 的 W1 小幅改善而 W3 回落。因此没有一个 reaction downstream 版本同时超过 W0 并通过两股门槛。

## 证据等级、限制与停止决定

- **已核实事实：**完整语料计数、时间规则、去重统计、gate、文章级训练/推理、下游文件和重载/唯一键核验。
- **探索性观察：**120m/240m article-level AR1 与四小时下游的数值差异。所有开发/later/Jun-Aug 结果都已暴露。
- **解释性假设：**文章后实现收益可能包含市场状态和共同冲击，直接平均进四小时分类器会把反应信息重复计权或引入时间漂移；本轮没有证明具体因果路径。
- **不能声称：**独立人工事件关联验收通过；AR2 全覆盖完成；reaction 特征已经提高主任务；LLM/FinBERT 微调已经带来提升。

停止规则：AR1 已完成预登记 horizon，120m/240m 的下游 W0–W3 已完成且没有两股稳定增量，不继续扩大 horizon、网格或下游组合。AR2 等待可复现的 FinBERT 模型二进制和独立质量检查后另立协议。

## 复现入口

- 协议：[PRE_REGISTRATION.md](../PRE_REGISTRATION.md)
- schema：[DATA_SCHEMA_AUDIT.md](DATA_SCHEMA_AUDIT.md)
- gate：[reaction_gate.json](reaction_gate.json)
- probe：[reaction_probe_gate.json](reaction_probe_gate.json)、[reaction_metrics.csv](reaction_metrics.csv)
- 下游：[reaction_downstream_metrics.csv](reaction_downstream_metrics.csv)、[reaction_downstream_predictions.csv](reaction_downstream_predictions.csv)
- 核验：[verification.json](verification.json)
