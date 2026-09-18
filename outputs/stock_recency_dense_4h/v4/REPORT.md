# v4：recency、稠密窗口与全语料反应输入

运行起点仓库提交：`c0383aca45022ec4a444509d445f4a02ab60d6e5`。本报告对应的新目录和代码在该提交之后生成；最终推送提交由项目日志记录。

## 结论先说

这一轮修正了 v3 的两个关键实现问题，并完成了预先登记的有限实验：

1. **recency 的 infinity 分支现在是真正的全样本等权重重训。** 它与历史 canonical reference 的逐概率最大误差为 `1.67e-15`，低于 `1e-10`；因此这条对照是可解释、可复现的。
2. **recency 没有通过外层 gate。** 训练期只用 2018-03—05 选择一个全局半衰期，所有方法都选到 20 个交易日；在 2018-06—08，AAPL 与 AMZN 的变化方向不一致或幅度不足，不能升级为改进方法。
3. **稠密窗口的正确 gate 只有 6 个月份行（两股共 12 个股票×月份行）。** D1 相对匹配的日归一化控制 D0_day 只提高 AAPL `+0.42pp`、AMZN `+0.78pp`，没有达到每股至少 `+1pp` 的预注册线，因此 D2 停止。

所有 development/later 数值都是已经暴露的历史回测，不能当作新的独立泛化测试。

## 实际做了什么

### Recency 分支

- 使用 `stock_paper_methods_4h/run.py` 的 canonical transform：F1 使用 J0；F2 使用 J2。F2 的 PCA(16) 在训练文章向量上拟合，再聚合到窗口，并保留 `log1p(news_count)` 与 `has_news`。
- 半衰期固定为 `infinity, 80, 40, 20` 个 XNYS session；选择只看 2018-03—05 chronological forward folds，按弱股票 BA、宏平均 BA、较简单的更长半衰期排序。
- 分开报告 `branch_recency`（加权文本分类器、等权 R1 fallback）和 `system_recency`（加权文本分类器、加权 R1 fallback）。主要比较是 `branch_recency` 对 infinity refit。
- 使用同一输入、同一 C schedule 重新拟合全部 fold；保存模型哈希、重载误差和逐窗口概率。

### 稠密窗口分支

- 30 分钟起点、每窗 48 根连续五分钟线，窗口是**重叠**的；并非“不重叠窗口”。
- 先做逐列输入 parity。重建的 R1 特征与官方输入在部分列不一致（例如 `return_1` 最大差约 `0.05046`，`history_age_hours` 最大差 `90`），所以没有把两种特征混合；官方控制和新增稠密样本统一使用 reconstructed generator。
- D1 与 D0_day 的外层比较固定为 2018-06、07、08 六个月份行。

## 关键结果

### Recency gate（外层 2018-06—08）

| 方法 | 选择半衰期 | AAPL ΔBA | AMZN ΔBA | 宏平均 ΔBA | 正向月份 | 通过 |
|---|---:|---:|---:|---:|---:|---|
| R1 | 20 | +2.80pp | −7.40pp | −2.30pp | 1/3 | 否 |
| F1 | 20 | +4.57pp | −0.38pp | +2.10pp | 2/3 | 否 |
| F2 | 20 | −4.34pp | +1.16pp | −1.59pp | 1/3 | 否 |

### 稠密窗口 gate（外层 2018-06—08）

| 控制/候选 | AAPL ΔBA | AMZN ΔBA | 宏平均 ΔBA | 正向月份 | 通过 |
|---|---:|---:|---:|---:|---|
| D1 − D0_day | +0.42pp | +0.78pp | +0.60pp | 3/3 | 否 |

“正向月份 3/3”仍未达到每股至少 `+1pp` 的要求，所以不能称为稳定改进。

## 如何解读

**已核实事实：**等权重 refit 没有重现错误；recency 代码、canonical F2 变换和 parity 检查通过。稠密 gate 使用了修正后的六个月范围，并且没有把不一致的官方/重建特征混在一起。

**观察：**半衰期改变了系数和预测方向，但 AAPL/AMZN 的收益互相抵消；稠密起点增加只带来小幅局部收益。

**解释性假设：**新闻“更近”或窗口“更密”并没有创造新的公司特定信息；它们主要改变了样本权重和相关窗口的分布。这个解释需要新时期数据才能进一步验证，不能当作因果结论。

## 停止与后续

- D2 组合分支按协议停止，因为 recency R1 与稠密 gate 都未通过。
- 本轮没有新增 LLM/FinBERT 微调、GNN、RL、Chronos 或外部数据。
- 下一条独立分支是全语料新闻反应审计和冻结 probe，见 `outputs/stock_reaction_features_4h/v1/REPORT.md`。它也必须由下游四小时方向结果决定是否保留。

## 复现入口

- 协议：[PRE_REGISTRATION.md](PRE_REGISTRATION.md)
- 修正说明：[CORRECTIONS.md](CORRECTIONS.md)
- recency 结果：[recency_metrics.csv](recency_metrics.csv)、[recency_selection.json](recency_selection.json)、[recency_refit_parity.csv](recency_refit_parity.csv)
- dense 结果：[dense_gate_corrected.json](dense_gate_corrected.json)、[dense_metrics.csv](dense_metrics.csv)、[dense_feature_parity.csv](dense_feature_parity.csv)
- 核验：[verification.json](verification.json)
