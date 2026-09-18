# v4 数据与输入审计

- 主任务是 AAPL/AMZN 四小时方向；官方窗口 1,607 个，原始标签和 cutoff 规则沿用课程主线。
- 稠密构造共得到 3,232 个候选窗口；每个窗口使用 48 根连续五分钟 regular-session bar，起点间隔 30 分钟，窗口之间重叠。缺失区间不插值。
- 官方输入与重建输入的逐列 parity 没有通过：`return_1` 最大差 `0.0504576`、`range_1` 最大差 `0.0353526`、`history_age_hours` 最大差 `90`，且存在官方缺失/重建缺失的列差异。因此 v4 的 official control、augmented training 和 evaluation 统一使用 reconstructed generator，避免输入来源混杂。
- v4 gate 的月份严格为 `2018-06`、`2018-07`、`2018-08`，D1 行数为 6（每股每月一行）；`n_month_stock_rows=12` 是将两股分别展开后的配对表行数。
- recency 年龄由 XNYS schedule 计算，训练样本标签必须在评估 cutoff 前成熟；F2 PCA 只在对应训练文章向量上拟合。
- v3 保留不变，仅作为历史记录。v3 的“non-overlapping windows”措辞已在 v4 纠正为 densely sampled overlapping windows。
- 原始新闻、价格、缓存、向量和模型二进制继续留在 `work/`，不进入 Git；本报告只发布聚合审计数值。
