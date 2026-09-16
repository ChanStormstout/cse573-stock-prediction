# 四小时新闻增量与条件组合：执行结果

所有结果来自已暴露历史，非新的独立测试。训练、抽取质量、预测收益分别记录。

## 冻结后续时期：2018-11 至 2019-02 初

| 方法 | AAPL BA | AAPL Brier | AMZN BA | AMZN Brier |
|---|---|---|---|---|
| 标题 baseline | 51.89% | 0.2694 | 48.92% | 0.2772 |
| 价格 baseline | 51.79% | 0.2720 | 46.20% | 0.2677 |
| 旧四小时组合 | 55.69% | 0.2630 | 48.67% | 0.2610 |
| 旧价格＋全文 | 51.74% | 0.2587 | 53.20% | 0.2608 |
| 旧价格＋FinBERT | 56.71% | 0.2856 | 52.22% | 0.2711 |
| 正文修正 | 54.09% | 0.2801 | 46.20% | 0.2677 |
| 语义修正 | 51.79% | 0.2720 | 46.20% | 0.2677 |
| 允许价格再修正＋正文 | 50.52% | 0.3251 | 46.20% | 0.2677 |
| 允许价格再修正＋语义 | 51.79% | 0.2720 | 46.20% | 0.2677 |
| 静态修正组合 | 51.79% | 0.2720 | 46.20% | 0.2677 |
| 条件修正组合 | 51.79% | 0.2720 | 46.20% | 0.2677 |
| 静态组合＋校准 | 52.97% | 0.2712 | 46.90% | 0.2641 |
| 条件组合＋校准 | 52.97% | 0.2712 | 46.90% | 0.2641 |
| 预设规则选择的最终系统 | 51.89% | 0.2694 | 48.92% | 0.2772 |

## 九至十月开发回放

| symbol | method | BA | Brier |
|---|---|---|---|
| AAPL | title | 0.5132 | 0.2596 |
| AAPL | static | 0.4959 | 0.2615 |
| AAPL | gate | 0.4959 | 0.2615 |
| AAPL | static_cal | 0.4911 | 0.2632 |
| AAPL | gate_cal | 0.4911 | 0.2632 |
| AMZN | title | 0.474 | 0.2767 |
| AMZN | static | 0.5009 | 0.262 |
| AMZN | gate | 0.5009 | 0.262 |
| AMZN | static_cal | 0.5074 | 0.2583 |
| AMZN | gate_cal | 0.5074 | 0.2583 |

## 在线更新：独立信息协议

| symbol | method | BA | Brier |
|---|---|---|---|
| AAPL | title | 0.5189 | 0.2694 |
| AAPL | price | 0.5179 | 0.272 |
| AAPL | static | 0.5179 | 0.272 |
| AAPL | gate | 0.5179 | 0.272 |
| AAPL | online_weights | 0.5349 | 0.273 |
| AAPL | online_price | 0.5177 | 0.2725 |
| AAPL | online_title | 0.5187 | 0.2694 |
| AMZN | title | 0.4892 | 0.2772 |
| AMZN | price | 0.462 | 0.2677 |
| AMZN | static | 0.462 | 0.2677 |
| AMZN | gate | 0.462 | 0.2677 |
| AMZN | online_weights | 0.462 | 0.2677 |
| AMZN | online_price | 0.4413 | 0.269 |
| AMZN | online_title | 0.4791 | 0.2783 |

## 训练期选择与停止

预设规则选择 title。完整候选的跨月BA、Brier限制和继续条件在 runs/zero_bias_v1/system_selection.json；未因后续成绩放宽规则。

## 后续逐月

| symbol | period | method | n | BA | Brier |
|---|---|---|---|---|---|
| AAPL | 2018-09 | title | 57 | 0.4975 | 0.2481 |
| AAPL | 2018-09 | static | 57 | 0.4791 | 0.2528 |
| AAPL | 2018-09 | gate | 57 | 0.4791 | 0.2528 |
| AAPL | 2018-09 | online_weights | 57 | 0.4791 | 0.2538 |
| AAPL | 2018-10 | title | 69 | 0.5269 | 0.2691 |
| AAPL | 2018-10 | static | 69 | 0.5141 | 0.2688 |
| AAPL | 2018-10 | gate | 69 | 0.5141 | 0.2688 |
| AAPL | 2018-10 | online_weights | 69 | 0.5141 | 0.2704 |
| AAPL | 2018-11 | title | 60 | 0.5206 | 0.2656 |
| AAPL | 2018-11 | static | 60 | 0.5044 | 0.2686 |
| AAPL | 2018-11 | gate | 60 | 0.5044 | 0.2686 |
| AAPL | 2018-11 | online_weights | 60 | 0.5044 | 0.2702 |
| AAPL | 2018-12 | title | 53 | 0.5681 | 0.2824 |
| AAPL | 2018-12 | static | 53 | 0.5712 | 0.2831 |
| AAPL | 2018-12 | gate | 53 | 0.5712 | 0.2831 |
| AAPL | 2018-12 | online_weights | 53 | 0.5712 | 0.2838 |
| AAPL | 2019-01 | title | 63 | 0.4574 | 0.2582 |
| AAPL | 2019-01 | static | 63 | 0.4688 | 0.2621 |
| AAPL | 2019-01 | gate | 63 | 0.4688 | 0.2621 |
| AAPL | 2019-01 | online_weights | 63 | 0.5151 | 0.2623 |
| AAPL | 2019-02 | title | 2 | nan | 0.3907 |
| AAPL | 2019-02 | static | 2 | nan | 0.3929 |
| AAPL | 2019-02 | gate | 2 | nan | 0.3929 |
| AAPL | 2019-02 | online_weights | 2 | nan | 0.4095 |
| AMZN | 2018-09 | title | 57 | 0.4403 | 0.2756 |
| AMZN | 2018-09 | static | 57 | 0.5283 | 0.2469 |
| AMZN | 2018-09 | gate | 57 | 0.5283 | 0.2469 |
| AMZN | 2018-09 | online_weights | 57 | 0.5283 | 0.2469 |
| AMZN | 2018-10 | title | 69 | 0.5074 | 0.2777 |
| AMZN | 2018-10 | static | 69 | 0.4732 | 0.2744 |
| AMZN | 2018-10 | gate | 69 | 0.4732 | 0.2744 |
| AMZN | 2018-10 | online_weights | 69 | 0.4732 | 0.2744 |
| AMZN | 2018-11 | title | 60 | 0.5139 | 0.2713 |
| AMZN | 2018-11 | static | 60 | 0.5208 | 0.2666 |
| AMZN | 2018-11 | gate | 60 | 0.5208 | 0.2666 |
| AMZN | 2018-11 | online_weights | 60 | 0.5208 | 0.2666 |
| AMZN | 2018-12 | title | 54 | 0.5135 | 0.2794 |
| AMZN | 2018-12 | static | 54 | 0.4203 | 0.2713 |
| AMZN | 2018-12 | gate | 54 | 0.4203 | 0.2713 |
| AMZN | 2018-12 | online_weights | 54 | 0.4203 | 0.2713 |
| AMZN | 2019-01 | title | 63 | 0.4489 | 0.2813 |
| AMZN | 2019-01 | static | 63 | 0.4289 | 0.2666 |
| AMZN | 2019-01 | gate | 63 | 0.4289 | 0.2666 |
| AMZN | 2019-01 | online_weights | 63 | 0.4289 | 0.2666 |
| AMZN | 2019-02 | title | 2 | 0.5 | 0.2618 |
| AMZN | 2019-02 | static | 2 | 0.5 | 0.2405 |
| AMZN | 2019-02 | gate | 2 | 0.5 | 0.2405 |
| AMZN | 2019-02 | online_weights | 2 | 0.5 | 0.2405 |

## 五日日期块区间：相对标题 baseline

| symbol | method | difference | low | high |
|---|---|---|---|---|
| AAPL | static | -0.001 | -0.0258 | 0.025 |
| AAPL | gate | -0.001 | -0.0258 | 0.025 |
| AAPL | old_integrated | 0.038 | -0.0174 | 0.0886 |
| AAPL | online_weights | 0.016 | -0.0135 | 0.0439 |
| AMZN | static | -0.0272 | -0.0668 | 0.0097 |
| AMZN | gate | -0.0272 | -0.0668 | 0.0097 |
| AMZN | old_integrated | -0.0025 | -0.0448 | 0.0395 |
| AMZN | online_weights | -0.0272 | -0.0668 | 0.0097 |

区间为10,000次共同日期配对重采样的描述性区间，不校正反复探索。BA均为比例，差值0.01等于1个百分点。单日/十日敏感性见CSV。

## 回退与案例

| symbol | split | gate_reason | windows |
|---|---|---|---|
| AAPL | test | STATIC_SELECTED | 178 |
| AAPL | validation | NO_NEWS_BASE | 2 |
| AAPL | validation | STATIC_SELECTED | 124 |
| AMZN | test | NO_NEWS_BASE | 95 |
| AMZN | test | STATIC_SELECTED | 84 |
| AMZN | validation | NO_NEWS_BASE | 53 |
| AMZN | validation | STATIC_SELECTED | 73 |

完整改对/改错、新闻分层与最多64例确定性面板见同目录CSV和CASE_CARDS.md。生成面板不等于全部人工审阅。

## 工程状态

核心模型使用同样1607分类窗口，231次主运行拟合。FinBERT冻结，未微调；baseline最终概率与旧运行重现一致。在线另有408次小截距拟合，另408次仅用于未来标签扰动验证，不作为新的候选实验。

初版v1因新代码概率共享内存错误中断；v2完成预测但部分NumPy元数据写为字符串；v3修正类型且预测与v2一致。zero_bias_v1是随后预登记的唯一机制扩展：新闻截距固定零。保留全部记录。

关系与新文章接入受独立复核条件限制。coverage目录是审计和复核材料，任何规则候选数量不能宣传为已增加的有效预测覆盖。

## 文件

metrics.csv / paired_intervals.csv / final_components.json / gate_usage.csv / case_changes.csv / news_slices.csv / CASE_CARDS.md；运行权重与OOF在 runs/zero_bias_v1；在线在 online/zero_bias_v1。
