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
| 正文修正 | 51.57% | 0.3076 | 46.20% | 0.2677 |
| 语义修正 | 52.62% | 0.3080 | 46.20% | 0.2677 |
| 允许价格再修正＋正文 | 50.52% | 0.3262 | 46.20% | 0.2677 |
| 允许价格再修正＋语义 | 51.60% | 0.3241 | 46.20% | 0.2677 |
| 静态修正组合 | 52.62% | 0.3080 | 46.20% | 0.2677 |
| 条件修正组合 | 51.52% | 0.3041 | 46.20% | 0.2677 |
| 静态组合＋校准 | 49.90% | 0.3050 | 46.90% | 0.2641 |
| 条件组合＋校准 | 50.45% | 0.3014 | 46.90% | 0.2641 |
| 预设规则选择的最终系统 | 51.89% | 0.2694 | 48.92% | 0.2772 |

## 九至十月开发回放

| symbol | method | BA | Brier |
|---|---|---|---|
| AAPL | title | 0.5132 | 0.2596 |
| AAPL | static | 0.5221 | 0.3053 |
| AAPL | gate | 0.5294 | 0.3016 |
| AAPL | static_cal | 0.5147 | 0.3033 |
| AAPL | gate_cal | 0.5221 | 0.3 |
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
| AAPL | static | 0.5262 | 0.308 |
| AAPL | gate | 0.5152 | 0.3041 |
| AAPL | online_weights | 0.5095 | 0.2987 |
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

最终预设规则选中标题 baseline。静态/条件组合在六月至八月两股平均BA有改善，但AAPL Brier超过预先约定的退化上限；校准版本又损失AMZN方向表现。未因看到后续成绩放宽规则。详细选择记录在 runs/v3/system_selection.json。

## 后续逐月

| symbol | period | method | n | BA | Brier |
|---|---|---|---|---|---|
| AAPL | 2018-09 | title | 57 | 0.4975 | 0.2481 |
| AAPL | 2018-09 | static | 57 | 0.5172 | 0.2876 |
| AAPL | 2018-09 | gate | 57 | 0.5345 | 0.2841 |
| AAPL | 2018-09 | online_weights | 57 | 0.5345 | 0.2803 |
| AAPL | 2018-10 | title | 69 | 0.5269 | 0.2691 |
| AAPL | 2018-10 | static | 69 | 0.5256 | 0.32 |
| AAPL | 2018-10 | gate | 69 | 0.5256 | 0.316 |
| AAPL | 2018-10 | online_weights | 69 | 0.5256 | 0.3076 |
| AAPL | 2018-11 | title | 60 | 0.5206 | 0.2656 |
| AAPL | 2018-11 | static | 60 | 0.5206 | 0.2925 |
| AAPL | 2018-11 | gate | 60 | 0.5033 | 0.2904 |
| AAPL | 2018-11 | online_weights | 60 | 0.5033 | 0.2873 |
| AAPL | 2018-12 | title | 53 | 0.5681 | 0.2824 |
| AAPL | 2018-12 | static | 53 | 0.5178 | 0.3561 |
| AAPL | 2018-12 | gate | 53 | 0.5178 | 0.3479 |
| AAPL | 2018-12 | online_weights | 53 | 0.4915 | 0.3364 |
| AAPL | 2019-01 | title | 63 | 0.4574 | 0.2582 |
| AAPL | 2019-01 | static | 63 | 0.5442 | 0.273 |
| AAPL | 2019-01 | gate | 63 | 0.5249 | 0.2714 |
| AAPL | 2019-01 | online_weights | 63 | 0.5249 | 0.2694 |
| AAPL | 2019-02 | title | 2 | nan | 0.3907 |
| AAPL | 2019-02 | static | 2 | nan | 0.5974 |
| AAPL | 2019-02 | gate | 2 | nan | 0.5825 |
| AAPL | 2019-02 | online_weights | 2 | nan | 0.5623 |
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
| AAPL | static | 0.0073 | -0.0487 | 0.0532 |
| AAPL | gate | -0.0037 | -0.0572 | 0.0388 |
| AAPL | old_integrated | 0.038 | -0.0174 | 0.0886 |
| AAPL | online_weights | -0.0095 | -0.0655 | 0.0354 |
| AMZN | static | -0.0272 | -0.0668 | 0.0097 |
| AMZN | gate | -0.0272 | -0.0668 | 0.0097 |
| AMZN | old_integrated | -0.0025 | -0.0448 | 0.0395 |
| AMZN | online_weights | -0.0272 | -0.0668 | 0.0097 |

区间为10,000次共同日期配对重采样的描述性区间，不校正反复探索。BA均为比例，差值0.01等于1个百分点。单日/十日敏感性见CSV。

## 回退与案例

| symbol | split | gate_reason | windows |
|---|---|---|---|
| AAPL | test | CONDITIONAL | 178 |
| AAPL | validation | CONDITIONAL | 124 |
| AAPL | validation | NO_NEWS_BASE | 2 |
| AMZN | test | CONDITIONAL | 84 |
| AMZN | test | NO_NEWS_BASE | 95 |
| AMZN | validation | CONDITIONAL | 73 |
| AMZN | validation | NO_NEWS_BASE | 53 |

完整改对/改错、新闻分层与最多64例确定性面板见同目录CSV和CASE_CARDS.md。人工读过哪些案例另在 CASE_REVIEW.md 逐一说明，生成面板不等于全部人工审阅。

## 工程状态

核心模型使用同样1607分类窗口，233次主运行拟合。FinBERT冻结，未微调；baseline最终概率与旧运行重现一致。在线另有408次小截距拟合，另408次仅用于未来标签扰动验证，不作为新的候选实验。

v1是新代码概率共享内存错误导致的中断，不能用于成绩；v2完成预测但部分NumPy布尔元数据被写为字符串；v3只修复JSON类型与依赖清单，预测与v2必须完全一致。两个问题均保留运行记录。

关系与新文章接入受独立复核条件限制。coverage目录是审计和复核材料，任何规则候选数量不能宣传为已增加的有效预测覆盖。

## 文件

metrics.csv / paired_intervals.csv / final_components.json / gate_usage.csv / case_changes.csv / news_slices.csv / CASE_CARDS.md；运行权重、OOF和来源在 runs/v3，在线状态在 online/v1。
