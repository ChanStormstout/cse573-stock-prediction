# Direct forecast case and behavior notes

## Completed-price behavior diagnostic

After all609 V2 price-only forecasts completed, joined outcomes for that branch.
This did not change prompts or selection for the still-running news/joint groups.
The descriptive analysis compares p_up with the supplied six-hour mean return;
it is not causal feature importance and not a new selected trading model.

- Later AAPL:91.57% direction agreement with the sign of mean historical return;
  Spearman probability/mean-return correlation0.868. LLM BA47.09%, simple sign46.49%.
- Later AMZN:95.53% agreement; Spearman0.824. LLM BA48.84%, simple sign51.19%.
- Development agreement85.71%/94.44%; correlation0.872/0.803.

Thus this frozen price prompt behaves similarly to continuation of the recent
mean trend. It does not demonstrate a learned profitable four-hour relationship.
The two later Brier errors0.2922/0.2846 exceed a constant .5 predictor's0.25.
Raw token preference should not be interpreted as calibrated market confidence.
Numbers alone cannot establish which internal representations caused a decision.
The simple sign comparator is post-hoc diagnostic, not an outcome-selected upgrade.

## Completed-news absence diagnostic

News-only later BA55.65%/52.54%, but Brier0.3684/0.3354. AMZN's95 no-news
windows are all DOWN, with p_up0.1645–0.2451 despite observed up rate49.47%.
Their BA is50% and Brier0.3286. The53 development no-news windows are also all
DOWN. These data demonstrate a negative default under this prompt; they do not
prove memorization or faithful interpretation. Added descriptive paired intervals for news-vs-news-LR and news-vs-title
baseline, alongside the originally planned joint intervals. This is a post-hoc
analysis addition, not a new prompt or outcome-based prediction selection.
No post-hoc missing-news fallback
was added to the completed predictions. Joint inference remains unchanged.

## 完整联合模型：16个固定案例已审阅

按股票×开发/后续时期×四种对错组合分层，在各格内按样本键SHA选第一条。
四种格都存在，共16例；已阅读每例全部实际提供的新闻片段和价格摘要。
这是助手事后案例检查，不是独立盲审。这里的LR是相同输入的新联合LR。
p为UP/DOWN答案token偏好，不是校准后的上涨概率。原文只留本地；以下为概述。

| 案例 | 时期/股票 | 实际方向；LLM p / LR p | 阅读观察与解释边界 |
|---|---|---|---|
| C01 | 后续 AAPL | 下；.004 / .476，双方对 | 需求疲软、App Store诉讼的重复报道与旧持仓混合，历史价格距截止约70小时。负面方向与结果一致，但不能证明重复信息或极端分数合理。 |
| C02 | 后续 AAPL | 上；.378 / .480，双方错 | 过去季度大跌、开始布局/反弹观点与XR销售资料并存；旧价格均值负。输入包含方向冲突，不能将过去下跌直接外推未来四小时。 |
| C03 | 后续 AAPL | 下；.378 / .586，仅LLM对 | 平板市场放缓、反弹观点、个人换机和旧持仓混杂；哈佛标题片段主要讲基金表现，未充分展开苹果交易。正确预测不等于输入已经完整。 |
| C04 | 后续 AAPL | 上；.011 / .611，仅LR对 | 条件性的贸易风险、沃尔玛电商排名、回购与过去基金交易；既有长期负面观点也有不同对象。LLM方向极端却错误，不能把不利叙述直接等同短期下跌。 |
| C05 | 开发 AAPL | 上；.991 / .513，双方对 | Canaccord目标价220→250在多个记录重复出现，另有旧持仓和长期比较；价格历史约91小时。事件有清晰变化，但一次判断正确无法验证99%偏好或重复计权。 |
| C06 | 开发 AAPL | 下；.905 / .579，双方错 | 硬件安全指控与公司否认，同其他技术看涨、品牌排名和评级列表并存。提供了相互冲突的信息，结果不足以证明模型正确权衡风险。 |
| C07 | 开发 AAPL | 下；.378 / .674，仅LLM对 | 前任CEO新事业、服务器安全调查、旧持仓和手机安全比较；部分新闻只是公司间接关联。不能把预测命中归于所有材料都有用。 |
| C08 | 开发 AAPL | 上；.033 / .533，仅LR对 | 已发生的FAANG下跌、延迟约38—44小时的市场/促销报道，与当天反弹描述混合；有三篇促销内容。支持继续检查时效与历史行情复述，不能证明它们单独导致失败。 |
| C09 | 后续 AMZN | 上；.651 / .690，双方对 | 无新闻；价格均值略正、历史约66小时。此例成功完全不能计作新闻理解贡献。 |
| C10 | 后续 AMZN | 下；.818 / .683，双方错 | 单篇苹果产品在Amazon销售的节日合作文章，价格均值负且历史约67小时。商业上的有利消息并未对应该四小时上涨。 |
| C11 | 后续 AMZN | 上；.731 / .340，仅LLM对 | 宏观/财报预告、Amazon Go长期机会和以Apple为主的财报片段。新闻覆盖有限且期限不同；正确输出未验证机制可跨月复用。 |
| C12 | 后续 AMZN | 上；.294 / .566，仅LR对 | 无新闻；过去均值负，LLM判下而实际上。属于价格关系/方向决策问题，不是事件抽取错误。 |
| C13 | 开发 AMZN | 下；.469 / .474，双方对 | 开盘综述中有服务器调查段，另含无关天气文字。两模型仅略偏下，不能根据接近阈值的一次成功夸大效果。 |
| C14 | 开发 AMZN | 上；.182 / .309，双方错 | 过去五个交易日下跌报道，混入GM和苹果活动；过去均值负。主要是已实现行情回顾，未来区间方向相反。 |
| C15 | 开发 AMZN | 下；.438 / .563，仅LLM对 | 无新闻；过去均值略负。改对来自价格/模型先验差异，不能归因于新增语义信息。 |
| C16 | 开发 AMZN | 下；.755 / .257，仅LR对 | 提高最低工资的两篇重复记录、3000美元长期看涨观点和Apple比较；成本与长期前景均可能相关，但LLM输出未验证四小时影响方向。 |

## 全样本变化比单例更有约束力

后续时期，新闻→联合：
- AAPL改变18例，改对4、改错14，BA下降5.57个百分点。
- AMZN改变67例，改对34、改错33。绝大多数变化来自95个无新闻窗口：
  60例改变（32改对、28改错）；84个有新闻窗口仅7例改变（2改对、5改错）。
  按完整类别分母计算，两部分BA贡献分别+0.55、−1.88个百分点，合计−1.33。
- AMZN无新闻时给模型补价格，不等于改进新闻理解；有新闻时也没有显示联合一定胜出。

相对课程标题baseline，新闻版后续BA差为AAPL+3.76、AMZN+3.62个百分点；
五日块描述性95%区间分别约[−4.32,+9.54]与[−5.84,+12.45]个百分点，都跨0。
联合版相对同输入联合LR为+1.56/+5.88个百分点，区间同样跨0。
这些已暴露数据上的事后区间不能恢复独立检验资格。

新闻版逐月也不稳定：AAPL十月BA43.33%；AMZN十二月58.65%、一月44.47%。
二月每股只有2个样本，AAPL只有一个真实类别，BA留空，不把它解释为一个稳定月份。

## 下一步的有限候选（未执行）

1. 给LR和LLM同时补截止前最近五分钟行情/部分小时摘要，单独检验“历史太旧”这一限制。
2. 用按句子完整性与公司对象组织的候选段落替代字符剪裁，保持文章预算；区分新变化、
   历史回顾、长期观点和未知，保留证据，不能直接假设抽取提高就会涨分。
3. 若要把LLM偏好用于融合，先在训练期样本外输出上学习受约束校准/组合，
   对比两个单模型；不能按这次后续成绩选股票专属赢家。

本轮并未完成这些改动、LLM微调、few-shot或新时期验证。16例用于解释观察到的失败方式，
不能建立新闻的因果影响，也不能还原未输出的LLM内部推理。
