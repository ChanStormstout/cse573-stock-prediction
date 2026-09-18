# ChatGPT Pro 审阅入口

每次讨论先打开本仓库默认分支的最新 README，再读取本页链接。请报告实际读取的 commit SHA；旧对话不会因为 Git push 自动重读全部文件。私有仓库需要通过 ChatGPT 的 GitHub 连接授权访问；普通网址本身不授予私有仓库访问权。

## 可复制提示词

> 请审阅这个仓库当前 main 分支，先报告你实际读取的 commit SHA 和文件清单。先读 README.md、docs/CURRENT_STATUS.md、outputs/PROJECT_LOG.md，以及 outputs/stock_nextgen_4h/ 下的 PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、INPUT_QUALITY.md、CALIBRATION.md、CASE_NOTES.md；再检查该目录的 Python 源码和 runs/v1 数值结果。请读取 docs/RESULT_FILES.json 中对应指标、逐窗口预测、训练期OOF、段落变化和配对区间，不要只复述报告。当前主任务是 AAPL/AMZN 未来四小时涨跌，早期一小时实验不能当作当前已实现机制。所有现有开发和后续时期都已经影响设计，只能视为探索性历史回放。区分代码事实、保存结果、作者解释和你提出的新假设。重点审查：P0--P3是否只改变登记变量；相同prompt的batch数值归一化是否合理；R0/R1/R2和S1/S2/S3是否时间安全且搜索预算匹配；无新闻回退、校准与融合是否只用过去OOF；AMZN目标段落覆盖是否限制了结论；市场数据分支的停止是否有证据。提出最多三个按成本/证据排序的后续机制，每项列数学或算法定义、匹配对照、只在过去数据上选参的方法、停止条件与预期失败方式。不要根据后续时期结果为两只股票分别挑赢家；不要承诺大幅涨分。对缺少原始新闻、模型权重或独立复核而无法核验的结论明确列出，不能声称已读取未提供文件。

## 证据索引

- [当前四小时实现与运行命令](../outputs/stock_nextgen_4h/README.md)
- [最新总报告](../outputs/stock_nextgen_4h/REPORT.md)
- [最新协议](../outputs/stock_nextgen_4h/PROTOCOL.md)
- [实现说明与已登记更正](../outputs/stock_nextgen_4h/IMPLEMENTATION_NOTES.md)
- [输入质量](../outputs/stock_nextgen_4h/INPUT_QUALITY.md)
- [LLM校准](../outputs/stock_nextgen_4h/CALIBRATION.md)
- [固定案例](../outputs/stock_nextgen_4h/CASE_NOTES.md)
- [固定案例的人工解释](../outputs/stock_nextgen_4h/CASE_INTERPRETATIONS.md)
- [本轮最终状态](../outputs/stock_nextgen_4h/FINAL_STATUS.md)
- [此前AMZN诊断](../outputs/stock_adaptive_4h/amzn_diagnosis/REPORT.md)
- [案例解释](../outputs/stock_adaptive_4h/amzn_diagnosis/CASE_INTERPRETATIONS.md)
- [近期论文与迁移限制](../outputs/stock_adaptive_4h/amzn_diagnosis/RECENT_PAPERS.md)
- [可读取结果文件](RESULT_FILES.json)
- [数据限制](DATA_AND_ARTIFACTS.md)

GitHub 访问不足时，请明确报告哪些文件没读到，不猜测附件内容或运行结果。

## 待审阅的新机制方案

[小型LLM→四小时事实修正](SMALL_LLM_4H_PLAN.md)已执行有限pilot，详见[实际报告](../outputs/stock_llm_4h/REPORT.md)与[指标](../outputs/stock_llm_4h/METRICS.csv)。完整四小时LLM分支仍未训练。请重点审查抽取适配的时间边界、模板一致性、有限事实质量验收与原价格＋文本精确回退，勿将小样本工程试验当作独立预测提升。

## 扩大标注后的新审阅重点

优先读取 [stock_llm_annotation 报告](../outputs/stock_llm_annotation/REPORT.md)、[标注清单](../outputs/stock_llm_annotation/INVENTORY.json)、[损失对照协议](../outputs/stock_llm_annotation/GATE_LOSS_PROTOCOL.md)与[案例笔记](../outputs/stock_llm_annotation/CASE_NOTES.md)，再核对相同目录的代码和结果表。

> 请重点检查：双轮GPT标注是否被误当独立金标准；正文候选为何选中大量历史持仓；补充开发/检查面板与原面板是否分开报告；长短答案的平均token损失如何影响事件存在判断；第二版辅助损失是否只改登记变量；checkpoint是否只依赖开发标签。先判断抽取组件是否学会了任务，再讨论四小时预测接入。不要把格式通过率、空输出比例或抽取精确率当作股票BA，也不要把未运行的下游分支写成完成。原文和adapter没有公开上传，明确你的核验边界。

## 9B与任务拆分对照

继续读取 [新实验入口](../outputs/stock_llm_model_compare/README.md) 和该目录的报告/协议。
区分只换模型、改提示方案、两阶段加规则三个变化；不要将家族与规模同时变化归因于参数量一个因素。
核对全体209篇分母、格式失败与FP的区别、gate丢失正例、规则拒绝、两阶段额外调用成本，
以及四月面板已暴露的限制。模型运行是冻结推理，不是新的训练或四小时预测成绩。

特别阅读CASE_NOTES与SAFETY_FIX：严格整篇拒绝可隐藏部分正确事实，但放宽后错抽很多；程序本身也有评级配对与语法缺口。安全修复受检查启发，不能冒充新保留集提升。分股票比较显示AAPL没有胜过旧微调，AMZN正例少且含重复事件。

## LLM直接四小时预测

优先读取 [直接预测报告](../outputs/stock_llm_direct_4h/REPORT.md)、CHOICE_PROTOCOL.md、
INPUT_AUDIT.md、CASE_NOTES.md、METRICS.csv、TRANSITIONS.csv 和 PAIRED_INTERVALS.csv。

> 请区分原JSON接口的部分试验与完整二选一实验。后者读取UP/DOWN答案token的相对概率，
> 没有微调、没有生成理由，不要编造其内部推理。核对609窗口三组是否完整、60次LR拟合、
> 不同CV月份导致新旧价格对照不同、新闻上限与缺失新闻、最新完整小时距截止55分钟或隔夜
> 的输入限制。检查新闻方向成绩与Brier是否一致改善、联合输入是否产生增量，
> 将行为关联与因果解释分开。现代LLM可能预训练见过历史新闻，所有时期已暴露；
> 不能把两股某单段BA超过50%直接写成独立泛化或校准成功。


## Latest finite mechanism round

Read [stock_goal60_4h report](../outputs/stock_goal60_4h/v1/REPORT.md), its PRE_REGISTRATION.md and IMPLEMENTATION_NOTES.md, then advancement.json, metrics.csv, training_evidence.json and CASE_NOTES.md. All16 new branches completed; none passed the declared gate. No final blend is claimed. Distinguish AMZN coverage recovery from predictive gain; independently review the matched price repair and strict float64 fallback. Alpaca remains AUTH_REQUIRED.
