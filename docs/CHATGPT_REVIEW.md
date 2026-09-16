# ChatGPT Pro 审阅入口

每次讨论先打开本仓库默认分支的最新 README，再读取本页链接。请报告实际读取的 commit SHA；旧对话不会因为 Git push 自动重读全部文件。私有仓库需要通过 ChatGPT 的 GitHub 连接授权访问；普通网址本身不授予私有仓库访问权。

## 可复制提示词

> 请审阅这个仓库当前 main 分支，先报告你实际读取的 commit SHA 和文件清单。先读 README.md、docs/CURRENT_STATUS.md、outputs/PROJECT_LOG.md、outputs/stock_adaptive_4h/amzn_diagnosis/REPORT.md、MECHANISM_STATUS.md、CASE_INTERPRETATIONS.md 和 RECENT_PAPERS.md；再检查 stock_adaptive_4h 的 core.py、run.py、online.py、report.py 及 tests。请读取 docs/RESULT_FILES.json 中对应指标/逐窗口预测，不要只复述报告。当前主任务是 AAPL/AMZN 未来四小时涨跌，早期一小时实验不能当作当前已实现机制。所有现有历史评价已经影响设计。区分代码事实、保存结果、作者解释和你提出的新假设。分析为什么最新组合没有稳定超越价格＋标题 baseline，特别检查 AMZN 新闻分支关闭、价格回退丢失联合训练系数、校准与排序质量、覆盖和样本依赖。提出最多三个按成本/证据排序的机制，每项列数学或算法定义、匹配对照、只在过去数据上选参的方法、停止条件与预期失败方式。不要根据后续时期结果为两只股票分别挑赢家；不要承诺大幅涨分。对缺少原始新闻或模型权重而无法核验的结论明确列出，不能声称已读取未提供文件。近期论文只作为机制依据，核对任务/数据/指标是否匹配。

## 证据索引

- [当前四小时实现与运行命令](../outputs/stock_adaptive_4h/README.md)
- [最新总报告](../outputs/stock_adaptive_4h/REPORT.md)
- [AMZN 诊断](../outputs/stock_adaptive_4h/amzn_diagnosis/REPORT.md)
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
