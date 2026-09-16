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
