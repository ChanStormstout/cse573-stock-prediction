# 数据与资产边界

## 纳入 Git

实验 Python 源码、Markdown 协议/报告/日志、环境版本记录，以及明确选入的四小时指标、逐窗口预测、选参对照和核验 JSON。未上传原始新闻全文、课程 slides/PDF、权重、缓存、虚拟环境或重复审阅 ZIP。

`RESULT_FILES.json` 是可读取结果的路径/大小/哈希清单。`LOCAL_ARTIFACTS.json` 是本地大型输入和模型文件的路径/大小/哈希清单；清单存在不代表这些文件可从 GitHub 下载。

## 数据来源与局限

课程提供的 AAPL/AMZN 新闻与五分钟价格。原新闻包偏向 AAPL；新闻可用时间与标注发布时间存在异常和延迟。见[原始审计](../outputs/stock_data_audit.md)和[覆盖审计](../outputs/stock_adaptive_4h/COVERAGE_REVIEW.md)。不得假设已经获得公开再分发许可。

新环境重训需要合法取得相同数据、既有审计输入及冻结向量。路径和哈希由原运行合同约束；搬迁后需显式重建合同并核验，不能删掉检查绕过。没有找到合格的新免费小时数据保留期。

公开发布前需单独核对报告中的新闻引用和课程信息。仓库未声明适用于第三方资料的开源许可。

## GPT标注与本地Qwen适配新增资产

本轮原文、标注包、浏览器回复、双轮比较、助手决议、sealed数据和adapter保存在
`work/stock-data/annotation/`，不随Git上传。`stock_llm_annotation/INVENTORY.json`
提供样本/排除统计，训练证据和核验文件提供权重哈希；不代表公开仓库包含训练数据。
这些`work/`资产不在仅扫描`outputs/`的旧二进制清单范围内。标注模型一致性不是
独立人类验收；公开汇总也不授予第三方新闻再分发许可。

9B模型文件、HF下载缓存、逐调用输入/输出、失败smoke和案例面板保存于
`work/stock-data/model_compare/`，同样不进入Git。公开的EXECUTION.json记录模型revision、
文件哈希与成本；模型对照仍使用上面的暂定标签，没有新增独立人工金标准。

## 直接LLM四小时预测

`work/stock-data/direct_4h/` 保存输入包、完整prompt、逐次输出、部分JSON失败试验、
固定案例原文与匹配LR权重。公开的 `outputs/stock_llm_direct_4h/` 只包含实现、协议、
报告、数字预测/指标、哈希及概述；不包含新闻原文或9B权重。
需要本地原课程资产与固定模型，才能重新运行；克隆仓库本身不提供这些输入。
