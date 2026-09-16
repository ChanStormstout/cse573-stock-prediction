# CSE 573 — News and price based stock direction prediction

**当前主任务：预测 AAPL / AMZN 未来四小时涨跌。** 本仓库管理实际实验代码、协议、报告、结果表及案例诊断；早期一小时实验保留作为历史记录。

## 从这里开始

- [当前结论与结果](docs/CURRENT_STATUS.md)
- [给 ChatGPT Pro 的审阅入口与提示词](docs/CHATGPT_REVIEW.md)
- [实验索引](docs/EXPERIMENT_INDEX.md)
- [完整项目日志](outputs/PROJECT_LOG.md)
- [扩展 GPT 标注与 Qwen 微调](outputs/stock_llm_annotation/README.md)
- [9B模型、提示词与分步抽取对照](outputs/stock_llm_model_compare/README.md)
- [LLM直接预测四小时：价格、新闻与联合对照](outputs/stock_llm_direct_4h/README.md)
- [数据与未上传文件说明](docs/DATA_AND_ARTIFACTS.md)
- [复现说明](docs/REPRODUCING.md) · [更新流程](docs/WORKFLOW.md)

## 目前的结论

完整四小时流水线已训练并保存结果，但尚未证明新组合对两只股票均有稳定提升。当前协议保留价格＋标题 TF-IDF＋L2 逻辑回归作为主 baseline；价格模型是辅助对照。不能依据后续时期的最高分事后挑选模型。

所有现有历史时期已经参与探索，结果称为**探索性历史回测**，不再称为全新独立测试。完整事件/Graph 分支尚未通过独立质量复核。FinBERT 在当前四小时流水线中冻结；实际拟合的是下游模型。早期 LoRA、GRU、RL 等实验不能当作四小时版本的已完成工作。

## 仓库范围

代码与报告保留原路径，便于核对历史记录。`docs/RESULT_FILES.json` 列出纳入 Git 的结果文件及 SHA-256。新闻原文、课程资料、运行环境、大模型权重和缓存不随 Git 上传；`docs/LOCAL_ARTIFACTS.json` 提供本地资产清单。因此仅克隆仓库可审阅实现和结果，**不足以直接重训全部实验或启动有权重的 demo**。

本文档不授予第三方新闻、论文或课程数据的再分发许可。
