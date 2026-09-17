# CSE 573 — News and price based stock direction prediction

**当前主任务：预测 AAPL / AMZN 未来四小时涨跌。** 本仓库管理实际实验代码、协议、报告、结果表及案例诊断；早期一小时实验保留作为历史记录。

## 从这里开始

- [当前结论与结果](docs/CURRENT_STATUS.md)
- [FinBERT事件适配与严格门控四小时实验](outputs/stock_finbert_event_adapter_4h/v1/REPORT.md)
- [最新完整段落／近期价格／共享模型实验](outputs/stock_nextgen_4h/REPORT.md)
- [最新案例解释](outputs/stock_nextgen_4h/CASE_INTERPRETATIONS.md)
- [给 ChatGPT Pro 的审阅入口与提示词](docs/CHATGPT_REVIEW.md)
- [实验索引](docs/EXPERIMENT_INDEX.md)
- [完整项目日志](outputs/PROJECT_LOG.md)
- [扩展 GPT 标注与 Qwen 微调](outputs/stock_llm_annotation/README.md)
- [9B模型、提示词与分步抽取对照](outputs/stock_llm_model_compare/README.md)
- [LLM直接预测四小时：价格、新闻与联合对照](outputs/stock_llm_direct_4h/README.md)
- [数据与未上传文件说明](docs/DATA_AND_ARTIFACTS.md)
- [复现说明](docs/REPRODUCING.md) · [更新流程](docs/WORKFLOW.md)

## 目前的结论

完整四小时流水线已训练并保存结果，但尚未证明新组合对两只股票均有稳定提升。最新事件实验实际训练A0／A1／A2三种FinBERT适配器；A1在暂定April检查集的完整事实F1达到79.17%，明显高于已有规则和Qwen对照，但训练期OOF仍选择不对F1基础概率作事件修正。因此新闻理解改善没有转化成可验证的四小时方向增量。价格＋全文F1在四个股票×时期单元的BA都高于50%，仍是当前跨时期最稳定的统一方法；价格＋标题F0继续作为主baseline，不能依据后续时期的最高分事后挑股票专属模型。

所有现有历史时期已经参与探索，结果称为**探索性历史回测**，不再称为全新独立测试。事件标签仍是双GPT＋助手裁决的暂定标签，独立质量复核尚未完成。旧F2中的FinBERT保持冻结；最新事件实验另外完成了顶部两层解冻和q/v LoRA，但没有直接用四小时涨跌标签全参数微调FinBERT。早期GRU、RL等实验不能当作四小时版本的已完成工作。

## 仓库范围

代码与报告保留原路径，便于核对历史记录。`docs/RESULT_FILES.json` 列出纳入 Git 的结果文件及 SHA-256。新闻原文、课程资料、运行环境、大模型权重和缓存不随 Git 上传；`docs/LOCAL_ARTIFACTS.json` 提供本地资产清单。因此仅克隆仓库可审阅实现和结果，**不足以直接重训全部实验或启动有权重的 demo**。

本文档不授予第三方新闻、论文或课程数据的再分发许可。
