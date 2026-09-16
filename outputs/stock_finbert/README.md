# E03：FinBERT 特征与新闻数量对照

完整动机、尝试历史、观察、作废结果、修复和下一步决定均维护在 [项目主日志](../PROJECT_LOG.md)。本文件只说明复现和文件结构。

## 有效结果

`results/` 是修复记录定位后的有效运行，`results_uuid_invalid/` 是按 UUID 错取新闻版本的作废运行。不要合并两者指标。

本轮保持 E02 的样本、标签、新闻窗口与价格特征，编码 5,783 条训练/验证使用的标题。冻结 ProsusAI/finbert 后，聚合正面、负面、中性概率和情绪分歧，用 LR 预测方向。另做价格 + 新闻数量对照。模型和数量对照使用相同 C 网格。

最终测试集未做特征推理、训练或评估。现有开发结果未证明 FinBERT 稳定提高下一小时预测能力。

## 文件

- `protocol.json`：本轮协议，模型 commit、聚合方式、预设网格及记录键规则。
- `run_finbert.py`：重建精确新闻窗口，核对与 E02 文本一致，本机推理、训练、验证和按日 bootstrap。
- `check_finbert.py` / `check_result.txt`：独立检查及实际通过结果。
- `news_review.csv`：48 条训练期标题，助手诊断标签，未经组员独立复核；包括原始记录键。
- `review_summary.json`：标题抽查计数，多标签可能重叠。
- `uuid_fix_audit.json`：错取 2 个版本所影响的窗口数。
- `results/inference.json`：模型固定版本、输入哈希、设备、运行时间、截断数。
- `results/article_probabilities.csv`：每条精确记录的概率；不能仅靠 UUID 做连接。
- `results/results.json`：有效选参后验证指标、月份切片、描述性差值区间。
- `results/{股票}/development_features.csv`：训练/验证特征、记录键、划分和标签。
- `results/{股票}/validation_grid.csv`：所有候选参数，避免只展示最好值。
- `results/{股票}/validation_predictions.csv`：验证概率。
- `results/{股票}/*.joblib`：模型与特征列表，仅加载可信模型文件。

## 当前工作区复现

依赖 E02 的 `outputs/stock_baseline/results`、本地 `work/stock-data/audit/news_index.pkl` 和已安装的虚拟环境。软件版本记录在 `requirements-record.txt`。首次运行会下载公开模型到 `work/stock-data/finbert-cache`；新闻推理在本机进行，不上传新闻到模型服务。

从任务根目录运行：

```sh
work/stock-data/finbert-env/bin/python outputs/stock_finbert/run_finbert.py
work/stock-data/finbert-env/bin/python outputs/stock_finbert/check_finbert.py
```

脚本会复用输入哈希与协议一致的标题概率缓存；重新训练 LR 并覆盖同名有效开发结果。修改实验设计时应另建 E04 目录与协议，保留 E03。

本次保留的是开发阶段比较；42 天的按日重采样区间未校正参数选择，也不足以完整处理跨日相关性。模型训练语料与历史年份的局限见主日志。
