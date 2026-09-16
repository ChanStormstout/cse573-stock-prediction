# E07：规则事件对照与本地模型修复试验

已完成：两股各四组 LR 对照、40 条本地模型事件/证据选择、40 条助手预先判断与对比、504 个验证窗口比较及独立检查。没有运行最终测试，没有全量 LLM 推理。结论与病例见 [完整探索报告](../EXPLORATION_REPORT.md)、[40 个案例](CASES_40.md)。

## 方法和文件

- `protocol.json`：新预测和 v3 输出前登记的设计与上限。
- `events.py`：固定标题分句、九类规则；只产生事件类别，不直接猜股价。
- `run.py`：按原窗口聚合事件比例，在固定训练期向前验证协议下训练四组 LR。
- `pilot_labels_before_inference.csv`：模型输出前的助手判断，非组员金标准。
- `pilot_v3.py`、`prompt_v3.txt`：本地模型单字母选择事件与证据，程序回填原文。
- `diagnose_decoding.py`：三条固定样例检查受限与不受限生成；不用于调参。
- `check.py`：时间、样本、特征、模型输出、选参、源证据、标签/规则哈希检查；生成按同一交易日联合重采样两股的描述性区间。
- `build_report.py`：从已保存结果生成 Markdown 报告及案例；不重新训练。
- `results/`：模型、特征、参数网格、逐窗口预测、JSON 指标、全部 504 个窗口比较与小样本输出。

## 当前工作区运行

在项目根目录运行，依赖本地原始文件、E02/E03/E04/E05 产物和两个虚拟环境，尚不是从空目录启动的独立发行包。

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 work/stock-data/finbert-env/bin/python outputs/stock_events/run.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 work/stock-data/structured-env/bin/python outputs/stock_events/pilot_v3.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 work/stock-data/structured-env/bin/python outputs/stock_events/diagnose_decoding.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 work/stock-data/finbert-env/bin/python outputs/stock_events/check.py
/Users/victor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 outputs/stock_events/build_report.py
```

**保留旧结果：** `run.py` 会覆盖同名预测结果；新实验应先另存整个实验目录并修改其协议。`pilot_v3.py` 检测已有输出会拒绝重跑，避免把不一致的新旧模型输出混合。E05/E06 的 prepare 脚本会重新生成质量表，包含审阅的版本必须先保留。单纯查看结果不需要重新训练。

FinBERT 环境和 MLX 环境分开安装。固定模型：`mlx-community/Qwen3-1.7B-4bit`，revision `3b1b1768f8f8cf8351c712464f906e86c2b8269e`。它使用已有本地缓存；本轮没有远程新闻推理或付费 API。后来的模型用于历史数据的局限见主报告。

## 验证与解释边界

`check_result.txt` 保存本轮实际通过的检查。程序检查不证明原始新闻时间戳全部正确。格式合法是选择器的结构约束，不等于模型的语义准确率。40 个检查样本的助手判断未经独立人工复核，不能推广为全语料质量；20 个新增样本与前 20 个既有诊断分开汇总。

所有报告中的预测结果都是开发验证成绩，最终测试未评价。原先 E03 按开发验证选参，本轮按训练期向前验证选参；不要把不同协议的分数混在同一公平比较表中。
