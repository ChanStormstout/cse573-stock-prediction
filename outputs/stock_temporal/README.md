# E08：相同信息下的时间模型对照

已完成当前快照 LR、六快照扁平 LR、小型 MLP、GRU。两种网络各用三个固定种子；训练期内选择参数，保留全部结果。**GRU 未证明优于同信息量 LR；最终测试未评价。**

- [完整结果与解释](REPORT.md)
- [16 张诊断卡](CASES_16.md)
- [预先登记协议](protocol.json)
- [独立检查结果](check_result.txt)

## 文件

- `prepare.py`：从现有 E03 开发特征构造当前及五个历史快照，保存来源索引、时间间隔、掩码和源文件哈希。
- `models.py`：共同缩放、825 参数 GRU、877 参数 MLP、训练与预测。
- `run.py`：按固定训练月份、候选参数和种子训练；结果存在时拒绝覆盖。
- `check.py`：样本、来源、时间边界、选参、缩放、权重与预测复现、掩码、代表重训检查，以及配对日区间。
- `report.py`：只读取已保存数据生成报告和程序诊断卡。
- `results/{股票}/data.npz`：同信息量张量、当前特征、目标、有效长度和来源索引。只有 `X` / `current` 进入模型；`y` 只用于监督训练/评价。
- `results/{股票}/manifest.csv`：训练/验证行与时间和记录键。
- `results/{股票}/*_training_grid.csv`：全部参数、月份、种子和指标。
- `results/{股票}/*.pt`、`*.joblib`：模型与训练部分拟合的缩放器，只加载可信文件。
- `results/{股票}/validation_predictions.csv`：每个种子的预测与反序诊断。
- `results/results.json`：训练、验证、月份、种子均值/标准差、训练损失轨迹、配置和代码哈希。
- `results/all_504_comparisons.csv`：全部验证窗口及预先固定种子 573 的诊断分组。

## 复现

从项目根目录运行，依赖现有 E03 特征与本地 `finbert-env`。不是从空目录自动下载数据的发行包。

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_temporal/prepare.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_temporal/run.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_temporal/check.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_temporal/report.py
```

已有结果不可直接覆盖：新实验先另存整个目录、更新协议和来源说明。`run.py` 会拒绝覆盖已存在的 `results.json`；不要仅删除该文件后混用其他旧输出。只查看结果无需重训。`check.py` 会额外重训一次 AAPL GRU seed 573 验证确定性，其余模型从权重复核。

随机种子 573/574/575；CPU 两个线程；软件版本见 `requirements-record.txt`。40 轮、三种 weight decay、一层隐层 8 GRU；所有选择均在训练期 6/7/8 月进行。主结果是单种子指标的平均，不是挑最佳种子，也不是集成概率的得分。

## 方法边界

StockNet 启发的时间组件实验，不包含完整变分模型和辅助目标。当前快照本身含六小时价格历史；六个快照会进一步延长有效历史，且并非连续六个自然小时。跨验证月份使用之前已知的特征属于向前预测，先前验证标签不用于更新模型或构造特征。原 FinBERT 和新闻时间元数据的局限继续存在。
