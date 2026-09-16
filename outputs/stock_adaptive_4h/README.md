# 四小时项目：复现与演示

阅读顺序：[结论](REPORT.md) → [案例](CASE_REVIEW.md) → [覆盖/质量](COVERAGE_REVIEW.md)。主任务是 **AAPL / AMZN 未来四小时方向**，不是股价数值回归或交易策略。

## 环境与现有文件

在项目根目录运行，使用现有 `work/stock-data/finbert-env` 环境；依赖版本见 `environment.lock.txt`。原始压缩新闻、五分钟价格、已审计输入及冻结 FinBERT 向量保留在原项目位置；本目录不重复打包数百 MB 数据。原始文章授权范围未扩展，不把课堂数据上传公开仓库。

```sh
cd /Users/victor/Documents/Codex/2026-09-14/wox
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

核心代码：

| 文件 | 职责 |
|---|---|
| core.py | 输入哈希合同、训练内特征变换、新闻 offset、校准、静态与条件权重、无标签推断 |
| run.py | 输入准备→分层前向训练→冻结评价；每次新目录、保存每次拟合及所有候选 |
| online.py | 成熟标签的有限在线更新及相同信息对照 |
| evidence.py / audit.py | 目标对象、动作、数值/时间证据抽取；原始覆盖扫描、近似分组、独立复核合同 |
| verify.py | 原始四小时标签、输入/代码/模型哈希、选参和权重重算、未来标签扰动、模型重载 |
| report.py | 所有时期/股票指标、配对区间、案例面板 |
| predict.py / demo.py | 实际加载保存模型进行历史时点推断；真实收益另行显示 |

## 快速核验现有结果

```sh
work/stock-data/finbert-env/bin/python -m unittest discover -s outputs/stock_adaptive_4h -p 'test_*.py' -v
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/predict.py --run outputs/stock_adaptive_4h/runs/zero_bias_v1 --symbol AMZN --start '2018-11-09 16:30:00+00:00'
```

这次真实输出的上涨概率：标题 baseline 53.89%，价格与条件修正47.89%；无新闻时修正严格为零。最后系统保留标题 baseline。预测函数去掉未来标签再调用模型。

## 从现有审计输入重新训练完整流水线

以下目录是新的示例名字；若已存在，换一个名字。默认拒绝覆盖。`run.py` 会验证旧输入/向量缓存指纹并保存本次 inputs、代码快照和模型；不是绕过数据准备直接读取旧预测表。

```sh
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/run.py --output outputs/stock_adaptive_4h/runs/reproduce_free --news-bias free
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/verify.py --run outputs/stock_adaptive_4h/runs/reproduce_free --output outputs/stock_adaptive_4h/runs/reproduce_free/verification.json
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/online.py --run outputs/stock_adaptive_4h/runs/reproduce_free --output outputs/stock_adaptive_4h/online/reproduce_free
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/report.py --run outputs/stock_adaptive_4h/runs/reproduce_free --online outputs/stock_adaptive_4h/online/reproduce_free --output outputs/stock_adaptive_4h/results/reproduce_free
```

第二版使用新目录 `reproduce_zero`，唯一机制参数改为 `--news-bias zero`，然后执行同样核验/在线/报告。报告和在线是各自独立输出目录，不改变训练权重。正式历史运行是 `runs/v3` 与 `runs/zero_bias_v1`；不要把中断的 v1 当结果。

## 覆盖和独立复核

```sh
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/audit.py --output outputs/stock_adaptive_4h/coverage/reproduce
```

正式审计是 `coverage/v3`。组员复核 `independent_check.csv` 时只填写 reviewer、critical_fields_correct、missed_target、uncertain、notes；先阅读对应原文证据，不根据后续股价评判抽取是否正确。复核提案、样本键和哈希不可修改。数值字段正确填1，错误填0；漏抽和不确定项也需要填写。代码中的 `quality_gate(review_path, expected_path)` 每次重新检查，不根据上次已运行条数放行。

当前没有独立复核，因此任何类型没有获准进入关系预测。不要修改 `prediction_accepted` 来手动绕过门槛。若独立检查完成，需要按原计划另立关系输入对照；本轮没有一个已经训练好的关系/Graph 模型可以直接启用。

## 离线 demo

```sh
work/stock-data/finbert-env/bin/python outputs/stock_adaptive_4h/demo.py --run outputs/stock_adaptive_4h/runs/zero_bias_v1 --port 8769
```

浏览器打开 http://127.0.0.1:8769 。可选股票与冻结时期的已审计时间；展示信息截止点、四小时目标区间、基础/分支/系统概率、权重、质量状态及两篇原文证据。勾选“显示实际结果”才另取历史实现收益；模型不接收它。

仅监听本机，页面无外部依赖。进程结束后按上述命令重启。它用于历史演示，不接受任意新数据作为已认证的实时预测。

## 保存与可移植性

每次运行在 sources.json / fits.json 和模型包中保存来源、训练键、时间、候选和哈希；verification.json 记录用于推断的精确文件。当前指纹含本机项目路径，移动项目时应重新生成/核验运行合同，不能删除指纹检查来运行。`environment.lock.txt` 固定版本，原始数据另外保留。学习到的关系不稳定和数据商时钟不能外部认证，是方法/数据限制，单元测试通过不消除这些限制。
