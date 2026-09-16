# CSE 573 · 全面机制实验交付

## 从哪里看

- [完整结果与解释](runs/v1/report/REPORT.md)
- [离线交互 demo](runs/v1/report/demo.html)：双击即可，无网络/API依赖；选择股票与历史小时。
- [完整逐月指标](runs/v1/evaluation/monthly.csv)、[逐窗口概率](runs/v1/evaluation/all_predictions.csv)
- [配对单日/五日块区间](runs/v1/evaluation/paired_intervals.json)
- [案例面板](runs/v1/evaluation/case_cards.json)、[改对/改错统计](runs/v1/evaluation/case_counts.csv)
- [事件复核表](runs/v1/M06/human_review.csv)、[漏抽复核材料](runs/v1/materials/article_recall_review.csv)
- [验证结果](runs/v1/verification.json)、[旧基线重现核对](runs/v1/materials/matched_reference_parity.json)
- [免费外部数据调查](EXTERNAL_DATA.md)
- [课堂报告与演示讲稿材料](COURSE_REPORT_MATERIALS.md)

## 已实现的机制与状态

| 机制 | 状态 | 主要文件 |
|---|---|---|
| M01 | 完成：固定协议配置入口、独立目录、哈希、模型、概率、运行成本、拒绝覆盖 | runner.py / core.py |
| M02 | 完成：108个容量折拟合、54个学习曲线拟合；12个最终容量模型 | diagnostics.py / runs/v1/M02 |
| M03 | 完成：四种更新方式，48个月度模型/副本；复现E13扩展参照 | diagnostics.py / runs/v1/M03 |
| M04 | 完成：5种到达前后区间、文章/在线近似事件组、盘外/迟到/缺失覆盖 | news_diagnostics.py / runs/v1/M04_v2 |
| M05 | 完成：全部78,055原始新闻记录正文覆盖检查；免费来源调查 | runs/v1/M05 / EXTERNAL_DATA.md |
| M06 | 代码、抽取、回归检查和材料完成；**未正式验收** | events.py / runs/v1/M06 |
| M07 | offset修正器、条件检查、精确回退已实现/测试；**未训练**，质量门槛未满足 | correction.py / runs/v1/M07_v2 |
| M08 | 完成：7,861篇标题重新编码；PCA8/16/32、3C、OOF权重融合 | semantic.py / runs/v1/M08 |
| M09 | 完成：两股×三个种子×20,000 PPO步，4类规则基线，3个成本设置 | trading.py / runs/v1/M09 |
| 注意力/新LoRA | 未进入：预设跨月收益门槛未通过 | runs/v1/advanced_gate.json |
| Graph | 证据关系存储完成；消息传递未进入：质量/规模门槛不足 | runs/v1/M06/evidence_graph.json |

本轮所有新成绩均为**探索性历史回测**。没有新的独立保留期，未使用付费服务，没有启动Sol作业。固定小型PPO在本地CPU足够；FinBERT为本地冻结推理。高级GPU训练用量为0。

## 重现

在项目根目录运行。已有数据与FinBERT缓存是先决条件；不会自动获取课程私有数据。依赖版本见 `requirements.txt`；本机环境为 `work/stock-data/finbert-env/bin/python`。新机器应复制有权限的课程数据与既有输出依赖，保持项目目录结构。

```bash
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/runner.py prepare --run outputs/stock_comprehensive/runs/replay_01
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/runner.py train --run outputs/stock_comprehensive/runs/replay_01 --mechanism all
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/runner.py evaluate --run outputs/stock_comprehensive/runs/replay_01
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/runner.py report --run outputs/stock_comprehensive/runs/replay_01
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/runner.py verify --run outputs/stock_comprehensive/runs/replay_01
```

配置默认 `configs/main.json`，也可显式 `--config`；这是本轮固定协议，未实现的配置改动会拒绝，不能静默扩大网格。可用 `--mechanism M02` 等分阶段运行，M07依赖M06。每个阶段默认拒绝覆盖；失败运行也保留，修复后用新实验目录。不能把重跑旧数据当作新测试。

可选图表和额外材料：

```bash
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/plots.py --run outputs/stock_comprehensive/runs/replay_01
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/finalize.py --run outputs/stock_comprehensive/runs/replay_01
```

`finalize.py` 含本轮浏览器人工检查记录，新的运行需另做UI核验，不应把该记录当成新运行已做浏览器检查。第一次模块开发期间的失败与修复记录保留在 `runs/v1/materials/provenance_note.json`。`sealed_inputs.json` 是封存清单，不把事后封存称作事前独立注册。

## 数据与模型细节

- 原分类3,233窗口保持不变。选择仍在6/7/8月训练期向前折内；9/10月开发，11月后为已暴露回测。
- M02学习曲线按两股共同完整日期抽样，含首尾日期、按时间分层，保持历史跨度；100%三个种子是相同数据的重复，并非三份独立数据。
- M08文本单模型仅用语义+新闻数量/有无，价格模型独立；融合权重只由训练期OOF选出。AAPL最终权重0必须明确展示。
- PPO rollout=100、batch=50，恰好20,000步，不因完整rollout超出预算。actor和critic各一层16单元；奖励无额外缩放。
- RL从原始5分钟线重建，平盘保留，缺必要线的整日剔除；执行价格为目标小时开盘，持仓连接下一成交开盘，末次到最后完整小时收盘。每个episode日终归零。
- M06审查单位为近似事件组，已见过的错误组进开发；锁定检查集各类型数量仍不足30，独立复核数为0。机构空值、纠错放弃与未定期间不能算完整事件通过。
- M07目前仅代码/合成边界测试，未运行获准的事件修正训练；不将该分支的代码存在写成效果证据。
- 初次独立运行的SemanticLR模型含`__main__`类路径；`verification.py`显式注册兼容别名。通过runner新运行将保存正常`semantic.SemanticLR`路径。

## 诚实的提交边界

这份交付包括可运行实验、结果、离线演示和课程报告材料。团队成员署名、贡献、最终排版/页数与教师最终rubric仍需组内核对；没有替组员完成独立质量签字，也未提交Canvas。

### 用实际保存权重重算一个窗口

```bash
work/stock-data/finbert-env/bin/python outputs/stock_comprehensive/predict.py --stock AMZN --time '2018-09-04 13:30:00+00:00'
```

该例重算价格62.8417%、文本46.9812%、融合50.9464%，与离线demo一致。仅接受原开发/已暴露时期的合法窗口与明确时区。
