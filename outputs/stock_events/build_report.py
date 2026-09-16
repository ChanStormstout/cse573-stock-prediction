"""Render Markdown findings from saved experiment results; does not retrain models."""
import json,csv,collections
from pathlib import Path
B=Path(__file__).resolve().parent;O=B.parent
read=lambda p:json.loads(p.read_text())
e5=read(O/'stock_content/results/results.json');e6=read(O/'stock_dedup/results/results.json');e7=read(B/'results/results.json');v3=read(B/'results/pilot_v3_summary.json');interval=read(B/'results/descriptive_intervals.json')
labels={int(r['index']):r for r in csv.DictReader((B/'pilot_labels_before_inference.csv').open())};rs=[json.loads(l) for l in (B/'results/pilot_v3_outputs.jsonl').read_text().splitlines()]
with (B/'results/pilot_review_40.csv').open('w') as f:
 fields=['index','partition','symbol','record_key','title','rule_event','event_type','evidence_id','evidence','acceptable_events','acceptable_evidence_ids','event_agreement','joint_event_evidence_agreement','rule_event_agreement','human_verified','review_note'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rs:
  lab=labels[r['index']];row={k:r.get(k,'') for k in fields};row.update(acceptable_events=lab['acceptable_events'],acceptable_evidence_ids=lab['acceptable_evidence_ids'],rule_event_agreement=r['rule_event'] in lab['acceptable_events'].split('|'),human_verified=False,review_note=lab['note']);w.writerow(row)
cards=['# 40 个事件抽取检查案例\n\n2026-09-15。前 20 个是此前模型已处理的诊断样本；后 20 个为新增检查样本。全部来自训练期质量开发集，未使用股票结果制定事件标签。标签为助手在本轮模型输出前对标题与所提供摘录的判断，未经组员独立复核；不是全文金标准。允许的多标签表示分类边界确有歧义，并非看到模型结果后放宽。\n']
for r in rs:
 lab=labels[r['index']];cards.append(f"## {r['index']:02d} · {r['symbol']} · {r['partition']}\n\n**标题：** {r['title']}\n\n- 记录：`{r['record_key']}`\n- 助手预先允许的事件：`{lab['acceptable_events']}`\n- 简单规则：`{r['rule_event']}`\n- 本地模型：`{r['event_type']}`\n- 事件判断一致：{r['event_agreement']}；事件与证据共同符合预先标记：{r['joint_event_evidence_agreement']}\n- 模型所选证据 `{r['evidence_id']}`：{r['evidence'] or '无'}\n- 诊断说明：{lab['note']}\n\n**完整的本次目标摘录（不代表完整新闻正文）：**\n\n"+'\n'.join('> '+line for line in r['target'].splitlines())+'\n')
(B/'CASES_40.md').write_text('\n'.join(cards))
pct=lambda x:f'{100*x:.2f}%'
table5=['| 输入与模型（均含原价格） | AAPL 验证 BA | AMZN 验证 BA |','|---|---:|---:|']
for n,label in [('title_sparse','标题＋稀疏词'),('lead_sparse','等长正文开头＋稀疏词'),('target_sparse','公司相关句＋稀疏词'),('title_sentiment','标题＋FinBERT'),('lead_sentiment','等长正文开头＋FinBERT'),('target_sentiment','公司相关句＋FinBERT')]:table5.append('|'+label+'|'+ '|'.join(pct(e5['stocks'][s][n]['validation']['balanced_accuracy']) for s in ['AAPL','AMZN'])+'|')
table6=['| 聚合方法 | AAPL 验证 BA | AMZN 验证 BA |','|---|---:|---:|']
for n,label in [('article_equal_control','文章等权'),('article_equal_with_counts','文章等权＋额外数量统计'),('cluster_equal','近似簇等权'),('cluster_equal_with_counts','近似簇等权＋额外数量统计')]:table6.append('|'+label+'|'+ '|'.join(pct(e6['stocks'][s]['models'][n]['validation']['balanced_accuracy']) for s in ['AAPL','AMZN'])+'|')
table7=['| 方法 | AAPL 验证 BA | AMZN 验证 BA | 两股等权平均 BA |','|---|---:|---:|---:|']
for n,label in [('price_count','价格＋新闻数量'),('price_finbert','价格＋FinBERT＋新闻数量'),('price_count_events','价格＋新闻数量＋规则事件'),('price_finbert_events','价格＋FinBERT＋新闻数量＋规则事件')]:
 values=[e7['stocks'][s]['models'][n]['validation']['balanced_accuracy'] for s in ['AAPL','AMZN']];table7.append('|'+label+'|'+'|'.join(pct(v) for v in values+[sum(values)/2])+'|')
ci=[]
for comparison,values in interval.items():
 for s,bounds in values.items():ci.append(f"| {comparison} | {s} | [{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}] |")
report='''# 新闻处理与事件特征：已完成对照，尚未找到稳定预测增益

2026-09-15 更新。本轮沿用户授权继续之前的探索；它们是团队选择的改进实验，不是老师指定必做项。

## 结论摘要

- **正文开头有一个候选信号，但不能宣布成功。** E05 的 AAPL 稀疏词模型从标题 52.72% 到正文开头 56.31%；训练期向前验证反而较差，按日差值区间跨零。公司相关句没有一致胜过等长正文开头。
- **去重未形成可靠收益。** E06 的 AAPL 变化约 0.01 个百分点；AMZN 添加数量特征的文章对照优于加数量的去重版本。
- **本轮新增规则事件特征也未形成跨股票收益。** 在 FinBERT 上加事件，AAPL 下降、AMZN 微升；全部 504 个验证窗口中 30 个由错变对、31 个由对变错。
- **本地小模型不进入全量运行。** E07 v3 在新增 20 例上，与预先助手事件判断一致率 10%，规则为 55%；不具备扩大运行的依据。格式合法不代表理解正确。

## 1. 为什么尝试、如何控制变量

已有案例显示：标题会遗漏公司事实、多个主体混杂、转载重复。提出的可检验假设是这些处理问题削弱了新闻特征，而不是事先假定所有新闻都能预测股价。

所有下游实验继续使用原先的完整小时目标、提前 5 分钟截止、4 小时新闻窗口与最近 6 根已完成常规小时价格；保留无新闻样本与全部原验证窗口。AAPL 996 训练 / 252 验证；AMZN 1,004 / 252。训练为 2018 年 1–8 月，开发验证为 9–10 月。C 只在训练期 6/7/8 月三次向前验证选取；词表、筛选和缩放在每折训练部分拟合。这里的开发月份已影响探索设计，不能称为无偏最终评估。

本文 BA 为上涨与下跌召回率的平均；一直猜同一类为 50%。MCC、普通准确率、Brier 和逐月分数均保留在结果 JSON。最终留出测试未做性能评估。本轮没有付费 API 调用，新闻没有发送到外部推理服务。

## 2. E05：含公司名的句子不一定是当前事件

提取器在每条正文中选最多三句含目标公司名称或代码的内容，优先代码、句首命中与金融关键词，保留原文字符位置；没有内容则回退标题。正文开头控制与目标摘录匹配长度，最多 254 内容 token 加两个特殊 token；共 5,951 个公司—文章对。

AAPL 标题回退率 0.99%，AMZN 1.63%；但回退少只能说明容易找到公司名，不能证明抽对。质量集 160 对，其中 40 个锁定样本的展示摘录已有助手审阅；其余 E05 标签仍未完成。未经组员独立复核，不报告金标准抽取准确率或全文片段召回。

'''+ '\n'.join(table5)+'''

**关键负例：** Sigma Planning 减持 Apple 的标题，被提取成去年股价涨幅、下次财报日期和 EPS 估计；Deutsche Bank 的标题含 175 美元目标价，摘录却主要剩评级小标题和历史走势。这说明当前排序更偏爱高频公司/金融表述，未能稳定保存新闻主事件。

AAPL 正文开头稀疏模型的训练期平均 BA 为 51.97%，原标题为 56.51%；开发验证虽达到 56.31%，不能跳过原先选择规则直接晋级。正文开头相对标题的 95% 描述性日分块差值区间为 [-1.98, +9.28] 个百分点。AAPL 公司相关句稀疏模型的全部系数被正则化为零，验证概率恒定，BA 为 50%；这是退化模型的真实结果，不是遗漏运行。

决定：保留正文开头作为探索观察；当前公司相关句提取不作为已经验证有效的最终替代。

## 3. E06：近似转载归并与数量对照

按新闻可用时间处理，仅与过去 24 小时标题比较。相似度阈值 0.95，并检查数字、方向词和股票代码集合；持仓模板更保守。窗口内只用已到达、原本就在窗口内的代表文章，不引入未来转载。

已对初版 100 对候选做助手标题级诊断，发现第 87 对加入新代码 BX，可能是更新，随后在预测结果出现前增加代码集合保护。初版审阅与第二版新抽样分别保留；未宣称 100 对全文人工金标准。归并后的“簇”只是近似文本组，不能等同于经人工验证的真实事件。

'''+ '\n'.join(table6)+'''

两股沿用各自 E05 训练期向前验证选出的表示：AAPL 标题稀疏词、AMZN 正文开头 FinBERT。故这里用于检验聚合改动，不用于直接比较两只股票哪一种表示更优。

AMZN “簇等权”相对“文章等权”的日重采样差值区间下界为 0，观察到的改对没有对应改错；但点数少，且加入数量后差值区间跨零，不能据此宣布稳定事件去重优势。

## 4. E07 前两版：多字段提取先在质量关口失败

前两版均处理同一批前 20 条，未完成原计划的 100 条。

- v1 要求完整 JSON 与逐字证据。严格 schema 合格 0/20；独立核对原文引用时 15/20 是逐字匹配。原汇总中的证据指标先要求 schema 合格，因此为 0；不能解释成全部引用都被改写。
- v2 改为选原文句子编号、由程序回填证据。schema 合格 11/20，schema 合格且证据编号可用 10/20。这是结构与引用可用性，不是内容正确率。
- 常见内容错误是把投资者持仓变化判成公司业绩指引，把 Prime Day 销量判成季度财报。原文有提及 earnings，不代表文章主要事件就是财报。

两个版本均停止，旧文件和失败记录保留。

## 5. 本轮 E07-R：只加一个可检查的事件类别特征

规则只读标题，先选含目标公司的分号分句，再按固定优先级判九类事件：财报、业绩指引、分析师评级、产品/经营、法律监管、持仓、行情回顾、其他、不确定。它不直接规定某类新闻一定导致涨或跌。

对每个原窗口计算九类文章所占比例，再让 LR 学习是否与目标方向相关。数量、价格、新闻集合、时间和选参预算保持一致；没有剔除困难文章，也没有同时启用 E06 近似去重。

'''+ '\n'.join(table7)+'''

在 FinBERT 上加规则事件后，AAPL 38 个方向发生变化：18 个改对、20 个改错；AMZN 23 个变化：12 个改对、11 个改错。全部 61 个变化中净少对 1 个。AMZN 的微小 BA 增幅不能代表整个方法成功。

### 配对日分块差值区间

按同一交易日联合重采样两股，保留同日相关性，500 次；下表单位为百分点。区间只描述开发数据上的有限不确定性，没有纠正多模型选择或全部跨日依赖。

| 改动减对照 | 股票/汇总 | 95% 描述性区间 |
|---|---|---:|
'''+ '\n'.join(ci)+'''

所有新增事件特征对照的区间都跨零。不能声称这些事件比例目前可靠提高了一小时预测。

## 6. 本轮 E07 v3：简化输出仍未通过内容检查

保持同一个本地 Qwen3-1.7B 4bit 固定版本，输入原标题和已有目标摘录；只选择主要事件与一个证据句。推理限制为允许的单字母选项，程序构造 JSON 并从输入回填原文。去掉了重要性、持续时间、实际/预期差异等字段。

40 条中，20 条是此前处理过的诊断样本，另 20 条是新检查样本；全部来自训练期质量开发集。助手在本轮模型输出前记录允许事件和证据，有歧义时预先允许两个事件。它不是独立人工金标准，更不是随机抽取全语料后的总体准确率估计。

| 样本 | 规则与助手事件判断一致 | LLM 与助手事件判断一致 | LLM 事件＋证据共同符合 |
|---|---:|---:|---:|
| 既有诊断 20 条 | 75% | 0% | 0% |
| 新增检查 20 条 | 55% | 10% | 10% |

模型 40 条中选了 37 次 earnings、3 次 product_business。格式正确率由代码约束为 100%，不可作为质量提升。分类任务与 v2 已改变，不能把 v3 的一致率与 v2 schema 合格率当作同一指标画提升曲线。

### 是否是解码程序的错误？

限制器通过人工构造分数检查：恰好九个候选保持有限值、预设最大值正确保留。对第 4、6、12 条的模型分数做独立读取，并用同一提示取消输出限制，三个样例仍输出 A（earnings）。支持“当前模型/提示组合的判断不可靠”，未证明这是该模型在所有任务上的能力上限，也未确定是训练、量化还是提示的具体原因。没有继续改提示追逐分数。

### 规则也有明显边界

新增 20 例的九个不一致项主要落入过宽的 other：公司收购、泛化标题下的具体正文事件、ETF 表现、股价突破等。第 56 条还有德语标题和杂乱链接摘要，尽管上游新闻元数据把它归入英语。规则简单、覆盖有限；这批数据也需要更可靠的主体与主事件标注。

**决定：** 小模型未达到新检查 90% 的候选质量门槛；本轮不做全量 LLM 推理，不进入注意力/GRU。规则事件模型作为已运行的探索对照保留，不替换当前基线。

## 7. 案例与可复核证据

- [40 个事件检查案例](stock_events/CASES_40.md)：给出记录键、预先判断、规则、模型、选中证据和完整输入摘录。
- [40 例可检查表](stock_events/results/pilot_review_40.csv)；[模型输出前的标签](stock_events/pilot_labels_before_inference.csv)。
- [全部 504 个预测比较](stock_events/results/all_504_event_comparisons.csv)：包含未翻转窗口，避免只看改对案例。
- [E05 摘录质量表](stock_content/quality_locked_40.csv)；[E06 初版 100 对标题诊断](stock_dedup/quality_100_pairs_v1.csv)。
- [E05/E06 检查](stock_content/check_result.txt)；[本轮独立检查](stock_events/check_result.txt)；[解码诊断](stock_events/results/decoding_diagnostics.json)。
- [E05 原始指标](stock_content/results/results.json)、[E06 指标](stock_dedup/results/results.json)、[E07-R 指标](stock_events/results/results.json)、[v3 小样本汇总](stock_events/results/pilot_v3_summary.json)。

## 8. 下一步的范围

这一轮完成了正文、去重、事件规则对照和受限 LLM 诊断；没有证明稳定预测提升，也没有完成完整 StockNet / CausalStock。事件提取字段不是因果发现。

若继续改进，最有依据的是先对少量文章建立经组员复核的“目标公司主事件＋证据”标签，把持仓模板的当前事件与旧财报/评级背景分开。一个后续候选可用事件引导的正文选择补足标题，但应另立新实验、增加新检查样本，不能反复在这 20 条上修规则再报告同一成绩。

在此之前，保留传统基线与 FinBERT 作为课程主比较，整理 demo 和报告更有价值。负结果可以解释哪些直觉没有得到数据支持；它不证明新闻从来无法预测股票，也不保证换大模型就会有收益。

## 9. 方法来源和复现边界

公司相关句与稀疏词思路借鉴指定 Alostad & Davulcu 论文；E07 借鉴结构化事件和多维新闻处理动机，未复现完整论文模型。FinBERT 使用固定公开 checkpoint 做金融情绪特征。Qwen 使用本地固定版本进行受限提取诊断。

来源：[指定论文](https://journals.sagepub.com/doi/pdf/10.3233/WEB-170349?download=true)、[FinBERT 模型](https://huggingface.co/ProsusAI/finbert)、[本地 Qwen checkpoint](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit)、[CausalStock](https://proceedings.neurips.cc/paper_files/paper/2024/file/54d689d58fe54c92aee2d732fc49fca8-Paper-Conference.pdf)。来源的前轮阅读记录见主日志和原方案。本轮结论主要依据本地实际实验，而非新增文献结论。

后来的预训练模型用于 2018 年历史文本，是回顾性特征实验；本项目未证明预训练数据与全部历史新闻无重合。模型效果不等于当年可以部署的交易策略。

[本轮代码与运行说明](stock_events/README.md)。
'''
(O/'EXPLORATION_REPORT.md').write_text(report)
