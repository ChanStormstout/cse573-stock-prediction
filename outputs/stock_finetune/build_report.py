from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent
R=json.loads((B/'results/training/results.json').read_text());M=json.loads((B/'results/prepared.json').read_text());A=json.loads((B/'results/audit.json').read_text())
assert set(R['models'])=={'frozen','lora'} and 'elapsed_seconds' in R
lines=['# E10：FinBERT 窗口级 LoRA 微调','','2026-09-15，训练、保存权重复现和输入审计均已完成。','','## 这次确实微调了什么','','以前的 FinBERT 情绪概率和隐藏向量都是冻结表示。本轮用整个预测窗口的方向标签，训练最后两层注意力 Query/Value 的低秩增量；FinBERT 原权重保持冻结。两股共享一个预测头，并保留每股截距。','',
'这是 [LoRA](https://arxiv.org/abs/2106.09685) 的有限应用，不是 FinBERT 全参数微调，也不是完整 StockNet 或 CausalStock。数学上将一层的 W 改成 W+BA，其中 A/B 是可训练的小矩阵；本轮 rank=4、alpha=4。','',
'**为什么按窗口训练：** 同一新闻可能在多个窗口出现，且对应不同涨跌结果。每篇文章分别复制窗口标签会制造错误的独立监督解释。这里先平均窗口内文章表示，再结合当时可用价格计算一个预测损失；梯度通过聚合、固定 PCA 投影和最后两层传到 LoRA。','','## 同输入冻结对照','','- 两臂使用完全相同的目标公司前缀＋原标题，最多 96 个 token，每窗口保留最近 16 条可用新闻，保持全部 2,504 个开发范围窗口及无新闻窗口。',
'- 冻结嵌入和前 10 层；第一次计算后的隐藏状态存为 float16，后两层使用 float32。两臂使用相同缓存。',
'- 文章 pooler 向量均值 → 训练有新闻窗口拟合的 PCA16 → 无新闻投影置零；拼接原 16 个价格特征、全量新闻数量、是否有新闻、保留文章平均年龄；训练期缩放。',
'- 共享线性头 35 个系数＋2 个股票截距。冻结臂 37 个可训练参数；LoRA 臂 24,613 个，其中适配参数 24,576 个。',
'- 头学习率 .003，LoRA .0001，weight decay .1；batch16，梯度裁剪1。每臂只试一个事先固定配置、三个种子 573/574/575。',
'- 每个训练前缀内部留最后20交易日监控 BCE，最多6轮、patience2；选定轮数后重置模型，用完整前缀重新拟合投影/缩放并训练相应轮数。六月/七月/八月外层月份与 Sep–Oct 均不用作早停。','','## 实际输入覆盖','','| 股票/划分 | 窗口 | 原文章出现次数 | 保留次数 | 保留比例 | 受16条上限影响的窗口 |','|---|---:|---:|---:|---:|---:|']
for s in ['AAPL','AMZN']:
 for split,name in [('train','训练'),('validation','开发')]:
  r=M['stocks'][s][split];lines.append(f'| {s}/{name} | {r["windows"]} | {r["original_occurrences"]} | {r["retained_occurrences"]} | {r["retained_fraction"]:.2%} | {r["capped_windows"]} |')
lines+=['',f'共 {M["unique_target_articles"]:,} 个“目标公司—文章”编码，实际有 {M["truncated_titles"]} 个文本超过 token 上限。出现次数包含同一文章在多个窗口重复使用，不等于独立新闻数。','','## 开发集结果','','| 股票 | 模型 | 训练 BA 均值 | 开发 BA 均值±标准差 | 开发 Brier 均值 | 开发 MCC 均值 | 选择轮数 |','|---|---|---:|---:|---:|---:|---|']
summary={}
for s in ['AAPL','AMZN']:
 summary[s]={}
 for kind,name in [('frozen','同输入冻结'),('lora','窗口 LoRA')]:
  runs=R['models'][kind]['seeds'];v=[x['stocks'][s]['validation']['overall'] for x in runs.values()];ba=np.array([x['balanced_accuracy'] for x in v]);br=np.mean([x['brier'] for x in v]);mcc=np.mean([x['mcc'] for x in v]);train=np.mean([x['stocks'][s]['training']['balanced_accuracy'] for x in runs.values()]);epochs=','.join(str(x['epochs']) for x in runs.values())
  summary[s][kind]=dict(BA_mean=float(ba.mean()),BA_std=float(ba.std(ddof=1)),Brier_mean=float(br),MCC_mean=float(mcc),train_BA_mean=float(train))
  lines.append(f'| {s} | {name} | {train:.2%} | {ba.mean()*100:.2f}±{ba.std(ddof=1)*100:.2f}% | {br:.4f} | {mcc:.3f} | {epochs} |')
lines+=['','“±”为三种子 BA 的样本标准差，单位为百分点；这里先算每个种子的指标，再平均，不是选最好种子或概率集成。','','| 股票 | 模型 | 训练期三折 BA 均值 | 九月 BA 均值 | 十月 BA 均值 |','|---|---|---:|---:|---:|']
for s in ['AAPL','AMZN']:
 for kind,name in [('frozen','同输入冻结'),('lora','窗口 LoRA')]:
  r=R['models'][kind];monthly={m:np.mean([x['stocks'][s]['validation']['months'][m]['balanced_accuracy'] for x in r['seeds'].values()]) for m in ['2018-09','2018-10']};lines.append(f'| {s} | {name} | {r["training_cv"][s]["balanced_accuracy"]:.2%} | {monthly["2018-09"]:.2%} | {monthly["2018-10"]:.2%} |')
lines+=['','## 实际改变了多少预测','','| 股票/种子 | 平均概率绝对变化 | 改变方向窗口数 | 改对/改错 |','|---|---:|---:|---|']
for s in ['AAPL','AMZN']:
 for seed,q in A['prediction_changes'][s].items():
  lines.append(f'| {s}/{seed} | {q["mean_absolute_probability_change"]:.6f} | {q["direction_changes"]}/252 | {q["fixed"]}/{q["broken"]} |')
lines+=['','选定的最终轮数为1/1/2。两臂总体预测十分接近；这次低秩更新只造成很小的输出变化。因此结果说明这套保守配置没有获得收益，不能据此说充分微调一定无效。真实梯度/参数更新与有效任务适配是不同的验收条件。','',
'## 配对差与判断','','按共同交易日成对重采样 2,000 次，对每次抽样计算三种子 BA 的均值。以下是 LoRA−同输入冻结的描述性区间：','','| 股票 | BA 差（百分点） | 描述性95%区间（百分点） |','|---|---:|---:|']
for s in ['AAPL','AMZN']:
 q=A['intervals'][s];lines.append(f'| {s} | {q["delta_BA"]*100:+.2f} | [{q["day_CI"][0]*100:+.2f}, {q["day_CI"][1]*100:+.2f}] |')
ci=A['intervals']['equal_stock_mean_CI'];lines += ['',f'两股等权平均区间为 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] 个百分点。该区间不包含全部训练/选方案的不确定性，也没有对反复探索作校正。','',
'应以这组同输入冻结模型判断微调增益。旧 E03 的文本、聚合和预测头不同；直接将两者分数之差称为微调效果是不恰当的。单个种子或月份略好不足以确认改善。','','## 验证与实际计算','','1. 前32条输入对照完整原模型，缓存重建 pooler 的最大绝对误差为 .001258、平均 .00003465；两臂共享该数值近似。',
'2. 零初始化 LoRA 与冻结表示的预飞行最大差为 2.15e−7；适配参数获得非零梯度，冻结基座无梯度。见 [preflight.json](preflight.json)。',
'3. 每个最终 LoRA 的所有原基座权重与缓存原权重逐个张量相同，LoRA B 矩阵实际发生非零更新。',
'4. [check.py](check.py) 核验全部2,504个文章袋的可用时间、最近16条排序、公司前缀、平均年龄、训练期投影和缩放，以及外层/内层时间边界与早停记录。',
'5. 六个保存模型均重新载入并重现504个开发窗口预测与指标；204个历史产物哈希未变。最终测试未评价。','',
f'缓存编码约 {M["encoding_seconds"]:.1f} 秒；两臂全部训练与训练/开发推理用时约 {R["elapsed_seconds"]/60:.1f} 分钟。设备为本地 MPS，没有付费推理。时间是本次实测，不是跨设备保证。','','## 适用边界与下一步','','- 只有两只股票、约2,000个训练窗口，新闻复用使独立信息量进一步受限。这里没有估计有效样本量或预测上限。',
'- 适配表示必须经过冻结表示上拟合的固定 PCA16，可能限制可学习方向；只适配最后两层及单一配置，也不能代表全部微调能力。',
'- 最多16条新闻改变了相对早期模型的输入信息量；匹配冻结对照控制了这点，但不能据此评价“全量全文 FinBERT 微调”。',
'- 现代预训练检查点作用于2018年数据，属于回溯性课程实验；没有完成预训练内容重叠审计。',
'- 本轮不按开发结果继续放大 rank、追加 epoch 或挑最好种子。保留全部结果，再决定课程最终比较；未运行完整 StockNet、CausalStock 或 GNN。','','检查命令（项目根目录）：','','```bash','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_finetune/check.py','```','','[训练协议](protocol.json)、[输入清单和来源](results/prepared.json)、[逐种子结果](results/training/results.json)、[检查结果](check_result.txt)、[完整审计](results/audit.json)。已有输入/缓存/结果受到防覆盖保护；新训练须另设版本与缓存路径。']
(B/'REPORT.md').write_text('\n'.join(lines)+'\n');(B/'results/summary.json').write_text(json.dumps(summary,indent=2));(B/'README.md').write_text('# E10 实验入口\n\n[完整报告](REPORT.md)、[协议](protocol.json)、[检查结果](check_result.txt)。\n\n依次 prepare.py → run.py → check.py；现有结果不可覆盖。重新检查已训练权重不需要重新训练。模型主体见 model.py，汇总报告可由 build_report.py 重建。\n')
print(json.dumps(summary,indent=2))
