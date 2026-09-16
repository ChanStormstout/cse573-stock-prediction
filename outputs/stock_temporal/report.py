"""Build inspectable development report from saved outputs; no training."""
from pathlib import Path
import json,csv
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;O=B/'results';R=json.loads((O/'results.json').read_text());D=json.loads((O/'data_summary.json').read_text());I=json.loads((O/'descriptive_intervals.json').read_text());names={'current_lr':'当前快照 LR','sequence_lr':'六快照扁平 LR','mlp':'六快照 MLP','gru':'六快照 GRU'}
pct=lambda x:f'{100*x:.2f}%';tab=['| 方法 | AAPL BA | AMZN BA | AAPL Brier | AMZN Brier |','|---|---:|---:|---:|---:|'];monthly=['| 股票 | 方法 | 9 月 BA | 10 月 BA |','|---|---|---:|---:|'];seedtable=['| 股票 | 模型 | 种子 | 训练 BA | 验证 BA | 验证 Brier |','|---|---|---:|---:|---:|---:|']
for k in names:
 vals=[];briers=[]
 for s in ['AAPL','AMZN']:
  m=R['stocks'][s]['models'][k];v=m.get('validation',m.get('seed_mean'));vals.append(pct(v['balanced_accuracy'])+(f" ± {100*m['seed_std']['balanced_accuracy']:.2f}" if 'seed_std' in m else ''));briers.append(f"{v['brier']:.4f}")
  months=[m['monthly'][mo]['balanced_accuracy'] if 'monthly' in m else np.mean([a['monthly'][mo]['balanced_accuracy'] for a in m['seeds'].values()]) for mo in ['2018-09','2018-10']];monthly.append(f'| {s} | {names[k]} | '+ ' | '.join(pct(v) for v in months)+' |')
  if 'seeds' in m:
   for seed,a in m['seeds'].items():seedtable.append(f"| {s} | {k} | {seed} | {pct(a['training']['balanced_accuracy'])} | {pct(a['validation']['balanced_accuracy'])} | {a['validation']['brier']:.4f} |")
 tab.append('| '+names[k]+' | '+' | '.join(vals+briers)+' |')
ci=['| 比较 | 股票/汇总 | BA 差值区间（百分点） |','|---|---|---:|']
for contrast,stocks in I.items():
 for s,(lo,hi) in stocks.items():ci.append(f'| {contrast} | {s} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
lookup=pd.read_csv(B.parent/'stock_finbert/results/article_probabilities.csv').set_index('record_key');allrows=[];cards=['# E08：16 张时间模型诊断卡\n\n程序按预先固定的种子 573，比较 GRU 与同信息量序列 LR；每股四种对错组合各选两例，按概率差优先展示。不是挑最好种子，不是随机总体样本，也不是经过逐篇正文人工审阅的因果解释。标题仅展示当前窗口最先到达的三条，完整来源见记录键。\n'];counts={}
for s in ['AAPL','AMZN']:
 p=pd.read_csv(O/s/'validation_predictions.csv');man=pd.read_csv(O/s/'manifest.csv').fillna({'news_record_keys':''});z=np.load(O/s/'data.npz');va=np.flatnonzero(man.split.eq('validation'));y=p.label.astype(bool);a=p.gru_573.ge(.5);b=p.sequence_lr.ge(.5);p['case_group']=np.select([a.eq(y)&~b.eq(y),~a.eq(y)&b.eq(y),a.eq(y)&b.eq(y)],['GRU改对','GRU改错','共同正确'],default='共同错误');p['probability_gap']=(p.gru_573-p.sequence_lr).abs();counts[s]=p.case_group.value_counts().to_dict();p.insert(0,'symbol',s);allrows.append(p)
 for group in ['GRU改对','GRU改错','共同错误','共同正确']:
  for j,row in p[p.case_group.eq(group)].sort_values('probability_gap',ascending=False).head(2).iterrows():
   i=va[j];n=z['lengths'][i];ix=z['source_indices'][i,:n];keys=man.iloc[i].news_record_keys.split('|') if man.iloc[i].news_record_keys else []
   cards.append(f"## {s} · {row.start_utc} · {group}\n\n- 真实方向：{'上涨' if row.label else '下跌'}\n- 当前快照 LR 上涨概率：{pct(row.current_lr)}\n- 六快照 LR：{pct(row.sequence_lr)}；MLP seed 573：{pct(row.mlp_573)}；GRU seed 573：{pct(row.gru_573)}\n- GRU 反序诊断：{pct(row.gru_reversed_573)}。这是输入扰动，不是另一个经训练选择的候选。\n\n| 快照截止 UTC | 距本次截止（小时） | 4h 新闻量 | 正面均值 | 负面均值 |\n|---|---:|---:|---:|---:|")
   for source in ix:
    age=(pd.Timestamp(man.iloc[i].cutoff_utc)-pd.Timestamp(man.iloc[source].cutoff_utc)).total_seconds()/3600;x=z['current'][source];cards.append(f'| {man.iloc[source].cutoff_utc} | {age:.2f} | {np.expm1(x[20]):.0f} | {x[16]:.3f} | {x[17]:.3f} |')
   cards.append('\n**当前窗口标题示例：**\n')
   for key in keys[:3]:cards.append(f'- {lookup.loc[key,"title"]}（`{key}`）')
   if not keys:cards.append('- 当前窗口无新闻；历史快照仍可能有新闻。')
   cards.append('\n观察边界：预测差异与时间快照共同出现，不能仅凭本卡确定是哪条新闻导致差异。\n')
pd.concat(allrows,ignore_index=True).to_csv(O/'all_504_comparisons.csv',index=False);(O/'case_group_counts_seed573.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2));(B/'CASES_16.md').write_text('\n'.join(cards))
text='''# E08：GRU 未超过同信息量的简单模型

2026-09-15。已实际完成时间建模实验，固定三种随机种子；全部结果为开发验证成绩，最终测试未评价。

## 结论

- **GRU 没有建立优势。** AAPL GRU 的平均 BA 为 49.39%，低于同信息量 LR 的 50.79%；AMZN 为 52.63%，也低于 LR 的 53.58%。不挑选最好种子报告。
- **更多历史信息有候选信号，但不稳定。** AMZN 序列 LR 的 9 月 BA 为 63.17%，10 月降至 44.01%；不能仅凭整体 53.58% 或 9 月高分宣布成功。
- **神经网络的训练表现与验证表现差距明显。** 训练 BA 约 73%–79%，验证接近 50%；Brier 明显变差，符合过拟合/泛化不足的表现。不是对具体市场机制变化的证明。

## 1. 这次用到了论文的什么？

借鉴 [StockNet](https://aclanthology.org/P18-1183/) 将文本与历史价格按时间组织的思路，做小型 GRU 组件实验。原论文还有连续隐变量、变分推断和时间辅助目标，本实验没有实现这些，也没有使用原论文的 Twitter 数据和日频标签。因此应写“受 StockNet 启发的时间模型对照”，不能写“复现 StockNet 失败”。

本次避开 E07 不可靠的 LLM 事件输出，沿用已缓存的原始标题 FinBERT 特征。先前关于结构化事件质量的停止条件继续适用于事件链路；这里单独检验时间建模是否提供帮助，并不声称输入新闻质量问题已解决。

## 2. 输入是什么？同样信息如何保证？

每个快照包含原来的 16 个价格特征与 6 个标题情绪/数量特征，追加距本次截止的时间间隔（log1p 小时）和有效位置标记，总计 24 维。

一个样本排列最多六个快照：**此前五个已结束预测时段的信息快照＋本次截止时刻的信息快照**。这里使用的是先前已保存的截止时刻特征，不把先前目标收益或涨跌标签塞进输入。当前快照本身已经含最近六个完成小时的价格历史；“当前快照 LR”不是只看一个价格点。

之前时段必须在本次截止前结束。因为目标前 5 分钟出预测，紧邻的上一目标小时有时尚未结束，会跳过。原样本只包含非平盘且历史完整的时段，所以六个快照不是必然连续六个自然小时。

- 原样本保持不变：AAPL 996 训练＋252 验证；AMZN 1,004＋252。
- 每股最早 6 行不足六个快照，右侧补零，并使用长度掩码；这些行保留。
- 最老快照距当前截止的中位数为 24 小时，最大 AAPL 97 小时、AMZN 96 小时，包含跨夜、周末及其他间隔。时间间隔显式输入。
- 每个快照仍是过去 4 小时新闻聚合；快照之间可能共享新闻，不能理解为六批独立信息。
- 每股张量分别为 `(1248,6,24)`、`(1256,6,24)`；全部只有训练与验证行。

序列 LR、MLP 和 GRU 读取完全相同的张量。数值缩放只在训练部分的有效位置拟合；补零不参与拟合，有效位置标记保留。GRU 使用 packed sequence 跳过填充。

## 3. 对照与参数选择

1. 当前快照 LR：原 FinBERT＋价格＋数量基线，用于判断新增历史信息是否有用。
2. 六快照扁平 LR：把同一张量展开成 144 维。
3. 六快照 MLP：普通非线性网络，隐层宽度 6，877 个参数。
4. 六快照 GRU：一层、隐层宽度 8，825 个参数。

MLP 的参数量接近 GRU，帮助区分普通非线性容量与递归结构；二者仍有结构与优化差别，不能视为完全匹配。

固定训练 40 轮，AdamW、学习率 0.003、batch 64、梯度裁剪 1，无 dropout，无早停。神经网络种子 573/574/575。LR 比较三个 C；神经网络比较三个 weight decay。均在训练期 6/7/8 月三次向前折选择，神经网络按三个种子和三个月份的平均 BA，再按 Brier 选择。没有用 9–10 月挑种子、轮数或隐层宽度。

两类神经网络各自每股 27 次训练折拟合＋3 次最终训练；共 120 次神经网络拟合。两种 LR 每股各 9 次折拟合＋1 次最终训练，共 40 次 LR 拟合。主训练脚本本机 CPU 用时约 '''+str(round(R['elapsed_seconds']))+''' 秒，不包含输入准备、后续复核与报告。另做一次代表 GRU 重训用于检查确定性，结果复现。

## 4. 同一验证集的结果

BA 是上涨与下跌召回率的平均；Brier 衡量概率误差，越低越好。神经网络“±”为三个种子指标的样本标准差，单位百分点；不是置信区间，也不是概率集成成绩。

'''+ '\n'.join(tab)+'''

### 各月份

'''+ '\n'.join(monthly)+'''

序列 LR 的总体提升主要来自 9 月；10 月两股都低于 50%。GRU 在 AMZN 10 月高于序列 LR，但仍低于 50%，同时整体 Brier 更差。不能挑一个月份为复杂模型背书。

### 种子与训练/验证差距

'''+ '\n'.join(seedtable)+'''

三个种子足以展示本轮初始化波动，不能代表所有随机性。训练日志保存了第 1/10/20/30/40 轮的训练损失。本轮没有根据事后发现再改轮数、加大正则或挑最好种子。

## 5. 不确定性与顺序敏感性

按同一交易日联合重采样两股，500 次；神经网络每次计算单种子 BA 后再取均值。以下是描述性区间，没有校正多模型选择、未穷尽跨日依赖，也不纳入所有可能训练随机性。

'''+ '\n'.join(ci)+'''

将训练完成的 GRU 输入有效位置反序，所有原特征（包括时间间隔）保留。平均上涨概率变化约 24–26 个百分点，说明模型对输入排列敏感；这并不能证明这种敏感性对真实预测有帮助。反序输入是分布扰动，不是另一个经过公平训练的模型，也不能把反序后的下降解释成因果证据。

## 6. 案例与检查

[16 张程序生成的诊断卡](CASES_16.md)固定使用种子 573，覆盖两股的改对、改错、共同错误和共同正确；展示每个快照的时间、新闻量、情绪均值及部分当前新闻标题。它们不是独立人工标注，也不是新闻导致股价变化的证明。[全部 504 个比较窗口](results/all_504_comparisons.csv)保留所有种子及未翻转样本。

独立检查实际通过：原样本和标签不变、每个历史快照合法、新闻截止边界正确、序列模型输入与缩放一致、训练折选参正确、保存权重重现预测、GRU 忽略填充、反序诊断复现、原当前 LR 与 E07 对照复现；代表 GRU 按相同种子重训后也复现。[检查输出](check_result.txt)。

## 7. Insight 与下一步决定

**已观察到：** 在这套固定数据和预算下，增加递归结构未超过同信息量 LR。神经网络能拟合训练数据，但这种拟合没有稳定转移到开发月份；概率置信程度也没有得到相应准确率支持。

**合理解释但尚未证实：** 样本少、快照重叠、弱新闻信号与市场随时间变化，都可能贡献这个差距。当前实验不能区分它们的份额。

**本轮决定：** 不将 GRU 晋级为最终主模型；保留序列 LR 的开发信号与所有负结果，不扩大网络或参数搜索。若另开一轮，最小的后续问题是训练期内部早停/更强正则能否缩小泛化差距；需要新协议与同预算对照，不能在已看的验证月份逐轮挑最优。

最终测试仍未评价。后续论文式完整模型、CausalStock 全量结构化和多股票因果图均未因此变成已完成事项。

## 文件与复现

[运行说明](README.md)、[预先登记协议](protocol.json)、[全部结果 JSON](results/results.json)、[主项目日志](../PROJECT_LOG.md)。
'''
(B/'REPORT.md').write_text(text)
print(json.dumps(counts,ensure_ascii=False));print('Report and 16 case cards written.')
