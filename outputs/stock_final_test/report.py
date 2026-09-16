"""Build the holdout report from saved results without changing any model."""
import json
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;O=B.parent;R=B/'results';P=json.loads((B/'protocol.json').read_text());M=json.loads((R/'prepared.json').read_text());E=json.loads((R/'evaluation/results.json').read_text());A=json.loads((R/'audit.json').read_text())
L=json.loads((O/'stock_robust/results/linear/results.json').read_text());L2=json.loads((O/'stock_text_regularization/results/results.json').read_text())
pred=pd.read_csv(R/'evaluation/predictions.csv');dev=pd.read_pickle(O/'stock_robust/results/data.pkl');test=pd.read_pickle(R/'test_inputs.pkl');names={'price':'价格LR','title_l1':'标题＋L1','body_l1':'全文＋L1','title_l2':'标题＋L2','body_l2':'全文＋L2','body_l1_temperature':'全文L1＋固定温度','training_prior':'训练期类别先验'}
summary=[];changes=[];panel=[]
for s,g in pred.groupby('symbol'):
    q=g.copy()
    for body,title in [('body_l1','title_l1'),('body_l2','title_l2')]:
        ca=q[title].ge(.5).eq(q.label);cb=q[body].ge(.5).eq(q.label);group=np.select([~ca&cb,ca&~cb,~ca&~cb],['改对','改错','共同错误'],default='共同正确');q[body+'_vs_title']=group
        c=pd.Series(group).value_counts();changes.append(dict(symbol=s,comparison=f'{body} minus {title}',fixed=int(c.get('改对',0)),broken=int(c.get('改错',0)),both_wrong=int(c.get('共同错误',0)),both_correct=int(c.get('共同正确',0))))
    panel.append(q)
    for method,spec in E['stocks'][s].items():summary.append(dict(symbol=s,method=method,n=len(g),**spec['overall']))
pd.DataFrame(summary).to_csv(R/'test_scores.csv',index=False);pd.DataFrame(changes).to_csv(R/'direction_changes.csv',index=False);pd.concat(panel,ignore_index=True).to_csv(R/'all_729_comparisons.csv',index=False)
stats=[]
for s in ['AAPL','AMZN']:
    for split,d in [('train',dev[dev.symbol.eq(s)&dev.split.eq('train')]),('validation',dev[dev.symbol.eq(s)&dev.split.eq('validation')]),('test',test[test.symbol.eq(s)])]:
        stats.append(dict(symbol=s,split=split,n=len(d),days=d.start_utc.str[:10].nunique(),up_fraction=float(d.label.mean()),news_coverage=float(d.has_news.mean()),mean_news_count=float(d.news_count.mean()),median_abs_return=float(d.target_return.abs().median())))
pd.DataFrame(stats).to_csv(R/'split_descriptives.csv',index=False)
lines=['# E12：全文 L1 / L2 的最终测试结果','','2026-09-15。用户明确授权后，固定比较方案并实际完成最终测试。','','## 结论','','**AAPL 没有保持开发集上的高分；AMZN 的全文 L2 在这次测试中较好。** 全文 L1 测试 BA 为 AAPL49.24%/AMZN50.92%；全文 L2 为48.06%/55.04%。不能把开发集AAPL59.44%当作最终泛化成绩，也没有一个方法在两股上都明显优于50%的恒定方向参照。','','AMZN全文L2比匹配标题L2提高4.65个百分点，比全文L1提高4.12个百分点。它在11月、12月、1月都超过匹配标题对照；但相对标题的提升区间对时间相关性的处理敏感，仍不足以宣称普遍稳定收益。','','## 1. 这次测试怎样做到固定比较','','- 测试范围：2018-11-01至2019-02-01，AAPL364窗口、AMZN365窗口，两股均62个交易日。所有原定合格窗口保留。',
'- **直接使用已保存的Jan–Aug模型。** 没有加入Sep–Oct重新训练，也没有在测试集选C、词表、卡方特征、阈值或温度。',
'- 主方法是全文L1和L2；加入相同正则的标题对照及价格对照。补充报告之前训练期拟合的全文L1温度校准，两股全部报告。',
'- 特征与任务不变：目标小时C>O；预测截止前五分钟；只用已完成历史价格和四小时内可用新闻；收益字段为C/O−1。',
'- 比较方案与源/模型哈希在首次测试预测前锁定，见[协议](protocol.json)。准备流程从原始价格和新闻重建，精确复现旧2504个训练/开发输入后才执行测试预测。',
'- 这里的“最终测试”指原来保留的时间段；它现在已经被评估和公开。后续基于本结果改模型，不能再将这个时段称为未见过的独立最终测试。','','### 固定的参数','','| 股票 | 价格C | 标题L1 C | 全文L1 C | 标题L2 C | 全文L2 C | 全文L1温度 |','|---|---:|---:|---:|---:|---:|---:|']
for s in ['AAPL','AMZN']:
    cs=[P['models'][s][k]['C'] for k in ['price','title_l1','body_l1','title_l2','body_l2']];lines.append('| '+s+' | '+' | '.join(f'{v:g}' for v in cs)+f' | {P["body_temperature"][s]:.3f} |')
lines+=['','L1/L2各自的C是此前训练期向前验证所选；同一个数值C不代表两种惩罚具有同样的有效复杂度。','','## 2. 从开发集到最终测试','','| 股票 | 方法 | 开发集BA | 最终测试BA | 变化（百分点） |','|---|---|---:|---:|---:|']
for s in ['AAPL','AMZN']:
    for name,source in [('body_l1',L),('body_l2',L2)]:
        a=source['stocks'][s]['paper_stem_body']['validation']['overall']['balanced_accuracy'];b=E['stocks'][s][name]['overall']['balanced_accuracy'];lines.append(f'| {s} | {names[name]} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |')
lines+=['','开发与测试的排名发生变化。分数下降或上升可能涉及时间分布变化、有限样本和此前开发选择等因素；本次测试没有分离出唯一成因。','','## 3. 全部对照与概率质量','','BA为上涨/下跌召回率的平均，越高越好；Brier为概率均方误差，越低越好。固定预测0.5的Brier为0.25；恒定方向的BA为50%。MCC正值表示正向关联，0表示无关联参照。','','| 股票 | 方法 | BA | 普通准确率 | MCC | Brier |','|---|---|---:|---:|---:|---:|']
for row in summary:
    lines.append(f'| {row["symbol"]} | {names[row["method"]]} | {row["balanced_accuracy"]:.2%} | {row["accuracy"]:.2%} | {row["mcc"]:.3f} | {row["brier"]:.4f} |')
lines+=['','关键区别：','','- AAPL全文L1虽然从开发59.44%回落到测试49.24%，仍比测试标题L1的46.33%高2.91个百分点；但它低于50%的恒定方向BA参照，不能说已经有可靠方向预测能力。',
'- AMZN全文L2达到55.04%，高于标题L2的50.39%和全文L1的50.92%；这是本次测试中较好的单股结果。',
'- AMZN全文L2的Brier为.2624，反而比标题L2的.2536差，也高于固定.5概率的.25。判断方向更好，不代表概率更准确。',
'- AAPL全文L1的固定温度将Brier从.3392降为.2567，BA不变；AMZN从.2708变为.3222，校准仍然失败。没有根据测试结果重新拟合温度，也没有临时给L2添加校准。','',
'若按两个股票等权平均，各固定全文方法的BA如下。这不是“各股挑最好方法”后的拼接成绩。','','| 固定方法 | 两股等权平均BA |','|---|---:|']
for k in ['body_l1','body_l2']:
    v=np.mean([E['stocks'][s][k]['overall']['balanced_accuracy'] for s in ['AAPL','AMZN']]);lines.append(f'| {names[k]} | {v:.2%} |')
lines+=['','## 4. 分月份：收益是否集中在一段时间','','| 股票/月 | 窗口数 | 标题L1 | 全文L1 | 标题L2 | 全文L2 |','|---|---:|---:|---:|---:|---:|']
for s in ['AAPL','AMZN']:
    for month,g in pred[pred.symbol.eq(s)].groupby(pred[pred.symbol.eq(s)].start_utc.str[:7]):
        vals=[E['stocks'][s][k]['monthly'][month]['balanced_accuracy'] for k in ['title_l1','body_l1','title_l2','body_l2']]
        lines.append(f'| {s}/{month} | {len(g)} | '+' | '.join(f'{v:.2%}' if v is not None else '不适用' for v in vals)+' |')
lines+=['','2019年2月只有2月1日，各股5个窗口，不能当作一个充分样本的月份。仍完整列出，没有因为分数极端而剔除。AAPL全文方法12月较好、1月回落；AMZN全文L2在三个完整测试月份均超过匹配标题模型。','','## 5. 配对不确定性','','在两股共同交易日上同时重采样，保留同一天全部小时，2,000次；另外使用五个相邻交易日的移动块做敏感性分析。区间是固定模型下的条件比较，没有包含重新训练、历史方案搜索的全部不确定性，没有做多重比较校正。','','| 比较 | 股票 | BA差（百分点） | 单日块95%区间 | 五日块95%区间 |','|---|---|---:|---:|---:|']
for name in ['body_l1 minus title_l1','body_l2 minus title_l2','body_l2 minus body_l1']:
    for s in ['AAPL','AMZN']:
        q=A['intervals']['block_1_days'][name][s];c=q['percentile95'];z=A['intervals']['block_5_days'][name][s]['percentile95'];lines.append(f'| {name} | {s} | {q["delta_BA"]*100:+.2f} | [{c[0]*100:+.2f}, {c[1]*100:+.2f}] | [{z[0]*100:+.2f}, {z[1]*100:+.2f}] |')
lines+=['','AMZN全文L2相对标题的单日区间下界仅+0.04个百分点，五日块区间变为[−1.30,+8.96]，跨零；不能忽略这个敏感性。它相对全文L1的区间在两种设定下为正，但这不等于已证明优于所有基线或在其他股票/时期有效。两股平均的主要对照区间均跨零。','','## 6. 方向变化与数据观察','','| 股票/比较 | 改对 | 改错 | 共同错误 | 共同正确 |','|---|---:|---:|---:|---:|']
for row in changes:lines.append(f'| {row["symbol"]}/{row["comparison"]} | {row["fixed"]} | {row["broken"]} | {row["both_wrong"]} | {row["both_correct"]} |')
lines+=['','这是全部窗口的程序统计，没有把正确预测自动解释为正确语义或因果。所有729个窗口的概率与分组见[完整比较](results/all_729_comparisons.csv)。','','| 股票/划分 | 窗口数 | 上涨比例 | 有新闻比例 | 每窗新闻均值 | 绝对收益中位数 |','|---|---:|---:|---:|---:|---:|']
for row in stats:lines.append(f'| {row["symbol"]}/{row["split"]} | {row["n"]} | {row["up_fraction"]:.2%} | {row["news_coverage"]:.2%} | {row["mean_news_count"]:.2f} | {row["median_abs_return"]:.3%} |')
lines+=['','AAPL测试全部364个窗口有新闻；AMZN195个有新闻、170个无新闻。AMZN全文L2在有新闻/无新闻子集BA为54.77%/54.66%，匹配标题为48.18%/52.61%。全文分支也改变了联合训练得到的价格系数，因此无新闻窗口的变化不能解释成当前窗口读取到了文本。子集BA因类别比例不同，不能直接按窗口数加权重建整体BA。','','## 7. 验证与现在的决定','','- 从原始价格/新闻重建输入，并与原样本、原2504个训练/开发文字特征逐项一致。',
'- 检查全部729个测试窗口的新闻可用时间、历史价格完成时间、目标区间、文章键、词集合与原标签一致。',
'- 对10份固定模型，在原Jan–Aug训练数据、相同配置下独立重训，系数、截距、特征集合与保存模型一致；所有测试概率及整体/月度指标复现。',
'- 固定温度预测与方向不变性质核验通过。历史204个产物和上一轮171个新实验文件仍保持原哈希。',
'- 源模型未覆盖，未新增测试期调参。详见[检查输出](check_result.txt)与[审计](results/audit.json)。','','**现在可写入课程报告的结论：** 全文表示有时优于标题，但收益依赖股票和时段；本次AMZN L2较好，AAPL开发集高分未保持。方向指标和概率指标也可能不同步。保留这组完整测试成绩，不拼接各股事后最优模型来制造一个未预先声明的系统。','','## 复现入口','','[固定协议与模型哈希](protocol.json)、[输入说明](results/prepared.json)、[原始测试结果](results/evaluation/results.json)、[精简得分表](results/test_scores.csv)、[训练/开发/测试描述](results/split_descriptives.csv)。prepared.json中的“未计算预测”仅记录准备阶段状态；evaluation/results.json中的test_evaluated=true记录此次已完成的评价。','','检查已有测试：','','```bash','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_final_test/check.py','```','','重新生成本报告：','','```bash','work/stock-data/finbert-env/bin/python outputs/stock_final_test/report.py','```','','register.py、prepare.py、run.py都保护已有协议或结果，不能覆盖后重选参数。check.py中的训练期重训用于复现验证，不用于更换已经评价的模型。']
(B/'REPORT.md').write_text('\n'.join(lines)+'\n')
(B/'README.md').write_text('# 最终测试已完成\n\n用户授权评估全文L1/L2后，已完成E12固定模型测试。[完整报告](REPORT.md)、[测试分数](results/test_scores.csv)、[协议](protocol.json)、[检查结果](check_result.txt)。\n\n测试区间2018-11-01至2019-02-01已被评估，不再是未见过的保留集。旧报告中“测试未评价”描述的是E12之前的阶段。原权重、原训练参数均未改变。\n')
print('Built final-test report, score table, 729 comparison rows and split diagnostics.')
