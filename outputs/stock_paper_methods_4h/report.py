"""Publish compact numerical report, paired intervals and descriptive cases."""
import json
import numpy as np
import pandas as pd
from run import HERE,WORK,metric

PAIRS=[('J1','J0'),('J3','J2'),('N1','N0'),('N1M','N0M'),('F1_calibrated','F1'),('F2_calibrated','F2'),('F1_temperature','F1'),('F2_temperature','F2')]
def intervals(d):
 records=[]
 for (phase,symbol),g in d[d.phase!='warmup'].groupby(['phase','symbol']):
  g=g.sort_values(['day','key']).reset_index(drop=True);daily=list(g.groupby('day',sort=True).indices.values());y=g.label.to_numpy()
  for days in (1,5):
   blocks=[np.concatenate(daily[i:i+days]) for i in range(0,len(daily),days)];rng=np.random.default_rng(573)
   resamples=[np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))]) for _ in range(1000)]
   for new,base in PAIRS:
    a=g[new].to_numpy();b=g[base].to_numpy();delta=[]
    for ix in resamples:
     yy=y[ix];up=yy==1;down=~up
     if not up.any() or not down.any():continue
     qa=a[ix]>=.5;qb=b[ix]>=.5
     ba=.5*((qa[up].mean()-qb[up].mean())+((~qa[down]).mean()-(~qb[down]).mean()))
     br=np.mean((a[ix]-yy)**2-(b[ix]-yy)**2);delta.append((ba,br))
    v=np.asarray(delta);lo,hi=np.quantile(v,[.025,.975],axis=0)
    records.append(dict(phase=phase,symbol=symbol,method=new,baseline=base,block_days=days,BA_low=lo[0],BA_high=hi[0],Brier_low=lo[1],Brier_high=hi[1]))
 return pd.DataFrame(records)

def main():
 p=HERE/'v1';d=pd.read_csv(p/'predictions.csv');m=pd.read_csv(p/'metrics.csv');sel=json.loads((p/'selection.json').read_text());man=json.loads((p/'model_manifest.json').read_text());cal=pd.read_csv(p/'calibration.csv');trans=pd.read_csv(p/'transitions.csv')
 ci=intervals(d);ci.to_csv(p/'paired_intervals.csv',index=False)
 text=['# 四小时文本联合、独立校准与全文章事件聚合实验','',
 '**已实际完成 Phase 1。没有机制通过预先登记的晋级条件，因此 FinModernBERT、Chronos-2、注意力、TabPFN 和 Graph 本轮没有运行。**',
 '', '本轮最清楚的正面观察是：正温度校准降低F2的概率误差，同时保留全部涨跌判断。R1进入FinBERT联合训练带来训练期方向增量，但概率误差超过护栏。全文章事件聚合在AMZN后续期较好，却没有训练期跨月方向增量。不能据后续分数晋级。','',
 '## 协议和实际工作','',
 '- 固定两股四小时任务、1,607原窗口和目标起点前5分钟截止。January–February 233窗口用于初始训练，不伪造OOF；March–August 765前向窗口，development 252，later 357。',
 '- March–August逐月重训并仅用更早月份选C；September之后使用August末冻结的模型。这样与旧基线的信息和训练边界匹配，没有利用开发/后续标签更新模型。',
 '- 训练266个LR模型，重载最大误差 %.3g；端到端约%.1f秒。FinBERT向量复用现有5,078文章缓存，没有新FinBERT/LLM权重训练。' % (max(x['reload_error'] for x in man['fits']),man['seconds']),
 '- J0全文和J2语义对照重现旧F1/F2概率，最大差分别%.3g和%.3g。速度来自缓存复用与小型分类器，不能描述为重新训练大模型。' % (man['parity']['J0'],man['parity']['J2']),
 '- 词表/卡方筛选/PCA16/缩放只拟合对应过去训练范围。候选C固定0.01/0.1/1；匹配旧基线的原始分支OOF选C，所有完整系统评价采用相同无新闻R1回退。',
 '- 原始有新闻、实际可用标题文本、A1合格事件三种状态单独记录。A1事件状态只用于分层，不是独立人工验收。所有现有评价期均为暴露历史回测。','',
 '## 训练期晋级结果','',
 '| 机制 / 匹配对照 | 月均BA变化 | 月均Brier变化 | BA改善月份 | AAPL BA变化 | AMZN BA变化 | 晋级 |','|---|---:|---:|---:|---:|---:|---|']
 for r in sel['advancement']:
  text.append('| %s / %s | %+.2f pp | %+.4f | %d/%d | %+.2f pp | %+.2f pp | %s |' % (r['method'],r['baseline'],100*r['delta_BA'],r['delta_Brier'],r['positive_months'],r['months'],100*r['stock_deltas']['AAPL']['BA'],100*r['stock_deltas']['AMZN']['BA'],'通过' if r['passes'] else '未通过'))
 text+=['','R1＋全文的AMZN训练月均BA下降2.16个百分点；R1＋FinBERT的两股BA均提高，但AAPL/AMZN Brier分别恶化0.0119/0.0240，超过0.002。聚合两组BA平均增量均不为正。以上门槛控制计算预算，不代表显著性判断。','',
 '## 完整系统结果','',
 '每格为 **BA / Brier / AUC**。BA/AUC越高越好，Brier越低越好。完整MCC、涨跌召回、上涨比例、恒定预测标记见metrics.csv；逐月和三种覆盖分层分别见monthly_metrics.csv及subgroup_metrics.csv。','',
 '| 方法 | AAPL开发 | AAPL后续 | AMZN开发 | AMZN后续 |','|---|---|---|---|---|']
 names={'F0':'F0 标题baseline','R1':'R1 近期价格','F1':'F1/J0 全文','F2':'F2/J2/N0 FinBERT文章均值','F6':'F6 旧融合','F1_temperature':'F1 正温度','F1_platt_shrunk':'F1 正斜率Platt（协议选中）','F2_temperature':'F2 正温度','F2_platt_shrunk':'F2 正斜率Platt（协议选中）','J1':'J1 R1＋全文','J3':'J3 R1＋FinBERT','N0M':'N0M 文章等权＋元信息','N1':'N1 事件等权','N1M':'N1M 事件等权＋元信息'}
 for method,name in names.items():
  cells=[]
  for sym,phase in [('AAPL','development'),('AAPL','later'),('AMZN','development'),('AMZN','later')]:
   r=m[(m.symbol==sym)&(m.phase==phase)&(m.method==method)].iloc[0];cells.append(f'{r.BA:.2%} / {r.Brier:.4f} / {r.AUC:.4f}')
  text.append('| '+name+' | '+' | '.join(cells)+' |')
 text+=['','## 校准改善了什么','',
 'F2正温度在四个评价单元均降低Brier，涨跌方向完全不变。AAPL后续从0.2856降至0.2503、AMZN从0.2617降至0.2462。F1正温度并非四格都改善：AMZN开发从0.2481变为0.2557。',
 '', '训练期全局选择规则为F1/F2都选了正斜率Platt，不能事后改成温度赢家。Platt在AAPL后续把F1推成全上涨（BA50%），F2后续BA降至49.32%。这表明根据过去标签学到的校准截距也可能漂移，校准本身同样需要跨时期验证。',
 '', '正温度保留0.5方向阈值；固定正单调变换保持同一新闻分支的排序。完整系统AUC可能因为未校准的无新闻R1与校准新闻分支重新排序而变化；逐月变换不同也会改变合并AUC。这些不应单独称为新闻区分能力提高。','',
 '| 分支/股票 | 方案 | 斜率 | 截距 | 温度(1/斜率) | 过去有新闻OOF数 |','|---|---|---:|---:|---:|---:|']
 for r in cal[(cal.month=='final')&(cal.kind!='identity')].itertuples():text.append(f'| {r.branch}/{r.symbol} | {r.kind} | {r.slope:.5f} | {r.intercept:.5f} | {r.temperature:.3f} | {r.n} |')
 text+=['','AMZN若干温度/斜率达到固定优化边界（温度约99.48），意味着过去分数被大幅压回0.5附近。不能解释为模型更懂新闻；边界没有根据后续表现调整。','',
 '## 全文章聚合、覆盖与归因','',
 '使用原F2每个窗口的完整文章集合。按可用时间顺序，以标题token Jaccard≥0.80且动作、数字和季度守卫一致形成近似转载组；组内平均向量，再对组等权。相似度须满足组内所有成员，避免冲突经中间文章传递合并。没有限制为LLM最多六篇输入。',
 '', 'N0M和N1M使用完全相同六个元信息：文章数、簇数、可识别来源数、最新到达年龄、中位到达年龄、是否全部为重复组。年龄按实际可用时间计算，不冒充首次事件披露时间。来源数不代表独立证据或受众规模。',
 '', f'全部1,607窗口中，{int((d.articles>d.clusters).sum())}窗口发生合并，{int((d.vector_delta>1e-10).sum())}窗口的768维聚合向量改变；完整前后向量与成员表保留私有目录，公开逐窗数量与向量差范数。','',
 '| 股票/时期 | 原始有新闻窗口 | 平均文章数 | 平均簇数 |','|---|---:|---:|---:|']
 for (sym,phase),g in d[d.phase!='warmup'].groupby(['symbol','phase']):text.append(f'| {sym}/{phase} | {int(g.has_original_news.sum())}/{len(g)} | {g.articles.mean():.3f} | {g.clusters.mean():.3f} |')
 text+=['','事后按成员键SHA顺序抽查8个多报道簇，均显示同标题的来源后缀或标点变体；全体共有271种多报道成员组合。这不是独立语义质量验收。守卫读取标题，无法保证正文中的所有季度、数字和否定关系均已核验，因此这里准确称为近似转载组。','',
 'N1M的AMZN后续BA为57.32%，高于F2的54.27%，但与匹配N0M比较的训练月均BA增量仅+0.12个百分点，宏平均为−0.26个百分点。因此记录这个局部好结果，但不据此升级注意力或指定AMZN专属赢家。','',
 '## 改对／改错与具体诊断','',
 '以下为事后描述性案例，不称预先盲选案例、不用于选择或改规则。自动取相关时期最早符合条件的窗口，完整转移数见transitions.csv。','']
 # Specific, verifiable score transitions without inventing semantic reasoning.
 for new,base,sym,condition in [('F1_calibrated','F1','AAPL','wrong'),('N1M','N0M','AMZN','right'),('N1M','N0M','AMZN','wrong'),('J3','J2','AAPL','right')]:
  g=d[(d.phase=='later')&(d.symbol==sym)].copy();a=(g[new]>=.5)==g.label;b=(g[base]>=.5)==g.label
  q=g[(a&~b) if condition=='right' else (~a&b)].sort_values('key')
  if len(q):
   r=q.iloc[0];text.append(f'- {r.key}：{base}概率{r[base]:.4f} → {new}概率{r[new]:.4f}，真实方向{"上涨" if r.label else "下跌"}，{"改对" if condition=="right" else "改错"}；文章/簇={int(r.articles)}/{int(r.clusters)}。模型在其他训练窗口也重新拟合，因此本窗文章数未变化不能证明聚类没有间接影响。')
 text+=['','| 后续股票 | 方法/对照 | 改对 | 改错 |','|---|---|---:|---:|']
 for r in trans[trans.phase=='later'].itertuples():text.append(f'| {r.symbol} | {r.method}/{r.baseline} | {r.changed_right} | {r.changed_wrong} |')
 text+=['','配对1日与5日交易日块区间见paired_intervals.csv（每组1,000次）。两股分别按相同日期重采样；这些是暴露时期的敏感性区间，不作为未经多次探索选择的显著性证据。','',
 '## 观察、解释与尚未验证','',
 '- **观察：** 温度改善F2四格Brier而BA不变；R1联合语义方向OOF更好但概率更差；事件聚合未过方向晋级线。',
 '- **解释：** 过度自信解释了一部分F2概率误差；更及时价格可能改变条件决策，但新的系数可能更不稳定。转载重复可能影响特定月份权重。',
 '- **尚未验证：** 标题近似组是否真实同一事件、现代编码器是否更好、长价格序列基础模型是否补足信息。这里没有独立人工聚类验收，也没有新时期泛化数据。',
 '- **未执行：** FinModernBERT、Chronos-2、注意力、TabPFN及Graph，原因是预先保存的Phase 1晋级线未通过；不是这些模型已经被证伪。',
 '', '## 课程报告建议','',
 '保留F0为主baseline、F1为跨原四格最稳妥的统一全文方法、F2为现代语义对照。新增正温度作为概率质量消融，明确它没有提高方向且不是本轮协议选中的校准器。A1继续作为新闻理解改善但无方向增量的独立结果。此次局部新高不能替代训练期选择。',
 '', '## 论文依据与复现','',
 '本轮只借鉴传播/新闻交互动机，不称完整论文复现：[FinGPT dissemination-aware](https://arxiv.org/abs/2412.10823)、[FININ](https://aclanthology.org/2024.findings-emnlp.189/)。它们的任务规模和指标不能直接移植为两股四小时涨分承诺。',
 '', '运行：`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_paper_methods_4h/run.py --run v2`；新run id防止覆盖。报告脚本当前读取v1；原文、成员表、768维向量、输入表和266个模型位于work/stock-data/paper_methods_4h/v1。公开模型manifest含训练边界、重载误差、权重哈希、来源与代码指纹。',
 '', '关键验证已通过：固定标签/截止、所有训练时间边界、过去月份选择、缓存标题与revision、模型重载、旧对照复现、无新闻精确回退、温度方向保持和数值/否定/季度聚类冲突守卫。']
 (p/'REPORT.md').write_text('\n'.join(text)+'\n')
 print('Report and paired intervals written')
if __name__=='__main__':main()
