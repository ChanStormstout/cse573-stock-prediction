"""Separate representation reconstruction and four-hour prediction outcomes."""
from run import *
def main():
 record=json.loads((PRIVATE/'done.json').read_text());d=pd.read_csv(PRIVATE/'predictions.csv',float_precision='round_trip');models=['B','RAW','RANDOM','SSL'];rows=[];monthly=[];seedrows=[];strata=[];z=np.load(PRIVATE/'seed_predictions.npz')
 for model in models:
  for (s,phase),g in d[d.phase.ne('warmup')].groupby(['symbol','phase']):rows.append(dict(method=model,symbol=s,phase=phase,**core.metric(g.label,g[model])))
  for (s,month),g in d[d.phase.ne('warmup')].groupby(['symbol','month']):monthly.append(dict(method=model,symbol=s,month=month,**core.metric(g.label,g[model])))
  for (s,phase,news),g in d[d.phase.ne('warmup')].groupby(['symbol','phase','has_original_news']):strata.append(dict(method=model,symbol=s,phase=phase,original_news=int(news),**core.metric(g.label,g[model])))
 for method in ('RANDOM','SSL'):
  for j,seed in enumerate(SEEDS):
   for (s,phase),g in d[d.phase.ne('warmup')].groupby(['symbol','phase']):seedrows.append(dict(method=method,seed=seed,symbol=s,phase=phase,**core.metric(g.label,z[method][j,g.index])))
 metrics=pd.DataFrame(rows);months=pd.DataFrame(monthly);metrics.to_csv(OUT/'metrics.csv',index=False);months.to_csv(OUT/'monthly_metrics.csv',index=False);pd.DataFrame(strata).to_csv(OUT/'strata_metrics.csv',index=False);sd=pd.DataFrame(seedrows);sd.to_csv(OUT/'seed_metrics.csv',index=False);sd.groupby(['method','symbol','phase'])[['BA','MCC','Brier']].agg(['mean','std']).to_csv(OUT/'seed_summary.csv');d.to_csv(OUT/'predictions.csv',index=False);core.dump(OUT/'training_evidence.json',record);core.dump(OUT/'cv.json',json.loads((PRIVATE/'cv.json').read_text()))
 # Joint-date resampling uses identical draws for both stocks.
 intervals=[]
 for phase,g in d[d.phase.ne('warmup')].groupby('phase'):
  days=sorted(g.day.unique());lookup={day:i for i,day in enumerate(days)};n=len(days)
  for block in (1,5):
   rng=np.random.default_rng(573);weights=[]
   for _ in range(1000):
    starts=rng.integers(0,n,size=int(np.ceil(n/block)));draw=np.concatenate([(start+np.arange(block))%n for start in starts])[:n];weights.append(np.bincount(draw,minlength=n))
   weights=np.array(weights)
   for s,gg in g.groupby('symbol'):
    w=weights[:,[lookup[day] for day in gg.day]];y=gg.label.to_numpy()
    def ba(q):return .5*((w[:,y==1]@q[y==1])/w[:,y==1].sum(1)+(w[:,y==0]@(~q[y==0]))/w[:,y==0].sum(1))
    for base in ('B','RAW','RANDOM'):
     delta=ba(gg.SSL.to_numpy()>=.5)-ba(gg[base].to_numpy()>=.5);lo,hi=np.nanquantile(delta,[.025,.975]);intervals.append(dict(symbol=s,phase=phase,base=base,block_days=block,BA_delta_low=lo,BA_delta_high=hi))
 pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
 outer=months[months.month.between('2018-06','2018-08')].groupby(['method','symbol']).BA.mean().unstack();outer.to_csv(OUT/'outer_monthly_BA.csv');gates=[]
 for base in ('B','RAW','RANDOM'):
  a=months[(months.method=='SSL')&months.month.between('2018-06','2018-08')].set_index(['symbol','month']);b=months[(months.method==base)&months.month.between('2018-06','2018-08')].set_index(['symbol','month']);delta=outer.loc['SSL']-outer.loc[base];gain=float(outer.loc['SSL'].min()-outer.loc[base].min());positive=int(((a.BA-b.BA).groupby('month').mean()>0).sum());gates.append(dict(base=base,weaker_gain=gain,AAPL_gain=float(delta.AAPL),AMZN_gain=float(delta.AMZN),positive_months=positive,passed=bool(gain>=.01 and delta.min()>=-.01 and positive>=2)))
 core.dump(OUT/'advancement.json',gates)
 old=pd.read_csv(ROOT/'outputs/stock_foundation_4h/v1/all_predictions.csv').set_index('key');transition=[]
 for s in ('AAPL','AMZN'):
  g=d[(d.symbol==s)&d.phase.eq('later')].set_index('key');og=old.loc[g.index];common=((og.F2>=.5)!=og.label)&((og.MODERN>=.5)!=og.label)
  for subset,mask in [('all',np.ones(len(g),bool)),('original_no_news',g.has_original_news.eq(0).to_numpy())]:
   a=g.loc[mask];wrong=common.loc[a.index]
   for model in models:
    q=(a[model]>=.5)==a.label;oldcorrect=(og.loc[a.index,'F2']>=.5)==a.label;transition.append(dict(symbol=s,subset=subset,method=model,n=len(a),old_common_errors=int(wrong.sum()),common_fixed=int((q&wrong).sum()),new_errors_vs_F2=int((~q&oldcorrect).sum()),corrected_vs_F2=int((q&~oldcorrect).sum())))
 pd.DataFrame(transition).to_csv(OUT/'error_transitions.csv',index=False)
 coverage=d.groupby(['symbol','phase']).has_sequence.agg(['count','sum']).reset_index();coverage.to_csv(OUT/'coverage.csv',index=False)
 # Preserve fixed case keys chosen before this round's outcomes.
 cases=[];indexed=d.set_index('key')
 for entry in json.loads((ROOT/'outputs/stock_goal60_4h/v1/case_selection.json').read_text()):
  r=indexed.loc[entry['key']];cases.append(dict(**entry,label=int(r.label),has_sequence=int(r.has_sequence),probabilities={m:float(r[m]) for m in models},correct={m:bool((r[m]>=.5)==r.label) for m in models}))
 core.dump(OUT/'cases.json',cases)
 lines=['# 小型价格自监督：实际实验结果','', '**这是有限的掩码重建试验，不是TS2Vec完整复现。全部时期是已暴露历史回测。**','', '先让编码器学习训练期五分钟价格结构，再冻结编码器训练四小时分类器。RANDOM是随机冻结编码器；不等于已经做过从头监督训练同一编码器。','', '| 方法 | AAPL外层月均 | AMZN外层月均 | AAPL开发 | AMZN开发 | AAPL后续 | AMZN后续 |','|---|---:|---:|---:|---:|---:|---:|']
 for m in models:
  vals=[outer.at[m,'AAPL'],outer.at[m,'AMZN']]+[metrics[(metrics.method==m)&(metrics.symbol==s)&(metrics.phase==p)].BA.iloc[0] for p in ('development','later') for s in ('AAPL','AMZN')];lines.append('| '+m+' | '+' | '.join(f'{v*100:.2f}%' for v in vals)+' |')
 lines+=['','B=既有新协议自身价格LR；RAW=相同48根线展平后PCA16；RANDOM=随机冻结卷积编码；SSL=相同初始化经过掩码重建后冻结。其余价格、LR候选、样本与回退相同。','', '## 晋级检查','']
 for g in gates:lines.append(f"- SSL相对{g['base']}：较弱股月均变化{100*g['weaker_gain']:+.2f}pp，AAPL{100*g['AAPL_gain']:+.2f}pp，AMZN{100*g['AMZN_gain']:+.2f}pp；{'通过' if g['passed'] else '未通过'}预设探索线。")
 lines+=['','## 实际训练与限制','',f"- 21次训练期内部早停拟合，21次按选定epoch重新预训练；{len(record['heads'])}次LR拟合。三个种子全部报告，没有选择最好种子。",'- 未标注片段按固定一小时步长取端点；它们重叠且相关，不能当作新增独立市场样本。','- 所有encoder、scaler、PCA只拟合各折过去数据；无新闻窗口也可使用价格编码。缺失必要行情时严格返回B。','- 掩码重建损失降低只证明能恢复部分过去价格结构，不证明获得未来方向信息。','- 时间、缓存、重载、逐指标复算、无输入回退与共同错误统计见verification.json。','- 初始重载测试用不同推理批大小出现float32卷积舍入差，已保留该中止运行；正式运行以相同批大小检查保存/加载等价，没有更改模型或选择指标。','- 没有通过结果扩大模型、调新阈值或融合。独立新时期验证仍缺失。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(outer);print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
