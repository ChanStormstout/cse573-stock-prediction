import json,re,sys
from pathlib import Path
import pandas as pd,numpy as np
B=Path(__file__).resolve().parent;ROOT=B.parents[1]/'work/stock-data'
sys.path.insert(0,str(B.parent/'stock_baseline'))
from run_baseline import scores
news=pd.read_pickle(ROOT/'audit/news_index.pkl');news['record_key']=news.archive+'::'+news.member;news=news.set_index('record_key')
probs=pd.read_csv(B.parent/'stock_finbert/results/article_probabilities.csv').set_index('record_key')
allrows=[];windows={}
for stock in ['AAPL','AMZN']:
 d=pd.read_csv(B.parent/f'stock_finbert/results/{stock}/development_features.csv').fillna({'news_record_keys':''})
 v=d[d.split.eq('validation')].reset_index(drop=True);pred=pd.read_csv(B.parent/f'stock_finbert/results/{stock}/validation_predictions.csv')
 assert pred.start_utc.tolist()==v.start_utc.tolist()
 for j,r in enumerate(v.itertuples()):
  keys=[k for k in r.news_record_keys.split('|') if k];w=news.loc[keys].copy();cut=pd.Timestamp(r.cutoff_utc)
  assert w.empty or (w.available_utc.le(cut)&w.available_utc.gt(cut-pd.Timedelta(hours=4))).all()
  w['publication_age_hours']=(cut-w.published_utc).dt.total_seconds()/3600
  w['template_flag']=w.title.str.contains(r'\b(?:stake|holding|holdings|position|shareholder|holder)\b',case=False,regex=True)
  w['multiple_flag']=w.title.str.contains(';',regex=False)|w.title.str.contains(r'\b(?:Apple|AAPL)\b',case=False,regex=True)&w.title.str.contains(r'\b(?:Amazon|AMZN)\b',case=False,regex=True)
  w=w.join(probs[['positive','negative','neutral']]);cid=f'{stock}-{j+1:03d}'
  c=pred.loc[j,'price_count_control'];f=pred.loc[j,'price_finbert'];cc=int(c>=.5)==r.label;fc=int(f>=.5)==r.label
  group='both_correct' if cc and fc else 'finbert_hurts' if cc else 'finbert_helps' if fc else 'both_wrong'
  row={'case_id':cid,'symbol':stock,'month':r.start_utc[:7],'start_utc':r.start_utc,'cutoff_utc':r.cutoff_utc,'label':r.label,'target_return':r.target_return,'news_count':len(w),'control_p_up':c,'finbert_p_up':f,'group':group,'p_delta':f-c,'control_correct':cc,'finbert_correct':fc,'history_age_hours':r.history_age_hours,'sentiment_std':r.sentiment_std,
       'lag_gt4_fraction':float(w.lag_hours.gt(4).mean()) if len(w) else None,'lag_gt24_fraction':float(w.lag_hours.gt(24).mean()) if len(w) else None,'median_publication_age_hours':float(w.publication_age_hours.median()) if len(w) else None,'template_fraction':float(w.template_flag.mean()) if len(w) else None,'multi_proxy_fraction':float(w.multiple_flag.mean()) if len(w) else None}
  row['news_band']='none' if not len(w) else '1-3' if len(w)<=3 else '4+'
  row['stale_band']='no_news' if not len(w) else 'majority_lag_gt4' if row['lag_gt4_fraction']>=.5 else 'majority_lag_le4'
  row['move_band']='under_0.1pct' if abs(r.target_return)<.001 else 'at_least_0.1pct'
  row['ny_hour']=pd.Timestamp(r.start_utc).tz_convert('America/New_York').hour
  allrows.append(row);windows[cid]=w.reset_index()[['record_key','title','published_utc','available_utc','lag_hours','publication_age_hours','template_flag','multiple_flag','positive','negative','neutral','archive','member']].to_dict('records')
a=pd.DataFrame(allrows);a.to_csv(B/'all_504_windows.csv',index=False)
# Stratified diagnostic set: up to six per stock/month/outcome cell, fill to 96 by stock/month.
chosen=[]
for _,g in a.groupby(['symbol','month','group']):chosen.extend(g.sample(min(6,len(g)),random_state=573).index)
for _,g in a.groupby(['symbol','month']):
 n=24-int(g.index.isin(chosen).sum());remaining=g.loc[~g.index.isin(chosen)]
 chosen.extend(remaining.sample(min(n,len(remaining)),random_state=574).index)
sel=a.loc[sorted(chosen)];assert len(sel)==96;sel.to_csv(B/'selected_96_cases.csv',index=False)
(B/'all_window_articles.json').write_text(json.dumps(windows,default=str,ensure_ascii=False,indent=2))
summary={'total_windows':len(a),'detailed_cases':len(sel),'case_type':'algorithmically generated diagnostic cards, not 96 manually reviewed causal explanations','stocks':{}}
for stock,g in a.groupby('symbol'):
 st={'outcome_counts':g.group.value_counts().to_dict(),'news_windows':int(g.news_count.gt(0).sum()),'median_news':float(g.news_count.median()),'mean_window_lag_gt4_fraction':float(g.lag_gt4_fraction.mean()),'median_window_template_fraction':float(g.template_fraction.median()),'slices':{}}
 for col in ['month','news_band','stale_band','move_band','ny_hour']:
  st['slices'][col]={str(k):{'n':len(z),'control':scores(z.label,z.control_p_up.to_numpy()),'finbert':scores(z.label,z.finbert_p_up.to_numpy())} for k,z in g.groupby(col)}
 summary['stocks'][stock]=st
(B/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
lines=['# 96 个分层诊断案例','', '这是程序生成的可追溯案例卡，不等于人工逐条审阅了 96 个案例。覆盖每只股票每个月 24 个窗口；优先覆盖四类正确/错误组合。全体 504 个窗口的统计才用于总体描述；以下选择不能估计错误原因占比。标题规则只作诊断线索，不能视为已验证的实体或事件标签。','', '每张卡只显示发布时间最接近截止点的最多 3 条标题，其余新闻在 `all_window_articles.json` 的同名 case_id 下完整保留。案例差异来自两个已选参模型，不能严格归因于单一情绪变量。','']
for r in sel.itertuples():
 lines += [f'## {r.case_id} | {r.month} | {r.group}', '',f'- 预测：{r.cutoff_utc}；目标开始：{r.start_utc}。实际收益：{r.target_return:.3%}。',f'- 上涨概率：数量对照 {r.control_p_up:.3f} → FinBERT {r.finbert_p_up:.3f}。',f'- 新闻 {r.news_count} 条；历史价格距截止点 {r.history_age_hours:.2f} 小时。']
 if r.news_count:
  lines += [f'- 窗口新闻中抓取滞后 >4h 比例 {r.lag_gt4_fraction:.1%}；持仓词线索 {r.template_fraction:.1%}；多公司代理规则 {r.multi_proxy_fraction:.1%}。']
  w=sorted(windows[r.case_id],key=lambda z:z['published_utc'],reverse=True)[:3]
  for n in w:lines += [f"  - {n['title']} | 发布 {n['published_utc']} | 可用 {n['available_utc']} | P(+/-/0)={n['positive']:.2f}/{n['negative']:.2f}/{n['neutral']:.2f}"]
 else:lines+=['- 无新闻：差异不能来自这个窗口的新情绪内容；加入特征后全局参数变化也会改变空窗口预测。']
 lines+=['- 解释范围：以上记录说明输入与预测差异；不证明新闻导致该次价格变化。','']
(B/'CASES_96.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
