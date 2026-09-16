from pathlib import Path
import sys,json,hashlib,zipfile,re
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B));from run import ROOT,W,metric
R=B/'runs/v1';O=B/'diagnosis';d=pd.read_pickle(R/'inputs.pkl');p=pd.read_csv(R/'predictions.csv');raw=pd.read_pickle(W/'audit/news_index.pkl');raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key');rows=[]
for (sym,split),g in d.groupby(['symbol','split']):
 keys=sorted({k for x in g.news_record_keys for k in x.split('|') if k});a=raw.loc[keys];ages=[]
 for row in g.itertuples():
  ids=[k for k in row.news_record_keys.split('|') if k]
  ages.extend((row.cutoff_utc-pd.to_datetime(raw.loc[ids,'published_utc'],utc=True)).dt.total_seconds().to_numpy()/3600)
 rows.append(dict(symbol=sym,split=split,windows=len(g),dates=g.day.nunique(),news_coverage=float(g.has_news.mean()),news_mean=float(g.news_count.mean()),unique_articles=len(keys),age_over4h=float(np.mean(np.array(ages)>4)) if ages else None,age_median=float(np.median(ages)) if ages else None,up_rate=float(g.label.mean()),abs_return_median=float(g.target_return.abs().median())))
pd.DataFrame(rows).to_csv(O/'coverage_shift.csv',index=False)
# Decompose expert preference, not a new predictive model.
rows=[];slices=[];cors=[]
for (sym,split),g in p[p.phase=='frozen'].groupby(['symbol','split']):
 expert='semantic' if sym=='AAPL' else 'body';a=(g[expert]>=.5)==g.label;b=(g.integrated>=.5)==g.label
 rows.append(dict(symbol=sym,split=split,expert=expert,expert_correct_fusion_wrong=int((a&~b).sum()),expert_wrong_fusion_correct=int((~a&b).sum()),agree=int(((g[expert]>=.5)==(g.integrated>=.5)).sum()),n=len(g)))
 for status,x in g.groupby('has_news'):
  for method in ['price','title','body','semantic','integrated']:slices.append(dict(symbol=sym,split=split,has_news=status,method=method,**metric(x.label,x[method])))
 for a,b in [('price','title'),('price','body'),('price','semantic'),('body','semantic')]:cors.append(dict(symbol=sym,split=split,branch_a=a,branch_b=b,probability_correlation=g[a].corr(g[b]),error_correlation=((g[a]>=.5)!=g.label).astype(int).corr(((g[b]>=.5)!=g.label).astype(int))))
pd.DataFrame(rows).to_csv(O/'mixing_changes.csv',index=False);pd.DataFrame(slices).to_csv(O/'news_slices.csv',index=False);pd.DataFrame(cors).to_csv(O/'branch_correlations.csv',index=False)
# Each of 16 cards is a distinct day: two stocks x two periods x four diagnostic categories.
full=p[p.phase=='frozen'].merge(d[['key','news_record_keys','target_return','cutoff_utc','news_count','text','history_age_hours','ny_hour']],on='key',validate='one_to_one');cards=[]
for (sym,split),g in full.groupby(['symbol','split']):
 expert='semantic' if sym=='AAPL' else 'body';ec=(g[expert]>=.5)==g.label;fc=(g.integrated>=.5)==g.label;used=set()
 masks={'expert_right_fusion_wrong':ec&~fc,'expert_wrong_fusion_right':~ec&fc,'both_wrong':~ec&~fc,'both_right':ec&fc}
 for category,mask in masks.items():
  cand=g[mask&g.has_news.eq(1)].copy();cand['rank']=(cand[expert]-cand.integrated).abs();cand=cand.sort_values(['rank','key'],ascending=[False,True]);cand=cand[~cand.day.isin(used)]
  if cand.empty:continue
  r=cand.iloc[0];used.add(r.day);ks=r.news_record_keys.split('|');articles=[]
  # First inspect the 3 newest available articles; full IDs/titles retained separately.
  order=raw.loc[ks].sort_values('available_utc',ascending=False)
  for a in order.head(3).itertuples():
   with zipfile.ZipFile(W/'raw/news'/a.archive) as z:body=json.loads(z.read(a.member)).get('text','')
   alias=r'\b(?:AAPL|Apple)\b' if sym=='AAPL' else r'\b(?:AMZN|Amazon)\b';paragraphs=[x for x in body.splitlines() if re.search(alias,x,re.I)];excerpt='\n'.join(paragraphs[:2])[:2200] or body[:2200]
   articles.append(dict(key=a.Index,title=a.title,published=str(a.published_utc),available=str(a.available_utc),lag_hours=float((pd.Timestamp(a.available_utc)-pd.Timestamp(a.published_utc)).total_seconds()/3600),excerpt=excerpt))
  cards.append(dict(id=f'{sym}_{split}_{category}',key=r.key,day=r.day,cutoff=str(r.cutoff_utc),label=int(r.label),return_4h=float(r.target_return),news_n=int(r.news_count),category=category,probabilities={k:float(r[k]) for k in ['price','title','body','semantic','integrated']},all_titles=raw.loc[ks,'title'].tolist(),articles=articles))
(O/'case_cards.json').write_text(json.dumps(cards,indent=2,ensure_ascii=False));lines=['# 16个四小时案例：诊断用选择，非独立质量金标准','已扫描全部1607窗口，按股票/开发与后续时期/专家与组合对错四象限选独立日期；每张卡保存全部标题并展示最近3篇的目标段落。判断不是新闻因果归因。']
for c in cards:
 lines.extend([f"\n## {c['id']}",f"{c['key']} 真值={c['label']}；4h收益={c['return_4h']:.4%}；新闻={c['news_n']}；概率={c['probabilities']}"])
 for a in c['articles']:lines.extend([f"### {a['title']}",f"published={a['published']} available={a['available']} delay={a['lag_hours']:.1f}h",a['excerpt']])
(O/'CASE_CARDS.md').write_text('\n\n'.join(lines));print(pd.DataFrame(rows).to_string(index=False));print(pd.read_csv(O/'coverage_shift.csv').to_string(index=False));print('cards',len(cards))
