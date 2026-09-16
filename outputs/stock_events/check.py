from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,joblib
from events import EVENTS
B=Path(__file__).resolve().parent;OUT=B/'results';res=json.loads((OUT/'results.json').read_text());old=json.loads((B.parent/'stock_improvement/results/results.json').read_text());events=pd.read_csv(OUT/'article_events.csv')
assert not events[['symbol','record_key']].duplicated().any();assert set(events.event_type)<=set(EVENTS)
idx=pd.read_pickle(B.parents[1]/'work/stock-data/audit/news_index.pkl');idx['record_key']=idx.archive+'::'+idx.member;idx=idx.set_index('record_key')
predictions={};flips=[]
for s in ['AAPL','AMZN']:
 ref=pd.read_csv(B.parent/f'stock_baseline/results/{s}/samples.csv');ref=ref[ref.split.ne('test')].reset_index(drop=True);ev=events[events.symbol.eq(s)].set_index('record_key');predictions[s]={}
 for name,m in res['stocks'][s]['models'].items():
  d=pd.read_pickle(OUT/s/f'{name}_development.pkl');assert d.start_utc.tolist()==ref.start_utc.tolist();assert d.label.tolist()==ref.label.tolist();assert d.split.isin(['train','validation']).all()
  totals=d[['event_'+e for e in EVENTS]].sum(axis=1);np.testing.assert_allclose(totals,d.has_news)
  for r in d.itertuples():
   keys=[k for k in r.news_record_keys.split('|') if k];cut=pd.Timestamp(r.cutoff_utc);times=idx.loc[keys,'available_utc'];assert times.le(cut).all() and times.gt(cut-pd.Timedelta(hours=4)).all()
   for e in EVENTS:assert abs(getattr(r,'event_'+e)-(sum(ev.loc[k,'event_type']==e for k in keys)/len(keys) if keys else 0))<1e-12
  fit=joblib.load(OUT/s/f'{name}.joblib');tr=d[d.split.eq('train')];va=d[d.split.eq('validation')];pred=pd.read_csv(OUT/s/f'{name}_validation_predictions.csv');p=fit.predict_proba(va)[:,1];np.testing.assert_allclose(p,pred.p_up,atol=1e-10)
  changed=va.copy();changed['label']=1-changed.label;changed['target_return']=999.;np.testing.assert_allclose(fit.predict_proba(changed)[:,1],p)
  cols=fit['features'].transformers_[0][2];np.testing.assert_allclose(fit['features'].named_transformers_['numeric'].mean_,tr[cols].mean())
  grid=pd.read_csv(OUT/s/f'{name}_training_cv.csv');g=grid.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]);assert g.iloc[0].C==m['C']
  predictions[s][name]=pred
 # Independent E04 controls under unchanged parameter protocol.
 for name,key in [('price_count','price_count'),('price_finbert','price_sentiment')]:
  if key not in old['stocks'][s]['models']:
   matches=[k for k in old['stocks'][s]['models'] if ('count' in k if name=='price_count' else k in ['price_finbert','price_sentiment','sentiment'])]
   if len(matches)!=1:raise ValueError((key,list(old['stocks'][s]['models'])))
   key=matches[0]
  assert abs(res['stocks'][s]['models'][name]['validation']['balanced_accuracy']-old['stocks'][s]['models'][key]['validation']['balanced_accuracy'])<1e-10
 a=predictions[s]['price_finbert_events'];b=predictions[s]['price_finbert'];new=a.p_up.ge(.5);base=b.p_up.ge(.5);truth=a.label.astype(bool)
 for i in range(len(a)):
  flips.append(dict(symbol=s,start_utc=a.iloc[i].start_utc,label=int(truth.iloc[i]),baseline_p=float(b.iloc[i].p_up),events_p=float(a.iloc[i].p_up),changed=bool(new.iloc[i]!=base.iloc[i]),baseline_correct=bool(base.iloc[i]==truth.iloc[i]),events_correct=bool(new.iloc[i]==truth.iloc[i])))
def ba(y,p):
 z=p>=.5
 return .5*((z[y==1]==1).mean()+(z[y==0]==0).mean())
contrasts=[('price_count_events','price_count'),('price_finbert_events','price_finbert')];intervals={}
for new,base in contrasts:
 name=new+' minus '+base;intervals[name]={};deltas={s:[] for s in ['AAPL','AMZN']};days=sorted(set().union(*[set(predictions[s][new].start_utc.str[:10]) for s in predictions]));rng=np.random.default_rng(577)
 groups={s:{day:np.flatnonzero(predictions[s][new].start_utc.str[:10].eq(day)) for day in days} for s in predictions}
 for _ in range(500):
  draw=rng.choice(days,len(days),replace=True)
  for s in predictions:
   ix=np.concatenate([groups[s][day] for day in draw]);a=predictions[s][new];b=predictions[s][base];y=a.label.to_numpy()[ix];deltas[s].append(ba(y,a.p_up.to_numpy()[ix])-ba(y,b.p_up.to_numpy()[ix]))
 for s in deltas:intervals[name][s]=np.quantile(deltas[s],[.025,.975]).tolist()
 intervals[name]['equal_stock_mean']=np.quantile(np.mean(list(deltas.values()),axis=0),[.025,.975]).tolist()
(OUT/'descriptive_intervals.json').write_text(json.dumps(intervals,indent=2));pd.DataFrame(flips).to_csv(OUT/'all_504_event_comparisons.csv',index=False)
rows=[json.loads(l) for l in (OUT/'pilot_v3_outputs.jsonl').read_text().splitlines()];assert len(rows)==40;assert len({(r['symbol'],r['record_key']) for r in rows})==40
for r in rows:
 sources=[r['title']]+[s for s in r['target'].split('\n') if s.strip()];mapping={f'S{i}':s for i,s in enumerate(sources)};assert r['evidence']==mapping.get(r['evidence_id'],'');assert r['event_type'] in EVENTS
summary=json.loads((OUT/'pilot_v3_summary.json').read_text());assert summary['label_sha256']==hashlib.sha256((B/'pilot_labels_before_inference.csv').read_bytes()).hexdigest();assert summary['rule_sha256']==res['rule_sha256']
print('PASS: identical original rows/outcomes; historical-only article windows; event proportions independently recomputed; probabilities reproduced; target mutation invariance; training scaling/C selection; E04 controls reproduced; v3 source evidence and frozen label/rule hashes. No final test evaluation. Descriptive paired-day intervals saved.')
