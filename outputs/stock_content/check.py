from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib
B=Path(__file__).resolve().parent;ROOT=B.parents[1]/'work/stock-data';sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
texts=pd.read_json(B/'results/article_texts.jsonl',lines=True);bodies=json.loads((B/'results/source_bodies.json').read_text());raw=pd.read_pickle(ROOT/'audit/news_index.pkl');raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key')
for r in texts.itertuples():
 if not r.fallback:assert '\n'.join(bodies[r.record_key][int(a):int(b)] for a,b in r.spans)==r.target
 assert r.target_tokens<=254 and r.lead_tokens<=254
assert not texts[['symbol','record_key']].duplicated().any()
summary={}
for stage in ['stock_content','stock_dedup']:
 base=B.parent/stage;results=json.loads((base/'results/results.json').read_text());intervals={}
 for s in ['AAPL','AMZN']:
  z=results['stocks'][s];models=z['models'] if stage=='stock_dedup' else z
  original=pd.read_csv(B.parent/f'stock_baseline/results/{s}/samples.csv');original=original[original.split.ne('test')].reset_index(drop=True)
  predictions={}
  for name,m in models.items():
   d=pd.read_pickle(base/f'results/{s}/{name}_development.pkl');assert d.start_utc.tolist()==original.start_utc.tolist();assert d.label.tolist()==original.label.tolist()
   assert d.split.isin(['train','validation']).all()
   for row in d.itertuples():
    keys=[k for k in row.news_record_keys.split('|') if k];t=raw.loc[keys,'available_utc'];cut=pd.Timestamp(row.cutoff_utc)
    assert t.le(cut).all() and t.gt(cut-pd.Timedelta(hours=4)).all()
    if stage=='stock_dedup':assert set(row.unit_keys.split('|'))-{''}<=set(keys)
   fit=joblib.load(base/f'results/{s}/{name}.joblib');tr=d[d.split.eq('train')];va=d[d.split.eq('validation')];pred=pd.read_csv(base/f'results/{s}/{name}_validation_predictions.csv')
   np.testing.assert_allclose(fit.predict_proba(va)[:,1],pred.p_up,atol=1e-8)
   cols=fit['features'].transformers_[0][2];np.testing.assert_allclose(fit['features'].named_transformers_['numeric'].mean_,tr[cols].mean())
   grid=pd.read_csv(base/f'results/{s}/{name}_training_cv.csv');g=grid.groupby('C')[['balanced_accuracy','brier']].mean().reset_index();best=g.sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];assert best.C==m['C']
   predictions[name]=pred
  if stage=='stock_content':
   # Original title controls reproduce E04 under the same CV protocol.
   e04=json.loads((B.parent/'stock_improvement/results/results.json').read_text())
   assert abs(models['title_sparse']['validation']['balanced_accuracy']-e04['stocks'][s]['models']['paper_price_bow']['validation']['balanced_accuracy'])<1e-10
   contrasts=[('lead_sparse','title_sparse'),('target_sparse','lead_sparse'),('target_sentiment','title_sentiment')]
  else:
   e05=json.loads((B/'results/results.json').read_text());selected=z['selected_e05']
   assert abs(models['article_equal_control']['validation']['balanced_accuracy']-e05['stocks'][s][selected]['validation']['balanced_accuracy'])<1e-10
   contrasts=[('cluster_equal','article_equal_control'),('cluster_equal_with_counts','article_equal_with_counts')]
  intervals[s]={}
  for new,old in contrasts:
   a=predictions[new];b=predictions[old];days=a.start_utc.str[:10];unique=days.unique();rng=np.random.default_rng(573);delta=[]
   for _ in range(500):
    ix=np.concatenate([np.flatnonzero(days.eq(day)) for day in rng.choice(unique,len(unique),replace=True)]);y=a.label.to_numpy()[ix]
    delta.append(scores(y,a.p_up.to_numpy()[ix])['balanced_accuracy']-scores(y,b.p_up.to_numpy()[ix])['balanced_accuracy'])
   intervals[s][new+' minus '+old]=np.quantile(delta,[.025,.975]).tolist()
 (base/'results/descriptive_intervals.json').write_text(json.dumps(intervals,indent=2));summary[stage]='passed'
c=pd.read_csv(B.parent/'stock_dedup/results/online_clusters.csv').fillna({'parent':''});c['time']=pd.to_datetime(c.available_utc,utc=True,format='mixed')
for s,g in c.groupby('symbol'):
 lookup=g.set_index('record_key')
 for row in g[g.parent.ne('')].itertuples():assert pd.Timedelta(0)<=row.time-lookup.loc[row.parent,'time']<=pd.Timedelta(hours=24)
print('PASS:',json.dumps(summary),'exact source spans; text limits; identical outcomes/rows; historical-only windows/clusters; train-only scaling/C selection; saved predictions and original controls reproduced. Final test not evaluated.')
