from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
r=json.loads((B/'results/results.json').read_text());old=json.loads((B.parent/'stock_baseline/results/results.json').read_text());ci={}
for s in ['AAPL','AMZN']:
 folder=B/'results'/s;d=pd.read_csv(folder/'feature_manifest.csv').fillna({'news_record_keys':''});e=np.load(folder/'development_embeddings.npy');base=pd.read_csv(B.parent/f'stock_baseline/results/{s}/samples.csv').fillna({'text':''});base=base[base.split.ne('test')].reset_index(drop=True)
 assert len(d)==len(e)==len(base);assert d.start_utc.tolist()==base.start_utc.tolist();assert d.label.tolist()==base.label.tolist();assert set(d.split)=={'train','validation'}
 assert (pd.to_datetime(d.recent_end,utc=True)<=pd.to_datetime(d.cutoff_utc,utc=True)).all()
 assert (np.expm1(d.fresh_count)<=d.news_count+1e-7).all()
 d['text']=base.text;d=pd.concat([d,pd.DataFrame(e,columns=[f'emb_{i}' for i in range(768)])],axis=1);tr=d[d.split.eq('train')];v=d[d.split.eq('validation')];pred=pd.read_csv(folder/'validation_predictions.csv')
 for kind,m in r['stocks'][s]['models'].items():
  fit=joblib.load(folder/f'{kind}.joblib');np.testing.assert_allclose(fit.predict_proba(v)[:,1],pred[kind],atol=1e-8)
  if kind=='price_embedding':
   pca=fit.named_steps['features'].named_transformers_['embedding'].named_steps['pca'];np.testing.assert_allclose(pca.mean_,e[d.split.eq('train')].mean(axis=0),atol=1e-7)
  if kind.startswith('paper'):
   lr=fit.named_steps['model'];assert lr.penalty=='l1';print(s,kind,'nonzero weights',int(np.count_nonzero(lr.coef_)),'of',lr.coef_.size)
  if kind!='paper_bow':
   feature=fit.named_steps['features'];cols=feature.transformers_[0][2];np.testing.assert_allclose(feature.named_transformers_['numeric'].mean_,tr[cols].mean().to_numpy(),atol=1e-7)
 grid=pd.read_csv(folder/'training_cv_grid.csv');assert set(grid.fold_month)=={6,7,8}
 for kind in r['stocks'][s]['models']:
  g=grid[grid.model.eq(kind)].groupby('C')[['balanced_accuracy','brier']].mean().reset_index();best=g.sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];assert best.C==r['stocks'][s]['models'][kind]['C']
 # Descriptive intervals for predeclared paper and recent-price contrasts.
 dates=pred.start_utc.str[:10];days=dates.unique();rng=np.random.default_rng(573);delta={'paper_price_bow':[],'recent_price':[],'paper_vs_l1':[]}
 for _ in range(1000):
  ix=np.concatenate([np.flatnonzero(dates.eq(day)) for day in rng.choice(days,len(days),replace=True)]);y=pred.label.to_numpy()[ix];ref=scores(y,pred.price.to_numpy()[ix])['balanced_accuracy']
  for k in delta:
   if k=='paper_vs_l1':delta[k].append(scores(y,pred.paper_price_bow.to_numpy()[ix])['balanced_accuracy']-scores(y,pred.price_l1_control.to_numpy()[ix])['balanced_accuracy'])
   else:delta[k].append(scores(y,pred[k].to_numpy()[ix])['balanced_accuracy']-ref)
 ci[s]={k:np.quantile(z,[.025,.975]).tolist() for k,z in delta.items()}
(B/'results/descriptive_intervals.json').write_text(json.dumps(ci,indent=2))
print('PASS: unchanged labels/rows; no test evaluation; recent price ends <= cutoff; freshness is subset; saved predictions reproduce; PCA/scaling train-only; C chosen solely from June-Aug training folds.')
