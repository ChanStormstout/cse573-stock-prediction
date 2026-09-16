"""Independent saved-output and temporal audit, plus paired-day diagnostics."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd,torch,joblib
from models import TemporalNet,fit_scaler,transform,predict,train_net
B=Path(__file__).resolve().parent;OUT=B/'results';R=json.loads((OUT/'results.json').read_text());P=R['protocol'];torch.set_num_threads(2);torch.use_deterministic_algorithms(True);info=json.loads((OUT/'data_summary.json').read_text());sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
raw=pd.read_pickle(B.parents[1]/'work/stock-data/audit/news_index.pkl');raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key');preds={};model_prob={}
for name,h in R['script_hashes'].items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==h
for s in ['AAPL','AMZN']:
 F=OUT/s;z=np.load(F/'data.npz');d=pd.read_csv(F/'manifest.csv').fillna({'news_record_keys':''});orig=pd.read_csv(B.parent/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''});assert hashlib.sha256((B.parent/f'stock_finbert/results/{s}/development_features.csv').read_bytes()).hexdigest()==info['source_hashes'][s]
 assert orig.start_utc.tolist()==d.start_utc.tolist();np.testing.assert_array_equal(z['y'],orig.label);assert set(d.split)=={'train','validation'};np.testing.assert_allclose(z['current'],orig[info['stocks'][s]['base_features']])
 tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));cut=pd.to_datetime(d.cutoff_utc,utc=True);end=pd.to_datetime(d.end_utc,utc=True);X=z['X'];length=z['lengths'];ix=z['source_indices'];assert X.shape==(len(d),6,24)
 for i,n in enumerate(length):
  expected=np.append(np.flatnonzero((np.arange(len(d))<i)&(end.le(cut.iloc[i]).to_numpy()))[-5:],i);np.testing.assert_array_equal(ix[i,:n],expected)
  np.testing.assert_allclose(X[i,:n,:22],z['current'][expected]);np.testing.assert_allclose(X[i,:n,-2],np.log1p((cut.iloc[i]-cut.iloc[expected]).dt.total_seconds()/3600));assert np.all(X[i,n:]==0);assert np.all(X[i,:n,-1]==1)
  keys=[k for k in d.iloc[i].news_record_keys.split('|') if k];times=raw.loc[keys,'available_utc'];assert times.le(cut.iloc[i]).all() and times.gt(cut.iloc[i]-pd.Timedelta(hours=4)).all()
 scaler=fit_scaler(X[tr]);aa=transform(X[tr],scaler);bb=transform(X[va],scaler);pred=pd.read_csv(F/'validation_predictions.csv');preds[s]=pred;model_prob[s]={}
 np.testing.assert_array_equal(pred.label,z['y'][va])
 for kind in ['current_lr','sequence_lr']:
  bundle=joblib.load(F/f'{kind}.joblib');sc=bundle['scaler'];true_mean=z['current'][tr].mean(axis=0) if kind=='current_lr' else X[tr,:,:-1][X[tr,:,-1]>0].mean(axis=0);np.testing.assert_allclose(sc.mean_,true_mean)
  data=sc.transform(z['current'][va]) if kind=='current_lr' else transform(X[va],sc).reshape(len(va),-1);pp=bundle['model'].predict_proba(data)[:,1];np.testing.assert_allclose(pp,pred[kind]);model_prob[s][kind]=pp[None,:]
  m=scores(z['y'][va],pp)
  for key,value in m.items():assert abs(value-R['stocks'][s]['models'][kind]['validation'][key])<1e-10
  grid=pd.read_csv(F/f'{kind}_training_grid.csv');best=grid.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];assert best.C==R['stocks'][s]['models'][kind]['C']
 old=pd.read_csv(B.parent/f'stock_events/results/{s}/price_finbert_validation_predictions.csv');np.testing.assert_allclose(pred.current_lr,old.p_up,atol=1e-10)
 for kind in ['mlp','gru']:
  saved_scaler=joblib.load(F/f'{kind}_scaler.joblib');np.testing.assert_allclose(saved_scaler.mean_,scaler.mean_);assert saved_scaler.n_samples_seen_==int(X[tr,:,-1].sum());m=R['stocks'][s]['models'][kind];ps=[]
  grid=pd.read_csv(F/f'{kind}_training_grid.csv');assert len(grid)==27;assert set(grid.seed)==set(P['neural']['seeds']);best=grid.groupby('weight_decay')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','weight_decay'],ascending=[False,True,False]).iloc[0];assert best.weight_decay==m['weight_decay']
  for seed in P['neural']['seeds']:
   net=TemporalNet(kind);net.load_state_dict(torch.load(F/f'{kind}_{seed}.pt',weights_only=True,map_location='cpu'));pp=predict(net,bb,length[va]);np.testing.assert_allclose(pp,pred[f'{kind}_{seed}'],atol=1e-7);ps.append(pp);assert sum(v.numel() for v in net.parameters())==m['parameters']
   expected=scores(z['y'][va],pp)
   for key,value in expected.items():assert abs(value-m['seeds'][str(seed)]['validation'][key])<1e-7
   if kind=='gru':
    # Packed GRU must ignore artificial values outside valid lengths.
    tiny=transform(X[:6],scaler);changed=tiny.copy()
    for i,n in enumerate(length[:6]):changed[i,n:]=999
    np.testing.assert_allclose(predict(net,tiny,length[:6]),predict(net,changed,length[:6]),atol=1e-7)
    reversed_x=bb.copy()
    for i,n in enumerate(length[va]):reversed_x[i,:n]=reversed_x[i,:n][::-1].copy()
    np.testing.assert_allclose(predict(net,reversed_x,length[va]),pred[f'gru_reversed_{seed}'],atol=1e-7)
  model_prob[s][kind]=np.stack(ps)
  for metric in ['balanced_accuracy','brier']:
   vals=[scores(z['y'][va],p)[metric] for p in ps];assert abs(np.mean(vals)-m['seed_mean'][metric])<1e-8;assert abs(np.std(vals,ddof=1)-m['seed_std'][metric])<1e-8
 # One representative deterministic re-fit checks recorded seed/runtime behavior.
 if s=='AAPL':
  net,_=train_net('gru',aa,z['y'][tr],length[tr],573,R['stocks'][s]['models']['gru']['weight_decay'],P['neural']);np.testing.assert_allclose(predict(net,bb,length[va]),pred.gru_573,atol=1e-7)

def mean_ba(y,ps):return np.mean([.5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean()) for p in ps])
contrasts=[('sequence_lr','current_lr'),('gru','sequence_lr'),('gru','mlp')];days=sorted(set().union(*[set(d.start_utc.str[:10]) for d in preds.values()]));groups={s:{day:np.flatnonzero(preds[s].start_utc.str[:10].eq(day)) for day in days} for s in preds};intervals={}
for new,base in contrasts:
 rng=np.random.default_rng(578);deltas={s:[] for s in preds}
 for _ in range(500):
  draw=rng.choice(days,len(days),replace=True)
  for s,d in preds.items():
   ids=np.concatenate([groups[s][day] for day in draw]);y=d.label.to_numpy()[ids];deltas[s].append(mean_ba(y,model_prob[s][new][:,ids])-mean_ba(y,model_prob[s][base][:,ids]))
 intervals[new+' minus '+base]={s:np.quantile(v,[.025,.975]).tolist() for s,v in deltas.items()};intervals[new+' minus '+base]['equal_stock_mean']=np.quantile(np.mean(list(deltas.values()),axis=0),[.025,.975]).tolist()
(OUT/'descriptive_intervals.json').write_text(json.dumps(intervals,indent=2))
print('PASS: source/code hashes; all original rows/outcomes; exact historical snapshot eligibility and news boundaries; shared feature tensors; training-only scaling/C/weight-decay selection; every checkpoint reproduces saved predictions/metrics; original current-LR control reproduces; GRU ignores padding; reverse diagnostic reproduces; representative deterministic GRU refit reproduces. No final test evaluated.')
