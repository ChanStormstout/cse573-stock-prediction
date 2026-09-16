import os,json,sys,time,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from models import fit_scaler,transform,train_net,predict,TemporalNet
B=Path(__file__).resolve().parent;OUT=B/'results';sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
P=json.loads((B/'protocol.json').read_text());N=P['neural'];torch.set_num_threads(N['threads']);torch.use_deterministic_algorithms(True)
if (OUT/'results.json').exists():raise RuntimeError('Existing experiment results: preserve and use another version before rerunning.')
result={'protocol':P,'versions':{'python':platform.python_version(),'torch':torch.__version__},'stocks':{}};start=time.time()
for s in ['AAPL','AMZN']:
 F=OUT/s;d=pd.read_csv(F/'manifest.csv');data=np.load(F/'data.npz');X=data['X'];cur=data['current'];y=data['y'];lengths=data['lengths'];tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));dates=pd.to_datetime(d.start_utc,utc=True);folds=[]
 for month in [6,7,8]:
  a=np.flatnonzero((dates<pd.Timestamp(f'2018-{month:02d}-01',tz='UTC'))&d.split.eq('train'));b=np.flatnonzero((dates>=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC'))&(dates<pd.Timestamp(f'2018-{month+1:02d}-01',tz='UTC'))&d.split.eq('train'))
  assert pd.to_datetime(d.end_utc.iloc[a],utc=True).max()<pd.to_datetime(d.cutoff_utc.iloc[b],utc=True).min();folds.append((month,a,b))
 result['stocks'][s]={'models':{}};pred=d.iloc[va][['start_utc','label']].reset_index(drop=True)
 def metric(p):return {'validation':scores(y[va],p),'monthly':{m:scores(y[va][mask],p[mask]) for m in ['2018-09','2018-10'] if (mask:=d.start_utc.iloc[va].str.startswith(m).to_numpy()).any()}}
 for kind in ['current_lr','sequence_lr']:
  grid=[]
  for C in [.01,.1,1.]:
   for month,a,b in folds:
    scaler=StandardScaler().fit(cur[a]) if kind=='current_lr' else fit_scaler(X[a]);aa=scaler.transform(cur[a]) if kind=='current_lr' else transform(X[a],scaler).reshape(len(a),-1);bb=scaler.transform(cur[b]) if kind=='current_lr' else transform(X[b],scaler).reshape(len(b),-1)
    model=LogisticRegression(C=C,l1_ratio=0,solver='liblinear',max_iter=3000,random_state=573).fit(aa,y[a]);grid.append(dict(C=C,month=month,n_train=len(a),n_valid=len(b),**scores(y[b],model.predict_proba(bb)[:,1])))
  g=pd.DataFrame(grid);g.to_csv(F/f'{kind}_training_grid.csv',index=False);best=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];C=float(best.C)
  scaler=StandardScaler().fit(cur[tr]) if kind=='current_lr' else fit_scaler(X[tr]);aa=scaler.transform(cur[tr]) if kind=='current_lr' else transform(X[tr],scaler).reshape(len(tr),-1);bb=scaler.transform(cur[va]) if kind=='current_lr' else transform(X[va],scaler).reshape(len(va),-1);model=LogisticRegression(C=C,l1_ratio=0,solver='liblinear',max_iter=3000,random_state=573).fit(aa,y[tr]);p=model.predict_proba(bb)[:,1];pred[kind]=p
  joblib.dump({'scaler':scaler,'model':model},F/f'{kind}.joblib');result['stocks'][s]['models'][kind]={'C':C,'training_cv_BA':float(best.balanced_accuracy),'training':scores(y[tr],model.predict_proba(aa)[:,1]),**metric(p)}
  print(s,kind,'CV',round(best.balanced_accuracy,4),'val',round(scores(y[va],p)['balanced_accuracy'],4),flush=True)
 for kind in ['mlp','gru']:
  grid=[]
  for wd in [.0001,.001,.01]:
   for month,a,b in folds:
    scaler=fit_scaler(X[a]);aa=transform(X[a],scaler);bb=transform(X[b],scaler)
    for seed in N['seeds']:
     net,trace=train_net(kind,aa,y[a],lengths[a],seed,wd,N);p=predict(net,bb,lengths[b]);grid.append(dict(weight_decay=wd,month=month,seed=seed,n_train=len(a),n_valid=len(b),training_bce=trace[-1]['training_bce'],**scores(y[b],p)))
    print(s,kind,'wd',wd,'fold',month,'three seeds done','elapsed',round(time.time()-start),flush=True)
  g=pd.DataFrame(grid);g.to_csv(F/f'{kind}_training_grid.csv',index=False);best=g.groupby('weight_decay')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','weight_decay'],ascending=[False,True,False]).iloc[0];wd=float(best.weight_decay)
  scaler=fit_scaler(X[tr]);joblib.dump(scaler,F/f'{kind}_scaler.joblib');aa=transform(X[tr],scaler);bb=transform(X[va],scaler);runs={}
  for seed in N['seeds']:
   net,trace=train_net(kind,aa,y[tr],lengths[tr],seed,wd,N);p=predict(net,bb,lengths[va]);pred[f'{kind}_{seed}']=p;torch.save(net.state_dict(),F/f'{kind}_{seed}.pt')
   runs[str(seed)]={'training':scores(y[tr],predict(net,aa,lengths[tr])),'trace':trace,**metric(p)}
   if kind=='gru':
    rev=bb.copy()
    for i,n in enumerate(lengths[va]):rev[i,:n]=rev[i,:n][::-1].copy()
    rp=predict(net,rev,lengths[va]);pred[f'gru_reversed_{seed}']=rp;runs[str(seed)]['reverse_order_diagnostic']={'validation':scores(y[va],rp),'mean_absolute_probability_change':float(np.mean(np.abs(rp-p)))}
  names=list(runs[str(N['seeds'][0])]['validation']);means={m:float(np.mean([r['validation'][m] for r in runs.values()])) for m in names};stds={m:float(np.std([r['validation'][m] for r in runs.values()],ddof=1)) for m in names}
  result['stocks'][s]['models'][kind]={'weight_decay':wd,'parameters':sum(p.numel() for p in net.parameters()),'training_cv_BA':float(best.balanced_accuracy),'seed_mean':means,'seed_std':stds,'seeds':runs}
  print(s,kind,'FINAL mean BA',round(means['balanced_accuracy'],4),'sd',round(stds['balanced_accuracy'],4),'params',sum(p.numel() for p in net.parameters()),flush=True)
 pred.to_csv(F/'validation_predictions.csv',index=False);(OUT/'results.json').write_text(json.dumps(result,indent=2))
result['elapsed_seconds']=time.time()-start;result['script_hashes']={f:hashlib.sha256((B/f).read_bytes()).hexdigest() for f in ['prepare.py','models.py','run.py','protocol.json']};(OUT/'results.json').write_text(json.dumps(result,indent=2));print('E08 completed in',round(time.time()-start),'seconds. Final test not evaluated.',flush=True)
