"""Matched grouped-random and chronological baselines, with sealed scores."""
import json,time,warnings
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from common import *

def schedules(d):
 groups=pd.read_csv(PRIVATE/'groups.csv');assert groups.row_id.tolist()==d.row_id.tolist();gg=groups.group.to_numpy();blocks=[]
 n=json.loads((OUT/'INPUT_AUDIT.json').read_text())['grouped_max_feasible_folds']
 if n:
  for seed in SEEDS:
   outer=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
   for fold,(tr,ev) in enumerate(outer.split(np.arange(len(d)),d.symbol+'_'+d.label.astype(str),gg)):
    ins=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed+fold+1000).split(tr,(d.symbol+'_'+d.label.astype(str)).iloc[tr],gg[tr]))
    for stock in ['AAPL','AMZN']:
     a=tr[(d.iloc[tr].symbol==stock).to_numpy()];b=ev[(d.iloc[ev].symbol==stock).to_numpy()]
     pairs=[]
     for i,j in ins:
      aa=tr[i];bb=tr[j];aa=aa[(d.iloc[aa].symbol==stock).to_numpy()];bb=bb[(d.iloc[bb].symbol==stock).to_numpy()]
      assert not set(gg[aa])&set(gg[bb]);assert set(d.iloc[aa].label)==set(d.iloc[bb].label)=={0,1};pairs.append((aa.tolist(),bb.tolist()))
     assert set(d.iloc[a].label)==set(d.iloc[b].label)=={0,1};assert not set(gg[a])&set(gg[b])
     blocks.append(dict(name=f'grouped_{seed}_{stock}_{fold}',protocol='grouped',seed=seed,stock=stock,fold=fold,train=a.tolist(),test=b.tolist(),inner=pairs))
 for stock in ['AAPL','AMZN']:
  for month in [f'2018-{m:02d}' for m in range(3,9)]+['final']:
   cut='2018-09' if month=='final' else month
   b=np.flatnonzero((d.symbol==stock)&((d.month>=cut) if month=='final' else (d.month==month)))
   a=np.flatnonzero((d.symbol==stock)&(d.month<cut)&(d.end_utc<d.iloc[b].cutoff_utc.min()))
   pairs=[]
   for mm in sorted(d.iloc[a].month.unique())[1:]:
    bb=a[(d.iloc[a].month==mm).to_numpy()];aa=a[((d.iloc[a].month<mm)&(d.iloc[a].end_utc<d.iloc[bb].cutoff_utc.min())).to_numpy()]
    if set(d.iloc[aa].label)==set(d.iloc[bb].label)=={0,1}:pairs.append((aa.tolist(),bb.tolist()))
   assert pairs
   blocks.append(dict(name=f'chronological_573_{stock}_{month}',protocol='chronological',seed=573,stock=stock,fold=month,train=a.tolist(),test=b.tolist(),inner=pairs))
 return blocks

def main():
 check_sources();d,emb,ids,means=load();blocks=schedules(d);seal=dict(sources=hashes(),common=sha(HERE/'common.py'),runner=sha(__file__),blocks=blocks)
 dest=PRIVATE/'diagnostic_baseline';dest.mkdir(exist_ok=True)
 if (dest/'seal.json').exists():assert json.loads((dest/'seal.json').read_text())==seal
 else:dump(dest/'seal.json',seal)
 begun=time.monotonic();total=0
 with threadpool_limits(2),warnings.catch_warnings():
  warnings.filterwarnings('ignore',category=FutureWarning);warnings.filterwarnings('ignore',message='Inconsistent values: penalty=')
  for block in blocks:
   run=dest/block['name']
   if (run/'complete.json').exists():
    s=json.loads((run/'complete.json').read_text());assert s['seal']==sha(dest/'seal.json')
    for n,h in s['artifacts'].items():assert sha(run/n)==h
    total+=s['fits'];continue
   if run.exists():raise RuntimeError('incomplete diagnostic block retained: '+str(run))
   run.mkdir();tr=np.array(block['train']);ev=np.array(block['test']);cv=[];ledger=[];choices={};inner_price={};outer_price=None
   pred=d.iloc[ev][['row_id','symbol','day','month','label','has_news']].copy()
   def fit(method,c,tf,a,b,name):
    tick=time.monotonic();m=classifier(method,c).fit(tf.transform(d,a,means),d.iloc[a].label);assert m.n_iter_.max()<3000
    p=m.predict_proba(tf.transform(d,b,means))[:,1]
    path=run/(name+'.joblib');joblib.dump(dict(model=m,transform=tf,train_indices=a.tolist(),evaluation_indices=b.tolist()),path,compress=3)
    q=joblib.load(path);err=float(np.max(np.abs(p-q['model'].predict_proba(q['transform'].transform(d,b,means))[:,1])));assert err<1e-12
    ledger.append(dict(name=path.name,method=method,C=c,sha256=sha(path),seconds=time.monotonic()-tick,reload_error=err))
    return p
   for method in ['PRICE','PAPER','FULL','FINBERT','MODERN']:
    ps={}
    for inf,(aa,bb) in enumerate(block['inner']):
     a=np.array(aa);b=np.array(bb);tf=Features(method).fit(d,a,emb,ids,means)
     for c in CS:
      p=fit(method,c,tf,a,b,f'{method}_inner{inf}_{c}')
      if method!='PRICE':p=apply_fallback(method,p,d.iloc[b].has_news.to_numpy()==0,inner_price[inf])
      ps[inf,c]=p;cv.append(dict(method=method,inner_fold=inf,C=c,**metric(d.iloc[b].label,p)))
    rows=[r for r in cv if r['method']==method];chosen=min(CS,key=lambda c:(-np.mean([r['BA'] for r in rows if r['C']==c]),np.mean([r['Brier'] for r in rows if r['C']==c]),c));choices[method]=chosen
    if method=='PRICE':inner_price={i:ps[i,chosen] for i in range(len(block['inner']))}
    tf=Features(method).fit(d,tr,emb,ids,means);p=fit(method,chosen,tf,tr,ev,f'{method}_selected')
    if method=='PRICE':outer_price=p.copy()
    else:p=apply_fallback(method,p,d.iloc[ev].has_news.to_numpy()==0,outer_price)
    pred[method]=p
   pred.to_csv(run/'predictions_SEALED.csv',index=False,float_format='%.17g');pd.DataFrame(cv).to_csv(run/'inner_cv.csv',index=False);dump(run/'training.json',dict(choices=choices,fits=ledger))
   dump(run/'complete.json',dict(seal=sha(dest/'seal.json'),fits=len(ledger),artifacts={p.name:sha(p) for p in run.iterdir() if p.is_file()}));total+=len(ledger)
   print(block['name'],'completed',len(ledger),'fits; outer scores sealed; elapsed',round(time.monotonic()-begun,1),flush=True)
 dump(OUT/'DIAGNOSTIC_EXECUTION.json',dict(status='COMPLETE_SEALED',blocks=len(blocks),fits=total,grouped_blocks=sum(b['protocol']=='grouped' for b in blocks),chronological_blocks=sum(b['protocol']=='chronological' for b in blocks),outer_metrics_released=False))
if __name__=='__main__':main()
