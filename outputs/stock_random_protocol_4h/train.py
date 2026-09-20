"""Nested random-CV baseline fitting; outer scores stay sealed until release."""
import argparse,time,json,sys,warnings
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from common import *

def select(records):
 return min(CS,key=lambda c:(-np.mean([r['BA'] for r in records if r['C']==c]),np.mean([r['Brier'] for r in records if r['C']==c]),c))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=SEEDS);args=ap.parse_args()
 check_sources();d,emb,ids,means=load();outer=pd.read_csv(PRIVATE/'outer_folds.csv');inner=json.loads((PRIVATE/'inner_folds.json').read_text())
 code={p.name:sha(p) for p in [HERE/'common.py',HERE/'train.py',HERE/'prepare.py']}
 stamp=dict(code=code,sources=hashes());seal=PRIVATE/'training_seal.json'
 if seal.exists():assert json.loads(seal.read_text())==stamp,'training code/source changed'
 else:dump(seal,stamp)
 started=time.monotonic();done=0
 with threadpool_limits(2),warnings.catch_warnings():
  warnings.filterwarnings('ignore',category=FutureWarning)
  for seed in ([args.seed] if args.seed else SEEDS):
   for stock in ['AAPL','AMZN']:
    for fold in range(10):
     run=PRIVATE/'baseline'/f'{seed}_{stock}_{fold}'
     if (run/'complete.json').exists():
      old=json.loads((run/'complete.json').read_text());assert old['seal']==sha(seal)
      for n,h in old['artifacts'].items():assert sha(run/n)==h
      done+=1;continue
     if run.exists():raise RuntimeError('Incomplete fit preserved: '+str(run))
     run.mkdir(parents=True)
     ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy(int)
     tr=np.array(sorted(set(np.flatnonzero(d.symbol==stock))-set(ev)))
     splits=[s for s in inner if s['seed']==seed and s['symbol']==stock and s['fold']==fold]
     ledger=[];cv=[];choices={};inner_price={};outer_price=None;prediction=d.iloc[ev][['row_id','symbol','day','month','label','has_news']].copy()
     def fit_record(method,c,tf,a,b,name):
      assert not(set(a)&set(b));assert not(set(a)&set(ev))
      x=tf.transform(d,a,means);xx=tf.transform(d,b,means);tick=time.monotonic()
      model=classifier(method,c).fit(x,d.iloc[a].label)
      if model.n_iter_.max()>=3000:raise RuntimeError('solver did not converge')
      p=model.predict_proba(xx)[:,1];seconds=time.monotonic()-tick
      path=run/(name+'.joblib');joblib.dump(dict(transform=tf,model=model,train_indices=list(map(int,a)),evaluation_indices=list(map(int,b))),path,compress=3)
      bundle=joblib.load(path);replay=bundle['model'].predict_proba(bundle['transform'].transform(d,b,means))[:,1]
      error=float(np.max(np.abs(replay-p)));assert error<=1e-12
      ledger.append(dict(name=path.name,method=method,C=c,train_n=len(a),eval_n=len(b),train_indices=list(map(int,a)),evaluation_indices=list(map(int,b)),seconds=seconds,reload_error=error,sha256=sha(path),iterations=int(model.n_iter_.max())))
      return p
     for method in ['PRICE','PAPER','FULL','FINBERT','MODERN']:
      values={}
      for s in splits:
       a=np.array(s['train']);b=np.array(s['validation']);inf=s['inner_fold']
       tf=Features(method).fit(d,a,emb,ids,means)
       for c in CS:
        p=fit_record(method,c,tf,a,b,f'{method}_inner{inf}_{c}')
        if method!='PRICE':p=apply_fallback(method,p,d.iloc[b].has_news.to_numpy()==0,inner_price[inf])
        values[inf,c]=p;cv.append(dict(method=method,inner_fold=inf,C=c,**metric(d.iloc[b].label,p)))
      chosen=select([r for r in cv if r['method']==method]);choices[method]=chosen
      if method=='PRICE':inner_price={s['inner_fold']:values[s['inner_fold'],chosen] for s in splits}
      # Keep these selected OOF scores for audit only, NOT honest correction training.
      oof=[]
      for s in splits:
       oof.extend(dict(index=i,p=float(p)) for i,p in zip(s['validation'],values[s['inner_fold'],chosen]))
      pd.DataFrame(oof).to_csv(run/f'{method}_selection_oof_NOT_META_TRAIN.csv',index=False)
      tf=Features(method).fit(d,tr,emb,ids,means);p=fit_record(method,chosen,tf,tr,ev,f'{method}_selected')
      if method=='PRICE':outer_price=p.copy()
      else:p=apply_fallback(method,p,d.iloc[ev].has_news.to_numpy()==0,outer_price)
      prediction[method]=p
     prediction.to_csv(run/'outer_predictions_SEALED.csv',index=False,float_format='%.17g')
     pd.DataFrame(cv).to_csv(run/'inner_selection.csv',index=False)
     dump(run/'training.json',dict(fits=ledger,selected_C=choices,outer_train=tr.tolist(),outer_test=ev.tolist(),outer_test_metrics_inspected=False))
     dump(run/'complete.json',dict(seal=sha(seal),fits=len(ledger),artifacts={p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file()}))
     done+=1;print(f'completed {seed} {stock} fold {fold}: {len(ledger)} actual fits; outer scores sealed; total blocks {done}; elapsed {time.monotonic()-started:.1f}s',flush=True)
 summary=dict(status='BASELINES_COMPLETE_SEALED' if done==60 else 'BASELINE_SUBSET_COMPLETE_SEALED',completed_blocks=done,actual_estimator_fits=sum(json.loads(p.read_text())['fits'] for p in (PRIVATE/'baseline').glob('*/complete.json')),outer_metrics_released=False,code=code)
 dump(OUT/'BASELINE_EXECUTION.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
