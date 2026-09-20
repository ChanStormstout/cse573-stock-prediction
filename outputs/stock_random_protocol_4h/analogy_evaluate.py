"""Finite shared analogy corrections. Chooses alpha on inner folds only."""
import json
import numpy as np,pandas as pd
from scipy.special import expit,logit
from common import *

ALPHAS=[0.,.1,.25,.5]
def correction(base,signal,gate,alpha):
 out=np.asarray(base,float).copy();gate=np.asarray(gate,bool)
 if alpha==0:return out
 out[gate]=expit(logit(np.clip(out[gate],1e-9,1-1e-9))+alpha*np.tanh(logit(np.clip(np.asarray(signal)[gate],1e-6,1-1e-6))))
 return out

def main():
 check_sources();src=PRIVATE/'analogy'
 complete=json.loads((src/'inference_complete.json').read_text());assert complete['output_sha256']==sha(src/'generated.jsonl')
 assert json.loads((OUT/'ANALOGY_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((OUT/'LLM_VERIFICATION.json').read_text())['status']=='PASS'
 target=PRIVATE/'analogy_downstream';target.mkdir(exist_ok=False)
 d,_,_,_=load();results={r['prompt_hash']:r for r in map(json.loads,(src/'generated.jsonl').read_text().splitlines())};jobs=json.loads((src/'jobs.json').read_text());by_scope={}
 for r in jobs:by_scope.setdefault(r['scope'],[]).append(r)
 decisions=[];predictions=[];fallback_mismatches=0
 def signal(scope,variant,indices):
  rows={j['index']:j for j in by_scope[scope]};p=[];gate=[]
  for i in indices:
   r=rows[int(i)]
   if variant=='VOTE':value=r['vote']
   else:value=results[r['prompts'][variant]]['p'] if variant in r['prompts'] else None
   gate.append(value is not None);p.append(value if value is not None else .5)
  return np.array(p),np.array(gate)
 for seed in SEEDS:
  for fold in range(10):
   for base in ['FINBERT','MODERN']:
    for variant in ['CURRENT','VOTE','ANALOGY']:
     candidate=[]
     for alpha in ALPHAS:
      stock_metrics=[]
      for stock in ['AAPL','AMZN']:
       run=PRIVATE/'baseline'/f'{seed}_{stock}_{fold}';oof=pd.read_csv(run/f'{base}_selection_oof_NOT_META_TRAIN.csv',float_precision='round_trip').set_index('index')
       y=[];old=[];new=[]
       for inf in range(3):
        scope=f'{seed}_{stock}_{fold}_inner{inf}';ii=np.array([j['index'] for j in by_scope[scope]])
        s,g=signal(scope,variant,ii);p=oof.loc[ii].p.to_numpy();q=correction(p,s,g,alpha)
        y.extend(d.iloc[ii].label);old.extend(p);new.extend(q)
       m=metric(y,new);b=metric(y,old);stock_metrics.append(dict(symbol=stock,BA=m['BA'],Brier=m['Brier'],base_BA=b['BA'],base_Brier=b['Brier']))
      candidate.append(dict(alpha=alpha,stocks=stock_metrics,eligible=all(s['Brier']<=s['base_Brier']+.002 for s in stock_metrics)))
     chosen=min([c for c in candidate if c['eligible']],key=lambda c:(-np.mean([s['BA'] for s in c['stocks']]),c['alpha']))['alpha']
     decisions.append(dict(seed=seed,fold=fold,base=base,variant=variant,alpha=chosen,candidates=candidate,selection='shared across stocks; finite inner-CV hyperparameter selection, no correction weights fitted'))
     for stock in ['AAPL','AMZN']:
      run=PRIVATE/'baseline'/f'{seed}_{stock}_{fold}';outer=pd.read_csv(run/'outer_predictions_SEALED.csv',float_precision='round_trip');lookup=dict(zip(d.row_id,d.index));ii=np.array([lookup[k] for k in outer.row_id])
      s,g=signal(f'{seed}_{stock}_{fold}_outer',variant,ii);p=outer[base].to_numpy();q=correction(p,s,g,chosen)
      fallback_mismatches+=int(np.count_nonzero(q[~g]!=p[~g]));assert np.array_equal(q[~g],p[~g])
      predictions.extend(dict(seed=seed,fold=fold,row_id=d.iloc[i].row_id,symbol=stock,method=f'{base}_{variant}',p=float(prob),base_probability=float(b),gate=bool(flag),alpha=chosen) for i,prob,b,flag in zip(ii,q,p,g))
 pd.DataFrame(predictions).to_csv(target/'predictions_SEALED.csv',index=False,float_format='%.17g');dump(target/'selection.json',decisions)
 dump(OUT/'ANALOGY_DOWNSTREAM_EXECUTION.json',dict(status='COMPLETE_SEALED',predictions=len(predictions),shared_alpha_decisions=len(decisions),new_gradient_training=False,fallback_mismatches=fallback_mismatches,outer_metrics_released=False))
if __name__=='__main__':main()
