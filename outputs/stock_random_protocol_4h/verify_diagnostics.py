"""Independent replay and time/group contracts for diagnostic baselines."""
import json
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from common import PRIVATE,OUT,HERE,CS,sha,hashes,load,dump

def main():
 d,emb,ids,means=load();dest=PRIVATE/'diagnostic_baseline';seal=json.loads((dest/'seal.json').read_text());groups=pd.read_csv(PRIVATE/'groups.csv').group.to_numpy()
 checks=dict(source_hashes=seal['sources']==hashes(),runner_hash=seal['runner']==sha(HERE/'diagnostic_train.py'),all_artifact_hashes=True,partitions=True,canonical_labels=True,group_isolation=True,chronological_label_maturity=True,selected_parameters=True,selected_model_membership=True,probability_replay=True,exact_fallback=True)
 nfit=0;nmodels=0;nrows=0;maxerr=0.
 with threadpool_limits(2):
  for block in seal['blocks']:
   run=dest/block['name'];complete=json.loads((run/'complete.json').read_text());t=json.loads((run/'training.json').read_text());p=pd.read_csv(run/'predictions_SEALED.csv',float_precision='round_trip');cv=pd.read_csv(run/'inner_cv.csv')
   checks['all_artifact_hashes'] &= complete['seal']==sha(dest/'seal.json') and all(sha(run/n)==h for n,h in complete['artifacts'].items())
   tr=np.array(block['train']);ev=np.array(block['test']);checks['partitions'] &= not bool(set(tr)&set(ev)) and p.row_id.tolist()==d.iloc[ev].row_id.tolist()
   checks['canonical_labels'] &= np.array_equal(p.label,d.iloc[ev].label)
   for a,b in [(tr,ev)]+[(np.array(a),np.array(b)) for a,b in block['inner']]:
    checks['partitions'] &= not bool(set(a)&set(b)) and len(a)>0 and len(b)>0
    if block['protocol']=='grouped':checks['group_isolation'] &= not bool(set(groups[a])&set(groups[b]))
    else:checks['chronological_label_maturity'] &= d.iloc[a].end_utc.max()<d.iloc[b].cutoff_utc.min()
   for method in ['PRICE','PAPER','FULL','FINBERT','MODERN']:
    rows=cv[cv.method==method];chosen=min(CS,key=lambda c:(-rows[rows.C==c].BA.mean(),rows[rows.C==c].Brier.mean(),c));checks['selected_parameters'] &= chosen==t['choices'][method]
    bundle=joblib.load(run/f'{method}_selected.joblib');tf=bundle['transform'];model=bundle['model'];checks['selected_model_membership'] &= bundle['train_indices']==tr.tolist() and tf.fit_ids==d.iloc[tr].row_id.tolist()
    raw=expit(np.asarray(tf.transform(d,ev,means)@model.coef_.ravel()).ravel()+model.intercept_[0])
    if method in ('FULL','FINBERT','MODERN'):
     none=d.iloc[ev].has_news.to_numpy()==0;raw[none]=p.PRICE.to_numpy()[none];checks['exact_fallback'] &= np.array_equal(p[method].to_numpy()[none],p.PRICE.to_numpy()[none])
    err=float(np.max(np.abs(raw-p[method].to_numpy())));maxerr=max(maxerr,err);checks['probability_replay'] &= err<=1e-12;nmodels+=1;nrows+=len(ev)
   nfit+=len(t['fits'])
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},blocks=len(seal['blocks']),fits_recorded=nfit,selected_models_replayed=nmodels,probabilities_replayed=nrows,max_error=maxerr,verifier_fits=0,outer_metrics_released=False)
 dump(OUT/'DIAGNOSTIC_VERIFICATION.json',result);print(json.dumps(result,indent=2))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
