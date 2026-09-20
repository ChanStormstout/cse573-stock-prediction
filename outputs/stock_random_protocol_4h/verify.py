"""Read-only independent selected-weight replay. Never calls fit or scores BA."""
import json
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from common import ROOT,HERE,OUT,PRIVATE,SOURCES,SEEDS,CS,OLD,RECENT,sha,hashes,load,dump

def main():
 d,emb,ids,means=load();manifest=json.loads((PRIVATE/'manifest.json').read_text());checks={}
 checks['source_hashes']=manifest['sources']==hashes()
 checks['prepared_artifacts']=all(sha(PRIVATE/n)==h for n,h in manifest['artifacts'].items())
 folds=pd.read_csv(PRIVATE/'outer_folds.csv');inners=json.loads((PRIVATE/'inner_folds.json').read_text())
 checks['exhaustive_unique_splits']=all(len(g)==len(d) and g['index'].nunique()==len(d) for _,g in folds.groupby('seed')) and set(folds.seed)==set(SEEDS)
 checks['inner_disjoint_and_exhaustive']=True
 for (seed,stock,fold),outer in folds.groupby(['seed','symbol','fold']):
  ev=set(outer['index']);tr=set(np.flatnonzero(d.symbol==stock))-ev
  ss=[s for s in inners if (s['seed'],s['symbol'],s['fold'])==(seed,stock,fold)]
  val=[]
  for s in ss:
   a,b=set(s['train']),set(s['validation']);checks['inner_disjoint_and_exhaustive'] &= not bool(a&b) and (a|b)==tr and not bool((a|b)&ev);val+=s['validation']
  checks['inner_disjoint_and_exhaustive'] &= len(val)==len(set(val))==len(tr)
 checks.update(all_3000_fit_hashes=True,all_train_memberships=True,canonical_labels=True,all_selected_transforms_train_only=True,all_C_choices_reconstructed=True,all_models_converged=True,all_selected_probabilities_replayed=True,no_news_exact_fallback=True)
 maxerr=0.;nfit=0;nmodel=0;nrow=0;overrides=0
 paths=list((PRIVATE/'baseline').glob('*/complete.json'));checks['sixty_completed_blocks']=len(paths)==60
 with threadpool_limits(2):
  for complete in paths:
   run=complete.parent;seed,stock,fold=run.name.split('_');seed=int(seed);fold=int(fold)
   stamp=json.loads(complete.read_text());checks['all_3000_fit_hashes'] &= all(sha(run/n)==h for n,h in stamp['artifacts'].items())
   t=json.loads((run/'training.json').read_text());cv=pd.read_csv(run/'inner_selection.csv');p=pd.read_csv(run/'outer_predictions_SEALED.csv',float_precision='round_trip')
   ev=folds[(folds.seed==seed)&(folds.symbol==stock)&(folds.fold==fold)]['index'].to_numpy(int)
   tr=np.array(sorted(set(np.flatnonzero(d.symbol==stock))-set(ev)))
   checks['all_train_memberships'] &= t['outer_train']==tr.tolist() and t['outer_test']==ev.tolist() and p.row_id.tolist()==d.iloc[ev].row_id.tolist()
   checks['canonical_labels'] &= np.array_equal(p.label,d.iloc[ev].label)
   for fit in t['fits']:
    a,b=set(fit['train_indices']),set(fit['evaluation_indices'])
    checks['all_train_memberships'] &= not bool(a&b) and a<=set(tr) and not bool(a&set(ev))
    checks['all_models_converged'] &= fit['iterations']<3000
    nfit+=1
   for method in ['PRICE','PAPER','FULL','FINBERT','MODERN']:
    rows=cv[cv.method==method];chosen=min(CS,key=lambda c:(-rows[rows.C==c].BA.mean(),rows[rows.C==c].Brier.mean(),c))
    checks['all_C_choices_reconstructed'] &= chosen==t['selected_C'][method]
    bundle=joblib.load(run/f'{method}_selected.joblib');tf=bundle['transform'];model=bundle['model']
    checks['all_selected_transforms_train_only'] &= tf.fit_ids==d.iloc[tr].row_id.tolist() and bundle['train_indices']==tr.tolist()
    if method in ('FINBERT','MODERN'):
     expected=sorted({k for i in tr for k in ids[i]});checks['all_selected_transforms_train_only'] &= tf.article_ids==expected
     checks['all_selected_transforms_train_only'] &= np.max(np.abs(tf.pca.mean_-emb[method][expected].mean(0)))<1e-12
    # Direct sigmoid replay does not invoke runner prediction/fallback helper.
    x=tf.transform(d,ev,means);actual=expit(np.asarray(x@model.coef_.ravel()).ravel()+model.intercept_[0])
    if method in ('FULL','FINBERT','MODERN'):
     no=d.iloc[ev].has_news.to_numpy()==0;actual[no]=p.PRICE.to_numpy()[no]
     mismatches=int(np.count_nonzero(p[method].to_numpy()[no]!=p.PRICE.to_numpy()[no]));overrides+=mismatches
     checks['no_news_exact_fallback'] &= mismatches==0
    error=float(np.max(np.abs(actual-p[method].to_numpy())));maxerr=max(maxerr,error)
    checks['all_selected_probabilities_replayed'] &= error<=1e-12;nmodel+=1;nrow+=len(ev)
 checks['expected_counts']=nfit==3000 and nmodel==300 and nrow==1607*3*5
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},actual_fits=nfit,selected_models_replayed=nmodel,row_probabilities_replayed=nrow,max_probability_error=maxerr,no_news_fallback_mismatches=overrides,predictive_metrics_read=False,verifier_estimator_fits=0)
 dump(OUT/'BASELINE_VERIFICATION.json',result);print(json.dumps(result,indent=2))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
