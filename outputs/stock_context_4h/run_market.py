"""Market v1 runner. Synthetic mode is test-only; real mode stays gated."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
METHODS={'M0':['p0'],'Mmeta':['p0','market_age_log1p_hours','market_prior_session_fraction','market_window_missing'],'M1':['p0','market_age_log1p_hours','market_prior_session_fraction','market_window_missing','spy_intrabar_return_60tm','spy_rv_60tm','qqq_intrabar_return_60tm','qqq_rv_60tm']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(d,tr,ev,cols,c):
 x=d.loc[tr,cols].to_numpy(float);mean=np.nanmean(x,0);mean=np.where(np.isfinite(mean),mean,0);scale=np.nanstd(x,0);scale=np.where((scale>1e-12)&np.isfinite(scale),scale,1);z=lambda a:np.where(np.isfinite((a[cols].to_numpy(float)-mean)/scale),(a[cols].to_numpy(float)-mean)/scale,0);m=LogisticRegression(C=c,solver='liblinear',random_state=573,max_iter=3000,tol=1e-8).fit(z(d.loc[tr]),d.loc[tr,'label']);return mean,scale,m.coef_,m.intercept_,m.predict_proba(z(d.loc[ev]))[:,1]
def metric(y,p):
 q=p>=.5;y=np.asarray(y);return {'n':len(y),'BA':float((q[y==1].mean()+(~q)[y==0].mean())/2),'Brier':float(np.mean((p-y)**2)),'constant':bool(q.min()==q.max())}
def run_synthetic(source,out):
 d=pd.read_csv(source);out.mkdir();models=out/'models';models.mkdir();pred=[];evidence=[]
 for s,g in d.groupby('symbol'):
  for month in sorted(g.month.unique()):
   tr=g.index[g.month<month].to_numpy();ev=g.index[g.month==month].to_numpy()
   if not len(tr):continue
   for method,cols in METHODS.items():
    mean,scale,coef,intercept,p=fit(d,tr,ev,cols,.1);name=f'{method}_{s}_{month}';np.savez(models/(name+'.npz'),mean=mean,scale=scale,coef=coef,intercept=intercept,classes=np.array([0,1]));meta={'method':method,'symbol':s,'fold':month,'C':.1,'columns':cols,'training_months':sorted(d.loc[tr,'month'].unique()),'train_n':len(tr),'eval_n':len(ev),'npz_sha256':sha(models/(name+'.npz'))};(models/(name+'.json')).write_text(json.dumps(meta));
    for key,v in zip(d.loc[ev,'key'],p):pred.append({'key':key,'symbol':s,'month':month,'label':int(d.loc[d.key==key,'label'].iloc[0]),method:float(v)})
    evidence.append(meta)
 # consolidate method rows
 frame=pd.DataFrame(pred).groupby(['key','symbol','month','label'],as_index=False).first();frame.to_csv(out/'predictions.csv',index=False);monthly=[]
 for (s,m),g in frame.groupby(['symbol','month']):
  for method in METHODS:monthly.append({'symbol':s,'month':m,'method':method,**metric(g.label,g[method])})
 pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False);pd.DataFrame(monthly).groupby(['symbol','method'],as_index=False)[['BA','Brier']].mean().to_csv(out/'metrics.csv',index=False);pd.DataFrame([{'symbol':s,'fold':m,'C':.1} for s in d.symbol.unique() for m in sorted(d.month.unique())]).to_csv(out/'m0_cv.csv',index=False);(out/'selection.json').write_text(json.dumps({'fixed_C':.1,'same_C_all_methods':True}));(out/'training_evidence.json').write_text(json.dumps(evidence));pd.DataFrame({'symbol':d.symbol,'month':d.month,'SPY_valid':1,'QQQ_valid':1}).drop_duplicates().to_csv(out/'market_coverage.csv',index=False);(out/'protocol_fingerprint.json').write_text(json.dumps({'methods':METHODS,'source_sha256':sha(source),'source_path':str(source)}));(out/'advancement.json').write_text(json.dumps({'outer_cells':6,'synthetic':True}));(out/'attribution.json').write_text(json.dumps({'comparison':'M1_vs_Mmeta','synthetic':True}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--approve-market-context-run',action='store_true');p.add_argument('--synthetic-input');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if a.synthetic_input:run_synthetic(Path(a.synthetic_input),out);return
 if not a.approve_market_context_run:raise SystemExit('Real market scoring is preregistered only.')
 raise SystemExit('STOP_NO_MARKET_RUN: AUTH_REQUIRED; real Mmeta/M1 not authorized')
if __name__=='__main__':main()
