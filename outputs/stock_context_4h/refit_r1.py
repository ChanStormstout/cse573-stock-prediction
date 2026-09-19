"""Independent canonical R1 refit: no historical fitter or selection import."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'audit_v2';PRIVATE=ROOT/'work/stock-data/context_increment_4h/audit_v2'
OLD=[f'{x}_{i}' for i in range(1,7) for x in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour'];REC=[]
for m in (5,15,30,60):REC += [f'recent_return_{m}',f'recent_range_{m}',f'recent_rv_{m}',f'recent_missing_{m}']
R1=OLD+REC+['overnight_gap','overnight_gap_missing','minutes_from_open','minutes_to_close']
def standard(train,ev):
 a=train[R1].to_numpy(float);mean=np.nanmean(a,0);mean=np.where(np.isfinite(mean),mean,0);scale=np.nanstd(a,0);scale=np.where((scale>1e-12)&np.isfinite(scale),scale,1);f=lambda x:np.where(np.isfinite((x[R1].to_numpy(float)-mean)/scale),(x[R1].to_numpy(float)-mean)/scale,0);return f(train),f(ev)
def main():
 d=pd.read_pickle(ROOT/'work/stock-data/nextgen_4h/price_v1/features.pkl').copy();d['month']=d.start_utc.dt.strftime('%Y-%m');pred=[];selection=[]
 for s,g in d.groupby('symbol'):
  records=[]
  for month in [f'2018-{m:02d}' for m in range(3,9)]:
   tr=g[g.month<month];ev=g[g.month==month]; candidates=[]
   for c in (.01,.1,1.):
    x,z=standard(tr,ev);p=LogisticRegression(C=c,solver='liblinear',random_state=573,max_iter=3000,tol=1e-8).fit(x,tr.label).predict_proba(z)[:,1];q=p>=.5;y=ev.label.to_numpy();ba=(q[y==1].mean()+((~q)[y==0].mean()))/2; candidates.append((c,ba,float(np.mean((p-y)**2))))
   chosen=.1 if not records else sorted([(c,-np.mean([x[1] for x in records if x[0]==c]),np.mean([x[2] for x in records if x[0]==c])) for c in (.01,.1,1.)],key=lambda x:(x[1],x[2],x[0]))[0][0]
   c=chosen;x,z=standard(tr,ev);p=LogisticRegression(C=c,solver='liblinear',random_state=573,max_iter=3000,tol=1e-8).fit(x,tr.label).predict_proba(z)[:,1];pred.extend(zip(ev.key,p));records += candidates;selection.append({'symbol':s,'fold':month,'C':c})
  tr=g[g.month<'2018-09'];ev=g[g.month>='2018-09'];chosen=sorted([(c,-np.mean([x[1] for x in records if x[0]==c]),np.mean([x[2] for x in records if x[0]==c])) for c in (.01,.1,1.)],key=lambda x:(x[1],x[2],x[0]))[0][0];x,z=standard(tr,ev);p=LogisticRegression(C=chosen,solver='liblinear',random_state=573,max_iter=3000,tol=1e-8).fit(x,tr.label).predict_proba(z)[:,1];pred.extend(zip(ev.key,p));selection.append({'symbol':s,'fold':'final','C':chosen})
 got=pd.DataFrame(pred,columns=['key','refit']);canon=pd.read_csv(ROOT/'outputs/stock_goal60_4h/v1/predictions.csv')[['key','R1']].dropna();j=canon.merge(got,on='key',validate='one_to_one');err=float(np.max(np.abs(j.R1-j.refit)));status='PASS' if len(j)==1374 and err<=1e-12 and bool(np.array_equal(j.R1>=.5,j.refit>=.5)) else 'BLOCKED_R1_PARITY';OUT.mkdir(exist_ok=True);(OUT/'r1_independent_refit.json').write_text(json.dumps({'status':status,'n':len(j),'max_abs_probability_error':err,'direction_parity':bool(np.array_equal(j.R1>=.5,j.refit>=.5)),'selection':selection},indent=2)+'\n');print(status,len(j),err)
if __name__=='__main__':main()
