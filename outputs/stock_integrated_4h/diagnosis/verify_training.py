from pathlib import Path
import sys,time,json,os,datetime
import numpy as np,pandas as pd,joblib
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B));from run import Transform,LogisticRegression,metric
R=B/'runs/v1';OUT=B/'diagnosis';d=pd.read_pickle(R/'inputs.pkl');z=np.load(B/'prepared/articles.npz');keys=z['keys'];emb=z['embeddings'].astype(np.float64);selection=json.loads((R/'selection.json').read_text());records=[]
for sym,kind in [('AAPL','semantic'),('AMZN','body')]:
 g=d[d.symbol==sym];tr=g[g.month<'2018-09'];ev=g[g.month>='2018-09'];C=next(x['C'] for x in selection if x['symbol']==sym and x['month']=='final' and x['branch']==kind);start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter();print('FIT START',os.getpid(),start,sym,kind,len(tr),C,flush=True)
 tf=Transform(kind).fit(tr,keys,emb);x=tf.transform(tr,keys,emb);model=LogisticRegression(C=C,solver='liblinear',max_iter=3000,random_state=573,tol=1e-7).fit(x,tr.label);p=model.predict_proba(tf.transform(ev,keys,emb))[:,1];saved=joblib.load(R/'models'/f'{sym}_final_{kind}.joblib');old=saved['classifier'].predict_proba(saved['transform'].transform(ev,keys,emb))[:,1];diff=float(np.max(np.abs(old-p)));np.testing.assert_allclose(old,p,rtol=0,atol=1e-12)
 record=dict(pid=os.getpid(),started_utc=start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stock=sym,branch=kind,training_rows=len(tr),training_dates=tr.day.nunique(),feature_count=x.shape[1],C=C,optimizer_iterations=model.n_iter_.tolist(),coefficient_norm=float(np.linalg.norm(model.coef_)),fit_seconds=time.perf_counter()-t,retrained_probability_max_error=diff,train=metric(tr.label,model.predict_proba(x)[:,1]),note='Two independent verification refits; not new candidate experiments. FinBERT encoder frozen, LR actually optimized.')
 records.append(record);joblib.dump(dict(transform=tf,classifier=model),OUT/f'verification_{sym}_{kind}.joblib');print('FIT DONE',json.dumps(record),flush=True)
(OUT/'training_verification.json').write_text(json.dumps(records,indent=2))
