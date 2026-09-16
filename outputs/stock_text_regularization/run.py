import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import prepared,sha,require_empty,verify_files,baseline,fold_indices,metric_details,scores
D,_=prepared();P=json.loads((B/'protocol.json').read_text());R=B/'results';require_empty(R);R.mkdir(exist_ok=True);(R/'started.json').write_text('{"status":"training"}')
paths=[B/'protocol.json',B/'run.py',O/'stock_robust/common.py',O/'stock_robust/results/prepared.json',O/'stock_robust/results/linear/results.json']
sources={str(p.relative_to(O)):sha(p) for p in paths};result=dict(protocol=P,stocks={},source_sha256=sources,test_evaluated=False);started=time.time()
for s,dd in D.groupby('symbol'):
    d=dd.reset_index(drop=True);tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));F=R/s;F.mkdir();result['stocks'][s]={};pred=d.iloc[va][['symbol','start_utc','label','has_news']].copy()
    for kind in P['representations']:
        grid=[]
        for C in P['C']:
            for month,a,b in fold_indices(d):
                fit=baseline(kind,C).set_params(model__l1_ratio=0).fit(d.iloc[a],d.label.iloc[a]);p=fit.predict_proba(d.iloc[b])[:,1]
                grid.append(dict(C=C,month=month,**scores(d.label.iloc[b],p)))
        g=pd.DataFrame(grid);best=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0]
        fit=baseline(kind,float(best.C)).set_params(model__l1_ratio=0).fit(d.iloc[tr],d.label.iloc[tr]);p=fit.predict_proba(d.iloc[va])[:,1]
        pred[kind]=p;g.to_csv(F/f'{kind}_grid.csv',index=False);joblib.dump(fit,F/f'{kind}.joblib')
        result['stocks'][s][kind]=dict(C=float(best.C),training_cv=dict(balanced_accuracy=float(best.balanced_accuracy),brier=float(best.brier)),training=scores(d.label.iloc[tr],fit.predict_proba(d.iloc[tr])[:,1]),validation=metric_details(d.iloc[va],p),nonzero=int(np.count_nonzero(fit['model'].coef_)))
        print(s,kind,'C',best.C,'BA',result['stocks'][s][kind]['validation']['overall']['balanced_accuracy'],'Brier',result['stocks'][s][kind]['validation']['overall']['brier'],flush=True)
    pred.to_csv(F/'validation_predictions.csv',index=False)
verify_files(O,sources);result['elapsed_seconds']=time.time()-started;(R/'results.json').write_text(json.dumps(result,indent=2));print('E11 complete. Final test not evaluated.',flush=True)
