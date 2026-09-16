import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.optimize import minimize_scalar
from scipy.special import expit
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import prepared,baseline,scores,metric_details,require_empty,sha,verify_files
D,M=prepared();P=json.loads((B/'protocol.json').read_text());R=B/'results';require_empty(R);R.mkdir(exist_ok=True);(R/'started.json').write_text('{"status":"fitting temperatures"}')
sources={str(p.relative_to(O)):sha(p) for p in [B/'protocol.json',B/'run.py',O/'stock_robust/common.py',O/'stock_robust/results/prepared.json',O/'stock_robust/results/linear/results.json']}
result=dict(protocol=P,stocks={},test_evaluated=False,source_sha256=sources);started=time.time()
for s,d in D.groupby('symbol'):
    d=d.reset_index(drop=True);dates=pd.to_datetime(d.start_utc,utc=True);tr=d.split.eq('train');va=d.split.eq('validation');pred=d.loc[va,['symbol','start_utc','label','has_news']].copy();F=R/s;F.mkdir();result['stocks'][s]={}
    for kind in P['methods']:
        rows=[];selection=[]
        for month in [6,7,8]:
            candidates=[]
            for C in [.01,.1,1.]:
                metrics=[]
                for inner in range(month-3,month):
                    lo=pd.Timestamp(f'2018-{inner:02d}-01',tz='UTC');hi=lo+pd.offsets.MonthBegin();a=(dates<lo)&tr;b=(dates>=lo)&(dates<hi)&tr
                    assert pd.to_datetime(d.loc[a,'end_utc'],utc=True).max()<pd.to_datetime(d.loc[b,'cutoff_utc'],utc=True).min()
                    fit=baseline(kind,C).fit(d.loc[a],d.loc[a,'label']);p=fit.predict_proba(d.loc[b])[:,1];m=scores(d.loc[b,'label'],p);metrics.append(m);selection.append(dict(outer_month=month,inner_month=inner,C=C,**m))
                candidates.append((np.mean([x['balanced_accuracy'] for x in metrics]),-np.mean([x['brier'] for x in metrics]),C))
            _,_,C=max(candidates);lo=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC');hi=lo+pd.offsets.MonthBegin();a=(dates<lo)&tr;b=(dates>=lo)&(dates<hi)&tr
            fit=baseline(kind,C).fit(d.loc[a],d.loc[a,'label']);logits=fit.decision_function(d.loc[b]);q=d.loc[b,['start_utc','cutoff_utc','label']].copy();q['logit']=logits;q['C']=C;q['outer_month']=month;q['training_end']=d.loc[a,'end_utc'].max();rows.append(q)
        oof=pd.concat(rows,ignore_index=True);y=oof.label.to_numpy();z=oof.logit.to_numpy()
        def loss(logT):
            v=z/np.exp(logT);return float(np.mean(np.logaddexp(0,v)-y*v))
        opt=minimize_scalar(loss,bounds=(np.log(.25),np.log(10.)),method='bounded',options={'xatol':1e-8});assert opt.success;T=float(np.exp(opt.x))
        file=O/f'stock_robust/results/linear/{s}/{kind}.joblib';sources[str(file.relative_to(O))]=sha(file);fit=joblib.load(file);logits=fit.decision_function(d.loc[va]);p=expit(logits/T);old=expit(logits);assert np.array_equal(p>=.5,old>=.5)
        pred[kind+'_uncalibrated']=old;pred[kind+'_temperature']=p
        result['stocks'][s][kind]=dict(temperature=T,oof_bce_before=loss(0.),oof_bce_after=loss(np.log(T)),calibration_rows=len(oof),selected_C_by_outer_month=oof.groupby('outer_month').C.first().to_dict(),uncalibrated=metric_details(d.loc[va],old),calibrated=metric_details(d.loc[va],p))
        oof.to_csv(F/f'{kind}_nested_oof.csv',index=False);pd.DataFrame(selection).to_csv(F/f'{kind}_inner_selection.csv',index=False)
        print(s,kind,'T',round(T,3),'BA',round(scores(d.loc[va,'label'],p)['balanced_accuracy'],4),'Brier',round(scores(d.loc[va,'label'],old)['brier'],4),'->',round(scores(d.loc[va,'label'],p)['brier'],4),flush=True)
    pred.to_csv(F/'validation_predictions.csv',index=False)
verify_files(O,sources);result['elapsed_seconds']=time.time()-started;(R/'results.json').write_text(json.dumps(result,indent=2));print('E09-C complete: directions unchanged, final test not evaluated.',flush=True)
