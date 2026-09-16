"""One fixed-model holdout evaluation; never fit or select on holdout labels."""
import json,sys,datetime
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.special import expit
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import verify_files,sha,require_empty,scores
P=json.loads((B/'protocol.json').read_text());R=B/'results';M=json.loads((R/'prepared.json').read_text())
assert sha(B/'protocol.json')==M['protocol_sha256'];verify_files(O.parent,P['source_sha256']);verify_files(R,M['artifact_sha256'])
F=R/'evaluation';require_empty(F);F.mkdir(exist_ok=True);(F/'started.json').write_text(json.dumps(dict(status='fixed evaluation started',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
D=pd.read_pickle(R/'test_inputs.pkl');assert set(D.split)=={'test'}
def details(d,p):
    return dict(overall=scores(d.label,p),monthly={month:dict(n=len(g),**scores(g.label,p[g.index])) for month,g in d.groupby(d.start_utc.str[:7])},coverage={name:dict(n=int(mask.sum()),**scores(d.loc[mask,'label'],p[mask])) for name,mask in [('with_news',d.has_news.eq(1).to_numpy()),('no_news',d.has_news.eq(0).to_numpy())] if mask.any()})
result=dict(protocol=P,stocks={},test_evaluated=True,evaluation_policy='Saved Jan-Aug models; no refitting, tuning, threshold choice or calibration fitting on holdout',prepared_sha256=sha(R/'prepared.json'))
preds=[]
for s,dd in D.groupby('symbol'):
    d=dd.reset_index(drop=True);pred=d[['symbol','start_utc','cutoff_utc','label','target_return','news_count','has_news']].copy();result['stocks'][s]={}
    for name,spec in P['models'][s].items():
        fit=joblib.load(O.parent/spec['path']);assert fit['model'].C==spec['C']
        assert fit['model'].l1_ratio==spec['l1_ratio']
        p=fit.predict_proba(d)[:,1];pred[name]=p;result['stocks'][s][name]=details(d,p)
        if name=='body_l1':
            q=expit(fit.decision_function(d)/P['body_temperature'][s]);assert np.array_equal(q>=.5,p>=.5)
            pred['body_l1_temperature']=q;result['stocks'][s]['body_l1_temperature']=details(d,q)
    p=np.full(len(d),P['training_up_prior'][s]);pred['training_prior']=p;result['stocks'][s]['training_prior']=details(d,p)
    print(s,json.dumps({k:v['overall'] for k,v in result['stocks'][s].items()}),flush=True);preds.append(pred)
pd.concat(preds,ignore_index=True).to_csv(F/'predictions.csv',index=False);verify_files(O.parent,P['source_sha256'])
(F/'results.json').write_text(json.dumps(result,indent=2));print('Fixed holdout evaluation complete; no model or parameter updated.',flush=True)
