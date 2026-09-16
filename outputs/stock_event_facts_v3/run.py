"""E14-P: explicitly limited fact projection. Not full-schema quality acceptance.
Issuer fields and guidance are excluded. Provisional historical ablation only.
"""
from pathlib import Path
import sys,json,datetime,time,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import *
BASE=['event_log_count','event_has','target_fraction','rating_fraction']
FACT=['pt_up','pt_down','pt_set','pt_maintain','pt_change','pt_pair_known','pt_value_known','rating_up','rating_down','rating_maintain','rating_pair_known']
AGE=['published_age_log','crawl_delay_log','headline_fraction']
def vector(f):
    pt=f['kind']=='analyst_target';rt=f['kind']=='analyst_rating';act=f['action'];change=f.get('change')
    return np.array([pt,rt,pt and act=='up',pt and act=='down',pt and act=='set',pt and act=='maintain',np.clip(change,-1,1) if change is not None else 0,pt and f.get('old_value') is not None and f.get('new_value') is not None,pt and f.get('new_value') is not None,rt and act=='up',rt and act=='down',rt and act=='maintain',rt and f.get('old_rating') is not None],dtype=float)
def features(D,events):
    lookup={}
    for f in events:
        if f['kind'] in ['analyst_target','analyst_rating']:lookup.setdefault((f['symbol'],f['record_key']),[]).append(f)
    rows=[]
    for r in D.itertuples():
        fs=[f for k in r.news_record_keys.split('|') if k for f in lookup.get((r.symbol,k),[])];n=len(fs);f=dict(event_log_count=float(np.log1p(n)),event_has=int(n>0));mean=np.zeros(13);shrunk=np.zeros(13);age=np.zeros(3)
        if n:
            avail=pd.to_datetime([x['available_utc'] for x in fs],utc=True,format='mixed');pub=pd.to_datetime([x['published_utc'] for x in fs],utc=True,format='mixed');assert (avail<=r.cutoff_utc).all() and (avail>r.cutoff_utc-pd.Timedelta(hours=4)).all();assert (pub<=avail).all()
            ages=(r.cutoff_utc-pub).total_seconds().to_numpy()/3600;q=np.array([x['q'] for x in fs]);w=q*np.exp2(-ages/4);v=np.stack([vector(x) for x in fs]);mean=v.mean(0);shrunk=(w[:,None]*v).sum(0)/(1+w.sum());age=np.array([np.log1p(ages.mean()),np.log1p(np.mean([x['lag_hours'] for x in fs])),np.mean([x['current_disclosure']=='headline_supported' for x in fs])])
        for i,c in enumerate(['target_fraction','rating_fraction']+FACT):f[c]=mean[i];f['shrunk_'+c]=shrunk[i]
        f.update(dict(zip(AGE,age)));rows.append(f)
    return pd.concat([D.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
def columns(kind):
    if kind=='B0':return []
    if kind=='B1':return BASE
    if kind=='B2':return BASE+FACT
    return BASE[:2]+['shrunk_'+c for c in BASE[2:]+FACT]+AGE
def factory(kind,C):
    parts=[('price',StandardScaler(),PRICE)]
    if columns(kind):parts.append(('events',StandardScaler(with_mean=False),columns(kind)))
    return Pipeline([('features',ColumnTransformer(parts)),('model',LogisticRegression(C=C,l1_ratio=0,solver='liblinear',max_iter=3000,random_state=573))])
if __name__=='__main__':
    R=B/'prediction';require_empty(R);R.mkdir(exist_ok=True);Q=json.loads((B/'results/quality.json').read_text());assert not Q['guidance_predictor_approved'];review=json.loads((B/'results/assistant_review.json').read_text());assert len(review)>=10 and all(r['predictor_fields_correct'] for r in review)
    paths=[B/'run.py',B/'results/events.jsonl',B/'results/quality.json',B/'results/assistant_review.json',O/'stock_adaptive/common.py',O/'stock_robust/results/data.pkl',O/'stock_final_test/results/test_inputs.pkl'];P=dict(id='E14_LIMITED_PROJECTION',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality_boundary='Full frame gate still FAILED on issuer fields; separate reduced projection excludes issuer entirely. Only exploratory evaluation of assistant-checked company/action/numeric/rating fields; no independent human validation or AMZN quality claim.',policy='monthly expanding fixed for all B0-B3; selected as common replay protocol, not stock-specific best E13 policy',models=['B0','B1','B2','B3'],C=[.01,.1,1.],selected_types=['analyst_target','analyst_rating'],excluded_types=['revenue_guidance'],selection='same last-three-month past-only folds, BA then Brier then larger C',B3='publication age proxy, H=4 hours, lambda=1, q=.5 or1 from frozen extractor; no H search; same candidate frames and all original windows',standardization='price centered; event columns variance-scaled without centering to preserve no-event zero',source_sha256={str(p.relative_to(O.parent)):sha(p) for p in paths});(B/'prediction_protocol.json').write_text(json.dumps(P,indent=2));events=[json.loads(x) for x in (B/'results/events.jsonl').read_text().splitlines()];D=features(load_data(),events);D.to_pickle(R/'data.pkl');preds=[];records=[];start=time.time()
    for s,sd in D.groupby('symbol'):
        for kind in P['models']:
            for month in ['2018-09','2018-10','2018-11','2018-12','2019-01','2019-02']:
                tr=train_rows(sd,month,'expanding');te=sd[sd.start_utc.dt.strftime('%Y-%m').eq(month)];folder=R/f'{s}_{kind}_{month}';folder.mkdir();C,grid=select(tr,month,'expanding',lambda c:factory(kind,c),P['C']);model=factory(kind,C).fit(tr,tr.label);p=model.predict_proba(te)[:,1];joblib.dump(model,folder/'model.joblib');grid.to_csv(folder/'cv.csv',index=False);assert tr.end_utc.max()<te.cutoff_utc.min()
                r=dict(symbol=s,kind=kind,month=month,C=C,features=PRICE+columns(kind),train_keys=row_keys(tr),test_keys=row_keys(te),training=metrics(tr,model.predict_proba(tr)[:,1]),evaluation=metrics(te,p));(folder/'manifest.json').write_text(json.dumps(r,indent=2));records.append(r);q=te[['symbol','start_utc','label','split','has_news','news_count','event_has','event_log_count']].copy();q['kind']=kind;q['month']=month;q['p']=p;preds.append(q);print(s,kind,month,round(r['evaluation']['balanced_accuracy'],4),flush=True)
    pd.concat(preds,ignore_index=True).to_csv(R/'predictions.csv',index=False);verify_files(O.parent,P['source_sha256']);(R/'results.json').write_text(json.dumps(dict(models=records,seconds=time.time()-start),indent=2));print('E14 reduced-projection ablation complete',flush=True)
