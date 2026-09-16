"""Independent saved-prediction checks and paired day-block summaries."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
from common import *
B=Path(__file__).resolve().parent;R=B/'results'
def ba(y,p):
    a=y==1;b=~a
    return ((p[a]>=.5).mean()+(p[b]<.5).mean())/2 if a.any() and b.any() else np.nan
def block_delta(g,a,b,length=5,n=1500):
    dates=g.start_utc.str[:10].to_numpy();days=sorted(set(dates));idx=[np.where(dates==x)[0] for x in days];rng=np.random.default_rng(573);res=[]
    y=g.label.to_numpy();pa=g[a].to_numpy();pb=g[b].to_numpy()
    for _ in range(n):
        starts=rng.integers(0,max(1,len(days)-length+1),int(np.ceil(len(days)/length)))
        ds=np.concatenate([np.arange(i,min(i+length,len(days))) for i in starts])[:len(days)];j=np.concatenate([idx[k] for k in ds]);res.append(ba(y[j],pa[j])-ba(y[j],pb[j]))
    return dict(delta=float(ba(y,pa)-ba(y,pb)),CI95=np.nanquantile(res,[.025,.975]).tolist(),block_days=length)
if __name__=='__main__':
    P=json.loads((B/'protocol.json').read_text());verify_files(O.parent,P['source_sha256']);verify_files(B,json.loads((B/'artifact_sha256.json').read_text()));D=load_data();p=pd.read_csv(R/'predictions.csv');records=json.loads((R/'results.json').read_text())['models'];assert len(records)==72
    old=pd.read_csv(O/'stock_final_test/results/evaluation/predictions.csv')
    for r in records:
        folder=R/f"{r['symbol']}_{r['penalty']}_{r['policy']}_{r['month']}";model=joblib.load(folder/'model.joblib')
        sd=D[D.symbol.eq(r['symbol'])];train=sd[sd.split.eq('train')] if r['policy']=='frozen' else train_rows(sd,r['month'],r['policy']);test=sd[sd.start_utc.dt.strftime('%Y-%m').eq(r['month'])]
        assert row_keys(train)==r['train_keys'] and row_keys(test)==r['test_keys'];assert train.end_utc.max()<test.cutoff_utc.min();assert model['model'].C==r['C']
        q=p[(p.symbol==r['symbol'])&(p.penalty==r['penalty'])&(p.policy==r['policy'])&(p.month==r['month'])]
        np.testing.assert_allclose(model.predict_proba(test)[:,1],q.p,atol=1e-12)
        if r['policy']!='frozen':
            grid=pd.read_csv(folder/'cv.csv');assert len(grid)==9;assert (pd.to_datetime(grid.train_end,utc=True)<pd.to_datetime(grid.valid_cutoff,utc=True)).all()
            best=grid.groupby('config')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','config'],ascending=[False,True,False]).iloc[0];assert best.config==r['C']
        elif r['month']>='2018-11':
            o=old[(old.symbol==r['symbol'])&old.start_utc.str.startswith(r['month'])];np.testing.assert_allclose(o['body_'+r['penalty']],q.p,atol=1e-12)
    # Independent representative refits of each updated policy and penalty.
    for policy in ['expanding','rolling6']:
        for penalty in ['l1','l2']:
            r=next(x for x in records if x['symbol']=='AAPL' and x['month']=='2018-11' and x['policy']==policy and x['penalty']==penalty)
            sd=D[D.symbol.eq('AAPL')];tr=train_rows(sd,r['month'],policy);model=baseline('paper_stem_body',r['C']).set_params(model__l1_ratio=int(penalty=='l1')).fit(tr,tr.label);saved=joblib.load(R/f'AAPL_{penalty}_{policy}_2018-11/model.joblib');np.testing.assert_allclose(model['model'].coef_,saved['model'].coef_,atol=1e-10)
    rows=[];intervals=[];cases=[]
    for period,mask in [('development',p.split.eq('validation')),('former_holdout',p.split.eq('test')),('all_replay',p.index>=0)]:
        for (s,pen,policy),g in p[mask].groupby(['symbol','penalty','policy']):rows.append(dict(period=period,symbol=s,penalty=pen,policy=policy,**metrics(g,g.p)))
        for (s,pen),g in p[mask].groupby(['symbol','penalty']):
            wide=g.pivot(index=['start_utc','label'],columns='policy',values='p').reset_index()
            for policy in ['expanding','rolling6']:
                intervals.append(dict(period=period,symbol=s,penalty=pen,contrast=policy+'-frozen',**block_delta(wide,policy,'frozen')))
                if period=='former_holdout':
                    a=wide[policy]>=.5;b=wide.frozen>=.5;y=wide.label.astype(bool);cases.append(dict(symbol=s,penalty=pen,policy=policy,fixes=int(((a==y)&(b!=y)).sum()),breaks=int(((a!=y)&(b==y)).sum())))
    S=pd.DataFrame(rows);S.to_csv(R/'summary.csv',index=False);pd.DataFrame(cases).to_csv(R/'direction_changes.csv',index=False);(R/'intervals.json').write_text(json.dumps(intervals,indent=2));monthly=pd.DataFrame([dict(symbol=r['symbol'],penalty=r['penalty'],policy=r['policy'],month=r['month'],C=r['C'],n_train=r['n_train'],**r['evaluation']) for r in records]);monthly.to_csv(R/'monthly.csv',index=False)
    audit=dict(passed=True,saved_models=72,predictions_replayed=len(p),representative_refits=4,original_E12_predictions_identical=True,past_only_checks=True,selection_checks=True,source_hashes_unchanged=True,warning='Historical adaptive replay after E12 exposure, not fresh holdout');(R/'audit.json').write_text(json.dumps(audit,indent=2))
    print(S[S.period.eq('former_holdout')].to_string(index=False));print(json.dumps(audit))
