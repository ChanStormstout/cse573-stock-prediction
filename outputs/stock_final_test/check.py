"""Reproduce the fixed holdout evaluation and paired date uncertainty."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from scipy.special import expit
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import prepared,verify_files,sha,scores,PRICE
P=json.loads((B/'protocol.json').read_text());R=B/'results';M=json.loads((R/'prepared.json').read_text());E=R/'evaluation';result=json.loads((E/'results.json').read_text())
assert result['test_evaluated'] is True and result['protocol']==P
assert M['protocol_sha256']==sha(B/'protocol.json') and result['prepared_sha256']==sha(R/'prepared.json')
verify_files(O.parent,P['source_sha256']);verify_files(R,M['artifact_sha256']);dev,_=prepared()
test=pd.read_pickle(R/'test_inputs.pkl');pred=pd.read_csv(E/'predictions.csv');assert test.start_utc.tolist()==pred.start_utc.tolist() and test.symbol.tolist()==pred.symbol.tolist();np.testing.assert_array_equal(test.label,pred.label)
articles=pd.read_csv(R/'test_article_index.csv').set_index('record_key');at=pd.to_datetime(articles.available_utc,utc=True,format='mixed');pub=pd.to_datetime(articles.published_utc,utc=True,format='mixed');crawl=pd.to_datetime(articles.crawled_utc,utc=True,format='mixed')
np.testing.assert_array_equal(at.astype('int64'),np.maximum(pub.astype('int64'),crawl.astype('int64')))
for row in test.itertuples():
    cut=pd.Timestamp(row.cutoff_utc);assert pd.Timestamp(row.start_utc)-cut==pd.Timedelta(minutes=5)
    assert pd.Timestamp(row.end_utc)-pd.Timestamp(row.start_utc)==pd.Timedelta(hours=1)
    history=pd.to_datetime(row.history_ends.split('|'),utc=True);assert len(history)==6 and history.max()<=cut
    keys=row.news_record_keys.split('|') if row.news_record_keys else []
    assert len(keys)==row.news_count and len(set(keys))==len(keys)
    assert at.loc[keys].le(cut).all() and at.loc[keys].gt(cut-pd.Timedelta(hours=4)).all()
    title=set().union(*(set(str(articles.loc[k,'stem_title']).split()) for k in keys));body=set().union(*(set(str(articles.loc[k,'stem_body']).split()) for k in keys))
    assert ' '.join(sorted(title))==row.stem_title and ' '.join(sorted(body))==row.stem_body
checks=[];frames={}
for s,dd in test.groupby('symbol'):
    d=dd.reset_index(drop=True);q=pred[pred.symbol.eq(s)].reset_index(drop=True);frames[s]=q;train=dev[dev.symbol.eq(s)&dev.split.eq('train')]
    assert len(d)==P['expected_test_windows'][s]
    assert pd.to_datetime(train.end_utc,utc=True).max()<pd.to_datetime(d.cutoff_utc,utc=True).min()
    for name,spec in P['models'][s].items():
        path=O.parent/spec['path'];assert sha(path)==spec['sha256'];model=joblib.load(path)
        np.testing.assert_allclose(model['features'].named_transformers_['numeric'].mean_,train[PRICE].mean(),atol=1e-12)
        # Same training data/configuration; independent re-fit for provenance,
        # never used to select a configuration or alter the saved predictor.
        refit=clone(model).fit(train,train.label)
        np.testing.assert_allclose(model['model'].coef_,refit['model'].coef_,atol=1e-9)
        np.testing.assert_allclose(model['model'].intercept_,refit['model'].intercept_,atol=1e-9)
        np.testing.assert_array_equal(model['features'].get_feature_names_out(),refit['features'].get_feature_names_out())
        p=model.predict_proba(d)[:,1];np.testing.assert_allclose(p,q[name],atol=1e-12)
        for field,value in scores(q.label,p).items():assert abs(value-result['stocks'][s][name]['overall'][field])<1e-10
        for month,g in q.groupby(q.start_utc.str[:7]):
            for field,value in scores(g.label,g[name].to_numpy()).items():
                saved=result['stocks'][s][name]['monthly'][month][field]
                assert saved==value if value is None else abs(value-saved)<1e-10
        checks.append(dict(symbol=s,model=name,training_refit_matches=True,test_prediction_max_error=float(np.max(np.abs(p-q[name])))))
    p=expit(joblib.load(O.parent/P['models'][s]['body_l1']['path']).decision_function(d)/P['body_temperature'][s]);np.testing.assert_allclose(p,q.body_l1_temperature)
    np.testing.assert_array_equal(p>=.5,q.body_l1.ge(.5));np.testing.assert_allclose(q.training_prior,P['training_up_prior'][s])
    for field,value in scores(q.label,p).items():assert abs(value-result['stocks'][s]['body_l1_temperature']['overall'][field])<1e-10

days=sorted(pred.start_utc.str[:10].unique());groups={s:{day:np.flatnonzero(q.start_utc.str[:10].eq(day)) for day in days} for s,q in frames.items()}
def ba(y,p):return .5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean())
intervals={};contrasts=P['primary_contrasts']+P['secondary_contrasts']
for block in [1,5]:
    rng=np.random.default_rng(583);deltas={f'{a} minus {b}':{s:[] for s in frames} for a,b in contrasts}
    for _ in range(2000):
        draw=rng.integers(0,len(days),size=len(days)) if block==1 else np.concatenate([np.arange(i,i+5) for i in rng.integers(0,len(days)-4,size=int(np.ceil(len(days)/5)))])[:len(days)]
        for s,q in frames.items():
            ix=np.concatenate([groups[s][days[i]] for i in draw]);y=q.label.to_numpy()[ix]
            for a,b in contrasts:deltas[f'{a} minus {b}'][s].append(ba(y,q[a].to_numpy()[ix])-ba(y,q[b].to_numpy()[ix]))
    intervals[f'block_{block}_days']={}
    for a,b in contrasts:
        name=f'{a} minus {b}';v=deltas[name];intervals[f'block_{block}_days'][name]={s:dict(delta_BA=float(ba(q.label.to_numpy(),q[a].to_numpy())-ba(q.label.to_numpy(),q[b].to_numpy())),percentile95=np.quantile(v[s],[.025,.975]).tolist()) for s,q in frames.items()}
        intervals[f'block_{block}_days'][name]['equal_stock_mean_interval']=np.quantile(np.mean(list(v.values()),axis=0),[.025,.975]).tolist()
for filename in ['before_sha256.json','new_experiment_sha256.json']:
    for p,h in json.loads((O/'stock_review_fixes'/filename).read_text()).items():assert sha(O.parent/p)==h,p
(R/'audit.json').write_text(json.dumps(dict(fixed_model_checks=checks,intervals=intervals,all_prior_artifacts_unchanged=True,test_evaluated=True),indent=2))
print(json.dumps(intervals,indent=2));print('PASS: 729 holdout rows/time bounds, 10 saved models reproduced by training-only refits, all test predictions/metrics, unchanged calibration directions, paired 1/5-day intervals and prior artifact hashes verified. No holdout fitting/selection.')
