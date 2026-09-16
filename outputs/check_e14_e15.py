"""Verify new numerical replays, timing, score selection and event data boundaries."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np,pandas as pd,joblib
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import *
from check_report import block_delta
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
F=module('fact_predictor_check',O/'stock_event_facts_v3/run.py');T=module('boost_predictor_check',O/'stock_small_boost/run.py')
def inspect(root,events=False):
    folder='prediction' if events else 'results';R=root/folder;P=json.loads((root/('prediction_protocol.json' if events else 'protocol.json')).read_text());verify_files(O.parent,P['source_sha256']);D=pd.read_pickle(R/'data.pkl') if events else load_data();D['log_news_count']=np.log1p(D.news_count);pred=pd.read_csv(R/'predictions.csv');records=json.loads((R/'results.json').read_text())['models'];assert len(records)==48
    for r in records:
        name=f"{r['symbol']}_{r['kind']}_{r['month']}" if events else f"{r['symbol']}_{r['feature']}_{r['kind']}_{r['month']}";f=R/name;model=joblib.load(f/'model.joblib');sd=D[D.symbol.eq(r['symbol'])];tr=train_rows(sd,r['month'],'expanding');te=sd[sd.start_utc.dt.strftime('%Y-%m').eq(r['month'])];assert row_keys(tr)==r['train_keys'] and row_keys(te)==r['test_keys'];assert tr.end_utc.max()<te.cutoff_utc.min();q=pred[(pred.symbol==r['symbol'])&(pred.kind==r['kind'])&(pred.month==r['month'])]
        if not events:q=q[q.feature.eq(r['feature'])]
        np.testing.assert_allclose(model.predict_proba(te)[:,1],q.p,atol=1e-12);g=pd.read_csv(f/'cv.csv');assert len(g)==9 and (pd.to_datetime(g.train_end,utc=True)<pd.to_datetime(g.valid_cutoff,utc=True)).all();best=g.groupby('config')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','config'],ascending=[False,True,False]).iloc[0];assert best.config==(r['C'] if events else r['config'])
        if r['month']=='2018-11' and ((events and r['kind']=='B3') or (not events and r['feature']=='price_count' and r['kind']=='boost')):
            refit=F.factory(r['kind'],r['C']) if events else T.factory(r['kind'],P['features'][r['feature']],r['config']);refit.fit(tr,tr.label);np.testing.assert_allclose(refit.predict_proba(te),model.predict_proba(te),atol=1e-12)
    rows=[];contrasts=[];flips=[];monthly=[];groups=['symbol','kind'] if events else ['symbol','feature','kind']
    for period,mask in [('development',pred.split.eq('validation')),('former_holdout',pred.split.eq('test')),('all_replay',pred.index>=0)]:
        for key,g in pred[mask].groupby(groups):rows.append(dict(period=period,**dict(zip(groups,key)),**metrics(g,g.p)))
        for key,g in pred[mask].groupby(['symbol'] if events else ['symbol','feature']):
            wide=g.pivot(index=['start_utc','label'],columns='kind',values='p').reset_index();key=key if isinstance(key,tuple) else (key,)
            for a,b in ([('B1','B0'),('B2','B1'),('B3','B2')] if events else [('boost','LR')]):
                contrasts.append(dict(period=period,symbol=key[0],feature='events' if events else key[1],contrast=a+'-'+b,**block_delta(wide,a,b)))
                if period=='former_holdout':
                    y=wide.label.astype(bool);pa=wide[a]>=.5;pb=wide[b]>=.5;flips.append(dict(symbol=key[0],feature='events' if events else key[1],contrast=a+'-'+b,fixes=int(((pa==y)&(pb!=y)).sum()),breaks=int(((pa!=y)&(pb==y)).sum()),both_wrong=int(((pa!=y)&(pb!=y)).sum())))
    for key,g in pred.groupby(groups+['month']):monthly.append(dict(zip(groups+['month'],key))|metrics(g,g.p))
    summary=pd.DataFrame(rows);summary.to_csv(R/'summary.csv',index=False);pd.DataFrame(monthly).to_csv(R/'monthly.csv',index=False);pd.DataFrame(flips).to_csv(R/'direction_changes.csv',index=False);(R/'intervals.json').write_text(json.dumps(contrasts,indent=2));print(root.name);print(summary[summary.period.eq('former_holdout')].to_string(index=False));return pred,D
if __name__=='__main__':
    p,d=inspect(O/'stock_event_facts_v3',True);q,_=inspect(O/'stock_small_boost');a=p[p.kind.eq('B0')];b=q[q.kind.eq('LR')&q.feature.eq('price')];np.testing.assert_allclose(a.p,b.p,atol=1e-12)
    # Independent event membership, quotations, numeric pairing, and zero/weak-evidence properties.
    B=O/'stock_event_facts_v3';ev=[json.loads(x) for x in (B/'results/events.jsonl').read_text().splitlines()];bodies=json.loads((O/'stock_event_facts/results/bodies.json').read_text());sys.path.insert(0,str(O/'stock_event_facts_v2'));from extract import clean
    pairs=pd.read_csv(O/'stock_event_facts/results/article_pairs.csv');pairset=set(zip(pairs.symbol,pairs.record_key))
    for f in ev:
        assert (f['symbol'],f['record_key']) in pairset
        evidence=' '.join(f['evidence'].split());body=' '.join(clean(bodies[f['record_key']]).split());assert evidence in body or evidence==' '.join(f['title'].split())
        assert pd.Timestamp(f['published_utc'])<=pd.Timestamp(f['available_utc'])
        if f['change'] is not None:assert abs(f['change']-(f['new_value']/f['old_value']-1))<1e-12
    allcols=F.BASE+F.FACT+['shrunk_'+c for c in F.BASE[2:]+F.FACT]+F.AGE;assert np.isfinite(d[allcols]).all().all();assert (d.loc[d.event_has.eq(0),allcols]==0).all().all()
    lookup={(f['symbol'],f['record_key']):f for f in ev if f['kind'] in ['analyst_target','analyst_rating']}
    for r in d.itertuples():
        fs=[lookup[(r.symbol,k)] for k in r.news_record_keys.split('|') if (r.symbol,k) in lookup]
        assert len(fs)==round(np.expm1(r.event_log_count))
        assert all(pd.Timestamp(f['available_utc'])<=r.cutoff_utc and pd.Timestamp(f['available_utc'])>r.cutoff_utc-pd.Timedelta(hours=4) for f in fs)
    w=2**-4;assert abs(w/(1+w)-1/17)<1e-12
    # E13 apparent improvement must also be inspected by month and model degeneracy.
    e13=[]
    for month in ['2018-11','2018-12','2019-01']:
        model=joblib.load(O/f'stock_adaptive/results/AAPL_l1_rolling6_{month}/model.joblib');e13.append(dict(month=month,nonzero_coefficients=int(np.count_nonzero(model['model'].coef_)),intercept=model['model'].intercept_.tolist()))
    historical={}
    for name in ['before_sha256.json','new_experiment_sha256.json']:
        h=json.loads((O/'stock_review_fixes'/name).read_text());verify_files(O.parent,h);historical[name]=len(h)
    h=json.loads((O/'stock_final_test/artifact_sha256.json').read_text());verify_files(O/'stock_final_test',h);historical['E12']=len(h)
    audit=dict(passed=True,numerical_models_replayed=96,representative_refits=4,matched_price_baselines_identical=True,event_quotes_and_numeric_pairs_checked=len(ev),all3233_event_membership_and_no_future_checks=True,zero_event_and_finite_features_checked=True,historical_unchanged=historical,E13_rolling_model_degeneracy=e13,quality_boundary='Full schema not accepted; reduced feature projection is assistant-reviewed exploratory only; AMZN quality unvalidated')
    (O/'E13_E15_VERIFICATION.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
