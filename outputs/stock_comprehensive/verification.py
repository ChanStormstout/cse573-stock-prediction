from core import *
from trading import reward_step,TradingEnv,evaluate_policy,raw_prices
from correction import OffsetCorrection,gate_check
from events import canonical_actor,extract,normalize
import tempfile

def run_checks(run):
    out=run/'verification.json'
    if out.exists():raise FileExistsError(out)
    checks={};d=load(run)
    def rejects(fn):
        try:fn()
        except (RuntimeError,ValueError,FileExistsError):return True
        raise AssertionError('Unsafe call accepted')
    checks['future_labels_rejected']=rejects(lambda:temporal(d.tail(10),d.head(10)))
    checks['overwrite_rejected']=rejects(lambda:prepare(run,{}))
    from guards import cache_metadata,cache_readable,pilot_gate,validate_resume
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'cache';p.write_text('v1');meta=Path(tmp)/'meta.json';expected={'text':'a','model':'fixed','max_length':256};dump(meta,cache_metadata(p,expected));assert cache_readable(p,meta,expected)
        checks['changed_text_cache_rejected']=rejects(lambda:cache_readable(p,meta,{**expected,'text':'changed'}));checks['changed_model_cache_rejected']=rejects(lambda:cache_readable(p,meta,{**expected,'model':'new'}));p.write_text('bad');checks['corrupt_cache_rejected']=rejects(lambda:cache_readable(p,meta,expected))
    failed=[dict(schema_valid=i<11,verbatim_evidence=True) for i in range(20)];checks['quality_stop_at20']=rejects(lambda:pilot_gate(failed));checks['quality_stop_on_resume21']=rejects(lambda:pilot_gate(failed+[dict(schema_valid=True,verbatim_evidence=True)]));checks['changed_resume_config']=rejects(lambda:validate_resume([],[],{'fingerprint':'old'},'new'))
    rng=np.random.default_rng(573);z=rng.normal(size=(100,4));base=np.linspace(.05,.95,100);g=np.arange(100)<70;y=rng.integers(0,2,100);m=OffsetCorrection(.1).fit(z,y,base,g);p=m.predict(z,base,g);assert np.array_equal(p[~g],base[~g]);checks['no_event_exact_fallback']=True;assert gate_check({},100,100)=='pending_independent_review';assert gate_check({'independent_approved':True},29,100)=='insufficient_events_or_windows';checks['event_quality_and_count_gates']=True
    assert canonical_actor('Wells Fargo & Co. The firm currently has a neutral rating')=='Wells Fargo & Co';assert canonical_actor('Morgan Stanley on optimism about services business')=='Morgan Stanley';assert canonical_actor('Over') is None;checks['known_actor_span_regressions']=True
    f=extract('AMZN','test','Morgan Stanley downgrades Amazon','Correction: Morgan Stanley only cut its price target for Amazon. The rating remains overweight.')
    assert not any(x['action']=='down' and x['kind']=='analyst_rating' for x in f);checks['rating_target_correction_conflict']=True
    f=extract('AAPL','test','Apple price target raised from $100 to $120','Previously, Apple price target was raised from $100 to $120 last year.')
    assert not any(x['current_disclosure']=='headline_supported' and x['evidence_source']=='normalized_body' and x['extraction_status']=='provisional' for x in f);checks['historical_body_not_current']=True
    # Analytic transaction examples include entry, reversal, terminal liquidation.
    r,turn,fee=reward_step(0,1,0,.001,True);np.testing.assert_allclose(np.exp(r),.999**2);assert turn==2
    r,turn,_=reward_step(0,-1,1,.001,True);np.testing.assert_allclose(np.exp(r),.998*.999);assert turn==3
    r1,_,_=reward_step(.05,1,0,.001,False);r2,_,_=reward_step(110/105-1,1,1,.001,True);np.testing.assert_allclose(np.exp(r1+r2),1.1*.999**2);assert reward_step(.1,0,0,.001,True)[0]==0;checks['cost_reversal_cash_terminal_math']=True
    # Check all original samples retain chronological inputs.
    n=pd.read_pickle(W/'audit/news_index.pkl');n['key']=n.archive+'::'+n.member;n=n.set_index('key');maxnews=0
    for r in d.itertuples():
        assert pd.to_datetime(r.history_ends.split('|'),utc=True).max()<=r.cutoff_utc
        ids=[k for k in r.news_record_keys.split('|') if k]
        if ids:assert n.loc[ids,'available_utc'].max()<=r.cutoff_utc;maxnews+=len(ids)
    checks['all_3233_time_cutoffs']=True;checks['news_memberships_checked']=maxnews
    # Evidence offsets and no train/check shared proxy event group.
    events=pd.read_json(run/'M06/events.jsonl',lines=True);bodies=json.loads((O/'stock_event_facts/results/bodies.json').read_text())
    for r in events.itertuples():
        if r.evidence_source=='unresolved':continue
        text=normalize(bodies[r.record_key]) if r.evidence_source=='normalized_body' else normalize(r.title);assert text[r.evidence_start:r.evidence_end]==r.evidence
    assert events.groupby('event_group').quality_split.nunique().max()==1;checks['event_offsets_group_disjoint']=True
    # Reload all saved sklearn predictive models against original saved probabilities.
    replays=0;features=pd.read_pickle(run/'M08/features.pkl')
    import semantic
    # Runs launched as a script serialize __main__.SemanticLR; register for portable replay.
    import __main__;__main__.SemanticLR=semantic.SemanticLR
    for mech in ['M02','M03','M08']:
        pr=pd.read_csv(run/mech/'predictions.csv');data=features if mech=='M08' else d
        for path in (run/mech).rglob('model.joblib'):
            manifest=json.loads((path.parent/'manifest.json').read_text());assert sha(path)==manifest['model_sha256'];idx=pd.Index(keys(data));ix=idx.get_indexer(manifest['prediction_keys']);assert (ix>=0).all();b=data.iloc[ix];model=joblib.load(path);pred=model.predict_proba(b)[:,1]
            method=path.parent.name;st=b.symbol.iloc[0];expected=pr[(pr.symbol==st)&(pr.method==method)].set_index('start_utc').loc[b.start_utc.astype(str),'p'].to_numpy();np.testing.assert_allclose(pred,expected,atol=1e-10,rtol=0);replays+=1
    checks['sklearn_models_reloaded']=replays
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1);trans=pd.read_csv(run/'M09/transitions.csv');rlreplays=0
    for stock in ['AAPL','AMZN']:
        folder=run/'M09'/stock;t=pd.read_csv(folder/'samples.csv');scaler=joblib.load(folder/'scaler.joblib');raw=raw_prices(stock)
        # Parity of history inputs with original classification windows; keep flats separate.
        orig=d[d.symbol.eq(stock)].assign(start_utc=lambda x:x.start_utc.astype(str));merged=t.merge(orig,on='start_utc',suffixes=('_rl','_cls'));np.testing.assert_allclose(merged[[x+'_rl' for x in PRICE]],merged[[x+'_cls' for x in PRICE]],atol=1e-12)
        for day,s in t.groupby('day'):
            np.testing.assert_allclose(np.prod(1+s.trade_return),s.close.iloc[-1]/s.open.iloc[0],atol=1e-12)
        for seed in [573,574,575]:
            m=PPO.load(folder/f'ppo_{seed}/model.zip',device='cpu');assert m.num_timesteps==20000;e=TradingEnv(t,scaler,.001);fresh=evaluate_policy(e,lambda o,e,i:int(m.predict(o,deterministic=True)[0]));oldp=trans[(trans.symbol==stock)&(trans.method==f'PPO_{seed}')&(trans.cost_bps==10)].reset_index(drop=True);np.testing.assert_array_equal(fresh.action,oldp.action);np.testing.assert_allclose(fresh.log_return,oldp.log_return,atol=1e-14);rlreplays+=1
    checks['PPO_full_trajectory_reloads']=rlreplays;checks['continuous_execution_and_price_feature_parity']=True
    for folder in run.iterdir():
        if folder.is_dir() and (folder/'hashes.json').exists():verify(json.loads((folder/'hashes.json').read_text()))
    historical=0
    for path,root in [(O/'stock_review_fixes/before_sha256.json',ROOT),(O/'stock_review_fixes/new_experiment_sha256.json',ROOT),(O/'stock_final_test/artifact_sha256.json',O/'stock_final_test'),(O/'E13_E15_ARTIFACT_SHA256.json',ROOT)]:
        data=json.loads(path.read_text());verify({str(root/k):v for k,v in data.items()});historical+=len(data)
    checks['historical_artifacts_unchanged']=historical
    dump(out,{'passed':True,'checks':checks});print(json.dumps(checks,indent=2))
