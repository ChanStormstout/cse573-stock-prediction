"""Independent saved-state and input audit for the completed E10 experiment."""
import json,sys,gc
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from model import WindowModel,probabilities,fit_transform,pool_bags
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import verify_files,sha,scores,inner_split
R=B/'results';M=json.loads((R/'prepared.json').read_text());C=Path(M['cache_dir']);P=M['protocol'];T=R/'training';result=json.loads((T/'results.json').read_text())
assert set(result['models'])=={'frozen','lora'} and 'elapsed_seconds' in result,'Training must finish before checking.'
assert result['test_evaluated'] is False and result['prepared_sha256']==sha(R/'prepared.json')
verify_files(O,M['source_sha256']);verify_files(C,M['cache_sha256']);verify_files(R,M['artifact_sha256'])
torch.set_num_threads(2);dev='mps' if torch.backends.mps.is_available() else 'cpu'
d=pd.read_csv(R/'manifest.csv').fillna({'news_record_keys':''});z=np.load(C/'data.npz');data={k:z[k] for k in ['bags','numeric','pooled','has','stock','y']}
data['prefix']=torch.load(C/'prefix.pt',weights_only=True,map_location='cpu');data['tail_state']=torch.load(C/'tail_state.pt',weights_only=True,map_location='cpu');data['config']=json.loads((C/'config.json').read_text())
articles=pd.read_csv(C/'articles.csv');upstream=pd.read_csv(O/'stock_finbert/results/article_probabilities.csv').set_index('record_key');at=pd.to_datetime(upstream.available_utc,utc=True,format='mixed')
assert set(d.split)=={'train','validation'} and len(d)==2504
np.testing.assert_array_equal(d.label,data['y']);np.testing.assert_array_equal(d.stock_index,data['stock'])
np.testing.assert_array_equal(d.has_news,data['has']);np.testing.assert_allclose(pool_bags(z['article_vectors'],data['bags']),data['pooled'])
for i,row in enumerate(d.itertuples()):
    keys=row.news_record_keys.split('|') if row.news_record_keys else [];cut=pd.Timestamp(row.cutoff_utc)
    expected=sorted(keys,key=lambda k:(at.loc[k],k),reverse=True)[:16];ids=data['bags'][i];actual=articles.iloc[ids[ids>=0]]
    assert actual.record_key.tolist()==expected and actual.symbol.eq(row.symbol).all()
    assert at.loc[expected].le(cut).all() and at.loc[expected].gt(cut-pd.Timedelta(hours=4)).all()
    company='Apple (AAPL)' if row.symbol=='AAPL' else 'Amazon (AMZN)'
    for article in actual.itertuples():assert article.text==f'Target company: {company}. News: {upstream.loc[article.record_key,"title"]}'
    age=float((cut-at.loc[expected]).dt.total_seconds().mean()/3600) if expected else 0.
    np.testing.assert_allclose(data['numeric'][i,-3:],[np.log1p(len(keys)),int(bool(keys)),age])
tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));assert len(tr)==2000 and len(va)==504
expected_transform=fit_transform(data['pooled'],data['numeric'],data['has'],tr);priors=np.array([data['y'][tr][data['stock'][tr]==s].mean() for s in [0,1]])
pred=pd.read_csv(T/'validation_predictions.csv');assert pred.start_utc.tolist()==d.start_utc.iloc[va].tolist();assert pred.symbol.tolist()==d.symbol.iloc[va].tolist()
ia,ib=inner_split(d,tr,20)
assert pd.to_datetime(d.end_utc.iloc[ia],utc=True).max()<pd.to_datetime(d.cutoff_utc.iloc[ib],utc=True).min()
checks=[]
def chosen(trace):
    best=float('inf');epoch=0
    for row in trace:
        if row['inner_bce']<best-P['training']['min_delta']:best=row['inner_bce'];epoch=row['epoch']
    return epoch
for kind in ['frozen','lora']:
    F=T/kind;g=pd.read_csv(F/'training_cv.csv');assert len(g)==18 and set(g.seed)=={573,574,575} and set(g.month)=={6,7,8}
    assert (pd.to_datetime(g.inner_train_end,utc=True)<pd.to_datetime(g.inner_monitor_start,utc=True)).all()
    assert (pd.to_datetime(g.inner_monitor_start,utc=True)<pd.to_datetime(g.outer_start,utc=True)).all()
    for entry in json.loads((F/'fold_traces.json').read_text()):
        assert chosen(entry['inner_trace'])==entry['epochs']==len(entry['refit_trace'])
    for seed in P['training']['seeds']:
        spec=result['models'][kind]['seeds'][str(seed)];assert chosen(spec['inner_trace'])==spec['epochs']==len(spec['refit_trace'])
        transform=joblib.load(F/f'{seed}_transform.joblib')
        for key,value in expected_transform.items():np.testing.assert_allclose(transform[key],value,atol=1e-9)
        net=WindowModel(kind,data['config'],data['tail_state'],transform,priors,seed,dev)
        net.load_state_dict(torch.load(F/f'{seed}.pt',weights_only=True,map_location=dev))
        assert sum(v.numel() for v in net.parameters() if v.requires_grad)==spec['checks']['trainable_parameters']
        if kind=='lora':
            updates=[]
            for key,value in net.tail.state_dict().items():
                if key.endswith(('.A','.B')):
                    if key.endswith('.B'):updates.append(float(value.abs().max().cpu()))
                    continue
                oldkey=key.replace('.query.base.','.query.').replace('.value.base.','.value.')
                assert torch.equal(value.cpu(),data['tail_state'][oldkey]),key
            assert max(updates)>0
        p=probabilities(net,data,va);err=float(np.max(np.abs(p-pred[f'{kind}_{seed}'])))
        np.testing.assert_allclose(p,pred[f'{kind}_{seed}'],atol=2e-6,rtol=1e-5)
        for s in ['AAPL','AMZN']:
            mask=pred.symbol.eq(s)
            for field,value in scores(pred.loc[mask,'label'],p[mask]).items():assert abs(value-spec['stocks'][s]['validation']['overall'][field])<2e-6
        checks.append(dict(kind=kind,seed=seed,max_abs_prediction_error=err));print('Reproduced',kind,seed,'max error',err,flush=True)
        del net;gc.collect()
        if dev=='mps':torch.mps.empty_cache()
def seed_ba(y,pp):return float(np.mean([.5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean()) for p in pp]))
days=sorted(pred.start_utc.str[:10].unique());rng=np.random.default_rng(581);delta={s:[] for s in ['AAPL','AMZN']};prob={};groups={}
for s in delta:
    q=pred[pred.symbol.eq(s)].reset_index(drop=True);prob[s]=(q.label.to_numpy(),q[[f'frozen_{n}' for n in [573,574,575]]].to_numpy().T,q[[f'lora_{n}' for n in [573,574,575]]].to_numpy().T)
    groups[s]={day:np.flatnonzero(q.start_utc.str[:10].eq(day)) for day in days}
for _ in range(2000):
    draw=rng.choice(days,len(days),replace=True)
    for s in delta:
        ix=np.concatenate([groups[s][day] for day in draw]);y,a,b=prob[s];delta[s].append(seed_ba(y[ix],b[:,ix])-seed_ba(y[ix],a[:,ix]))
intervals={s:dict(delta_BA=seed_ba(prob[s][0],prob[s][2])-seed_ba(prob[s][0],prob[s][1]),day_CI=np.quantile(v,[.025,.975]).tolist()) for s,v in delta.items()}
intervals['equal_stock_mean_CI']=np.quantile(np.mean(list(delta.values()),axis=0),[.025,.975]).tolist()
changes={}
for s in ['AAPL','AMZN']:
    q=pred[pred.symbol.eq(s)];changes[s]={}
    for seed in [573,574,575]:
        a=q[f'frozen_{seed}'].to_numpy();b=q[f'lora_{seed}'].to_numpy();y=q.label.to_numpy()
        correct_a=(a>=.5)==y;correct_b=(b>=.5)==y
        changes[s][str(seed)]=dict(mean_absolute_probability_change=float(np.abs(a-b).mean()),max_absolute_probability_change=float(np.abs(a-b).max()),direction_changes=int(((a>=.5)!=(b>=.5)).sum()),fixed=int((~correct_a&correct_b).sum()),broken=int((correct_a&~correct_b).sum()))
historical=json.loads((O/'stock_review_fixes/before_sha256.json').read_text())
for p,h in historical.items():assert sha(O.parent/p)==h,p
(R/'audit.json').write_text(json.dumps(dict(checks=checks,intervals=intervals,prediction_changes=changes,interval_note='2000 paired-day draws, average of seed-specific BA; descriptive development comparison, not an ensemble and not corrected for repeated exploration.',test_evaluated=False),indent=2))
print(json.dumps(intervals,indent=2));print('PASS: all 2504 input bags/time bounds, training-only transforms/epochs, frozen base weights, adapter updates, six saved predictors and metrics verified; 204 historical artifacts unchanged. Final test not evaluated.')
