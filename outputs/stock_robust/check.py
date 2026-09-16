"""Check saved fits, selection, train-only transforms and paired-day effects."""
import sys,json
import numpy as np,pandas as pd,torch,joblib
from common import B,O,prepared,FEATURES,sharing_design,metric_details,scores,sha
sys.path.insert(0,str(O/'stock_temporal'));from models import TemporalNet,fit_scaler,transform,predict
D,M=prepared();R=B/'results';L=json.loads((R/'linear/results.json').read_text());T=json.loads((R/'temporal/results.json').read_text());P=L['protocol'];N=P['temporal'];torch.set_num_threads(2)
preds={};probs={};logs=[]
for s,d in D.groupby('symbol'):
    d=d.reset_index(drop=True);tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));a=pd.read_csv(R/f'linear/{s}/validation_predictions.csv');t=pd.read_csv(R/f'temporal/{s}/validation_predictions.csv');np.testing.assert_array_equal(a.label,d.label.iloc[va]);assert a.start_utc.tolist()==t.start_utc.tolist();preds[s]=a;probs[s]={}
    for kind in P['baselines']:
        fit=joblib.load(R/f'linear/{s}/{kind}.joblib');p=fit.predict_proba(d.iloc[va])[:,1];np.testing.assert_allclose(p,a[kind]);probs[s][kind]=p[None,:]
        for k,v in scores(a.label,p).items():assert abs(v-L['stocks'][s][kind]['validation']['overall'][k])<1e-10
        g=pd.read_csv(R/f'linear/{s}/{kind}_grid.csv');chosen=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];assert chosen.C==L['stocks'][s][kind]['C']
        if 'numeric' in fit['features'].named_transformers_:
            cols=fit['features'].transformers_[0][2];np.testing.assert_allclose(fit['features'].named_transformers_['numeric'].mean_,d.iloc[tr][cols].mean())
    old=pd.read_csv(O/f'stock_improvement/results/{s}/validation_predictions.csv');np.testing.assert_allclose(a.price,old.price);np.testing.assert_allclose(a.price_finbert,old.price_sentiment)
    z=np.load(O/f'stock_temporal/results/{s}/data.npz');X=z['X'];y=z['y'];length=z['lengths'];old_seq=pd.read_csv(O/f'stock_temporal/results/{s}/validation_predictions.csv');probs[s]['old_gru']=old_seq[['gru_573','gru_574','gru_575']].to_numpy().T;probs[s]['sequence_lr']=old_seq[['sequence_lr']].to_numpy().T
    for kind in N['variants']:
        spec=T['stocks'][s][kind];sc=joblib.load(R/f'temporal/{s}/{kind}_scaler.joblib');np.testing.assert_allclose(sc.mean_,fit_scaler(X[tr]).mean_);bb=transform(X[va],sc);vv=[]
        g=pd.read_csv(R/f'temporal/{s}/{kind}_grid.csv');best=g.groupby('weight_decay')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','weight_decay'],ascending=[False,True,False]).iloc[0];assert best.weight_decay==spec['weight_decay']
        assert (pd.to_datetime(g.inner_train_end,utc=True)<pd.to_datetime(g.inner_validation_start,utc=True)).all();assert (pd.to_datetime(g.inner_validation_start,utc=True)<pd.to_datetime(g.outer_validation_start,utc=True)).all()
        for seed in N['seeds']:
            net=TemporalNet(kind.split('_')[0],hidden=spec['hidden']);net.load_state_dict(torch.load(R/f'temporal/{s}/{kind}_{seed}.pt',weights_only=True));p=predict(net,bb,length[va]);np.testing.assert_allclose(p,t[f'{kind}_{seed}'],atol=1e-7);vv.append(p)
            run=spec['seeds'][str(seed)];best_loss=float('inf');epoch=0
            for row in run['early_stop_trace']:
                if row['inner_bce']<best_loss-N['min_delta']:best_loss=row['inner_bce'];epoch=row['epoch']
            assert epoch==run['epochs']
            for k,v in scores(a.label,p).items():assert abs(v-run['validation']['overall'][k])<1e-7
        probs[s][kind]=np.stack(vv)
joint=pd.read_csv(R/'linear/sharing_predictions.csv');tr=np.flatnonzero(D.split.eq('train'));va=np.flatnonzero(D.split.eq('validation'))
for mode in ['separate','shared','partial']:
    bundle=joblib.load(R/f'linear/sharing_{mode}.joblib');sc=bundle['scaler'];np.testing.assert_allclose(sc.mean_,D.iloc[tr][FEATURES].mean());x=sharing_design(sc.transform(D.iloc[va][FEATURES]),D.stock_index.iloc[va],mode);p=bundle['model'].predict_proba(x)[:,1];np.testing.assert_allclose(p,joint[mode]);g=pd.read_csv(R/f'linear/sharing_{mode}_grid.csv');best=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];assert best.C==L['sharing'][mode]['C']
    for s in ['AAPL','AMZN']:probs[s]['sharing_'+mode]=p[joint.symbol.eq(s)][None,:]


def ba(y,p):return float(np.mean([.5*((q[y==1]>=.5).mean()+(q[y==0]<.5).mean()) for q in p]))


contrasts=[('paper_stem_body','paper_stem_title'),('sharing_shared','sharing_separate'),('sharing_partial','sharing_separate'),('gru_early_h8','old_gru'),('gru_early_h8','sequence_lr')]
days=sorted(set(preds['AAPL'].start_utc.str[:10]));groups={s:{day:np.flatnonzero(preds[s].start_utc.str[:10].eq(day)) for day in days} for s in preds};intervals={}
for new,old in contrasts:
    rng=np.random.default_rng(579);delta={s:[] for s in preds}
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True)
        for s in preds:
            ix=np.concatenate([groups[s][day] for day in draw]);y=preds[s].label.to_numpy()[ix];delta[s].append(ba(y,probs[s][new][:,ix])-ba(y,probs[s][old][:,ix]))
    intervals[new+' minus '+old]={s:dict(delta_BA=ba(preds[s].label.to_numpy(),probs[s][new])-ba(preds[s].label.to_numpy(),probs[s][old]),descriptive_day_CI=np.quantile(v,[.025,.975]).tolist()) for s,v in delta.items()}
    intervals[new+' minus '+old]['equal_stock_mean_CI']=np.quantile(np.mean(list(delta.values()),axis=0),[.025,.975]).tolist()
(R/'descriptive_intervals.json').write_text(json.dumps(intervals,indent=2))
for k,v in intervals.items():print(k,json.dumps(v))
historical=json.loads((O/'stock_review_fixes/before_sha256.json').read_text())
for p,h in historical.items():assert sha(O.parent/p)==h,p
print('PASS: all historical artifacts unchanged; all saved E09 predictions/metrics reproduce; selected configurations and early-stop epochs match training-only records; scaler means match training; baseline parity; 2000 paired-day descriptive intervals. Final test not evaluated.')
