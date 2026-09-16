import json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import prepared,verify_files,scores
D,_=prepared();R=B/'results';result=json.loads((R/'results.json').read_text());verify_files(O,result['source_sha256']);assert result['test_evaluated'] is False
for s,dd in D.groupby('symbol'):
    d=dd.reset_index(drop=True);tr=d.split.eq('train');va=d.split.eq('validation');pred=pd.read_csv(R/s/'validation_predictions.csv')
    assert pred.start_utc.tolist()==d.loc[va,'start_utc'].tolist()
    for kind,spec in result['stocks'][s].items():
        fit=joblib.load(R/s/f'{kind}.joblib');assert fit['model'].l1_ratio==0 and fit['model'].C==spec['C']
        g=pd.read_csv(R/s/f'{kind}_grid.csv');best=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0];assert best.C==spec['C']
        old=joblib.load(O/f'stock_robust/results/linear/{s}/{kind}.joblib')
        # Deterministic supervised feature selection must be identical to L1;
        # only the fitted classifier and selected regularization may differ.
        np.testing.assert_array_equal(fit['features'].get_feature_names_out(),old['features'].get_feature_names_out())
        columns=fit['features'].transformers_[0][2];np.testing.assert_allclose(fit['features'].named_transformers_['numeric'].mean_,d.loc[tr,columns].mean())
        p=fit.predict_proba(d.loc[va])[:,1];np.testing.assert_allclose(p,pred[kind])
        for key,v in scores(pred.label,p).items():assert abs(v-spec['validation']['overall'][key])<1e-10
days=sorted(D.loc[D.split.eq('validation'),'start_utc'].str[:10].unique());intervals={}
for kind in result['protocol']['representations']:
    data={};groups={};delta={s:[] for s in ['AAPL','AMZN']}
    for s in delta:
        new=pd.read_csv(R/s/'validation_predictions.csv');old=pd.read_csv(O/f'stock_robust/results/linear/{s}/validation_predictions.csv');assert new.start_utc.tolist()==old.start_utc.tolist();data[s]=(new.label.to_numpy(),old[kind].to_numpy(),new[kind].to_numpy());groups[s]={day:np.flatnonzero(new.start_utc.str[:10].eq(day)) for day in days}
    def ba(y,p):return .5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean())
    rng=np.random.default_rng(582)
    for _ in range(2000):
        draw=rng.choice(days,len(days),replace=True)
        for s in delta:
            ix=np.concatenate([groups[s][day] for day in draw]);y,a,b=data[s];delta[s].append(ba(y[ix],b[ix])-ba(y[ix],a[ix]))
    intervals[kind]={s:dict(delta_BA=ba(data[s][0],data[s][2])-ba(data[s][0],data[s][1]),day_CI=np.quantile(v,[.025,.975]).tolist()) for s,v in delta.items()};intervals[kind]['equal_stock_mean_CI']=np.quantile(np.mean(list(delta.values()),axis=0),[.025,.975]).tolist()
(R/'descriptive_intervals.json').write_text(json.dumps(intervals,indent=2));print(json.dumps(intervals,indent=2));print('PASS: identical feature selection, training-only scalers/C selection, all saved L2 predictions and metrics reproduced. Final test not evaluated.')
