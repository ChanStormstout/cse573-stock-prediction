"""Reproduce nested calibration predictions without changing fitted artifacts."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.optimize import minimize_scalar
from scipy.special import expit
B=Path(__file__).resolve().parent;O=B.parent
sys.path.insert(0,str(O/'stock_robust'))
from common import prepared,verify_files,baseline,scores
D,_=prepared();R=B/'results';result=json.loads((R/'results.json').read_text())
verify_files(O,result['source_sha256']);assert result['test_evaluated'] is False
count=0
for s,dd in D.groupby('symbol'):
    d=dd.reset_index(drop=True);dates=pd.to_datetime(d.start_utc,utc=True)
    train=d.split.eq('train');val=d.split.eq('validation')
    pred=pd.read_csv(R/s/'validation_predictions.csv')
    assert pred.start_utc.tolist()==d.loc[val,'start_utc'].tolist()
    for kind,spec in result['stocks'][s].items():
        oof=pd.read_csv(R/s/f'{kind}_nested_oof.csv')
        grid=pd.read_csv(R/s/f'{kind}_inner_selection.csv')
        assert set(grid.outer_month)=={6,7,8}
        for month,g in grid.groupby('outer_month'):
            assert set(g.C)=={.01,.1,1.}
            assert set(g.inner_month)==set(range(month-3,month))
            best=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0]
            q=oof[oof.outer_month.eq(month)]
            assert q.C.eq(best.C).all()
            lo=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC');hi=lo+pd.offsets.MonthBegin()
            a=train&(dates<lo);b=train&(dates>=lo)&(dates<hi)
            assert pd.to_datetime(d.loc[a,'end_utc'],utc=True).max()<pd.to_datetime(d.loc[b,'cutoff_utc'],utc=True).min()
            assert q.start_utc.tolist()==d.loc[b,'start_utc'].tolist()
            fit=baseline(kind,best.C).fit(d.loc[a],d.loc[a,'label'])
            np.testing.assert_allclose(fit.decision_function(d.loc[b]),q.logit,atol=1e-9)
            count+=1
        y=oof.label.to_numpy();z=oof.logit.to_numpy()
        def loss(logT):
            v=z/np.exp(logT);return float(np.mean(np.logaddexp(0,v)-y*v))
        opt=minimize_scalar(loss,bounds=(np.log(.25),np.log(10.)),method='bounded',options={'xatol':1e-8})
        np.testing.assert_allclose(np.exp(opt.x),spec['temperature'],rtol=1e-6)
        fit=joblib.load(O/f'stock_robust/results/linear/{s}/{kind}.joblib')
        logit=fit.decision_function(d.loc[val]);p=expit(logit/spec['temperature'])
        np.testing.assert_allclose(p,pred[kind+'_temperature'])
        np.testing.assert_allclose(expit(logit),pred[kind+'_uncalibrated'])
        np.testing.assert_array_equal(p>=.5,pred[kind+'_uncalibrated'].ge(.5))
        for field,value in scores(d.loc[val,'label'],p).items():
            assert abs(value-spec['calibrated']['overall'][field])<1e-10
print(f'PASS: {count} nested out-of-fold models refit and logits reproduced; C selection, time boundaries, fitted temperatures, saved predictions and all metrics verified. Every direction decision unchanged; no final-test evaluation.')
