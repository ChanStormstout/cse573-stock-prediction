import json,time
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import B,O,FEATURES,prepared,baseline,fold_indices,metric_details,sharing_design,scores,require_empty,sha
D,M=prepared();P=json.loads((B/'protocol.json').read_text());R=B/'results/linear';require_empty(R);R.mkdir(exist_ok=True);(R/'started.json').write_text('{"status":"training"}')
result=dict(protocol=P,stocks={},sharing={},test_evaluated=False);started=time.time()
for s,d in D.groupby('symbol'):
    d=d.reset_index(drop=True);tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));F=R/s;F.mkdir();pred=d.iloc[va][['symbol','start_utc','label','has_news']].copy();result['stocks'][s]={}
    for kind in P['baselines']:
        rows=[]
        for C in P['C_grid']:
            for month,a,b in fold_indices(d):
                fit=baseline(kind,C).fit(d.iloc[a],d.label.iloc[a]);p=fit.predict_proba(d.iloc[b])[:,1]
                rows.append(dict(C=C,month=month,n_train=len(a),n_valid=len(b),**scores(d.label.iloc[b],p)))
        grid=pd.DataFrame(rows);best=grid.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0]
        fit=baseline(kind,float(best.C)).fit(d.iloc[tr],d.label.iloc[tr]);p=fit.predict_proba(d.iloc[va])[:,1];pred[kind]=p
        joblib.dump(fit,F/f'{kind}.joblib');grid.to_csv(F/f'{kind}_grid.csv',index=False)
        result['stocks'][s][kind]=dict(C=float(best.C),training_cv_BA=float(best.balanced_accuracy),training=scores(d.label.iloc[tr],fit.predict_proba(d.iloc[tr])[:,1]),validation=metric_details(d.iloc[va],p))
        print(s,kind,'CV',round(best.balanced_accuracy,4),'VAL',round(scores(d.label.iloc[va],p)['balanced_accuracy'],4),flush=True)
    pred.to_csv(F/'validation_predictions.csv',index=False)
    (R/'results.json').write_text(json.dumps(result,indent=2))
# Joint models: same scaling, stock intercepts and three-C budget for all modes.
tr=np.flatnonzero(D.split.eq('train'));va=np.flatnonzero(D.split.eq('validation'));joint=D.iloc[va][['symbol','start_utc','label','has_news']].copy()
for mode in ['separate','shared','partial']:
    grid=[]
    for C in P['C_grid']:
        for month,a,b in fold_indices(D):
            sc=StandardScaler().fit(D.iloc[a][FEATURES]);xa=sharing_design(sc.transform(D.iloc[a][FEATURES]),D.stock_index.iloc[a],mode);xb=sharing_design(sc.transform(D.iloc[b][FEATURES]),D.stock_index.iloc[b],mode)
            fit=LogisticRegression(C=C,l1_ratio=0,fit_intercept=False,solver='liblinear',max_iter=3000,random_state=573).fit(xa,D.label.iloc[a]);p=fit.predict_proba(xb)[:,1]
            for s in ['AAPL','AMZN']:
                mask=D.symbol.iloc[b].eq(s).to_numpy();grid.append(dict(C=C,month=month,symbol=s,**scores(D.label.iloc[b].to_numpy()[mask],p[mask])))
    g=pd.DataFrame(grid);best=g.groupby('C')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','C'],ascending=[False,True,False]).iloc[0]
    sc=StandardScaler().fit(D.iloc[tr][FEATURES]);xa=sharing_design(sc.transform(D.iloc[tr][FEATURES]),D.stock_index.iloc[tr],mode);xb=sharing_design(sc.transform(D.iloc[va][FEATURES]),D.stock_index.iloc[va],mode)
    fit=LogisticRegression(C=float(best.C),l1_ratio=0,fit_intercept=False,solver='liblinear',max_iter=3000,random_state=573).fit(xa,D.label.iloc[tr]);p=fit.predict_proba(xb)[:,1];joint[mode]=p
    joblib.dump(dict(scaler=sc,model=fit,mode=mode,features=FEATURES),R/f'sharing_{mode}.joblib');g.to_csv(R/f'sharing_{mode}_grid.csv',index=False)
    result['sharing'][mode]=dict(C=float(best.C),training_cv_BA=float(best.balanced_accuracy),stocks={s:metric_details(D.iloc[va].loc[mask],p[mask.to_numpy()]) for s in ['AAPL','AMZN'] if (mask:=D.symbol.iloc[va].eq(s)).any()})
    print('SHARING',mode,'CV',round(best.balanced_accuracy,4),'VAL',[round(result['sharing'][mode]['stocks'][s]['overall']['balanced_accuracy'],4) for s in ['AAPL','AMZN']],flush=True)
joint.to_csv(R/'sharing_predictions.csv',index=False);result['elapsed_seconds']=time.time()-started;result['prepared_sha256']=sha(B/'results/prepared.json');(R/'results.json').write_text(json.dumps(result,indent=2));print('Linear finished',round(result['elapsed_seconds']),'seconds',flush=True)
