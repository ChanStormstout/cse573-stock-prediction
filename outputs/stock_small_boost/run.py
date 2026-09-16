"""E15: shallow boosting vs matched LR, fixed monthly expanding policy."""
from pathlib import Path
import sys,json,datetime,time
import pandas as pd,numpy as np,joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import *
def factory(kind,features,config):
    model=LogisticRegression(C=config,solver='liblinear',l1_ratio=0,max_iter=3000,random_state=573) if kind=='LR' else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=4,max_depth=2,min_samples_leaf=30,learning_rate=.05,l2_regularization=config,early_stopping=False,random_state=573)
    return Pipeline([('features',ColumnTransformer([('numeric',StandardScaler(),features)])),('model',model)])
if __name__=='__main__':
    R=B/'results';require_empty(R);R.mkdir(exist_ok=True);paths=[B/'run.py',O/'stock_adaptive/common.py',O/'stock_robust/results/data.pkl',O/'stock_final_test/results/test_inputs.pkl'];P=dict(id='E15',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='bounded low-capacity nonlinear control, no neural expansion',policy='monthly expanding from January; same past data, features and three-candidate search per model',features={'price':PRICE,'price_count':PRICE+['log_news_count','has_news']},grid={'LR':[.01,.1,1.],'boost':[1.,10.,100.]},tree='100 iterations, depth2, <=4 leaves, >=30 samples per leaf; lr .05; no random early stopping',selection='last three completed months within past training; mean BA then Brier then larger candidate',evaluation='post-E12 historical replay; no independent holdout',source_sha256={str(p.relative_to(O.parent)):sha(p) for p in paths});(B/'protocol.json').write_text(json.dumps(P,indent=2));D=load_data();D['log_news_count']=np.log1p(D.news_count);preds=[];records=[];start=time.time()
    for s,sd in D.groupby('symbol'):
        for feature,cols in P['features'].items():
            for kind in ['LR','boost']:
                for month in ['2018-09','2018-10','2018-11','2018-12','2019-01','2019-02']:
                    tr=train_rows(sd,month,'expanding');te=sd[sd.start_utc.dt.strftime('%Y-%m').eq(month)];name=f'{s}_{feature}_{kind}_{month}';f=R/name;f.mkdir();make=lambda c:factory(kind,cols,c);config,grid=select(tr,month,'expanding',make,P['grid'][kind]);model=make(config).fit(tr,tr.label);p=model.predict_proba(te)[:,1];joblib.dump(model,f/'model.joblib');grid.to_csv(f/'cv.csv',index=False);assert tr.end_utc.max()<te.cutoff_utc.min()
                    r=dict(symbol=s,feature=feature,kind=kind,month=month,config=config,train_keys=row_keys(tr),test_keys=row_keys(te),training=metrics(tr,model.predict_proba(tr)[:,1]),evaluation=metrics(te,p));(f/'manifest.json').write_text(json.dumps(r,indent=2));records.append(r);q=te[['symbol','start_utc','label','split','has_news','news_count']].copy();q['feature']=feature;q['kind']=kind;q['month']=month;q['p']=p;preds.append(q);print(name,round(r['evaluation']['balanced_accuracy'],4),flush=True)
    pd.concat(preds,ignore_index=True).to_csv(R/'predictions.csv',index=False);verify_files(O.parent,P['source_sha256']);(R/'results.json').write_text(json.dumps(dict(models=records,seconds=time.time()-start),indent=2));(B/'artifact_sha256.json').write_text(json.dumps(snapshot(B),indent=2));print('E15 complete',round(time.time()-start),flush=True)
