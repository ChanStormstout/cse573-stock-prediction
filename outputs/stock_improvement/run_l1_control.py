from pathlib import Path
import sys,json,joblib
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
old=json.loads((B.parent/'stock_baseline/results/results.json').read_text());result=json.loads((B/'results/results.json').read_text());result['protocol_addendum']=json.loads((B/'protocol_l1_control.json').read_text())
for s in ['AAPL','AMZN']:
 folder=B/'results'/s;d=pd.read_csv(folder/'feature_manifest.csv');tr=d[d.split.eq('train')];va=d[d.split.eq('validation')];dates=pd.to_datetime(tr.start_utc,utc=True);cols=old['stocks'][s]['features'];grid=[];candidates=[]
 def make(C):return Pipeline([('features',ColumnTransformer([('numeric',StandardScaler(),cols)])),('model',LogisticRegression(C=C,l1_ratio=1,solver='liblinear',max_iter=3000,random_state=573))])
 for C in [.01,.1,1]:
  ss=[]
  for month in [6,7,8]:
   start=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC');end=start+pd.offsets.MonthBegin();train=tr[dates<start];valid=tr[(dates>=start)&(dates<end)];fit=make(C).fit(train,train.label);m=scores(valid.label,fit.predict_proba(valid)[:,1]);grid.append(dict(model='price_l1_control',C=C,fold_month=month,**m));ss.append(m)
  candidates.append((np.mean([m['balanced_accuracy'] for m in ss]),-np.mean([m['brier'] for m in ss]),C))
 ba,_,C=max(candidates);fit=make(C).fit(tr,tr.label);p=fit.predict_proba(va)[:,1];m=scores(va.label,p);result['stocks'][s]['models']['price_l1_control']={'C':C,'training_cv_ba':ba,'validation':m,'monthly':{month:scores(va.loc[mask,'label'],p[mask.to_numpy()]) for month in ['2018-09','2018-10'] if (mask:=va.start_utc.str.startswith(month)).any()}}
 pred=pd.read_csv(folder/'validation_predictions.csv');pred['price_l1_control']=p;pred.to_csv(folder/'validation_predictions.csv',index=False);joblib.dump(fit,folder/'price_l1_control.joblib')
 previous=pd.read_csv(folder/'training_cv_grid.csv');previous=previous[previous.model.ne('price_l1_control')];pd.concat([previous,pd.DataFrame(grid)]).to_csv(folder/'training_cv_grid.csv',index=False)
 wordmodel=joblib.load(folder/'paper_price_bow.joblib');names=wordmodel['features'].get_feature_names_out();coef=wordmodel['model'].coef_[0]
 pd.DataFrame({'feature':names,'coefficient':coef}).query('coefficient != 0').to_csv(folder/'paper_nonzero_weights.csv',index=False)
 print(s,'L1 control',C,ba,m)
(B/'results/results.json').write_text(json.dumps(result,indent=2))
