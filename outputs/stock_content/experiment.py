"""Shared evaluation: fit every transformation within training folds, never test."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
OLD=json.loads((B.parent/'stock_baseline/results/results.json').read_text())
def fit_variant(d,symbol,name,family,extra,folder):
 price=OLD['stocks'][symbol]['features'];num=price+extra
 def make(C):
  transforms=[('numeric',StandardScaler(),num)]
  if family=='sparse':transforms.append(('words',make_pipeline(CountVectorizer(binary=True,ngram_range=(1,1),min_df=3,stop_words='english',max_features=10000),SelectKBest(chi2,k=500)),'text'))
  return Pipeline([('features',ColumnTransformer(transforms)),('model',LogisticRegression(C=C,l1_ratio=1 if family=='sparse' else 0,solver='liblinear',max_iter=3000,random_state=573))])
 tr=d[d.split.eq('train')];va=d[d.split.eq('validation')];dates=pd.to_datetime(tr.start_utc,utc=True);grid=[];candidates=[]
 for C in [.01,.1,1]:
  ms=[]
  for month in [6,7,8]:
   start=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC');end=start+pd.offsets.MonthBegin();a=tr[dates<start];b=tr[(dates>=start)&(dates<end)]
   assert pd.to_datetime(a.end_utc,utc=True).max()<pd.to_datetime(b.cutoff_utc,utc=True).min()
   fit=make(C).fit(a,a.label);m=scores(b.label,fit.predict_proba(b)[:,1]);ms.append(m);grid.append(dict(variant=name,C=C,month=month,n_train=len(a),n_valid=len(b),**m))
  candidates.append((np.mean([m['balanced_accuracy'] for m in ms]),-np.mean([m['brier'] for m in ms]),C))
 cv,_,C=max(candidates);fit=make(C).fit(tr,tr.label);p=fit.predict_proba(va)[:,1];met=scores(va.label,p)
 folder.mkdir(exist_ok=True,parents=True);joblib.dump(fit,folder/f'{name}.joblib');pd.DataFrame(grid).to_csv(folder/f'{name}_training_cv.csv',index=False)
 pred=va[['start_utc','label']].copy();pred['p_up']=p;pred.to_csv(folder/f'{name}_validation_predictions.csv',index=False)
 np.testing.assert_allclose(fit['features'].named_transformers_['numeric'].mean_,tr[num].mean().to_numpy())
 d.to_pickle(folder/f'{name}_development.pkl')
 out=dict(C=C,training_cv_BA=cv,validation=met,monthly={mo:scores(va.loc[mask,'label'],p[mask.to_numpy()]) for mo in ['2018-09','2018-10'] if (mask:=va.start_utc.str.startswith(mo)).any()},numeric_features=num)
 print(symbol,name,'C',C,'CV',round(cv,4),'validation',round(met['balanced_accuracy'],4),flush=True)
 return out
