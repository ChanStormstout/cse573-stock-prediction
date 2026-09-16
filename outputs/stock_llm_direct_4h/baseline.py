"""Past-only C selection; LR controls use the same source pack as the LLM."""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss
from prepare import PRICE,NEWSNUM
from common import B,sha,dump

def metric(y,p):
 y=np.array(y);p=np.array(p);z=p>=.5
 return dict(n=len(y),BA=float(balanced_accuracy_score(y,z)) if len(set(y))==2 else None,MCC=float(matthews_corrcoef(y,z)),Brier=float(brier_score_loss(y,p)),pred_up=float(z.mean()),constant=bool(len(set(z))==1),up_rate=float(y.mean()))
def estimator(v,c):
 num=(PRICE if v!='news' else [])+(NEWSNUM if v!='price' else [])
 parts=[('numeric',StandardScaler(),num)]
 if v!='price':parts.append(('text',TfidfVectorizer(max_features=500,min_df=2,ngram_range=(1,2),sublinear_tf=True),'packed_text'))
 return Pipeline([('features',ColumnTransformer(parts)),('model',LogisticRegression(C=c,solver='liblinear',max_iter=3000,random_state=573))])
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 seal=json.loads((a.inputs/'manifest.json').read_text());assert sha(a.inputs/'features.pkl')==seal['features_sha'];d=pd.read_pickle(a.inputs/'features.pkl');cv=[];pred=[];fits=[];selected=[];t=time.time()
 for sym,s in d.groupby('symbol'):
  for v in ['price','news','joint']:
   grid=[]
   for c in [.01,.1,1.]:
    for mo in [6,7,8]:
     lo=pd.Timestamp(f'2018-{mo:02d}-01',tz='UTC');ev=s[(s.start_utc>=lo)&(s.start_utc<lo+pd.DateOffset(months=1))];tr=s[s.end_utc<ev.cutoff_utc.min()];assert tr.end_utc.max()<ev.cutoff_utc.min()
     m=estimator(v,c).fit(tr,tr.label);z=metric(ev.label,m.predict_proba(ev)[:,1]);r=dict(symbol=sym,variant=v,C=c,month=mo,**z);grid.append(r);cv.append(r);fits.append(dict(symbol=sym,variant=v,C=c,month=mo,train_keys=tr.key.tolist(),eval_keys=ev.key.tolist(),max_label_end=str(tr.end_utc.max()),cutoff=str(ev.cutoff_utc.min())))
   ranks=pd.DataFrame(grid).groupby('C')[['BA','Brier']].mean().reset_index().sort_values(['BA','Brier','C'],ascending=[False,True,True]);c=float(ranks.iloc[0].C)
   ev=s[s.split!='train'];tr=s[(s.split=='train')&(s.end_utc<ev.cutoff_utc.min())];m=estimator(v,c).fit(tr,tr.label);p1=m.predict_proba(ev)[:,1];path=a.out/f'{sym}_{v}.joblib';joblib.dump(m,path);p2=joblib.load(path).predict_proba(ev)[:,1];np.testing.assert_allclose(p1,p2,atol=1e-12)
   z=ev[['key','symbol','split','day','month','label','has_news']].copy();z['method']='matched_'+v;z['p']=p1;pred.append(z);selected.append(dict(symbol=sym,variant=v,C=c,cv_BA=float(ranks.iloc[0].BA),model_sha=sha(path),reload_max_error=float(np.max(np.abs(p1-p2)))))
   fits.append(dict(symbol=sym,variant=v,C=c,month='final',train_keys=tr.key.tolist(),eval_keys=ev.key.tolist(),max_label_end=str(tr.end_utc.max()),cutoff=str(ev.cutoff_utc.min())));print(sym,v,c,flush=True)
 pd.concat(pred).to_csv(a.out/'predictions.csv',index=False);pd.DataFrame(cv).to_csv(a.out/'cv.csv',index=False);dump(a.out/'fits.json',fits);dump(a.out/'summary.json',dict(completed=True,actual_fits=len(fits),seconds=time.time()-t,selected=selected,input_sha=seal['features_sha'],code={f:sha(B/f) for f in ['baseline.py','prepare.py','common.py','PROTOCOL.md']}))
if __name__=='__main__':main()
