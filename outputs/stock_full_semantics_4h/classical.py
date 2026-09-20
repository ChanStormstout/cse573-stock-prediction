"""Finite nested text-only comparisons on the unchanged random partitions."""
import sys,json,time,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import load,check_sources,PRIVATE as BASE,sha,dump,metric
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
DEST=HERE.parents[1]/'work/stock-data/full_semantics_4h/v1/classical'
OUT=HERE/'v1'
CAND={'TFIDF_LR':[.01,.1,1.], 'TFIDF_SVM':[.01,.1,1.], 'TFIDF_RF':[3,6]}
def model(method,value,seed):
 def text():return TfidfVectorizer(min_df=3,max_features=5000,sublinear_tf=True)
 if method=='TFIDF_LR':return make_pipeline(text(),LogisticRegression(C=value,solver='liblinear',max_iter=3000,tol=1e-7,random_state=seed))
 if method=='TFIDF_RF':return make_pipeline(text(),RandomForestClassifier(n_estimators=200,max_depth=value,min_samples_leaf=10,max_features='sqrt',random_state=seed,n_jobs=2))
 # Each calibration split fits its own vocabulary, IDF and decision model.
 return CalibratedClassifierCV(make_pipeline(text(),LinearSVC(C=value,max_iter=10000,random_state=seed)),method='sigmoid',cv=StratifiedKFold(3,shuffle=True,random_state=seed),ensemble=True)
def main():
 check_sources();d,*_=load();outer=pd.read_csv(BASE/'outer_folds.csv');inner=json.loads((BASE/'inner_folds.json').read_text())
 DEST.mkdir(parents=True,exist_ok=True)
 seal={'code':sha(__file__),'protocol':sha(OUT/'protocol.json'),'source_manifest':sha(BASE/'manifest.json')}
 p=DEST/'seal.json'
 if p.exists():assert json.loads(p.read_text())==seal
 else:dump(p,seal)
 start=time.monotonic()
 with threadpool_limits(2),warnings.catch_warnings():
  warnings.simplefilter('ignore',FutureWarning)
  for seed in [573,574,575]:
   for stock in ['AAPL','AMZN']:
    for fold in range(10):
     root=DEST/f'{seed}_{stock}_{fold}'
     if (root/'complete.json').exists():
      saved=json.loads((root/'complete.json').read_text());assert saved['seal']==sha(p)
      for n,h in saved['files'].items():assert sha(root/n)==h
      continue
     root.mkdir(exist_ok=False)
     ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy(int)
     tr=np.array(sorted(set(np.flatnonzero(d.symbol==stock))-set(ev)))
     splits=[s for s in inner if s['seed']==seed and s['symbol']==stock and s['fold']==fold]
     records=[];ledger=[];pred=d.iloc[ev][['row_id','symbol','day','label']].copy();pred['seed']=seed;pred['fold']=fold
     for method,candidates in CAND.items():
      for value in candidates:
       for s in splits:
        a=s['train'];b=s['validation'];assert not(set(a)&set(b)) and not(set(a)&set(ev))
        m=model(method,value,seed);m.fit(d.iloc[a].stem_body.fillna(''),d.iloc[a].label)
        prob=m.predict_proba(d.iloc[b].stem_body.fillna(''))[:,1]
        records.append(dict(method=method,value=value,inner_fold=s['inner_fold'],**metric(d.iloc[b].label,prob)))
      rs=[r for r in records if r['method']==method]
      chosen=min(candidates,key=lambda v:(-np.mean([r['BA'] for r in rs if r['value']==v]),np.mean([r['Brier'] for r in rs if r['value']==v]),candidates.index(v)))
      m=model(method,chosen,seed);m.fit(d.iloc[tr].stem_body.fillna(''),d.iloc[tr].label)
      prob=m.predict_proba(d.iloc[ev].stem_body.fillna(''))[:,1]
      f=root/(method+'.joblib');joblib.dump(m,f,compress=3)
      replay=joblib.load(f).predict_proba(d.iloc[ev].stem_body.fillna(''))[:,1];assert np.max(abs(prob-replay))<1e-12
      pred[method]=prob;ledger.append(dict(method=method,selected=chosen,train=tr.tolist(),evaluation=ev.tolist(),model_sha256=sha(f)))
     pd.DataFrame(records).to_csv(root/'inner_metrics.csv',index=False);pred.to_csv(root/'predictions.csv',index=False,float_format='%.17g');dump(root/'models.json',ledger)
     dump(root/'complete.json',{'seal':sha(p),'estimator_fit_calls':27,'files':{q.name:sha(q) for q in root.iterdir() if q.is_file()}})
     print(seed,stock,fold,'completed',round(time.monotonic()-start,1),flush=True)
 dump(OUT/'CLASSICAL_EXECUTION.json',{'status':'COMPLETE_PENDING_INDEPENDENT_VERIFICATION','blocks':60,'top_level_estimator_fit_calls':1620,'selected_models':180,'note':'SVM calibration includes three internally fitted text pipelines per estimator call; not 1620 unique independent samples'})
if __name__=='__main__':main()
