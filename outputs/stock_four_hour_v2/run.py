"""Frozen protocol: nested monthly price-context experiment and matched horizons."""
import argparse,json,time,hashlib,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss,roc_auc_score
from config import Config
from features_price import load_sources,build_price_snapshot,VALUES,MASKS
ROOT=Path(__file__).resolve().parents[2]
BASE=[f'{v}_{i}' for i in range(1,7) for v in ['return','range']]+['history_age_hours','return_mean','return_std']+['at1030','at1130']
META=MASKS+['log_quote_age']
ARMS={'O':BASE,'M':BASE+META,'S':BASE+['stale_'+x for x in VALUES]+META,'R':BASE+VALUES+META}
LAM=[.2,.02,.002]
def metrics(y,p,threshold=.5):
 y=np.asarray(y);p=np.asarray(p);q=p>=threshold
 return dict(BA=balanced_accuracy_score(y,q),MCC=matthews_corrcoef(y,q),Brier=brier_score_loss(y,p),AUC=roc_auc_score(y,p) if len(set(y))==2 else None,up_recall=float(q[y==1].mean()) if (y==1).any() else None,down_recall=float((~q[y==0]).mean()) if (y==0).any() else None,pred_up=float(q.mean()),n=len(y),coverage=1.,constant=float(len(set(q))==1))
def prefix(d,month):
 ev=d[d.month==month];tr=d[d.month<month]
 if ev.empty or tr.empty:raise ValueError('Missing fold')
 if not (tr.end_utc.max()<ev.cutoff_utc.min()):raise ValueError('Future training label')
 return tr,ev

def main(out,lag='0min',main_only=False):
 cfg=Config(availability_lag=lag)
 start=time.monotonic();out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
 source=ROOT/'outputs/stock_horizons/runs/v1/data.pkl';d=pd.read_pickle(source);d['month']=d.day.str[:7];d['key']=d.symbol+'|'+d.start_utc.astype(str)
 four=d[d.horizon=='4h'].copy();assert len(four)==1607 and not four.key.duplicated().any()
 rec=[];provenance=[]
 for sym,g in four.groupby('symbol'):
  b,c=load_sources(ROOT/'work/stock-data',sym)
  for row in g.itertuples():
   f=build_price_snapshot(sym,row.start_utc,row.cutoff_utc,b,c,availability_lag=lag);s=build_price_snapshot(sym,row.start_utc,row.cutoff_utc,b,c,asof_end=row.history_end,availability_lag=lag)
   rec.append({'key':row.key,**{k:f.values[k] for k in VALUES},**f.masks,'log_quote_age':np.log1p(f.values['quote_age_hours']*60),**{'stale_'+k:s.values[k] for k in VALUES}})
   provenance.append({'key':row.key,'fresh':f.provenance,'stale':s.provenance})
 feature=pd.DataFrame(rec);feature.to_csv(out/'new_features.csv',index=False)
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 keys=set(four.key)&set(d[d.horizon=='1h'].key)
 datasets={'main4h':four,'matched4h':four[four.key.isin(keys)],'matched1h':d[(d.horizon=='1h')&d.key.isin(keys)]}
 if main_only:datasets={'main4h':four}
 (out/'config.json').write_text(json.dumps(cfg.__dict__,indent=2))
 fits=[];preds=[];selection=[];count=0;cache={}
 def fit(tr,ev,cols,lam,name):
  nonlocal count
  count+=1
  if count>500 or time.monotonic()-start>7200:raise RuntimeError('Budget exceeded')
  assert tr.end_utc.max()<ev.cutoff_utc.min()
  model=make_pipeline(StandardScaler(),LogisticRegression(solver='liblinear',C=1/(lam*len(tr)),random_state=573,max_iter=3000,tol=1e-9))
  model.fit(tr[cols],tr.label);p=model.predict_proba(ev[cols])[:,1]
  path=out/'models'/f'{name}.joblib';joblib.dump(model,path)
  assert np.array_equal(p,joblib.load(path).predict_proba(ev[cols])[:,1])
  fits.append(dict(name=name,lambda_=lam,C=1/(lam*len(tr)),W=len(tr),features=cols,train_keys=tr.key.tolist(),eval_keys=ev.key.tolist(),label_end_max=str(tr.end_utc.max()),fit_cutoff=str(ev.cutoff_utc.min()),train=metrics(tr.label,model.predict_proba(tr[cols])[:,1]),evaluation=metrics(ev.label,p),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),iterations=int(model[-1].n_iter_.max())))
  return p
 for ds,frame in datasets.items():
  frame=frame.merge(feature,on='key',validate='one_to_one');frame['at1030']=(frame.ny_hour==10.5).astype(int);frame['at1130']=(frame.ny_hour==11.5).astype(int)
  frame.to_pickle(out/f'{ds}.pkl')
  for sym,g in frame.groupby('symbol'):
   for m in pd.period_range('2018-03','2018-08',freq='M').astype(str):
    tr,ev=prefix(g,m)
    for lam in LAM:
     p=fit(tr,ev,BASE,lam,f'{ds}_{sym}_inner_{m}_{lam}');cache[(ds,sym,m,lam)]=metrics(ev.label,p)
   for m in ['2018-06','2018-07','2018-08','2018-09']:
    months=[x for x in pd.period_range('2018-03','2018-08',freq='M').astype(str) if x<m]
    ranked=sorted(LAM,key=lambda l:(-np.mean([cache[ds,sym,x,l]['BA'] for x in months]),np.mean([cache[ds,sym,x,l]['Brier'] for x in months]),-l));lam=ranked[0]
    selection.append(dict(dataset=ds,symbol=sym,fit_month=m,inner_months=months,lambda_=lam,candidates={str(l):{metric:float(np.mean([cache[ds,sym,x,l][metric] for x in months])) for metric in ['BA','Brier']} for l in LAM}))
    tr,ev=prefix(g,m)
    if m=='2018-09':ev=g[g.month>=m]
    for arm in (['O','M','S','R'] if ds=='main4h' else ['O','R']):
     p=fit(tr,ev,ARMS[arm],lam,f'{ds}_{sym}_{m}_{arm}');q=ev[['key','symbol','day','month','ny_hour','label','split']].copy();q['p']=p;q['arm']=arm;q['dataset']=ds;q['phase']='outer' if m<'2018-09' else 'frozen_replay';q['train_prior']=tr.label.mean();preds.append(q)
  print('Completed',ds,'fits',count,flush=True)
 p=pd.concat(preds,ignore_index=True);p.to_csv(out/'predictions.csv',index=False)
 (out/'fits.json').write_text(json.dumps(fits,indent=2));(out/'selection.json').write_text(json.dumps(selection,indent=2))
 rows=[]
 for groupcols in [['dataset','symbol','phase','arm'],['dataset','symbol','phase','arm','month'],['dataset','symbol','phase','arm','ny_hour']]:
  for key,g in p.groupby(groupcols):
   for threshold in ['0.5','train_prior']:
    rows.append({**dict(zip(groupcols,key)),'threshold':threshold,**metrics(g.label,g.p,.5 if threshold=='0.5' else g.train_prior)})
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
 gates=[]
 for sym,g in p[(p.dataset=='main4h')&(p.phase=='outer')].groupby('symbol'):
  scores={(a,m):metrics(x.label,x.p) for (a,m),x in g.groupby(['arm','month'])};months=sorted(g.month.unique())
  mean=lambda a,k:np.mean([scores[a,m][k] for m in months])
  diff=[scores['R',m]['BA']-scores['O',m]['BA'] for m in months]
  gates.append(dict(symbol=sym,R_BA=mean('R','BA'),delta_BA=mean('R','BA')-mean('O','BA'),delta_MCC=mean('R','MCC')-mean('O','MCC'),delta_Brier=mean('R','Brier')-mean('O','Brier'),nonnegative_months=sum(x>=0 for x in diff),pass_gate=bool(mean('R','BA')>.5 and np.mean(diff)>=.01 and sum(x>=0 for x in diff)>=2 and mean('R','MCC')>=mean('O','MCC') and mean('R','Brier')-mean('O','Brier')<=.002),R_minus_M=mean('R','BA')-mean('M','BA'),R_minus_S=mean('R','BA')-mean('S','BA')))
 (out/'gates.json').write_text(json.dumps(gates,indent=2))
 (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE',fits=count,elapsed_seconds=time.monotonic()-start,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),code_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')},interpretation='Exploratory historical replay; all periods previously exposed'),indent=2))
 print(json.dumps(gates,indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lag',choices=['0min','5min'],default='0min');parser.add_argument('--main-only',action='store_true');args=parser.parse_args();main(args.output,args.lag,args.main_only)
