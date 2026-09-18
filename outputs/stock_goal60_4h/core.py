"""Past-only, actual-issued-probability selection shared by finite experiments."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/stock-data'; OUT=ROOT/'outputs/stock_goal60_4h/v1'; PRIVATE=W/'goal60_4h/v1'
OLD=[f'{v}_{i}' for i in range(1,7) for v in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour']
RECENT=[f'recent_{v}_{n}' for n in (5,15,30,60) for v in ('return','range','rv','missing')]+['overnight_gap','overnight_gap_missing','minutes_from_open','minutes_to_close']
MONTHS=[f'2018-{m:02d}' for m in range(3,9)]+['final']; CS=[.01,.1,1.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,default=lambda a:a.item() if hasattr(a,'item') else str(a),allow_nan=False)+'\n')
def data():return pd.read_pickle(W/'paper_methods_4h/v1/inputs.pkl').reset_index(drop=True)
def metric(y,p):
 y=np.array(y);p=np.array(p);q=p>=.5
 return dict(n=len(y),BA=float(balanced_accuracy_score(y,q)) if len(set(y))==2 else None,MCC=float(matthews_corrcoef(y,q)),Brier=float(brier_score_loss(y,p)),pred_up=float(q.mean()),constant=int(len(set(q))==1))
def choose(records,default=.1):
 if not records:return default
 a=pd.DataFrame(records);s=a.groupby(['C','symbol']).BA.mean().unstack();return min(s.index,key=lambda c:(-s.loc[c].min(),-s.loc[c].mean(),c))
def split(d,s,m):
 boundary='2018-09' if m=='final' else m
 ev=np.flatnonzero((d.symbol==s)&((d.month>=boundary) if m=='final' else (d.month==m)))
 tr=np.flatnonzero((d.symbol==s)&(d.month<boundary)&(d.end_utc<d.iloc[ev].cutoff_utc.min()))
 assert d.iloc[tr].end_utc.max()<d.iloc[ev].cutoff_utc.min()
 return tr,ev

def load_bars(symbol):
 p=W/f"raw/CHARTS/{'APPLE' if symbol=='AAPL' else 'AMAZON'}5.csv";b=pd.read_csv(p,header=None,names=['date','time','open','high','low','close','activity']);b.index=pd.to_datetime(b.date+' '+b.time,format='%Y.%m.%d %H:%M',utc=True);assert not b.index.duplicated().any();return b.sort_index()
def schedule():
 a=pd.read_csv(W/'audit/xnys_schedule.csv');a['open']=pd.to_datetime(a.open,utc=True);a['close']=pd.to_datetime(a.close,utc=True);return a.sort_values('open').reset_index(drop=True)
def window(b,begin,end):
 idx=pd.date_range(begin,end-pd.Timedelta('5min'),freq='5min');g=b.reindex(idx)
 if len(g)==0 or g[['open','high','low','close']].isna().any().any():return None
 return dict(ret=float(np.log(g.close.iloc[-1]/g.open.iloc[0])),rv=float(np.sqrt(np.square(np.log(g.close/g.open)).sum())),range=float((g.high.max()-g.low.min())/g.open.iloc[0]))
def cross_features(d):
 bars={s:load_bars(s) for s in ('AAPL','AMZN')};sc=schedule();rows=[];audit=[]
 for r in d.itertuples():
  other='AMZN' if r.symbol=='AAPL' else 'AAPL';j=int(np.flatnonzero(sc.open.dt.date==r.start_utc.date())[0]);row={};latest=[]
  for minutes in (15,60):
   begin=r.cutoff_utc-pd.Timedelta(minutes=minutes);v=window(bars[other],begin,r.cutoff_utc) if begin>=sc.iloc[j].open else None
   row.update({f'x_other_return_{minutes}':v['ret'] if v else np.nan,f'x_other_rv_{minutes}':v['rv'] if v else np.nan,f'x_other_missing_{minutes}':int(v is None),f'x_relative_return_{minutes}':getattr(r,f'recent_return_{minutes}')-v['ret'] if v and not getattr(r,f'recent_missing_{minutes}') else np.nan})
   if v:latest.append(r.cutoff_utc)
  prev=sc.iloc[j-1] if j else None;v=window(bars[other],prev.open,prev.close) if prev is not None else None;own=window(bars[r.symbol],prev.open,prev.close) if prev is not None else None
  row.update(x_other_day_return=v['ret'] if v else np.nan,x_other_day_range=v['range'] if v else np.nan,x_relative_day_return=own['ret']-v['ret'] if v and own else np.nan,x_day_missing=int(not(v and own)))
  if v:latest.append(prev.close)
  assert not latest or max(latest)<=r.cutoff_utc
  rows.append(row);audit.append(dict(key=r.key,latest_used=str(max(latest)) if latest else None,cutoff=str(r.cutoff_utc)))
 return pd.DataFrame(rows),audit

def run_fingerprint(name,builder):
 files=[W/'paper_methods_4h/v1/inputs.pkl',OUT.parent/'PRE_REGISTRATION.md',Path(__file__),Path(sys.modules[builder.__module__].__file__)]
 if name.startswith('P_cross'):
  files += [W/f'raw/CHARTS/{s}5.csv' for s in ('APPLE','AMAZON')]+[W/'audit/xnys_schedule.csv']
 if 'tabpfn' in name:files += [W/'goal60_4h/tabpfn_model/tabpfn-v2.5-classifier-v2.5_default.ckpt']
 if not name.startswith('P_'):files += [PRIVATE/'P_own_lr/predictions.csv']
 if name.startswith(('F2','T_mix')):files += [ROOT/'outputs/stock_integrated_4h/prepared/articles.npz']
 if name=='T_mix':files += [W/'foundation_4h/v1/modern_embeddings.npz']
 if name.startswith('T_original'):files += [PRIVATE/'dense_original.npz']
 if name.startswith('T_A1'):files += [PRIVATE/f'dense_A1_{s}.npz' for s in (573,574,575)]
 if name.startswith('H'):files += [PRIVATE/'history_embeddings.npz',PRIVATE/'history_members.json',PRIVATE/'history_reactions.pkl',W/'audit/news_index.pkl']
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}

def run_model(name, builder, candidates=CS, fallback=None,gate=None):
 """builder(tr,ev,C) returns p, serializable evidence; scores AFTER fallback."""
 d=data();dest=PRIVATE/name;dest.mkdir(parents=True,exist_ok=True);fp=run_fingerprint(name,builder)
 if (dest/'done.json').exists():
  old=json.loads((dest/'done.json').read_text());assert old['fingerprint']==fp,'resume mismatch';return pd.read_csv(dest/'predictions.csv',float_precision='round_trip').p.to_numpy()
 started=dest/'started.json'
 if started.exists():assert json.loads(started.read_text())==fp,'incomplete resume mismatch'
 else:dump(started,fp)
 pred=np.full(len(d),np.nan);cv=[];fits=[];choices=[]
 for m in MONTHS:
  c=choose(cv,default=candidates[0] if len(candidates)==1 else .1);choices.append(dict(month=m,C=c,selection_months=sorted({r['month'] for r in cv})))
  for s in ('AAPL','AMZN'):
   tr,ev=split(d,s,m)
   for cc in ([c] if m=='final' else candidates):
    t=time.monotonic();p,evidence=builder(tr,ev,cc);p=np.asarray(p,dtype=np.float64).copy();raw=p.copy()
    if gate is not None:p[~gate[ev]]=fallback[ev][~gate[ev]]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    fits.append(dict(symbol=s,month=m,C=cc,train_n=len(tr),eval_n=len(ev),train_end=str(d.iloc[tr].end_utc.max()),eval_cutoff=str(d.iloc[ev].cutoff_utc.min()),seconds=time.monotonic()-t,**evidence))
    np.savez_compressed(dest/f'{s}_{m}_{cc}_pred.npz',indices=ev,raw=raw,issued=p)
    if m!='final':cv.append(dict(symbol=s,month=m,C=cc,**metric(d.iloc[ev].label,p)))
    if cc==c:pred[ev]=p
  print(name,m,'fits',len(fits),flush=True)
 pd.DataFrame({'key':d.key,'p':pred}).to_csv(dest/'predictions.csv',index=False);pd.DataFrame(cv).to_csv(dest/'cv.csv',index=False);dump(dest/'done.json',dict(fingerprint=fp,fits=fits,choices=choices))
 return pred

def price(name,cross=False,kind='lr'):
 d=data();x=d[OLD+RECENT].to_numpy(float)
 if cross:
  features,audit=cross_features(d);x=np.c_[x,features.to_numpy()];dump(PRIVATE/'cross_audit.json',audit);features.to_pickle(PRIVATE/'cross.pkl')
 dest=PRIVATE/name;dest.mkdir(parents=True,exist_ok=True)
 def builder(tr,ev,c):
  if kind=='lr':model=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,solver='liblinear',random_state=573,max_iter=3000,tol=1e-7))
  else:model=HistGradientBoostingClassifier(max_depth=3,max_iter=100,learning_rate=.05,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=573)
  model.fit(x[tr],d.iloc[tr].label);p=model.predict_proba(x[ev])[:,1];path=dest/f'model_{tr[-1]}_{c}.joblib';joblib.dump(model,path);p2=joblib.load(path).predict_proba(x[ev])[:,1];err=float(np.max(abs(p-p2)));assert err<1e-12
  return p,dict(model=path.name,sha256=sha(path),reload_error=err)
 with threadpool_limits(limits=2):return run_model(name,builder,candidates=CS if kind=='lr' else [1.])
if __name__=='__main__':
 PRIVATE.mkdir(parents=True,exist_ok=True)
 for cross in (False,True):
  for kind in ('lr','hgb'):price(f'P_{"cross" if cross else "own"}_{kind}',cross,kind)
