"""Frozen horizon pilot: past-only LR, all results exposed historical replay."""
from pathlib import Path
import sys,json,hashlib,time,platform
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss
B=Path(__file__).resolve().parent;O=B.parent;ROOT=O.parent;W=ROOT/'work/stock-data'
sys.path.insert(0,str(O/'stock_baseline'))
from run_baseline import build,window_indices
PRICE=[f'{v}_{k}' for k in range(1,7) for v in ['return','range']]+['history_age_hours','return_mean','return_std','ny_hour']
PROTOCOL=dict(status='exploratory historical replay; all periods previously exposed',horizons={'1h':'every complete hourly start','4h':'every hourly start with four continuous regular-session hours','day':'session open to session close; includes short sessions'},cutoff='target start minus five minutes',news='same legacy filters/dedup; past four hours of availability; title TFIDF',history='six fully completed regular-session hours; identical across horizons at shared cutoff',flat='exclude flat target independently per horizon; report exclusions',models=['price','price_count','price_text'],text='price + counts + unigram/bigram TFIDF max_features100 min_df2 L2',C=[.01,.1,1.],selection='mean BA June July August forward folds; ties Brier then smaller C',evaluation='frozen Jan-Aug fit, Sep-Oct development and Nov-Feb exposed test; report each month and common open anchors',boundaries='train labels end strictly before earliest validation cutoff; no overnight targets',uncertainty='paired trading day blocks of 1 and 5 days; descriptive, not selection corrected',seed=573)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);z=p>=.5
 return dict(n=len(y),ba=float(balanced_accuracy_score(y,z)) if len(set(y))==2 else None,mcc=float(matthews_corrcoef(y,z)),brier=float(brier_score_loss(y,p)),up_rate=float(y.mean()),pred_up=float(z.mean()),constant=bool(z.min()==z.max()))
def estimator(kind,C):
 num=PRICE+(['log_news_count','has_news'] if kind!='price' else [])
 parts=[('numeric',StandardScaler(),num)]
 if kind=='price_text':parts.append(('text',TfidfVectorizer(max_features=100,min_df=2,ngram_range=(1,2),sublinear_tf=True),'text'))
 return Pipeline([('features',ColumnTransformer(parts)),('model',LogisticRegression(C=C,solver='liblinear',max_iter=3000,random_state=573))])
def dataset():
 cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d')
 for c in ['open','close']:cal[c]=pd.to_datetime(cal[c],utc=True)
 rows=[];cov=[];checks=[]
 for stock,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
  legacy,_,_,news=build(W,stock,prefix);times=news.available_utc.astype('int64').to_numpy()
  raw=pd.read_csv(W/f'raw/CHARTS/{prefix}5.csv',header=None,names=['date','time','open','high','low','close','activity']);raw.index=pd.to_datetime(raw.date+' '+raw.time,format='%Y.%m.%d %H:%M',utc=True);raw=raw.sort_index();assert not raw.index.duplicated().any()
  def aggregate(start,end):
   x=raw.reindex(pd.date_range(start,end-pd.Timedelta(minutes=5),freq='5min'))
   if x[['open','high','low','close']].isna().any().any():return None
   return dict(start=start,end=end,open=x.open.iloc[0],close=x.close.iloc[-1],high=x.high.max(),low=x.low.min())
  bars=[];candidates=[]
  for day,c in cal.iterrows():
   if c.open<raw.index.min() or c.open>raw.index.max():continue
   for start in pd.date_range(c.open,c.close-pd.Timedelta(hours=1),freq='h'):
    end=start+pd.Timedelta(hours=1);b=aggregate(start,end)
    if b:bars.append(b)
    candidates.append((day,'1h',start,end,start==c.open))
    if start+pd.Timedelta(hours=4)<=c.close:candidates.append((day,'4h',start,start+pd.Timedelta(hours=4),start==c.open))
   candidates.append((day,'day',c.open,c.close,True))
  bars=pd.DataFrame(bars)
  for day,h,start,end,opening in candidates:
   r=dict(symbol=stock,horizon=h,day=day,start_utc=start,end_utc=end,opening=opening);b=aggregate(start,end);cut=start-pd.Timedelta(minutes=5);hist=bars[bars.end<=cut].tail(6)
   reason='missing_prices' if b is None else 'insufficient_history' if len(hist)<6 else 'flat' if b['open']==b['close'] else 'accepted'
   cov.append(dict(**r,reason=reason))
   if reason!='accepted':continue
   left,right=window_indices(times,cut);a=news.iloc[left:right]
   assert hist.end.max()<=cut and (a.empty or (a.available_utc.max()<=cut and a.available_utc.min()>cut-pd.Timedelta(hours=4)))
   r.update(cutoff_utc=cut,label=int(b['close']>b['open']),target_return=b['close']/b['open']-1,text=' . '.join(a.title.fillna('')),news_count=len(a),log_news_count=np.log1p(len(a)),has_news=int(len(a)>0),article_keys='|'.join(a.archive.astype(str)+'::'+a.member.astype(str)),history_end=hist.end.max(),split='train' if start<pd.Timestamp('2018-09-01',tz='UTC') else 'validation' if start<pd.Timestamp('2018-11-01',tz='UTC') else 'test')
   for k,x in enumerate(hist.iloc[::-1].itertuples(),1):r[f'return_{k}']=np.log(x.close/x.open);r[f'range_{k}']=(x.high-x.low)/x.open
   r.update(history_age_hours=(cut-hist.end.iloc[-1]).total_seconds()/3600,return_mean=np.mean([r[f'return_{k}'] for k in range(1,7)]),return_std=np.std([r[f'return_{k}'] for k in range(1,7)]),ny_hour=start.tz_convert('America/New_York').hour+.5)
   rows.append(r)
  one=pd.DataFrame([r for r in rows if r['symbol']==stock and r['horizon']=='1h']).sort_values('start_utc');legacy=legacy.sort_values('start_utc')
  assert list(one.start_utc)==list(legacy.start_utc);np.testing.assert_array_equal(one.label,legacy.label);np.testing.assert_allclose(one[PRICE],legacy[PRICE],atol=1e-12);assert list(one.text)==list(legacy.text)
  checks.append(dict(symbol=stock,legacy_hourly_parity=True,rows=len(one)))
 return pd.DataFrame(rows),pd.DataFrame(cov),checks

def main():
 out=B/'runs/v1';out.mkdir(parents=True,exist_ok=False);t=time.monotonic();dump(out/'protocol.json',PROTOCOL)
 sources=[Path(__file__),O/'stock_baseline/run_baseline.py',W/'audit/news_index.pkl',W/'audit/xnys_schedule.csv']+list((W/'raw/CHARTS').glob('*5.csv'));dump(out/'sources.json',{str(p):sha(p) for p in sources})
 d,cov,checks=dataset();d.to_pickle(out/'data.pkl');d.drop(columns='text').to_csv(out/'samples.csv',index=False);cov.to_csv(out/'coverage.csv',index=False);dump(out/'checks.json',checks)
 preds=[];cv=[];selected=[];man=[]
 for (stock,h),s in d.groupby(['symbol','horizon']):
  for kind in PROTOCOL['models']:
   candidates=[]
   for C in PROTOCOL['C']:
    for mo in [6,7,8]:
     lo=pd.Timestamp(f'2018-{mo:02d}-01',tz='UTC');b=s[(s.start_utc>=lo)&(s.start_utc<lo+pd.DateOffset(months=1))];a=s[s.end_utc<b.cutoff_utc.min()];assert a.end_utc.max()<b.cutoff_utc.min()
     m=estimator(kind,C).fit(a,a.label);p=m.predict_proba(b)[:,1];r=dict(symbol=stock,horizon=h,method=kind,C=C,month=str(lo)[:7],train_n=len(a),**metric(b.label,p));candidates.append(r);cv.append(r)
   grid=pd.DataFrame(candidates).groupby('C')[['ba','brier']].mean().reset_index().sort_values(['ba','brier','C'],ascending=[False,True,True]);C=float(grid.iloc[0].C);selected.append(dict(symbol=stock,horizon=h,method=kind,C=C,cv_ba=float(grid.iloc[0].ba)))
   b=s[~s.split.eq('train')];a=s[s.split.eq('train') & (s.end_utc<b.cutoff_utc.min())];m=estimator(kind,C).fit(a,a.label);p=m.predict_proba(b)[:,1];path=out/f'{stock}_{h}_{kind}.joblib';joblib.dump(m,path);np.testing.assert_allclose(joblib.load(path).predict_proba(b)[:,1],p,atol=1e-12)
   z=b[['symbol','horizon','day','start_utc','end_utc','cutoff_utc','opening','label','split','has_news','news_count']].copy();z['method']=kind;z['p']=p;preds.append(z)
   man.append(dict(model=path.name,sha256=sha(path),train_keys=a.start_utc.astype(str).tolist(),predict_keys=b.start_utc.astype(str).tolist(),train_n=len(a),train_end=a.end_utc.max(),first_cutoff=b.cutoff_utc.min(),train_metrics=metric(a.label,m.predict_proba(a)[:,1])))
  print(stock,h,'complete',flush=True)
 p=pd.concat(preds,ignore_index=True);p.to_csv(out/'predictions.csv',index=False);pd.DataFrame(cv).to_csv(out/'cv.csv',index=False);dump(out/'selected.json',selected);dump(out/'models.json',man)
 summary=[]
 for (stock,h,method,split),g in p.groupby(['symbol','horizon','method','split']):summary.append(dict(symbol=stock,horizon=h,method=method,period=split,**metric(g.label,g.p),days=g.day.nunique(),news_ratio=g.has_news.mean()))
 for (stock,h,method,month),g in p.groupby(['symbol','horizon','method',p.day.str[:7]]):summary.append(dict(symbol=stock,horizon=h,method=method,period=month,**metric(g.label,g.p),days=g.day.nunique(),news_ratio=g.has_news.mean()))
 pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
 # Shared opening anchors: identical information and dates, separate target labels.
 op=p[p.opening].copy();sets=[set(g.day) for _,g in op.groupby(['symbol','horizon','method'])];common=set.intersection(*sets);op=op[op.day.isin(common)];op.to_csv(out/'common_open_predictions.csv',index=False)
 common_metrics=[]
 for key,g in op.groupby(['symbol','horizon','method','split']):common_metrics.append(dict(zip(['symbol','horizon','method','split'],key),**metric(g.label,g.p)))
 pd.DataFrame(common_metrics).to_csv(out/'common_open_metrics.csv',index=False)
 # Paired day/block bootstrap within each horizon, never resample rows independently.
 ci=[];rng=np.random.default_rng(573)
 for (stock,h,split),g in p.groupby(['symbol','horizon','split']):
  wide=g.pivot(index=['day','start_utc','label'],columns='method',values='p').reset_index();days=sorted(wide.day.unique());groups=[np.flatnonzero(wide.day.eq(day)) for day in days]
  for method in ['price_count','price_text']:
   for block in [1,5]:
    vals=[]
    for _ in range(1000):
     ids=[]
     while len(ids)<len(days):
      start=int(rng.integers(0,max(1,len(days)-block+1)));ids.extend(range(start,min(start+block,len(days))))
     ix=np.concatenate([groups[i] for i in ids[:len(days)]]);z=wide.iloc[ix]
     if z.label.nunique()<2:continue
     vals.append([balanced_accuracy_score(z.label,z[method]>=.5)-balanced_accuracy_score(z.label,z.price>=.5),brier_score_loss(z.label,z[method])-brier_score_loss(z.label,z.price)])
    low,hi=np.quantile(vals,[.025,.975],axis=0);ci.append(dict(symbol=stock,horizon=h,split=split,method=method,block_days=block,ba_low=low[0],ba_high=hi[0],brier_low=low[1],brier_high=hi[1]))
 pd.DataFrame(ci).to_csv(out/'paired_intervals.csv',index=False)
 dump(out/'status.json',dict(state='complete',seconds=time.monotonic()-t,models=len(man),cv_fits=len(cv),python=platform.python_version(),common_open_days=len(common)))
 print('COMPLETE',time.monotonic()-t,flush=True)
if __name__=='__main__':main()
