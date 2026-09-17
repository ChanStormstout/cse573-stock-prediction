"""Frozen Chronos forecasts aligned to existing four-hour open-to-close labels."""
from pathlib import Path
import sys,json,time,warnings
import numpy as np,pandas as pd,torch,joblib
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from experiment import ROOT,HERE,W,PRIVATE,OUT,SOURCE,init,inputs,sha,dump,metric,CS,MONTHS,choose_c,summarize
sys.path.insert(0,str(W/'runtime'))
from chronos import Chronos2Pipeline

def load_bars(symbol):
 p=ROOT/'work/stock-data/raw/CHARTS'/('APPLE5.csv' if symbol=='AAPL' else 'AMAZON5.csv')
 b=pd.read_csv(p,header=None,names=['date','time','open','high','low','close','activity']);b.index=pd.to_datetime(b.date+' '+b.time,format='%Y.%m.%d %H:%M',utc=True);assert b.index.is_unique
 return p,b.sort_index()

def prepare(d):
 sched=ROOT/'work/stock-data/audit/xnys_schedule.csv';s=pd.read_csv(sched);idx=pd.DatetimeIndex(np.concatenate([pd.date_range(r.open,r.close,freq='5min',inclusive='left').to_numpy() for r in s.itertuples()]))
 assert idx.is_unique and idx.is_monotonic_increasing
 contexts=np.full((len(d),2,512),np.nan,np.float32);audit=[];sources={str(sched.relative_to(ROOT)):sha(sched)}
 for sym in ('AAPL','AMZN'):
  path,b=load_bars(sym);sources[str(path.relative_to(ROOT))]=sha(path);values=b.reindex(idx)[['open','close']].to_numpy()
  for i,r in d[d.symbol==sym].iterrows():
   past=np.flatnonzero(idx+pd.Timedelta('5min')<=r.cutoff_utc);last=int(past[-1]) if len(past) else -1;hist=values[max(0,last-511):last+1].T
   if hist.shape[1]:contexts[i,:,-hist.shape[1]:]=hist
   first=idx.get_indexer([r.start_utc])[0];final=idx.get_indexer([r.end_utc-pd.Timedelta('5min')])[0];step_open=first-last;step_close=final-last
   valid=np.isfinite(contexts[i]).all(0);observed=int(valid.sum());eligible=observed>=48 and first>=0 and final>=0 and 1<=step_open<=2 and step_close<=49
   latest=str(idx[last]+pd.Timedelta('5min')) if last>=0 else None
   assert latest is None or pd.Timestamp(latest)<=r.cutoff_utc
   if first>=0 and final>=0:
    assert final-first==47
    actual=b.loc[r.end_utc-pd.Timedelta('5min'),'close']>b.loc[r.start_utc,'open'];assert int(actual)==r.label
   anchor=float(contexts[i,1,np.flatnonzero(valid)[-1]]) if observed else np.nan
   audit.append(dict(key=r.key,cutoff=str(r.cutoff_utc),latest_context_end=latest,observed_bars=observed,missing_bars=512-observed,open_step=step_open,close_step=step_close,eligible=eligible,anchor=anchor))
 a=pd.DataFrame(audit).set_index('key').loc[d.key].reset_index();assert list(a.key)==list(d.key)
 return contexts,a,sources

def infer(d):
 contexts,a,sources=prepare(d);mp=W/'models/chronos';fingerprint=dict(sources=sources,input=sha(SOURCE/'inputs.pkl'),protocol=sha(HERE/'PRE_REGISTRATION.md'),code=sha(Path(__file__)),weights=sha(mp/'model.safetensors'),config=sha(mp/'config.json'),revision='29ec3766d36d6f73f0696f85560a422f50e8498c',context=512,prediction_length=49,quantiles=[.1,.5,.9],cross_learning=False,dtype='float32')
 fp=PRIVATE/'chronos_cache.json'
 if fp.exists():assert json.loads(fp.read_text())==fingerprint,'chronos cache mismatch'
 else:dump(fp,fingerprint)
 path=PRIVATE/'chronos_forecasts.npz'
 if path.exists():
  z=np.load(path);assert np.array_equal(z['keys'],d.key);return contexts,a,z['features']
 a.to_csv(OUT/'chronos_alignment.csv',index=False);torch.manual_seed(573);torch.set_num_threads(4);tick=time.monotonic();pipeline=Chronos2Pipeline.from_pretrained(str(mp),device_map='cpu',torch_dtype=torch.float32);pipeline.model.eval();pipeline.model.requires_grad_(False)
 chunk=PRIVATE/'chronos_chunks';chunk.mkdir(exist_ok=True);features=np.full((len(d),3),np.nan);quantiles=np.full((len(d),2,3),np.nan);calls=0
 active=np.flatnonzero(a.eligible.to_numpy());first_batch=None;first_output=None
 for start in range(0,len(active),32):
  ix=active[start:start+32];p=chunk/f'{start:05d}.npz'
  if p.exists():
   z=np.load(p);assert np.array_equal(z['keys'],d.iloc[ix].key);qq=z['quantiles']
  else:
   batch=torch.tensor(contexts[ix]);q,_=pipeline.predict_quantiles(batch,prediction_length=49,quantile_levels=[.1,.5,.9],context_length=512,batch_size=64,cross_learning=False)
   qq=np.array([[v[0,int(a.iloc[i].open_step)-1,:].numpy(),v[1,int(a.iloc[i].close_step)-1,:].numpy()] for i,v in zip(ix,q)])
   assert np.isfinite(qq).all() and (qq>0).all() and (np.diff(qq,axis=-1)>=-1e-5).all()
   np.savez_compressed(p,keys=np.asarray(d.iloc[ix].key,dtype=str),quantiles=qq);calls+=1
   if first_batch is None:first_batch=batch[:1];first_output=qq[0];first_index=ix[0]
  anchor=a.iloc[ix].anchor.to_numpy();quantiles[ix]=qq;features[ix]=np.c_[(qq[:,1,1]-qq[:,0,1])/anchor,(qq[:,0,2]-qq[:,0,0])/anchor,(qq[:,1,2]-qq[:,1,0])/anchor]
  print('chronos',start+len(ix),'/',len(active),'seconds',round(time.monotonic()-tick,1),flush=True)
  if time.monotonic()-tick>8*3600:raise TimeoutError('inference budget exceeded')
 replay=None
 if first_batch is not None:
  # Reload weights and independently replay one task: batch isolation and weight replay.
  reloaded=Chronos2Pipeline.from_pretrained(str(mp),device_map='cpu',torch_dtype=torch.float32);reloaded.model.eval()
  q,_=reloaded.predict_quantiles(first_batch,prediction_length=49,quantile_levels=[.1,.5,.9],context_length=512,batch_size=64,cross_learning=False)
  i=first_index;observed=np.array([q[0][0,int(a.iloc[i].open_step)-1,:].numpy(),q[0][1,int(a.iloc[i].close_step)-1,:].numpy()]);replay=float(np.max(np.abs(observed-first_output)));assert np.allclose(observed,first_output,rtol=1e-5,atol=1e-5)
 np.savez_compressed(path,keys=np.asarray(d.key,dtype=str),features=features,quantiles=quantiles)
 evidence=dict(model='amazon/chronos-2',revision=fingerprint['revision'],device='cpu',encoder_training=False,parameters=sum(p.numel() for p in pipeline.model.parameters()),trainable_parameters=sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad),gradients_present=any(p.grad is not None for p in pipeline.model.parameters()),windows=len(d),eligible=int(a.eligible.sum()),new_batches=calls,seconds=time.monotonic()-tick,reload_single_task_error=replay,fingerprint=fingerprint,torch=torch.__version__,package='chronos-forecasting==2.2.2')
 dump(OUT/'chronos_inference.json',evidence);return contexts,a,features

def fit(d,contexts,a,features):
 if (PRIVATE/'chronos_complete.json').exists():raise FileExistsError('chronos classifiers already complete')
 ok=a.eligible.to_numpy() & np.isfinite(features).all(1);anchor=a.anchor.to_numpy();lag=np.log(contexts/anchor[:,None,None]).reshape(len(d),-1);xxs={'HISTORY_LR':lag,'CHRONOS_LR':features};fits=[];cv=[];choices=[]
 for symbol in ['AAPL','AMZN']:
  for month in MONTHS+['final']:
   tr=np.flatnonzero((d.symbol==symbol)&(d.month<('2018-09' if month=='final' else month))&ok);ev=np.flatnonzero((d.symbol==symbol)&((d.month>='2018-09') if month=='final' else (d.month==month)));assert d.iloc[tr].end_utc.max()<d.iloc[ev].cutoff_utc.min()
   for name,xx in xxs.items():
    prior=[r for r in cv if r['symbol']==symbol and r['method']==name];chosen=choose_c(prior)
    with warnings.catch_warnings():
     warnings.simplefilter('ignore',RuntimeWarning);mean=np.nan_to_num(np.nanmean(xx[tr],axis=0));scale=np.nanstd(xx[tr],axis=0);scale=np.where(np.isfinite(scale)&(scale>1e-12),scale,1.)
    x=np.nan_to_num((xx[tr]-mean)/scale);xe=np.nan_to_num((xx[ev]-mean)/scale)
    for c in ([chosen] if month=='final' else CS):
     path=PRIVATE/'models'/f'{symbol}_{month}_{name}_{c}.joblib'
     if path.exists():raise FileExistsError(path)
     tick=time.monotonic();model=LogisticRegression(C=c,solver='liblinear',tol=1e-7,max_iter=3000,random_state=573).fit(x,d.iloc[tr].label);assert model.n_iter_.max()<3000
     p=model.predict_proba(xe)[:,1];p[~ok[ev]]=d.iloc[ev].R1.to_numpy()[~ok[ev]];joblib.dump(dict(model=model,mean=mean,scale=scale,keys=d.iloc[tr].key.tolist()),path);saved=joblib.load(path);p2=saved['model'].predict_proba(np.nan_to_num((xx[ev]-saved['mean'])/saved['scale']))[:,1];p2[~ok[ev]]=d.iloc[ev].R1.to_numpy()[~ok[ev]];error=float(np.max(np.abs(p-p2)));assert error<1e-12
     fits.append(dict(name=path.name,train_n=len(tr),eval_n=len(ev),seconds=time.monotonic()-tick,reload_error=error,sha256=sha(path),train_end=str(d.iloc[tr].end_utc.max()),cutoff_min=str(d.iloc[ev].cutoff_utc.min())))
     if month!='final':cv.append(dict(symbol=symbol,month=month,method=name,C=c,**metric(d.iloc[ev].label,p)))
     if c==chosen:d.loc[ev,name]=p
    choices.append(dict(symbol=symbol,month=month,method=name,C=chosen,selection_months=sorted({r['month'] for r in prior})))
   print('chronos classifiers',symbol,month,len(fits),flush=True)
 summarize(d,['F0','R1','F1','F2','F6','HISTORY_LR','CHRONOS_LR'],'chronos');pd.DataFrame(cv).to_csv(OUT/'chronos_cv.csv',index=False)
 dump(OUT/'chronos_training.json',dict(fits=fits,choices=choices,code_sha256=sha(Path(__file__))))
 raw=[]
 for (phase,symbol),g in d[(d.phase!='warmup')&ok].groupby(['phase','symbol']):
  for name in ['CHRONOS_MEDIAN','R1','HISTORY_LR']:
   p=(features[g.index,0]>=0).astype(float) if name=='CHRONOS_MEDIAN' else g[name].to_numpy();m=metric(g.label,p);m.pop('Brier');m.pop('AUC');raw.append(dict(phase=phase,symbol=symbol,method=name,**m))
 pd.DataFrame(raw).to_csv(OUT/'chronos_raw_direction.csv',index=False);dump(PRIVATE/'chronos_complete.json',{'fits':len(fits)})

def main():
 init();d,*_=inputs();contexts,a,features=infer(d)
 with threadpool_limits(2):fit(d,contexts,a,features)
if __name__=='__main__':main()
