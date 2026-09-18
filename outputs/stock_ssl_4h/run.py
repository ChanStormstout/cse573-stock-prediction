"""Finite masked-reconstruction experiment; no stock labels in pretraining."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'outputs/stock_goal60_4h'))
import core
W=core.W;HERE=Path(__file__).resolve().parent;OUT=HERE/'v1';PRIVATE=W/'ssl_4h/v1';SEEDS=(573,574,575)
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv1d(7,16,3,padding=1),nn.GELU(),nn.Conv1d(16,16,3,padding=4,dilation=4),nn.GELU(),nn.Conv1d(16,16,3,padding=16,dilation=16),nn.GELU());self.decoder=nn.Conv1d(16,4,1)
 def forward(self,x,mask=None):
  if mask is None:mask=torch.zeros((len(x),48),dtype=torch.bool)
  a=x.clone();a[:,:,:4]=a[:,:,:4].masked_fill(mask[:,:,None],0);z=self.layers(torch.cat([a,mask[:,:,None].float()],2).transpose(1,2));return self.decoder(z).transpose(1,2),torch.cat([z.mean(2),z[:,:,-1]],1)
def normalize(x,mean,scale):return ((x-mean)/scale).astype(np.float32)
def scaler(raw,indices):
 v=raw[np.unique(indices)];mean=v.mean(0);scale=v.std(0);return mean,np.where(scale>1e-8,scale,1)
def prepare():
 d=core.data();sc=core.schedule();grid=pd.to_datetime(np.concatenate([pd.date_range(r.open,r.close-pd.Timedelta('5min'),freq='5min').to_numpy() for r in sc.itertuples()]),utc=True);raws=[];times=[];symbols=[]
 for s in ('AAPL','AMZN'):
  b=core.load_bars(s).reindex(grid);prev=b.close.shift();span=(b.high-b.low).to_numpy();gap=np.r_[1,(np.diff(grid.asi8)>pd.Timedelta('5min').value).astype(float)];ny=grid.tz_convert('America/New_York');tod=(ny.hour*60+ny.minute-570)/390
  v=np.column_stack([np.log(b.close/b.open),np.log(b.high/b.low),np.divide((b.close-b.low).to_numpy(),span,out=np.full(len(b),.5),where=span!=0),np.log(b.open/prev),gap,tod]);v[~np.isfinite(v).all(1)]=np.nan;raws.append(v);times.extend(grid);symbols.extend([s]*len(grid))
 raw=np.concatenate(raws);ends=pd.DatetimeIndex(times)+pd.Timedelta('5min');syms=np.array(symbols);n=len(grid);supervised=np.full((len(d),48),-1,dtype=int);audit=[];unlab=[]
 for i,r in enumerate(d.itertuples()):
  off=0 if r.symbol=='AAPL' else n;j=np.searchsorted(grid.asi8,(r.cutoff_utc-pd.Timedelta('5min')).value,side='right')-1;idx=off+np.arange(j-47,j+1);valid=j>=47 and np.isfinite(raw[idx]).all()
  if valid:supervised[i]=idx;assert ends[idx].max()<=r.cutoff_utc
  audit.append(dict(key=r.key,valid=bool(valid),cutoff=str(r.cutoff_utc),latest_end=str(ends[idx].max()) if valid else None))
 for off in (0,n):
  for j in range(47,n,12):
   idx=off+np.arange(j-47,j+1)
   if ends[idx[-1]]>=pd.Timestamp('2018-01-01',tz='UTC') and np.isfinite(raw[idx]).all():unlab.append(idx)
 unlab=np.array(unlab);np.savez_compressed(PRIVATE/'inputs.npz',raw=raw,ends=ends.asi8,symbols=syms,supervised=supervised,unlabeled=unlab);core.dump(OUT/'input_audit.json',audit);return d,raw,ends,supervised,unlab

def fit_ssl(x,seed,epochs,validation=None):
 torch.manual_seed(seed);m=Encoder();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);x=torch.tensor(x);rng=np.random.default_rng(seed);curve=[];best=float('inf');epochbest=1;stale=0;maxgrad=0;start=time.monotonic()
 if validation is not None:
  v=torch.tensor(validation);gen=torch.Generator().manual_seed(873);vm=torch.rand((len(v),48),generator=gen)<.25;vlosszero=float(nn.functional.smooth_l1_loss(torch.zeros_like(v[:,:,:4])[vm],v[:,:,:4][vm]))
 else:vlosszero=None
 for epoch in range(1,epochs+1):
  loss_sum=0.;count=0;m.train()
  for ids in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/128))):
   a=x[ids];mask=torch.rand((len(a),48))<.25;opt.zero_grad();recon,_=m(a,mask);loss=nn.functional.smooth_l1_loss(recon[mask],a[:,:,:4][mask]);loss.backward();grad=float(sum(p.grad.square().sum() for p in m.parameters() if p.grad is not None).sqrt());maxgrad=max(maxgrad,grad);opt.step();loss_sum+=float(loss.detach())*len(a);count+=len(a)
  vl=None
  if validation is not None:
   m.eval()
   with torch.no_grad():rec,_=m(v,vm);vl=float(nn.functional.smooth_l1_loss(rec[vm],v[:,:,:4][vm]))
   if vl<best-1e-6:best=vl;epochbest=epoch;stale=0
   else:stale+=1
  curve.append(dict(epoch=epoch,train_loss=loss_sum/count,validation_loss=vl))
  if validation is not None and stale>=4:break
 assert maxgrad>0
 return m,(epochbest if validation is not None else epochs),dict(curve=curve,max_gradient=maxgrad,seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),validation_zero_loss=vlosszero)

def representations(raw,ends,sup,unlab,cutoff,month):
 permitted=unlab[(ends[unlab[:,-1]]<cutoff)&(ends[unlab[:,-1]].strftime('%Y-%m')<('2018-09' if month=='final' else month))];dates=ends[permitted[:,-1]].strftime('%Y-%m-%d');unique=sorted(set(dates));boundary=unique[int(.85*len(unique))];val=permitted[dates>=boundary];train=permitted[ends[permitted[:,-1]]<ends[val[:,0]].min()-pd.Timedelta('5min')];assert len(train)>0 and len(val)>0 and ends[train[:,-1]].max()<ends[val[:,0]].min()-pd.Timedelta('5min');mi,si=scaler(raw,train);mf,sf=scaler(raw,permitted);valid=sup[:,0]>=0;allx=normalize(raw[sup[valid]],mf,sf);res={'random':[],'ssl':[]};records=[]
 for seed in SEEDS:
  tick=time.monotonic();_,epochs,early=fit_ssl(normalize(raw[train],mi,si),seed,20,normalize(raw[val],mi,si));m,_,final=fit_ssl(normalize(raw[permitted],mf,sf),seed,epochs);m.eval();torch.manual_seed(seed);random=Encoder().eval();path=PRIVATE/f'encoder_{month}_{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mf,scale=sf,epochs=epochs),path);saved=torch.load(path,weights_only=False,map_location='cpu');reload=Encoder();reload.load_state_dict(saved['state']);reload.eval()
  with torch.no_grad():
   ss=m(torch.tensor(allx))[1].numpy();rr=random(torch.tensor(allx))[1].numpy();check=reload(torch.tensor(allx))[1].numpy();err=float(abs(check-ss).max());assert err<1e-7
  for name,v in [('random',rr),('ssl',ss)]:
   a=np.zeros((len(sup),32));a[valid]=v;res[name].append(a)
  records.append(dict(month=month,seed=seed,unlabeled_n=len(permitted),unique_endpoint_days=len(unique),inner_train_n=len(train),inner_validation_n=len(val),latest_pretrain_end=str(ends[permitted[:,-1]].max()),evaluation_cutoff=str(cutoff),selected_epochs=epochs,early=early,final=final,device='cpu',seconds=time.monotonic()-tick,checkpoint_sha256=core.sha(path),reload_error=err));print('encoder',month,seed,'epochs',epochs,'sequences',len(permitted),flush=True)
 return res,records

def run():
 torch.set_num_threads(2);PRIVATE.mkdir(parents=True,exist_ok=True)
 sources=[Path(__file__),HERE/'PRE_REGISTRATION.md',ROOT/'outputs/stock_goal60_4h/core.py',W/'paper_methods_4h/v1/inputs.pkl',W/'goal60_4h/v1/P_own_lr/predictions.csv',W/'audit/xnys_schedule.csv']+[W/f'raw/CHARTS/{s}5.csv' for s in ('APPLE','AMAZON')];fp={str(p.relative_to(ROOT)):core.sha(p) for p in sources};manifest=PRIVATE/'fingerprint.json'
 if manifest.exists():assert json.loads(manifest.read_text())==fp,'cache mismatch'
 else:core.dump(manifest,fp)
 if (PRIVATE/'done.json').exists():print('matching completed run exists');return
 d,raw,ends,sup,unlab=prepare();valid=sup[:,0]>=0;fallback=pd.read_csv(W/'goal60_4h/v1/P_own_lr/predictions.csv',float_precision='round_trip').p.to_numpy();methods=['RAW','RANDOM','SSL'];pred={m:np.full(len(d),np.nan) for m in methods};seedpred={m:np.full((3,len(d)),np.nan) for m in ['RANDOM','SSL']};cv={m:[] for m in methods};choices=[];enc=[];fits=[]
 for month in core.MONTHS:
  boundary='2018-09' if month=='final' else month;ev_all=np.flatnonzero(d.month.ge(boundary) if month=='final' else d.month.eq(month));rep,records=representations(raw,ends,sup,unlab,d.iloc[ev_all].cutoff_utc.min(),month);enc+=records;selected={m:core.choose(cv[m]) for m in methods}
  for method in methods:choices.append(dict(method=method,month=month,C=selected[method],past_months=sorted({r['month'] for r in cv[method]})))
  for symbol in ('AAPL','AMZN'):
   tr,ev=core.split(d,symbol,month);good=tr[valid[tr]];price=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler()).fit(d.iloc[tr][core.OLD+core.RECENT]);px=price.transform(d[core.OLD+core.RECENT]);bm,bs=scaler(raw,sup[good]);flat=np.zeros((len(d),48*6));flat[valid]=normalize(raw[sup[valid]],bm,bs).reshape(sum(valid),-1);pca=PCA(n_components=16,random_state=573,svd_solver='randomized').fit(flat[good]);z=pca.transform(flat);z[~valid]=0
   for method in methods:
    for c in ([selected[method]] if month=='final' else core.CS):
     predictions=[]
     for seedidx,embedding in enumerate([z] if method=='RAW' else rep[method.lower()]):
      start=time.monotonic();scale=StandardScaler().fit(embedding[tr]);x=np.c_[px,scale.transform(embedding)];model=LogisticRegression(C=c,solver='liblinear',tol=1e-7,max_iter=3000,random_state=573).fit(x[tr],d.iloc[tr].label);p=model.predict_proba(x[ev])[:,1].astype(float);path=PRIVATE/f'head_{method}_{month}_{symbol}_{c}_{seedidx}.joblib';joblib.dump(dict(model=model,price=price,embedding_scale=scale,pca=pca if method=='RAW' else None,bar_mean=bm,bar_scale=bs),path);reload=joblib.load(path);err=float(abs(reload['model'].predict_proba(x[ev])[:,1]-p).max());assert err<1e-12;p[~valid[ev]]=fallback[ev][~valid[ev]];predictions.append(p);fits.append(dict(method=method,month=month,symbol=symbol,C=c,seed=SEEDS[seedidx],train_n=len(tr),train_label_end=str(d.iloc[tr].end_utc.max()),eval_cutoff=str(d.iloc[ev].cutoff_utc.min()),seconds=time.monotonic()-start,reload_error=err,file=path.name,sha256=core.sha(path)))
     prob=np.mean(predictions,axis=0);prob[~valid[ev]]=fallback[ev][~valid[ev]]
     if month!='final':cv[method].append(dict(symbol=symbol,month=month,C=c,**core.metric(d.iloc[ev].label,prob)))
     if c==selected[method]:
      pred[method][ev]=prob
      if method!='RAW':seedpred[method][:,ev]=np.array(predictions)
  core.dump(PRIVATE/'progress.json',dict(completed_month=month,encoder_fits=len(enc),heads=len(fits)));print('complete month',month,flush=True)
 out=d[['key','symbol','day','month','phase','label','has_original_news']].copy();out['has_sequence']=valid.astype(int);out['B']=fallback
 for m,p in pred.items():out[m]=p
 out.to_csv(PRIVATE/'predictions.csv',index=False);np.savez_compressed(PRIVATE/'seed_predictions.npz',**seedpred);core.dump(PRIVATE/'done.json',dict(fingerprint=fp,encoders=enc,heads=fits,selections=choices));core.dump(PRIVATE/'cv.json',cv)
if __name__=='__main__':
 with threadpool_limits(limits=2):run()
