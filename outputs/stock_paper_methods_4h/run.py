"""Finite, past-only calibration, joint-price and news aggregation experiment."""
from pathlib import Path
import argparse, hashlib, json, re, time, warnings
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
WORK=ROOT/'work/stock-data'; INTEGRATED=ROOT/'outputs/stock_integrated_4h'
CS=(.01,.1,1.); MONTHS=list(pd.period_range('2018-03','2018-08',freq='M').astype(str))
OLD=[f'{v}_{i}' for i in range(1,7) for v in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour']
RECENT=[f'recent_{v}_{n}' for n in (5,15,30,60) for v in ('return','range','rv','missing')]+['overnight_gap','overnight_gap_missing','minutes_from_open','minutes_to_close']
META=['log_articles','log_clusters','log_sources','log_newest_age','log_median_age','only_duplicates']
METHODS=['J0','J1','J2','J3','N0M','N1','N1M']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x),allow_nan=False)+'\n')
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);q=p>=.5
 assert len(y)==len(p) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 return dict(n=len(y),BA=float(balanced_accuracy_score(y,q)) if len(set(y))==2 else None,MCC=float(matthews_corrcoef(y,q)),Brier=float(brier_score_loss(y,p)),AUC=float(roc_auc_score(y,p)) if len(set(y))==2 else None,up_recall=float(q[y==1].mean()) if (y==1).any() else None,down_recall=float((~q[y==0]).mean()) if (y==0).any() else None,pred_up=float(q.mean()),constant=bool(len(set(q))==1))
def scores(d,col):
 rows=[dict(symbol=s,month=m,**metric(g.label,g[col])) for (s,m),g in d.groupby(['symbol','month'])]
 return pd.DataFrame(rows)
def score(d,col):
 a=scores(d,col);return {k:float(a[k].mean()) for k in ['BA','Brier']}
def eligible(d,col,base):
 a=scores(d,col);b=scores(d,base)
 return bool(a.Brier.mean()<=b.Brier.mean()+.002 and all(a[a.symbol==s].Brier.mean()<=b[b.symbol==s].Brier.mean()+.002 for s in ('AAPL','AMZN')))

def signature(title):
 t=title.lower(); tokens=set(re.findall(r'[a-z0-9]+',t))
 groups=[r'\b(?:rais\w*|upgrad\w*|increas\w*|boost\w*)\b',r'\b(?:lower\w*|downgrad\w*|cut\w*|reduc\w*)\b',r'\b(?:maintain\w*|reiterat\w*)\b',r'\b(?:initiat\w*)\b',r'\b(?:deni\w*|deny\w*|not|no|refut\w*)\b',r'\b(?:accus\w*|alleg\w*)\b']
 guard=(tuple(bool(re.search(x,t)) for x in groups),tuple(sorted(re.findall(r'\d+(?:[.,]\d+)*',t))),tuple(sorted(re.findall(r'\b(?:q[1-4]|first|second|third|fourth)\b',t))))
 return tokens,guard

def compatible(a,b):
 ta,ga=signature(a);tb,gb=signature(b)
 return ga==gb and bool(ta|tb) and len(ta&tb)/len(ta|tb)>=.8

def prepare(private):
 sources=[WORK/'nextgen_4h/price_v1/features.pkl',INTEGRATED/'prepared/data.pkl',INTEGRATED/'prepared/articles.npz',INTEGRATED/'prepared/embedding_inputs.json',WORK/'audit/news_index.pkl',WORK/'nextgen_4h/fusion_v1/predictions.csv',WORK/'nextgen_4h/price_v1/feature_audit.csv',WORK/'finbert_event_adapter_4h/v1/downstream/predictions.csv']
 fingerprints={str(p.relative_to(ROOT)):sha(p) for p in sources}
 d=pd.read_pickle(sources[0]).sort_values(['start_utc','symbol']).reset_index(drop=True)
 original=pd.read_pickle(sources[1]);assert len(d)==1607 and not d.key.duplicated().any()
 for c in ('start_utc','end_utc','cutoff_utc'): d[c]=pd.to_datetime(d[c],utc=True)
 assert ((d.end_utc-d.start_utc)==pd.Timedelta('4h')).all()
 assert ((d.start_utc-d.cutoff_utc)==pd.Timedelta('5min')).all()
 ref=original.assign(key=original.symbol+'|'+pd.to_datetime(original.start_utc,utc=True).astype(str)).set_index('key')
 assert np.array_equal(d.label,ref.loc[d.key,'label'])
 audit=pd.read_csv(sources[6]);a=pd.to_datetime(audit.latest_used_bar_end,utc=True);b=pd.to_datetime(audit.cutoff,utc=True);assert (a.isna()|(a<=b)).all()
 raw=pd.read_pickle(sources[4]);raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key')
 raw['available_utc']=pd.to_datetime(raw.available_utc,utc=True)
 z=np.load(sources[2]);keys=z['keys'];emb=z['embeddings'].astype(float);lookup={k:i for i,k in enumerate(keys)}
 manifest=json.loads(sources[3].read_text());titles=dict(manifest['keys_and_titles'])
 assert manifest['revision']=='4556d13015211d73dccd3fdd39d39232506f3e43' and manifest['max_length']==256
 for k in keys: assert titles[k]==(raw.at[k,'title'] if pd.notna(raw.at[k,'title']) else '')
 d['news_record_keys']=d.news_record_keys.fillna(''); ids=[];means=[];events=[];stats=[];members=[]
 for r in d.itertuples():
  ks=[k for k in r.news_record_keys.split('|') if k];ks=sorted(ks,key=lambda k:(raw.at[k,'available_utc'],k));ii=[lookup[k] for k in ks];ids.append(ii)
  assert all(raw.at[k,'available_utc']<=r.cutoff_utc for k in ks)
  clusters=[]
  for k in ks:
   # All-pairs agreement prevents transitive merging of conflicting reports.
   target=next((c for c in clusters if all(compatible(titles[k],titles[j]) for j in c)),None)
   if target is None:clusters.append([k])
   else:target.append(k)
  means.append(emb[ii].mean(0) if ii else np.zeros(emb.shape[1]))
  events.append(np.mean([emb[[lookup[k] for k in c]].mean(0) for c in clusters],axis=0) if clusters else np.zeros(emb.shape[1]))
  ages=[(r.cutoff_utc-raw.at[k,'available_utc']).total_seconds()/3600 for k in ks]
  sites={str(raw.at[k,'site']).strip().lower() for k in ks if pd.notna(raw.at[k,'site']) and str(raw.at[k,'site']).strip()}
  stat=dict(key=r.key,articles=len(ks),clusters=len(clusters),sources=len(sites),newest_age=min(ages) if ages else 0,median_age=float(np.median(ages)) if ages else 0,only_duplicates=int(bool(clusters) and all(len(c)>1 for c in clusters)),has_usable_text=int(any(titles[k].strip() for k in ks)))
  stats.append(stat);members.append(dict(key=r.key,clusters=clusters))
 stats=pd.DataFrame(stats).set_index('key');d=d.join(stats,on='key');d['has_original_news']=d.has_news.astype(int)
 # Event flag is from the already selected A1 gate, not a new quality assertion.
 gates=pd.read_csv(sources[7]).set_index('key').D4_event_gate
 d['has_qualified_event']=d.key.map(gates).astype('Int64')
 for dest,src in zip(META[:-1],['articles','clusters','sources','newest_age','median_age']):d[dest]=np.log1p(d[src])
 prior=pd.read_csv(sources[5]).set_index('key')
 for c in ['F0','R1','F1','F2','F6']:d[c]=d.key.map(prior[c])
 d['phase']=np.where(d.month<'2018-03','warmup',np.where(d.month<'2018-09','train_forward_oof',np.where(d.month<'2018-11','development','later')))
 d['vector_delta']=np.linalg.norm(np.asarray(means)-np.asarray(events),axis=1)
 np.savez_compressed(private/'aggregates.npz',article_mean=means,event_mean=events)
 d.to_pickle(private/'inputs.pkl');dump(private/'clusters.json',members)
 return d,emb,ids,np.asarray(means),np.asarray(events),fingerprints

class Transform:
 def __init__(self,method):self.method=method
 def fit(self,d,indices,emb,ids,means,events):
  tr=d.iloc[indices]; self.columns=OLD+(RECENT if self.method in ('J1','J3') else [])
  x=tr[self.columns].to_numpy(float);self.mean=np.nanmean(x,axis=0);self.mean=np.nan_to_num(self.mean);self.scale=np.nanstd(x,axis=0);self.scale=np.where(self.scale>1e-12,self.scale,1.)
  if self.method in ('J0','J1'):
   self.vectorizer=CountVectorizer(binary=True,min_df=3);v=self.vectorizer.fit_transform(tr.stem_body.fillna(''));s=chi2(v,tr.label)[0];terms=self.vectorizer.get_feature_names_out();self.keep=np.lexsort((terms,-np.nan_to_num(s,nan=-np.inf)))[:500]
  else:
   articles=sorted({k for i in indices for k in ids[i]});self.pca=PCA(n_components=16,svd_solver='randomized',random_state=573).fit(emb[articles]);self.semantic_scaler=StandardScaler().fit(self.semantic(d,indices,means,events))
  return self
 def semantic(self,d,ii,means,events):
  z=self.pca.transform((events if self.method in ('N1','N1M') else means)[ii]);no=d.iloc[ii].has_original_news.to_numpy()==0;z[no]=0
  extra=d.iloc[ii][META].to_numpy() if self.method in ('N0M','N1M') else np.c_[np.log1p(d.iloc[ii].news_count),d.iloc[ii].has_news]
  return np.c_[z,extra]
 def transform(self,d,ii,means,events):
  g=d.iloc[ii];x=np.nan_to_num((g[self.columns].to_numpy(float)-self.mean)/self.scale)
  if self.method in ('J0','J1'):return sparse.hstack([sparse.csr_matrix(x),self.vectorizer.transform(g.stem_body.fillna(''))[:,self.keep]]).tocsr()
  return np.c_[x,self.semantic_scaler.transform(self.semantic(d,ii,means,events))]

def choose_c(records):
 if not records:return .1
 return min(CS,key=lambda c:(-np.mean([r['BA'] for r in records if r['C']==c]),np.mean([r['Brier'] for r in records if r['C']==c]),c))

def train(d,emb,ids,means,events,private):
 fits=[];cv=[];selections=[];start=time.monotonic()
 for symbol in ('AAPL','AMZN'):
  for month in MONTHS+['final']:
   tr=np.flatnonzero((d.symbol==symbol)&(d.month<('2018-09' if month=='final' else month)))
   ev=np.flatnonzero((d.symbol==symbol)&((d.month>='2018-09') if month=='final' else (d.month==month)))
   assert d.iloc[tr].end_utc.max()<d.iloc[ev].cutoff_utc.min()
   for method in METHODS:
    prior=[r for r in cv if r['symbol']==symbol and r['method']==method];chosen=choose_c(prior)
    tf=Transform(method).fit(d,tr,emb,ids,means,events);x=tf.transform(d,tr,means,events);xx=tf.transform(d,ev,means,events)
    candidates=[chosen] if month=='final' else CS
    for c in candidates:
     tick=time.monotonic();model=LogisticRegression(C=c,solver='liblinear',tol=1e-7,max_iter=3000,random_state=573).fit(x,d.iloc[tr].label)
     assert model.n_iter_.max()<3000
     p=model.predict_proba(xx)[:,1];name=f'{symbol}_{month}_{method}_{c}.joblib';path=private/'models'/name;joblib.dump(dict(transform=tf,model=model),path)
     reload=joblib.load(path);p2=reload['model'].predict_proba(reload['transform'].transform(d,ev,means,events))[:,1];err=float(np.max(np.abs(p-p2)));assert err<1e-12
     raw_score=metric(d.iloc[ev].label,p)
     none=d.iloc[ev].has_original_news.to_numpy()==0;p[none]=d.iloc[ev].R1.to_numpy()[none]
     fits.append(dict(name=name,train_n=len(tr),eval_n=len(ev),train_end=str(d.iloc[tr].end_utc.max()),cutoff_min=str(d.iloc[ev].cutoff_utc.min()),seconds=time.monotonic()-tick,reload_error=err,sha256=sha(path)))
     if month!='final':cv.append(dict(symbol=symbol,month=month,method=method,C=c,**raw_score))
     if c==chosen:d.loc[ev,method]=p
    selections.append(dict(symbol=symbol,month=month,method=method,C=chosen,selection_months=sorted({r['month'] for r in prior})))
   print('joint',symbol,month,'fits',len(fits),'seconds',round(time.monotonic()-start,1),flush=True)
 d['N0']=d.J2
 dump(private/'training.json',dict(fits=fits,selections=selections,seconds=time.monotonic()-start));pd.DataFrame(cv).to_csv(private/'cv.csv',index=False)
 return fits,selections


def calibration_fit(y,p,kind):
 if kind=='identity' or len(y)<30:return 1.,0.
 l=logit(np.clip(np.asarray(p),1e-6,1-1e-6));y=np.asarray(y)
 def objective(v):
  a=np.exp(v[0]);b=0 if kind=='temperature' else v[1];z=a*l+b
  loss=np.logaddexp(0,z).sum()-np.dot(y,z)
  if kind!='temperature':loss+=10*((a-1)**2+b*b)/2
  return loss
 init=[0.] if kind=='temperature' else [0.,0.]
 opt=minimize(objective,init,method='L-BFGS-B',bounds=[(-4.6,4.6)]+([] if kind=='temperature' else [(-5,5)]))
 if not opt.success:raise RuntimeError(opt.message)
 return float(np.exp(opt.x[0])),float(opt.x[1]) if len(opt.x)>1 else 0.

def calibrate(d):
 records=[];choices=[]
 for branch in ('F1','F2'):
  for month in MONTHS+['final']:
   cutoff='2018-09' if month=='final' else month
   previous=d[(d.month<cutoff)&(d.month>='2018-03')]
   # Select scheme only on prior outer calibrated predictions, not fitted scores.
   selection=previous[previous.month>='2018-04']
   kinds=['identity','temperature','platt_shrunk'];chosen='identity'
   if len(selection):
    valid=[k for k in kinds if eligible(selection,f'{branch}_{k}',branch)]
    chosen=min(valid,key=lambda k:(score(selection,f'{branch}_{k}')['Brier'],-score(selection,f'{branch}_{k}')['BA'],k))
   for symbol in ('AAPL','AMZN'):
    prior=previous[previous.symbol==symbol];mask=(d.symbol==symbol)&((d.month>='2018-09') if month=='final' else (d.month==month))
    for kind in kinds:
     # Calibrate only original-news branch; identical no-news R1 policy.
     fitting=prior[prior.has_original_news==1];a,b=calibration_fit(fitting.label,fitting[branch],kind)
     p=d.loc[mask,branch].to_numpy().copy();yes=d.loc[mask,'has_original_news'].to_numpy()==1
     if kind!='identity':p[yes]=expit(a*logit(np.clip(p[yes],1e-6,1-1e-6))+b)
     d.loc[mask,f'{branch}_{kind}']=p
     records.append(dict(branch=branch,symbol=symbol,month=month,kind=kind,slope=a,intercept=b,temperature=1/a,n=len(fitting),last_fit_month=str(fitting.month.max()) if len(fitting) else None))
    d.loc[mask,f'{branch}_calibrated']=d.loc[mask,f'{branch}_{chosen}']
   choices.append(dict(branch=branch,month=month,kind=chosen,selection_months=sorted(selection.month.unique().tolist())))
 return records,choices

def advancement(d):
 pairs=[('J1','J0'),('J3','J2'),('N1','N0'),('N1M','N0M')];out=[]
 for new,base in pairs:
  g=d[d.phase=='train_forward_oof'];a=scores(g,new).set_index(['symbol','month']);b=scores(g,base).set_index(['symbol','month']);delta=a[['BA','Brier']]-b[['BA','Brier']];months=delta.groupby('month').BA.mean();stocks=delta.groupby('symbol').mean()
  passed=len(months)>=3 and (months>0).sum()>=2 and delta.BA.mean()>=.01 and stocks.BA.min()>=-.01 and stocks.Brier.max()<=.002
  out.append(dict(method=new,baseline=base,months=len(months),positive_months=int((months>0).sum()),delta_BA=float(delta.BA.mean()),delta_Brier=float(delta.Brier.mean()),stock_deltas=stocks.to_dict('index'),passes=bool(passed)))
 return out

def tables(d,public):
 methods=['F0','R1','F1','F2','F6']+[f'{b}_{k}' for b in ('F1','F2') for k in ('temperature','platt_shrunk','calibrated')]+METHODS+['N0']
 rows=[];monthly=[];sub=[];trans=[]
 for (phase,sym),g in d[d.phase!='warmup'].groupby(['phase','symbol']):
  for m in methods:
   rows.append(dict(phase=phase,symbol=sym,method=m,**metric(g.label,g[m])))
   for month,x in g.groupby('month'):monthly.append(dict(phase=phase,symbol=sym,month=month,method=m,**metric(x.label,x[m])))
   for flag in ('has_original_news','has_usable_text','has_qualified_event'):
    for value,x in g.groupby(flag):sub.append(dict(phase=phase,symbol=sym,flag=flag,value=value,method=m,**metric(x.label,x[m])))
  for m,b in [('J1','J0'),('J3','J2'),('N1','N0'),('N1M','N0M'),('F1_calibrated','F1'),('F2_calibrated','F2')]:
   a=(g[m]>=.5)==g.label;bb=(g[b]>=.5)==g.label
   trans.append(dict(phase=phase,symbol=sym,method=m,baseline=b,changed_right=int((a&~bb).sum()),changed_wrong=int((~a&bb).sum())))
 for name,data in [('metrics',rows),('monthly_metrics',monthly),('subgroup_metrics',sub),('transitions',trans)]:pd.DataFrame(data).to_csv(public/f'{name}.csv',index=False)
 cols=['key','symbol','day','month','phase','label','has_original_news','has_usable_text','has_qualified_event','articles','clusters','sources','newest_age','median_age','only_duplicates','vector_delta']+methods
 d[cols].to_csv(public/'predictions.csv',index=False)
 return pd.DataFrame(rows)

def main(out):
 start=time.monotonic();public=HERE/out;private=WORK/'paper_methods_4h'/out
 if public.exists() or private.exists():raise FileExistsError('Use a new run id; historical runs cannot be overwritten')
 public.mkdir(parents=True);(private/'models').mkdir(parents=True)
 with threadpool_limits(limits=2):
  d,emb,ids,means,events,source_hashes=prepare(private);print('prepared',len(d),'changed aggregation',int((d.vector_delta>1e-10).sum()),flush=True)
  fits,selections=train(d,emb,ids,means,events,private)
  calibration,choices=calibrate(d);gate=advancement(d)
  # Exact replay of unchanged J0/J2 validates the reusable transform path.
  usable=d.phase!='warmup';parity={m:float(np.max(np.abs(d.loc[usable,m]-d.loc[usable,b]))) for m,b in [('J0','F1'),('J2','F2')]}
  assert max(parity.values())<1e-8,parity
  methods=METHODS+['N0','F1_calibrated','F2_calibrated']
  for m in methods:
   mask=usable&(d.has_original_news==0);assert np.array_equal(d.loc[mask,m],d.loc[mask,'R1'])
  metrics=tables(d,public)
  pd.DataFrame(calibration).to_csv(public/'calibration.csv',index=False)
  dump(public/'selection.json',dict(joint=selections,calibration=choices,advancement=gate))
  manifest=dict(fits=fits,source_hashes=source_hashes,code_sha256=sha(__file__),protocol_sha256=sha(HERE/'PRE_REGISTRATION.md'),parity=parity,rows=1607,evaluable_rows=int(usable.sum()),seconds=time.monotonic()-start,exact_no_news_fallback=True,status='COMPLETE',phase2_eligible=any(x['passes'] for x in gate),all_evaluation_exposed=True)
  assert all(sha(ROOT/p)==h for p,h in source_hashes.items())
  dump(public/'model_manifest.json',manifest);d.to_pickle(private/'results.pkl')
  print(json.dumps(gate,indent=2),flush=True)
  print(metrics[metrics.phase.isin(['development','later'])][['phase','symbol','method','BA','Brier','AUC']].to_string(index=False),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',default='v1');main(parser.parse_args().run)
