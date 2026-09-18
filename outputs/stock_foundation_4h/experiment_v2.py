"""v2 matched FinBERT/Fin-ModernBERT comparison.

This is intentionally a separate run family.  It first reproduces the
canonical FinBERT J2 path, then substitutes only the encoder embeddings.
"""
from pathlib import Path
import sys,json,time,argparse,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/stock_paper_methods_4h'))
from run import Transform,metric,sha,dump,CS,MONTHS,choose_c
W=ROOT/'work/stock-data/foundation_4h'; PRIVATE=W/'v2'; OUT=HERE/'v2'; SOURCE=ROOT/'work/stock-data/paper_methods_4h/v1'

def inputs():
 d=pd.read_pickle(SOURCE/'inputs.pkl');z=np.load(ROOT/'outputs/stock_integrated_4h/prepared/articles.npz');keys=z['keys'];emb=z['embeddings'].astype(float);lookup={k:i for i,k in enumerate(keys)}
 ids=[[lookup[k] for k in s.split('|') if k] for s in d.news_record_keys]
 v=np.load(SOURCE/'aggregates.npz');return d,keys,emb,ids,v['article_mean'],v['event_mean']

def init():
 PRIVATE.mkdir(parents=True,exist_ok=True);(PRIVATE/'models').mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 sources=[SOURCE/'inputs.pkl',SOURCE/'aggregates.npz',ROOT/'outputs/stock_integrated_4h/prepared/articles.npz',HERE/'v2'/'PRE_REGISTRATION.md']
 fingerprint={str(p.relative_to(ROOT)):sha(p) for p in sources};p=PRIVATE/'source_hashes.json'
 if p.exists():assert json.loads(p.read_text())==fingerprint,'input/protocol mismatch'
 else:dump(p,fingerprint)
 return fingerprint

def summarize(d,methods,stage):
 rows=[];monthly=[]
 for (phase,symbol),g in d[d.phase!='warmup'].groupby(['phase','symbol']):
  for method in methods:
   rows.append(dict(phase=phase,symbol=symbol,method=method,**metric(g.label,g[method])))
   for month,q in g.groupby('month'):monthly.append(dict(phase=phase,symbol=symbol,month=month,method=method,**metric(q.label,q[method])))
 pd.DataFrame(rows).to_csv(OUT/f'{stage}_metrics.csv',index=False);pd.DataFrame(monthly).to_csv(OUT/f'{stage}_monthly.csv',index=False)
 cols=['key','symbol','day','month','phase','label','has_original_news','has_usable_text','has_qualified_event']+methods
 d[cols].to_csv(OUT/f'{stage}_predictions.csv',index=False)

def train_text(d,emb,ids,means,events,specs,stage):
 if (PRIVATE/f'{stage}_complete.json').exists():raise FileExistsError('completed stage')
 fits=[];cv=[];choices=[]
 for symbol in ['AAPL','AMZN']:
  for month in MONTHS+['final']:
   tr=np.flatnonzero((d.symbol==symbol)&(d.month<('2018-09' if month=='final' else month)))
   ev=np.flatnonzero((d.symbol==symbol)&((d.month>='2018-09') if month=='final' else (d.month==month)))
   assert d.iloc[tr].end_utc.max()<d.iloc[ev].cutoff_utc.min()
   for name,method,fixed in specs:
    prior=[r for r in cv if r['symbol']==symbol and r['method']==name];chosen=fixed if fixed is not None else choose_c(prior)
    tf=Transform(method).fit(d,tr,emb,ids,means,events);x=tf.transform(d,tr,means,events);xx=tf.transform(d,ev,means,events)
    for c in ([chosen] if fixed is not None or month=='final' else CS):
     path=PRIVATE/'models'/f'{stage}_{symbol}_{month}_{name}_{c}.joblib'
     if path.exists():raise FileExistsError(path)
     tick=time.monotonic();model=LogisticRegression(C=c,solver='liblinear',max_iter=3000,tol=1e-7,random_state=573).fit(x,d.iloc[tr].label);assert model.n_iter_.max()<3000
     p=model.predict_proba(xx)[:,1];raw=metric(d.iloc[ev].label,p)
     joblib.dump(dict(transform=tf,model=model,keys=d.iloc[tr].key.tolist()),path);loaded=joblib.load(path);error=float(np.max(np.abs(p-loaded['model'].predict_proba(loaded['transform'].transform(d,ev,means,events))[:,1])));assert error<1e-12
     p[d.iloc[ev].has_original_news.to_numpy()==0]=d.iloc[ev].R1.to_numpy()[d.iloc[ev].has_original_news.to_numpy()==0]
     fits.append(dict(name=path.name,train_n=len(tr),eval_n=len(ev),seconds=time.monotonic()-tick,reload_error=error,sha256=sha(path),train_end=str(d.iloc[tr].end_utc.max()),cutoff_min=str(d.iloc[ev].cutoff_utc.min())))
     if month!='final':cv.append(dict(symbol=symbol,month=month,method=name,C=c,**raw))
     if c==chosen:d.loc[ev,name]=p
    choices.append(dict(symbol=symbol,month=month,method=name,C=chosen,selection_months=sorted({r['month'] for r in prior})))
   print(stage,symbol,month,'fits',len(fits),flush=True)
 methods=['F0','R1','F1','F2','F6']+[x[0] for x in specs];summarize(d,methods,stage)
 result=dict(fits=fits,choices=choices,code_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'v2'/'PRE_REGISTRATION.md'));dump(OUT/f'{stage}_training.json',result);dump(PRIVATE/f'{stage}_complete.json',{'fits':len(fits)});pd.DataFrame(cv).to_csv(OUT/f'{stage}_cv.csv',index=False)
 return d

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['aggregation','modern']);a=p.parse_args();init();d,keys,emb,ids,means,events=inputs()
 with threadpool_limits(2):
  if a.stage=='aggregation':
   train_text(d,emb,ids,means,events,[('A01','N0M',.01),('A1','N0M',1.),('E01','N1M',.01),('E1','N1M',1.)],'aggregation')
  else:
   z=np.load(PRIVATE/'modern_embeddings.npz');assert np.array_equal(z['keys'],keys)
   # Separate matched old encoder run and new encoder run, identical selection.
   d=train_text(d,emb,ids,means,events,[('M_F2','J2',None)],'finbert_match')
   usable=d.phase!='warmup';assert np.max(np.abs(d.loc[usable,'M_F2']-d.loc[usable,'F2']))<1e-10
   emb=z['embeddings'].astype(float);means=np.asarray([emb[ii].mean(0) if ii else np.zeros(emb.shape[1]) for ii in ids]);train_text(d,emb,ids,means,means,[('MODERN','J2',None)],'modern')
if __name__=='__main__':main()
