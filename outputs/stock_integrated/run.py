from pathlib import Path
import sys,json,time,hashlib,itertools
import numpy as np,pandas as pd,joblib
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss,accuracy_score
B=Path(__file__).resolve().parent;ROOT=B.parents[1];O=ROOT/'outputs';W=ROOT/'work/stock-data'
sys.path.insert(0,str(O/'stock_four_hour_v2'))
from text_vectorizer import DeterministicTfidf
PRICE=[f'{v}_{i}' for i in range(1,7) for v in ['return','range']]+['history_age_hours','return_mean','return_std','ny_hour']
BRANCHES=['price','title','body','semantic'];CS=[.01,.1,1.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str))
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);q=p>=.5
 return dict(n=len(y),BA=balanced_accuracy_score(y,q) if len(set(y))==2 else None,MCC=matthews_corrcoef(y,q),Brier=brier_score_loss(y,p),accuracy=accuracy_score(y,q),up_recall=float(q[y==1].mean()) if (y==1).any() else None,down_recall=float((~q[y==0]).mean()) if (y==0).any() else None,pred_up=float(q.mean()),coverage=1.)
def monthly(y,p,months):
 return {k:float(np.mean([metric(np.asarray(y)[months==m],np.asarray(p)[months==m])[k] for m in sorted(set(months))])) for k in ['BA','Brier']}
class Transform:
 def __init__(self,kind):self.kind=kind
 def fit(self,d,keys,emb):
  self.scaler=StandardScaler().fit(d[PRICE])
  if self.kind=='title':self.text=DeterministicTfidf(500,2).fit(d.text)
  if self.kind=='body':
   self.text=CountVectorizer(binary=True,min_df=3);x=self.text.fit_transform(d.stem_body);scores=chi2(x,d.label)[0];terms=self.text.get_feature_names_out();self.keep=np.lexsort((terms,-np.nan_to_num(scores,nan=-np.inf)))[:500]
  if self.kind=='semantic':
   lookup={k:i for i,k in enumerate(keys)};self.training_article_keys=sorted({k for s in d.news_record_keys for k in s.split('|') if k});ids=[lookup[k] for k in self.training_article_keys]
   self.pca=PCA(n_components=16,svd_solver='randomized',random_state=573).fit(emb[ids]);self.semantic_scaler=StandardScaler().fit(self.semantic(d,keys,emb))
  return self
 def semantic(self,d,keys,emb):
  lookup={k:i for i,k in enumerate(keys)};z=[]
  for s in d.news_record_keys:
   ids=[lookup[k] for k in s.split('|') if k];z.append(self.pca.transform(emb[ids].mean(0,keepdims=True))[0] if ids else np.zeros(16))
  return np.c_[z,np.log1p(d.news_count),d.has_news]
 def transform(self,d,keys,emb):
  x=self.scaler.transform(d[PRICE])
  if self.kind=='title':return sparse.hstack([sparse.csr_matrix(x),self.text.transform(d.text)]).tocsr()
  if self.kind=='body':return sparse.hstack([sparse.csr_matrix(x),self.text.transform(d.stem_body)[:,self.keep]]).tocsr()
  if self.kind=='semantic':return np.c_[x,self.semantic_scaler.transform(self.semantic(d,keys,emb))]
  return x

def weights_grid(forbidden=None,multiple=False):
 for ints in itertools.product(range(5),repeat=4):
  if sum(ints)!=4 or (forbidden is not None and ints[BRANCHES.index(forbidden)]):continue
  if multiple and sum(v>0 for v in ints)<2:continue
  yield np.array(ints)/4

def combine(d,w):
 p=d[BRANCHES].to_numpy()@w;p[d.has_news.to_numpy()==0]=d.price.to_numpy()[d.has_news.to_numpy()==0];return p

def select_fusion(ledger,forbidden=None,multiple=False):
 months=ledger.month.to_numpy();base=monthly(ledger.label,ledger.title,months);grid=[]
 for w in weights_grid(forbidden,multiple):
  score=monthly(ledger.label,combine(ledger,w),months);grid.append(dict(weights=w.tolist(),**score,eligible=score['Brier']<=base['Brier']+.002))
 eligible=[x for x in grid if x['eligible']]
 if not eligible:return dict(weights=None,fallback='unchanged_title',selection_months=sorted(set(months))),grid
 chosen=sorted(eligible,key=lambda r:(-r['BA'],r['Brier'],sum(x>0 for x in r['weights']),r['weights']))[0]
 return {**chosen,'selection_months':sorted(set(months))},grid

def main(out):
 out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();t=time.monotonic()
 inputs=[O/'stock_robust/results/data.pkl',O/'stock_final_test/results/test_inputs.pkl',O/'stock_comprehensive/runs/v1/M08/articles.npz',O/'stock_comprehensive/runs/v1/M08/embedding_inputs.json',W/'audit/news_index.pkl',B/'PROTOCOL.md',Path(__file__),O/'stock_four_hour_v2/text_vectorizer.py']
 hashes={str(p):sha(p) for p in inputs};dump(out/'sources.json',hashes)
 d=pd.concat([pd.read_pickle(inputs[0]),pd.read_pickle(inputs[1])],ignore_index=True)
 for col in ['start_utc','end_utc','cutoff_utc']:d[col]=pd.to_datetime(d[col],utc=True)
 d['key']=d.symbol+'|'+d.start_utc.astype(str);d['month']=d.start_utc.dt.strftime('%Y-%m');d['day']=d.start_utc.dt.strftime('%Y-%m-%d');d['news_record_keys']=d.news_record_keys.fillna('');assert len(d)==3233 and not d.key.duplicated().any()
 z=np.load(inputs[2]);keys=z['keys'];emb=z['embeddings'].astype(np.float64);fingerprint=json.loads(inputs[3].read_text());raw=pd.read_pickle(inputs[4]);raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');titles=dict(fingerprint['keys_and_titles']);needed={k for s in d.news_record_keys for k in s.split('|') if k}
 assert needed<=set(keys) and all(titles[k]==(raw.loc[k,'title'] if pd.notna(raw.loc[k,'title']) else '') for k in needed)
 assert fingerprint['revision']=='4556d13015211d73dccd3fdd39d39232506f3e43' and fingerprint['max_length']==256
 for r in d.itertuples():
  ids=[k for k in r.news_record_keys.split('|') if k]
  if ids:assert (pd.to_datetime(raw.loc[ids,'available_utc'],utc=True)<=r.cutoff_utc).all()
 d.drop(columns=[x for x in d if x.startswith('finbert_')],errors='ignore').to_pickle(out/'inputs.pkl');pred=[];selection=[];cv=[];fits=[];ledgers=[];grids=[];fitcount=0
 def train(tr,ev,kind,C,transform,name):
  nonlocal fitcount
  assert tr.end_utc.max()<ev.cutoff_utc.min();a=transform.transform(tr,keys,emb);b=transform.transform(ev,keys,emb)
  model=LogisticRegression(C=C,solver='liblinear',max_iter=3000,random_state=573,tol=1e-7).fit(a,tr.label);fitcount+=1
  if model.n_iter_.max()>=3000:raise RuntimeError('Did not converge')
  path=out/'models'/f'{name}.joblib';joblib.dump({'transform':transform,'classifier':model},path);p=model.predict_proba(b)[:,1];loaded=joblib.load(path);reloaded=loaded['classifier'].predict_proba(loaded['transform'].transform(ev,keys,emb))[:,1];np.testing.assert_allclose(p,reloaded,rtol=0,atol=1e-12);np.testing.assert_array_equal(p>=.5,reloaded>=.5)
  fits.append(dict(name=name,C=C,train_keys=tr.key.tolist(),eval_keys=ev.key.tolist(),label_end_max=str(tr.end_utc.max()),fit_cutoff=str(ev.cutoff_utc.min()),model_hash=sha(path),reload_max_error=float(np.max(np.abs(p-reloaded))),train_metrics=metric(tr.label,model.predict_proba(a)[:,1])))
  return p
 for sym,g in d.groupby('symbol'):
  history=[];ledger=[]
  for month in pd.period_range('2018-03','2018-08',freq='M').astype(str):
   tr=g[g.month<month];ev=g[g.month==month];row=ev[['key','symbol','day','month','label','has_news','split']].copy()
   for kind in BRANCHES:
    prior=[r for r in history if r['branch']==kind];C=.1 if not prior else sorted(CS,key=lambda c:(-np.mean([r['BA'] for r in prior if r['C']==c]),np.mean([r['Brier'] for r in prior if r['C']==c]),c))[0]
    transform=Transform(kind).fit(tr,keys,emb)
    for candidate in CS:
     p=train(tr,ev,kind,candidate,transform,f'{sym}_{month}_{kind}_{candidate}');score=dict(symbol=sym,month=month,branch=kind,C=candidate,**metric(ev.label,p));history.append(score);cv.append(score)
     if candidate==C:row[kind]=p
    selection.append(dict(symbol=sym,month=month,branch=kind,C=C,selection_months=sorted({r['month'] for r in prior})))
   if month>='2018-06':
    previous=pd.concat(ledger);outer=row.copy();outer['phase']='outer'
    for name,forbidden,multiple in [('integrated',None,False),('multi_only',None,True),('no_semantic','semantic',False),('no_body','body',False),('no_title','title',False)]:
     choice,grid=select_fusion(previous,forbidden,multiple);selection.append(dict(symbol=sym,month=month,branch=name,**choice));grids.extend([dict(symbol=sym,month=month,branch=name,**x) for x in grid]);outer[name]=outer.title if choice['weights'] is None else combine(outer,choice['weights'])
    outer['uniform']=combine(outer,np.ones(4)/4);pred.append(outer)
   ledger.append(row);print(sym,month,'complete',fitcount,flush=True)
  ledger=pd.concat(ledger);ledgers.append(ledger);tr=g[g.month<'2018-09'];ev=g[g.month>='2018-09'];row=ev[['key','symbol','day','month','label','has_news','split']].copy();row['phase']='frozen'
  for kind in BRANCHES:
   prior=[r for r in history if r['branch']==kind];C=sorted(CS,key=lambda c:(-np.mean([r['BA'] for r in prior if r['C']==c]),np.mean([r['Brier'] for r in prior if r['C']==c]),c))[0];tf=Transform(kind).fit(tr,keys,emb);row[kind]=train(tr,ev,kind,C,tf,f'{sym}_final_{kind}');selection.append(dict(symbol=sym,month='final',branch=kind,C=C,selection_months=sorted({r['month'] for r in prior})))
  for name,forbidden,multiple in [('integrated',None,False),('multi_only',None,True),('no_semantic','semantic',False),('no_body','body',False),('no_title','title',False)]:
   choice,grid=select_fusion(ledger,forbidden,multiple);selection.append(dict(symbol=sym,month='final',branch=name,**choice));grids.extend([dict(symbol=sym,month='final',branch=name,**x) for x in grid]);row[name]=row.title if choice['weights'] is None else combine(row,choice['weights'])
  row['uniform']=combine(row,np.ones(4)/4);pred.append(row)
 p=pd.concat(pred,ignore_index=True);p.to_csv(out/'predictions.csv',index=False);pd.concat(ledgers).to_csv(out/'oof.csv',index=False);pd.DataFrame(cv).to_csv(out/'cv.csv',index=False);dump(out/'selection.json',selection);dump(out/'fusion_grid.json',grids);dump(out/'fits.json',fits)
 rows=[];methods=BRANCHES+['integrated','multi_only','no_semantic','no_body','no_title','uniform']
 for (sym,phase),g in p.groupby(['symbol','phase']):
  groups=[('all',g)]+[(m,x) for m,x in g.groupby('month')]
  if phase=='frozen':groups += [(s,x) for s,x in g.groupby('split')]
  for period,x in groups:
   for method in methods:rows.append(dict(symbol=sym,phase=phase,period=period,method=method,**metric(x.label,x[method])))
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
 assert hashes=={str(p):sha(p) for p in inputs}
 dump(out/'status.json',dict(status='COMPLETE',LR_fits=fitcount,seconds=time.monotonic()-t,source_hashes_unchanged=True,rows=3233,interpretation='Exploratory historical replay, not untouched test'))
 print(pd.DataFrame(rows).query("phase=='frozen' and period=='test'")[['symbol','method','BA','Brier']].to_string(index=False),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.output)
