"""Isolated random-CV study. No historical fitted predictors are imported."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,brier_score_loss,matthews_corrcoef,accuracy_score

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'v1'
PRIVATE=ROOT/'work/stock-data/random_protocol_4h/v1'
OLD=[f'{v}_{i}' for i in range(1,7) for v in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour']
RECENT=[f'recent_{v}_{n}' for n in (5,15,30,60) for v in ('return','range','rv','missing')]+['overnight_gap','overnight_gap_missing','minutes_from_open','minutes_to_close']
METHODS=['PAPER','PRICE','FULL','FINBERT','MODERN']
CS=[.01,.1,1.]
SEEDS=[573,574,575]
SOURCES={
 'rows':ROOT/'work/stock-data/paper_methods_4h/v1/inputs.pkl',
 'finbert':ROOT/'outputs/stock_integrated_4h/prepared/articles.npz',
 'modern':ROOT/'work/stock-data/foundation_4h/v2/modern_embeddings.npz',
 'modern_evidence':ROOT/'outputs/stock_foundation_4h/v2/modern_inference.json',
 'titles':ROOT/'outputs/stock_integrated_4h/prepared/embedding_inputs.json',
 'raw_index':ROOT/'work/stock-data/audit/news_index.pkl',
 'canonical':ROOT/'outputs/stock_integrated_4h/prepared/data.pkl',
 'canonical_manifest':ROOT/'outputs/stock_integrated_4h/prepared/manifest.json',
 'price_audit':ROOT/'work/stock-data/nextgen_4h/price_v1/feature_audit.csv',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
def hashes():return {k:sha(p) for k,p in SOURCES.items()}|{'protocol':sha(HERE/'PROTOCOL.md')}
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);q=p>=.5
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 return dict(n=len(y),BA=float(balanced_accuracy_score(y,q)),Brier=float(brier_score_loss(y,p)),MCC=float(matthews_corrcoef(y,q)),accuracy=float(accuracy_score(y,q)),predicted_up=float(q.mean()),constant=bool(q.min()==q.max()))
def load():
 d=pd.read_pickle(SOURCES['rows']).reset_index(drop=True)
 # Hard whitelist: discard stored probabilities/returns/reader gates before training.
 cols=['key','symbol','day','month','label','start_utc','end_utc','cutoff_utc','stem_body','news_record_keys','news_count','has_news']+OLD+RECENT
 d=d[cols].copy();d['row_id']=[digest(k)[:24] for k in d.key]
 for c in ['start_utc','end_utc','cutoff_utc']:d[c]=pd.to_datetime(d[c],utc=True)
 z=np.load(SOURCES['finbert']);m=np.load(SOURCES['modern']);assert np.array_equal(z['keys'],m['keys'])
 lookup={k:i for i,k in enumerate(z['keys'])}
 ids=[[lookup[k] for k in str(s).split('|') if k] for s in d.news_record_keys]
 embeddings={'FINBERT':z['embeddings'].astype(float),'MODERN':m['embeddings'].astype(float)}
 means={name:np.array([e[ii].mean(0) if ii else np.zeros(e.shape[1]) for ii in ids]) for name,e in embeddings.items()}
 return d,embeddings,ids,means

class Scale:
 def fit(self,x):
  x=np.asarray(x,float);self.mean=np.nan_to_num(np.nanmean(x,axis=0));self.std=np.nanstd(x,axis=0);self.std=np.where(self.std>1e-12,self.std,1.);return self
 def transform(self,x):return np.nan_to_num((np.asarray(x,float)-self.mean)/self.std)

class Features:
 def __init__(self,method):self.method=method
 def fit(self,d,tr,emb,ids,means):
  self.fit_ids=d.iloc[tr].row_id.tolist();self.cols=OLD+RECENT if self.method=='PRICE' else OLD
  self.price_scale=Scale().fit(d.iloc[tr][self.cols])
  if self.method in ('PAPER','FULL'):
   self.vectorizer=CountVectorizer(binary=True,min_df=3)
   v=self.vectorizer.fit_transform(d.iloc[tr].stem_body.fillna(''))
   scores=np.nan_to_num(chi2(v,d.iloc[tr].label)[0],nan=-np.inf)
   self.keep=np.lexsort((self.vectorizer.get_feature_names_out(),-scores))[:500]
  elif self.method in ('FINBERT','MODERN'):
   self.article_ids=sorted({k for i in tr for k in ids[i]})
   self.pca=PCA(n_components=16,svd_solver='randomized',random_state=573).fit(emb[self.method][self.article_ids])
   self.semantic_scale=Scale().fit(self.semantic(d,tr,means))
  return self
 def semantic(self,d,ii,means):
  x=self.pca.transform(means[self.method][ii]);x[d.iloc[ii].has_news.to_numpy()==0]=0
  return np.c_[x,np.log1p(d.iloc[ii].news_count),d.iloc[ii].has_news]
 def transform(self,d,ii,means):
  price=self.price_scale.transform(d.iloc[ii][self.cols])
  if self.method=='PRICE':return price
  if self.method in ('PAPER','FULL'):
   text=self.vectorizer.transform(d.iloc[ii].stem_body.fillna(''))[:,self.keep]
   return text if self.method=='PAPER' else sparse.hstack([price,text]).tocsr()
  return np.c_[price,self.semantic_scale.transform(self.semantic(d,ii,means))]

def classifier(method,c):return LogisticRegression(C=c,penalty='l1' if method=='PAPER' else 'l2',solver='liblinear',max_iter=3000,tol=1e-7,random_state=573)
def apply_fallback(method,p,no_news,price):
 p=np.asarray(p).copy()
 if method in ('FULL','FINBERT','MODERN'):p[no_news]=price[no_news]
 return p
def check_sources():
 manifest=json.loads((PRIVATE/'manifest.json').read_text())
 if manifest['sources']!=hashes():raise ValueError('source/protocol fingerprint mismatch')
 for name,h in manifest['artifacts'].items():
  if sha(PRIVATE/name)!=h:raise ValueError('prepared artifact mismatch: '+name)
 return manifest
