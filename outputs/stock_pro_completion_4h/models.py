"""Frozen finite mechanisms; all transformations fit within current training scope."""
import os,sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_representation_4h'))
import engine as previous
from engine import *
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit,logit
from scipy.optimize import minimize
OUT=ROOT/'outputs/stock_pro_completion_4h/v1'
LOCAL=ROOT/'work/stock-data/pro_completion_4h/v1'
CHECKPOINT=ROOT/'work/stock-data/goal60_4h/tabpfn_model/tabpfn-v2.5-classifier-v2.5_default.ckpt'
STATE=['return_mean','return_std','range_1']
DATA=None
TIMES=[]
def initialize():
 global DATA
 if DATA is None:
  d,emb,ids,means=load();data=joblib.load(previous.WORK/'chunks.joblib');inp=joblib.load(LOCAL/'inputs.joblib')
  av=np.load(previous.SEM/'FINBERT_articles.npz');arts=[json.loads(s) for s in (previous.SEM/'articles.jsonl').read_text().splitlines()];lookup={(r['symbol'],r['record_key']):i for i,r in enumerate(arts)}
  aid=[[lookup[r.symbol,k] for k in r.news_record_keys.split('|') if k and arts[lookup[r.symbol,k]]['chunks']] for r in d.itertuples()]
  mv=np.array([av['vectors'][ii].mean(0) if ii else np.zeros(768) for ii in aid]);gate=np.array([bool(ii) for ii in aid])
  DATA=dict(d=d,emb=emb,ids=ids,means=means,chunks=data,inp=inp,av=av['vectors'],aid=aid,mv=mv,gate=gate)
 return DATA

def indices(x):return np.asarray(x).reshape(-1).astype(int)
class Lexical(BaseEstimator,TransformerMixin):
 def __init__(self,kind='UNI'):self.kind=kind
 def fit(self,x,y=None):
  ii=indices(x);db=initialize();self.train=ii.tolist()
  docs=np.array(db['inp']['documents'][self.kind],object)[ii]
  self.v=TfidfVectorizer(min_df=3,max_features=5000,sublinear_tf=True,tokenizer=str.split,token_pattern=None,lowercase=False).fit(docs)
  return self
 def transform(self,x):return self.v.transform(np.array(initialize()['inp']['documents'][self.kind],object)[indices(x)])
class NBRatio(BaseEstimator,TransformerMixin):
 def fit(self,x,y):
  z=(x>0).astype(float);p=1+np.asarray(z[np.asarray(y)==1].sum(0)).ravel();q=1+np.asarray(z[np.asarray(y)==0].sum(0)).ravel();self.r=np.log((p/p.sum())/(q/q.sum()));return self
 def transform(self,x):return (x>0).astype(float).multiply(self.r).tocsr()

# A train-scope cache avoids rebuilding the SAME unsupervised geometry per C.
GEOMETRY={}
class KernelFeatures(BaseEstimator,TransformerMixin):
 def __init__(self,kind='K_WP',gamma=1.):self.kind=kind;self.gamma=gamma
 def fit(self,x,y=None):
  ii=indices(x);key=tuple(ii);db=initialize();d=db['d']
  if key not in GEOMETRY:
   tf=previous.SetTransform().fit(d,ii,db['chunks']);tf.state=Scale().fit(d.iloc[ii][STATE]);GEOMETRY[key]=tf
  self.tf=GEOMETRY[key];self.train=ii.tolist();return self
 def transform(self,x):
  ii=indices(x);db=initialize();d=db['d'];tf=self.tf;data=db['chunks'];key=tuple(self.train)
  if key not in MATRICES:
   z=(data['vectors']-tf.center)/tf.std;ph=np.sqrt(2/256)*np.cos(z@tf.omega+tf.phase)
   mean=np.stack([z[jj].T@w if len(jj) else np.zeros(768) for jj,w in zip(data['window_chunks'],data['weights'])])
   each=np.stack([ph[jj].T@w if len(jj) else np.zeros(256) for jj,w in zip(data['window_chunks'],data['weights'])])
   after=np.sqrt(2/256)*np.cos(mean@tf.omega+tf.phase);after[~db['gate']]=0
   MATRICES[key]={'base':tf.base(d,np.arange(len(d))),'K_MEAN':mean/np.sqrt(768),'K_AFTER':after,'K_SET':each,'state':tf.state.transform(d[STATE])/np.sqrt(3)}
  vals=MATRICES[key];base=vals['base'][ii]
  if self.kind=='K_WP':return base
  sem=vals['K_SET' if self.kind=='K_INTERACTION' else self.kind][ii];blocks=[base,sparse.csr_matrix(sem)]
  if self.kind=='K_INTERACTION':blocks.append(sparse.csr_matrix(np.sqrt(self.gamma)*np.einsum('ij,ik->ijk',sem,vals['state'][ii]).reshape(len(ii),-1)))
  return sparse.hstack(blocks).tocsr()
MATRICES={}

def svm_model(kind,c,seed,cv,gamma=1.):
 if kind in ['UNI','BIGRAM','NB_BIGRAM']:
  steps=[('features',Lexical('BIGRAM' if kind=='NB_BIGRAM' else kind))]
  if kind=='NB_BIGRAM':steps.append(('nb',NBRatio()))
  steps.append(('classifier',LinearSVC(C=c,max_iter=10000,random_state=seed)))
 else:steps=[('features',KernelFeatures(kind,gamma)),('classifier',SVC(C=c,kernel='linear',max_iter=10000,random_state=seed))]
 return CalibratedClassifierCV(Pipeline(steps),method='sigmoid',cv=cv,ensemble=True)

def timed_fit(m,x,y,tag):
 t=time.monotonic();m.fit(x,y);TIMES.append(dict(tag=tag,seconds=time.monotonic()-t,n=len(y)));return m

class Table:
 def fit(self,a,y):
  db=initialize();d=db['d'];self.train=list(map(int,a));self.v=TfidfVectorizer(min_df=3,max_features=5000,sublinear_tf=True,tokenizer=str.split,token_pattern=None,lowercase=False)
  docs=np.array(db['inp']['documents']['BIGRAM'],object);v=self.v.fit_transform(docs[a]);scores=np.nan_to_num(chi2(v,y)[0],nan=-np.inf);self.keep=np.lexsort((self.v.get_feature_names_out(),-scores))[:128]
  self.price=Scale().fit(d.iloc[a][OLD]);self.article_ids=sorted({j for i in a for j in db['aid'][i]});self.pca=PCA(n_components=32,svd_solver='randomized',random_state=573).fit(db['av'][self.article_ids]);sem=self.pca.transform(db['mv'][a]);self.sem_scale=Scale().fit(sem[db['gate'][a]])
  self.meta=Scale().fit(np.c_[np.log1p(d.iloc[a].news_count),db['gate'][a]]);return self
 def transform(self,ii,semantic):
  db=initialize();d=db['d'];docs=np.array(db['inp']['documents']['BIGRAM'],object)
  x=np.c_[self.v.transform(docs[ii])[:,self.keep].toarray(),self.price.transform(d.iloc[ii][OLD]),self.meta.transform(np.c_[np.log1p(d.iloc[ii].news_count),db['gate'][ii]])]
  if semantic:
   sem=self.sem_scale.transform(self.pca.transform(db['mv'][ii]));sem[~db['gate'][ii]]=0;x=np.c_[x,sem]
  return x

def tabpfn(seed):
 os.environ['TABPFN_DISABLE_TELEMETRY']='1';sys.path.insert(0,str(ROOT/'work/stock-data/goal60_4h/runtime'))
 from tabpfn import TabPFNClassifier
 return TabPFNClassifier(n_estimators=8,device='cpu',model_path=str(CHECKPOINT),random_state=seed,n_preprocessing_jobs=1)

class Offset:
 def fit(self,z,y,p,c):
  off=logit(np.clip(p,1e-6,1-1e-6));y=np.asarray(y)
  def objective(w):
   eta=off+z@w;return np.logaddexp(0,eta).sum()-y@eta+.5/c*(w@w),z.T@(expit(eta)-y)+w/c
  r=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8});assert r.success,r.message
  self.coef=r.x;self.iterations=r.nit;return self
 def predict(self,z,p,gate):
  out=np.asarray(p).copy();out[gate]=expit(logit(np.clip(out[gate],1e-6,1-1e-6))+z[gate]@self.coef);return out

class Semantic16:
 def fit(self,a):
  db=initialize();self.train=list(map(int,a));self.article_ids=sorted({j for i in a for j in db['aid'][i]});self.pca=PCA(n_components=16,svd_solver='randomized',random_state=573).fit(db['av'][self.article_ids]);self.scale=Scale().fit(self.pca.transform(db['mv'][a])[db['gate'][a]]);return self
 def transform(self,ii):
  db=initialize();z=self.scale.transform(self.pca.transform(db['mv'][ii]));z[~db['gate'][ii]]=0;return z

class TabPFNRemote:
 def __init__(self,path,seed):self.path=str(path);self.seed=seed
 def fit(self,x,y):
  self.x=np.asarray(x);self.y=np.asarray(y);return self
 def predict_proba(self,x):
  import subprocess,tempfile
  mode='predict' if Path(self.path).exists() else 'fit'
  with tempfile.TemporaryDirectory(dir=LOCAL if LOCAL.exists() else None) as t:
   p=Path(t);np.savez(p/'input.npz',x=self.x,y=self.y,evaluation=x)
   env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'work/stock-data/goal60_4h/runtime');env['TABPFN_DISABLE_TELEMETRY']='1'
   cmd=[sys.executable,str(Path(__file__).with_name('tabpfn_worker.py')),mode,str(p/'input.npz'),self.path,str(p/'output.npy'),'--seed',str(self.seed)]
   subprocess.run(cmd,env=env,check=True,stdout=subprocess.DEVNULL);q=np.load(p/'output.npy');self.last_execution=json.loads((p/'output.npy.json').read_text());return np.c_[1-q,q]
