"""Finite preprocessing, grouped SVM and distributional aggregation study."""
import sys,json,time,re,html,zipfile,warnings,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import *
import joblib
from functools import lru_cache
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from nltk.stem.snowball import EnglishStemmer
from threadpoolctl import threadpool_limits
PUB=ROOT/'outputs/stock_representation_4h/v1'
WORK=ROOT/'work/stock-data/representation_4h/v1'
SEM=ROOT/'work/stock-data/full_semantics_4h/v1/semantic'
CLASSIC=ROOT/'work/stock-data/full_semantics_4h/v1/classical'
VARIANTS=['FREQUENCY','NEGATION','NUMERIC','NO_STEM']
BAD=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',re.I)
NUM=re.compile(r'(?<![\w])[$€£]?\d+(?:[,.]\d+)*(?:%|[bBmM])?(?![\w])')
STEM=EnglishStemmer()
@lru_cache(maxsize=200000)
def stem(w):return STEM.stem(w)
def article_views(title,body):
 text=html.unescape(re.sub(r'<[^>]+>',' ',body));text=' '.join(x for x in text.splitlines() if not BAD.search(x));text=re.sub(r'https?://\S+',' ',text)
 combined=text+' '+title
 words=[w for w in re.findall('[a-z]+',combined.lower()) if len(w)>1]
 usual=[stem(w) for w in words if w not in ENGLISH_STOP_WORDS]
 # Digit tokens are added without changing the historical alphabetic extraction.
 numbers=['num_'+re.sub(r'\W',lambda m: {'$':'usd','€':'eur','£':'gbp','%':'pct',',':'comma','.':'dot'}.get(m[0],'_'),m[0].lower()) for m in NUM.finditer(combined)]
 return {'CONTROL':usual,'FREQUENCY':usual,'NEGATION':[stem(w) for w in words if w not in ENGLISH_STOP_WORDS or w in {'not','no','nor'}], 'NUMERIC':usual+numbers,'NO_STEM':[w for w in words if w not in ENGLISH_STOP_WORDS]}
def assemble(views,keys,method):
 vals=[w for k in keys for w in views[k][method]]
 return ' '.join(sorted(vals if method=='FREQUENCY' else set(vals)))
def read(p):return pd.read_csv(p,float_precision='round_trip')
def select(rs):
 return min(CS,key=lambda c:(-np.mean([r['BA'] for r in rs if r['C']==c]),np.mean([r['Brier'] for r in rs if r['C']==c]),c))
def calibration(a,d,seed,groups=None):
 if groups is None:cv=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(a)),d.iloc[a].label))
 else:cv=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(a)),d.iloc[a].label,groups[a]))
 for x,y in cv:
  assert not set(x)&set(y) and set(d.iloc[np.asarray(a)[x]].label)=={0,1} and set(d.iloc[np.asarray(a)[y]].label)=={0,1}
  if groups is not None:assert not set(groups[np.asarray(a)[x]])&set(groups[np.asarray(a)[y]])
 return cv
class Tracked(TfidfVectorizer):
 def fit_transform(self,docs,y=None):
  docs=list(docs);self.document_hashes_=[digest(s) for s in docs]
  return super().fit_transform(docs,y)
def svm(c,seed,cv):
 return CalibratedClassifierCV(Pipeline([('text',Tracked(min_df=3,max_features=5000,sublinear_tf=True)),('svm',LinearSVC(C=c,max_iter=10000,random_state=seed))]),method='sigmoid',cv=cv,ensemble=True)
def random_blocks(d):
 outer=read(PRIVATE/'outer_folds.csv');inner=json.loads((PRIVATE/'inner_folds.json').read_text());ans=[]
 for seed in SEEDS:
  for stock in ['AAPL','AMZN']:
   for fold in range(10):
    b=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].tolist();a=sorted(set(np.flatnonzero(d.symbol==stock))-set(b))
    ins=[(s['train'],s['validation']) for s in inner if (s['seed'],s['symbol'],s['fold'])==(seed,stock,fold)]
    ans.append(dict(name=f'{seed}_{stock}_{fold}',seed=seed,stock=stock,train=a,test=b,inner=ins))
 return ans

def seal_check():
 s=json.loads((WORK/'seal.json').read_text())
 for p,h in s['files'].items():assert sha(ROOT/p)==h,'seal mismatch '+p
 return sha(WORK/'seal.json')
def begin_block(stage,b):
 root=WORK/stage/b['name'];seal=seal_check()
 if (root/'complete.json').exists():
  c=json.loads((root/'complete.json').read_text());assert c['seal']==seal
  for f,h in c['files'].items():assert sha(root/f)==h
  return None
 assert not root.exists(),'Incomplete block preserved '+str(root)
 root.mkdir(parents=True);return root

def savefit(root,name,model,a,b,x,y,labels,ledger,metadata):
 t=time.monotonic();model.fit(x,labels);p=model.predict_proba(y)[:,1]
 obj=dict(model=model,train=list(map(int,a)),evaluation=list(map(int,b)),**metadata)
 joblib.dump(obj,root/(name+'.joblib'),compress=3)
 ledger.append(dict(file=name+'.joblib',seconds=time.monotonic()-t,train=obj['train'],evaluation=obj['evaluation'],**{k:v for k,v in metadata.items() if k!='transform'}))
 return p

def finish(root,b,out,ledger,records,inner):
 out.to_csv(root/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(records).to_csv(root/'selection.csv',index=False,float_format='%.17g')
 pd.DataFrame(inner).to_csv(root/'inner_predictions.csv',index=False,float_format='%.17g');dump(root/'training.json',ledger)
 dump(root/'complete.json',dict(seal=sha(WORK/'seal.json'),files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
 print(root.parent.name,b['name'],'complete',len(ledger),'containers',flush=True)

def train_text(stage):
 seal_check();d,*_=load();inp=joblib.load(WORK/'inputs.joblib');groups=inp['groups'];blocks=inp['random'] if stage=='preprocess' else inp['grouped'];methods=VARIANTS if stage=='preprocess' else ['GROUP_SVM']
 for b in blocks:
  root=begin_block(stage,b)
  if root is None:continue
  a=np.array(b['train']);e=np.array(b['test']);seed=b['seed'];ledger=[];records=[];inner=[]
  base=PRIVATE/('baseline' if stage=='preprocess' else 'diagnostic_baseline')/b['name'];old=read(base/('outer_predictions_SEALED.csv' if stage=='preprocess' else 'predictions_SEALED.csv'))
  assert old.row_id.tolist()==d.iloc[e].row_id.tolist();out=old.copy();out['seed']=seed;out['block']=b['name']
  for method in methods:
   docs=np.array(inp['documents'][method] if stage=='preprocess' else d.stem_body.tolist(),dtype=object)
   for inf,(aa,bb) in enumerate(b['inner']):
    cv=calibration(aa,d,seed,None if stage=='preprocess' else groups)
    for c in CS:
     p=savefit(root,f'{method}_inner{inf}_{c}',svm(c,seed,cv),aa,bb,docs[aa].tolist(),docs[bb].tolist(),d.iloc[aa].label,ledger,dict(method=method,C=c,inner_fold=inf,cv=[[x.tolist(),y.tolist()] for x,y in cv]))
     records.append(dict(method=method,C=c,inner_fold=inf,**metric(d.iloc[bb].label,p)))
     inner.extend(dict(method=method,C=c,inner_fold=inf,index=int(i),p=float(q)) for i,q in zip(bb,p))
   c=select([r for r in records if r['method']==method]);cv=calibration(a,d,seed,None if stage=='preprocess' else groups)
   p=savefit(root,f'{method}_selected',svm(c,seed,cv),a,e,docs[a].tolist(),docs[e].tolist(),d.iloc[a].label,ledger,dict(method=method,C=c,inner_fold=None,cv=[[x.tolist(),y.tolist()] for x,y in cv]))
   out[method+'_DIRECT']=p;out[method]=np.where(d.iloc[e].has_news,p,old.PRICE)
  if stage=='preprocess':
   prior=read(CLASSIC/b['name']/'predictions.csv');assert prior.row_id.tolist()==out.row_id.tolist();out['CONTROL']=np.where(d.iloc[e].has_news,prior.TFIDF_SVM,old.PRICE)
  finish(root,b,out,ledger,records,inner)

class SetTransform:
 """Fixed RFF, train-only geometry, same article/chunk weights for both orders."""
 def fit(self,d,a,data):
  self.train=list(map(int,a));self.v=Tracked(min_df=3,max_features=5000,sublinear_tf=True).fit(d.iloc[a].stem_body)
  self.price=Scale().fit(d.iloc[a][OLD]);self.chunk_ids=sorted({int(c) for i in a for c in data['window_chunks'][i]})
  v=data['vectors'];self.center=v[self.chunk_ids].mean(0);self.std=v[self.chunk_ids].std(0);self.std=np.where(self.std>1e-12,self.std,1.)
  rng=np.random.default_rng(573);ix=rng.choice(self.chunk_ids,min(1024,len(self.chunk_ids)),replace=False);z=(v[ix]-self.center)/self.std
  jj=rng.permutation(len(ix));dist=np.sum((z-z[jj])**2,axis=1);self.sigma=float(np.sqrt(np.median(dist[dist>0])))
  self.omega=rng.normal(size=(768,256))/self.sigma;self.phase=rng.uniform(0,2*np.pi,256)
  return self
 def representations(self,d,data):
  z=(data['vectors']-self.center)/self.std;phi=np.sqrt(2/256)*np.cos(z@self.omega+self.phase)
  wm=np.stack([z[ix].T@w if len(ix) else np.zeros(768) for ix,w in zip(data['window_chunks'],data['weights'])])
  each=np.stack([phi[ix].T@w if len(ix) else np.zeros(256) for ix,w in zip(data['window_chunks'],data['weights'])])
  after=np.sqrt(2/256)*np.cos(wm@self.omega+self.phase);gate=np.array([len(x)>0 for x in data['window_chunks']]);after[~gate]=0;wm[~gate]=0
  return {'MEAN_LINEAR':normalize(wm),'MEAN_THEN_MAP':normalize(after),'MAP_THEN_MEAN':normalize(each)}
 def base(self,d,ii):return sparse.hstack([self.v.transform(d.iloc[ii].stem_body),self.price.transform(d.iloc[ii][OLD])/4]).tocsr()

def train_set():
 seal_check();d,*_=load();inp=joblib.load(WORK/'inputs.joblib');data=joblib.load(WORK/'chunks.joblib');methods=['SET_BASE','MEAN_LINEAR','MEAN_THEN_MAP','MAP_THEN_MEAN']
 for b in inp['random']:
  root=begin_block('aggregation',b)
  if root is None:continue
  a=np.array(b['train']);e=np.array(b['test']);base=PRIVATE/'baseline'/b['name'];old=read(base/'outer_predictions_SEALED.csv');price=read(base/'PRICE_selection_oof_NOT_META_TRAIN.csv').set_index('index').p
  out=old.copy();out['seed']=b['seed'];out['block']=b['name'];ledger=[];records=[];inner=[]
  for inf,(aa,bb) in enumerate(b['inner']):
   tf=SetTransform().fit(d,aa,data);reps=tf.representations(d,data);x=tf.base(d,aa);y=tf.base(d,bb)
   for method in methods:
    xx=x if method=='SET_BASE' else sparse.hstack([x,reps[method][aa]]).tocsr();yy=y if method=='SET_BASE' else sparse.hstack([y,reps[method][bb]]).tocsr()
    for c in CS:
     p=savefit(root,f'{method}_inner{inf}_{c}',classifier('FULL',c),aa,bb,xx,yy,d.iloc[aa].label,ledger,dict(method=method,C=c,inner_fold=inf,transform=tf))
     p=np.where(d.iloc[bb].has_news,p,price.loc[bb]);records.append(dict(method=method,C=c,inner_fold=inf,**metric(d.iloc[bb].label,p)))
     inner.extend(dict(method=method,C=c,inner_fold=inf,index=int(i),p=float(q)) for i,q in zip(bb,p))
  tf=SetTransform().fit(d,a,data);reps=tf.representations(d,data);x=tf.base(d,a);y=tf.base(d,e)
  for method in methods:
   c=select([r for r in records if r['method']==method]);xx=x if method=='SET_BASE' else sparse.hstack([x,reps[method][a]]).tocsr();yy=y if method=='SET_BASE' else sparse.hstack([y,reps[method][e]]).tocsr()
   p=savefit(root,method+'_selected',classifier('FULL',c),a,e,xx,yy,d.iloc[a].label,ledger,dict(method=method,C=c,inner_fold=None,transform=tf))
   out[method+'_DIRECT']=p;out[method]=np.where(d.iloc[e].has_news,p,old.PRICE)
  prior=read(CLASSIC/b['name']/'predictions.csv');out['CONTROL']=np.where(d.iloc[e].has_news,prior.TFIDF_SVM,old.PRICE)
  finish(root,b,out,ledger,records,inner)

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preprocess','grouped','aggregation']);args=ap.parse_args()
 with (WORK/'run.lock').open('a') as f,threadpool_limits(2),warnings.catch_warnings():
  fcntl.flock(f,fcntl.LOCK_EX);warnings.simplefilter('ignore',FutureWarning)
  train_set() if args.stage=='aggregation' else train_text(args.stage)
