"""Frozen finite text/joint experiment primitives. Raw content stays private."""
import sys,json,re,html,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import *
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
PUBLIC=ROOT/'outputs/stock_text_joint_4h/v1'
WORK=ROOT/'work/stock-data/text_joint_4h/v1'
SEM=ROOT/'work/stock-data/full_semantics_4h/v1/semantic'
CLASSIC=ROOT/'work/stock-data/full_semantics_4h/v1/classical'
TOKEN=re.compile(r"[$€£]?\d+(?:[,.]\d+)*(?:%|[bBmM])?|[A-Za-z]+(?:['’][A-Za-z]+)?")
BAD=re.compile(r'^(?:advertisement|privacy policy|terms of use|copyright|subscribe|sign up now|click here|newsletter)\b',re.I)
def clean(title,body):
 # Block HTML and physical lines remain hard boundaries. Only entire obvious
 # template lines are excluded; URLs removed, never arbitrary financial clauses.
 body=re.sub(r'<(?:script|style)\b[^>]*>.*?</(?:script|style)>',' ',str(body),flags=re.I|re.S)
 body=re.sub(r'</?(?:p|div|br|li|h[1-6]|tr)\b[^>]*>','\n',body,flags=re.I)
 body=html.unescape(re.sub(r'<[^>]+>',' ',body))
 parts=[];removed=[]
 for origin,text in [('title',str(title)),('body',body)]:
  for lineno,line in enumerate(text.splitlines()):
   line=re.sub(r'\s+',' ',re.sub(r'https?://\S+',' ',line)).strip()
   if not line:continue
   if origin=='body' and BAD.search(line):removed.append({'line':lineno,'hash':digest(line)});continue
   # Conservative hard punctuation boundaries. A decimal point is not a boundary.
   for sentence in re.split(r'(?<!\d)[.!?;]+|[.!?;]+(?!\d)',line):
    tokens=[m.group().lower() for m in TOKEN.finditer(sentence)]
    if tokens:parts.append(dict(origin=origin,line=lineno,text=sentence.strip(),tokens=tokens))
 return parts,removed

def features(parts,bigrams):
 out=[]
 for p in parts:
  ts=p['tokens'];out.extend(ts)
  if bigrams:out.extend(a+' '+b for a,b in zip(ts,ts[1:]))
 return out
class Analyzer:
 def __init__(self,bigrams=False):self.bigrams=bigrams
 def __call__(self,doc):return doc['b' if self.bigrams else 'u']
class TrackedTfidf(TfidfVectorizer):
 def fit_transform(self,raw_documents,y=None):
  docs=list(raw_documents);self.fit_row_ids_=[x['row_id'] for x in docs]
  return super().fit_transform(docs,y)
def text_model(method,c,seed):
 v=TrackedTfidf(analyzer=Analyzer(method!='A1'),min_df=3,max_features=5000,sublinear_tf=True,norm='l2',lowercase=False)
 if method!='A3':return Pipeline([('text',v),('classifier',classifier('FULL',c))])
 return CalibratedClassifierCV(Pipeline([('text',v),('classifier',LinearSVC(C=c,max_iter=10000,random_state=seed))]),method='sigmoid',cv=StratifiedKFold(3,shuffle=True,random_state=seed),ensemble=True)
def read_csv(p):return pd.read_csv(p,float_precision='round_trip')
def semantic(d):
 articles=[json.loads(s) for s in (SEM/'articles.jsonl').read_text().splitlines()]
 z=dict(np.load(SEM/'FINBERT_articles.npz'));assert z['keys'].tolist()==[r['article_id'] for r in articles]
 lookup={(r['symbol'],r['record_key']):i for i,r in enumerate(articles)}
 ids=[[lookup[r.symbol,k] for k in r.news_record_keys.split('|') if k and z['has_evidence'][lookup[r.symbol,k]]] for r in d.itertuples()]
 means=np.array([z['vectors'][ix].mean(0) if ix else np.zeros(z['vectors'].shape[1]) for ix in ids])
 return z['vectors'],ids,means,np.array([bool(x) for x in ids])
class Joint:
 def fit(self,d,tr,emb,ids,means,method):
  self.method=method;self.train=list(map(int,tr));self.full=Features('FULL').fit(d,tr,{},[],{})
  self.article_ids=sorted({j for i in tr for j in ids[i]})
  self.pca=PCA(16,svd_solver='randomized',random_state=573).fit(emb[self.article_ids]) if method=='B1' else None
  z=self.pca.transform(means) if self.pca is not None else means.copy()
  gate=np.array([bool(x) for x in ids]);z[~gate]=0
  self.scale=Scale().fit(z[tr]);return self
 def transform(self,d,ix,means,gate,rho):
  full=self.full.transform(d,ix,{})
  if rho==0:return full
  z=self.pca.transform(means[ix]) if self.pca is not None else means[ix].copy()
  z[~gate[ix]]=0;z=self.scale.transform(z);z[~gate[ix]]=0
  return sparse.hstack([full,rho*z]).tocsr()
def choose(records,method):
 rs=[r for r in records if r['method']==method]
 candidates=sorted({(float(r['C']),float(r['rho'])) for r in rs})
 return min(candidates,key=lambda cr:(-np.mean([r['BA'] for r in rs if (r['C'],r['rho'])==cr]),np.mean([r['Brier'] for r in rs if (r['C'],r['rho'])==cr]),cr[1],cr[0]))
def shrink(p,has,a):
 q=np.asarray(p).copy();q[has]=.5+a*(q[has]-.5)
 assert np.array_equal(q>=.5,np.asarray(p)>=.5),'floating boundary changed direction'
 return q
def protect_check():
 check_sources();p=WORK/'protected.json';h=json.loads(p.read_text())
 for f,v in h.items():
  if sha(ROOT/f)!=v:raise ValueError('Protected artifact changed: '+f)
 return len(h)
def cache_check(seal):
 actual={'protocol':sha(PUBLIC/'protocol.json'),'inputs':sha(WORK/'inputs.joblib'),'protected':sha(WORK/'protected.json'),'code':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('train.py')]}}
 if seal!=actual:raise ValueError('Cache input/config/code fingerprint mismatch')
 return actual
def block_guard(root,seal):
 if (root/'complete.json').exists():
  old=json.loads((root/'complete.json').read_text());assert old['seal']==seal
  for n,h in old['files'].items():assert sha(root/n)==h
  return False
 if root.exists():raise RuntimeError('Incomplete run preserved: '+str(root))
 root.mkdir(parents=True);return True
