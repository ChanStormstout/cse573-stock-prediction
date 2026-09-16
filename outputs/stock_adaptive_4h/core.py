"""Small, chronological four-hour news corrections. No labels at inference."""
from pathlib import Path
import sys, json, hashlib, itertools
import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
OLD = B.parent / 'stock_integrated_4h'
sys.path.insert(0, str(OLD))
import run as legacy
Transform = legacy.Transform
PRICE = legacy.PRICE
CS = [.01, .1, 1.]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, obj):
    def encode(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        return str(value)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=encode, allow_nan=False)+'\n')
def check_hashes(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest: raise ValueError(f'Fingerprint mismatch: {path}')
def temporal_check(train, evaluation):
    if len(train) and train.end_utc.max() >= evaluation.cutoff_utc.min():
        raise ValueError('Future/unmatured label in training')
def score(y, p): return legacy.metric(y, p)
def mean_months(d, p):
    q=d[['label','month']].copy(); q['p']=np.asarray(p)
    v=[score(x.label,x.p) for _,x in q.groupby('month')]
    return {k:float(np.mean([r[k] for r in v if r[k] is not None])) for k in ['BA','Brier']}
def add_delta(q, delta, present):
    # Some installed pandas/NumPy combinations ignore np.array(Series, copy=True).
    # Copy after conversion so corrections cannot alter upstream probabilities.
    p=np.asarray(q,dtype=float).copy(); g=np.asarray(present,dtype=bool)
    p[g]=expit(logit(np.clip(p[g],1e-8,1-1e-8))+np.asarray(delta)[g])
    return p

class NewsTransform:
    def __init__(self, kind): self.kind=kind
    def fit(self, d, keys, embeddings):
        self.train_keys_=d.key.tolist()
        if self.kind=='body':
            self.vectorizer=CountVectorizer(binary=True,min_df=3)
            try:
                x=self.vectorizer.fit_transform(d.stem_body)
                terms=self.vectorizer.get_feature_names_out()
                s=chi2(x,d.label)[0]
                self.keep=np.lexsort((terms,-np.nan_to_num(s,nan=-np.inf)))[:100]
                self.empty=False
            except ValueError as exc:
                if 'vocabulary' not in str(exc) and 'terms remain' not in str(exc): raise
                self.empty=True
        else:
            lookup={k:i for i,k in enumerate(keys)}
            self.article_keys_=sorted({k for s in d.news_record_keys for k in s.split('|') if k})
            if len(self.article_keys_)<16: raise ValueError('PCA16 requires 16 unique training articles')
            self.pca=PCA(n_components=16,svd_solver='randomized',random_state=573).fit(embeddings[[lookup[k] for k in self.article_keys_]])
            self.scaler=StandardScaler().fit(self.raw_semantic(d,keys,embeddings))
        return self
    def raw_semantic(self,d,keys,embeddings):
        lookup={k:i for i,k in enumerate(keys)}; rows=[]
        for s in d.news_record_keys:
            ids=[lookup[k] for k in s.split('|') if k]
            rows.append(self.pca.transform(embeddings[ids].mean(0,keepdims=True))[0] if ids else np.zeros(16))
        return np.asarray(rows)
    def transform(self,d,keys,embeddings):
        if self.kind=='body':
            return np.zeros((len(d),1)) if self.empty else self.vectorizer.transform(d.stem_body)[:,self.keep].toarray().astype(float)
        return self.scaler.transform(self.raw_semantic(d,keys,embeddings))

class OffsetModel:
    """Convex raw offset fit; clip correction only during candidate evaluation/inference."""
    def __init__(self,C=.1,joint=False,news_bias='free'): self.C=C; self.joint=joint;self.news_bias=news_bias
    def design(self,z,g,x=None):
        a=np.c_[z*np.asarray(g)[:,None],g if getattr(self,'news_bias','free')=='free' else np.zeros(len(z))]
        if self.joint: a=np.c_[a,x,np.ones(len(z))]
        return a
    def objective(self,w,a,y,o,penalty):
        t=o+a@w
        return float(np.sum(np.logaddexp(0,t)-y*t)+np.dot(penalty*w,w)/(2*self.C)), a.T@(expit(t)-y)+penalty*w/self.C
    def fit(self,z,g,q,y,x=None):
        a=self.design(z,g,x); self.penalty_=np.ones(a.shape[1]); self.penalty_[z.shape[1]]=0
        if self.joint: self.penalty_[-1]=0
        o=logit(np.clip(np.asarray(q),1e-8,1-1e-8)); y=np.asarray(y)
        r=minimize(self.objective,np.zeros(a.shape[1]),args=(a,y,o,self.penalty_),jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7})
        if not r.success: raise RuntimeError(str(r.message))
        self.coef_=r.x; self.iterations_=int(r.nit); self.loss_=float(r.fun); return self
    def correction(self,z,g,x=None):
        raw=self.design(z,g,x)@self.coef_
        return np.clip(raw,-1,1),raw

class Calibration:
    def fit(self,p,y):
        t=logit(np.clip(np.asarray(p),1e-8,1-1e-8)); y=np.asarray(y)
        def f(w):
            a,b=w; s=a*t+b; e=expit(s)-y
            return np.mean(np.logaddexp(0,s)-y*s)+.5*((a-1)**2+b*b), np.array([np.mean(e*t)+a-1,np.mean(e)+b])
        r=minimize(f,[1.,0.],jac=True,bounds=[(.5,2),(-.5,.5)],method='L-BFGS-B')
        if not r.success: raise RuntimeError(r.message)
        self.coef_=r.x; self.iterations_=int(r.nit); return self
    def predict(self,p): return expit(self.coef_[0]*logit(np.clip(p,1e-8,1-1e-8))+self.coef_[1])

def static_grid():
    return [np.array(x)/4 for x in itertools.product(range(5),repeat=3) if sum(x)==4]
def combine(d,w):
    w=np.asarray(w)
    delta=d.db.to_numpy()*w[...,1]+d.ds.to_numpy()*w[...,2]
    return add_delta(d.price,delta,d.has_news)
def select_static(d):
    if len(d)==0: return [1.,0.,0.],[]
    ref=mean_months(d,d.price); grid=[]
    for w in static_grid():
        m=mean_months(d,combine(d,w)); grid.append({'weights':w.tolist(),**m,'eligible':m['Brier']<=ref['Brier']+.002})
    chosen=sorted([v for v in grid if v['eligible']],key=lambda a:(-a['BA'],a['Brier'],sum(x>0 for x in a['weights']),-a['weights'][0],a['weights']))[0]
    return chosen['weights'],grid

CONTEXT=['log_count','log_age','age_unknown','past_volatility','base_confidence','expert_disagreement','expert_agreement','hour']
def context(d):
    return np.c_[np.log1p(d.news_count),np.log1p(d.age_median),d.age_unknown,d.return_std,np.abs(d.price-.5),np.abs(d.db-d.ds),(np.sign(d.db)==np.sign(d.ds)).astype(float),d.ny_hour]

class Router:
    def __init__(self,alpha=10): self.alpha=alpha
    def fit(self,d):
        self.support_={'rows':len(d),'common_days':len(set(d[d.symbol=='AAPL'].day)&set(d[d.symbol=='AMZN'].day))}
        self.stock_support_={s:{'news_rows':int(g.has_news.sum()),'news_days':g.loc[g.has_news.astype(bool),'day'].nunique()} for s,g in d.groupby('symbol')}
        self.ready_=self.support_['rows']>=200 and self.support_['common_days']>=60
        self.train_keys_=d.key.tolist()
        if not self.ready_: return self
        z=context(d); self.fill_=np.nanmedian(z,axis=0);self.fill_=np.nan_to_num(self.fill_)
        z=np.where(np.isfinite(z),z,self.fill_); self.lo_=np.quantile(z,.01,axis=0);self.hi_=np.quantile(z,.99,axis=0)
        self.scale=StandardScaler().fit(z); a=np.c_[self.scale.transform(z),(d.symbol=='AMZN').astype(float)]
        y=d.label.to_numpy(); base=(y-d.price.to_numpy())**2
        target=np.c_[(y-d.body.to_numpy())**2-base,(y-d.semantic.to_numpy())**2-base]
        self.model=Ridge(alpha=self.alpha).fit(a,target)
        return self
    def weights(self,d,static,rho):
        w=np.tile(np.asarray(static),(len(d),1)); reason=np.full(len(d),'STATIC_INSUFFICIENT_HISTORY',dtype=object)
        if self.ready_:
            z=context(d); z=np.where(np.isfinite(z),z,self.fill_);bad=((z<self.lo_)|(z>self.hi_)).sum(1)>=3
            a=np.c_[self.scale.transform(z),(d.symbol=='AMZN').astype(float)]
            advantage=np.clip(self.model.predict(a),-1,1)
            dynamic=softmax(-np.c_[np.zeros(len(d)),advantage]/.05,axis=1)
            for i,s in enumerate(d.symbol):
                support=self.stock_support_.get(s,{'news_rows':0,'news_days':0})
                if support['news_rows']<30 or support['news_days']<20: continue
                if bad[i]: reason[i]='STATIC_CONTEXT_OUTSIDE_RANGE'; continue
                w[i]=(1-rho)*w[i]+rho*dynamic[i];reason[i]='CONDITIONAL' if rho else 'STATIC_SELECTED'
        empty=d.has_news.to_numpy()==0;w[empty]=[1.,0.,0.];reason[empty]='NO_NEWS_BASE'
        return w,reason

def prepare_inputs():
    p=OLD/'prepared'; manifest=json.loads((p/'manifest.json').read_text())
    check_hashes({str(p/name):value for name,value in manifest['prepared_hashes'].items()})
    check_hashes(manifest['archive_hashes'])
    d=pd.read_pickle(p/'data.pkl').copy()
    for c in ['start_utc','end_utc','cutoff_utc','history_end']: d[c]=pd.to_datetime(d[c],utc=True)
    d['key']=d.symbol+'|'+d.start_utc.astype(str);d['month']=d.start_utc.dt.strftime('%Y-%m');d['day']=d.start_utc.dt.strftime('%Y-%m-%d');d.news_record_keys=d.news_record_keys.fillna('')
    assert len(d)==1607 and not d.key.duplicated().any() and d.horizon.eq('4h').all()
    assert (d.end_utc-d.start_utc).eq(pd.Timedelta('4h')).all()
    assert (d.start_utc-d.cutoff_utc).eq(pd.Timedelta('5min')).all()
    assert (d.history_end<=d.cutoff_utc).all()
    raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key')
    fingerprint=json.loads((p/'embedding_inputs.json').read_text());titles=dict(fingerprint['keys_and_titles'])
    assert fingerprint['revision']=='4556d13015211d73dccd3fdd39d39232506f3e43' and fingerprint['max_length']==256
    ages=[];unknown=[]
    for r in d.itertuples():
        ids=[k for k in r.news_record_keys.split('|') if k]
        if not ids: ages.append(np.nan);unknown.append(1.);continue
        a=raw.loc[ids]
        if not ((a.available_utc<=r.cutoff_utc)&(a.available_utc>r.cutoff_utc-pd.Timedelta('4h'))).all(): raise ValueError('Future/outside-window news')
        if not all(titles[k]==(raw.loc[k,'title'] if pd.notna(raw.loc[k,'title']) else '') for k in ids): raise ValueError('Changed title in embedding cache')
        age=(r.cutoff_utc-a.published_utc).dt.total_seconds()/3600
        if (age.dropna()<0).any():raise ValueError('Future published timestamp')
        ages.append(float(age.median()) if age.notna().any() else np.nan);unknown.append(float(age.isna().mean()))
    d['age_median']=ages;d['age_unknown']=unknown
    z=np.load(p/'articles.npz');return d.sort_values(['start_utc','symbol']).reset_index(drop=True),z['keys'],z['embeddings'].astype(float)

def predict_bundle(bundle,d,keys,emb):
    result=d[['key','symbol','day','month','start_utc','end_utc','cutoff_utc','has_news','news_count','age_median','age_unknown','return_std','ny_hour','split']].copy()
    for kind in ['price','title']:
        obj=bundle[kind];result[kind]=obj['model'].predict_proba(obj['transform'].transform(d,keys,emb))[:,1]
    for name,kind,joint in [('body','body',False),('semantic','semantic',False),('joint_body','body',True),('joint_semantic','semantic',True)]:
        obj=bundle.get(name); delta=np.zeros(len(d));raw=delta.copy()
        if obj is not None:
            z=obj['transform'].transform(d,keys,emb);x=obj['price_scaler'].transform(d[PRICE]) if joint else None
            delta,raw=obj['model'].correction(z,d.has_news.to_numpy(),x)
        result[name]=add_delta(result.price,delta,np.ones(len(d),bool) if joint else d.has_news)
        if not joint:result['db' if name=='body' else 'ds']=delta;result['raw_b' if name=='body' else 'raw_s']=raw
    result['static']=combine(result,bundle['static'])
    w,reason=bundle['router'].weights(result,bundle['static'],bundle['rho'])
    result['gate']=combine(result,w);result['gate_reason']=reason
    for j,n in enumerate(['zero','body','semantic']):result['weight_'+n]=w[:,j]
    for name in ['price','static','gate']:result[name+'_cal']=bundle['calibration'].predict(result[name])
    return result
