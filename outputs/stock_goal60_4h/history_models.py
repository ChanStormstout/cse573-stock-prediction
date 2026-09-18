"""Matched current/history/reaction finite LR comparisons."""
from core import *
from sklearn.decomposition import PCA

class HistoryTransform:
 def __init__(self,kind,emb,ids,meta,react):self.kind=kind;self.emb=emb;self.ids=ids;self.meta=meta;self.react=react
 def fit(self,d,tr):
  jj=sorted({j for i in tr for bucket in self.ids[i] for j in bucket});self.pca=PCA(n_components=8,svd_solver='randomized',random_state=573).fit(self.emb[jj]);self.projected=self.pca.transform(self.emb);self.emb=None;self.scale=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler()).fit(self.raw(d,tr));return self
 def raw(self,d,ii):
  rows=[]
  for i in ii:
   row=[]
   for bucket in self.ids[i]:row.extend(self.projected[bucket].mean(0) if bucket else np.zeros(8))
   row.extend(self.meta[i]);
   if self.kind=='H2':row.extend(self.react[i])
   rows.append(row)
  return np.c_[d.iloc[ii][OLD+RECENT].to_numpy(float),rows]
 def transform(self,d,ii):return self.scale.transform(self.raw(d,ii))
def run(kind):
 d=data();z=np.load(PRIVATE/'history_embeddings.npz');emb=z['embeddings'];lookup={k:i for i,k in enumerate(z['keys'])};members=json.loads((PRIVATE/'history_members.json').read_text());raw=pd.read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;avail=raw.set_index('key').available_utc;ids=[];meta=[];gate=[]
 for i,m in enumerate(members):
  buckets=[m['current'],m['old'] if kind!='H0' else []];ids.append([[lookup[k] for k in b] for b in buckets]);row=[]
  for b in buckets:
   age=np.mean([(d.iloc[i].cutoff_utc-avail[k]).total_seconds()/3600 for k in b]) if b else 0;row.extend([np.log1p(len(b)),np.log1p(age),int(bool(b))])
  meta.append(row);gate.append(any(buckets))
 gate=np.array(gate);fallback=pd.read_csv(PRIVATE/'P_own_lr/predictions.csv',float_precision='round_trip').p.to_numpy();react=pd.read_pickle(PRIVATE/'history_reactions.pkl').to_numpy();dest=PRIVATE/kind;dest.mkdir(parents=True,exist_ok=True)
 def builder(tr,ev,c):
  tf=HistoryTransform(kind,emb,ids,np.array(meta),react).fit(d,tr);x=tf.transform(d,tr);xx=tf.transform(d,ev);model=LogisticRegression(C=c,solver='liblinear',max_iter=3000,tol=1e-7,random_state=573).fit(x,d.iloc[tr].label);p=model.predict_proba(xx)[:,1];path=dest/f'model_{tr[-1]}_{c}.joblib';joblib.dump(dict(transform=tf,model=model),path);reload=joblib.load(path);p2=reload['model'].predict_proba(reload['transform'].transform(d,ev))[:,1];err=float(np.max(abs(p-p2)));assert err<1e-12
  return p,dict(model=path.name,sha256=sha(path),reload_error=err)
 with threadpool_limits(limits=2):run_model(kind,builder,fallback=fallback,gate=gate)
if __name__=='__main__':
 for kind in ('H0','H1','H2'):run(kind)
