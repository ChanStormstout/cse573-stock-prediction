"""Fixed finite text controls, dense adaptation and rank-two interactions."""
from core import *
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from scipy import sparse
import torch
from torch import nn
STATE=['recent_return_15','recent_return_60','recent_rv_15','recent_rv_60','return_mean','return_std','minutes_from_open','overnight_gap']

class TextTransform:
 def __init__(self,kind,embs,ids,means,gate):self.kind=kind;self.embs=embs;self.ids=ids;self.means=means;self.gate=gate
 def fit(self,d,tr):
  self.pricecols=OLD if self.kind in ('F1_new','F2_new','T_mix') else OLD+RECENT
  self.price=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler()).fit(d.iloc[tr][self.pricecols]);self.state=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler()).fit(d.iloc[tr][STATE]);self.pcas=[]
  if self.kind=='F1_new':
   self.vocab=CountVectorizer(binary=True,min_df=3);a=self.vocab.fit_transform(d.iloc[tr].stem_body.fillna(''));scores=chi2(a,d.iloc[tr].label)[0];self.keep=np.lexsort((self.vocab.get_feature_names_out(),-np.nan_to_num(scores,nan=-np.inf)))[:500]
  else:
   for emb,ids in zip(self.embs,self.ids):
    jj=sorted({j for i in tr for j in ids[i]});n=8 if self.kind in ('T_mix','H0','H1','H2') else 16;self.pcas.append(PCA(n_components=n,svd_solver='randomized',random_state=573).fit(emb[jj]))
   self.textscale=StandardScaler().fit(self.semantic(d,tr))
  return self
 def semantic(self,d,ii):
  chunks=[]
  for pca,mean,ids in zip(self.pcas,self.means,self.ids):
   z=pca.transform(mean[ii]);z[np.array([not ids[i] for i in ii])]=0;chunks.append(z)
  return np.column_stack(chunks+[np.log1p(d.iloc[ii].news_count.to_numpy()),d.iloc[ii].has_original_news.to_numpy()])
 def transform(self,d,ii):
  price=self.price.transform(d.iloc[ii][self.pricecols])
  if self.kind=='F1_new':return sparse.hstack([sparse.csr_matrix(price),self.vocab.transform(d.iloc[ii].stem_body.fillna(''))[:,self.keep]]).tocsr()
  return np.c_[price,self.textscale.transform(self.semantic(d,ii))]

def arrays(path,target=False):
 z=np.load(path);lookup={k:i for i,k in enumerate(z['keys'])};d=data();ids=[[lookup[(r.symbol+'|' if target else '')+k] for k in str(r.news_record_keys or '').split('|') if k and k!='nan'] for r in d.itertuples()];emb=z['embeddings'].astype(float);means=np.array([emb[ii].mean(0) if ii else np.zeros(emb.shape[1]) for ii in ids]);return emb,ids,means
class Interaction(nn.Module):
 def __init__(self,n):
  super().__init__();self.linear=nn.Linear(n,1);self.u=nn.Linear(8,2,bias=False);self.v=nn.Linear(16,2,bias=False)
 def forward(self,x,s):return self.linear(x).squeeze(-1)+(self.u(s)*self.v(x[:,len(OLD+RECENT):len(OLD+RECENT)+16])).sum(-1)
def fit_net(x,s,y,seed,validation=None,epochs=80):
 torch.manual_seed(seed);model=Interaction(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1);xx=torch.tensor(x,dtype=torch.float32);ss=torch.tensor(s,dtype=torch.float32);yy=torch.tensor(np.asarray(y),dtype=torch.float32);best=float('inf');bestepoch=1;stale=0;losses=[];gradient=0.
 for epoch in range(1,epochs+1):
  model.train();opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xx,ss),yy);loss.backward();gradient=max(gradient,float(sum(p.grad.square().sum() for p in model.parameters()).sqrt()));opt.step();vl=None
  if validation is not None:
   a,b,c=validation;model.eval()
   with torch.no_grad():vl=float(nn.functional.binary_cross_entropy_with_logits(model(torch.tensor(a,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)),torch.tensor(np.asarray(c),dtype=torch.float32)))
   if vl<best-1e-6:best=vl;bestepoch=epoch;stale=0
   else:stale+=1
  losses.append(dict(epoch=epoch,train=float(loss.detach()),validation=vl))
  if validation is not None and stale>=8:break
 return model,(bestepoch if validation is not None else epochs),losses,gradient

def run(kind,interaction=False):
 d=data();gate=d.has_original_news.to_numpy().astype(bool);fallback=pd.read_csv(PRIVATE/'P_own_lr/predictions.csv',float_precision='round_trip').p.to_numpy();sources=[]
 if kind=='T_mix':sources=[arrays(ROOT/'outputs/stock_integrated_4h/prepared/articles.npz'),arrays(W/'foundation_4h/v1/modern_embeddings.npz')]
 elif kind.startswith('T_A1'):sources=[arrays(PRIVATE/f'dense_A1_{s}.npz',True) for s in (573,574,575)]
 elif kind.startswith('T_original'):sources=[arrays(PRIVATE/'dense_original.npz',True)]
 elif kind=='F2_new':sources=[arrays(ROOT/'outputs/stock_integrated_4h/prepared/articles.npz')]
 seeds=(573,574,575) if interaction or kind.startswith('T_A1') else (573,)
 name=kind+('_interaction' if interaction else '');dest=PRIVATE/name;dest.mkdir(parents=True,exist_ok=True)
 def gettf(seed,tr):
  selected=sources if kind=='T_mix' else ([sources[seed-573]] if kind.startswith('T_A1') else sources)
  return TextTransform(kind,[x[0] for x in selected],[x[1] for x in selected],[x[2] for x in selected],gate).fit(d,tr)
 def builder(tr,ev,c):
  predictions=[];evidence=[]
  for seed in seeds:
   start=time.monotonic();tf=gettf(seed,tr);x=tf.transform(d,tr);xx=tf.transform(d,ev);path=dest/f'model_{tr[-1]}_{c}_{seed}.joblib'
   if not interaction:
    model=LogisticRegression(C=c,solver='liblinear',random_state=seed,max_iter=3000,tol=1e-7).fit(x,d.iloc[tr].label);p=model.predict_proba(xx)[:,1];joblib.dump(dict(transform=tf,model=model),path);bundle=joblib.load(path);p2=bundle['model'].predict_proba(bundle['transform'].transform(d,ev))[:,1];detail=dict(epochs=None,gradient=None)
   else:
    dates=sorted(d.iloc[tr].day.unique());boundary=dates[int(len(dates)*.8)];va=tr[d.iloc[tr].day.to_numpy()>=boundary];it=tr[(d.iloc[tr].day.to_numpy()<boundary)&(d.iloc[tr].end_utc.to_numpy()<d.iloc[va].cutoff_utc.min())];assert len(it)>0 and d.iloc[it].end_utc.max()<d.iloc[va].cutoff_utc.min();inner=gettf(seed,it)
    _,epochs,curve,_=fit_net(inner.transform(d,it),inner.state.transform(d.iloc[it][STATE]),d.iloc[it].label,seed,(inner.transform(d,va),inner.state.transform(d.iloc[va][STATE]),d.iloc[va].label))
    model,_,traincurve,grad=fit_net(x,tf.state.transform(d.iloc[tr][STATE]),d.iloc[tr].label,seed,epochs=epochs);model.eval()
    with torch.no_grad():p=torch.sigmoid(model(torch.tensor(xx,dtype=torch.float32),torch.tensor(tf.state.transform(d.iloc[ev][STATE]),dtype=torch.float32))).numpy()
    joblib.dump(dict(transform=tf,weights={k:v.detach().numpy() for k,v in model.state_dict().items()},input_dim=x.shape[1]),path);bundle=joblib.load(path);reload=Interaction(bundle['input_dim']);reload.load_state_dict({k:torch.tensor(v) for k,v in bundle['weights'].items()});reload.eval()
    with torch.no_grad():p2=torch.sigmoid(reload(torch.tensor(bundle['transform'].transform(d,ev),dtype=torch.float32),torch.tensor(bundle['transform'].state.transform(d.iloc[ev][STATE]),dtype=torch.float32))).numpy()
    detail=dict(epochs=epochs,gradient=grad,inner_train_n=len(it),inner_validation_n=len(va),early_stop_curve=curve,train_curve=traincurve,parameters=sum(a.numel() for a in model.parameters()));assert grad>0
   err=float(np.max(abs(p-p2)));assert err<1e-7;predictions.append(p);evidence.append(dict(seed=seed,seconds=time.monotonic()-start,sha256=sha(path),reload_error=err,**detail))
  np.savez_compressed(dest/f'seeds_{tr[-1]}_{c}.npz',indices=ev,seeds=seeds,probabilities=np.array(predictions))
  return np.mean(predictions,axis=0),dict(seed_training=evidence,device='cpu')
 torch.set_num_threads(2)
 with threadpool_limits(limits=2):run_model(name,builder,candidates=[1.] if interaction else CS,fallback=fallback,gate=gate)
if __name__=='__main__':
 import sys
 for kind in (sys.argv[1:] or ['F1_new','F2_new','T_mix']):
  run(kind)
  if kind.startswith(('T_A1','T_original')):run(kind,True)
