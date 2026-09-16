from core import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator,ClassifierMixin

class SemanticLR(ClassifierMixin,BaseEstimator):
    def __init__(self,dim=16,C=.1):self.dim=dim;self.C=C
    def fit(self,X,y):
        self.pca_=PCA(n_components=self.dim,svd_solver='randomized',random_state=573).fit(X.loc[X.has_news.eq(1),[f'e{i}' for i in range(768)]])
        z=self.features(X);self.scaler_=StandardScaler().fit(z);self.lr_=LogisticRegression(C=self.C,solver='liblinear',max_iter=3000,random_state=573).fit(self.scaler_.transform(z),y);self.classes_=self.lr_.classes_;return self
    def features(self,X):
        z=self.pca_.transform(X[[f'e{i}' for i in range(768)]]);z[X.has_news.eq(0)]=0
        return np.c_[z,np.log1p(X.news_count),X.has_news]
    def predict_proba(self,X):return self.lr_.predict_proba(self.scaler_.transform(self.features(X)))

def encode(d,out):
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    keys=sorted({k for s in d.news_record_keys.fillna('') for k in s.split('|') if k});news=pd.read_pickle(W/'audit/news_index.pkl');news['key']=news.archive+'::'+news.member;news=news.set_index('key');titles=news.loc[keys,'title'].fillna('').tolist()
    revision='4556d13015211d73dccd3fdd39d39232506f3e43';kw=dict(revision=revision,cache_dir=str(W/'finbert-cache'),local_files_only=True,trust_remote_code=False)
    manifest=dict(keys_and_titles=list(zip(keys,titles)),revision=revision,max_length=256,pooling='mean last hidden non-special nonpadding; equal mean articles within original window',torch=torch.__version__,transformers=transformers.__version__,code=sha(__file__))
    dump(out/'embedding_inputs.json',manifest)
    torch.manual_seed(573);torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',**kw);m=AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert',**kw).to(device).eval();arr=[]
    for i in range(0,len(keys),32):
        batch=tok(titles[i:i+32],padding=True,truncation=True,max_length=256,return_special_tokens_mask=True,return_tensors='pt').to(device);special=batch.pop('special_tokens_mask');mask=(batch['attention_mask']*(1-special)).unsqueeze(-1)
        with torch.inference_mode():v=(m.bert(**batch).last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1)
        arr.append(v.cpu().numpy())
        if i%1280==0:print('FinBERT new fingerprints',i,len(keys),flush=True)
    emb=np.concatenate(arr);np.savez_compressed(out/'articles.npz',keys=np.array(keys),embeddings=emb);lookup={k:i for i,k in enumerate(keys)};bags=[[lookup[k] for k in s.split('|') if k] for s in d.news_record_keys.fillna('')];pooled=np.array([emb[b].mean(0) if b else np.zeros(768) for b in bags]);dump(out/'bags.json',bags)
    d=pd.concat([d.reset_index(drop=True),pd.DataFrame(pooled,columns=[f'e{i}' for i in range(768)])],axis=1);d.to_pickle(out/'features.pkl');dump(out/'embedding_fingerprint.json',{str(out/x):sha(out/x) for x in ['embedding_inputs.json','articles.npz','features.pkl']});return d

def run(d,out):
    d=encode(d,out);cv=[];pred=[];decisions={};oofout=[]
    for stock,s in d.groupby('symbol'):
        specs=[('price',None)]+[(f'text{dim}',dim) for dim in [8,16,32]];chosen={};oofs={};final={};a=s[s.split.eq('train')];b=s[~s.split.eq('train')]
        for name,dim in specs:
            candidates=[];cache={}
            for C in [.01,.1,1.]:
                pieces=[]
                for month,ft,fv in folds(s):
                    m=(price(C) if dim is None else SemanticLR(dim,C)).fit(ft,ft.label);p=m.predict_proba(fv)[:,1];r=dict(symbol=stock,method=name,C=C,month=month,**metrics(fv.label,p));cv.append(r);pieces.append(pred_rows(fv,p,name,C=C));candidates.append(r)
                cache[C]=pd.concat(pieces)
            g=pd.DataFrame(candidates).groupby('C')[['balanced_accuracy','brier']].mean().sort_values(['balanced_accuracy','brier'],ascending=[False,True]);C=float(g.index[0]);chosen[name]=C;oofs[name]=cache[C].reset_index(drop=True);oofout.append(oofs[name])
            m=(price(C) if dim is None else SemanticLR(dim,C)).fit(a,a.label);p=m.predict_proba(b)[:,1];save_model(m,a,b,out/stock/name,p);final[name]=p;pred.append(pred_rows(b,p,name))
        fusion=[]
        for dim in [8,16,32]:
            text=oofs[f'text{dim}'];base=oofs['price'];assert keys(text)==keys(base)
            for w in [0,.25,.5,.75,1.]:
                p=(1-w)*base.p.to_numpy()+w*text.p.to_numpy()
                for month,ix in base.groupby(base.start_utc.dt.strftime('%Y-%m')).groups.items():fusion.append(dict(dim=dim,weight=w,month=month,**metrics(base.loc[ix,'label'],p[ix])))
        fg=pd.DataFrame(fusion);chosenf=fg.groupby(['dim','weight'])[['balanced_accuracy','brier']].mean().sort_values(['balanced_accuracy','brier'],ascending=[False,True]).index[0];dim,w=int(chosenf[0]),float(chosenf[1]);p=(1-w)*final['price']+w*final[f'text{dim}'];pred.append(pred_rows(b,p,'fusion'))
        z=oofs['price'].copy();z['p']=(1-w)*z.p+w*oofs[f'text{dim}'].p;z['method']='fusion';oofout.append(z);fg.to_csv(out/f'{stock}_fusion_grid.csv',index=False);decisions[stock]=dict(C=chosen,fusion_dim=dim,text_weight=w)
        print('semantic selected',stock,decisions[stock],flush=True)
    pd.DataFrame(cv).to_csv(out/'cv.csv',index=False);pd.concat(pred).to_csv(out/'predictions.csv',index=False);pd.concat(oofout).to_csv(out/'selected_oof.csv',index=False);dump(out/'selected.json',decisions)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();d=load(a.run.resolve());stage(a.run.resolve(),'M08',lambda out:run(d,out))
