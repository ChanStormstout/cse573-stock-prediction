#!/usr/bin/env python3
"""Train and evaluate preregistered L1-L5 on augmented chronological data."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,matthews_corrcoef,roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'v1'
PRIVATE=ROOT/'work/stock-data/fnspid_llm_4h/v1';SOURCE=ROOT/'work/stock-data/fnspid_augmented_4h/v1';MODELS=PRIVATE/'models'
CS=(.01,.1,1.);MONTHS=[f'2018-{m:02d}' for m in range(3,9)]
OLD=[f'{v}_{i}' for i in range(1,7) for v in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour']
F_NAMES=['accepted_log','quality_mean','quality_max','novelty_mean','novelty_max','signed_change','positive_mass','negative_mass','repeat_mass','fnspid_log']

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def metric(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);q=p>=.5
 return {'n':len(y),'BA':float(balanced_accuracy_score(y,q)) if len(set(y))==2 else None,'accuracy':float(accuracy_score(y,q)),'MCC':float(matthews_corrcoef(y,q)),'Brier':float(brier_score_loss(y,p)),'AUC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'pred_up':float(q.mean()),'constant':bool(len(set(q))==1)}
def choose_c(rows,symbol,method):
 x=[r for r in rows if r['symbol']==symbol and r['method']==method]
 if not x:return .1
 return min(CS,key=lambda c:(-np.mean([r['BA'] for r in x if r['C']==c]),np.mean([r['Brier'] for r in x if r['C']==c]),c))
def corrected(base,signal,gate,alpha):
 out=np.asarray(base,float).copy();gate=np.asarray(gate,bool)
 if alpha:out[gate]=expit(logit(np.clip(out[gate],1e-8,1-1e-8))+alpha*np.tanh(logit(np.clip(np.asarray(signal)[gate],1e-6,1-1e-6))))
 return out

class L1Transform:
 def fit(self,d,ii,emb,ids,means):
  x=d.iloc[ii][OLD].to_numpy(float);self.pm=np.nan_to_num(np.nanmean(x,0));self.ps=np.nanstd(x,0);self.ps=np.where(self.ps>1e-12,self.ps,1.)
  articles=sorted({j for i in ii for j in ids[i]});self.pca=PCA(16,svd_solver='randomized',random_state=573).fit(emb[articles]);raw=self.raw(d,ii,means);self.ss=StandardScaler().fit(raw);return self
 def raw(self,d,ii,means):
  z=self.pca.transform(means[ii]);no=d.iloc[ii].l1_has_news.to_numpy()==0;z[no]=0
  return np.c_[z,np.log1p(d.iloc[ii].l1_articles),d.iloc[ii].l1_has_news,np.log1p(d.iloc[ii].l1_fnspid)]
 def transform(self,d,ii,means):
  price=np.nan_to_num((d.iloc[ii][OLD].to_numpy(float)-self.pm)/self.ps);return np.c_[price,self.ss.transform(self.raw(d,ii,means))]

class ResidualModel:
 def fit(self,x,gate,y,base,C):
  active=np.asarray(gate,bool);x=np.asarray(x,float);y=np.asarray(y,float);base=np.asarray(base,float)
  if len(y)==0 or not active.any():
   self.mean=np.zeros(x.shape[1]);self.scale=np.ones(x.shape[1]);self.beta=np.zeros(x.shape[1]);self.C=C;return self
  self.mean=np.nanmean(x[active],0);self.scale=np.nanstd(x[active],0);self.scale=np.where(self.scale>1e-12,self.scale,1.);z=np.nan_to_num((x-self.mean)/self.scale);offset=logit(np.clip(base,1e-7,1-1e-7))
  def loss(beta):
   score=offset+active*(z@beta);return float(np.logaddexp(0,score).sum()-y@score+.5*np.dot(beta,beta)/C)
  opt=minimize(loss,np.zeros(x.shape[1]),method='L-BFGS-B');
  if not opt.success:raise RuntimeError(opt.message)
  self.beta=opt.x;self.C=C;return self
 def predict(self,x,gate,base):
  gate=np.asarray(gate,bool);z=np.nan_to_num((x-self.mean)/self.scale);out=np.asarray(base,float).copy();out[gate]=expit(logit(np.clip(out[gate],1e-7,1-1e-7))+z[gate]@self.beta);return out

def main():
 global OUT,MODELS
 ap=argparse.ArgumentParser()
 ap.add_argument('--max-stage',type=int,choices=(3,5),default=5)
 ap.add_argument('--output',default='outputs/stock_fnspid_llm_4h/v1')
 ap.add_argument('--models',default='work/stock-data/fnspid_llm_4h/v1/models')
 args=ap.parse_args();OUT=ROOT/args.output;MODELS=ROOT/args.models;include_l4=args.max_stage>=5
 OUT.mkdir(parents=True,exist_ok=True)
 if MODELS.exists():raise FileExistsError('immutable run already trained')
 if not (PRIVATE/'article_scores.jsonl').exists():raise FileNotFoundError('complete article inference first')
 if include_l4 and not (PRIVATE/'analogy_scores.jsonl').exists():raise FileNotFoundError('complete both LLM inference stages first')
 MODELS.mkdir(parents=True)
 d=pd.read_pickle(SOURCE/'inputs.pkl').sort_values(['start_utc','symbol']).reset_index(drop=True)
 base=pd.read_csv(ROOT/'outputs/stock_fnspid_augmented_4h/v1/predictions.csv',float_precision='round_trip').set_index('key');d['PRICE_R1']=d.key.map(base.PRICE_R1);d['AUG_UNFILTERED_FINBERT']=d.key.map(base.AUG_FINBERT_LR);d['ORIG_FINBERT']=d.key.map(base.ORIG_FINBERT_LR)
 scores={(x['article_key'],x['task']):x['p_A'] for x in jsonl(PRIVATE/'article_scores.jsonl')};analogy={x['key']:x['p_A'] for x in jsonl(PRIVATE/'analogy_scores.jsonl')} if include_l4 else {}
 z=np.load(SOURCE/'combined_embeddings.npz');keys=z['keys'].astype(str);lookup={k:i for i,k in enumerate(keys)};emb=z['finbert'].astype(float)
 l1keys=[];features=[]
 for r in d.itertuples(index=False):
  course=[k for k in str(r.news_record_keys).split('|') if k];ext=[k for k in str(r.fnspid_record_keys).split('|') if k];accepted=[k for k in ext if scores[(k,'quality')]>=.60];l1keys.append(course+accepted)
  q=np.array([scores[(k,'quality')] for k in accepted]);n=np.array([scores[(k,'novelty')] for k in accepted]);pol=np.array([scores[(k,'polarity')] for k in accepted])
  if len(accepted):features.append([np.log1p(len(accepted)),q.mean(),q.max(),n.mean(),n.max(),np.sum(n*(2*pol-1)),np.sum(n*pol),np.sum(n*(1-pol)),np.sum(1-n),np.log1p(len(ext))])
  else:features.append([0]*9+[np.log1p(len(ext))])
 d['l1_articles']=[len(x) for x in l1keys];d['l1_fnspid']=[sum(k.startswith('FNSPID::') for k in x) for x in l1keys];d['l1_has_news']=(d.l1_articles>0).astype(int);d['l2_gate']=(d.l1_fnspid>0).astype(int)
 ids=[[lookup[k] for k in x] for x in l1keys];means=np.asarray([emb[ii].mean(0) if ii else np.zeros(emb.shape[1]) for ii in ids]);norm=means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12);features=np.asarray(features,float)
 def vote(i,k):
  r=d.iloc[i];mask=(d.symbol.eq(r.symbol)&(d.end_utc<r.cutoff_utc));
  if r.month>='2018-09':mask&=d.month.lt('2018-09')
  pool=np.flatnonzero(mask.to_numpy());sim=norm[pool]@norm[i];order=np.argsort(-sim,kind='stable')[:k];w=np.exp(5*(sim[order]-sim[order].max()));return float((np.dot(w,d.iloc[pool[order]].label)+1)/(w.sum()+2))
 vote_cache={(i,k):vote(i,k) for i in range(len(d)) if d.iloc[i].phase!='warmup' for k in (3,5,10)}
 cv=[];sel=[];evidence=[];started=time.time()
 with threadpool_limits(2):
  for symbol in ('AAPL','AMZN'):
   for month in MONTHS+['final']:
    boundary='2018-09' if month=='final' else month;tr=np.flatnonzero((d.symbol==symbol)&(d.month<boundary));ev=np.flatnonzero((d.symbol==symbol)&((d.month>='2018-09') if month=='final' else (d.month==month)))
    assert d.iloc[tr].end_utc.max()<d.iloc[ev].cutoff_utc.min();base_tr=d.iloc[tr].PRICE_R1.to_numpy();base_ev=d.iloc[ev].PRICE_R1.to_numpy()
    # L1
    chosen=choose_c(cv,symbol,'L1_FILTERED_FINBERT');candidates=(chosen,) if month=='final' else CS
    tf=L1Transform().fit(d,tr,emb,ids,means);xt=tf.transform(d,tr,means);xe=tf.transform(d,ev,means)
    for c in candidates:
     model=LogisticRegression(C=c,solver='liblinear',tol=1e-7,max_iter=3000,random_state=573).fit(xt,d.iloc[tr].label);p=model.predict_proba(xe)[:,1];path=MODELS/f'{symbol}_{month}_L1_{c}.joblib';joblib.dump({'transform':tf,'model':model},path,compress=3);again=joblib.load(path)['model'].predict_proba(joblib.load(path)['transform'].transform(d,ev,means))[:,1];err=float(np.max(np.abs(p-again)));assert err<1e-12
     if month!='final':cv.append({'symbol':symbol,'month':month,'method':'L1_FILTERED_FINBERT','C':c,**metric(d.iloc[ev].label,p)})
     if c==chosen:d.loc[ev,'L1_FILTERED_FINBERT']=p
     evidence.append({'symbol':symbol,'month':month,'method':'L1','C':c,'train_n':len(tr),'eval_n':len(ev),'train_end':str(d.iloc[tr].end_utc.max()),'eval_cutoff':str(d.iloc[ev].cutoff_utc.min()),'reload_error':err,'sha256':sha(path)})
    sel.append({'symbol':symbol,'month':month,'method':'L1_FILTERED_FINBERT','choice':chosen})
    # L2
    chosen=choose_c(cv,symbol,'L2_FACT_CHANGE');candidates=(chosen,) if month=='final' else CS
    for c in candidates:
     valid=np.isfinite(base_tr);model=ResidualModel().fit(features[tr][valid],d.iloc[tr].l2_gate.to_numpy()[valid],d.iloc[tr].label.to_numpy()[valid],base_tr[valid],c);p=model.predict(features[ev],d.iloc[ev].l2_gate,base_ev);path=MODELS/f'{symbol}_{month}_L2_{c}.joblib';joblib.dump(model,path,compress=3);again=joblib.load(path).predict(features[ev],d.iloc[ev].l2_gate,base_ev);err=float(np.max(np.abs(p-again)));assert err<1e-12
     assert np.array_equal(p[d.iloc[ev].l2_gate.to_numpy()==0],base_ev[d.iloc[ev].l2_gate.to_numpy()==0])
     if month!='final':cv.append({'symbol':symbol,'month':month,'method':'L2_FACT_CHANGE','C':c,**metric(d.iloc[ev].label,p)})
     if c==chosen:d.loc[ev,'L2_FACT_CHANGE']=p
     evidence.append({'symbol':symbol,'month':month,'method':'L2','C':c,'train_n':int(valid.sum()),'base_oof_only':True,'eval_n':len(ev),'train_end':str(d.iloc[tr][valid].end_utc.max()) if valid.any() else None,'eval_cutoff':str(d.iloc[ev].cutoff_utc.min()),'reload_error':err,'sha256':sha(path)})
    sel.append({'symbol':symbol,'month':month,'method':'L2_FACT_CHANGE','choice':chosen})
    # L3 candidates and past-only choice
    prior=[x for x in cv if x['symbol']==symbol and x['method']=='L3_CASE_VOTE']
    choice=(5,0.) if not prior else min(((k,a) for k in (3,5,10) for a in (0,.25,.5)),key=lambda ka:(-np.mean([x['BA'] for x in prior if x['k']==ka[0] and x['alpha']==ka[1]]),np.mean([x['Brier'] for x in prior if x['k']==ka[0] and x['alpha']==ka[1]]),ka))
    for k in (3,5,10):
     signal=np.array([vote_cache[(i,k)] for i in ev])
     for alpha in (0,.25,.5):
      p=corrected(base_ev,signal,np.ones(len(ev),bool),alpha)
      if month!='final':cv.append({'symbol':symbol,'month':month,'method':'L3_CASE_VOTE','k':k,'alpha':alpha,**metric(d.iloc[ev].label,p)})
      if (k,alpha)==choice:d.loc[ev,'L3_CASE_VOTE']=p
    sel.append({'symbol':symbol,'month':month,'method':'L3_CASE_VOTE','choice':str(choice)})
    if include_l4:
     # L4 candidates and past-only choice
     prior=[x for x in cv if x['symbol']==symbol and x['method']=='L4_LLM_ANALOGY'];choice=0. if not prior else min((0,.25,.5),key=lambda a:(-np.mean([x['BA'] for x in prior if x['alpha']==a]),np.mean([x['Brier'] for x in prior if x['alpha']==a]),a))
     signal=np.array([analogy.get(d.iloc[i].key,.5) for i in ev]);gate=np.array([d.iloc[i].key in analogy for i in ev])
     for alpha in (0,.25,.5):
      p=corrected(base_ev,signal,gate,alpha)
      if month!='final':cv.append({'symbol':symbol,'month':month,'method':'L4_LLM_ANALOGY','alpha':alpha,**metric(d.iloc[ev].label,p)})
      if alpha==choice:d.loc[ev,'L4_LLM_ANALOGY']=p
     sel.append({'symbol':symbol,'month':month,'method':'L4_LLM_ANALOGY','choice':choice})
    print(symbol,month,'complete',round(time.time()-started,1),flush=True)
 # L5 OOF gate
 candidates=['L1_FILTERED_FINBERT','L2_FACT_CHANGE','L3_CASE_VOTE']+(['L4_LLM_ANALOGY'] if include_l4 else []);oof=d[d.phase=='train_forward_oof'];gates=[]
 for method in candidates:
  month_delta=[];stock_delta={};brier_delta={};constant={}
  for sym in ('AAPL','AMZN'):
   q=oof[oof.symbol==sym];a=metric(q.label,q[method]);b=metric(q.label,q.PRICE_R1);stock_delta[sym]=a['BA']-b['BA'];brier_delta[sym]=a['Brier']-b['Brier'];constant[sym]=a['constant']
  for mon,q in oof.groupby('month'):
   vals=[]
   for sym,s in q.groupby('symbol'):vals.append(metric(s.label,s[method])['BA']-metric(s.label,s.PRICE_R1)['BA'])
   month_delta.append((mon,float(np.mean(vals))))
  macro=float(np.mean(list(stock_delta.values())));passes=macro>=.01 and min(stock_delta.values())>0 and sum(v>0 for _,v in month_delta)>=4 and max(brier_delta.values())<=.002 and not any(constant.values())
  gates.append({'method':method,'macro_delta_BA':macro,'stock_delta_BA':stock_delta,'stock_delta_Brier':brier_delta,'positive_macro_months':sum(v>0 for _,v in month_delta),'month_deltas':dict(month_delta),'constant':constant,'passes':bool(passes)})
 passing=[x for x in gates if x['passes']]
 if passing:
  def macro_score(method):
   values=[metric(q.label,q[method]) for _,q in oof.groupby('symbol')]
   return np.mean([x['BA'] for x in values]),np.mean([x['Brier'] for x in values])
  chosen=min(passing,key=lambda x:(-macro_score(x['method'])[0],macro_score(x['method'])[1],candidates.index(x['method'])))['method']
 else:chosen='PRICE_R1'
 d['L5_GUARDED_SYSTEM']=d[chosen]
 methods=['PRICE_R1','AUG_UNFILTERED_FINBERT','ORIG_FINBERT']+candidates+['L5_GUARDED_SYSTEM'];rows=[];monthly=[];trans=[]
 for (phase,sym),q in d[d.phase!='warmup'].groupby(['phase','symbol']):
  for method in methods:
   rows.append({'phase':phase,'symbol':sym,'method':method,**metric(q.label,q[method])})
   for mon,s in q.groupby('month'):monthly.append({'phase':phase,'symbol':sym,'month':mon,'method':method,**metric(s.label,s[method])})
  for method in candidates+['L5_GUARDED_SYSTEM']:
   a=(q[method]>=.5)==q.label;b=(q.PRICE_R1>=.5)==q.label;trans.append({'phase':phase,'symbol':sym,'method':method,'repaired':int((a&~b).sum()),'introduced':int((~a&b).sum()),'changed':int(((q[method]>=.5)!=(q.PRICE_R1>=.5)).sum())})
 pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False);pd.DataFrame(trans).to_csv(OUT/'transitions.csv',index=False);pd.DataFrame(cv).to_csv(OUT/'cv_metrics.csv',index=False);pd.DataFrame(sel).to_csv(OUT/'selections.csv',index=False)
 cols=['key','symbol','day','month','phase','label','has_original_news','has_aug_news','l1_has_news','l1_fnspid','l2_gate']+methods;d[cols].to_csv(OUT/'predictions.csv',index=False,float_format='%.17g')
 dump(OUT/'promotion_gate.json',{'components':gates,'L5_choice':chosen,'selection_period':'March-August forward OOF only'})
 source_hashes={str((SOURCE/'inputs.pkl').relative_to(ROOT)):sha(SOURCE/'inputs.pkl'),str((SOURCE/'combined_embeddings.npz').relative_to(ROOT)):sha(SOURCE/'combined_embeddings.npz'),str((PRIVATE/'article_scores.jsonl').relative_to(ROOT)):sha(PRIVATE/'article_scores.jsonl'),str((HERE/'PRE_REGISTRATION.md').relative_to(ROOT)):sha(HERE/'PRE_REGISTRATION.md')}
 if include_l4:source_hashes[str((PRIVATE/'analogy_scores.jsonl').relative_to(ROOT))]=sha(PRIVATE/'analogy_scores.jsonl')
 dump(OUT/'training_evidence.json',{'status':'COMPLETE_PENDING_VERIFICATION','max_stage':args.max_stage,'models':len(evidence),'evidence':evidence,'max_reload_error':max(x['reload_error'] for x in evidence),'seconds':time.time()-started,'source_hashes':source_hashes})
 print(pd.DataFrame(rows).query("phase in ['development','later']").to_string(index=False));print(json.dumps({'L5':chosen,'gates':gates},indent=2))
if __name__=='__main__':main()
