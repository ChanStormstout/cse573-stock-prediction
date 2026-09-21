#!/usr/bin/env python3
"""Independent replay and protocol checks for L1-L5."""
from __future__ import annotations
import argparse,ast,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'v1';PRIVATE=ROOT/'work/stock-data/fnspid_llm_4h/v1';SOURCE=ROOT/'work/stock-data/fnspid_augmented_4h/v1'
OLD=[f'{v}_{i}' for i in range(1,7) for v in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour'];F_NAMES=['accepted_log','quality_mean','quality_max','novelty_mean','novelty_max','signed_change','positive_mass','negative_mass','repeat_mass','fnspid_log'];CS=(.01,.1,1.)
class L1Transform:
 def raw(self,d,ii,means):
  z=self.pca.transform(means[ii]);z[d.iloc[ii].l1_has_news.to_numpy()==0]=0;return np.c_[z,np.log1p(d.iloc[ii].l1_articles),d.iloc[ii].l1_has_news,np.log1p(d.iloc[ii].l1_fnspid)]
 def transform(self,d,ii,means):return np.c_[np.nan_to_num((d.iloc[ii][OLD].to_numpy(float)-self.pm)/self.ps),self.ss.transform(self.raw(d,ii,means))]
class ResidualModel:
 def predict(self,x,gate,base):
  gate=np.asarray(gate,bool);z=np.nan_to_num((x-self.mean)/self.scale);out=np.asarray(base,float).copy();out[gate]=expit(logit(np.clip(out[gate],1e-7,1-1e-7))+z[gate]@self.beta);return out
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def corrected(base,signal,gate,alpha):
 out=np.asarray(base,float).copy();gate=np.asarray(gate,bool)
 if alpha:out[gate]=expit(logit(np.clip(out[gate],1e-8,1-1e-8))+alpha*np.tanh(logit(np.clip(np.asarray(signal)[gate],1e-6,1-1e-6))))
 return out
def expected_c(cv,symbol,method,boundary):
 x=cv[(cv.symbol==symbol)&(cv.method==method)&(cv.month<boundary)]
 if x.empty:return .1
 return min(CS,key=lambda c:(-x[np.isclose(x.C,c)].BA.mean(),x[np.isclose(x.C,c)].Brier.mean(),c))
def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--max-stage',type=int,choices=(3,5),default=5);ap.add_argument('--output',default='outputs/stock_fnspid_llm_4h/v1');ap.add_argument('--models',default='work/stock-data/fnspid_llm_4h/v1/models');args=ap.parse_args()
 OUT=ROOT/args.output;models=ROOT/args.models;include_l4=args.max_stage>=5
 checks={};p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');cv=pd.read_csv(OUT/'cv_metrics.csv');sel=pd.read_csv(OUT/'selections.csv');evd=json.loads((OUT/'training_evidence.json').read_text());gate_saved=json.loads((OUT/'promotion_gate.json').read_text());d=pd.read_pickle(SOURCE/'inputs.pkl').sort_values(['start_utc','symbol']).reset_index(drop=True);base=pd.read_csv(ROOT/'outputs/stock_fnspid_augmented_4h/v1/predictions.csv',float_precision='round_trip').set_index('key');d['PRICE_R1']=d.key.map(base.PRICE_R1)
 checks['canonical_rows']=len(d)==1607 and p.key.tolist()==d.key.tolist() and int((p.phase!='warmup').sum())==1374
 checks['source_hashes']=all(sha(ROOT/k)==v for k,v in evd['source_hashes'].items());checks['chronology']=all(x['train_end'] is None or pd.Timestamp(x['train_end'])<pd.Timestamp(x['eval_cutoff']) for x in evd['evidence']);checks['model_count']=evd['models']==76 and len(evd['evidence'])==76;checks['runner_reload']=evd['max_reload_error']<=1e-12
 article_jobs=jsonl(PRIVATE/'article_jobs.jsonl');article_scores=jsonl(PRIVATE/'article_scores.jsonl');checks['article_jobs_complete']=len(article_jobs)==len(article_scores)==3966 and {x['job_id'] for x in article_jobs}=={x['job_id'] for x in article_scores} and all(0<=x['p_A']<=1 for x in article_scores)
 if include_l4:
  analogy_jobs=jsonl(PRIVATE/'analogy_jobs.jsonl');analogy_scores=jsonl(PRIVATE/'analogy_scores.jsonl');audit=json.loads((OUT/'ANALOGY_JOB_AUDIT.json').read_text());checks['analogy_jobs_complete']=len(analogy_jobs)==len(analogy_scores)==audit['jobs'] and audit['future_case_violations']==0 and {x['job_id'] for x in analogy_jobs}=={x['job_id'] for x in analogy_scores}
 scores={(x['article_key'],x['task']):x['p_A'] for x in article_scores};z=np.load(SOURCE/'combined_embeddings.npz');keys=z['keys'].astype(str);lookup={k:i for i,k in enumerate(keys)};emb=z['finbert'].astype(float)
 l1keys=[];features=[]
 for r in d.itertuples(index=False):
  course=[k for k in str(r.news_record_keys).split('|') if k];ext=[k for k in str(r.fnspid_record_keys).split('|') if k];accepted=[k for k in ext if scores[(k,'quality')]>=.60];l1keys.append(course+accepted);q=np.array([scores[(k,'quality')] for k in accepted]);n=np.array([scores[(k,'novelty')] for k in accepted]);pol=np.array([scores[(k,'polarity')] for k in accepted]);features.append([np.log1p(len(accepted)),q.mean() if len(q) else 0,q.max() if len(q) else 0,n.mean() if len(n) else 0,n.max() if len(n) else 0,np.sum(n*(2*pol-1)),np.sum(n*pol),np.sum(n*(1-pol)),np.sum(1-n),np.log1p(len(ext))])
 d['l1_articles']=[len(x) for x in l1keys];d['l1_fnspid']=[sum(k.startswith('FNSPID::') for k in x) for x in l1keys];d['l1_has_news']=(d.l1_articles>0).astype(int);d['l2_gate']=(d.l1_fnspid>0).astype(int);ids=[[lookup[k] for k in x] for x in l1keys];means=np.asarray([emb[ii].mean(0) if ii else np.zeros(emb.shape[1]) for ii in ids]);features=np.asarray(features)
 selection_errors=[];maxerr=0.;replay_errors=[]
 for r in sel[sel.method.isin(['L1_FILTERED_FINBERT','L2_FACT_CHANGE'])].itertuples(index=False):
  boundary='2018-09' if r.month=='final' else r.month;want=expected_c(cv,r.symbol,r.method,boundary);choice=float(r.choice)
  if not np.isclose(want,choice):selection_errors.append((r.symbol,r.month,r.method,choice,want))
  ii=np.flatnonzero((d.symbol==r.symbol)&((d.month>='2018-09') if r.month=='final' else (d.month==r.month)))
  if r.method=='L1_FILTERED_FINBERT':
   x=joblib.load(models/f'{r.symbol}_{r.month}_L1_{choice}.joblib');q=x['model'].predict_proba(x['transform'].transform(d,ii,means))[:,1]
  else:q=joblib.load(models/f'{r.symbol}_{r.month}_L2_{choice}.joblib').predict(features[ii],d.iloc[ii].l2_gate,d.iloc[ii].PRICE_R1)
  err=float(np.max(np.abs(q-p.iloc[ii][r.method])));maxerr=max(maxerr,err)
  if err>1e-12:replay_errors.append((r.symbol,r.month,r.method,err))
 checks['past_only_C_selection']=not selection_errors;checks['selected_model_replay']=not replay_errors and maxerr<=1e-12
 # Independently reconstruct the non-parametric L3 vote, L4 correction and
 # their past-only monthly choices rather than trusting runner columns.
 norm=means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12)
 def vote(i,k):
  r=d.iloc[i];mask=(d.symbol.eq(r.symbol)&(d.end_utc<r.cutoff_utc))
  if r.month>='2018-09':mask&=d.month.lt('2018-09')
  pool=np.flatnonzero(mask.to_numpy());sim=norm[pool]@norm[i];order=np.argsort(-sim,kind='stable')[:k];w=np.exp(5*(sim[order]-sim[order].max()));return float((np.dot(w,d.iloc[pool[order]].label)+1)/(w.sum()+2))
 analogy={x['key']:x['p_A'] for x in analogy_scores} if include_l4 else {}
 nonparam_selection_errors=[];nonparam_replay_errors=[]
 for r in sel[sel.method.isin(['L3_CASE_VOTE']+(['L4_LLM_ANALOGY'] if include_l4 else []))].itertuples(index=False):
  boundary='2018-09' if r.month=='final' else r.month;prior=cv[(cv.symbol==r.symbol)&(cv.method==r.method)&(cv.month<boundary)]
  ii=np.flatnonzero((d.symbol==r.symbol)&((d.month>='2018-09') if r.month=='final' else (d.month==r.month)));base_prob=d.iloc[ii].PRICE_R1.to_numpy()
  if r.method=='L3_CASE_VOTE':
   want=(5,0.) if prior.empty else min(((k,a) for k in (3,5,10) for a in (0,.25,.5)),key=lambda ka:(-prior[(prior.k==ka[0])&np.isclose(prior.alpha,ka[1])].BA.mean(),prior[(prior.k==ka[0])&np.isclose(prior.alpha,ka[1])].Brier.mean(),ka))
   got=ast.literal_eval(r.choice);signal=np.array([vote(i,int(got[0])) for i in ii]);q=corrected(base_prob,signal,np.ones(len(ii),bool),float(got[1]))
  else:
   want=0. if prior.empty else min((0,.25,.5),key=lambda a:(-prior[np.isclose(prior.alpha,a)].BA.mean(),prior[np.isclose(prior.alpha,a)].Brier.mean(),a))
   got=float(r.choice);signal=np.array([analogy.get(d.iloc[i].key,.5) for i in ii]);active=np.array([d.iloc[i].key in analogy for i in ii]);q=corrected(base_prob,signal,active,got)
  if want!=got:nonparam_selection_errors.append((r.symbol,r.month,r.method,got,want))
  err=float(np.max(np.abs(q-p.iloc[ii][r.method])));maxerr=max(maxerr,err)
  if err>1e-12:nonparam_replay_errors.append((r.symbol,r.month,r.method,err))
 checks['past_only_nonparam_selection']=not nonparam_selection_errors;checks['L3_L4_independent_replay']=not nonparam_replay_errors
 evaluated=p.phase!='warmup';no=evaluated&(p.l2_gate==0);checks['L2_exact_fallback']=np.array_equal(p.loc[no,'L2_FACT_CHANGE'].to_numpy(),p.loc[no,'PRICE_R1'].to_numpy())
 candidates=['PRICE_R1','L1_FILTERED_FINBERT','L2_FACT_CHANGE','L3_CASE_VOTE']+(['L4_LLM_ANALOGY'] if include_l4 else [])
 checks['L5_global_choice']=gate_saved['L5_choice'] in candidates and np.array_equal(p.loc[evaluated,'L5_GUARDED_SYSTEM'].to_numpy(),p.loc[evaluated,gate_saved['L5_choice']].to_numpy())
 cols=candidates+['L5_GUARDED_SYSTEM'];checks['probability_bounds']=bool((p.loc[evaluated,cols].ge(0)&p.loc[evaluated,cols].le(1)).all().all())
 result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'selection_errors':selection_errors+nonparam_selection_errors,'replay_errors':replay_errors+nonparam_replay_errors,'maximum_selected_model_probability_error':maxerr};(OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2));
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
