from pathlib import Path
import json, hashlib, math, sys, time
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, brier_score_loss, roc_auc_score

ROOT=Path('/Users/victor/Documents/Codex/2026-09-14/wox')
HERE=Path(__file__).resolve().parent
OUT=HERE/'v1'; MODELS=HERE/'private_models'
W=ROOT/'work/stock-data'
C_GRID=(.01,.1,1.)
BLOCKS={
 'D':['D'], 'U':['U'], 'A':['A'], 'AL':['A','L'],
 'DUA':['D','U','A'], 'DUAL':['D','U','A','L'],
 'DUAL_PARTIAL':['D','U','A','L']}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False,default=lambda z:z.item() if hasattr(z,'item') else str(z))+'\n')
def safe_logit(p): return logit(np.clip(np.asarray(p,float),1e-6,1-1e-6))

def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);q=p>=.5
 both=len(np.unique(y))==2
 return dict(n=len(y),BA=float(balanced_accuracy_score(y,q)) if both else None,
   MCC=float(matthews_corrcoef(y,q)),Brier=float(brier_score_loss(y,p)),
   Accuracy=float(np.mean(q==y)),AUC=float(roc_auc_score(y,p)) if both else None,
   predicted_up=float(q.mean()),constant=bool(len(np.unique(q))==1))

def load_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def assemble():
 d=pd.read_pickle(W/'paper_methods_4h/v1/inputs.pkl').reset_index(drop=True)
 base=pd.read_csv(ROOT/'outputs/stock_foundation_4h/v2/modern_predictions.csv',float_precision='round_trip')
 base=base[['key','phase','F1','R1']]
 d=d.drop(columns=[c for c in ['phase','F1','R1'] if c in d],errors='ignore').merge(base,on='key',how='left',validate='one_to_one')
 combo=pd.read_csv(ROOT/'outputs/stock_combination_4h/v1/predictions.csv',float_precision='round_trip')
 keep=['key','articles','clusters','vector_delta','has_original_news','has_usable_text','has_qualified_event']
 d=d.drop(columns=[c for c in keep[1:] if c in d],errors='ignore').merge(combo[keep],on='key',how='left',validate='one_to_one')
 prep=pd.read_csv(W/'analogy_4h/v2/prepared.csv',float_precision='round_trip')
 pub=pd.read_csv(ROOT/'outputs/stock_analogy_4h/v2/predictions.csv',float_precision='round_trip')
 d=d.merge(prep[['key','n_cases','P1']],on='key',how='left',validate='one_to_one')
 d=d.merge(pub[['key','P2','P3']],on='key',how='left',validate='one_to_one')
 assert len(d)==1607 and d.key.nunique()==1607
 return d

def event_features(d):
 facts=load_jsonl(W/'finbert_event_adapter_4h/v1/corpus_extraction/facts_A1_ensemble.jsonl')
 by={}
 for r in facts: by.setdefault((r['symbol'],r['record_key']),[]).append(r)
 names=['event_count','event_groups','rating_raise','rating_lower','rating_maintain',
        'target_raise','target_lower','target_unknown','target_pos_max','target_neg_min','evidence_mean']
 rows=[]
 for r in d.itertuples():
  keys=str(r.news_record_keys).split('|') if isinstance(r.news_record_keys,str) and r.news_record_keys else []
  ev=[]
  for k in keys:
   for a in by.get((r.symbol,k),[]):
    if pd.Timestamp(a['available_utc'])>pd.Timestamp(r.cutoff_utc): raise AssertionError('future event fact')
    for e in a['events']:
     action=e.get('action','unknown');change=e.get('target_change')
     if change is not None and ((action=='raise' and change<0) or (action=='lower' and change>0)):
      continue
     sig=(a.get('event_group'),e.get('kind'),action,str(e.get('old')),str(e.get('new')),str(e.get('unit')))
     ev.append((sig,a,e))
  uniq={x[0]:x for x in ev}; ev=list(uniq.values())
  vals={n:0. for n in names}; vals['event_count']=len(ev);vals['event_groups']=len(set(x[1].get('event_group') for x in ev))
  probs=[];changes=[]
  for _,a,e in ev:
   k=e.get('kind');ac=e.get('action','unknown');probs.append(float(e.get('evidence_probability',0)))
   if k=='rating' and ac in ('raise','lower','maintain'):vals[f'rating_{ac}']+=1
   if k=='target_price':
    vals['target_'+(ac if ac in ('raise','lower') else 'unknown')]+=1
    if e.get('target_change') is not None:changes.append(float(e['target_change']))
  vals['target_pos_max']=max([x for x in changes if x>0],default=0)
  vals['target_neg_min']=min([x for x in changes if x<0],default=0)
  vals['evidence_mean']=float(np.mean(probs)) if probs else 0
  rows.append(vals)
 return pd.DataFrame(rows,index=d.index)

def analogy_features(d):
 retrieval=json.loads((W/'analogy_4h/v2/retrieval.json').read_text())
 arr=np.load(W/'analogy_4h/v2/retrieval_arrays.npz');rets=arr['returns']
 assert len(retrieval)==len(d)==len(rets)
 rows=[]
 for i,(r,a) in enumerate(zip(d.itertuples(),retrieval)):
  js=a['case_indices'];ws=np.asarray(a['scores'],float)
  if js:
   cr=np.asarray([rets[j] for j in js],float)
   for j in js:
    past=d.iloc[j]
    assert pd.Timestamp(past.end_utc)<pd.Timestamp(r.cutoff_utc)
    assert str(past.month)<min(str(r.month),'2018-09')
   sw=ws.sum();mean=float(np.sum(ws*cr)/sw);var=float(np.sum(ws*(cr-mean)**2)/sw)
   pos=float(np.sum(ws*(cr>0))/sw);neff=float(sw*sw/np.sum(ws*ws))
   rows.append(dict(analog_count=len(js),analog_mean=mean,analog_std=math.sqrt(var),analog_pos=pos,
     analog_consistency=abs(2*pos-1),analog_neff=neff,analog_max_similarity=float(ws.max()),analog_mean_similarity=float(ws.mean())))
  else:rows.append(dict(analog_count=0,analog_mean=0,analog_std=0,analog_pos=.5,analog_consistency=0,
     analog_neff=0,analog_max_similarity=0,analog_mean_similarity=0))
 return pd.DataFrame(rows,index=d.index)

def build_raw(d):
 e=event_features(d);a=analogy_features(d)
 articles=d.articles.fillna(0).astype(float);clusters=d.clusters.fillna(0).astype(float)
 D=pd.DataFrame(dict(log_articles=np.log1p(articles),log_clusters=np.log1p(clusters),
   duplicate_fraction=np.where(articles>0,np.maximum(articles-clusters,0)/articles,0),
   vector_delta=d.vector_delta.fillna(0).astype(float),usable=d.has_usable_text.fillna(0).astype(float)),index=d.index)
 A=a.copy();A['vote_delta']=safe_logit(d.P1.fillna(d.F1))-safe_logit(d.R1.fillna(d.F1))
 L=pd.DataFrame(dict(llm_outcome_delta=safe_logit(d.P3.fillna(d.F1))-safe_logit(d.P2.fillna(d.F1))),index=d.index)
 gates={'D':d.has_original_news.fillna(0).astype(float).to_numpy(),
        'U':(e.event_count>0).astype(float).to_numpy(),
        'A':(a.analog_count>0).astype(float).to_numpy(),
        'L':((a.analog_count>0)&d.P2.notna()&d.P3.notna()).astype(float).to_numpy()}
 return {'D':D,'U':e,'A':A,'L':L},gates

def design(raw,gates,blocks,train_ix,apply_ix,symbols,partial=False):
 train=[];apply=[];spec={};active=np.zeros(len(train_ix),bool)
 for b in blocks:
  x=raw[b].to_numpy(float);g=gates[b]
  active|=g[train_ix]>0
  mask=g[train_ix]>0
  mu=x[train_ix][mask].mean(0) if mask.any() else np.zeros(x.shape[1])
  sd=x[train_ix][mask].std(0) if mask.any() else np.ones(x.shape[1]);sd=np.where(sd<1e-8,1,sd)
  def t(ix):return g[ix,None]*np.c_[np.ones(len(ix)),(x[ix]-mu)/sd]
  train.append(t(train_ix));apply.append(t(apply_ix));spec[b]={'mean':mu,'std':sd,'cols':raw[b].columns.tolist()}
 X=np.concatenate(train,1);Z=np.concatenate(apply,1)
 alpha=None
 if partial:
  cnt={s:int(np.sum(active & (symbols[train_ix]==s))) for s in ('AAPL','AMZN')}
  alpha={s:cnt[s]/(cnt[s]+60.) for s in cnt}
  code=np.array([1 if s=='AAPL' else -1 for s in symbols])
  Xdev=X*code[train_ix,None]*np.array([alpha[s] for s in symbols[train_ix]])[:,None]
  Zdev=Z*code[apply_ix,None]*np.array([alpha[s] for s in symbols[apply_ix]])[:,None]
  X=np.c_[X,Xdev];Z=np.c_[Z,Zdev]
 return X,Z,dict(blocks=spec,partial=partial,alpha=alpha)

def fit_model(d,raw,gates,method,C,train_ix,apply_ix):
 partial=method=='DUAL_PARTIAL';blocks=BLOCKS[method]
 if len(train_ix)<30:return np.asarray(d.F1.iloc[apply_ix],float),dict(identity=True,C=C,n=len(train_ix),method=method,blocks=blocks)
 X,Z,spec=design(raw,gates,blocks,train_ix,apply_ix,d.symbol.to_numpy(),partial)
 if X.shape[1]==0 or not np.any(np.abs(X)>0):return np.asarray(d.F1.iloc[apply_ix],float),dict(identity=True,C=C,n=len(train_ix),method=method,blocks=blocks)
 y=d.label.to_numpy(int)[train_ix];off=safe_logit(d.F1.to_numpy(float)[train_ix]);beta=np.zeros(X.shape[1])
 def fun(b):
  eta=off+X@b;p=expit(eta)
  return float(np.logaddexp(0,eta).sum()-np.dot(y,eta)+.5*np.dot(b,b)/C), X.T@(p-y)+b/C
 res=minimize(lambda b:fun(b),beta,jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
 if not res.success:raise RuntimeError(res.message)
 p=expit(safe_logit(d.F1.to_numpy(float)[apply_ix])+Z@res.x)
 no=np.ones(len(apply_ix),bool)
 for b in blocks:no &= gates[b][apply_ix]==0
 # The registered inactive path is an exact copy, not a numerically equivalent
 # sigmoid(logit(base)) round trip.
 p[no]=d.F1.to_numpy(float)[apply_ix][no]
 assert np.array_equal(p[no],d.F1.to_numpy(float)[apply_ix][no])
 return p,dict(identity=False,C=C,n=len(train_ix),method=method,blocks=blocks,beta=res.x,spec=spec,objective=float(res.fun),iterations=int(res.nit))

def choose(records,months):
 if not records:return C_GRID[0]
 score=[]
 for C in C_GRID:
  r=[x for x in records if x['C']==C and x['month'] in months]
  if not r:continue
  score.append((-np.mean([x['macro_BA'] for x in r]),np.mean([x['macro_Brier'] for x in r]),C))
 return min(score)[2] if score else C_GRID[0]

def group_score(d,ix,p):
 vals=[];bs=[]
 for s in ('AAPL','AMZN'):
  take=np.array([i for i in ix if d.symbol.iloc[i]==s],int)
  if not len(take):continue
  pos={v:i for i,v in enumerate(ix)};pp=np.array([p[pos[j]] for j in take])
  m=metrics(d.label.iloc[take],pp);vals.append(m['BA']);bs.append(m['Brier'])
 return float(np.mean(vals)),float(np.mean(bs))

def execute(d,raw,gates):
 eligible=d.phase!='warmup';assert d.loc[eligible,'F1'].notna().all()
 months=[f'2018-{x:02d}' for x in range(3,9)];all_pred={'BASE':d.F1.to_numpy(float).copy()};selection=[];models=[]
 for method in BLOCKS:
  pred=d.F1.to_numpy(float).copy();cand_records=[]
  for month in months:
   apply_ix=np.flatnonzero((d.month==month).to_numpy());train_ix=np.flatnonzero(((d.phase=='train_forward_oof')&(d.month<month)).to_numpy())
   prior=[m for m in months if m<month];issued=choose(cand_records,prior)
   p,art=fit_model(d,raw,gates,method,issued,train_ix,apply_ix);pred[apply_ix]=p
   selection.append(dict(method=method,issued_month=month,C=issued,training_rows=len(train_ix)))
   for C in C_GRID:
    q,_=fit_model(d,raw,gates,method,C,train_ix,apply_ix);ba,br=group_score(d,apply_ix,q)
    cand_records.append(dict(method=method,month=month,C=C,macro_BA=ba,macro_Brier=br))
   if not art.get('identity'):
    path=MODELS/f'{method}_{month}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(art,path);models.append(path)
  finalC=choose(cand_records,months);train_ix=np.flatnonzero((d.phase=='train_forward_oof').to_numpy());apply_ix=np.flatnonzero(d.phase.isin(['development','later']).to_numpy())
  p,art=fit_model(d,raw,gates,method,finalC,train_ix,apply_ix);pred[apply_ix]=p
  selection.append(dict(method=method,issued_month='final',C=finalC,training_rows=len(train_ix)))
  path=MODELS/f'{method}_final.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(art,path);models.append(path)
  all_pred[method]=pred
 return all_pred,pd.DataFrame(selection),models

def report_tables(d,preds):
 rows=[];monthly=[];trans=[]
 for method,p in preds.items():
  for (phase,sym),g in d[d.phase!='warmup'].groupby(['phase','symbol']):
   ix=g.index.to_numpy();rows.append(dict(method=method,phase=phase,symbol=sym,**metrics(g.label,p[ix])))
   for month,h in g.groupby('month'):
    j=h.index.to_numpy();monthly.append(dict(method=method,phase=phase,symbol=sym,month=month,**metrics(h.label,p[j])))
   base=preds['BASE'][ix];old=(base>=.5)==g.label.to_numpy();new=(p[ix]>=.5)==g.label.to_numpy()
   trans.append(dict(method=method,phase=phase,symbol=sym,changed=int(np.sum((base>=.5)!=(p[ix]>=.5))),corrected=int(np.sum(new&~old)),broken=int(np.sum(old&~new))))
 return pd.DataFrame(rows),pd.DataFrame(monthly),pd.DataFrame(trans)

def gates_table(monthly):
 out=[]
 base=monthly[(monthly.method=='BASE')&monthly.month.isin(['2018-06','2018-07','2018-08'])].set_index(['symbol','month'])
 for m in BLOCKS:
  a=monthly[(monthly.method==m)&monthly.month.isin(['2018-06','2018-07','2018-08'])].set_index(['symbol','month'])
  delta=a[['BA','Brier']]-base[['BA','Brier']];stock=delta.groupby('symbol').mean();macro=delta.groupby('month').BA.mean()
  passed=len(delta)==6 and bool((stock.BA>=.01).all()) and int((macro>0).sum())>=2 and bool((stock.Brier<=.002).all()) and not bool(a.constant.any())
  out.append(dict(method=m,AAPL_delta_BA=stock.loc['AAPL','BA'],AMZN_delta_BA=stock.loc['AMZN','BA'],
   AAPL_delta_Brier=stock.loc['AAPL','Brier'],AMZN_delta_Brier=stock.loc['AMZN','Brier'],positive_macro_months=int((macro>0).sum()),passes=passed))
 return pd.DataFrame(out)

def main():
 if OUT.exists() or MODELS.exists():raise FileExistsError('preserve run')
 OUT.mkdir(parents=True);tick=time.time();d=assemble();raw,gates=build_raw(d)
 # Save outcome-blind feature/coverage audit before fitting.
 audit={'rows':len(d),'keys':d.key.nunique(),'phase_counts':d.phase.value_counts().to_dict(),
  'coverage':{b:{s:int(np.sum(gates[b][d.symbol.to_numpy()==s])) for s in ('AAPL','AMZN')} for b in gates},
  'sources':{str(p.relative_to(ROOT)):sha(p) for p in [W/'paper_methods_4h/v1/inputs.pkl',ROOT/'outputs/stock_foundation_4h/v2/modern_predictions.csv',ROOT/'outputs/stock_combination_4h/v1/predictions.csv',W/'analogy_4h/v2/prepared.csv',W/'analogy_4h/v2/retrieval.json',W/'analogy_4h/v2/retrieval_arrays.npz',W/'finbert_event_adapter_4h/v1/corpus_extraction/facts_A1_ensemble.jsonl']}}
 dump(OUT/'INPUT_AUDIT.json',audit)
 preds,selection,models=execute(d,raw,gates);metrics_df,monthly,trans=report_tables(d,preds);gate=gates_table(monthly)
 keep=['key','symbol','day','month','phase','label','has_original_news','has_usable_text','articles','clusters','n_cases']
 q=d[keep].copy()
 for m,p in preds.items():q[m]=p
 q.to_csv(OUT/'predictions.csv',index=False);metrics_df.to_csv(OUT/'metrics.csv',index=False);monthly.to_csv(OUT/'monthly_metrics.csv',index=False);trans.to_csv(OUT/'transitions.csv',index=False);selection.to_csv(OUT/'selection.csv',index=False);gate.to_csv(OUT/'advancement.csv',index=False)
 selected='BASE';passing=gate[gate.passes]
 if len(passing):selected=passing.sort_values(['AAPL_delta_BA','AMZN_delta_BA'],ascending=False).method.iloc[0]
 manifest={str(p.name):sha(p) for p in models}
 dump(OUT/'MODEL_MANIFEST.json',manifest)
 dump(OUT/'EXECUTION.json',dict(status='COMPLETE',seconds=time.time()-tick,selected=selected,models=len(models),protocol_sha256=sha(HERE/'PRE_REGISTRATION.md'),code_sha256=sha(__file__),all_periods_exposed=True,new_llm_calls=0))
 print(metrics_df[metrics_df.phase.isin(['development','later'])].to_string(index=False));print(gate.to_string(index=False));print('selected',selected)

if __name__=='__main__':main()
