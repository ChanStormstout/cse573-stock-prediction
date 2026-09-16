"""Join outcomes only after forecasts; aggregate metrics and fixed paired cases."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from baseline import metric
from common import B,ROOT,rows,sha,dump

def paired_ci(g,baseline,block,method="llm_joint"):
 days=sorted(g.day.unique());groups={d:np.flatnonzero(g.day.eq(d).to_numpy()) for d in days};rng=np.random.default_rng(573);diff=[]
 for _ in range(500):
  sample=[]
  while len(sample)<len(days):
   start=int(rng.integers(len(days)));sample.extend(days[start:min(len(days),start+block)])
  ix=np.concatenate([groups[d] for d in sample[:len(days)]])
  x=g.iloc[ix]
  if x.label.nunique()<2:continue
  diff.append(metric(x.label,x[method])['BA']-metric(x.label,x[baseline])['BA'])
 return dict(block_days=block,replicates=len(diff),delta_BA=metric(g.label,g[method])['BA']-metric(g.label,g[baseline])['BA'],lo=float(np.quantile(diff,.025)),hi=float(np.quantile(diff,.975)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True,type=Path);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 run=json.loads((a.run/'summary.json').read_text());is_choice=run.get('interface')=='binary_choice_logprobs';assert run['completed'];assert not json.loads((a.run/'manifest.json').read_text())['smoke'];a.out.mkdir(parents=True,exist_ok=False)
 seal=json.loads((a.inputs/'manifest.json').read_text());assert sha(a.inputs/'features.pkl')==seal['features_sha'];d=pd.read_pickle(a.inputs/'features.pkl');ev=d[d.split!='train'].copy();assert len(ev)==609
 meta=ev[['key','symbol','split','day','month','label','has_news']];parts=[];raw_by={}
 for v in ['price','news','joint']:
  rr=rows(a.run/f'{v}.jsonl');raw_by[v]={r['key']:r for r in rr};assert len(rr)==609 and set(raw_by[v])==set(meta.key)
  pp=pd.DataFrame([{k:r[k] for k in ['key','p','valid','evidence_valid','seconds','prompt_tokens','output_tokens']} for r in rr]);z=meta.merge(pp,on='key',validate='one_to_one');z['method']='llm_'+v;parts.append(z)
 base=pd.read_csv(a.baseline/'predictions.csv');assert len(base)==609*3
 for v,g in base.groupby('method'):
  check=meta.merge(g[['key','label']],on='key',validate='one_to_one');assert len(check)==609 and (check.label_x==check.label_y).all()
 base['valid']=True;base['evidence_valid']=np.nan;parts.append(base)
 oldpath=ROOT/'outputs/stock_integrated_4h/runs/v1/predictions.csv';old=pd.read_csv(oldpath);old=old[old.phase.eq('frozen')]
 for v in ['price','title','body','semantic','integrated']:
  g=meta.merge(old[['key','label',v]],on='key',validate='one_to_one',suffixes=('','_old'));assert len(g)==609 and (g.label==g.label_old).all();g=g.drop(columns='label_old').rename(columns={v:'p'});g['method']='legacy_'+v;g['valid']=True;g['evidence_valid']=np.nan;parts.append(g)
 predictions=pd.concat(parts,ignore_index=True);predictions.to_csv(a.out/'predictions.csv',index=False)
 scores=[]
 for (method,symbol),g in predictions.groupby(['method','symbol']):
  for period,x in list(g.groupby('split'))+list(g.groupby('month')):
   scores.append(dict(method=method,symbol=symbol,period=period,scope='all',**metric(x.label,x.p),invalid_n=int((~x.valid).sum()),news_ratio=float(x.has_news.mean()),days=x.day.nunique()))
   if method.startswith('llm_') and (~x.valid).any() and x.valid.any():scores.append(dict(method=method,symbol=symbol,period=period,scope='valid_only_diagnostic',**metric(x[x.valid].label,x[x.valid].p),invalid_n=0,news_ratio=float(x[x.valid].has_news.mean()),days=x[x.valid].day.nunique()))
 pd.DataFrame(scores).to_csv(a.out/'metrics.csv',index=False)
 wide=predictions.pivot(index=['key','symbol','split','day','month','label'],columns='method',values='p').reset_index()
 wide=wide.merge(meta[['key','has_news']],on='key',validate='one_to_one');transitions=[]
 for (symbol,split),g in wide.groupby(['symbol','split']):
  denominators={y:int((g.label==y).sum()) for y in [0,1]}
  for old,new in [('llm_price','llm_joint'),('llm_news','llm_joint'),('matched_joint','llm_joint'),('legacy_title','llm_joint'),('matched_news','llm_news')]:
   for news,sub in g.groupby('has_news'):
    aok=(sub[old]>=.5)==sub.label;bok=(sub[new]>=.5)==sub.label
    contribution=sum(.5*float(((bok.astype(int)-aok.astype(int))[sub.label==y]).sum())/denominators[y] for y in [0,1])
    transitions.append(dict(symbol=symbol,split=split,has_news=int(news),from_method=old,to_method=new,n=len(sub),changed=int((aok!=bok).sum()),corrected=int((~aok&bok).sum()),broken=int((aok&~bok).sum()),both_correct=int((aok&bok).sum()),both_wrong=int((~aok&~bok).sum()),delta_BA_contribution=contribution))
 pd.DataFrame(transitions).to_csv(a.out/'transitions.csv',index=False)
 cis=[];cases=[];empty=[]
 for (symbol,split),g in wide.groupby(['symbol','split']):
  g=g.reset_index(drop=True)
  for method,base in [('llm_joint','matched_joint'),('llm_joint','legacy_title'),('llm_news','matched_news'),('llm_news','legacy_title')]:
   for block in [1,5]:cis.append(dict(symbol=symbol,split=split,method=method,baseline=base,**paired_ci(g,base,block,method)))
  llm=(g.llm_joint>=.5)==g.label;lr=(g.matched_joint>=.5)==g.label
  groups={'both_correct':llm&lr,'both_wrong':~llm&~lr,'llm_only_correct':llm&~lr,'lr_only_correct':~llm&lr}
  for outcome,mask in groups.items():
   sub=g[mask].copy()
   if sub.empty:empty.append(dict(symbol=symbol,split=split,outcome=outcome));continue
   sub['order']=sub.key.map(lambda k:hashlib.sha256(k.encode()).hexdigest());r=sub.sort_values('order').iloc[0];cases.append(dict(id=f'C{len(cases)+1:02d}',key=r.key,symbol=symbol,split=split,outcome=outcome,label=int(r.label),llm_p=float(r.llm_joint),lr_p=float(r.matched_joint)))
 pd.DataFrame(cis).to_csv(a.out/'paired_intervals.csv',index=False);dump(a.out/'case_selection.json',dict(cases=cases,empty_cells=empty,protocol_sha=sha(B/'PROTOCOL.md')))
 inp={r['key']:r for r in rows(a.inputs/'inputs.jsonl')};cards=[dict(**c,input=inp[c['key']],forecast=raw_by['joint'][c['key']]) for c in cases];dump(a.out/'private_cases.json',cards)
 coverage=pd.read_csv(a.inputs/'coverage.csv');cov=[]
 for (sym,split),g in coverage.groupby(['symbol','split']):cov.append(dict(symbol=sym,split=split,n=len(g),news_windows=int((g.eligible>0).sum()),capped_windows=int((g.omitted>0).sum()),article_occurrences=int(g.eligible.sum()),included_occurrences=int(g.included.sum()),omitted_occurrences=int(g.omitted.sum()),body_chars_included=int(g.body_chars_included.sum()),body_chars_omitted=int(g.body_chars_omitted.sum())))
 dump(a.out/'coverage_summary.json',cov)
 quality={}
 for v,rr in raw_by.items():
  values=list(rr.values());quality[v]=dict(n=len(values),invalid=sum(not r['valid'] for r in values),forecast_gate_bad_citations=sum(not r['evidence_valid'] for r in values),p_exact_half=sum(r['p']==.5 for r in values),max_prompt_tokens=max(r['prompt_tokens'] for r in values),seconds=sum(r['seconds'] for r in values),length_stops=sum(r['finish_reason']=='length' for r in values),raw_json_failures=0,bad_citations=0,citations_not_evaluable=0,raw_p_exact_half=0,direction_probability_conflicts=0)
  for r in values:
   text=json.dumps(r['parsed']) if is_choice else r['raw'].strip()
   if text.startswith('```') and text.endswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
   try:obj=json.loads(text)
   except (ValueError,TypeError):obj=None
   if not isinstance(obj,dict):
    quality[v]['raw_json_failures']+=1;quality[v]['citations_not_evaluable']+=1;continue
   pp=obj.get('p_up');direction=obj.get('direction');pv=isinstance(pp,(int,float)) and not isinstance(pp,bool) and np.isfinite(pp) and 0<=pp<=1
   quality[v]['raw_p_exact_half']+=int(pv and pp==.5)
   quality[v]['direction_probability_conflicts']+=int(pv and direction in ['UP','DOWN'] and direction!=('UP' if pp>=.5 else 'DOWN'))
   allowed=set()
   if v!='news':allowed.update(x['id'] for x in inp[r['key']]['price_rows'])
   if v!='price':allowed.update(x['id'] for x in inp[r['key']]['news'])
   ids=obj.get('evidence_ids',[]);ok=isinstance(ids,list) and len(ids)<=2 and all(isinstance(i,str) and i in allowed for i in ids)
   quality[v]['bad_citations']+=int(not ok)
  if is_choice:
   quality[v].update(interface='binary_choice_logprobs',citations_not_requested=True,min_choice_mass=min(r['choice_mass'] for r in values),median_choice_mass=float(np.median([r['choice_mass'] for r in values])),max_choice_mass=max(r['choice_mass'] for r in values),unconstrained_off_label=sum(r['unconstrained_token'] not in ['UP','DOWN'] for r in values))
 native=[];native_rows=[]
 for v,rr in raw_by.items():
  for key,r in rr.items():
   raw=json.dumps(r['parsed']) if is_choice else r['raw'].strip()
   if raw.startswith('```') and raw.endswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
   try:x=json.loads(raw)
   except (ValueError,TypeError):x={}
   if not isinstance(x,dict):x={}
   direction=x.get('direction');p=x.get('p_up');pv=isinstance(p,(int,float)) and not isinstance(p,bool) and np.isfinite(p) and 0<=p<=1
   native_rows.append(dict(key=key,method='llm_'+v,native_up=int(direction!='DOWN'),direction_valid=direction in ['UP','DOWN'],raw_p=float(p) if pv else .5,probability_valid=bool(pv)))
 native_frame=meta.merge(pd.DataFrame(native_rows),on='key',validate='one_to_many')
 for (method,symbol),g in native_frame.groupby(['method','symbol']):
  for period,x in list(g.groupby('split'))+list(g.groupby('month')):
   mm=metric(x.label,x.native_up);mm.pop('Brier')
   native.append(dict(method=method,symbol=symbol,period=period,**mm,raw_probability_Brier=metric(x.label,x.raw_p)['Brier'],direction_failures=int((~x.direction_valid).sum()),probability_failures=int((~x.probability_valid).sum())))
 pd.DataFrame(native).to_csv(a.out/'native_direction_diagnostic.csv',index=False)
 native_frame.to_csv(a.out/'native_predictions.csv',index=False)
 dump(a.out/'quality.json',quality);dump(a.out/'verification.json' ,dict(completed=True,windows=609,all_original_windows=1607,input_seal=seal['input_sha'],forecast_hashes={v:sha(a.run/f'{v}.jsonl') for v in raw_by},baseline_summary_sha=sha(a.baseline/'summary.json'),legacy_predictions_sha=sha(oldpath),interface=run.get('interface','generated_json'),new_training='60 LR fits, no LLM weight training',independent_holdout=False,analysis_sha=sha(__file__)))
 print(pd.DataFrame(scores).query("period=='test' and scope=='all'")[['method','symbol','BA','Brier','pred_up','invalid_n']].to_string(index=False),flush=True)
if __name__=='__main__':main()
