"""Verified fixed results, paired uncertainty and coverage; no model fitting."""
from core import *
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

NEW=['A1','A2','A3','B1','B2','FULL_SHRINK']
BASE=['PAPER','PRICE','FULL','TFIDF_LR','TFIDF_SVM','TFIDF_RF','TFIDF_LR_PRICE_FALLBACK','TFIDF_SVM_PRICE_FALLBACK','TFIDF_RF_PRICE_FALLBACK']
HIST=['TITLE_FINBERT_FUSION','TITLE_MODERN_FUSION','PARAGRAPH_FINBERT_FUSION','PARAGRAPH_MODERN_FUSION']
def extended(y,p):
 y=np.asarray(y);p=np.asarray(p);q=p>=.5
 if len(np.unique(y))<2:
  r=dict(n=len(y),BA=None,Brier=float(np.mean((y-p)**2)),MCC=None,accuracy=float((q==y).mean()),predicted_up=float(q.mean()),constant=bool(q.min()==q.max()))
 else:r=metric(y,p)
 r.update(log_loss=float(log_loss(y,p,labels=[0,1])),TPR=float(q[y==1].mean()) if (y==1).any() else None,TNR=float((~q[y==0]).mean()) if (y==0).any() else None)
 return r
def fmt_table(frame):return frame.to_markdown(index=False,floatfmt='.4f')
def main():
 assert json.loads((PUBLIC/'VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((PUBLIC/'INPUT_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((PUBLIC/'PREPROCESSING_VERIFICATION.json').read_text())['status']=='PASS'
 d,*_=load();_,ids,_,gate=semantic(d);p=read_csv(PUBLIC/'predictions.csv')
 for filename,prefix in [('TITLE_FUSION_PREDICTIONS.csv','TITLE'),('SEMANTIC_PREDICTIONS.csv','PARAGRAPH')]:
  q=read_csv(ROOT/'outputs/stock_full_semantics_4h/v1'/filename)
  for enc in ['FINBERT','MODERN']:
   z=q[q.encoder==enc][['seed','row_id','p']].rename(columns={'p':f'{prefix}_{enc}_FUSION'})
   p=p.merge(z,on=['seed','row_id'],validate='one_to_one')
 candidate_rows=[]
 for root in sorted((WORK/'training').iterdir()):
  seed,stock,fold=root.name.split('_');z=read_csv(root/'selection.csv');z['seed']=int(seed);z['symbol']=stock;z['fold']=int(fold);candidate_rows.append(z)
 pd.concat(candidate_rows,ignore_index=True).to_csv(PUBLIC/'inner_candidate_metrics.csv',index=False)
 methods=BASE+NEW+HIST;p=p.merge(d[['row_id','month','has_news']].rename(columns={'month':'canonical_month','has_news':'canonical_has_news'}),on='row_id',validate='many_to_one')
 p['coverage']=np.where(p.canonical_has_news==0,'no_news',np.where(p.gate,'target_paragraph','news_without_target_paragraph'))
 metrics=[];monthly=[];changes=[];coverage=[];upper=[]
 for (stock,seed),g in p.groupby(['symbol','seed']):
  for m in methods:
   metrics.append(dict(symbol=stock,seed=seed,method=m,**extended(g.label,g[m])))
   for mon,h in g.groupby('canonical_month'):monthly.append(dict(symbol=stock,seed=seed,method=m,month=mon,**extended(h.label,h[m])))
  for ref in ['FULL','TFIDF_SVM']:
   for m in NEW+['TFIDF_SVM_PRICE_FALLBACK']:
    for cls in [0,1]:
     h=g[g.label==cls];old=h[ref].to_numpy()>=.5;new=h[m].to_numpy()>=.5;y=h.label.to_numpy()
     changes.append(dict(symbol=stock,seed=seed,reference=ref,method=m,true_label=cls,n=len(h),changed=int((old!=new).sum()),repaired=int(((old!=y)&(new==y)).sum()),introduced=int(((old==y)&(new!=y)).sum()),delta_BA=extended(g.label,g[m])['BA']-extended(g.label,g[ref])['BA']))
  for group,h in g.groupby('coverage'):
   for m in ['FULL','TFIDF_SVM']+NEW:
    ref=h.FULL.to_numpy()>=.5;new=h[m].to_numpy()>=.5;y=h.label.to_numpy();mutable=(h.canonical_has_news>0 if m.startswith('A') else h.gate) if m in NEW[:5] else np.zeros(len(h),bool)
    coverage.append(dict(symbol=stock,seed=seed,coverage=group,method=m,n=len(h),up=int(y.sum()),down=int((y==0).sum()),errors=int((new!=y).sum()),can_modify=int(np.asarray(mutable).sum()),changed=int((new!=ref).sum()),repaired=int(((new==y)&(ref!=y)).sum()),introduced=int(((new!=y)&(ref==y)).sum()),**{k:v for k,v in extended(y,h[m]).items() if k!='n'}))
  for scope,mask in [('A_news',g.canonical_has_news.to_numpy()>0),('B_target_paragraph',g.gate.to_numpy(bool))]:
   q=g.FULL.to_numpy()>=.5;q[mask]=g.label.to_numpy()[mask];upper.append(dict(symbol=stock,seed=seed,scope=scope,mutable=int(mask.sum()),oracle_BA=float(balanced_accuracy_score(g.label,q)),note='UNDEPLOYABLE KNOWS-ANSWERS UPPER BOUND; not selection evidence'))
 met=pd.DataFrame(metrics);met.to_csv(PUBLIC/'metrics.csv',index=False);pd.DataFrame(monthly).to_csv(PUBLIC/'monthly_metrics.csv',index=False);pd.DataFrame(changes).to_csv(PUBLIC/'direction_changes.csv',index=False);pd.DataFrame(coverage).to_csv(PUBLIC/'coverage_diagnostic.csv',index=False);pd.DataFrame(upper).to_csv(PUBLIC/'oracle_scope_diagnostic.csv',index=False)
 summary=met.groupby(['method','symbol'])[['BA','accuracy','MCC','Brier','log_loss','predicted_up','TPR','TNR']].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index();summary.to_csv(PUBLIC/'summary.csv',index=False)
 # Common cluster weights across both stocks and all seeds; no iid seed assumption.
 groups=read_csv(PRIVATE/'groups.csv').set_index('row_id').group.reindex(d.row_id).to_numpy();days=np.array(sorted(d.day.unique()));rng=np.random.default_rng(573);boot={}
 for name,labels in [('day1',d.day.to_numpy()),('day5',d.day.to_numpy()),('association_group',groups)]:
  levels=np.unique(labels);inverse=np.searchsorted(levels,labels);n=len(levels);counts=np.zeros((1000,n),int)
  for j in range(1000):
   sample=rng.integers(n,size=n) if name!='day5' else ((rng.integers(n,size=int(np.ceil(n/5)))[:,None]+np.arange(5))%n).ravel()[:n]
   counts[j]=np.bincount(sample,minlength=n)
  boot[name]=counts[:,inverse]
 intervals=[]
 for stock in ['AAPL','AMZN']:
  ix=np.flatnonzero(d.symbol==stock);y=d.iloc[ix].label.to_numpy();rowids=d.iloc[ix].row_id
  for ref in ['FULL','TFIDF_SVM']:
   for m in NEW+['TFIDF_SVM_PRICE_FALLBACK']:
    delta=[]
    for seed in SEEDS:
     h=p[(p.symbol==stock)&(p.seed==seed)].set_index('row_id').loc[rowids]
     delta.append(((h[m].to_numpy()>=.5)==y).astype(float)-((h[ref].to_numpy()>=.5)==y).astype(float))
    delta=np.mean(delta,axis=0);point=.5*(delta[y==0].mean()+delta[y==1].mean())
    for name,weights in boot.items():
     w=weights[:,ix];den0=w[:,y==0].sum(1);den1=w[:,y==1].sum(1);valid=(den0>0)&(den1>0)
     scores=.5*((w[:,y==0]@delta[y==0])/np.maximum(den0,1)+(w[:,y==1]@delta[y==1])/np.maximum(den1,1));scores=scores[valid]
     intervals.append(dict(symbol=stock,reference=ref,method=m,resampling=name,delta_BA=point,low=float(np.quantile(scores,.025)),high=float(np.quantile(scores,.975)),replicates=len(scores)))
 pd.DataFrame(intervals).to_csv(PUBLIC/'paired_intervals.csv',index=False)
 paired=[]
 for (stock,seed),g in met.groupby(['symbol','seed']):
  indexed=g.set_index('method')
  for ref in ['FULL','TFIDF_SVM','TFIDF_SVM_PRICE_FALLBACK']:
   for m in NEW+['TFIDF_SVM_PRICE_FALLBACK']:
    paired.append(dict(symbol=stock,seed=seed,reference=ref,method=m,**{f'delta_{k}':float(indexed.loc[m,k]-indexed.loc[ref,k]) for k in ['BA','accuracy','MCC','Brier','log_loss']}))
 pd.DataFrame(paired).to_csv(PUBLIC/'paired_metric_deltas.csv',index=False)
 # Cases fixed by preregistered hash rule; originals only in the private pack.
 inp=joblib.load(WORK/'inputs.joblib');cases=[];public=[];panel_counts=[];lookup=d.set_index('row_id');g0=p[p.seed==573]
 raw_index=pd.read_pickle(SOURCES['raw_index']);raw_index['rk']=raw_index.archive+'::'+raw_index.member;raw_titles=raw_index.set_index('rk').title.to_dict()
 for stock in ['AAPL','AMZN']:
  g=g0[g0.symbol==stock]
  for m in NEW[:5]:
   for typ,okbase,oknew in [('repaired',False,True),('harmed',True,False),('both_wrong',False,False),('both_right',True,True)]:
    h=g[((g.FULL>=.5)==g.label)==okbase];h=h[((h[m]>=.5)==h.label)==oknew]
    keys=sorted(h.row_id,key=lambda k:digest(stock+'|'+m+'|'+typ+'|'+k))[:2]
    panel_counts.append(dict(symbol=stock,method=m,category=typ,available=len(h),selected=len(keys)))
    for key in keys:cases.append(dict(symbol=stock,method=m,category=typ,row_id=key))
  tags={}
  for r in g.itertuples():
   row=lookup.loc[r.row_id];keys=[k for k in row.news_record_keys.split('|') if k];text=' '.join(t for k in keys for s in inp['articles'][k]['parts'] for t in s['tokens'])
   tags[r.row_id]=dict(no_news=not keys,negation=bool(re.search(r'\b(?:not|no|nor)\b',text)),numbers=bool(re.search(r'\d',text)),multi_company=bool(re.search(r'\b(?:apple|aapl)\b',text) and re.search(r'\b(?:amazon|amzn)\b',text)),near_repost=False)
   # Frozen association group >1 article-linked windows is only a candidate,
   # not a certified duplicate event. Use exact normalized title duplicates here.
   titles=[str(raw_titles[k]).lower() for k in keys];bags=[set(re.findall(r'[a-z]+|[0-9]+',t)) for t in titles]
   tags[r.row_id]['near_repost']=any(len(bags[i]&bags[j])/max(1,len(bags[i]|bags[j]))>=.8 and re.findall(r'\d+',titles[i])==re.findall(r'\d+',titles[j]) for i in range(len(titles)) for j in range(i))
  for tag in ['no_news','negation','numbers','multi_company','near_repost']:
   ks=sorted([k for k,t in tags.items() if t[tag]],key=lambda k:digest(stock+'|'+tag+'|'+k))[:2]
   panel_counts.append(dict(symbol=stock,method='A3',category='supplement_'+tag,available=sum(t[tag] for t in tags.values()),selected=len(ks)))
   for key in ks:cases.append(dict(symbol=stock,method='A3',category='supplement_'+tag,row_id=key))
 for c in cases:
  row=lookup.loc[c['row_id']];q=g0.set_index('row_id').loc[c['row_id']];keys=[k for k in row.news_record_keys.split('|') if k]
  # Evidence is readable locally. Public notes contain identifiers and factual
  # model changes, not redistributable course-news excerpts.
  c.update(day=row.day,label=int(row.label),FULL=float(q.FULL),new=float(q[c['method']]),article_keys=keys,evidence=[{'key':k,'sentences':inp['articles'][k]['parts']} for k in keys])
  public.append({k:v for k,v in c.items() if k not in ['article_keys','evidence']})
 dump(PUBLIC/'CASE_PANEL_COVERAGE.json',dict(rule='fixed stable hash; no substitute when category empty',counts=panel_counts));dump(WORK/'case_evidence.json',cases);pd.DataFrame(public).to_csv(PUBLIC/'case_index.csv',index=False)
 notes=['# Fixed case panel','', 'Seed 573; stable hashes select two cases per stock/method/outcome category. These are diagnosis examples, never training or parameter-selection data. Raw evidence remains in the private case pack. Article labels are not financial-event gold.','',fmt_table(pd.DataFrame(public)),'','No-news rows: A1–A3 and B1–B2 retain PRICE exactly. A word or paragraph cue can change a prediction without reliably determining the subsequent market reaction. Model differences alone do not prove the original news caused the return.']
 (PUBLIC/'CASE_NOTES.md').write_text('\n'.join(notes)+'\n')
 tables=summary.pivot(index='method',columns='symbol',values=['BA_mean','BA_std','Brier_mean','log_loss_mean']).reindex(methods);tables.columns=['_'.join(x) for x in tables.columns]
 report=['# Ordered text and joint FinBERT study','', '**EXPLORATORY RANDOM HISTORICAL BACKTEST.** Same exposed 1,607 windows, three seeds and nested ten-fold protocol. No chronological scores are mixed here. FULL remains the fixed reference; SVM the strong classical comparator.','', '## Actual execution','',json.dumps(json.loads((PUBLIC/'training_evidence.json').read_text()),indent=2),'', '## Three-seed results','',fmt_table(tables.reset_index()),'', 'A1: ordered raw unigrams + LR; A2: add within-sentence bigrams; A3: same text + calibrated SVM. A methods use exact PRICE fallback. B1/B2: original FULL features plus frozen target-company paragraph vectors, PCA16/uncompressed, in one LR. They are not complete-full-body encodings. No encoder was retrained.','', '## Interpretation boundaries','', '- A1 versus old TF-IDF changes order/frequency/numeric/negation retention together; it cannot identify one cause. A1/A2 and A2/A3 are closer component controls. A versus FULL changes both price handling and text/classifier choices.','- B1/B2 share inputs, gating and classifier; this is the matched PCA comparison. Older probability fusion has different metadata/scales/selection and is a historical system reference, not an isolated fusion-location ablation.','- FULL_SHRINK cannot change directions and supplies no new information. Its Brier/log-loss improvement is a confidence control.','- All transforms and SVM calibration text pipelines fit only allowed train rows. Selected FULL inner OOF is finite selection data, never learned stacking training.','- Day/group intervals are descriptive after repeated exploration, not fresh significance tests. Seeds/folds are not independent market samples.','- No per-stock outer-score winner is assembled. No attention, LoRA, RL, new data or LLM continuation.','', '## Coverage and correction scope','', 'BA/MCC are undefined (blank) in single-class subgroups; TPR/TNR are reported only where their true class exists. Full-stock metrics contain both classes.','',fmt_table(pd.DataFrame(coverage).groupby(['symbol','coverage','method'])[['n','errors','changed','repaired','introduced']].mean().reset_index()),'', '## Paired differences','',fmt_table(pd.DataFrame(intervals)),'', 'See metrics.csv, monthly_metrics.csv, selection_records.csv, direction_changes.csv, oracle_scope_diagnostic.csv and CASE_NOTES.md. The oracle scope is explicitly undeployable and never used for selection.']
 (PUBLIC/'REPORT.md').write_text('\n'.join(report)+'\n');print(tables.to_string())
if __name__=='__main__':
 with threadpool_limits(2):main()
