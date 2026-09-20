"""Report only independently verified stages; never selects using outer scores."""
from engine import *

def measured(y,p):
 if len(set(y))<2:return dict(n=len(y),BA=np.nan,MCC=np.nan,Brier=float(np.mean((np.array(y)-p)**2)),predicted_up=float(np.mean(p>=.5)),constant=bool(np.all(p>=.5) or np.all(p<.5)))
 return metric(y,p)
def main():
 assert json.loads((PUB/'INPUT_VERIFICATION.json').read_text())['status']=='PASS'
 d,*_=load();methods={'preprocess':['PRICE','FULL','CONTROL']+VARIANTS,'grouped':['PRICE','FULL','GROUP_SVM'],'aggregation':['PRICE','FULL','CONTROL','SET_BASE','MEAN_LINEAR','MEAN_THEN_MAP','MAP_THEN_MEAN']}
 metrics=[];monthly=[];foldmetrics=[];coverage=[];changes=[];intervals=[];cases=[];evidence=[];allstage={};selected=[]
 days=np.array(sorted(d.day.unique()));rng=np.random.default_rng(573);weights={}
 for typ in ['day1','day5']:
  counts=[]
  for _ in range(1000):
   ix=rng.integers(len(days),size=len(days)) if typ=='day1' else ((rng.integers(len(days),size=int(np.ceil(len(days)/5)))[:,None]+np.arange(5))%len(days)).ravel()[:len(days)]
   counts.append(np.bincount(ix,minlength=len(days)))
  weights[typ]=np.array(counts)[:,np.searchsorted(days,d.day)]
 for stage,ms in methods.items():
  vf=PUB/(stage+'_VERIFICATION.json');assert json.loads(vf.read_text())['status']=='PASS'
  p=read(PUB/(stage+'_predictions.csv'));p=p.merge(d[['row_id','month','has_news']],on='row_id',suffixes=('_saved',''),validate='many_to_one');allstage[stage]=p
  for (stock,seed,block),g in p.groupby(['symbol','seed','block']):
   for m in ms:foldmetrics.append(dict(stage=stage,symbol=stock,seed=seed,block=block,method=m,**measured(g.label.to_numpy(),g[m].to_numpy())))
  ref='FULL' if stage=='grouped' else 'CONTROL'
  for (stock,seed),g in p.groupby(['symbol','seed']):
   for m in ms:
    metrics.append(dict(stage=stage,symbol=stock,seed=seed,method=m,**measured(g.label.to_numpy(),g[m].to_numpy())))
    for mon,h in g.groupby('month'):monthly.append(dict(stage=stage,symbol=stock,seed=seed,method=m,month=mon,**measured(h.label.to_numpy(),h[m].to_numpy())))
    for has,h in g.groupby(g.has_news>0):coverage.append(dict(stage=stage,symbol=stock,seed=seed,method=m,has_news=bool(has),**measured(h.label.to_numpy(),h[m].to_numpy())))
    if m in ['PRICE','FULL','CONTROL']:continue
    for reference in list(dict.fromkeys([ref,'FULL']+(['SET_BASE','MEAN_THEN_MAP'] if stage=='aggregation' else []))):
     y=g.label.to_numpy();old=g[reference].to_numpy()>=.5;new=g[m].to_numpy()>=.5
     changes.append(dict(stage=stage,symbol=stock,seed=seed,method=m,reference=reference,changed=int(sum(old!=new)),repaired=int(sum((old!=y)&(new==y))),introduced=int(sum((old==y)&(new!=y))),delta_BA=metric(y,g[m])['BA']-metric(y,g[reference])['BA']))
    if seed==573:
     for cat,oldok,newok in [('repaired',False,True),('harmed',True,False),('both_wrong',False,False),('both_right',True,True)]:
      h=g[(((g[ref]>=.5)==g.label)==oldok)&(((g[m]>=.5)==g.label)==newok)].copy();h['rank']=[digest(stage+stock+m+cat+k) for k in h.row_id]
      for row in h.sort_values('rank').head(2).itertuples():cases.append(dict(stage=stage,symbol=stock,method=m,category=cat,row_id=row.row_id,day=row.day,label=row.label,reference=ref,base_probability=float(getattr(row,ref)),new_probability=float(getattr(row,m))))
  for stock in ['AAPL','AMZN']:
   ix=np.flatnonzero(d.symbol==stock);y=d.iloc[ix].label.to_numpy();ids=d.iloc[ix].row_id
   for m in ms:
    if m in ['PRICE','FULL','CONTROL']:continue
    for reference in list(dict.fromkeys([ref,'FULL']+(['SET_BASE','MEAN_THEN_MAP'] if stage=='aggregation' else []))):
     delta=[]
     for seed in SEEDS:
      g=p[(p.symbol==stock)&(p.seed==seed)].set_index('row_id').loc[ids];delta.append(((g[m].to_numpy()>=.5)==y).astype(float)-((g[reference].to_numpy()>=.5)==y).astype(float))
     delta=np.mean(delta,0)
     for typ,w in weights.items():
      w=w[:,ix];a=w[:,y==0].sum(1);b=w[:,y==1].sum(1);ok=(a>0)&(b>0);z=.5*((w[:,y==0]@delta[y==0])/np.maximum(a,1)+(w[:,y==1]@delta[y==1])/np.maximum(b,1))
      intervals.append(dict(stage=stage,symbol=stock,method=m,reference=reference,bootstrap=typ,delta_BA=.5*(delta[y==0].mean()+delta[y==1].mean()),low=float(np.quantile(z[ok],.025)),high=float(np.quantile(z[ok],.975))))
  for root in (WORK/stage).iterdir():
   if not root.is_dir():continue
   ledger=json.loads((root/'training.json').read_text());evidence.extend(dict(stage=stage,block=root.name,**{k:r[k] for k in ['file','seconds','method','C','inner_fold']}) for r in ledger)
 met=pd.DataFrame(metrics);met.to_csv(PUB/'metrics.csv',index=False);summary=met.groupby(['stage','method','symbol'])[['BA','MCC','Brier','predicted_up']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(PUB/'summary.csv',index=False)
 for name,rows in [('monthly_metrics',monthly),('fold_metrics',foldmetrics),('coverage',coverage),('direction_changes',changes),('paired_intervals',intervals),('case_index',cases)]:pd.DataFrame(rows).to_csv(PUB/(name+'.csv'),index=False)
 ev=pd.DataFrame(evidence);dump(PUB/'training_evidence.json',dict(status='COMPLETE_VERIFIED',top_level_fit_calls=len(ev),SVM_internal_classifier_fits=int((ev.stage!='aggregation').sum()*3),sigmoid_calibrators=int((ev.stage!='aggregation').sum()*3),LR_fits=int((ev.stage=='aggregation').sum()),fit_seconds=float(ev.seconds.sum()),device='CPU',threads=2,encoder_training=False,encoder_inference=False,LLM_continued=False,counts=ev.groupby('stage').size().to_dict()))
 # Frozen descriptive budget screen, never a claim of independent significance.
 gate={};s=met[met.stage=='aggregation']
 for stock in ['AAPL','AMZN']:
  g=s[s.symbol==stock].pivot(index='seed',columns='method',values=['BA','Brier']);entry={}
  for reference in ['SET_BASE','MEAN_THEN_MAP']:
   delta=g['BA']['MAP_THEN_MEAN']-g['BA'][reference];db=g['Brier']['MAP_THEN_MEAN']-g['Brier'][reference];entry[reference]=dict(mean_delta_BA=float(delta.mean()),positive_seeds=int((delta>0).sum()),mean_delta_Brier=float(db.mean()),passes=bool(delta.mean()>0 and (delta>0).sum()>=2 and db.mean()<=.002))
  gate[stock]=entry
 passes=all(v['passes'] for stock in gate.values() for v in stock.values());dump(PUB/'FOLLOWUP_SCREEN.json',dict(status='DESCRIPTIVE_BUDGET_SCREEN_ONLY',passes=passes,values=gate,decision='Interaction proposal eligible; requires separate finite registration' if passes else 'Do not expand aggregation to price interactions; no automatic TabPFN or model grid'))
 lines=['# Representation diagnostics and aggregation order','','**EXPOSED EXPLORATORY RANDOM HISTORICAL BACKTEST.** All 1,607 original four-hour windows, three fixed seeds. No future generalization claim. Inner-only C selection. Grouped testing is a separate robustness protocol.','','## Results: three-seed means and standard deviations','']
 for stage in methods:
  lines+=['### '+stage,'',summary[summary.stage==stage].to_markdown(index=False,floatfmt='.5f'),'']
 lines+=['## Interpretation rules','','- Preprocessing changes exactly one historical lexical operation per branch. Frequency can also change the 5,000-term vocabulary through term-frequency selection; that is part of this mechanism, not isolated coefficient-only reweighting.','- Grouped SVM has group-disjoint outer, inner and calibration splits. Historical ordinary SVM used stratified calibration. Thus cross-protocol score movement includes the necessary calibration isolation change. Components are association groups, not independently reviewed financial events.','- Aggregation uses a new matched TF-IDF + price LR reference (SET_BASE). It is not silently relabeled as historical FULL or SVM. All three semantic representations share that exact classifier/input design and C budget.','- MEAN_LINEAR keeps768 dimensions; the other two both use the same256-dimensional random mapping. MEAN_THEN_MAP vs MAP_THEN_MEAN is the strict aggregation-order comparison; linear vs nonlinear additionally changes feature geometry/dimension.','- The chunk pool already averages tokens within each chunk and selects target-company paragraphs. Mapping cannot recover missing passages, word-level relations or absent AMZN news.','- No-news windows return saved PRICE exactly; news with no qualifying paragraph use zero semantic block. No evidence-quality acceptance is implied.','- Paired intervals are descriptive after repeated exploration. Fixed thresholds, no outer-selected stock-specific winner or best seed.','', '## Actual training','', '```json',json.dumps(json.loads((PUB/'training_evidence.json').read_text()),indent=2),'```','','## Follow-up screen','', '```json',json.dumps(json.loads((PUB/'FOLLOWUP_SCREEN.json').read_text()),indent=2),'```','','See monthly_metrics.csv, coverage.csv, direction_changes.csv, paired_intervals.csv and selected_C files. All source text, vectors and weights remain private.']
 (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n');(PUB/'CASE_NOTES.md').write_text('# Fixed diagnostic case panel\n\nStable hash first two per outcome category, seed573. These are prediction observations, not causal explanations. Raw articles remain private.\n\n'+pd.DataFrame(cases).to_markdown(index=False)+'\n')
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(2):main()
