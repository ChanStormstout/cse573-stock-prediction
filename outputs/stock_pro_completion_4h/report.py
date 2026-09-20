"""Summarize verified runs; no tuning or model choice based on outer results."""
from models import *
STAGES={'lexical':['UNI','BIGRAM','NB_BIGRAM'],'kernel':['K_WP','K_MEAN','K_AFTER','K_SET','K_INTERACTION'],'table':['TABLE_'+l+'_'+s for l in ['LR','RF','TABPFN'] for s in ['NOSEM','SEM']],'residual':['RBASE','R16']}
def measured(y,p):
 return metric(y,p) if len(set(y))==2 else dict(n=len(y),BA=np.nan,MCC=np.nan,Brier=float(np.mean((np.array(y)-p)**2)),predicted_up=float(np.mean(p>=.5)),constant=bool(np.all(p>=.5) or np.all(p<.5)))
def refs(stage,name):
 if stage=='lexical':return ['UNI','FULL']
 if stage=='kernel':return ['K_WP','K_AFTER','FULL']
 if stage=='table':return [name.replace('_SEM','_NOSEM'),'FULL'] if name.endswith('_SEM') else ['FULL']
 return ['RBASE']
def main():
 d,*_=load();metrics=[];monthly=[];folds=[];coverage=[];changes=[];intervals=[];cases=[];evidence={};selection=[]
 days=np.array(sorted(d.day.unique()));rng=np.random.default_rng(573);weights={}
 for size in [1,5]:
  w=[]
  for _ in range(1000):
   ix=((rng.integers(len(days),size=int(np.ceil(len(days)/size)))[:,None]+np.arange(size))%len(days)).ravel()[:len(days)];w.append(np.bincount(ix,minlength=len(days)))
  weights[size]=np.array(w)[:,np.searchsorted(days,d.day)]
 for stage,names in STAGES.items():
  vf=json.loads((OUT/f'{stage}_VERIFICATION.json').read_text());assert vf['status']=='PASS';evidence[stage]=vf
  p=read(OUT/f'{stage}_predictions.csv');methods=['PRICE','FULL','FINBERT','MODERN']+names
  for (stock,seed),g in p.groupby(['symbol','seed']):
   for name in methods:
    metrics.append(dict(stage=stage,symbol=stock,seed=seed,method=name,**measured(g.label.to_numpy(),g[name].to_numpy())))
    for month,h in g.groupby('month'):monthly.append(dict(stage=stage,symbol=stock,seed=seed,month=month,method=name,**measured(h.label.to_numpy(),h[name].to_numpy())))
    for block,h in g.groupby('block'):folds.append(dict(stage=stage,symbol=stock,seed=seed,block=block,method=name,**measured(h.label.to_numpy(),h[name].to_numpy())))
    for news,h in g.groupby(g.has_news>0):coverage.append(dict(stage=stage,symbol=stock,seed=seed,has_news=bool(news),method=name,**measured(h.label.to_numpy(),h[name].to_numpy())))
    if name not in names:continue
    for ref in dict.fromkeys(refs(stage,name)):
     if ref==name:continue
     y=g.label.to_numpy();old=g[ref].to_numpy()>=.5;new=g[name].to_numpy()>=.5
     changes.append(dict(stage=stage,symbol=stock,seed=seed,method=name,reference=ref,changed=int(sum(old!=new)),repaired=int(sum((old!=y)&(new==y))),introduced=int(sum((old==y)&(new!=y)))))
     if seed==573 and ref==refs(stage,name)[0]:
      for cat,a,b in [('repaired',False,True),('harmed',True,False),('both_wrong',False,False),('both_right',True,True)]:
       h=g[(((g[ref]>=.5)==g.label)==a)&(((g[name]>=.5)==g.label)==b)].copy();h['rank']=[digest(stage+name+cat+r) for r in h.row_id]
       for _,r in h.sort_values('rank').head(2).iterrows():cases.append(dict(stage=stage,method=name,symbol=stock,category=cat,row_id=r.row_id,day=r.day,label=int(r.label),reference=ref,p_base=r[ref],p_new=r[name]))
  for stock in ['AAPL','AMZN']:
   ix=np.flatnonzero(d.symbol==stock);y=d.iloc[ix].label.to_numpy();ids=d.iloc[ix].row_id
   for name in names:
    for ref in dict.fromkeys(refs(stage,name)):
     if ref==name:continue
     delta=[]
     for seed in SEEDS:
      g=p[(p.symbol==stock)&(p.seed==seed)].set_index('row_id').loc[ids];delta.append(((g[name].to_numpy()>=.5)==y).astype(float)-((g[ref].to_numpy()>=.5)==y).astype(float))
     delta=np.mean(delta,0)
     for size,w in weights.items():
      w=w[:,ix];a=w[:,y==0].sum(1);b=w[:,y==1].sum(1);ok=(a>0)&(b>0);z=.5*((w[:,y==0]@delta[y==0])/np.maximum(a,1)+(w[:,y==1]@delta[y==1])/np.maximum(b,1));intervals.append(dict(stage=stage,method=name,reference=ref,symbol=stock,block_days=size,delta_BA=.5*(delta[y==0].mean()+delta[y==1].mean()),low=float(np.quantile(z[ok],.025)),high=float(np.quantile(z[ok],.975))))
  for root in (LOCAL/stage).iterdir():
   if not root.is_dir():continue
   for name in names:
    obj=joblib.load(root/f'{name}.joblib');selection.append(dict(stage=stage,block=root.name,method=name,C=obj.get('C'),gamma=obj.get('gamma')))
 met=pd.DataFrame(metrics);summary=met.groupby(['stage','method','symbol'])[['BA','MCC','Brier','predicted_up']].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
 for name,rs in [('metrics',metrics),('monthly_metrics',monthly),('fold_metrics',folds),('coverage',coverage),('direction_changes',changes),('paired_intervals',intervals),('case_index',cases),('selected_parameters',selection)]:pd.DataFrame(rs).to_csv(OUT/f'{name}.csv',index=False)
 summary.to_csv(OUT/'summary.csv',index=False);dump(OUT/'training_evidence.json',evidence)
 lines=['# GPT Pro finite mechanism completion','','**EXPOSED EXPLORATORY RANDOM HISTORICAL BACKTEST.** Same 1,607 four-hour windows; 3 seeds x10 outer folds. These are not past-to-future scores. Prior grouped tests show strong split sensitivity. No best seed or per-stock outer winner is selected.','','## Three-seed results','']
 for stage in STAGES:lines+=['### '+stage,'',summary[summary.stage==stage].to_markdown(index=False,floatfmt='.5f'),'']
 lines+=['## What was tested','','- Strict real-adjacency bigrams versus same unigram/calibrated LinearSVC; optional NB weighting is a separate inspired variant.','- SVC with summed word, price and semantic kernels. Mean-before/nonlinear-after and nonlinear-before/mean-after use the same frozen Gaussian map. Price interactions are tested regardless of semantic main-effect results. This is fixed-weight finite MKL-inspired comparison, not a full SimpleMKL optimizer.','- Independent compact table: 128 selected word/phrase features, 16 price features, 2 metadata features, optionally32 PCA semantic features. Same-table LR, shallow RF and fixed local TabPFN2.5/6.3.0. TabPFN uses training labels as context, not new gradient finetuning. Existing checkpoint was fine-tuned on real data; not synthetic-only.','- R16 uses fresh nested cross-fitted base probabilities in every residual-training scope; no old selection OOF is used for training the correction. OFF is a candidate; missing evidence preserves base exactly.','- Full versus no-evidence base fallbacks and unchanged no-news PRICE remain explicit. FinBERT vectors are frozen existing caches; no encoder or LLM inference/training occurred.','','## Limits','','The course period has been exposed repeatedly. Random overlap can make similar news and market regimes appear across folds. Numerical gains are not prospective validation or causal attribution. Paired day/block intervals are descriptive, not multiple-search-adjusted. FinBERT/TabPFN historical pretraining overlap is not independently ruled out. No independent extraction quality approval is implied.','','TabSTAR, learned retrieval, new LLM calls, attention and finetuning remain deferred later-stage proposals. Completing these finite main mechanisms does not mean every paper component has been reproduced.','','## Evidence','','See protocol.json, INPUT_AUDIT.json, *_VERIFICATION.json, training_evidence.json, monthly_metrics.csv, direction_changes.csv, paired_intervals.csv and selected_parameters.csv. All original text/cache/weights remain private.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'CASE_NOTES.md').write_text('# Fixed diagnostic cases\n\nHash-first two examples per outcome category on seed573. Raw sources remain private. Prediction flips alone do not establish linguistic or market causation.\n\n'+pd.DataFrame(cases).to_markdown(index=False)+'\n');print(summary.to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(2):main()
