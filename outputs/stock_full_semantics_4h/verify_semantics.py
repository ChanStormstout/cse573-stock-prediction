"""Read-only source, selection and exact probability/fallback replay."""
from train_semantics import *
def verify_embedding_cache(name):
 articles=[json.loads(s) for s in (SRC/'articles.jsonl').read_text().splitlines()];chunks=[json.loads(s) for s in (SRC/'chunks.jsonl').read_text().splitlines()];values={};counts={}
 for path in sorted((SRC/name).glob('batch_*.npz')):
  meta=json.loads(path.with_suffix('.json').read_text());assert meta['sha256']==sha(path);assert meta['seal_sha256']==sha(SRC/name/'seal.json');z=np.load(path)
  for k,v,n in zip(z['keys'],z['vectors'],z['counts']):
   assert k not in values;values[k]=v;counts[k]=n
 assert set(values)=={r['chunk_id'] for r in chunks}
 z=dict(np.load(SRC/(name+'_articles.npz')));assert z['keys'].tolist()==[a['article_id'] for a in articles]
 max_error=0.
 for i,a in enumerate(articles):
  expected=sum((values[k].astype(float)*counts[k] for k in a['chunks']),np.zeros(z['vectors'].shape[1]))/sum(counts[k] for k in a['chunks']) if a['chunks'] else np.zeros(z['vectors'].shape[1])
  max_error=max(max_error,float(np.max(abs(expected-z['vectors'][i]))));assert bool(z['has_evidence'][i])==bool(a['chunks'])
 assert max_error<1e-12
 return max_error

def main():
 check_sources();d,*_=load();outer=pd.read_csv(PRIVATE/'outer_folds.csv');frames=[];selections=[];error=0.;fallback=0;models=0;fit_count=0
 embedding_errors={n:verify_embedding_cache(n) for n in ['FINBERT','MODERN']}
 dest=NEW/'semantic_training';seal=dest/'seal.json';stamp=json.loads(seal.read_text());assert stamp['code']==sha(Path(__file__).with_name('train_semantics.py'));assert stamp['protocol']==sha(PUBLIC/'SEMANTIC_PROTOCOL.json')
 for n,h in stamp['base_hashes'].items():assert sha(PRIVATE/n)==h
 for name in ['FINBERT','MODERN']:
  assert stamp['encoder_seals'][name]==sha(SRC/name/'seal.json');assert json.loads((PUBLIC/(name+'_ENCODING.json')).read_text())['output_sha256']==sha(SRC/(name+'_articles.npz'));emb,ids,means=semantic_input(d,name)
  for root in sorted(dest.glob(name+'_*')):
   complete=json.loads((root/'complete.json').read_text());assert complete['seal']==sha(seal)
   for n,h in complete['files'].items():assert sha(root/n)==h
   bundle=joblib.load(root/'selected.joblib');p=pd.read_csv(root/'predictions.csv',float_precision='round_trip');cv=pd.read_csv(root/'selection.csv',float_precision='round_trip');ev=bundle['evaluation'];tr=bundle['train'];assert not(set(tr)&set(ev));assert p.row_id.tolist()==d.iloc[ev].row_id.tolist();assert p.label.tolist()==d.iloc[ev].label.tolist()
   seed=int(p.seed.iloc[0]);stock=p.symbol.iloc[0];fold=int(p.fold.iloc[0]);assert p.seed.nunique()==p.symbol.nunique()==p.fold.nunique()==1
   expected_ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_list();assert ev==expected_ev;assert tr==sorted(set(np.flatnonzero(d.symbol==stock))-set(expected_ev))
   base=pd.read_csv(PRIVATE/'baseline'/f'{seed}_{stock}_{fold}'/'outer_predictions_SEALED.csv',float_precision='round_trip');assert p.row_id.tolist()==base.row_id.tolist();assert np.array_equal(p.FULL,base.FULL)
   assert bundle['model'].classes_.tolist()==[0,1];assert bundle['model'].n_iter_.max()<bundle['model'].max_iter
   fit_count+=len(cv[['C','inner_fold']].drop_duplicates())+1
   chosen=min([(c,w) for c in CS for w in WEIGHTS],key=lambda cw:(-cv[(cv.C==cw[0])&(cv.weight==cw[1])].BA.mean(),cv[(cv.C==cw[0])&(cv.weight==cw[1])].Brier.mean(),cw[1],cw[0]))
   assert chosen==(bundle['C'],bundle['weight']);tf=bundle['transform'];assert tf['article_ids']==sorted({j for i in tr for j in ids[i]})
   x=np.c_[d[OLD],tf['pca'].transform(means),np.log1p(d.news_count),d.has_news];x=tf['scale'].transform(x)
   from scipy.special import expit
   prob=expit(x[ev]@bundle['model'].coef_[0]+bundle['model'].intercept_[0]);gate=np.array([bool(ids[i]) for i in ev]);assert np.array_equal(gate,p.gate)
   expected=p.FULL.to_numpy().copy();w=bundle['weight']
   if w:expected[gate]=(1-w)*expected[gate]+w*prob[gate]
   error=max(error,float(np.max(abs(prob-p.semantic_p))),float(np.max(abs(expected-p.p))));fallback+=int(np.count_nonzero(p.p.to_numpy()[~gate]!=p.FULL.to_numpy()[~gate]));assert error<1e-12 and fallback==0
   selections.append(dict(encoder=name,symbol=p.symbol.iloc[0],seed=int(p.seed.iloc[0]),fold=int(p.fold.iloc[0]),C=bundle['C'],weight=w,train_n=len(tr),train_evidence_n=sum(bool(ids[i]) for i in tr)))
   frames.append(p);models+=1
 assert models==120;allp=pd.concat(frames,ignore_index=True);assert len(allp)==len(d)*6 and not allp[['encoder','seed','row_id']].duplicated().any();metrics=[]
 for (name,stock,seed),g in allp.groupby(['encoder','symbol','seed']):
  y=g.label.to_numpy();base=g.FULL.to_numpy()>=.5;q=g.p.to_numpy()>=.5
  metrics.append(dict(encoder=name,symbol=stock,seed=seed,**metric(y,g.p),base_BA=metric(y,g.FULL)['BA'],base_Brier=metric(y,g.FULL)['Brier'],coverage=float(g.gate.mean()),changed=int(sum(q!=base)),repaired=int(sum((base!=y)&(q==y))),introduced=int(sum((base==y)&(q!=y)))))
 pd.DataFrame(selections).to_csv(PUBLIC/'SEMANTIC_SELECTIONS.csv',index=False);pd.DataFrame(metrics).to_csv(PUBLIC/'SEMANTIC_METRICS.csv',index=False);allp.to_csv(PUBLIC/'SEMANTIC_PREDICTIONS.csv',index=False,float_format='%.17g')
 dump(PUBLIC/'SEMANTIC_VERIFICATION.json',dict(status='PASS',fit_calls=0,embedding_aggregation_errors=embedding_errors,models=models,actual_fit_count_reconstructed=fit_count,rows=len(allp),max_probability_error=error,no_evidence_fallback_mismatches=fallback,checks=['source hashes','code and protocol seal','model/artifact hashes','labels and row keys','canonical outer-fold membership','canonical FULL probability parity','disjoint training','training-only PCA membership','inner C and weight reconstruction','direct sigmoid and fusion replay','exact no-evidence fallback']))
 print('SEMANTIC PASS',models,error)
if __name__=='__main__':main()
