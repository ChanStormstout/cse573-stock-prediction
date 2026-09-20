"""Read-only source, selection and exact probability/fallback replay."""
from train_semantics import *
def main():
 check_sources();d,*_=load();frames=[];error=0.;fallback=0;models=0
 dest=NEW/'semantic_training';seal=dest/'seal.json';stamp=json.loads(seal.read_text());assert stamp['code']==sha(Path(__file__).with_name('train_semantics.py'));assert stamp['protocol']==sha(PUBLIC/'SEMANTIC_PROTOCOL.json')
 for name in ['FINBERT','MODERN']:
  assert stamp['inputs'][name]==sha(SRC/(name+'_articles.npz'));emb,ids,means=semantic_input(d,name)
  for root in sorted(dest.glob(name+'_*')):
   complete=json.loads((root/'complete.json').read_text());assert complete['seal']==sha(seal)
   for n,h in complete['files'].items():assert sha(root/n)==h
   bundle=joblib.load(root/'selected.joblib');p=pd.read_csv(root/'predictions.csv',float_precision='round_trip');cv=pd.read_csv(root/'selection.csv',float_precision='round_trip');ev=bundle['evaluation'];tr=bundle['train'];assert not(set(tr)&set(ev));assert p.row_id.tolist()==d.iloc[ev].row_id.tolist();assert p.label.tolist()==d.iloc[ev].label.tolist()
   chosen=min([(c,w) for c in CS for w in WEIGHTS],key=lambda cw:(-cv[(cv.C==cw[0])&(cv.weight==cw[1])].BA.mean(),cv[(cv.C==cw[0])&(cv.weight==cw[1])].Brier.mean(),cw[1],cw[0]))
   assert chosen==(bundle['C'],bundle['weight']);tf=bundle['transform'];assert tf['article_ids']==sorted({j for i in tr for j in ids[i]})
   x=np.c_[d[OLD],tf['pca'].transform(means),np.log1p(d.news_count),d.has_news];x=tf['scale'].transform(x)
   from scipy.special import expit
   prob=expit(x[ev]@bundle['model'].coef_[0]+bundle['model'].intercept_[0]);gate=np.array([bool(ids[i]) for i in ev]);assert np.array_equal(gate,p.gate)
   expected=p.FULL.to_numpy().copy();w=bundle['weight']
   if w:expected[gate]=(1-w)*expected[gate]+w*prob[gate]
   error=max(error,float(np.max(abs(prob-p.semantic_p))),float(np.max(abs(expected-p.p))));fallback+=int(np.count_nonzero(p.p.to_numpy()[~gate]!=p.FULL.to_numpy()[~gate]));assert error<1e-12 and fallback==0
   frames.append(p);models+=1
 assert models==120;allp=pd.concat(frames,ignore_index=True);metrics=[]
 for (name,stock,seed),g in allp.groupby(['encoder','symbol','seed']):
  y=g.label.to_numpy();base=g.FULL.to_numpy()>=.5;q=g.p.to_numpy()>=.5
  metrics.append(dict(encoder=name,symbol=stock,seed=seed,**metric(y,g.p),base_BA=metric(y,g.FULL)['BA'],base_Brier=metric(y,g.FULL)['Brier'],coverage=float(g.gate.mean()),changed=int(sum(q!=base)),repaired=int(sum((base!=y)&(q==y))),introduced=int(sum((base==y)&(q!=y)))))
 pd.DataFrame(metrics).to_csv(PUBLIC/'SEMANTIC_METRICS.csv',index=False);allp.to_csv(PUBLIC/'SEMANTIC_PREDICTIONS.csv',index=False,float_format='%.17g')
 dump(PUBLIC/'SEMANTIC_VERIFICATION.json',dict(status='PASS',fit_calls=0,models=models,rows=len(allp),max_probability_error=error,no_evidence_fallback_mismatches=fallback,checks=['source hashes','code and protocol seal','model/artifact hashes','labels and row keys','disjoint training','training-only PCA membership','inner C and weight reconstruction','direct sigmoid and fusion replay','exact no-evidence fallback']))
 print('SEMANTIC PASS',models,error)
if __name__=='__main__':main()
