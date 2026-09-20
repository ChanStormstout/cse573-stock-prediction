"""Fit-free saved-model replay and independent membership/selection checks."""
from engine import *
import argparse

def forbidden(*a,**k):raise RuntimeError('Verifier must never fit')
def vocabulary_check(v,docs):
 from collections import Counter
 df=Counter();tf=Counter();analyze=v.build_analyzer()
 for s in docs:
  ts=analyze(s);df.update(set(ts));tf.update(ts)
 terms=sorted(df);mask=np.array([df[t]>=3 for t in terms]);ix=np.flatnonzero(mask)
 if len(ix)>5000:ix=ix[np.argsort(-np.array([tf[t] for t in terms])[mask])[:5000]]
 expected=sorted(terms[j] for j in ix)
 assert v.get_feature_names_out().tolist()==expected
 idf=np.array([np.log((1+len(docs))/(1+df[t]))+1 for t in expected])
 np.testing.assert_allclose(v.idf_,idf,rtol=0,atol=1e-12)

def verify(stage,wait_pid=None):
 seal_check();d,*_=load();inp=joblib.load(WORK/'inputs.joblib');data=joblib.load(WORK/'chunks.joblib') if stage=='aggregation' else None
 blocks=inp['grouped'] if stage=='grouped' else inp['random'];allout=[];err=0.;count=0;cal=0;selected=[];reps_cache={}
 # Disable all fitting paths used by the runner after read-only setup.
 for cls in [LogisticRegression,LinearSVC,CalibratedClassifierCV,TfidfVectorizer,Tracked,Scale,SetTransform]:
  if hasattr(cls,'fit'):cls.fit=forbidden
  if hasattr(cls,'fit_transform'):cls.fit_transform=forbidden
 for b in blocks:
  root=WORK/stage/b['name']
  while wait_pid and not (root/'complete.json').exists():
   import os
   os.kill(wait_pid,0);time.sleep(3)
  complete=json.loads((root/'complete.json').read_text());assert complete['seal']==sha(WORK/'seal.json')
  for f,h in complete['files'].items():assert sha(root/f)==h,'artifact mismatch '+f
  out=read(root/'predictions.csv');ip=read(root/'inner_predictions.csv');scores=read(root/'selection.csv');ev=np.array(b['test']);tr=np.array(b['train'])
  assert out.row_id.tolist()==d.iloc[ev].row_id.tolist();assert out.label.tolist()==d.iloc[ev].label.tolist()
  if stage=='preprocess':assert np.array_equal(out.CONTROL.to_numpy(),np.where(d.iloc[ev].has_news,read(CLASSIC/b['name']/'predictions.csv').TFIDF_SVM,out.PRICE))
  base=PRIVATE/('diagnostic_baseline' if stage=='grouped' else 'baseline')/b['name']
  pp=None if stage=='grouped' else read(base/'PRICE_selection_oof_NOT_META_TRAIN.csv').set_index('index').p
  ledger=json.loads((root/'training.json').read_text());reps_cache={}
  for rec in ledger:
   obj=joblib.load(root/rec['file']);a=np.array(obj['train']);e=np.array(obj['evaluation']);method=obj['method'];inf=obj['inner_fold'];model=obj['model']
   expected=(b['train'],b['test']) if inf is None else b['inner'][inf];assert a.tolist()==list(expected[0]) and e.tolist()==list(expected[1]);assert not set(a)&set(e)
   if inf is not None:assert not (set(a)|set(e))&set(ev)
   if stage!='aggregation':
    docs=np.array(d.stem_body.tolist() if stage=='grouped' else inp['documents'][method],dtype=object)
    cv=obj['cv'];assert len(model.calibrated_classifiers_)==3
    for (xx,yy),cl in zip(cv,model.calibrated_classifiers_):
     aa=a[xx];bb=a[yy];assert not set(aa)&set(bb);assert sorted(xx+yy)==list(range(len(a)))
     assert cl.estimator.named_steps['text'].document_hashes_==[digest(s) for s in docs[aa]]
     if stage=='grouped':assert not set(inp['groups'][aa])&set(inp['groups'][bb]);assert not set(inp['groups'][a])&set(inp['groups'][e])
     if inf is None:vocabulary_check(cl.estimator.named_steps['text'],docs[aa].tolist())
     cal+=1
    p=model.predict_proba(docs[e].tolist())[:,1]
   else:
    tf=obj['transform'];assert tf.train==a.tolist()
    ids=sorted({int(c) for i in a for c in data['window_chunks'][i]});assert ids==tf.chunk_ids
    key=tuple(a)
    if key not in reps_cache:
     np.testing.assert_allclose(tf.center,data['vectors'][ids].mean(0),rtol=0,atol=1e-12)
     std=data['vectors'][ids].std(0);std=np.where(std>1e-12,std,1);np.testing.assert_allclose(tf.std,std,rtol=0,atol=1e-12)
     vocabulary_check(tf.v,d.iloc[a].stem_body.tolist())
     # Independent mapping and weighted aggregation, no runner representations call.
     rng=np.random.default_rng(573);ix=rng.choice(ids,min(1024,len(ids)),replace=False);zz=(data['vectors'][ix]-tf.center)/std;j=rng.permutation(len(ix));dist=np.sum((zz-zz[j])**2,axis=1);sigma=np.sqrt(np.median(dist[dist>0]));assert abs(sigma-tf.sigma)<1e-12
     np.testing.assert_array_equal(tf.omega,rng.normal(size=(768,256))/sigma);np.testing.assert_array_equal(tf.phase,rng.uniform(0,2*np.pi,256))
     z=(data['vectors']-tf.center)/std;f=np.cos(z@tf.omega+tf.phase)*np.sqrt(2/256);means=[];sets=[]
     for ix,w in zip(data['window_chunks'],data['weights']):
      means.append(np.sum(z[ix]*w[:,None],axis=0) if len(ix) else np.zeros(768));sets.append(np.sum(f[ix]*w[:,None],axis=0) if len(ix) else np.zeros(256))
     means=np.array(means);post=np.cos(means@tf.omega+tf.phase)*np.sqrt(2/256);post[[len(x)==0 for x in data['window_chunks']]]=0
     reps_cache[key]={'MEAN_LINEAR':normalize(means),'MEAN_THEN_MAP':normalize(post),'MAP_THEN_MEAN':normalize(np.array(sets))}
    x=tf.base(d,e)
    if method!='SET_BASE':x=sparse.hstack([x,reps_cache[key][method][e]]).tocsr()
    p=model.predict_proba(x)[:,1]
   if inf is None:
    expectedp=out[method+'_DIRECT'].to_numpy();delta=float(np.max(abs(p-expectedp)));assert delta<1e-10;err=max(err,delta)
    final=np.where(d.iloc[e].has_news,p,out.PRICE);np.testing.assert_allclose(final,out[method],rtol=0,atol=1e-10)
    nn=d.iloc[e].has_news.to_numpy()==0;assert np.array_equal(out.loc[nn,method].to_numpy(),out.loc[nn,'PRICE'].to_numpy())
    rows=scores[scores.method==method].to_dict('records');chosen=min(CS,key=lambda c:(-np.mean([r['BA'] for r in rows if r['C']==c]),np.mean([r['Brier'] for r in rows if r['C']==c]),c));assert chosen==obj['C'];selected.append(dict(stage=stage,block=b['name'],stock=b['stock'],seed=b['seed'],method=method,C=chosen))
   else:
    if stage=='aggregation':p=np.where(d.iloc[e].has_news,p,pp.loc[e])
    expectedrows=ip[(ip.method==method)&(ip.C==obj['C'])&(ip.inner_fold==inf)];assert expectedrows['index'].tolist()==e.tolist()
    delta=float(np.max(abs(p-expectedrows.p.to_numpy())));assert delta<1e-10;err=max(err,delta)
    measured=metric(d.iloc[e].label,p);r=scores[(scores.method==method)&(scores.C==obj['C'])&(scores.inner_fold==inf)].iloc[0]
    for k in ['BA','Brier','MCC']:assert abs(measured[k]-r[k])<1e-10
   count+=1
  allout.append(out);print('Verified',stage,b['name'],flush=True)
 out=pd.concat(allout,ignore_index=True);assert len(out)==4821
 out.to_csv(PUB/(stage+'_predictions.csv'),index=False,float_format='%.17g');pd.DataFrame(selected).to_csv(PUB/(stage+'_selected_C.csv'),index=False)
 dump(PUB/(stage+'_VERIFICATION.json'),dict(status='PASS',blocks=len(blocks),rows=len(out),reloaded_models=count,max_probability_error=err,calibration_memberships_checked=cal,inner_metrics_recomputed=True,selection_reconstructed=True,exact_PRICE_fallback=True,fit_calls=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preprocess','grouped','aggregation']);ap.add_argument('--wait-pid',type=int);a=ap.parse_args()
 with threadpool_limits(2):verify(a.stage,a.wait_pid)
