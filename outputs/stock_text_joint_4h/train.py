"""One finite nested run; never reads outer scores for decisions."""
from core import *
from threadpoolctl import threadpool_limits
import time,warnings,fcntl

def main():
 protect_check();d,*_=load();inp=joblib.load(WORK/'inputs.joblib');docs=inp['docs'];emb,ids,means,gate=semantic(d)
 outer=read_csv(PRIVATE/'outer_folds.csv');splits_all=json.loads((PRIVATE/'inner_folds.json').read_text())
 sealfile=WORK/'training_seal.json'
 actual={'protocol':sha(PUBLIC/'protocol.json'),'inputs':sha(WORK/'inputs.joblib'),'protected':sha(WORK/'protected.json'),'code':{p.name:sha(p) for p in [Path(__file__).with_name('core.py'),Path(__file__)]}}
 if sealfile.exists():cache_check(json.loads(sealfile.read_text()))
 else:dump(sealfile,actual)
 started=time.monotonic()
 for seed in SEEDS:
  for stock in ['AAPL','AMZN']:
   for fold in range(10):
    name=f'{seed}_{stock}_{fold}';root=WORK/'training'/name;base=PRIVATE/'baseline'/name
    if not block_guard(root,sha(sealfile)):continue
    ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy(int)
    tr=np.array(sorted(set(np.flatnonzero(d.symbol==stock))-set(ev)))
    splits=[s for s in splits_all if (s['seed'],s['symbol'],s['fold'])==(seed,stock,fold)]
    old=read_csv(base/'outer_predictions_SEALED.csv');assert old.row_id.tolist()==d.iloc[ev].row_id.tolist()
    base_oof=read_csv(base/'FULL_selection_oof_NOT_META_TRAIN.csv').set_index('index').p
    price_oof=read_csv(base/'PRICE_selection_oof_NOT_META_TRAIN.csv').set_index('index').p
    records=[];innerpred=[];ledger=[];out=old.copy();out['seed']=seed;out['fold']=fold;out['gate']=gate[ev]
    def fit_save(method,c,rho,m,a,b,x,y,tf=None,inf=None):
     t=time.monotonic();m.fit(x,d.iloc[a].label);p=m.predict_proba(y)[:,1]
     fname=f'{method}_{"selected" if inf is None else "inner"+str(inf)}_C{c}_rho{rho}.joblib'
     bundle=dict(model=m,transform=tf,train=list(map(int,a)),evaluation=list(map(int,b)),method=method,C=c,rho=rho,inner_fold=inf)
     joblib.dump(bundle,root/fname,compress=3)
     ledger.append(dict(file=fname,method=method,C=c,rho=rho,inner_fold=inf,train=list(map(int,a)),evaluation=list(map(int,b)),seconds=time.monotonic()-t,classifier_fits=3 if method=='A3' else 1,calibrators=3 if method=='A3' else 0))
     return p,fname
    def record(method,c,rho,inf,b,p):
     records.append(dict(method=method,C=c,rho=rho,inner_fold=inf,**metric(d.iloc[b].label,p)))
     innerpred.extend(dict(method=method,C=c,rho=rho,inner_fold=inf,index=int(i),p=float(v)) for i,v in zip(b,p))
    for method in ['A1','A2','A3']:
     for s in splits:
      a=s['train'];b=s['validation'];assert not(set(a)&set(ev)) and not(set(a)&set(b))
      for c in CS:
       p,_=fit_save(method,c,0.,text_model(method,c,seed),a,b,[docs[i] for i in a],[docs[i] for i in b],inf=s['inner_fold'])
       p[d.iloc[b].has_news.to_numpy()==0]=price_oof.loc[b].to_numpy()[d.iloc[b].has_news.to_numpy()==0]
       record(method,c,0.,s['inner_fold'],b,p)
     c,rho=choose(records,method);p,f=fit_save(method,c,rho,text_model(method,c,seed),tr,ev,[docs[i] for i in tr],[docs[i] for i in ev]);p[d.iloc[ev].has_news.to_numpy()==0]=old.PRICE.to_numpy()[d.iloc[ev].has_news.to_numpy()==0];out[method]=p
    # rho=0 reuses each original FULL candidate exactly. No duplicate fitting.
    oldC=json.loads((base/'training.json').read_text())['selected_C']['FULL']
    for method in ['B1','B2']:
     for s in splits:
      a=s['train'];b=s['validation'];inf=s['inner_fold'];tf=Joint().fit(d,a,emb,ids,means,method)
      for c in CS:
       saved=joblib.load(base/f'FULL_inner{inf}_{c}.joblib');assert saved['train_indices']==a and saved['evaluation_indices']==b
       p0=saved['model'].predict_proba(saved['transform'].transform(d,b,{}))[:,1];nn=d.iloc[b].has_news.to_numpy()==0;p0[nn]=price_oof.loc[b].to_numpy()[nn]
       record(method,c,0.,inf,b,p0)
       for rho in [.1,1.]:
        p,_=fit_save(method,c,rho,classifier('FULL',c),a,b,tf.transform(d,a,means,gate,rho),tf.transform(d,b,means,gate,rho),tf,inf)
        p[~gate[b]]=base_oof.loc[b].to_numpy()[~gate[b]];record(method,c,rho,inf,b,p)
     c,rho=choose(records,method)
     if rho==0:
      assert c==oldC;out[method]=old.FULL.to_numpy();dump(root/(method+'_reuse.json'),dict(method=method,C=c,rho=0.,source=str((base/'FULL_selected.joblib').relative_to(ROOT)),sha256=sha(base/'FULL_selected.joblib')))
     else:
      tf=Joint().fit(d,tr,emb,ids,means,method);p,f=fit_save(method,c,rho,classifier('FULL',c),tr,ev,tf.transform(d,tr,means,gate,rho),tf.transform(d,ev,means,gate,rho),tf);p[~gate[ev]]=old.FULL.to_numpy()[~gate[ev]];out[method]=p
    oofids=base_oof.index.to_numpy(int);bp=base_oof.to_numpy();has=d.iloc[oofids].has_news.to_numpy().astype(bool)
    sr=[dict(a=a,Brier=float(np.mean((d.iloc[oofids].label.to_numpy()-shrink(bp,has,a))**2))) for a in [.25,.5,.75,1.]]
    a=min(sr,key=lambda r:(r['Brier'],-r['a']))['a'];out['FULL_SHRINK']=shrink(old.FULL.to_numpy(),d.iloc[ev].has_news.to_numpy().astype(bool),a)
    classical=read_csv(CLASSIC/name/'predictions.csv');assert classical.row_id.tolist()==out.row_id.tolist()
    for m in ['TFIDF_LR','TFIDF_SVM','TFIDF_RF']:
     out[m]=classical[m].to_numpy();out[m+'_PRICE_FALLBACK']=np.where(d.iloc[ev].has_news.to_numpy(),out[m],out.PRICE)
    out.to_csv(root/'predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(records).to_csv(root/'selection.csv',index=False,float_format='%.17g');pd.DataFrame(innerpred).to_csv(root/'inner_predictions.csv',index=False,float_format='%.17g')
    dump(root/'shrink.json',dict(candidates=sr,selected=a));dump(root/'training.json',ledger)
    dump(root/'complete.json',dict(seal=sha(sealfile),files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
    print(name,'complete; fits',sum(r['classifier_fits'] for r in ledger),'elapsed',round(time.monotonic()-started,1),flush=True)
 ledger=[r for p in (WORK/'training').glob('*/training.json') for r in json.loads(p.read_text())]
 dump(PUBLIC/'training_evidence.json',dict(status='COMPLETE_PENDING_NO_FIT_VERIFICATION',blocks=60,top_level_fit_calls=len(ledger),actual_classifier_fits=sum(r['classifier_fits'] for r in ledger),SVM_sigmoid_calibrators=sum(r['calibrators'] for r in ledger),device='CPU',threads=2,encoder_fits=0,seconds=sum(r['seconds'] for r in ledger),wall_seconds=time.monotonic()-started,LLM_continued=False))
if __name__=='__main__':
 with (WORK/'training.lock').open('a') as lock,threadpool_limits(2),warnings.catch_warnings():
  fcntl.flock(lock,fcntl.LOCK_EX);warnings.simplefilter('ignore',FutureWarning);main()
