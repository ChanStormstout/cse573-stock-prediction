"""Finite completion. Each completed block sealed; partial blocks stop on restart."""
import models as m
from models import *
import argparse,fcntl

def seal_check():
 for p,h in json.loads((LOCAL/'seal.json').read_text())['files'].items():assert sha(ROOT/p)==h,p

def split(a,seed):
 d=initialize()['d'];a=np.asarray(a);return [(a[x].tolist(),a[y].tolist()) for x,y in StratifiedKFold(3,shuffle=True,random_state=seed).split(a,d.iloc[a].label)]
def choose(rows):
 keys=sorted({(r['C'],r.get('gamma',1.)) for r in rows})
 return min(keys,key=lambda k:(-np.mean([r['BA'] for r in rows if (r['C'],r.get('gamma',1.))==k]),np.mean([r['Brier'] for r in rows if (r['C'],r.get('gamma',1.))==k]),k))
def predict_saved(obj,ii):
 db=initialize();typ=obj['type']
 if typ=='svm':return obj['model'].predict_proba(np.asarray(ii).reshape(-1,1))[:,1]
 if typ=='table':return obj['model'].predict_proba(obj['transform'].transform(ii,obj['semantic']))[:,1]
 if typ=='base':
  p=obj['FULL']['model'].predict_proba(obj['FULL']['transform'].transform(db['d'],ii,db['means']))[:,1];q=obj['PRICE']['model'].predict_proba(obj['PRICE']['transform'].transform(db['d'],ii,db['means']))[:,1];return np.where(db['d'].iloc[ii].has_news,p,q)
 if typ=='residual':
  p=predict_saved(obj['base'],ii)
  return p if obj['C']==0 else obj['model'].predict(obj['transform'].transform(ii),p,db['gate'][ii])
 raise ValueError(typ)
def save(root,name,obj,ii):
 obj['evaluation']=list(map(int,ii));p=predict_saved(obj,ii);joblib.dump(obj,root/(name+'.joblib'),compress=3);return p

def run_svm(stage,b,root,out):
 db=initialize();d=db['d'];a=b['train'];e=b['test'];methods=['UNI','BIGRAM','NB_BIGRAM'] if stage=='lexical' else ['K_WP','K_MEAN','K_AFTER','K_SET','K_INTERACTION'];rs=[];ip=[];cache={}
 for inf,(aa,bb) in enumerate(b['inner']):
  for name in methods:
   for c in CS:
    for gamma in ([.1,1.] if name=='K_INTERACTION' else [1.]):
     mod=timed_fit(svm_model(name,c,b['seed'],previous.calibration(aa,d,b['seed']),gamma),np.array(aa)[:,None],d.iloc[aa].label,f'{name}/inner')
     p=mod.predict_proba(np.array(bb)[:,None])[:,1]
     if stage=='kernel' and name!='K_WP':p=np.where(db['gate'][bb],p,cache[(inf,c,1.)])
     if name=='K_WP':cache[(inf,c,gamma)]=p.copy()
     # No-news replacement during kernel selection uses a separately trained price model on aa.
     if stage=='kernel':
      if ('price',inf) not in cache:cache['price',inf]=base_select(aa,split(aa,b['seed']),b['seed'],only_price=True)[0]
      pr=cache['price',inf];q=pr['model'].predict_proba(pr['transform'].transform(d,bb,db['means']))[:,1];p=np.where(d.iloc[bb].has_news,p,q)
     rs.append(dict(method=name,C=c,gamma=gamma,inner_fold=inf,**metric(d.iloc[bb].label,p)))
     ip.extend(dict(method=name,C=c,gamma=gamma,inner_fold=inf,index=int(i),p=float(v)) for i,v in zip(bb,p))
 for name in methods:
  c,gamma=choose([r for r in rs if r['method']==name]);mod=timed_fit(svm_model(name,c,b['seed'],previous.calibration(a,d,b['seed']),gamma),np.array(a)[:,None],d.iloc[a].label,f'{name}/selected');obj=dict(type='svm',model=mod,train=a,C=c,gamma=gamma,method=name)
  p=save(root,name,obj,e);out[name+'_DIRECT']=p
  if stage=='kernel' and name!='K_WP':p=np.where(db['gate'][e],p,out.K_WP)
  out[name]=np.where(d.iloc[e].has_news,p,out.PRICE)
 return rs,ip

def base_select(a,folds,seed,only_price=False):
 db=initialize();d=db['d'];bundle={};records=[];price_cv={};selected_price={}
 for name in (['PRICE'] if only_price else ['PRICE','FULL']):
  rr=[]
  for inf,(aa,bb) in enumerate(folds):
   tf=Features(name).fit(d,aa,db['emb'],db['ids'],db['means']);x=tf.transform(d,aa,db['means']);v=tf.transform(d,bb,db['means'])
   for c in CS:
    model=timed_fit(classifier(name,c),x,d.iloc[aa].label,'nested_base/'+name);p=model.predict_proba(v)[:,1]
    if name=='PRICE':price_cv[inf,c]=p.copy()
    else:p=np.where(d.iloc[bb].has_news,p,selected_price[inf])
    rr.append(dict(method=name,C=c,inner_fold=inf,**metric(d.iloc[bb].label,p)))
  c,_=choose(rr)
  if name=='PRICE':selected_price={inf:price_cv[inf,c] for inf in range(len(folds))}
  tf=Features(name).fit(d,a,db['emb'],db['ids'],db['means']);model=timed_fit(classifier(name,c),tf.transform(d,a,db['means']),d.iloc[a].label,'nested_base/'+name+'/selected');bundle[name]=dict(model=model,transform=tf,C=c,train=list(a));records.extend(rr)
 return (bundle['PRICE'] if only_price else dict(type='base',**bundle,train=list(a))),records

def run_table(b,root,out):
 db=initialize();d=db['d'];a=b['train'];e=b['test'];rs=[];ip=[]
 for inf,(aa,bb) in enumerate(b['inner']):
  tf=Table().fit(aa,d.iloc[aa].label)
  for sem in [False,True]:
   name='TABLE_LR_'+('SEM' if sem else 'NOSEM');x=tf.transform(aa,sem);v=tf.transform(bb,sem)
   for c in CS:
    model=timed_fit(classifier('FULL',c),x,d.iloc[aa].label,name+'/inner');p=model.predict_proba(v)[:,1]
    rs.append(dict(method=name,C=c,inner_fold=inf,**metric(d.iloc[bb].label,p)));ip.extend(dict(method=name,C=c,inner_fold=inf,index=int(i),p=float(q)) for i,q in zip(bb,p))
 tf=Table().fit(a,d.iloc[a].label)
 for learner in ['LR','RF','TABPFN']:
  for sem in [False,True]:
   name='TABLE_'+learner+'_'+('SEM' if sem else 'NOSEM');c=choose([r for r in rs if r['method']==name])[0] if learner=='LR' else None
   model=classifier('FULL',c) if learner=='LR' else (RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=10,random_state=b['seed'],n_jobs=1) if learner=='RF' else TabPFNRemote(root/(name+'_weights.bin'),b['seed']))
   model=timed_fit(model,tf.transform(a,sem),d.iloc[a].label,name+'/selected');obj=dict(type='table',model=model,transform=tf,semantic=sem,C=c,train=a,method=name)
   p=save(root,name,obj,e);out[name+'_DIRECT']=p
   if learner=='TABPFN':TIMES.append(dict(tag=name+'/conditioning_and_inference',seconds=model.last_execution['seconds'],n=len(a),actual_prior_fit=True))
   if sem:p=np.where(db['gate'][e],p,out['TABLE_'+learner+'_NOSEM'])
   out[name]=np.where(d.iloc[e].has_news,p,out.PRICE)
 return rs,ip

def residual_scope(a,evals,folds,seed,root,tag):
 db=initialize();d=db['d'];a=np.array(a);oof=np.zeros(len(a));loc={int(i):j for j,i in enumerate(a)};audit=[]
 for k,(aa,bb) in enumerate(folds):
  sub=split(aa,seed);obj,rs=base_select(aa,sub,seed);p=save(root,tag+f'_cross{k}',obj,bb);oof[[loc[i] for i in bb]]=p
  audit.append(dict(train=aa,evaluation=bb,selection_folds=sub,selected_C={n:obj[n]['C'] for n in ['FULL','PRICE']}))
 base,br=base_select(a.tolist(),folds,seed);save(root,tag+'_base',base,evals)
 tf=Semantic16().fit(a);z=tf.transform(a);vals={}
 for c in CS:
  t=time.monotonic();model=Offset().fit(z,d.iloc[a].label,oof,c);TIMES.append(dict(tag='residual/'+tag,seconds=time.monotonic()-t,n=len(a)))
  obj=dict(type='residual',C=c,model=model,transform=tf,base=base,train=a.tolist());vals[c]=(obj,predict_saved(obj,evals))
 vals[0]=(dict(type='residual',C=0,base=base,train=a.tolist()),predict_saved(base,evals))
 dump(root/(tag+'_crossfit.json'),dict(outer_training=a.tolist(),crossfits=audit,base_selection_folds=folds,base_records=br,OOF=[dict(index=int(i),p=float(q)) for i,q in zip(a,oof)]))
 return vals

def run_residual(b,root,out):
 db=initialize();d=db['d'];rs=[];ip=[]
 for inf,(aa,bb) in enumerate(b['inner']):
  vals=residual_scope(aa,bb,split(aa,b['seed']),b['seed'],root,f'inner{inf}')
  for c,(obj,p) in vals.items():
   joblib.dump(obj,root/f'inner{inf}_{c}.joblib',compress=3);rs.append(dict(method='R16',C=c,inner_fold=inf,**metric(d.iloc[bb].label,p)));ip.extend(dict(method='R16',C=c,inner_fold=inf,index=int(i),p=float(q)) for i,q in zip(bb,p))
 c,_=choose(rs);vals=residual_scope(b['train'],b['test'],b['inner'],b['seed'],root,'outer');obj,p=vals[c];out['RBASE']=save(root,'RBASE',obj['base'],b['test']);out['R16']=save(root,'R16',obj,b['test']);return rs,ip

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['lexical','kernel','table','residual']);ap.add_argument('--limit',type=int);args=ap.parse_args();seal_check();db=initialize();d=db['d'];blocks=db['inp']['blocks']
 with (LOCAL/'run.lock').open('a') as lock,threadpool_limits(2):
  fcntl.flock(lock,fcntl.LOCK_EX)
  for b in blocks[:args.limit]:
   root=LOCAL/args.stage/b['name']
   if (root/'complete.json').exists():
    saved=json.loads((root/'complete.json').read_text());assert saved['seal']==sha(LOCAL/'seal.json')
    for p,h in saved['files'].items():assert sha(root/p)==h
    continue
   assert not root.exists(),'Incomplete block preserved: '+str(root);root.mkdir(parents=True);TIMES.clear();m.GEOMETRY.clear();m.MATRICES.clear()
   out=read(PRIVATE/'baseline'/b['name']/'outer_predictions_SEALED.csv');assert out.row_id.tolist()==d.iloc[b['test']].row_id.tolist();out['seed']=b['seed'];out['block']=b['name']
   if args.stage in ['lexical','kernel']:rs,ip=run_svm(args.stage,b,root,out)
   elif args.stage=='table':rs,ip=run_table(b,root,out)
   else:rs,ip=run_residual(b,root,out)
   out.to_csv(root/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(rs).to_csv(root/'selection.csv',index=False,float_format='%.17g');pd.DataFrame(ip).to_csv(root/'inner_predictions.csv',index=False,float_format='%.17g');dump(root/'training.json',TIMES)
   dump(root/'complete.json',dict(seal=sha(LOCAL/'seal.json'),files={p.name:sha(p) for p in root.iterdir() if p.is_file()}));print(args.stage,b['name'],'complete',len(TIMES),'fits',flush=True)
if __name__=='__main__':main()
