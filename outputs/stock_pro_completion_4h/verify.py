"""Read-only fitted-model replay; this verifier never calls fit."""
from models import *
import argparse

def replay(obj,ii):
 db=initialize();d=db['d']
 if obj['type']=='svm':return obj['model'].predict_proba(np.array(ii)[:,None])[:,1]
 if obj['type']=='table':return obj['model'].predict_proba(obj['transform'].transform(ii,obj['semantic']))[:,1]
 if obj['type']=='base':
  vals={k:v['model'].predict_proba(v['transform'].transform(d,ii,db['means']))[:,1] for k,v in obj.items() if k in ['FULL','PRICE']}
  return np.where(d.iloc[ii].has_news,vals['FULL'],vals['PRICE'])
 if obj['type']=='residual':
  p=replay(obj['base'],ii)
  if obj['C']==0:return p
  q=p.copy();g=db['gate'][ii];z=obj['transform'].transform(ii);q[g]=expit(logit(np.clip(p[g],1e-6,1-1e-6))+z[g]@obj['model'].coef);return q
 raise ValueError(obj['type'])
def verify_membership(obj,a,e):
 assert set(obj['train'])==set(a) and not set(a)&set(e)
 db=initialize()
 if obj['type']=='svm':
  for f in obj['model'].calibrated_classifiers_:
   t=f.estimator.named_steps['features'];assert set(t.train)<=set(a) and not set(t.train)&set(e)
   if isinstance(t,KernelFeatures):assert set(t.tf.chunk_ids)=={int(j) for i in t.train for j in db['chunks']['window_chunks'][i]}
 elif obj['type'] in ['table','residual'] and 'transform' in obj:
  t=obj['transform'];assert set(t.train)==set(a);assert set(t.article_ids)=={j for i in a for j in db['aid'][i]}
 if obj['type']=='base':
  for name in ['FULL','PRICE']:assert set(obj[name]['transform'].fit_ids)==set(db['d'].iloc[a].row_id)
 if obj['type']=='residual':verify_membership(obj['base'],a,e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['lexical','kernel','table','residual']);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 for p,h in json.loads((LOCAL/'seal.json').read_text())['files'].items():assert sha(ROOT/p)==h,p
 db=initialize();d=db['d'];count=0;maxerr=0;rows=[];fitcount=0;seconds=0;cal=0
 with threadpool_limits(2):
  for b in db['inp']['blocks']:
   root=LOCAL/args.stage/b['name']
   if not (root/'complete.json').exists():assert args.partial,root;continue
   sealed=json.loads((root/'complete.json').read_text());assert sealed['seal']==sha(LOCAL/'seal.json')
   for p,h in sealed['files'].items():assert sha(root/p)==h,p
   out=read(root/'predictions.csv');rs=read(root/'selection.csv');e=b['test'];a=b['train'];assert out.row_id.tolist()==d.iloc[e].row_id.tolist();assert np.array_equal(out.label,d.iloc[e].label)
   old=read(PRIVATE/'baseline'/b['name']/'outer_predictions_SEALED.csv')
   for name in ['PRICE','FULL','FINBERT','MODERN']:assert np.array_equal(out[name],old[name])
   methods=['UNI','BIGRAM','NB_BIGRAM'] if args.stage=='lexical' else (['K_WP','K_MEAN','K_AFTER','K_SET','K_INTERACTION'] if args.stage=='kernel' else (['TABLE_'+l+'_'+s for l in ['LR','RF','TABPFN'] for s in ['NOSEM','SEM']] if args.stage=='table' else ['RBASE','R16']))
   for name in methods:
    obj=joblib.load(root/(name+'.joblib'));verify_membership(obj,a,e);p=replay(obj,e);saved=out[name+'_DIRECT'] if name+'_DIRECT' in out else out[name];err=float(np.max(abs(p-saved)));maxerr=max(maxerr,err);assert err<= (1e-5 if 'TABPFN' in name else 1e-10),(name,err);count+=1
    if name not in ['RBASE','R16']:
     if name.startswith('K_') and name!='K_WP':p=np.where(db['gate'][e],p,out.K_WP)
     if name.startswith('TABLE_') and name.endswith('_SEM'):p=np.where(db['gate'][e],p,out[name[:-3]+'NOSEM'])
     p=np.where(d.iloc[e].has_news,p,out.PRICE)
    assert np.max(abs(p-out[name]))<= (1e-5 if 'TABPFN' in name else 1e-10)
    if name=='R16':assert np.array_equal(out.R16[~db['gate'][e]],out.RBASE[~db['gate'][e]])
    r=rs[rs.method==name]
    if len(r):
     cols=['C']+(['gamma'] if 'gamma' in r else []);agg=r.groupby(cols,dropna=False)[['BA','Brier']].mean().reset_index().sort_values(['BA','Brier']+cols,ascending=[False,True]+[True]*len(cols));win=agg.iloc[0]
     assert obj['C']==win.C
     if 'gamma' in cols:assert obj['gamma']==win.gamma
    if obj['type']=='svm':cal+=len(obj['model'].calibrated_classifiers_)
   if args.stage=='residual':
    assert np.max(abs(out.RBASE-out.FULL))<1e-10,'canonical base parity'
    for path in root.glob('*_crossfit.json'):
     audit=json.loads(path.read_text());scope=set(audit['outer_training']);seen=[]
     for k,r in enumerate(audit['crossfits']):
      aa=r['train'];bb=r['evaluation'];assert set(aa)|set(bb)==scope and not set(aa)&set(bb);seen+=bb
      for x,y in r['selection_folds']:assert set(x)|set(y)==set(aa) and not set(x)&set(y) and not (set(x)|set(y))&set(bb)
      tag=path.stem.replace('_crossfit','');obj=joblib.load(root/f'{tag}_cross{k}.joblib');verify_membership(obj,aa,bb);p=replay(obj,bb);saved={r['index']:r['p'] for r in audit['OOF']};err=max(abs(q-saved[i]) for i,q in zip(bb,p));assert err<1e-10;maxerr=max(maxerr,err);count+=1
     assert sorted(seen)==sorted(scope)
   train=json.loads((root/'training.json').read_text());fitcount+=len(train);seconds+=sum(r['seconds'] for r in train);rows.append(out)
   import models;models.GEOMETRY.clear();models.MATRICES.clear()
   print('verified',args.stage,b['name'],flush=True)
 if not args.partial:assert len(rows)==60
 result=dict(status='PASS' if len(rows)==60 else 'PASS_PARTIAL',blocks=len(rows),model_replays=count,max_probability_error=maxerr,calibration_memberships=cal,fits_recorded=fitcount,fit_seconds=seconds,checks=['seal hashes','row and label parity','canonical baseline equality','fit membership','train-only semantic geometry','selected C and gamma reconstructed','direct prediction replay','fallback replay','residual exact no-evidence fallback and nested membership where applicable'],verifier_fits=0)
 if len(rows)==60:
  pd.concat(rows).to_csv(OUT/f'{args.stage}_predictions.csv',index=False,float_format='%.17g');dump(OUT/f'{args.stage}_VERIFICATION.json',result)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
