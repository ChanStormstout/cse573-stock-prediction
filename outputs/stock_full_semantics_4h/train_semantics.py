"""Joint finite C/weight selection, with no learned stacking coefficients."""
import sys,json,time,warnings,argparse,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import *
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import joblib
PUBLIC=Path(__file__).resolve().parent/'v1';NEW=ROOT/'work/stock-data/full_semantics_4h/v1';SRC=NEW/'semantic'
WEIGHTS=[0.,.25,.5,.75,1.]
def semantic_input(d,name):
 a=[json.loads(s) for s in (SRC/'articles.jsonl').read_text().splitlines()];z=dict(np.load(SRC/(name+'_articles.npz')));assert z['keys'].tolist()==[r['article_id'] for r in a]
 lookup={(r['symbol'],r['record_key']):i for i,r in enumerate(a)};ids=[[lookup[r.symbol,k] for k in r.news_record_keys.split('|') if k and z['has_evidence'][lookup[r.symbol,k]]] for r in d.itertuples()]
 means=np.array([z['vectors'][ix].mean(0) if ix else np.zeros(z['vectors'].shape[1]) for ix in ids]);return z['vectors'],ids,means
def fit_transform(d,tr,emb,ids,means):
 articles=sorted({j for i in tr for j in ids[i]});pca=PCA(16,svd_solver='randomized',random_state=573).fit(emb[articles]);x=np.c_[d[OLD],pca.transform(means),np.log1p(d.news_count),d.has_news];scale=Scale().fit(x[tr]);return {'pca':pca,'scale':scale,'article_ids':articles},scale.transform(x)
def blend(base,semantic,gate,w):
 out=np.asarray(base).copy()
 if w:out[gate]=(1-w)*out[gate]+w*np.asarray(semantic)[gate]
 return out
def main(encoder=None):
 check_sources();d,*_=load();outer=pd.read_csv(PRIVATE/'outer_folds.csv');inner=json.loads((PRIVATE/'inner_folds.json').read_text());dest=NEW/'semantic_training';dest.mkdir(exist_ok=True)
 base_hashes={}
 for block in sorted((PRIVATE/'baseline').iterdir()):
  complete=json.loads((block/'complete.json').read_text())
  for n,h in complete['artifacts'].items():assert sha(block/n)==h
  for n in ['FULL_selection_oof_NOT_META_TRAIN.csv','outer_predictions_SEALED.csv','complete.json']:base_hashes[str((block/n).relative_to(PRIVATE))]=sha(block/n)
 stamp={'base_hashes':base_hashes,'code':sha(__file__),'protocol':sha(PUBLIC/'SEMANTIC_PROTOCOL.json'),'encoder_seals':{name:sha(SRC/name/'seal.json') for name in ['FINBERT','MODERN']},'input_manifest':sha(SRC/'manifest.json'),'source_manifest':sha(PRIVATE/'manifest.json')}
 seal=dest/'seal.json'
 if seal.exists():assert json.loads(seal.read_text())==stamp
 else:dump(seal,stamp)
 for name in ([encoder] if encoder else ['FINBERT','MODERN']):
  evidence=json.loads((PUBLIC/(name+'_ENCODING.json')).read_text());assert evidence['status']=='COMPLETE' and evidence['output_sha256']==sha(SRC/(name+'_articles.npz'))
  emb,ids,means=semantic_input(d,name);gate=np.array([bool(ix) for ix in ids])
  for seed in SEEDS:
   for stock in ['AAPL','AMZN']:
    for fold in range(10):
     run=dest/f'{name}_{seed}_{stock}_{fold}';base=PRIVATE/'baseline'/f'{seed}_{stock}_{fold}'
     if (run/'complete.json').exists():
      old=json.loads((run/'complete.json').read_text());assert old['seal']==sha(seal)
      for n,h in old['files'].items():assert sha(run/n)==h
      continue
     run.mkdir(exist_ok=False);splits=[s for s in inner if s['seed']==seed and s['symbol']==stock and s['fold']==fold]
     ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy(int);tr=np.array(sorted(set(np.flatnonzero(d.symbol==stock))-set(ev)))
     baseoof=pd.read_csv(base/'FULL_selection_oof_NOT_META_TRAIN.csv',float_precision='round_trip').set_index('index');records=[]
     for s in splits:
      a=np.array(s['train']);b=np.array(s['validation']);tf,x=fit_transform(d,a,emb,ids,means);bp=baseoof.loc[b].p.to_numpy()
      for c in CS:
       m=classifier('FULL',c).fit(x[a],d.iloc[a].label);p=m.predict_proba(x[b])[:,1]
       for w in WEIGHTS:records.append(dict(C=c,weight=w,inner_fold=s['inner_fold'],**metric(d.iloc[b].label,blend(bp,p,gate[b],w))))
     best=min([(c,w) for c in CS for w in WEIGHTS],key=lambda cw:(-np.mean([r['BA'] for r in records if (r['C'],r['weight'])==cw]),np.mean([r['Brier'] for r in records if (r['C'],r['weight'])==cw]),cw[1],cw[0]));c,w=best
     tf,x=fit_transform(d,tr,emb,ids,means);m=classifier('FULL',c).fit(x[tr],d.iloc[tr].label)
     pred=pd.read_csv(base/'outer_predictions_SEALED.csv',float_precision='round_trip');assert pred.row_id.tolist()==d.iloc[ev].row_id.tolist();p=m.predict_proba(x[ev])[:,1];bp=pred.FULL.to_numpy();q=blend(bp,p,gate[ev],w);assert np.array_equal(q[~gate[ev]],bp[~gate[ev]])
     joblib.dump(dict(model=m,transform=tf,train=tr.tolist(),evaluation=ev.tolist(),C=c,weight=w),run/'selected.joblib',compress=3)
     pred=pred[['row_id','symbol','day','label']].copy();pred['seed']=seed;pred['fold']=fold;pred['encoder']=name;pred['FULL']=bp;pred['semantic_p']=p;pred['gate']=gate[ev];pred['p']=q;pred.to_csv(run/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(records).to_csv(run/'selection.csv',index=False)
     dump(run/'complete.json',dict(seal=sha(seal),fits=10,files={f.name:sha(f) for f in run.iterdir() if f.is_file()}));print(name,seed,stock,fold,'complete',flush=True)
 completed=list(dest.glob('*/complete.json'));dump(PUBLIC/'SEMANTIC_TRAINING.json',dict(status='COMPLETE_PENDING_VERIFICATION' if len(completed)==120 else 'PARTIAL_PENDING_OTHER_ENCODER',fits=sum(json.loads(p.read_text())['fits'] for p in completed),selected_models=len(completed),learned_meta_coefficients=0))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--encoder',choices=['FINBERT','MODERN']);args=parser.parse_args()
 with (NEW/'semantic_training.lock').open('a') as lock,threadpool_limits(2),warnings.catch_warnings():
  fcntl.flock(lock,fcntl.LOCK_EX);warnings.simplefilter('ignore',FutureWarning);main(args.encoder)
