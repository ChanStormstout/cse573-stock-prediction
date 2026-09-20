"""Outcome-blind preparation and finite preregistration before any new fit."""
from engine import *
import subprocess

def main():
 assert not PUB.exists() and not WORK.exists();check_sources();d,*_=load();assert len(d)==1607
 PUB.mkdir(parents=True);WORK.mkdir(parents=True)
 protocol=dict(reference_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),status='FROZEN_BEFORE_FITS',interpretation='EXPOSED EXPLORATORY RANDOM HISTORICAL BACKTEST',rows=1607,seeds=SEEDS,C=CS,threshold=.5,
 preprocessing=dict(methods=VARIANTS,control='saved old TF-IDF calibrated SVM + matched PRICE fallback',cleaning='exact historical body cleaning and alphabetic word extraction; same title and article membership',FREQUENCY='keep repeated occurrences rather than sorted unique tokens only',NEGATION='remove only no/not/nor from stoplist',NUMERIC='add unique namespaced numeric expressions; historical alphabetic extraction unchanged',NO_STEM='disable stemming only',selection='text-only inner mean BA descending, Brier ascending, smaller C, matching historical SVM',calibration='same 3 shuffled stratified folds, independently refit TFIDF and classifier per calibration fold',vocabulary='min_df3 max5000 sublinearTF L2; unigram only',fallback='same saved PRICE on every no-news window'),
 grouped=dict(method='same fixed CONTROL SVM, original association-group outer and inner partitions',calibration='three StratifiedGroupKFold splits within each current fit; same group cannot cross calibration boundary',selection='same text-only inner BA/Brier/C rule',interpretation='association components not certified event IDs; separate robustness test, not a replacement main score'),
 aggregation=dict(methods=['SET_BASE','MEAN_LINEAR','MEAN_THEN_MAP','MAP_THEN_MEAN'],base='TFIDF original stem_body max5000 plus train-standardized OLD16 divided by sqrt(16); L2 logistic regression',semantic_weight=1.,candidate_C=CS,mean_linear_dimensions=768,RFF_dimensions=256,RFF_seed=573,kernel='Gaussian random Fourier features, exp(-squared_distance/(2*sigma^2))',geometry='standardize unique training chunks only; sigma median positive pair distance of seed573 permutation of up to1024 unique train chunks',chunk_weights='existing token-count weights within each article; equal evidence articles within window',normalization='L2 window semantic vector; same for all representations; zero if no evidence',same_mapping_for_before_after=True,fit_scope='vocabulary, scaler, distance bandwidth fit separately within each training partition',selection='fallback-applied inner mean BA descending, Brier ascending, smaller C',fallback='exact saved PRICE when no news; news without evidence retains trained joint classifier with zero semantic block',no_new_encoder=True,no_PCA=True,comparison='same LR and same word/price block; compare to FULL and fixed old calibrated SVM as external controls'),
 followup=dict(interactions='deferred unless MAP_THEN_MEAN improves both-stock three-seed mean BA vs SET_BASE and MEAN_THEN_MAP, improves at least2seeds perstock, Brier harm <=.002 each stock; descriptive budget screen not independent significance',TabPFN='later conditional proposal only; not auto-run',LLM='remain paused; no continuation',no_best_seed_or_stock_winner=True),
 reporting=dict(metrics=['BA','MCC','Brier','predicted_up','constant'],paired_intervals='1000 common-day and 5-day-block paired bootstrap; shared weights across seeds; descriptive',cases='seed573 hash-first2 per stock/method corrected, harmed, shared wrong, shared right; no causal claim',verification='all saved model containers reloaded without fit; inner selection recomputed; calibration membership and fallbacks; raw reconstruction; preservation'),budget=dict(preprocess_top_level_fits=2400,preprocess_svm_internal_classifiers=7200,grouped_top_level_fits=600,grouped_svm_internal_classifiers=1800,aggregation_LR_fits=2400))
 dump(PUB/'protocol.json',protocol)
 oldledger=json.loads((ROOT/'work/stock-data/text_joint_4h/v1/protected.json').read_text());paths={ROOT/p for p in oldledger}
 for parent in [ROOT/'outputs/stock_text_joint_4h/v1',ROOT/'work/stock-data/text_joint_4h/v1']:
  paths.update(p for p in parent.rglob('*') if p.is_file() and p.suffix in {'.joblib','.json','.csv','.pkl','.npz','.jsonl'})
 protected={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)};dump(WORK/'protected.json',protected)
 raw=pd.read_pickle(SOURCES['raw_index']);raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key');needed=sorted({k for s in d.news_record_keys for k in s.split('|') if k});views={}
 for archive,g in raw.loc[needed].groupby('archive'):
  with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive) as z:
   for key,r in g.iterrows():views[key]=article_views(str(r.title),json.loads(z.read(r.member)).get('text',''))
 documents={m:[] for m in ['CONTROL']+VARIANTS}
 for row in d.itertuples():
  keys=[k for k in row.news_record_keys.split('|') if k]
  for m in documents:documents[m].append(assemble(views,keys,m))
 assert documents['CONTROL']==d.stem_body.tolist(),'historical reconstruction mismatch'
 groups=read(PRIVATE/'groups.csv');assert groups.row_id.tolist()==d.row_id.tolist()
 rb=random_blocks(d);gb=json.loads((PRIVATE/'diagnostic_baseline/seal.json').read_text())['blocks'];gb=[b for b in gb if b['protocol']=='grouped']
 assert len(rb)==len(gb)==60
 for b in gb:
  for a,e in b['inner']+[(b['train'],b['test'])]:
   assert not set(groups.group.iloc[a])&set(groups.group.iloc[e]);calibration(a,d,b['seed'],groups.group.to_numpy())
 joblib.dump(dict(documents=documents,groups=groups.group.to_numpy(),random=rb,grouped=gb),WORK/'inputs.joblib',compress=3)
 arts=[json.loads(s) for s in (SEM/'articles.jsonl').read_text().splitlines()];chunks=[json.loads(s) for s in (SEM/'chunks.jsonl').read_text().splitlines()];keys=[];vec=[];counts=[]
 for p in sorted((SEM/'FINBERT').glob('batch_*.npz')):
  assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];z=np.load(p);keys.extend(z['keys'].tolist());vec.append(z['vectors']);counts.extend(z['counts'])
 assert keys==[r['chunk_id'] for r in chunks];v=np.concatenate(vec).astype(float);cnt=np.array(counts,float);lookup={k:i for i,k in enumerate(keys)};alook={(r['symbol'],r['record_key']):r for r in arts};ixs=[];weights=[]
 for row in d.itertuples():
  eligible=[alook[row.symbol,k]['chunks'] for k in row.news_record_keys.split('|') if k and alook[row.symbol,k]['chunks']];ii=[];ww=[]
  for ks in eligible:
   ids=np.array([lookup[k] for k in ks]);ii.extend(ids);ww.extend(cnt[ids]/cnt[ids].sum()/len(eligible))
  ixs.append(np.array(ii,int));weights.append(np.array(ww,float));assert not ww or abs(sum(ww)-1)<1e-12
 joblib.dump(dict(vectors=v,counts=cnt,window_chunks=ixs,weights=weights),WORK/'chunks.joblib',compress=3)
 # Compare weighted reconstruction to accepted article means (float32 archive tolerance).
 cached=np.load(SEM/'FINBERT_articles.npz');av=cached['vectors'];ai={(r['symbol'],r['record_key']):i for i,r in enumerate(arts)};errors=[]
 for row,ii,ww in zip(d.itertuples(),ixs,weights):
  jj=[ai[row.symbol,k] for k in row.news_record_keys.split('|') if k and alook[row.symbol,k]['chunks']]
  if jj:errors.append(float(np.max(abs(v[ii].T@ww-av[jj].mean(0)))))
 assert max(errors)<1e-5
 deps=[Path(__file__),Path(__file__).with_name('engine.py'),Path(__file__).with_name('run.py'),PUB/'protocol.json',WORK/'inputs.joblib',WORK/'chunks.joblib',WORK/'protected.json',PRIVATE/'outer_folds.csv',PRIVATE/'inner_folds.json',PRIVATE/'diagnostic_baseline/seal.json',SEM/'manifest.json']
 dump(WORK/'seal.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in deps}})
 dump(PUB/'INPUT_AUDIT.json',dict(status='PASS_FIT_FREE',rows=len(d),articles=len(views),chunks=len(v),original_text_exact_parity_rows=len(d),random_blocks=len(rb),grouped_blocks=len(gb),group_calibration_feasibility='PASS',max_chunk_reaggregation_error=max(errors),protected_files=len(protected),protected_sha=sha(WORK/'protected.json'),seal_sha=sha(WORK/'seal.json'),fits=0,coverage={s:dict(rows=len(g),no_news=int((g.has_news==0).sum())) for s,g in d.groupby('symbol')},document_changes={m:int(sum(x!=y for x,y in zip(documents[m],documents['CONTROL']))) for m in VARIANTS}))
 print('Prepared and sealed',flush=True)
if __name__=='__main__':main()
