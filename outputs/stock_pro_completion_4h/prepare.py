"""Freeze protocol and reconstruct real-adjacency bigrams; zero estimator fits."""
from models import *
import subprocess

def main():
 assert not OUT.exists() and not LOCAL.exists();previous.seal_check();check_sources();d,*_=load();OUT.mkdir(parents=True);LOCAL.mkdir(parents=True)
 protocol=dict(status='FROZEN_BEFORE_FITS',reference_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),interpretation='EXPOSED EXPLORATORY RANDOM HISTORICAL BACKTEST; overlapping news/windows may cross random folds; not prospective performance',rows=1607,seeds=SEEDS,outer_folds=10,inner_folds=3,C=CS,threshold=.5,
 lexical=dict(methods=['UNI','BIGRAM','NB_BIGRAM'],unigrams='exact old stem_body token set, default length>=2 token restriction',bigrams='unique pairs of adjacent retained alphabetic tokens within each original sentence/paragraph/title; stopword removal DOES NOT bridge a gap; stemmer/stoplist unchanged; no invented pairs from sorted stem_body',representation='same min_df3 max5000 sublinear TFIDF, 3-fold sigmoid calibrated ensemble LinearSVC; NB uses binary occurrence log smoothed class-count ratio within each calibration training partition',selection='text-only inner mean BA descending, Brier ascending, smaller C; same as historical SVM'),
 kernel=dict(methods=['K_WP','K_MEAN','K_AFTER','K_SET','K_INTERACTION'],solver='linear-kernel SVC on explicit concatenated maps, equivalent to sum of block dot-product kernels; 3-fold calibrated ensemble',base='historical unigram TFIDF5000 + standardized OLD16 /sqrt16',mean='train-standardized chunk mean /sqrt768',after='Gaussian RFF of ordinary mean',set='weighted mean of Gaussian chunk RFF; token weights within article, equal article weights; no final L2 normalization',RFF_dimensions=256,RFF_seed=573,geometry='unique training chunks only; median positive sampled paired distance, matching earlier geometry',semantic_weight=1,interaction='K_set elementwise-multiplied by K_state implemented by tensor product',state=STATE,state_scale='train standardized /sqrt3',gamma_candidates=[.1,1.],selection='mean text-and-price inner BA descending, Brier ascending, C ascending, gamma ascending',no_main_effect_advancement_gate=True,no_evidence='use matching K_WP probability exactly',no_news='saved same-fold PRICE probability exactly'),
 table=dict(methods=['TABLE_LR','TABLE_RF','TABLE_TABPFN'],word_features=128,semantic_dimensions=32,price_features=16,metadata=['log1p(news_count)','evidence_flag'],max_dimensions=178,feature_selection='train chi2 top128 of same bigram TFIDF; PCA32 fit unique training evidence articles',LR_C=CS,RF=dict(n_estimators=200,max_depth=3,min_samples_leaf=10),TabPFN=dict(version='6.3.0',checkpoint_version='2.5 default fine-tuned on real data',checkpoint_sha=sha(CHECKPOINT),n_estimators=8,device='cpu',n_preprocessing_jobs=1,license='Prior Labs License; retained existing local authorized checkpoint; no redistribution',task_gradient_training=False),selection='LR only: inner mean BA/Brier/C; RF/TabPFN fixed without inner model tuning',variants=['NOSEM','SEM'],no_evidence='same learner NOSEM probability',no_news='saved same-fold PRICE probability',independent_of_kernel_result=True),
 residual=dict(name='R16',base='FULL binary chi2top500 OLD16 LR plus PRICE36 fallback; same C grid and inner selection',fit='offset logistic 16 coefficients, no intercept, sum logloss + ||w||^2/(2C)',OOF='fresh 3-fold nested crossfit inside EVERY residual training scope; each base C selected only inside that crossfit complement; never use old selection_oof_NOT_META_TRAIN for fitting',selection='C by original inner folds; include OFF, ties prefer OFF; outer base uses canonical inner folds',PCA='unique training evidence articles only, train evidence scale',gate='no evidence returns base probability bit-exact'),
 reporting=dict(all_seeds=True,no_stock_winners=True,monthly=True,paired_intervals='1000 paired day and 5day blocks, descriptive only',cases='hash-first per repair/harm/sharedwrong/sharedright, seed573; examples are diagnostic not causal'),deferred=['TabSTAR later-stage joint pretrained architecture','new LLM calls/learned retrieval','attention','LoRA','RL','pseudo-label consistency'],verification='no-fit reload of ALL final models; inner selection replay; membership, exact fallbacks, cache seals, protected artifacts; TabPFN deterministic CPU tolerance1e-5, other probabilities1e-10')
 dump(OUT/'protocol.json',protocol)
 raw=pd.read_pickle(SOURCES['raw_index']);raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key');needed=sorted({k for v in d.news_record_keys for k in v.split('|') if k});pairs={}
 for archive,g in raw.loc[needed].groupby('archive'):
  with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive) as z:
   for k,r in g.iterrows():
    text=html.unescape(re.sub(r'<[^>]+>',' ',json.loads(z.read(r.member)).get('text','')))
    lines=[line for line in text.splitlines() if not previous.BAD.search(line)]+[str(r.title)];ps=set()
    for line in lines:
     line=re.sub(r'https?://\S+',' ',line)
     for sentence in re.split(r'[.!?;\n]',line):
      tokens=re.findall('[a-z]+',sentence.lower())
      for a,b in zip(tokens,tokens[1:]):
       if len(a)>1 and len(b)>1 and a not in ENGLISH_STOP_WORDS and b not in ENGLISH_STOP_WORDS:
        ps.add('bg_'+previous.stem(a)+'__'+previous.stem(b))
    pairs[k]=ps
 uni=[' '.join(w for w in s.split() if len(w)>1) for s in d.stem_body];big=[]
 for row,u in zip(d.itertuples(),uni):big.append(' '.join(sorted(set(u.split())|{p for k in row.news_record_keys.split('|') if k for p in pairs[k]})))
 joblib.dump(dict(documents={'UNI':uni,'BIGRAM':big},blocks=previous.random_blocks(d)),LOCAL/'inputs.joblib',compress=3)
 protected=json.loads((previous.WORK/'protected.json').read_text())
 for parent in [previous.PUB,previous.WORK]:
  for p in parent.rglob('*'):
   if p.is_file() and p.suffix in {'.json','.csv','.joblib','.md'}:protected[str(p.relative_to(ROOT))]=sha(p)
 dump(LOCAL/'protected.json',protected)
 paths=[LOCAL/'inputs.joblib',LOCAL/'protected.json',OUT/'protocol.json',previous.WORK/'chunks.joblib',CHECKPOINT,PRIVATE/'outer_folds.csv',PRIVATE/'inner_folds.json']+list(Path(__file__).parent.glob('*.py'))
 dump(LOCAL/'seal.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
 dump(OUT/'INPUT_AUDIT.json',dict(status='PASS_FIT_FREE',rows=len(d),blocks=60,articles=len(pairs),bigram_windows=sum(u!=b for u,b in zip(uni,big)),protected_files=len(protected),fits=0,checkpoint_sha=sha(CHECKPOINT)))
 print('Protocol and inputs frozen; zero fits',flush=True)
if __name__=='__main__':main()
