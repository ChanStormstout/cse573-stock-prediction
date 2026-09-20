"""Fit-free source audit and immutable preregistration."""
from core import *
import zipfile,subprocess,collections

def main():
 assert not PUBLIC.exists() and not WORK.exists(),'Never overwrite an existing run'
 check_sources();d,*_=load();assert len(d)==1607
 # Bind all prior model/prediction/input files, not merely their own manifest.
 paths=set(SOURCES.values())
 for base in [PRIVATE,ROOT/'work/stock-data/full_semantics_4h/v1']:
  paths.update(p for p in base.rglob('*') if p.is_file() and p.suffix in {'.joblib','.pkl','.npz','.csv','.json','.jsonl'} and 'log' not in p.name and 'live' not in str(p))
 paths.add(ROOT/'outputs/stock_integrated_4h/prepare.py');paths.add(ROOT/'outputs/stock_random_protocol_4h/common.py')
 manifest=json.loads(SOURCES['canonical_manifest'].read_text());paths.update(Path(p) for p in manifest['archive_hashes'])
 paths.add(ROOT/'outputs/stock_random_protocol_4h/PROTOCOL.md')
 print('Hashing protected artifacts',len(paths),flush=True)
 h={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
 for p,v in manifest['archive_hashes'].items():assert sha(p)==v
 llm=PRIVATE/'analogy/generated.jsonl';assert sum(1 for _ in llm.open())==1972
 assert sha(llm)=='c37cbc45187fd6cbd86a3144cd06e294963df509088471fc57664e817e8d8ae7'
 states=subprocess.check_output(['ps','-p','2333,3030','-o','pid=,state='],text=True)
 assert len(states.splitlines())==2 and all('T' in s for s in states.splitlines())
 protocol=dict(status='FROZEN_BEFORE_NEW_PREDICTIVE_FITS',reference_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),interpretation='EXPLORATORY RANDOM HISTORICAL BACKTEST; all periods exposed',seeds=SEEDS,outer_folds=10,inner_folds=3,rows=1607,stocks={'AAPL':803,'AMZN':804},threshold=.5,C=CS,
 methods={'A1':'raw ordered unigram TFIDF LR','A2':'same unigram+within-sentence bigram TFIDF LR','A3':'same A2 text; calibrated LinearSVC; full pipeline refit in each of 3 calibration splits','B1':'original FULL + frozen target-paragraph FinBERT PCA16 jointly fitted LR','B2':'same FULL + uncompressed 768-d FinBERT jointly fitted LR'},
 text=dict(min_df=3,max_features=5000,sublinear_tf=True,norm='l2',token_regex=TOKEN.pattern,stopwords=None,stemming=False,frequency='preserved including repeated article occurrences in each canonical window',boundaries='title/body, physical paragraph and conservative punctuation hard boundaries; decimals retained; no cross-boundary bigrams',cleaning='strip script/style, HTML block tags -> newline; remaining tags stripped, HTML unescape, URLs removed, whitespace normalized; only lines starting fixed template pattern removed',template_pattern=BAD.pattern),
 semantics=dict(source='existing verified target-company paragraphs, not complete full body',rho=[0.,.1,1.],PCA='unique train-member evidence articles only',scale='per-coordinate train-window mean/std; no-evidence semantic block zero after transform; scale then rho; no subsequent scaling',base='original OLD16+binary min_df3 chi2<=500 L2 FULL',no_new_metadata=True,rho0='reuse original per-C FULL inner models; original selected FULL outer path; zero-rho best C must equal original FULL selection',fallback='nonzero-rho no evidence exactly selected FULL; no news exactly saved PRICE'),
 selection='higher inner mean BA, lower inner mean Brier, smaller rho then smaller C; A rho=0',A_fallback='saved legal PRICE inner/outer predictions on no news',SVM_direction='calibrated ensemble probability >=0.5; not raw margin',shrink=dict(a=[.25,.5,.75,1.],selection='minimum pooled inner Brier, tie larger a; selected inner FULL OOF used only for finite selection',no_news='exact PRICE',direction='must remain identical'),
 budget=dict(A_top_level_fits=1800,A_LR_classifier_fits=1200,A_SVM_internal_classifier_fits=1800,A_SVM_sigmoid_calibrators=1800,B_inner_classifier_fits=2160,B_outer_classifier_fits_max=120,B_outer_rho0_reuses='no new fit',total_new_predictive_classifier_fits_max=5280),
 checks='no-fit verifier, source bindings, calibration memberships, selection reconstruction, exact fallbacks/rho0, reload probabilities, unchanged windows, boundary/frequency tests',intervals='1000 paired resamples each one-day, circular-five-day and frozen association groups; shared weights across stocks and seeds; report seed-mean delta, not independent seeds; descriptive after exploration',cases='per stock and new method stable hash first two from repaired/harmed/both wrong/both right against FULL at seed573; supplement no-news/negation/numeric/multi-company/near-repost by hash without correctness filter',stop='no grid expansion or advanced branches; preserve failed/incomplete runs and report',LLM='remain paused at1972; no automatic continuation')
 PUBLIC.mkdir(parents=True);WORK.mkdir(parents=True)
 dump(PUBLIC/'protocol.json',protocol);dump(WORK/'protected.json',h)
 raw=pd.read_pickle(SOURCES['raw_index']);raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key');needed=sorted({k for s in d.news_record_keys for k in s.split('|') if k});articles={}
 for archive,g in raw.loc[needed].groupby('archive'):
  with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive) as z:
   for key,r in g.iterrows():
    body=json.loads(z.read(r.member)).get('text','');title='' if pd.isna(r.title) else str(r.title);parts,removed=clean(title,body)
    articles[key]=dict(raw_body_sha256=digest(body),title_sha256=digest(title),parts=parts,removed=removed)
 docs=[];mapping=[]
 for r in d.itertuples():
  ks=[k for k in r.news_record_keys.split('|') if k];parts=[p for k in ks for p in articles[k]['parts']]
  docs.append(dict(row_id=r.row_id,u=features(parts,False),b=features(parts,True)))
  mapping.append(dict(row_id=r.row_id,article_keys=ks,segments=len(parts),tokens=sum(len(p['tokens']) for p in parts)))
 # Independent old reconstruction confirms the set/sort, stopword and number loss.
 from nltk.stem.snowball import EnglishStemmer
 from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
 assert {'not','no','nor'}<=ENGLISH_STOP_WORDS
 assert all(x== ' '.join(sorted(set(x.split()))) for x in d.stem_body)
 assert not any(re.search(r'\d',x) for x in d.stem_body)
 joblib.dump(dict(docs=docs,articles=articles,mapping=mapping),WORK/'inputs.joblib',compress=3)
 d[['row_id','key','symbol','day','label','start_utc','end_utc','cutoff_utc','news_record_keys']].to_pickle(WORK/'canonical.pkl')
 _,ids,_,gate=semantic(d)
 audit=dict(status='PASS_FIT_FREE',rows=len(d),article_keys=len(needed),row_mapping_sha256=digest(json.dumps(mapping)),input_sha256=sha(WORK/'inputs.joblib'),canonical_sha256=sha(WORK/'canonical.pkl'),protected_files=len(h),protected_ledger_sha256=sha(WORK/'protected.json'),old_stem_body=dict(sorted_unique_all_rows=True,numeric_tokens=0,negations_in_stoplist=['not','no','nor'],source= str((ROOT/'outputs/stock_integrated_4h/prepare.py').relative_to(ROOT))),
 coverage={s:dict(rows=len(g),no_news=int((g.has_news==0).sum()),news_without_target_paragraph=int(((g.has_news>0)&~gate[g.index]).sum()),target_paragraph=int(gate[g.index].sum())) for s,g in d.groupby('symbol')},
 new_text=dict(tokens=sum(len(x['u']) for x in docs),numeric_tokens=sum(sum(bool(re.search(r'\d',t)) for t in x['u']) for x in docs),negations=sum(sum(t in ['no','not','nor'] for t in x['u']) for x in docs),repeated_occurrences=sum(len(x['u'])-len(set(x['u'])) for x in docs),empty_news_windows=[d.iloc[i].row_id for i,x in enumerate(docs) if d.iloc[i].has_news and not x['u']]),LLM=dict(records=1972,sha256=sha(llm),process_states=states,continued=False),predictive_fits=0)
 dump(PUBLIC/'INPUT_AUDIT.json',audit)
 (PUBLIC/'INPUT_AUDIT.md').write_text('# Input audit\n\nOld sorted binary stem sets remain valid historical inputs, but discard frequency/order/numbers and stoplisted negation. New A inputs reconstruct the same article membership from raw title/body, preserving within-sentence sequence and repetitions. B reuses existing target-paragraph vectors unchanged. No model fits were performed during preparation.\n\n```json\n'+json.dumps(audit,indent=2)+'\n```\n')
 print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
