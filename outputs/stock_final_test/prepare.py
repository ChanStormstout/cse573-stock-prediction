"""Reconstruct the unchanged holdout inputs; no model prediction or fitting."""
import json,sys,re,html,zipfile
from pathlib import Path
from functools import lru_cache
import numpy as np,pandas as pd
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
B=Path(__file__).resolve().parent;O=B.parent;ROOT=O.parent/'work/stock-data'
sys.path.insert(0,str(O/'stock_robust'))
from common import sha,verify_files,require_empty,prepared,PRICE
from run_baseline import build,window_indices
P=json.loads((B/'protocol.json').read_text());verify_files(O.parent,P['source_sha256'])
R=B/'results';require_empty(R);R.mkdir(exist_ok=True);(R/'preparing.json').write_text('{"status":"preparing; no predictions"}')
olddev,_=prepared();frames=[];allnews={};raw=pd.read_pickle(ROOT/'audit/news_index.pkl');raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key');assert raw.index.is_unique
for s,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
    d,cols,ledger,n=build(ROOT,s,prefix);assert cols==PRICE
    old=pd.read_csv(O/f'stock_baseline/results/{s}/samples.csv').fillna({'news_ids':'','text':''})
    assert d.shape==old.shape
    for c in ['start_utc','end_utc','cutoff_utc']:
        np.testing.assert_array_equal(d[c].astype('int64'),pd.to_datetime(old[c],utc=True).astype('int64'))
    for c in ['symbol','split','text','news_ids','history_ends']:
        assert d[c].tolist()==old[c].tolist(),(s,c)
    np.testing.assert_allclose(d[PRICE+['label','news_count','has_news','target_return']],old[PRICE+['label','news_count','has_news','target_return']],atol=1e-12)
    n['record_key']=n.archive+'::'+n.member;times=n.available_utc.astype('int64').to_numpy();keys=[]
    for r in d.itertuples():
        a,b=window_indices(times,r.cutoff_utc);w=n.iloc[a:b]
        assert ' . '.join(w.title.fillna(''))==r.text and len(w)==r.news_count
        assert w.empty or (w.available_utc.le(r.cutoff_utc)&w.available_utc.gt(r.cutoff_utc-pd.Timedelta(hours=4))).all()
        keys.append('|'.join(w.record_key));allnews.update({key:None for key in w.record_key})
    d['news_record_keys']=keys;frames.append(d)
bodies={}
for archive,g in raw.loc[sorted(allnews)].groupby('archive'):
    with zipfile.ZipFile(ROOT/'raw/news'/archive) as z:
        for r in g.itertuples():
            value=json.loads(z.read(r.member)).get('text','');assert isinstance(value,str)
            bodies[r.Index]=value
    print('Read raw bodies',archive,len(g),flush=True)
prior_bodies=json.loads((O/'stock_content/results/source_bodies.json').read_text())
for key in set(prior_bodies)&set(bodies):assert prior_bodies[key]==bodies[key]
stemmer=EnglishStemmer();bad=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',re.I)
@lru_cache(maxsize=150000)
def stem(w):return stemmer.stem(w)
def tokens(s):return set(stem(w) for w in re.findall(r'[a-z]+',s.lower()) if len(w)>1 and w not in ENGLISH_STOP_WORDS)
title={};body={}
for key in sorted(allnews):
    title[key]=tokens(str(raw.loc[key,'title']))
    text=html.unescape(re.sub(r'<[^>]+>',' ',bodies[key]));text=' '.join(line for line in text.splitlines() if not bad.search(line));text=re.sub(r'https?://\S+',' ',text)
    body[key]=tokens(text)|title[key]
done=[];parity={};usage={}
for s,d in zip(['AAPL','AMZN'],frames):
    texts=[]
    for r in d.itertuples():
        keys=r.news_record_keys.split('|') if r.news_record_keys else []
        texts.append(dict(stem_title=' '.join(sorted(set().union(*(title[k] for k in keys)))),stem_body=' '.join(sorted(set().union(*(body[k] for k in keys))))))
    d=pd.concat([d,pd.DataFrame(texts)],axis=1)
    for c in ['start_utc','end_utc','cutoff_utc']:d[c]=d[c].astype(str)
    prev=olddev[olddev.symbol.eq(s)].reset_index(drop=True);now=d[d.split.ne('test')].reset_index(drop=True)
    for c in ['start_utc','news_record_keys','stem_title','stem_body','label']:
        assert now[c].tolist()==prev[c].tolist(),(s,c)
    parity[s]=len(now);t=d[d.split.eq('test')].reset_index(drop=True)
    assert len(t)==P['expected_test_windows'][s]
    assert pd.to_datetime(t.start_utc,utc=True).min()>=pd.Timestamp(P['test_start'],tz='UTC')
    assert pd.to_datetime(t.end_utc,utc=True).max()<pd.Timestamp(P['test_end_exclusive'],tz='UTC')
    done.append(t);usage[s]=set(k for seq in t.news_record_keys for k in seq.split('|') if k)
test=pd.concat(done,ignore_index=True);test.to_pickle(R/'test_inputs.pkl');test.drop(columns=['text','stem_title','stem_body']).to_csv(R/'manifest.csv',index=False)
needed=set.union(*usage.values());index=raw.loc[sorted(needed),['title','published_utc','crawled_utc','available_utc','archive','member']].reset_index()
index['body_sha256']=[__import__('hashlib').sha256(bodies[k].encode()).hexdigest() for k in index.record_key]
index['stem_title']=index.record_key.map(lambda k:' '.join(sorted(title[k])));index['stem_body']=index.record_key.map(lambda k:' '.join(sorted(body[k])));index.to_csv(R/'test_article_index.csv',index=False)
verify_files(O.parent,P['source_sha256'])
meta=dict(protocol_sha256=sha(B/'protocol.json'),artifact_sha256={name:sha(R/name) for name in ['test_inputs.pkl','manifest.csv','test_article_index.csv']},development_rows_reconstructed_exactly=parity,stocks={s:dict(test_windows=len(t),first=t.start_utc.min(),last=t.start_utc.max(),trading_days=t.start_utc.str[:10].nunique(),with_news=int(t.has_news.sum()),unique_articles=len(usage[s])) for s,t in test.groupby('symbol')},test_predictions_computed=False)
(R/'prepared.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta['stocks'],indent=2));print('All 2504 development inputs exactly reproduced; holdout inputs prepared, no predictions computed.',flush=True)
