import json,re,html
from functools import lru_cache
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from common import B,O,sha,require_empty
R=B/'results';require_empty(R);R.mkdir(exist_ok=True)
(R/'preparing.json').write_text('{"status":"preparing"}')
paths=[O/'stock_finbert/results/article_probabilities.csv',O/'stock_content/results/source_bodies.json',O/'stock_baseline/results/results.json',B/'protocol.json',B/'common.py',B/'prepare.py',B/'run_linear.py',B/'run_temporal.py']
paths += [O/f'stock_finbert/results/{s}/development_features.csv' for s in ['AAPL','AMZN']]
paths += [O/f'stock_temporal/results/{s}/{f}' for s in ['AAPL','AMZN'] for f in ['data.npz','manifest.csv']]
paths += [O/'stock_temporal/models.py',O/'stock_baseline/run_baseline.py',O/'stock_review_fixes/guards.py']
source={str(p.relative_to(O)):sha(p) for p in paths}
a=pd.read_csv(paths[0]).set_index('record_key');bodies=json.loads(paths[1].read_text())
stemmer=EnglishStemmer()
@lru_cache(maxsize=150000)
def stem(w):return stemmer.stem(w)
bad=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',re.I)
def tokens(s):
    return set(stem(w) for w in re.findall(r'[a-z]+',s.lower()) if len(w)>1 and w not in ENGLISH_STOP_WORDS)
title={k:tokens(str(v)) for k,v in a.title.items()};body={}
for k in a.index:
    text=html.unescape(re.sub(r'<[^>]+>',' ',bodies[k]));text=' '.join(line for line in text.splitlines() if not bad.search(line));text=re.sub(r'https?://\S+',' ',text)
    body[k]=tokens(text)|title[k]
frames=[]
for si,s in enumerate(['AAPL','AMZN']):
    d=pd.read_csv(O/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''})
    assert set(d.split)=={'train','validation'};rows=[]
    for r in d.itertuples():
        keys=r.news_record_keys.split('|') if r.news_record_keys else []
        times=pd.to_datetime(a.loc[keys,'available_utc'],utc=True);cut=pd.Timestamp(r.cutoff_utc)
        assert times.le(cut).all() and times.gt(cut-pd.Timedelta(hours=4)).all()
        rows.append(dict(text=' . '.join(a.loc[keys,'title']),stem_title=' '.join(sorted(set().union(*(title[k] for k in keys)))),stem_body=' '.join(sorted(set().union(*(body[k] for k in keys))))))
    d=pd.concat([d,pd.DataFrame(rows)],axis=1);d['stock_index']=si;frames.append(d)
d=pd.concat(frames,ignore_index=True);d.to_pickle(R/'data.pkl')
meta=dict(status='prepared',source_sha256=source,artifact_sha256={'data.pkl':sha(R/'data.pkl')},rows=d.groupby(['symbol','split']).size().to_dict().__str__(),note='Stemming is deterministic, no fitted vocabulary here; binary unigram uses window token union.')
(R/'prepared.json').write_text(json.dumps(meta,indent=2));print('Prepared',len(d),'development-only windows',flush=True)
