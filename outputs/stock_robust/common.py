from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
B=Path(__file__).resolve().parent;O=B.parent
sys.path.insert(0,str(O/'stock_review_fixes'))
from guards import sha,verify_files,require_empty,fingerprint
sys.path.insert(0,str(O/'stock_baseline'))
from run_baseline import scores
PRICE=json.loads((O/'stock_baseline/results/results.json').read_text())['stocks']['AAPL']['features']
SENT=['finbert_positive','finbert_negative','finbert_neutral','sentiment_std']
COUNT=['log_news_count','has_news'];FEATURES=PRICE+SENT+COUNT


def fold_indices(d):
    t=pd.to_datetime(d.start_utc,utc=True)
    for month in [6,7,8]:
        lo=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC');hi=lo+pd.offsets.MonthBegin()
        a=np.flatnonzero((t<lo)&d.split.eq('train'));b=np.flatnonzero((t>=lo)&(t<hi)&d.split.eq('train'))
        assert pd.to_datetime(d.end_utc.iloc[a],utc=True).max()<pd.to_datetime(d.cutoff_utc.iloc[b],utc=True).min()
        yield month,a,b


def inner_split(d,ids,n_days=20):
    days=d.start_utc.str[:10];unique=sorted(days.iloc[ids].unique());assert len(unique)>n_days+20
    threshold=unique[-n_days];a=ids[(days.iloc[ids]<threshold).to_numpy()];b=ids[(days.iloc[ids]>=threshold).to_numpy()]
    assert pd.to_datetime(d.end_utc.iloc[a],utc=True).max()<pd.to_datetime(d.cutoff_utc.iloc[b],utc=True).min()
    return a,b


def metric_details(d,p):
    return dict(overall=scores(d.label,p),months={m:scores(d.loc[mask,'label'],p[mask.to_numpy()]) for m in ['2018-09','2018-10'] if (mask:=d.start_utc.str.startswith(m)).any()},coverage={name:dict(n=int(mask.sum()),**scores(d.loc[mask,'label'],p[mask.to_numpy()])) for name,mask in [('with_news',d.has_news.eq(1)),('no_news',d.has_news.eq(0))] if mask.any()})


def baseline(kind,C):
    parts=[]
    num={'price':PRICE,'text':[],'combined':PRICE,'price_count':PRICE+COUNT,'finbert_only':SENT+COUNT,'price_finbert':FEATURES,'paper_stem_title':PRICE,'paper_stem_body':PRICE}[kind]
    if num:parts.append(('numeric',StandardScaler(),num))
    if kind in ['text','combined']:parts.append(('text',TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=3,max_df=.98,sublinear_tf=True),'text'))
    if kind.startswith('paper_'):
        parts.append(('words',make_pipeline(CountVectorizer(binary=True,min_df=3,max_features=10000),SelectKBest(chi2,k=500)),'stem_body' if kind.endswith('body') else 'stem_title'))
    return Pipeline([('features',ColumnTransformer(parts)),('model',LogisticRegression(C=C,l1_ratio=int(kind.startswith('paper_')),solver='liblinear',max_iter=3000,random_state=573))])


def sharing_design(x,stock,mode):
    onehot=np.eye(2)[np.asarray(stock,dtype=int)];blocks=[x*onehot[:,i,None] for i in range(2)]
    return np.concatenate(([x] if mode=='shared' else blocks if mode=='separate' else [x]+blocks)+[onehot],axis=1)


def prepared():
    meta=json.loads((B/'results/prepared.json').read_text());verify_files(O,meta['source_sha256']);verify_files(B/'results',meta['artifact_sha256'])
    return pd.read_pickle(B/'results/data.pkl'),meta
