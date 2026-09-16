from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib
B=Path(__file__).resolve().parent;old=B.parent/'stock_baseline/results';out=B/'results'
r=json.loads((old/'results.json').read_text());protocol=json.loads((B/'protocol.json').read_text())
meta=json.loads((out/'inference.json').read_text());assert meta['protocol']==protocol
probs=pd.read_csv(out/'article_probabilities.csv').set_index('record_key')
assert not probs.index.duplicated().any()
assert probs[['positive','negative','neutral']].ge(0).all().all()
np.testing.assert_allclose(probs[['positive','negative','neutral']].sum(axis=1),1,atol=1e-5)
news=pd.read_pickle(B.parents[1]/'work/stock-data/audit/news_index.pkl')
news['record_key']=news.archive+'::'+news.member
news=news.set_index('record_key')
used=set()
for s in ['AAPL','AMZN']:
 raw=old/s/'samples.csv';assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['stocks'][s]['samples_sha256']
 original=pd.read_csv(raw).fillna({'news_ids':'','news_record_keys':''});d=pd.read_csv(out/s/'development_features.csv').fillna({'news_ids':'','news_record_keys':''})
 assert d.split.isin(['train','validation']).all()
 assert d.start_utc.tolist()==original.loc[original.split.ne('test'),'start_utc'].tolist()
 for row in d.itertuples():
  ids=[u for u in row.news_record_keys.split('|') if u];used.update(ids)
  assert len(ids)==row.news_count
  if ids:
   times=news.loc[ids,'available_utc'];cutoff=pd.Timestamp(row.cutoff_utc)
   assert times.le(cutoff).all() and times.gt(cutoff-pd.Timedelta(hours=4)).all()
  expected_std=float((probs.loc[ids,'positive']-probs.loc[ids,'negative']).std(ddof=0)) if ids else 0
  np.testing.assert_allclose(row.sentiment_std,expected_std,atol=1e-7)
  for c in ['positive','negative','neutral']:
   expected=probs.loc[ids,c].mean() if ids else 0
   np.testing.assert_allclose(getattr(row,'finbert_'+c),expected,atol=1e-7)
 for kind in protocol['models']:
  bundle=joblib.load(out/s/f'{kind}.joblib');cols=bundle['columns'];model=bundle['model']
  assert not set(cols)&{'label','target_return','end_utc','text','start_utc'}
  np.testing.assert_allclose(model.named_steps['standardscaler'].mean_,d.loc[d.split.eq('train'),cols].mean(),atol=1e-10)
assert used==set(probs.index)
print('PASS: baseline files unchanged; identical development rows; no test rows encoded; normalized sentiment probabilities; all per-window means; train-only scaling; no outcome features.')
