"""Frozen FinBERT feature experiment on the existing train/validation rows."""
import os,sys,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'stock_baseline'))
from run_baseline import scores
ROOT=BASE.parents[1]/'work/stock-data'
protocol=json.loads((BASE/'protocol.json').read_text())
OUT=BASE/'results';OUT.mkdir(exist_ok=True)
cache=ROOT/'finbert-cache';cache.mkdir(exist_ok=True)
torch.manual_seed(573);torch.set_num_threads(4)
frames={s:pd.read_csv(BASE.parent/f'stock_baseline/results/{s}/samples.csv').fillna({'news_ids':'','text':''}) for s in ['AAPL','AMZN']}
frames={s:d[d.split.ne('test')].reset_index(drop=True) for s,d in frames.items()}
raw_news=pd.read_pickle(ROOT/'audit/news_index.pkl')
raw_news['record_key']=raw_news.archive+'::'+raw_news.member
assert not raw_news.record_key.duplicated().any()
used=set()
for symbol,d in frames.items():
 pattern=r'\b(?:AAPL|Apple)\b' if symbol=='AAPL' else r'\b(?:AMZN|Amazon)\b'
 n=raw_news[raw_news.language.eq('english') & raw_news.chars.ge(100) & raw_news.lag_hours.ge(0)]
 n=n[n.title.fillna('').str.contains(pattern,case=False,regex=True)&~n.physician]
 n=n.sort_values(['available_utc','archive','member']).drop_duplicates('normalized_hash').drop_duplicates('title_norm')
 keys=[]
 for row in d.itertuples():
  cutoff=pd.Timestamp(row.cutoff_utc)
  w=n[n.available_utc.gt(cutoff-pd.Timedelta(hours=4))&n.available_utc.le(cutoff)]
  assert ' . '.join(w.title.fillna(''))==row.text, 'Text differs from original baseline'
  assert '|'.join(w.uuid.astype(str))==row.news_ids
  assert len(w)==row.news_count
  keys.append('|'.join(w.record_key));used.update(w.record_key)
 d['news_record_keys']=keys
news=raw_news[raw_news.record_key.isin(used)].sort_values('record_key').reset_index(drop=True)
assert set(news.record_key)==used
fingerprint=hashlib.sha256(news[['record_key','title']].to_csv(index=False).encode()).hexdigest()
print('Unique titles to encode:',len(news),flush=True)
record=dict(protocol=protocol,input_hash=fingerprint,n_titles=len(news),torch=torch.__version__)
encoded=OUT/'article_probabilities.csv'
if encoded.exists() and (OUT/'inference.json').exists() and json.loads((OUT/'inference.json').read_text())['input_hash']==fingerprint and json.loads((OUT/'inference.json').read_text())['protocol']==protocol:
 probs=pd.read_csv(encoded);record=json.loads((OUT/'inference.json').read_text())
else:
 tok=AutoTokenizer.from_pretrained(protocol['model_id'],revision=protocol['revision'],cache_dir=str(cache),trust_remote_code=False)
 model=AutoModelForSequenceClassification.from_pretrained(protocol['model_id'],revision=protocol['revision'],cache_dir=str(cache),trust_remote_code=False)
 device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device).eval()
 print('Device:',device,'labels:',model.config.id2label,flush=True)
 output=[];truncated=0;started=time.time()
 for i in range(0,len(news),32):
  titles=news.title.iloc[i:i+32].fillna('').tolist()
  lengths=tok(titles,truncation=False,add_special_tokens=True)['input_ids'];truncated+=sum(len(x)>256 for x in lengths)
  batch=tok(titles,padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
  with torch.inference_mode(): p=model(**batch).logits.softmax(-1).cpu().numpy()
  output.append(p)
  if i%640==0:print(f'Encoded {min(i+32,len(news))}/{len(news)} ({time.time()-started:.0f}s)',flush=True)
 probs=news[['record_key','uuid','title','available_utc']].copy();arr=np.concatenate(output)
 for k,v in model.config.id2label.items():probs[v.lower()]=arr[:,int(k)]
 assert np.allclose(probs[['positive','negative','neutral']].sum(axis=1),1,atol=1e-5)
 probs.to_csv(encoded,index=False)
 record.update(device=device,id2label=model.config.id2label,truncated_titles=truncated,seconds=time.time()-started)
 (OUT/'inference.json').write_text(json.dumps(record,indent=2))
lookup=probs.set_index('record_key')
old=json.loads((BASE.parent/'stock_baseline/results/results.json').read_text())
result={'protocol':protocol,'stocks':{}}
for symbol,d in frames.items():
 feature_rows=[]
 for r in d.itertuples():
  ids=[u for u in r.news_record_keys.split('|') if u];p=lookup.loc[ids] if ids else None
  if ids:
   available=news.set_index('record_key').loc[ids,'available_utc']
   assert available.max()<=pd.Timestamp(r.cutoff_utc)
   assert available.min()>pd.Timestamp(r.cutoff_utc)-pd.Timedelta(hours=4)
   f={f'finbert_{c}':float(p[c].mean()) for c in ['positive','negative','neutral']}
   f['sentiment_std']=float((p.positive-p.negative).std(ddof=0))
  else:f={k:0. for k in ['finbert_positive','finbert_negative','finbert_neutral','sentiment_std']}
  f.update(log_news_count=float(np.log1p(len(ids))),has_news=int(bool(ids)));feature_rows.append(f)
 features=pd.DataFrame(feature_rows);d=pd.concat([d,features.drop(columns=['has_news'])],axis=1)
 price=old['stocks'][symbol]['features'];sentiment=['finbert_positive','finbert_negative','finbert_neutral','sentiment_std'];count=['log_news_count','has_news']
 variants={'price_count_control':price+count,'finbert_only':sentiment+count,'price_finbert':price+sentiment+count}
 tr=d[d.split.eq('train')];va=d[d.split.eq('validation')]
 folder=OUT/symbol;folder.mkdir(exist_ok=True);d.drop(columns='text').to_csv(folder/'development_features.csv',index=False)
 preds=va[['start_utc','label','news_count']].reset_index(drop=True);grid=[];selected={}
 for kind,cols in variants.items():
  candidates=[]
  for C in protocol['C_grid']:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='liblinear',max_iter=2000,random_state=573)).fit(tr[cols],tr.label)
   p=model.predict_proba(va[cols])[:,1];m=scores(va.label,p);grid.append(dict(model=kind,C=C,**m));candidates.append((m,C,model,p))
  m,C,model,p=max(candidates,key=lambda x:(x[0]['balanced_accuracy'],-x[0]['brier']))
  selected[kind]=dict(C=C,**m);preds[kind]=p;joblib.dump({'model':model,'columns':cols},folder/f'{kind}.joblib')
  np.testing.assert_allclose(model.named_steps['standardscaler'].mean_,tr[cols].mean().to_numpy())
 pd.DataFrame(grid).to_csv(folder/'validation_grid.csv',index=False);preds.to_csv(folder/'validation_predictions.csv',index=False)
 baseline=pd.read_csv(BASE.parent/f'stock_baseline/results/{symbol}/validation_predictions.csv')
 assert baseline.start_utc.tolist()==preds.start_utc.tolist()
 assert baseline.label.tolist()==preds.label.tolist()
 # Resample whole dates, retaining all six hourly observations within each draw.
 days=pd.to_datetime(preds.start_utc).dt.strftime('%Y-%m-%d');unique=days.unique();rng=np.random.default_rng(573)
 deltas={k:[] for k in ['price','price_count_control']}
 for _ in range(1000):
  idx=np.concatenate([np.flatnonzero(days.eq(day)) for day in rng.choice(unique,len(unique),replace=True)])
  y=preds.label.to_numpy()[idx];new=scores(y,preds.price_finbert.to_numpy()[idx])['balanced_accuracy']
  for k in deltas:
   ref=baseline.price if k=='price' else preds.price_count_control
   deltas[k].append(new-scores(y,ref.to_numpy()[idx])['balanced_accuracy'])
 ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in deltas.items()}
 monthly={month:{k:scores(preds.loc[mask,'label'],preds.loc[mask,k].to_numpy()) for k in selected} for month in ['2018-09','2018-10'] if (mask:=days.str.startswith(month)).any()}
 result['stocks'][symbol]={'train':len(tr),'validation':len(va),'selected':selected,'paired_day_bootstrap_95pct_delta_BA':ci,'monthly':monthly,'validation_days':len(unique),'sample_alignment_passed':True}
 print(symbol,json.dumps(selected),flush=True)
(OUT/'results.json').write_text(json.dumps(result,indent=2))
print('Done. No final-test inference or evaluation.',flush=True)
