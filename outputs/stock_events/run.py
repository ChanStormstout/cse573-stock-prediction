import json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from events import EVENTS,rule_event
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B.parent/'stock_content'));from experiment import fit_variant
OUT=B/'results';a=pd.read_json(B.parent/'stock_content/results/article_texts.jsonl',lines=True)
rows=[]
for r in a.itertuples():
 event,clause=rule_event(r.title,r.symbol);rows.append(dict(symbol=r.symbol,record_key=r.record_key,title=r.title,event_type=event,evidence=clause))
events=pd.DataFrame(rows);events.to_csv(OUT/'article_events.csv',index=False)
result=dict(protocol=json.loads((B/'protocol.json').read_text()),rule_sha256=hashlib.sha256((B/'events.py').read_bytes()).hexdigest(),stocks={})
for s in ['AAPL','AMZN']:
 d=pd.read_csv(B.parent/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''});lookup=events[events.symbol.eq(s)].set_index('record_key');matrix=[]
 for r in d.itertuples():
  keys=[k for k in r.news_record_keys.split('|') if k];counts=lookup.loc[keys,'event_type'].value_counts() if keys else {}
  matrix.append([counts.get(event,0)/len(keys) if keys else 0. for event in EVENTS])
 cols=['event_'+x for x in EVENTS];d[cols]=matrix
 assert set(d.split)<= {'train','validation'}
 result['stocks'][s]={'event_article_counts':lookup.event_type.value_counts().to_dict(),'models':{}}
 count=['log_news_count','has_news'];sent=['finbert_positive','finbert_negative','finbert_neutral','sentiment_std']
 for name,extra in [('price_count',count),('price_finbert',sent+count),('price_count_events',count+cols),('price_finbert_events',sent+count+cols)]:
  result['stocks'][s]['models'][name]=fit_variant(d,s,name,'sentiment',extra,OUT/s)
  (OUT/'results.json').write_text(json.dumps(result,indent=2))
print('E07-R finished; final test not evaluated.',flush=True)
