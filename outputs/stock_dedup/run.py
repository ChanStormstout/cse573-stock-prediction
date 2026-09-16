import sys,json
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;OUT=B/'results';sys.path.insert(0,str(B.parent/'stock_content'));from experiment import fit_variant
prior=json.loads((B.parent/'stock_content/results/results.json').read_text());texts=pd.read_json(B.parent/'stock_content/results/article_texts.jsonl',lines=True);probs=pd.read_csv(B.parent/'stock_content/results/text_probabilities.csv');clusters=pd.read_csv(OUT/'online_clusters.csv').fillna({'site':''})
result={'protocol':json.loads((B/'protocol.json').read_text()),'stocks':{}}
for s in ['AAPL','AMZN']:
 selected=max(prior['stocks'][s],key=lambda k:prior['stocks'][s][k]['training_cv_BA']);variant,family=selected.split('_');print('Selected on training CV',s,selected,flush=True)
 t=texts[texts.symbol.eq(s)].set_index('record_key');p=probs[probs.symbol.eq(s)&probs.variant.eq(variant)].set_index('record_key');c=clusters[clusters.symbol.eq(s)].set_index('record_key');d=pd.read_csv(B.parent/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''})
 result['stocks'][s]={'selected_e05':selected,'models':{},'counts':{}}
 for name in ['article_equal_control','article_equal_with_counts','cluster_equal','cluster_equal_with_counts']:
  rows=[];strings=[];unit_total={split:0 for split in ['train','validation']};article_total={split:0 for split in ['train','validation']}
  for r in d.itertuples():
   keys=[k for k in r.news_record_keys.split('|') if k];w=c.loc[keys].copy()
   if len(w):
    # Every representative is selected from the current window, not from future or older out-of-window text.
    w['parsed_time']=pd.to_datetime(w.available_utc,utc=True,format='mixed');w=w.sort_values(['parsed_time','record_key'])
    representatives=w.drop_duplicates('cluster').index.tolist();units=representatives if name.startswith('cluster') else keys
   else:units=[]
   q=p.loc[units];strings.append(' . '.join(t.loc[units,variant]) if units else '')
   f={f'p_{label}':float(q[label].mean()) if units else 0. for label in ['positive','negative','neutral']};f['p_std']=float((q.positive-q.negative).std(ddof=0)) if units else 0.;f['count']=float(np.log1p(len(keys)));f['has']=int(bool(keys))
   f['cluster_count']=float(np.log1p(w.cluster.nunique())) if keys else 0.;f['source_count']=float(np.log1p(w.site.nunique())) if keys else 0.;f['article_count']=float(np.log1p(len(keys)));f['unit_keys']='|'.join(units);rows.append(f);unit_total[r.split]+=len(units);article_total[r.split]+=len(keys)
  dd=d.copy();dd['text']=strings;dd=pd.concat([dd,pd.DataFrame(rows)],axis=1)
  extra=[] if family=='sparse' else ['p_positive','p_negative','p_neutral','p_std','count','has']
  if name.endswith('with_counts'):extra+=['cluster_count','source_count','article_count']
  result['stocks'][s]['models'][name]=fit_variant(dd,s,name,family,extra,OUT/s)
  result['stocks'][s]['counts'][name]={'article_occurrences':article_total,'unit_occurrences':unit_total}
  (OUT/'results.json').write_text(json.dumps(result,indent=2))
print('E06 complete, no test evaluation.',flush=True)
