"""Fit-free source and split audit; does not print predictive statistics."""
import re,json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import *

def main():
 if (PRIVATE/'manifest.json').exists():check_sources();print('existing matching preparation');return
 d,emb,ids,means=load();assert len(d)==1607 and d.row_id.nunique()==1607
 assert d.groupby('symbol').size().to_dict()=={'AAPL':803,'AMZN':804}
 assert ((d.start_utc-d.cutoff_utc)==pd.Timedelta('5min')).all()
 assert ((d.end_utc-d.start_utc)==pd.Timedelta('4h')).all()
 c=pd.read_pickle(SOURCES['canonical']);c['key']=c.symbol+'|'+pd.to_datetime(c.start_utc,utc=True).astype(str);c=c.set_index('key').loc[d.key]
 assert np.array_equal(c.label,d.label)
 for col in OLD+['stem_body','news_record_keys']:
  if col in OLD:assert np.allclose(c[col],d[col],atol=1e-12,rtol=0,equal_nan=True),col
  else:assert c[col].tolist()==d[col].tolist(),col
 manifest=json.loads(SOURCES['canonical_manifest'].read_text())
 assert sha(SOURCES['finbert'])==manifest['prepared_hashes']['articles.npz']
 assert sha(SOURCES['canonical'])==manifest['prepared_hashes']['data.pkl']
 evidence=json.loads(SOURCES['modern_evidence'].read_text());assert evidence['embeddings_sha256']==sha(SOURCES['modern'])
 assert evidence['fingerprint']['pooling']=='special_token_excluded_mean'
 raw=pd.read_pickle(SOURCES['raw_index']);raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key')
 raw['available_utc']=pd.to_datetime(raw.available_utc,utc=True)
 titles=dict(json.loads(SOURCES['titles'].read_text())['keys_and_titles'])
 for k,v in titles.items():assert v==(raw.at[k,'title'] if pd.notna(raw.at[k,'title']) else '')
 for r in d.itertuples():assert all(raw.at[k,'available_utc']<=r.cutoff_utc for k in r.news_record_keys.split('|') if k)
 pa=pd.read_csv(SOURCES['price_audit']);end=pd.to_datetime(pa.latest_used_bar_end,utc=True);cut=pd.to_datetime(pa.cutoff,utc=True);assert (end.isna()|(end<=cut)).all()
 outer=[];inner=[]
 for seed in SEEDS:
  for stock in ['AAPL','AMZN']:
   ii=np.flatnonzero(d.symbol==stock)
   for fold,(a,b) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(ii,d.iloc[ii].label)):
    tr,ev=ii[a],ii[b]
    outer.extend(dict(seed=seed,symbol=stock,fold=fold,index=int(i),row_id=d.iloc[i].row_id) for i in ev)
    for inf,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1000).split(tr,d.iloc[tr].label)):
     inner.append(dict(seed=seed,symbol=stock,fold=fold,inner_fold=inf,train=tr[it].tolist(),validation=tr[iv].tolist()))
 pd.DataFrame(outer).to_csv(PRIVATE/'outer_folds.csv',index=False);dump(PRIVATE/'inner_folds.json',inner)
 # Outcome-blind dependence components, including exact shared news and same-day cross-stock windows.
 parent=list(range(len(d)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):parent[find(i)]=find(j)
 seen={};article_owner={}
 for i,r in d.iterrows():
  day=str(r.day)
  if day in seen:union(i,seen[day])
  seen[day]=i
  for k in r.news_record_keys.split('|'):
   if not k:continue
   if k in article_owner:union(i,article_owner[k])
   article_owner[k]=i
 def sig(t):
  t=t.lower();return (set(re.findall('[a-z0-9]+',t)),tuple(re.findall(r'\d+(?:\.\d+)?',t)),tuple(bool(re.search(x,t)) for x in ['rais|upgrad|boost','lower|downgrad|cut','maintain|reiterat','initiat','deni|deny|not|refut']))
 by_guard={};links=0
 for k,i in article_owner.items():
  terms,nums,actions=sig(titles[k]);guard=(nums,actions)
  for old_terms,j in by_guard.get(guard,[]):
   if terms and len(terms&old_terms)/len(terms|old_terms)>=.8:union(i,j);links+=1
  by_guard.setdefault(guard,[]).append((terms,i))
 groups=np.array([find(i) for i in range(len(d))]);d[['row_id','symbol','day']].assign(group=groups).to_csv(PRIVATE/'groups.csv',index=False)
 class_groups=d.assign(group=groups).groupby(['symbol','label']).group.nunique();feasible=min(10,int(class_groups.min()))
 audit=dict(status='PASS',rows=len(d),stocks=d.groupby('symbol').size().to_dict(),folds=60,inner_splits=len(inner),estimator_fits=0,
  same_day_shared_and_near_duplicate_components=len(set(groups)),largest_component=int(pd.Series(groups).value_counts().max()),near_duplicate_links=links,
  grouping_class_group_counts={str(k):int(v) for k,v in class_groups.items()},grouped_max_feasible_folds=feasible if feasible>=3 else 0,
  features={'PRICE':len(OLD+RECENT),'other_joint_price':len(OLD),'PCA':16},no_saved_probabilities_imported=True,
  modern_pooling='corrected_v2_special_token_excluded_mean',external_entity_lane_untouched=True)
 dump(OUT/'INPUT_AUDIT.json',audit)
 dump(PRIVATE/'manifest.json',dict(sources=hashes(),artifacts={n:sha(PRIVATE/n) for n in ['outer_folds.csv','inner_folds.json','groups.csv']}))
 dump(OUT/'SOURCE_MANIFEST.json',dict(sources=hashes(),private_artifacts={n:sha(PRIVATE/n) for n in ['outer_folds.csv','inner_folds.json','groups.csv']}))
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
