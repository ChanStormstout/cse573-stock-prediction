"""Deterministic, outcome-blind pair manifest; retrieval is not event identity."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'audit_v1'; PRIVATE=ROOT/'work/stock-data/context_increment_4h'
def tok(s):return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?",str(s).lower()))
def main():
 d=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl').copy();d['target']=d.apply(lambda r:'AAPL' if r.title_aapl else ('AMZN' if r.title_amzn else None),axis=1);d=d[d.target.notna()].sort_values('available_utc');d['month']=d.available_utc.dt.strftime('%Y-%m');current=d[d.month.between('2018-03','2018-08')].copy();current['_sample']=current.uuid.map(lambda x:hashlib.sha256(str(x).encode()).hexdigest());current=current.sort_values('_sample').groupby('target').head(1500);by_target={s:g.reset_index(drop=True) for s,g in d.groupby('target')};rows=[]
 for r in current.itertuples():
  base=by_target[r.target];times=base.available_utc.astype('int64').to_numpy();right=times.searchsorted(r.available_utc.value,side='left');left=times.searchsorted((r.available_utc-pd.Timedelta(days=7)).value,side='left');prior=base.iloc[left:right]
  if prior.empty:continue
  t=tok(r.title);prior=prior.assign(sim=prior.title.map(lambda x:len(t&tok(x))/max(1,len(t|tok(x)))))
  p=prior.sort_values(['sim','available_utc'],ascending=[False,False]).iloc[0];numbers=lambda s:set(re.findall(r'\d+(?:\.\d+)?',str(s)));same_numbers=numbers(r.title)==numbers(p.title)
  if p.sim>=.8: strata='exact_or_near_repetition'
  elif p.sim>=.25 and not same_numbers: strata='similar_differing_numbers_actions'
  elif p.sim>=.25: strata='similar_target_different_actor_period'
  else:strata='weak_ambiguous'
  group=hashlib.sha256((str(r.normalized_hash)+'|'+str(p.normalized_hash)).encode()).hexdigest();rows.append({'pair_id':hashlib.sha256((str(r.uuid)+'|'+str(p.uuid)).encode()).hexdigest()[:16],'group_id':group,'target':r.target,'current_key':f'{r.archive}::{r.member}','past_key':f'{p.archive}::{p.member}','current_available_utc':r.available_utc.isoformat(),'past_available_utc':p.available_utc.isoformat(),'similarity':float(p.sim),'stratum':strata,'split':'locked_check' if int(group[:2],16)%2 else 'pilot'})
 candidates=pd.DataFrame(rows).sort_values(['stratum','target','pair_id']).drop_duplicates('group_id');take=[]
 for s in ['exact_or_near_repetition','similar_differing_numbers_actions','similar_target_different_actor_period','weak_ambiguous']:
  x=candidates[candidates.stratum.eq(s)];take.append(x.head(16))
 selected=pd.concat(take).drop_duplicates('group_id').head(64)
 PRIVATE.mkdir(parents=True,exist_ok=True);selected.to_json(PRIVATE/'news_pairs_private.jsonl',orient='records',lines=True);selected.drop(columns=['current_key','past_key']).to_csv(OUT/'news_pair_manifest.csv',index=False)
 (OUT/'NEWS_PAIR_PROTOCOL.md').write_text('# Pair protocol\n\n64 (or fewer if strata are unavailable) distinct, outcome-blind pairs use same-target seven-day lexical retrieval. Retrieval is not event identity. Split is deterministic SHA group parity; independent review is `QUALITY_UNVERIFIED`. Current/past full passages remain private.\n')
 print(json.dumps({'pairs':len(selected),'by_stratum':selected.stratum.value_counts().to_dict(),'by_split':selected.split.value_counts().to_dict()},indent=2))
if __name__=='__main__':main()
