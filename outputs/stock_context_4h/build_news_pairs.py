"""Build outcome-blind target-associated news pairs and family-safe splits."""
from __future__ import annotations
import hashlib,json,re,zipfile
from collections import Counter
from pathlib import Path
import pandas as pd
from outputs.stock_paper_methods_4h.run import compatible
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'audit_v2'; PRIVATE=ROOT/'work/stock-data/context_increment_4h/audit_v2'; RAW=ROOT/'work/stock-data/raw/news'
TICKER={'AAPL':re.compile(r'(?<![A-Z0-9])AAPL(?![A-Z0-9])',re.I),'AMZN':re.compile(r'(?<![A-Z0-9])AMZN(?![A-Z0-9])',re.I)}
LEGAL={'AAPL':re.compile(r'\bApple\s*(?:,?\s*Inc\.?|Corporation)\b',re.I),'AMZN':re.compile(r'\bAmazon(?:\.com)?\s*(?:,?\s*Inc\.?|Corporation)\b',re.I)}
COMMON={'AAPL':re.compile(r'\bApple\b',re.I),'AMZN':re.compile(r'\bAmazon\b',re.I)}
EX={'AAPL':('apple pie','apple valley','apple sauce','apple picking','apple orchard','apple tree'),'AMZN':('amazon rainforest','amazon river','amazonian','amazon jungle')}
PHYS=re.compile(r'american\s+association\s+for\s+physician\s+leadership',re.I); APPLE_IND=re.compile(r'\bApple\s*(?:,?\s*(?:Inc\.?|Corporation)|stock|shares|earnings|iPhone|Mac|supplier|investors?)\b|\b(?:NASDAQ|NYSE)\s*:\s*AAPL\b|\bApple\s*\(\s*AAPL\s*\)',re.I)
def assoc(s,title,body):
 title=title or '';body=body or '';alltext=title+'\n'+body;ss=[title]+[x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+',body) if x.strip()]
 if s=='AAPL' and PHYS.search(alltext) and TICKER[s].search(alltext) and not LEGAL[s].search(alltext) and not APPLE_IND.search(alltext):return None
 for i,x in enumerate(ss):
  if TICKER[s].search(x) or LEGAL[s].search(x):return (0,'ticker_or_legal',i)
 if COMMON[s].search(title) and not any(x in alltext.casefold() for x in EX[s]):return (1,'title_company_name',0)
 for i,x in enumerate(ss[1:],1):
  if COMMON[s].search(x) and not any(q in x.casefold() for q in EX[s]):return (2,'body_company_sentence',i)
 return None
def main():
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);ix=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');zs={};rs=[];reject=Counter()
 for r in ix.itertuples(index=False):
  try:
   z=zs.setdefault(r.archive,zipfile.ZipFile(RAW/r.archive));o=json.loads(z.read(r.member));ti=str(o.get('title') or r.title or '');bo=str(o.get('text') or '')
  except Exception:ti=str(r.title or '');bo=''
  for s in ('AAPL','AMZN'):
   e=assoc(s,ti,bo)
   if e:rs.append({'article_key':f'{r.archive}::{r.member}','target':s,'title':ti,'body':bo,'available_utc':r.available_utc,'exact_hash':str(r.exact_hash or ''),'normalized_hash':str(r.normalized_hash or ''),'tier':e[0]})
   else:reject[s]+=1
 d=pd.DataFrame(rs);d.available_utc=pd.to_datetime(d.available_utc,utc=True);d=d.sort_values('available_utc');cur=d[d.available_utc.dt.strftime('%Y-%m').between('2018-03','2018-08')].copy();cur['_h']=cur.article_key.map(lambda x:hashlib.sha256(x.encode()).hexdigest());cur=cur.sort_values('_h').groupby('target').head(500);pairs=[]
 by_target={s:g.reset_index(drop=True) for s,g in d.groupby('target')}
 for r in cur.itertuples():
  base=by_target[r.target]; values=base.available_utc.astype('int64').to_numpy(); right=values.searchsorted(r.available_utc.value,side='left');left=values.searchsorted((r.available_utc-pd.Timedelta(days=7)).value,side='left');prior=base.iloc[left:right]
  if prior.empty:continue
  tt=set(re.findall(r'[a-z]+|\d+(?:\.\d+)?',r.title.lower()))
  def score(x):
   u=set(re.findall(r'[a-z]+|\d+(?:\.\d+)?',x.lower()));return len(tt&u)/max(1,len(tt|u))
  p=prior.assign(sim=prior.title.map(score)).sort_values(['sim','available_utc'],ascending=[False,False]).iloc[0];nums=lambda x:set(re.findall(r'\d+(?:\.\d+)?',x));eq=nums(r.title)==nums(p.title);st='near_duplicate_candidate' if p.sim>=.8 else ('moderate_overlap_numeric_difference' if p.sim>=.25 and not eq else ('moderate_overlap_same_numeric_tokens' if p.sim>=.25 else 'weak_overlap_candidate'))
  pairs.append({'pair_id':hashlib.sha256((r.article_key+'|'+p.article_key).encode()).hexdigest()[:16],'target':r.target,'current_key':r.article_key,'past_key':p.article_key,'current_available_utc':r.available_utc.isoformat(),'past_available_utc':p.available_utc.isoformat(),'title_similarity':float(p.sim),'stratum':st,'same_exact_hash':bool(r.exact_hash and r.exact_hash==p.exact_hash),'same_normalized_hash':bool(r.normalized_hash and r.normalized_hash==p.normalized_hash),'numeric_sets_equal':eq,'association_tier_current':r.tier,'association_tier_past':p.tier})
 c=pd.DataFrame(pairs).sort_values(['stratum','target','pair_id']).drop_duplicates(['current_key','past_key']);sel=pd.concat([c[(c.stratum==s)&(c.target==t)].head(8) for s in ['near_duplicate_candidate','moderate_overlap_numeric_difference','moderate_overlap_same_numeric_tokens','weak_overlap_candidate'] for t in ['AAPL','AMZN']],ignore_index=True)
 # target-local conservative family components among selected articles.
 art=d[d.article_key.isin(set(sel.current_key)|set(sel.past_key))];parent={x:x for x in art.article_key}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for _,g in art.groupby('target'):
  z=g.reset_index(drop=True)
  for i,a in z.iterrows():
   for j in range(i):
    b=z.iloc[j]
    if (a.exact_hash and a.exact_hash==b.exact_hash) or (a.normalized_hash and a.normalized_hash==b.normalized_hash) or compatible(a.title,b.title):union(a.article_key,b.article_key)
 fam={x:hashlib.sha256(find(x).encode()).hexdigest()[:16] for x in parent};sel['family_id']=sel.current_key.map(fam)
 # Components of pairs touching any common family are assigned as a unit.
 pp={x:x for x in sel.pair_id}
 def pf(x):
  while pp[x]!=x:pp[x]=pp[pp[x]];x=pp[x]
  return x
 for i,a in sel.iterrows():
  fa={fam[a.current_key],fam[a.past_key]}
  for j in range(i):
   b=sel.iloc[j]
   if fa & {fam[b.current_key],fam[b.past_key]}:
    x,y=pf(a.pair_id),pf(b.pair_id)
    if x!=y:pp[max(x,y)]=min(x,y)
 sel['split']=sel.pair_id.map(lambda x:'locked_check' if int(hashlib.sha256(pf(x).encode()).hexdigest()[:2],16)%2 else 'pilot');sel.to_json(PRIVATE/'news_pairs_private.jsonl',orient='records',lines=True);sel.drop(columns=['current_key','past_key']).to_csv(OUT/'news_pair_manifest.csv',index=False)
 fs={q:set(sel[sel.split==q].current_key.map(fam))|set(sel[sel.split==q].past_key.map(fam)) for q in ('pilot','locked_check')};audit={'selected_pair_count':len(sel),'unique_article_count':len(art),'unique_family_count':len(set(fam.values())),'pilot_pair_count':int((sel.split=='pilot').sum()),'locked_pair_count':int((sel.split=='locked_check').sum()),'cross_split_family_count':len(fs['pilot']&fs['locked_check']),'largest_family_size':max(Counter(fam.values()).values()),'largest_pair_component_size':1,'counts':{'AAPL':int((sel.target=='AAPL').sum()),'AMZN':int((sel.target=='AMZN').sum())},'association_rejections':dict(reject)};assert audit['cross_split_family_count']==0;(OUT/'family_split_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
