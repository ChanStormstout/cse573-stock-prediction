"""Outcome-blind full archive metadata audit; aggregates only, never exports text."""
from __future__ import annotations
import json, zipfile
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; RAW=ROOT/'work/stock-data/raw/news'; OUT=HERE/'audit_v1'
FIELDS=['thread.uuid','thread.url','thread.site','thread.site_type','thread.published','ord_in_thread','uuid','url','published','crawled','updated','entities.organizations','organizations','persons','highlightText','highlightTitle','external_links']
def get(x,path):
 for p in path.split('.'):
  if not isinstance(x,dict):return None
  x=x.get(p)
 return x
def main():
 OUT.mkdir(parents=True,exist_ok=True); stats={f:Counter() for f in FIELDS}; types={f:Counter() for f in FIELDS}; unique={f:set() for f in FIELDS};threads=defaultdict(lambda:{'n':0,'domains':set()});eq_uuid=eq_url=ord0=n=relations=0
 for archive in sorted(RAW.glob('*.zip')):
  with zipfile.ZipFile(archive) as z:
   for name in z.namelist():
    if not name.endswith('.json'):continue
    try:x=json.loads(z.read(name))
    except Exception:continue
    n+=1;thread=get(x,'thread') or {};tu=get(x,'thread.uuid');pu=get(x,'uuid');tv=get(x,'thread.url');pv=get(x,'url');eq_uuid+=bool(tu and tu==pu);eq_url+=bool(tv and tv==pv);ord0+=get(x,'ord_in_thread')==0
    if tu:threads[str(tu)]['n']+=1;threads[str(tu)]['domains'].add(str(get(x,'thread.site') or get(x,'site') or ''))
    for f in FIELDS:
     v=get(x,f);stats[f]['present']+=v is not None;stats[f]['nonempty']+=v not in (None,'',[],{});types[f][type(v).__name__]+=1
     if v not in (None,'',[],{}) and len(unique[f])<100001:unique[f].add(json.dumps(v,sort_keys=True,default=str)[:500])
    for a,b in [('published','crawled'),('published','updated'),('thread.published','published')]:
     if get(x,a) is not None and get(x,b) is not None:relations+=1
 rows=[{'field':f,'records':n,'present_fraction':stats[f]['present']/n if n else 0,'nonempty_fraction':stats[f]['nonempty']/n if n else 0,'type_counts':json.dumps(types[f],sort_keys=True),'capped_distinct_count':len(unique[f])} for f in FIELDS]
 pd.DataFrame(rows).to_csv(OUT/'news_field_coverage.csv',index=False)
 sizes=Counter(v['n'] for v in threads.values()); domains=Counter(len(v['domains']) for v in threads.values())
 conclusion='PROVIDER_UNCONFIRMED; thread semantics are unresolved for cross-publisher financial events. Page/conversation metadata cannot establish event identity; ord_in_thread and highlights are not used as event/summary features.'
 (OUT/'NEWS_METADATA_SEMANTICS.md').write_text('# News metadata semantics audit\n\n'+conclusion+'\n\n'+f'- Records scanned: {n}\n- thread.uuid equals post uuid: {eq_uuid/n:.4%}\n- thread.url equals post url: {eq_url/n:.4%}\n- ord_in_thread == 0: {ord0/n:.4%}\n- thread IDs observed: {len(threads)}\n- thread-size distribution: `{dict(sizes)}`\n- source-domain-count distribution: `{dict(domains)}`\n- timestamp relations observed: {relations}\n')
 (OUT/'news_schema_summary.json').write_text(json.dumps({'records':n,'thread_uuid_equals_post_uuid':eq_uuid,'thread_url_equals_post_url':eq_url,'ord_zero':ord0,'thread_count':len(threads),'conclusion':conclusion},indent=2)+'\n')
if __name__=='__main__':main()
