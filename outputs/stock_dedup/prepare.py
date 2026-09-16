import re,json,difflib
from pathlib import Path
import pandas as pd
B=Path(__file__).resolve().parent;ROOT=B.parents[1]/'work/stock-data';OUT=B/'results';OUT.mkdir(exist_ok=True)
a=pd.read_json(B.parent/'stock_content/results/article_texts.jsonl',lines=True)
n=pd.read_pickle(ROOT/'audit/news_index.pkl');n['record_key']=n.archive+'::'+n.member;n=n.set_index('record_key')
HOLD=re.compile(r'\b(?:holding|holdings|stake|position|shareholder)\b',re.I)
DIRECTION=re.compile(r'\b(?:up|down|rise|fall|rises|falls|raised|lowered|increased|decreased|cut|buy|sell|bought|sold|upgrade|downgrade)\b',re.I)
def tickers(t):
 return set(re.findall(r'\b[A-Z]{2,5}\b',t))-{'NYSE','USA','CEO','CFO','SEC','DOJ','US','UK','SI','EPS'}
def norm(t):
 t=t.casefold().replace('’',"'");t=re.sub(r'\s+(?:-|–|—|\|)\s+(?:marketwatch|daily political|the.*?daily|reuters|yahoo finance|business insider|barron.?s|key gazette|nasdaq).*$', '',t)
 return ' '.join(re.findall(r'[a-z0-9]+',t))
rows=[];pairs=[]
for s,g in a.groupby('symbol'):
 g=g.copy();g['time']=pd.to_datetime(g.available_utc,utc=True,format="mixed");g=g.sort_values(['time','record_key']);past=[];clusters={}
 for r in g.itertuples():
  t=norm(r.title);nums=set(re.findall(r'\b\d+\b',t));directions=set(DIRECTION.findall(t));past=[x for x in past if x['time']>=r.time-pd.Timedelta(hours=24)];best=None
  for x in past:
   ts=set(t.split());xs=set(x['norm'].split())
   if len(ts&xs)/max(1,len(ts|xs))<.45:continue
   sim=difflib.SequenceMatcher(None,t,x['norm'],autojunk=False).ratio()
   if sim<.80:continue
   guard=nums==x['nums'] and directions==x['directions'] and tickers(r.title)==tickers(x['title']) and not ((HOLD.search(t) or HOLD.search(x['norm'])) and t!=x['norm'])
   accept=sim>=.95 and guard
   if r.training_used and x['training_used']:
    pairs.append({'symbol':s,'new_key':r.record_key,'old_key':x['key'],'new_title':r.title,'old_title':x['title'],'similarity':sim,'guards_pass':guard,'accepted':accept})
   if accept and (best is None or sim>best[0]):best=(sim,x)
  cluster=best[1]['cluster'] if best else r.record_key
  rows.append({'symbol':s,'record_key':r.record_key,'cluster':cluster,'parent':best[1]['key'] if best else '', 'similarity':best[0] if best else 1.,'available_utc':str(r.time),'normalized_title':t,'site':n.loc[r.record_key,'site']})
  past.append({'key':r.record_key,'time':r.time,'norm':t,'nums':nums,'directions':directions,'cluster':cluster,'title':r.title,'training_used':r.training_used})
pd.DataFrame(rows).to_csv(OUT/'online_clusters.csv',index=False);p=pd.DataFrame(pairs).drop_duplicates(['symbol','new_key','old_key']);p.to_csv(OUT/'all_training_candidate_pairs.csv',index=False)
q=[]
for accepted,g in p.groupby('accepted'):q.append(g.sample(min(50,len(g)),random_state=573))
quality=pd.concat(q);remaining=p.drop(quality.index)
if len(quality)<100:quality=pd.concat([quality,remaining.sample(min(100-len(quality),len(remaining)),random_state=574)])
quality['review_status']='unreviewed';quality.to_csv(B/'quality_100_pairs.csv',index=False)
print(pd.DataFrame(rows).groupby('symbol').agg(articles=('record_key','size'),clusters=('cluster','nunique')))
print('quality pairs',len(quality),'accepted',quality.accepted.sum(),flush=True)
