"""Extract target text with exact raw character spans. No stock outcome labels used."""
import re,json,hashlib,zipfile
from pathlib import Path
import pandas as pd,numpy as np
from transformers import AutoTokenizer
B=Path(__file__).resolve().parent;ROOT=B.parents[1]/'work/stock-data';OUT=B/'results';OUT.mkdir(exist_ok=True)
if any(OUT.iterdir()) or (B/'quality_160.csv').exists():raise RuntimeError('Preserve existing E05 inputs and reviewed quality labels. Prepare a new experiment version.')
tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',revision='4556d13015211d73dccd3fdd39d39232506f3e43',cache_dir=str(ROOT/'finbert-cache'),local_files_only=True)
raw=pd.read_pickle(ROOT/'audit/news_index.pkl');raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key')
uses={};trainuses={}
for s in ['AAPL','AMZN']:
 d=pd.read_csv(B.parent/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''})
 uses[s]=set(k for x in d.news_record_keys for k in x.split('|') if k);trainuses[s]=set(k for x in d.loc[d.split.eq('train'),'news_record_keys'] for k in x.split('|') if k)
needed=set.union(*uses.values());texts={}
for archive,g in raw.loc[sorted(needed)].groupby('archive'):
 with zipfile.ZipFile(ROOT/'raw/news'/archive) as z:
  for r in g.itertuples():texts[r.Index]=json.loads(z.read(r.member)).get('text','')
 print('Read',archive,len(g),flush=True)
# Sentence boundaries skip common abbreviations; offsets always refer to original body.
ABBR=re.compile(r'(?:Inc|Corp|Co|Ltd|Mr|Ms|Dr|vs|U\.S|U\.K)\.$',re.I)
BAD=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|all done|read more|advertisement|copyright|subscribe|click here',re.I)
FIN=re.compile(r'earnings|revenue|profit|guidance|price target|upgrad|downgrad|reiterat|overweight|underweight|lawsuit|sued|acquis|buyback|recall|forecast|demand|sales|dividend',re.I)
ALIASES={'AAPL':re.compile(r'\b(?:AAPL|Apple)\b',re.I),'AMZN':re.compile(r'\b(?:AMZN|Amazon(?:\.com)?)\b',re.I)}
def sentences(body):
 starts=0;out=[]
 for m in re.finditer(r'\n+|(?<=[.!?])\s+',body):
  if '\n' not in m.group() and ABBR.search(body[max(starts,m.start()-12):m.start()]):continue
  a,b=starts,m.start();starts=m.end()
  while a<b and body[a].isspace():a+=1
  while b>a and body[b-1].isspace():b-=1
  if b>a:out.append((a,b,body[a:b]))
 if starts<len(body):
  a=starts
  while a<len(body) and body[a].isspace():a+=1
  if a<len(body):out.append((a,len(body),body[a:]))
 return out

def clip(s,n=254):
 # Offset mapping avoids decode changing the original content; split sentence at token boundary.
 enc=tok(s,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
 if len(enc['input_ids'])<=n:return s
 return s[:enc['offset_mapping'][n-1][1]]

def extract(body,title,symbol):
 ss=sentences(body);clean=[x for x in ss if len(x[2])>=35 and not BAD.search(x[2])]
 candidates=[]
 for a,b,t in clean:
  if not ALIASES[symbol].search(t):continue
  score=3*bool(re.search(r'\b'+symbol+r'\b',t,re.I))+2*bool(ALIASES[symbol].match(t))+bool(FIN.search(t))
  # Penalize gigantic unsegmented paragraphs and title echoes, while retaining them as fallbacks.
  score-=int(len(t)>1800);score-=int(t.strip().casefold()==title.strip().casefold())
  candidates.append((score,a,b,t))
 chosen=sorted(sorted(candidates,key=lambda x:(-x[0],x[1]))[:3],key=lambda x:x[1])
 if chosen:
  budget=254;pieces=[];spans=[]
  for _,a,b,t in chosen:
   if budget<8:break
   piece=clip(t,budget);used=len(tok(piece,add_special_tokens=False)['input_ids']);budget-=used
   pieces.append(piece);spans.append([a,a+len(piece)])
  target='\n'.join(pieces);fallback=False
 else:target=clip(title);spans=[];fallback=True
 target=clip(target);target_n=len(tok(target,add_special_tokens=False)['input_ids'])
 leadtext='\n'.join(t for _,_,t in clean) if clean else body
 lead=clip(leadtext,max(target_n,1)) if leadtext.strip() else clip(title,max(target_n,1))
 return dict(target=target,lead=lead,fallback=fallback,spans=spans,target_tokens=target_n,lead_tokens=len(tok(lead,add_special_tokens=False)['input_ids']),candidates=len(candidates),body_chars=len(body))
rows=[]
for s in ['AAPL','AMZN']:
 for key in sorted(uses[s]):
  r=raw.loc[key];f=extract(texts[key],r.title,s)
  rows.append(dict(symbol=s,record_key=key,title=r.title,training_used=key in trainuses[s],available_utc=str(r.available_utc),published_utc=str(r.published_utc),lag_hours=r.lag_hours,body_sha256=hashlib.sha256(texts[key].encode()).hexdigest(),**f))
d=pd.DataFrame(rows);d.to_json(OUT/'article_texts.jsonl',orient='records',lines=True,force_ascii=False)
quality=[]
for s,g in d[d.training_used].groupby('symbol'):
 g=g.copy();g['stratum']=np.select([g.title.str.contains(r'\b(?:stake|holding|position|shareholder)\b',case=False,regex=True),g.title.str.contains(';',regex=False)|(g.title.str.contains('Apple|AAPL',case=False,regex=True)&g.title.str.contains('Amazon|AMZN',case=False,regex=True)),g.body_chars.lt(800)],['holdings','multiple','short_body'],default='other')
 selected=[]
 for name,z in g.groupby('stratum'):selected.extend(z.sample(min(20,len(z)),random_state=573).index)
 if len(selected)<80:selected.extend(g.loc[~g.index.isin(selected)].sample(80-len(selected),random_state=574).index)
 q=g.loc[selected].copy();q['quality_split']='development';locked=q.sample(20,random_state=575).index;q.loc[locked,'quality_split']='locked_check';quality.append(q)
q=pd.concat(quality);q['review_status']='unreviewed';q['human_verified']=False;q['review_note']='';q.to_csv(B/'quality_160.csv',index=False)
# Keep exact raw bodies for review and verification, local only.
(OUT/'source_bodies.json').write_text(json.dumps(texts,ensure_ascii=False))
summary={'stock_article_pairs':len(d),'quality_rows':len(q),'quality_split':q.quality_split.value_counts().to_dict(),'stocks':{s:{'pairs':len(g),'fallback':int(g.fallback.sum()),'fallback_rate':float(g.fallback.mean()),'median_target_tokens':float(g.target_tokens.median()),'median_lead_tokens':float(g.lead_tokens.median()),'lead_shorter_than_target':int((g.lead_tokens<g.target_tokens).sum())} for s,g in d.groupby('symbol')}}
(OUT/'extraction_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
