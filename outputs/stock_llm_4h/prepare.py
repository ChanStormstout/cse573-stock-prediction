"""Prepare early-period, no-price-label input panel with exact original spans."""
import argparse,re,zipfile,collections
from common import *
import pandas as pd
ALIAS={'AAPL':r'\b(?:Apple|AAPL)\b','AMZN':r'\b(?:Amazon(?:\.com)?|AMZN)\b'}
ACTION=re.compile(r'price target|target price|upgrad|downgrad|reiterat|reaffirm|\brating\b|\braised\b|\blowered\b',re.I)
BAD=re.compile(r'privacy policy|subscribe|newsletter|copyright|sign up',re.I)

def split_sentences(body):
 # Keep explicit raw offsets and punctuation; abbreviation handling protects common company suffixes.
 starts=[0]
 for m in re.finditer(r'\n+|(?<=[.!?])\s+(?=[A-Z])',body):
  if '\n' not in m.group() and re.search(r'(?:Inc|Co|Corp|Ltd|Mr|U\.S)\.$',body[max(0,m.start()-10):m.start()]):continue
  starts.append(m.end())
 spans=[]
 for a,b in zip(starts,starts[1:]+[len(body)]):
  t=body[a:b];l=len(t)-len(t.lstrip());r=len(t.rstrip());a+=l;b=a+r-l
  if b>a:spans.append((a,b,body[a:b]))
 return spans

def select_input(symbol,title,body):
 ss=split_sentences(body);valid=[i for i,(_,_,t) in enumerate(ss) if 30<=len(t)<=1500 and not BAD.search(t)]
 target=[i for i in valid if re.search(ALIAS[symbol],ss[i][2],re.I)]
 # Headline/lead plus top actionable target mentions with neighbours. Content-only, no outcomes.
 ranked=sorted(target,key=lambda i:(-int(bool(ACTION.search(ss[i][2]))),i))[:3]
 ids=set(valid[:2])
 for i in ranked:
  ids.update(j for j in [i-1,i,i+1] if j in valid)
 sentences={'S0':title};mapping={'S0':{'source':'title','start':0,'end':len(title)}};chars=len(title)
 for i in sorted(ids):
  a,b,t=ss[i]
  if chars+len(t)>4300:continue
  k=f'S{len(sentences)}';sentences[k]=t;mapping[k]={'source':'body','start':a,'end':b};chars+=len(t)
 return sentences,mapping

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=new_run(args.out)
 p=ROOT/'outputs/stock_adaptive_4h/coverage/v3/article_candidates.jsonl';d=pd.read_json(p,lines=True)
 d=d[(d.available_utc>='2018-01-01')&(d.available_utc<'2018-03-01')].copy()
 d['period']=d.available_utc.map(lambda x:'train' if x<'2018-02-01' else 'valid' if x<'2018-02-15' else 'check')
 d['stratum']=d.apply(lambda r:'action_title' if ACTION.search(r.title) and re.search(ALIAS[r.symbol],r.title,re.I) else 'other',axis=1)
 # Same underlying article stays in one group, plus existing past-only event groups.
 d=d.sort_values(['available_utc','record_key']).drop_duplicates(['symbol','record_key']).drop_duplicates(['symbol','event_group'])
 chosen=[];seen_titles=[];seen_records=set()
 for period,npos,nneg in [('train',16,8),('valid',6,2),('check',6,2)]:
  for stratum,n in [('action_title',npos),('other',nneg)]:
   x=d[(d.period==period)&(d.stratum==stratum)].copy();x['order']=x.record_key.map(lambda k:digest(k.encode()));x=x.sort_values('order')
   for sym in ['AMZN','AAPL']:
    limit=n//2 if sym=='AMZN' else n-sum(r['period']==period and r['stratum']==stratum for r in chosen)
    added=0
    for r in x[x.symbol==sym].to_dict('records'):
     if r['record_key'] in seen_records:continue
     ts=set(re.findall('[a-z]+',r['title'].lower()))
     if any(len(ts&s)/max(1,len(ts|s))>=.72 for s in seen_titles):continue
     chosen.append(r);seen_titles.append(ts);seen_records.add(r['record_key']);added+=1
     if added>=limit:break
 result=[];archives={}
 try:
  for i,r in enumerate(chosen):
   archive,member=r['record_key'].split('::')
   if archive not in archives:archives[archive]=zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive)
   body=json.loads(archives[archive].read(member)).get('text','');sentences,mapping=select_input(r['symbol'],r['title'],body)
   result.append({'id':f'L{i:02d}','symbol':r['symbol'],'record_key':r['record_key'],'group':r['event_group'],'split':r['period'],'stratum':r['stratum'],'available_utc':r['available_utc'],'published_utc':r['published_utc'],'title':r['title'],'sentences':sentences,'spans':mapping,'body_sha256':digest(body.encode()),'body':body})
 finally:
  for z in archives.values():z.close()
 write_rows(out/'inputs.jsonl',result)
 dump(out/'manifest.json',{'source_sha256':file_sha(p),'code_sha256':file_sha(__file__),'input_sha256':file_sha(out/'inputs.jsonl'),'n':len(result),'counts':dict(collections.Counter(r['split']+'|'+r['symbol']+'|'+r['stratum'] for r in result)),'no_price_labels':True,'status':'UNLABELLED'})
 print(json.dumps(json.loads((out/'manifest.json').read_text()),indent=2))
 for r in result:
  print('\n'+r['id']+' '+r['split']+' '+r['symbol']+' '+r['available_utc'])
  for k,t in r['sentences'].items():print(k+': '+t)
if __name__=='__main__':main()
