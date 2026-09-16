"""Rebuild past-only inputs and seal all 1,607 original windows before inference."""
import argparse,json,re,zipfile
import numpy as np,pandas as pd
from common import B,ROOT,W,sha,dump,validate_boundary
PRICE=[f'{v}_{k}' for k in range(1,7) for v in ['return','range']]+['history_age_hours','return_mean','return_std','ny_hour']
NEWSNUM=['included_news','omitted_news','publication_age_mean','collection_delay_mean','unknown_publication']
def excerpt(text,symbol):
 spans=[(0,min(len(text),200))] if text else []
 m=re.search(r'\b(?:Apple|AAPL)\b' if symbol=='AAPL' else r'\b(?:Amazon|AMZN)\b',text,re.I)
 if m:
  start=max(0,m.start()-100)
  if start<200:start=200
  if start<len(text):spans.append((start,min(len(text),start+400)))
 elif len(text)>200:spans.append((200,min(len(text),600)))
 return [{'start':a,'end':b,'text':text[a:b]} for a,b in spans if b>a]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=__import__('pathlib').Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'outputs/stock_integrated_4h/prepared';old=json.loads((source/'manifest.json').read_text())
 for name,value in old['prepared_hashes'].items():assert sha(source/name)==value
 for name,value in old['archive_hashes'].items():assert sha(name)==value
 d=pd.read_pickle(source/'data.pkl').copy();d['key']=d.symbol+'|'+d.start_utc.astype(str);d=d.sort_values(['start_utc','symbol']).reset_index(drop=True)
 assert len(d)==1607 and not d.key.duplicated().any()
 raw=pd.read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key')
 keys=sorted({k for s in d.news_record_keys.fillna('') for k in s.split('|') if k});bodies={}
 for arc,g in raw.loc[keys].groupby('archive'):
  with zipfile.ZipFile(W/'raw/news'/arc) as z:
   for r in g.itertuples():bodies[r.Index]=json.loads(z.read(r.member)).get('text','') or ''
 cal=pd.read_csv(W/'audit/xnys_schedule.csv');cal['open']=pd.to_datetime(cal.open,utc=True);cal['close']=pd.to_datetime(cal.close,utc=True)
 histories={};sources={str(source/'data.pkl'):sha(source/'data.pkl'),str(W/'audit/news_index.pkl'):sha(W/'audit/news_index.pkl'),str(W/'audit/xnys_schedule.csv'):sha(W/'audit/xnys_schedule.csv'),**old['archive_hashes']}
 for sym,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
  p=W/f'raw/CHARTS/{prefix}5.csv';sources[str(p)]=sha(p)
  x=pd.read_csv(p,header=None,names=['date','time','open','high','low','close','activity']);x.index=pd.to_datetime(x.date+' '+x.time,format='%Y.%m.%d %H:%M',utc=True);x=x.sort_index();assert not x.index.duplicated().any()
  bs=[]
  for c in cal.itertuples():
   if c.open<x.index.min() or c.open>x.index.max():continue
   for start in pd.date_range(c.open,c.close-pd.Timedelta('1h'),freq='h'):
    end=start+pd.Timedelta('1h');q=x.reindex(pd.date_range(start,end-pd.Timedelta('5min'),freq='5min'))
    if q[['open','high','low','close']].isna().any().any():continue
    bs.append(dict(start=start,end=end,ret=np.log(q.close.iloc[-1]/q.open.iloc[0]),rng=(q.high.max()-q.low.min())/q.open.iloc[0]))
  histories[sym]=pd.DataFrame(bs)
  for r in d[d.symbol.eq(sym)].itertuples():
   q=x.reindex(pd.date_range(r.start_utc,r.end_utc-pd.Timedelta('5min'),freq='5min'));assert len(q)==48 and not q[['open','close']].isna().any().any()
   assert int(q.close.iloc[-1]>q.open.iloc[0])==r.label
 inputs=[];features=[];coverage=[]
 for r in d.itertuples():
  hist=histories[r.symbol];hist=hist[hist.end<=r.cutoff_utc].tail(6);assert len(hist)==6
  for k,h in enumerate(hist.iloc[::-1].itertuples(),1):
   np.testing.assert_allclose([h.ret,h.rng],[getattr(r,f'return_{k}'),getattr(r,f'range_{k}')],atol=1e-12)
  assert hist.end.iloc[-1]==r.history_end
  pr=[dict(id=f'P{i}',start=h.start.isoformat(),end=h.end.isoformat(),log_return_pct=round(float(h.ret)*100,6),range_pct=round(float(h.rng)*100,6)) for i,h in enumerate(hist.itertuples(),1)]
  kk=[k for k in str(r.news_record_keys).split('|') if k and k!='nan'];articles=raw.loc[kk].sort_values(['available_utc','key'] if 'key' in raw.columns else ['available_utc'],ascending=False) if kk else raw.iloc[:0]
  # Stable explicit key tie break; never rank using future outcomes.
  chosen=sorted(kk,key=lambda k:(-raw.loc[k,'available_utc'].value,k))[:6]
  if kk:
   assert ((articles.available_utc<=r.cutoff_utc)&(articles.available_utc>r.cutoff_utc-pd.Timedelta('4h'))).all()
  news=[];ages=[];delays=[]
  for j,k in enumerate(chosen,1):
   z=raw.loc[k];pub=z.published_utc;pub=None if pd.isna(pub) else pub
   if pub is not None:assert pub<=z.available_utc
   spans=excerpt(bodies[k],r.symbol);title=str(z.title) if pd.notna(z.title) else ''
   age=(r.cutoff_utc-pub).total_seconds()/3600 if pub is not None else None;lag=(z.available_utc-pub).total_seconds()/3600 if pub is not None else None
   if age is not None:ages.append(age);delays.append(lag)
   news.append(dict(id=f'N{j}',record_key=k,title=title[:240],published_at=pub.isoformat() if pub is not None else None,available_at=z.available_utc.isoformat(),publication_age_hours=round(age,6) if age is not None else None,collection_delay_hours=round(lag,6) if lag is not None else None,passages=spans,body_chars_omitted=len(bodies[k])-sum(v['end']-v['start'] for v in spans),title_chars_omitted=max(0,len(title)-240)))
  item=dict(key=r.key,symbol=r.symbol,cutoff=r.cutoff_utc.isoformat(),interval_start=r.start_utc.isoformat(),interval_end=r.end_utc.isoformat(),timezone='UTC; regular session America/New_York',price_rows=pr,price_summary=dict(mean_log_return_pct=float(r.return_mean)*100,std_log_return_pct=float(r.return_std)*100,history_age_hours=float(r.history_age_hours),target_start_ny_hour=float(r.ny_hour)),news=news,news_summary=dict(eligible_articles=len(kk),included_articles=len(news),omitted_articles=len(kk)-len(news)))
  validate_boundary(item);inputs.append(item)
  f={k:float(getattr(r,k)) for k in PRICE};f.update(key=r.key,symbol=r.symbol,split=r.split,day=r.day,month=str(r.start_utc)[:7],start_utc=r.start_utc,end_utc=r.end_utc,cutoff_utc=r.cutoff_utc,label=int(r.label),has_news=int(bool(kk)),included_news=len(news),omitted_news=len(kk)-len(news),publication_age_mean=float(np.mean(ages)) if ages else 0.,collection_delay_mean=float(np.mean(delays)) if delays else 0.,unknown_publication=len(news)-len(ages),packed_text=' '.join(n['title']+' '+ ' '.join(p['text'] for p in n['passages']) for n in news));features.append(f)
  coverage.append(dict(key=r.key,symbol=r.symbol,split=r.split,eligible=len(kk),included=len(news),omitted=len(kk)-len(news),body_chars_included=sum(len(p['text']) for n in news for p in n['passages']),body_chars_omitted=sum(n['body_chars_omitted'] for n in news)))
 with (a.out/'inputs.jsonl').open('x') as f:
  for r in inputs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 pd.DataFrame(features).to_pickle(a.out/'features.pkl');pd.DataFrame(coverage).to_csv(a.out/'coverage.csv',index=False)
 dump(a.out/'manifest.json',dict(n=len(d),split_counts={f'{s}|{p}':int(n) for (s,p),n in d.groupby(['symbol','split']).size().items()},input_sha=sha(a.out/'inputs.jsonl'),features_sha=sha(a.out/'features.pkl'),coverage_sha=sha(a.out/'coverage.csv'),sources=sources,code={p.name:sha(p) for p in B.glob('*.py')},protocol_sha=sha(B/'PROTOCOL.md'),historical_price_and_label_parity=True))
 print('SEALED',len(d),'samples; eval',sum(d.split!='train'),flush=True)
if __name__=='__main__':main()
