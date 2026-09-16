from core import *
from trading import raw_prices
import re,zipfile

def groups_online(n):
    """Greedy online title grouping. Fixed 72h history, numeric and ticker guards."""
    history=[];groups=[]
    for r in n.itertuples():
        text=str(r.title).lower();tokens=set(re.findall(r'[a-z]+|\d+(?:\.\d+)?',text));nums=set(re.findall(r'\d+(?:\.\d+)?',text));t=r.available_utc
        history=[v for v in history if t-v[0]<=pd.Timedelta(hours=72)];group=r.key
        for _,previous,pnums,g in reversed(history):
            if nums!=pnums:continue
            if len(tokens&previous)/max(1,len(tokens|previous))>=.7:group=g;break
        history.append((t,tokens,nums,group));groups.append(group)
    return groups

def timing(d,out):
    n=pd.read_pickle(W/'audit/news_index.pkl');n['key']=n.archive+'::'+n.member;n=n.set_index('key',drop=False);cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d');rows=[];articles=[]
    for stock,s in d.groupby('symbol'):
        used=sorted({k for x in s.news_record_keys.fillna('') for k in x.split('|') if k});q=n.loc[used].reset_index(drop=True).sort_values(['available_utc','key']);q['event_group']=groups_online(q);m=raw_prices(stock)
        for r in q.itertuples(index=False):
            available=r.available_utc;anchor=available.ceil('5min');day=anchor.strftime('%Y-%m-%d');intraday=False
            if day in cal.index:
                op,cl=pd.to_datetime(cal.loc[day,'open'],utc=True),pd.to_datetime(cal.loc[day,'close'],utc=True);intraday=op<=available<cl
            context='intraday' if intraday else 'outside_session';late=r.lag_hours>1
            base=dict(symbol=stock,key=r.key,event_group=r.event_group,published_utc=r.published_utc,available_utc=available,anchor=anchor,alignment_wait_minutes=(anchor-available).total_seconds()/60,context=context,late=late,crawl_delay_hours=r.lag_hours,title=r.title)
            articles.append(base)
            for horizon in [-60,15,30,60,120]:
                start=anchor+pd.Timedelta(minutes=horizon) if horizon<0 else anchor;end=anchor if horizon<0 else anchor+pd.Timedelta(minutes=horizon)
                stamps=pd.date_range(start,end-pd.Timedelta(minutes=5),freq='5min');x=m.reindex(stamps);valid=intraday and start>=op and end<=cl and not x[['open','close']].isna().any().any()
                ret=float(x.close.iloc[-1]/x.open.iloc[0]-1) if valid else None
                vol=float(np.sqrt(np.sum(np.diff(np.log(np.r_[x.open.iloc[0],x.close.to_numpy()]))**2))) if valid else None
                rows.append(dict(**base,horizon_minutes=horizon,valid=valid,return_value=ret,realized_volatility=vol))
    a=pd.DataFrame(articles);a.to_csv(out/'articles.csv',index=False);z=pd.DataFrame(rows);z.to_csv(out/'responses.csv',index=False);summ=[]
    for level in ['article','event_first_arrival']:
        v=z if level=='article' else z.sort_values('available_utc').drop_duplicates(['symbol','event_group','horizon_minutes'])
        for key,g in v.groupby(['symbol','context','late','horizon_minutes']):
            h=g[g.valid];summ.append(dict(level=level,symbol=key[0],context=key[1],late=key[2],horizon=key[3],total=len(g),valid=len(h),mean_return=h.return_value.mean(),median_abs_return=h.return_value.abs().median(),mean_vol=h.realized_volatility.mean()))
    pd.DataFrame(summ).to_csv(out/'summary.csv',index=False)
    dump(out/'interpretation.json',{'anchor':'first five-minute opening at/after availability; no price before receipt is treated as executable','missing':'Any absent bar or outside regular same session invalidates interval; overnight articles retained in denominator, no overnight-to-open substitution','grouping':'Online near-title 72h Jaccard>=0.7 matching numeric tokens; proxy groups require manual review, not verified identical economic events','inference':'Descriptive associations, no causal claim or horizon selection'})

def coverage(d,out):
    n=pd.read_pickle(W/'audit/news_index.pkl');n['key']=n.archive+'::'+n.member;records=[];examples=[];cp={'AAPL':r'\b(?:Apple|AAPL)\b','AMZN':r'\b(?:Amazon|AMZN)\b'}
    for archive,sub in n.groupby('archive'):
        with zipfile.ZipFile(W/'raw/news'/archive) as z:
            for r in sub.itertuples():
                body=json.loads(z.read(r.member)).get('text','') or '';valid=r.language=='english' and r.chars>=100 and r.lag_hours>=0 and not r.physician
                for stock,pattern in cp.items():
                    title=bool(re.search(pattern,str(r.title),re.I));hit=re.search(pattern,body,re.I)
                    if not (hit or title):continue
                    excerpt=body[max(0,hit.start()-160):hit.end()+300] if hit else ''
                    category='comparison_candidate' if re.search(r'compet|rival|ahead|behind|versus|\bvs\b',excerpt,re.I) else 'direct_event_candidate' if re.search(pattern+r'.{0,70}\b(?:announc\w*|report\w*|rais\w*|launch\w*|expect\w*)',excerpt,re.I) else 'incidental_or_uncertain'
                    records.append(dict(symbol=stock,key=r.key,title_hit=title,body_hit=bool(hit),valid_base=valid,category=category,published_utc=r.published_utc,available_utc=r.available_utc,body_sha256=hashlib.sha256(body.encode()).hexdigest()))
                    # Deterministic coverage sample, one per archive/category/symbol; preserve candidates, no training admission.
                    if valid and hit and not title and not any(e['archive']==archive and e['symbol']==stock and e['category']==category for e in examples):examples.append(dict(archive=archive,symbol=stock,key=r.key,category=category,title=r.title,excerpt=excerpt,assistant_status='heuristic candidate; not an accepted event',human_classification='',human_note=''))
        print('coverage scanned',archive,flush=True)
    a=pd.DataFrame(records);a.to_csv(out/'candidates.csv',index=False);pd.DataFrame(examples).to_csv(out/'candidate_review.csv',index=False);a.groupby(['symbol','valid_base','title_hit','body_hit','category']).size().rename('n').reset_index().to_csv(out/'summary.csv',index=False)
    dump(out/'protocol.json',{'raw_records':len(n),'title_and_body':'Counts before dedup; raw tags are not object evidence','category':'Heuristic routing only; direct/comparison/incidental require review','admission':'No added candidates admitted to training in this run','independent_review':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mechanism',choices=['M04','M05']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();d=load(a.run.resolve());stage(a.run.resolve(),a.mechanism+'_v2' if a.mechanism=='M04' else a.mechanism,lambda out:(timing if a.mechanism=='M04' else coverage)(d,out))
