"""Independent slow reconstruction validates vectorized reaction features."""
from core import *
def main():
 d=data();raw=pd.read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');members=json.loads((PRIVATE/'history_members.json').read_text());expected=pd.read_pickle(PRIVATE/'history_reactions.pkl');sc=schedule();checks=[]
 # Three fixed indices per stock, spanning history; selection excludes no labels.
 for symbol in ('AAPL','AMZN'):
  b=load_bars(symbol);ii=np.flatnonzero(d.symbol==symbol)
  for i in [ii[0],ii[len(ii)//2],ii[-1]]:
   r=d.iloc[i];keys=list(dict.fromkeys(members[i]['current']+members[i]['old']));values=[];unknown=0;overnight=0
   for key in keys:
    a=raw.at[key,'available_utc'];intervals=[]
    for ses in sc[(sc.close>a)&(sc.open<r.cutoff_utc)].itertuples():
     begin=max(ses.open,a.ceil('5min'));end=min(ses.close,r.cutoff_utc.floor('5min'))
     if begin<end:intervals.append((begin,end))
    parts=[window(b,start,end) for start,end in intervals]
    if not parts or any(p is None for p in parts):unknown+=1;continue
    overnight+=int(len(intervals)>1);values.append((np.log(b.loc[intervals[-1][1]-pd.Timedelta('5min'),'close']/b.loc[intervals[0][0],'open']),np.sqrt(sum(p['rv']**2 for p in parts))))
   actual=np.array([np.mean([v[0] for v in values]) if values else 0,np.mean([v[1] for v in values]) if values else 0,unknown/max(1,len(keys)),overnight/max(1,len(keys)),np.log1p(len(values))]);error=float(abs(actual-expected.iloc[i].to_numpy()).max());assert error<1e-10,(r.key,error);checks.append(dict(key=r.key,articles=len(keys),max_error=error))
 dump(OUT/'history_independent_checks.json',checks);print('history slow-reference checks',len(checks))
if __name__=='__main__':main()
