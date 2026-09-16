"""Cutoff-safe price context. Never accepts labels or future target prices."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
VALUES=['prior_tail30','quote_vs_prior_close','opening_gap','session_return','recent15','prior_rms']
MASKS=[x+'_valid' for x in VALUES]
@dataclass
class FeatureRow:
 values: dict
 masks: dict
 provenance: dict

def load_sources(root,symbol):
 prefix={'AAPL':'APPLE','AMZN':'AMAZON'}[symbol]
 b=pd.read_csv(root/f'raw/CHARTS/{prefix}5.csv',header=None,names=['date','time','open','high','low','close','activity'])
 b.index=pd.to_datetime(b.date+' '+b.time,format='%Y.%m.%d %H:%M',utc=True);b=b.sort_index();b['end']=b.index+pd.Timedelta('5min')
 if b.index.duplicated().any():raise ValueError('Duplicate raw bar')
 c=pd.read_csv(root/'audit/xnys_schedule.csv',index_col=0);c.index=pd.to_datetime(c.index).strftime('%Y-%m-%d')
 for col in ['open','close']:c[col]=pd.to_datetime(c[col],utc=True)
 b['day']=b.index.tz_convert('America/New_York').strftime('%Y-%m-%d');opens=b.day.map(c.open);closes=b.day.map(c.close)
 b['rth']=(b.index>=opens)&(b.end<=closes)
 return b,c

def build_price_snapshot(symbol,target_start,cutoff,bars,calendar,asof_end=None,quote_scope='rth',availability_lag='0min'):
 if quote_scope not in ['rth','raw_assumption']:raise ValueError('Unknown quote scope')
 cutoff=pd.Timestamp(cutoff);asof=cutoff if asof_end is None else pd.Timestamp(asof_end);lag=pd.Timedelta(availability_lag)
 if asof>cutoff or lag<pd.Timedelta(0):raise ValueError('Future asof or negative lag')
 day=pd.Timestamp(target_start).tz_convert('America/New_York').strftime('%Y-%m-%d')
 v=dict.fromkeys(VALUES,0.);mask=dict.fromkeys(MASKS,0);prov={'symbol':symbol,'cutoff':str(cutoff),'asof':str(asof),'scope':quote_scope,'lag':str(lag),'used':{}}
 available=bars[(bars.end+lag<=asof)&(bars.rth if quote_scope=='rth' else True)]
 def put(key,value,x):
  if not len(x) or not np.isfinite(value):return
  assert (x.end+lag<=asof).all()
  v[key]=float(value);mask[key+'_valid']=1;prov['used'][key]={'first_start':str(x.index.min()),'last_end':str(x.end.max()),'n_bars':len(x)}
 def complete(start,end):
  idx=pd.date_range(start,end-pd.Timedelta('5min'),freq='5min');x=available.reindex(idx)
  return x if len(idx)>0 and not x[['open','high','low','close','end']].isna().any().any() else available.iloc[:0]
 prev=calendar[calendar.index<day]
 if len(prev):
  pc=prev.iloc[-1];closebar=complete(pc.close-pd.Timedelta('5min'),pc.close);tail=complete(pc.close-pd.Timedelta('30min'),pc.close);full=complete(pc.open,pc.close)
  if len(tail):put('prior_tail30',np.log(tail.close.iloc[-1]/tail.open.iloc[0]),tail)
  if len(full):put('prior_rms',np.sqrt(np.mean(np.log(full.close/full.open)**2)),full)
  today=available[(available.day==day)&available.rth];session_open=calendar.loc[day,'open'];first=complete(session_open,session_open+pd.Timedelta('5min'))
  if len(closebar) and len(available):put('quote_vs_prior_close',np.log(available.close.iloc[-1]/closebar.close.iloc[-1]),pd.concat([closebar,available.tail(1)]).drop_duplicates())
  if len(first) and len(closebar):put('opening_gap',np.log(first.open.iloc[0]/closebar.close.iloc[-1]),pd.concat([closebar,first]))
  if len(first) and len(today):
   continuous=complete(session_open,today.end.iloc[-1])
   if len(continuous):put('session_return',np.log(continuous.close.iloc[-1]/first.open.iloc[0]),continuous)
 if len(available):
  latest=available.iloc[-1];recent=complete(latest.end-pd.Timedelta('15min'),latest.end)
  if len(recent) and recent.day.nunique()==1:put('recent15',np.log(recent.close.iloc[-1]/recent.open.iloc[0]),recent)
  v['quote_age_hours']=(cutoff-latest.end).total_seconds()/3600;prov['quote_end']=str(latest.end)
 else:v['quote_age_hours']=0.;prov['quote_end']=None
 return FeatureRow(v,mask,prov)
