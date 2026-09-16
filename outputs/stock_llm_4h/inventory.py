"""Availability inventory, not a semantic event count."""
from common import *
import pandas as pd

def main():
 source=ROOT/'work/stock-data/audit/news_index.pkl';raw=pd.read_pickle(source)
 r=raw[raw.lag_hours.ge(0)&raw.language.eq('english')&(raw.available_utc>='2018-01-01')&(raw.available_utc<'2018-03-01')]
 counts={}
 for sym,alias in [('AAPL','Apple|AAPL'),('AMZN','Amazon|AMZN')]:
  x=r[r.title.str.contains(alias,case=False,na=False)&r.title.str.contains(r'price target|target price|rating|downgrad',case=False,na=False)&~r.title.str.contains(r'stake|position|holding|earnings|stock market|futures',case=False,na=False)].drop_duplicates('normalized_hash')
  counts[sym]={'candidate_records':len(x),'months':x.groupby(x.available_utc.dt.strftime('%Y-%m')).size().to_dict()}
 dump(B/'data_inventory.json',{'source_sha256':file_sha(source),'early_eligible_records':len(r),'period':['2018-01-01','2018-03-01'],'filters':'English, nonnegative lag, Jan-Feb; title contains target and rating/target keywords, excluding common holdings/market keywords; normalized-content dedup','candidate_counts':counts,'warning':'Title candidates only, NOT confirmed events or a complete body audit. AMZN article may concern another company.'})
if __name__=='__main__':main()
