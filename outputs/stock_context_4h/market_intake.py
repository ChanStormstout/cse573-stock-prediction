"""Bounded credential-safe Alpaca data probe; it never calls trading endpoints."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request,urlopen
HERE=Path(__file__).resolve().parent; OUT=HERE/'audit_v1'/'market_access.json'
def main():
 key=os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY'); secret=os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
 row={'provider':'Alpaca','endpoint':'https://data.alpaca.markets/v2/stocks/{symbol}/bars','feed':'sip','adjustment':'raw','symbols':['SPY','QQQ'],'credentials_present':bool(key and secret),'attempts':0}
 if not(key and secret): row.update(status='AUTH_REQUIRED',required_environment=['APCA_API_KEY_ID','APCA_API_SECRET_KEY'])
 else:
  url='https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=5Min&start=2018-01-03T14%3A30%3A00Z&end=2018-01-03T15%3A00%3A00Z&feed=sip&adjustment=raw&limit=10'
  for attempt in range(2):
   row['attempts']=attempt+1
   try:
    with urlopen(Request(url,headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret}),timeout=20) as r: payload=json.loads(r.read());row.update(status='HTTP_OK',http_status=r.status,bar_count=len(payload.get('bars') or []));break
   except HTTPError as e: row.update(status='AUTH_OR_ENTITLEMENT_ERROR' if e.code in (401,403) else 'UPSTREAM_HTTP_ERROR',http_status=e.code)
   except (URLError,TimeoutError) as e: row.update(status='UPSTREAM_NETWORK_ERROR',error_type=type(e).__name__)
   time.sleep(.2*(attempt+1))
 row['acquired_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
if __name__=='__main__':main()
