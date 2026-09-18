"""Minimal no-secret Alpaca minute-data feasibility probe."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'v1'
DATES=['2018-01-10','2018-04-27','2018-09-04','2018-12-03']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    key=os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
    secret=os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
    rows=[]
    for symbol in ('SPY','QQQ'):
        for day in DATES:
            url=(f'https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Min'
                 f'&start={day}T14%3A30%3A00Z&end={day}T21%3A00%3A00Z&limit=10&feed=iex')
            rec={'symbol':symbol,'date':day,'feed':'iex','timeframe':'1Min','url_host':'data.alpaca.markets'}
            if not (key and secret):
                rec.update(status='AUTH_REQUIRED_NO_CREDENTIALS',http_status=None,bar_count=None)
            else:
                req=Request(url,headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret})
                try:
                    with urlopen(req,timeout=20) as response:
                        payload=json.loads(response.read().decode())
                    bars=payload.get('bars') or []
                    rec.update(status='HTTP_OK',http_status=200,bar_count=len(bars),first_bar=bars[0].get('t') if bars else None,
                               keys=sorted(payload.keys()))
                except HTTPError as exc:
                    rec.update(status='HTTP_ERROR',http_status=exc.code,bar_count=None)
                except (URLError,TimeoutError,ValueError) as exc:
                    rec.update(status='REQUEST_FAILED',http_status=None,bar_count=None,error_type=type(exc).__name__)
            rows.append(rec)
    statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    result={'dates':DATES,'symbols':['SPY','QQQ'],'requested_regular_session':'14:30Z-21:00Z',
            'credentials_present':bool(key and secret),'rows':rows,'status_counts':statuses,
            'market_branch_ready':all(r['status']=='HTTP_OK' and r['bar_count'] for r in rows),
            'checked_at_epoch':time.time(),
            'decision':'STOP_AUTH_OR_DATA_AUDIT' if not all(r['status']=='HTTP_OK' and r['bar_count'] for r in rows) else 'PROCEED_TO_BAR_SCHEMA_AUDIT'}
    (OUT/'alpaca_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('credentials_present','status_counts','market_branch_ready','decision')},indent=2))

if __name__=='__main__':main()
