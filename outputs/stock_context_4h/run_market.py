"""Guarded market runner; real Mmeta/M1 remains unapproved."""
from __future__ import annotations
import argparse,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--approve-market-context-run',action='store_true');p.add_argument('--output',default=str(HERE/'market_v1'));a=p.parse_args()
 if not a.approve_market_context_run:raise SystemExit('Real market scoring is preregistered only.')
 parity=HERE/'audit_v2/r1_independent_refit.json';access=HERE/'audit_v1/market_access.json'
 if not parity.exists() or json.loads(parity.read_text()).get('status')!='PASS':raise SystemExit('BLOCKED_R1_PARITY')
 if not access.exists() or json.loads(access.read_text()).get('status')!='AVAILABLE':raise SystemExit('STOP_NO_MARKET_RUN: market data unavailable')
 if Path(a.output).exists():raise SystemExit('refuse overwrite')
 raise SystemExit('Real Mmeta/M1 execution is not authorized in this task.')
if __name__=='__main__':main()
