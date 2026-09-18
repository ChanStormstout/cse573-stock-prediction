"""Future real run entrypoint intentionally requires later explicit authorization."""
from __future__ import annotations
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--approve-market-context-run',action='store_true');a=p.parse_args()
 if not a.approve_market_context_run:raise SystemExit('Market v1 is preregistered only; no real Mmeta/M1 fitting is authorized.')
 raise SystemExit('APPROVE_MARKET_CONTEXT_RUN is not authorized in this task.')
if __name__=='__main__':main()
