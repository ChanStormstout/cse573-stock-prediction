"""Market Context v1 runner; authenticated real scoring remains prohibited."""
from __future__ import annotations
import argparse
from pathlib import Path
from .market_core import run_pipeline

def repo_root(): return Path(__file__).resolve().parents[2]
def real_source_root(): return repo_root()/'work/stock-data/context_increment_4h/market_v1_authenticated'
def canonical_goal60_predictions_path(): return repo_root()/'outputs/stock_goal60_4h/v1/predictions.csv'
def main():
 p=argparse.ArgumentParser();p.add_argument('--synthetic-input');p.add_argument('--approve-market-context-run',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 if a.synthetic_input: run_pipeline(Path(a.synthetic_input),Path(a.output),'synthetic'); return
 if not a.approve_market_context_run: raise SystemExit('STOP_NO_MARKET_RUN: real path is preregistered and requires explicit approval')
 if not canonical_goal60_predictions_path().is_file(): raise SystemExit('STOP_NO_MARKET_RUN: canonical Goal60 R1 predictions are unavailable')
 if not real_source_root().exists(): raise SystemExit('STOP_NO_MARKET_RUN: AUTH_REQUIRED; no authenticated frozen Alpaca SIP source artifact')
 run_pipeline(real_source_root(),Path(a.output),'real')
if __name__=='__main__':main()
