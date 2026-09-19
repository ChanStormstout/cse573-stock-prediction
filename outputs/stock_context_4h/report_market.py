"""PASS-gated future market report."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);v=root/'verification.json'
 if not v.exists() or json.loads(v.read_text()).get('status')!='PASS':raise SystemExit('report blocked: verifier is not PASS')
 (root/'REPORT.md').write_text('# Market context v1\n\nEXPOSED EXPLORATORY HISTORICAL BACKTEST.\n')
if __name__=='__main__':main()
