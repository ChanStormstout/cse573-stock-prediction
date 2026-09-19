"""PASS-gated descriptive report for Market Context v1."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);v=root/'verification.json'
 if not v.exists() or json.loads(v.read_text()).get('status')!='PASS':raise SystemExit('report blocked: verifier is not PASS')
 fp=json.loads((root/'protocol_fingerprint.json').read_text());cov=pd.read_csv(root/'market_coverage.csv');met=pd.read_csv(root/'metrics.csv');gate=json.loads((root/'advancement.json').read_text());attr=json.loads((root/'attribution.json').read_text());cv=pd.read_csv(root/'m0_cv.csv')
 lines=['# Market Context v1 report','',f"Source mode: `{fp['mode']}`. Provider contract: Alpaca/SIP/raw/5Min (synthetic fixture when mode is synthetic).",'','## Coverage','',cov.to_markdown(index=False),'','## March-August forward OOF and later periods','',met.to_markdown(index=False),'','## Registered June-August M1 minus M0 gate','',json.dumps(gate,indent=2),'','## M1 minus Mmeta attribution','',json.dumps(attr,indent=2),'','## Selected M0 C by stock/fold','',cv[['symbol','fold','C']].drop_duplicates().to_markdown(index=False),'','Development and later rows, if present, are **EXPOSED EXPLORATORY HISTORICAL BACKTEST**. This artifact has no authenticated market-data run.']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
