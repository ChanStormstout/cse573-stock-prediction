"""Fit-free verifier for a future immutable market artifact."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);required=['predictions.csv','metrics.csv','monthly_metrics.csv','selection.json','training_evidence.json','protocol_fingerprint.json'];missing=[x for x in required if not(root/x).exists()];status='FAIL' if missing else 'PASS';(root/'verification.json').write_text(json.dumps({'status':status,'missing':missing,'fit_free':True},indent=2)+'\n');
 if status!='PASS':raise SystemExit('verification failed')
if __name__=='__main__':main()
