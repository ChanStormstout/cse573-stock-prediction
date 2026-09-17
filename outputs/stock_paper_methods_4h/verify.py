"""Necessary checks only: time, replay, selection, calibration and cluster conflicts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import HERE,ROOT,WORK,sha,metric,compatible,calibration_fit

def main(run='v1'):
 p=HERE/run;m=json.loads((p/'model_manifest.json').read_text());d=pd.read_csv(p/'predictions.csv');s=json.loads((p/'selection.json').read_text())
 assert len(d)==1607 and not d.key.duplicated().any()
 assert m['code_sha256']==sha(HERE/'run.py') and m['protocol_sha256']==sha(HERE/'PRE_REGISTRATION.md')
 assert all(sha(ROOT/k)==v for k,v in m['source_hashes'].items())
 assert all(pd.Timestamp(x['train_end'])<pd.Timestamp(x['cutoff_min']) and x['reload_error']<1e-12 for x in m['fits'])
 for x in s['joint']+s['calibration']:
  boundary='2018-09' if x['month']=='final' else x['month'];assert all(mm<boundary for mm in x['selection_months'])
 for r in pd.read_csv(p/'metrics.csv').itertuples():
  q=d[(d.phase==r.phase)&(d.symbol==r.symbol)];actual=metric(q.label,q[r.method])
  for k in ['BA','MCC','Brier','AUC']:assert abs(actual[k]-getattr(r,k))<1e-12
 mask=(d.phase!='warmup')&(d.has_original_news==0)
 for name in ['J0','J1','J2','J3','N0','N0M','N1','N1M','F1_calibrated','F2_calibrated']:
  assert np.array_equal(d.loc[mask,name],d.loc[mask,'R1'])
 for branch in ['F1','F2']:
  mask=d.phase!='warmup';assert np.array_equal(d.loc[mask,branch]>=.5,d.loc[mask,f'{branch}_temperature']>=.5)
 assert compatible('Apple raises target to $200 Q2','Apple raises target to $200 Q2 report')
 for a,b in [('Apple raises target to $200','Apple lowers target to $200'),('Apple raises target to $200 Q2','Apple raises target to $200 Q3'),('Apple denies chip allegations','Apple chip allegations'),('Apple raises target to $200','Apple raises target to $180')]:assert not compatible(a,b)
 calibrators=pd.read_csv(p/'calibration.csv');assert (calibrators.slope>0).all()
 assert not m['phase2_eligible'] or any(x['passes'] for x in s['advancement'])
 print(json.dumps(dict(status='PASS',fixed_windows=len(d),evaluated_windows=int((d.phase!='warmup').sum()),fits=len(m['fits']),metric_rows=len(pd.read_csv(p/'metrics.csv')),max_reload_error=max(x['reload_error'] for x in m['fits']),raw_controls_reproduced=m['parity'],strict_fallback=True,positive_calibration=True),indent=2))
if __name__=='__main__':main()
