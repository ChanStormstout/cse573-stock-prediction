"""Independently recompute public result invariants; does not fit models."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent; OUT=HERE/'v1'
d=pd.read_csv(OUT/'predictions.csv'); m=pd.read_csv(OUT/'metrics.csv'); c=pd.read_csv(OUT/'calibrators.csv'); e=json.loads((OUT/'execution.json').read_text())
assert len(d)==1607 and d.key.is_unique
for r in m.itertuples():
 g=d[(d.phase==r.phase)&(d.symbol==r.symbol)]; p=g[r.method]
 assert len(g)==r.n
 for actual,expected in [(balanced_accuracy_score(g.label,p>=.5),r.BA),(matthews_corrcoef(g.label,p>=.5),r.MCC),(brier_score_loss(g.label,p),r.Brier)]:
  assert abs(actual-expected)<1e-12
u=d[d.phase!='warmup']; no=u.has_original_news==0
for a,b in [('B0','B1'),('B2','B3')]:
 assert np.array_equal(u[a]>=.5,u[b]>=.5)
 assert np.array_equal(u.loc[no,b],u.loc[no,'R1'])
assert (c.slope>0).all()
for r in c.itertuples():
 if pd.notna(r.fit_last_month):assert r.fit_last_month<('2018-09' if r.month=='final' else r.month)
a=pd.read_csv(OUT/'aggregation_interventions.csv').set_index('key'); ref=d.set_index('key').loc[a.index]
for left,right in [('N0M_article','N0M'),('N1M_event','N1M')]:assert np.allclose(a[left],ref[right],rtol=0,atol=1e-12)
days=pd.read_csv(OUT/'amzn_day_contributions.csv')
for phase,g in days.groupby('phase'):
 q=m[(m.symbol=='AMZN')&(m.phase==phase)].set_index('method')
 assert abs(g.BA_contribution.sum()-(q.loc['N1M','BA']-q.loc['N0M','BA']))<1e-12
for path,key in [(HERE/'run.py','code_sha256'),(HERE/'PRE_REGISTRATION.md','protocol_sha256'),(ROOT/'outputs/stock_paper_methods_4h/v1/predictions.csv','source_sha256')]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==e[key]
result={'status':'PASS','recomputed_metric_rows':len(m),'rows':len(d),'intervention_rows':len(a),'checks':['unique keys','BA MCC Brier recomputed','directions preserved','strict fallback','past-only calibration','intervention endpoint parity','day contributions sum','code protocol source hashes'],'browser_checks':['stock selection','no-news exact fallback display','reveal actual label','period change resets reveal','visual card layout']}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
