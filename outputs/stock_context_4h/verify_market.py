"""Fit-free independent artifact verifier."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);checks=[]
 def ck(name,ok):checks.append({'check':name,'pass':bool(ok)})
 req=['predictions.csv','metrics.csv','monthly_metrics.csv','m0_cv.csv','selection.json','training_evidence.json','market_coverage.csv','protocol_fingerprint.json','advancement.json','attribution.json'];ck('required',all((root/x).exists() for x in req))
 if checks[-1]['pass']:
  d=pd.read_csv(root/'predictions.csv');ck('unique_keys',not d.key.duplicated().any());ck('methods',all(x in d for x in ('M0','Mmeta','M1')));ck('probability_range',all(((d[x]>=0)&(d[x]<=1)).all() for x in ('M0','Mmeta','M1')));ck('monthly_complete',len(pd.read_csv(root/'monthly_metrics.csv'))>=6);ck('coverage',pd.read_csv(root/'market_coverage.csv')[['SPY_valid','QQQ_valid']].ge(.95).all().all())
  fp=json.loads((root/'protocol_fingerprint.json').read_text());ck('source_hash',Path(fp['source_path']).exists() and sha(fp['source_path'])==fp['source_sha256']);source=pd.read_csv(fp['source_path']) if Path(fp['source_path']).exists() else pd.DataFrame();ev=json.loads((root/'training_evidence.json').read_text());ok=True;recon={}
  for x in ev:
   z=root/'models'/f"{x['method']}_{x['symbol']}_{x['fold']}.npz";ok &= z.exists() and sha(z)==x['npz_sha256'];
   if z.exists():
    q=np.load(z,allow_pickle=False);ok &= list(q.files)==['mean','scale','coef','intercept','classes'] and len(x['columns'])==q['coef'].shape[1]
    evrows=source[(source.symbol==x['symbol'])&(source.month==x['fold'])];xx=np.where(np.isfinite((evrows[x['columns']].to_numpy(float)-q['mean'])/q['scale']),(evrows[x['columns']].to_numpy(float)-q['mean'])/q['scale'],0);recon.update({k:float(v) for k,v in zip(evrows.key,1/(1+np.exp(-(xx@q['coef'].T+q['intercept']).ravel())) )})
  ck('plain_models_hash_and_shape',ok);ck('labels',source.set_index('key').loc[d.key,'label'].to_numpy().astype(int).tolist()==d.label.astype(int).tolist());ck('direct_prediction_reconstruction',np.max(np.abs(np.array([recon[k] for k in d.key])-d.M1.to_numpy()))<1e-12);sel=json.loads((root/'selection.json').read_text());ck('same_C',sel.get('same_C_all_methods') is True)
 status='PASS' if all(x['pass'] for x in checks) else 'FAIL';(root/'verification.json').write_text(json.dumps({'status':status,'checks':checks,'fit_free':True},indent=2)+'\n');
 if status!='PASS':raise SystemExit('verification failed')
if __name__=='__main__':main()
