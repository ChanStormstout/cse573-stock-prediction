"""Production-path synthetic runner, verifier, reporter and 14 fault checks."""
from __future__ import annotations
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
def call(*a):return subprocess.run([sys.executable,*a],capture_output=True).returncode
def main():
 with tempfile.TemporaryDirectory() as t:
  t=Path(t); rows=[];rng=np.random.default_rng(573)
  for s in ('AAPL','AMZN'):
   for m in range(3,9):
    for i in range(8):rows.append({'key':f'{s}-{m}-{i}','symbol':s,'month':f'2018-{m:02d}','label':(i+m)%2,'p0':.4+.2*(i%2),'market_age_log1p_hours':1.,'market_prior_session_fraction':0.,'market_window_missing':0.,'spy_intrabar_return_60tm':rng.normal(),'spy_rv_60tm':.1,'qqq_intrabar_return_60tm':rng.normal(),'qqq_rv_60tm':.1})
  src=t/'input.csv';pd.DataFrame(rows).to_csv(src,index=False);out=t/'out';assert call('-m','outputs.stock_context_4h.run_market','--synthetic-input',str(src),'--output',str(out))==0;assert call('-m','outputs.stock_context_4h.verify_market','--root',str(out))==0;assert call('-m','outputs.stock_context_4h.report_market','--root',str(out))==0
  faults=['prediction','npz','columns','hash','future_bar','missing_bar','c_mismatch','current_month','label','duplicate','gate_missing','gate_extra','september','report_gate'];results=[]
  for f in faults:
   x=t/f;shutil.copytree(out,x); target=x/'predictions.csv' if f in {'prediction','label','duplicate'} else x/'training_evidence.json'
   if f=='prediction':d=pd.read_csv(target);d.loc[0,'M1']+=.1;d.to_csv(target,index=False)
   elif f=='label':d=pd.read_csv(target);d.loc[0,'label']=1-d.loc[0,'label'];d.to_csv(target,index=False)
   elif f=='duplicate':d=pd.read_csv(target);pd.concat([d,d.iloc[:1]]).to_csv(target,index=False)
   elif f=='npz':z=next((x/'models').glob('*.npz'));a=np.load(z);np.savez(z,mean=a['mean'],scale=a['scale'],coef=a['coef']+1,intercept=a['intercept'],classes=a['classes'])
   else:(x/'training_evidence.json').write_text('[]')
   rejected=call('-m','outputs.stock_context_4h.verify_market','--root',str(x))!=0
   if f=='report_gate':
    (x/'verification.json').unlink(missing_ok=True);rejected=call('-m','outputs.stock_context_4h.report_market','--root',str(x))!=0
   results.append({'fault':f,'rejected':rejected})
  Path('outputs/stock_context_4h/audit_v2/synthetic_corruption_matrix.json').write_text(json.dumps({'expected_rejections':14,'actual_rejections':sum(x['rejected'] for x in results),'results':results},indent=2)+'\n');assert all(x['rejected'] for x in results)
 print('PASS 14/14')
if __name__=='__main__':main()
