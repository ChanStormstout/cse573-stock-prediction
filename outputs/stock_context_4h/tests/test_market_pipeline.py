"""Separate-process, fault-specific production-path synthetic verification."""
from __future__ import annotations
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from outputs.stock_context_4h.market_core import R1_FEATURES

ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(*args):return subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True)
def fixture(root):
 root.mkdir(); rng=np.random.default_rng(573); schedule=[]; rows=[]; bars={'SPY':[],'QQQ':[]}; keys=[]
 for month in range(1,11):
  day=pd.Timestamp(f'2018-{month:02d}-05 14:30:00+00:00'); close=day+pd.Timedelta(hours=6,minutes=30);schedule.append({'open':day,'close':close})
  starts=pd.date_range(day,close-pd.Timedelta(minutes=5),freq='5min')
  for sym,base in [('SPY',270),('QQQ',160)]:
   for j,t in enumerate(starts):bars[sym].append({'bar_start_utc':t.isoformat(),'open':base+.01*j,'close':base+.01*j+(.03 if j%2 else -.01)})
  cutoff=day+pd.Timedelta(minutes=61)
  for stock in ('AAPL','AMZN'):
   for i in range(10):
    row={'key':f'{stock}-{month:02d}-{i}','symbol':stock,'month':f'2018-{month:02d}','label':int((i+month+(stock=='AMZN'))%2),'cutoff_utc':cutoff.isoformat(),'session_id':str(day.date())}
    row.update({c:float(rng.normal()+.05*i) for c in R1_FEATURES});rows.append(row)
    if month>=3:keys.append(row['key'])
 pd.DataFrame(rows).to_csv(root/'r1_rows.csv',index=False);pd.DataFrame(schedule).to_csv(root/'schedule.csv',index=False)
 for sym in bars:pd.DataFrame(bars[sym]).to_csv(root/f'bars_{sym}.csv',index=False)
 manifest={'provider':'Synthetic','feed':'synthetic','adjustment':'raw','timeframe':'5Min','symbols':['SPY','QQQ'],'bar_hashes':{s:sha(root/f'bars_{s}.csv') for s in bars},'request_range':'2018-01 through 2018-10','acquired_at':'synthetic','duplicate_timestamp_count':0,'unexpected_timestamp_count':0}
 (root/'source_manifest.json').write_text(json.dumps(manifest));(root/'expected_keys.json').write_text(json.dumps({'keys':keys}))
def mutate(name,x):
 if name=='prediction':
  p=x/'predictions.csv';d=pd.read_csv(p);d.loc[0,'M1']+=.1;d.to_csv(p,index=False)
 elif name=='npz':
  p=next((x/'models').glob('M1_*.npz'));a=np.load(p);np.savez(p,mean=a['mean'],scale=a['scale'],coef=a['coef']+.01,intercept=a['intercept'],classes=a['classes'])
 elif name=='columns':
  p=next((x/'models').glob('M1_*.json'));d=json.loads(p.read_text());d['columns'][0],d['columns'][1]=d['columns'][1],d['columns'][0];p.write_text(json.dumps(d))
 elif name=='hash':
  p=x/'protocol_fingerprint.json';d=json.loads(p.read_text());d['market_feature_code_sha256']='0'*64;p.write_text(json.dumps(d))
 elif name=='future_bar':
  p=x/'market_features.csv';d=pd.read_csv(p);d.loc[0,'spy_intrabar_return_60tm']+=.1;d.to_csv(p,index=False)
 elif name=='missing_bar':
  src=Path(json.loads((x/'protocol_fingerprint.json').read_text())['source_dir']);p=src/'bars_SPY.csv';d=pd.read_csv(p);d=d.iloc[1:];d.to_csv(p,index=False)
 elif name=='c_mismatch':
  p=next((x/'models').glob('M1_*.json'));d=json.loads(p.read_text());d['C']=1.0 if d['C']!=1 else .01;p.write_text(json.dumps(d))
 elif name=='current_month':
  p=x/'m0_cv.csv';d=pd.read_csv(p);d.loc[(d.fold=='2018-04')&(d.C==.01),'BA']=1.;d.to_csv(p,index=False)
 elif name=='label':
  p=x/'predictions.csv';d=pd.read_csv(p);d.loc[0,'label']=1-d.loc[0,'label'];d.to_csv(p,index=False)
 elif name=='duplicate':
  p=x/'predictions.csv';d=pd.read_csv(p);pd.concat([d,d.iloc[[0]]]).to_csv(p,index=False)
 elif name=='gate_missing':
  p=x/'advancement.json';d=json.loads(p.read_text());d['cells']=d['cells'][:-1];p.write_text(json.dumps(d))
 elif name=='gate_extra':
  p=x/'advancement.json';d=json.loads(p.read_text());d['cells'].append(dict(d['cells'][0],month='2018-09'));p.write_text(json.dumps(d))
 elif name=='september':
  p=next((x/'models').glob('M0_AAPL_2018-09.json'));d=json.loads(p.read_text());d['training_months'].append('2018-09');p.write_text(json.dumps(d))
def main():
 faults={'prediction':'all_three_probability_reconstruction','npz':'model_hash_columns_and_scaler','columns':'model_hash_columns_and_scaler','hash':'protocol_fingerprint','future_bar':'raw_bar_time_safe_feature_reconstruction','missing_bar':'raw_bar_time_safe_feature_reconstruction','c_mismatch':'model_hash_columns_and_scaler','current_month':'chronological_C_and_no_later_leakage','label':'labels_exact','duplicate':'complete_expected_keys','gate_missing':'primary_gate_exact_six_cells','gate_extra':'primary_gate_exact_six_cells','september':'model_hash_columns_and_scaler'}
 with tempfile.TemporaryDirectory() as t:
  t=Path(t);src=t/'fixture';fixture(src);out=t/'clean';r=call('-m','outputs.stock_context_4h.run_market','--synthetic-input',str(src),'--output',str(out));assert r.returncode==0,r.stderr
  r=call('-m','outputs.stock_context_4h.verify_market','--root',str(out));assert r.returncode==0,r.stderr
  assert call('-m','outputs.stock_context_4h.report_market','--root',str(out)).returncode==0
  result=[]
  for name,expected in faults.items():
   x=t/name;shutil.copytree(out,x); own=t/f'{name}_source';shutil.copytree(src,own)
   fp=x/'protocol_fingerprint.json';z=json.loads(fp.read_text());z['source_dir']=str(own);fp.write_text(json.dumps(z))
   mutate(name,x);r=call('-m','outputs.stock_context_4h.verify_market','--root',str(x));v=json.loads((x/'verification.json').read_text());failed=[z['check'] for z in v['checks'] if not z['pass']];result.append({'fault':name,'mutation_description':name,'expected_verifier_check':expected,'actual_failed_checks':failed,'rejected':r.returncode!=0});assert expected in failed,(name,failed)
  x=t/'report_gate';shutil.copytree(out,x);(x/'verification.json').unlink();r=call('-m','outputs.stock_context_4h.report_market','--root',str(x));result.append({'fault':'report_gate','mutation_description':'remove PASS verification','expected_verifier_check':'reporter_requires_PASS','actual_failed_checks':['reporter_requires_PASS'] if r.returncode else [],'rejected':r.returncode!=0});assert r.returncode!=0
  dest=ROOT/'outputs/stock_context_4h/audit_v3';dest.mkdir(exist_ok=True);(dest/'synthetic_corruption_matrix.json').write_text(json.dumps({'status':'PASS','clean_e2e':'PASS','expected_rejections':14,'actual_rejections':sum(x['rejected'] for x in result),'results':result},indent=2)+'\n')
 print('PASS clean E2E and 14 distinct fault checks')
if __name__=='__main__':main()
