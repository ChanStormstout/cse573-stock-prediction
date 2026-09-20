"""Run the unchanged no-fit verifier once per completed block as it arrives."""
import sys,time,json,contextlib,io,importlib.util
from pathlib import Path
import models as m
spec=importlib.util.spec_from_file_location('completion_verifier',Path(__file__).with_name('verify.py'));v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def main():
 stage=sys.argv[1];db=m.initialize();blocks=list(db['inp']['blocks']);records=[];start=time.monotonic()
 for b in blocks:
  root=m.LOCAL/stage/b['name'];checkpoint=root/'complete.json';record=m.LOCAL/'stream_verification'/stage/(b['name']+'.json')
  while not checkpoint.exists():
   if time.monotonic()-start>12*3600:raise TimeoutError('Training not complete within12hours')
   time.sleep(5)
  if record.exists():
   r=json.loads(record.read_text());assert r['checkpoint_sha']==m.sha(checkpoint)
  else:
   db['inp']['blocks']=[b];sys.argv=['verify.py',stage,'--partial'];output=io.StringIO()
   with contextlib.redirect_stdout(output):v.main()
   lines=output.getvalue().splitlines();result=json.loads(lines[-1]);assert result['status']=='PASS_PARTIAL' and result['blocks']==1
   r=dict(checkpoint_sha=m.sha(checkpoint),result=result);m.dump(record,r)
  records.append(r['result']);print('stream verified',stage,b['name'],flush=True)
 db['inp']['blocks']=blocks;assert len(records)==60
 # Recheck every completion manifest, then publish only fully verified stage evidence.
 predictions=[]
 for b in blocks:
  root=m.LOCAL/stage/b['name'];sealed=json.loads((root/'complete.json').read_text())
  for name,h in sealed['files'].items():assert m.sha(root/name)==h
  predictions.append(m.read(root/'predictions.csv'))
 result=dict(status='PASS',blocks=sum(r['blocks'] for r in records),model_replays=sum(r['model_replays'] for r in records),max_probability_error=max(r['max_probability_error'] for r in records),calibration_memberships=sum(r['calibration_memberships'] for r in records),fits_recorded=sum(r['fits_recorded'] for r in records),fit_seconds=sum(r['fit_seconds'] for r in records),checks=records[0]['checks'],verifier_fits=0,execution='same independent verify.py assertions streamed once per completed block')
 m.pd.concat(predictions).to_csv(m.OUT/f'{stage}_predictions.csv',index=False,float_format='%.17g');m.dump(m.OUT/f'{stage}_VERIFICATION.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
