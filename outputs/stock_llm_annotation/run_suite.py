"""Sequential local suite; requires completed original SFT, preserves every run."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
 base=ROOT/'work/stock-data/annotation';runs=base/'runs';first=json.loads((runs/'qlora_v2/summary.json').read_text());assert first['actual_training'] and first['reload_greedy_equal']
 py=ROOT/'work/stock-data/structured-env/bin/python';train=ROOT/'outputs/stock_llm_4h/train_adapter.py';infer=ROOT/'outputs/stock_llm_4h/infer.py'
 stages=[('gate_training',[py,train,'--data',base/'student_v2','--out',runs/'qlora_gate_v1','--expanded-contract','--balance-events','--event-gate-loss','--seed','573','--max-length','2048']),('frozen_inference',[py,infer,'--data',base/'student_v3','--out',runs/'frozen_v3','--split','valid,check','--expanded-contract']),('ordinary_inference',[py,infer,'--data',base/'student_v3','--out',runs/'tuned_v3','--split','check','--expanded-contract','--adapter',runs/'qlora_v2/selected']),('gate_inference',[py,infer,'--data',base/'student_v3','--out',runs/'tuned_gate_v3','--split','check','--expanded-contract','--adapter',runs/'qlora_gate_v1/selected'])]
 records=[]
 for name,cmd in stages:
  print('START',name,flush=True);start=time.time()
  with (out/(name+'.log')).open('x') as f:r=subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  records.append({'stage':name,'command':list(map(str,cmd)),'exit_code':r.returncode,'seconds':time.time()-start});(out/'stages.json').write_text(json.dumps(records,indent=2)+'\n');print('END',name,'exit',r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
if __name__=='__main__':main()
