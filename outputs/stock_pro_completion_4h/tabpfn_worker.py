"""Isolated pre-existing sklearn1.7/TabPFN6.3 runtime; no network or telemetry."""
import os,argparse,time,json
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import numpy as np,joblib,torch
from tabpfn import TabPFNClassifier
from threadpoolctl import threadpool_limits
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','predict']);p.add_argument('bundle');p.add_argument('model');p.add_argument('output');p.add_argument('--seed',type=int,default=573);a=p.parse_args();z=np.load(a.bundle);torch.set_num_threads(2)
 with threadpool_limits(2):
  t=time.monotonic()
  if a.mode=='fit':
   m=TabPFNClassifier(n_estimators=8,device='cpu',model_path=str(ROOT/'work/stock-data/goal60_4h/tabpfn_model/tabpfn-v2.5-classifier-v2.5_default.ckpt'),random_state=a.seed,n_preprocessing_jobs=1);m.fit(z['x'],z['y']);joblib.dump(m,a.model,compress=3)
  else:m=joblib.load(a.model)
  q=m.predict_proba(z['evaluation'])[:,1];np.save(a.output,q)
  Path(a.output+'.json').write_text(json.dumps(dict(mode=a.mode,seconds=time.monotonic()-t,fits=int(a.mode=='fit'))))
if __name__=='__main__':main()
