"""Fixed synthetic-only TabPFN2.5; private conditioning bundles, no telemetry."""
import os
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from core import *

def run(cross):
 name='P_cross_tabpfn' if cross else 'P_own_tabpfn';d=data();x=d[OLD+RECENT].to_numpy(float)
 if cross:x=np.c_[x,pd.read_pickle(PRIVATE/'cross.pkl').to_numpy()]
 checkpoint=W/'goal60_4h/tabpfn_model/tabpfn-v2.5-classifier-v2.5_default.ckpt';h=sha(checkpoint);dest=PRIVATE/name;dest.mkdir(parents=True,exist_ok=True)
 def create():return TabPFNClassifier(n_estimators=1,device='cpu',model_path=str(checkpoint),random_state=573,n_preprocessing_jobs=1)
 def builder(tr,ev,c):
  model=create();model.fit(x[tr],d.iloc[tr].label.to_numpy());p=model.predict_proba(x[ev])[:,1];path=dest/f'conditioning_{tr[-1]}.npz';np.savez_compressed(path,x=x[tr],y=d.iloc[tr].label.to_numpy(),eval_x=x[ev],eval_indices=ev)
  # Reload conditioning data and a fresh frozen prior on a fixed subset.
  z=np.load(path);reloaded=create();reloaded.fit(z['x'],z['y']);p2=reloaded.predict_proba(z['eval_x'][:3])[:,1];err=float(np.max(abs(p2-p[:3])));assert err<1e-5
  return p,dict(conditioning_bundle=path.name,sha256=sha(path),checkpoint_sha256=h,reload_error=err,gradient_training=False,device='cpu')
 torch.set_num_threads(4)
 with threadpool_limits(limits=4):run_model(name,builder,candidates=[1.])
if __name__=='__main__':
 for cross in (False,True):run(cross)
