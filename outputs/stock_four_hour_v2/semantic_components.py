"""Tested building blocks for the gated text experiment; not an executed text trial.
Inputs to calibration MUST be chronological out-of-fold base probabilities.
PCA coordinates MUST be fitted separately for each training prefix.
"""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.decomposition import PCA

def require_independent_review(path):
 d=pd.read_csv(path,keep_default_na=False)
 required={'sample_id','subject_correct','reviewer','template_or_mixed','missed_target'}
 if not required.issubset(d.columns) or len(d)!=40 or d.sample_id.duplicated().any():raise ValueError('Incomplete 40-article review')
 good=pd.to_numeric(d.subject_correct,errors='coerce')
 if not good.isin([0,1]).all() or not d.reviewer.str.strip().ne('').all():raise ValueError('WAITING_INDEPENDENT_REVIEW')
 if d.reviewer.str.contains('assistant|codex|chatgpt|gpt',case=False).any():raise ValueError('Assistant review cannot substitute for independent review')
 if not (d.template_or_mixed.str.strip().ne('').all() and d.missed_target.str.strip().ne('').all()):raise ValueError('Coverage/quality annotations incomplete')
 if good.mean()<.9:raise ValueError('Quality gate failed')
 return {'n':40,'subject_accuracy':float(good.mean()),'review_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'caveat':'Reviewer identity is declared; this function cannot independently authenticate the human reviewer.'}

class ForwardCalibrator:
 def fit(self,p,y):
  t=logit(np.clip(p,1e-6,1-1e-6));y=np.asarray(y);x=np.column_stack([np.ones(len(t)),t])
  def objective(w):
   z=x@w;loss=np.mean(np.logaddexp(0,z)-y*z)+.1/2*np.dot(w,w);grad=x.T@(expit(z)-y)/len(y)+.1*w
   return loss,grad
  r=minimize(objective,[0.,1.],jac=True,bounds=[(None,None),(0,None)],method='L-BFGS-B')
  if not r.success:raise RuntimeError(r.message)
  self.coef_=r.x;return self
 def predict(self,p):return expit(self.coef_[0]+self.coef_[1]*logit(np.clip(p,1e-6,1-1e-6)))

class FixedBaseOffset:
 def fit(self,calibrated_p,x,y,gate,days):
  x=np.asarray(x);y=np.asarray(y);gate=np.asarray(gate,dtype=bool);days=np.asarray(days);self.coef_=np.zeros(x.shape[1]);self.status_='INSUFFICIENT_DATES'
  if len(set(days[gate]))<20 or any(len(set(days[gate&(y==v)]))<5 for v in [0,1]):return self
  # All windows stay in the objective; invalid windows contribute fixed-base loss only.
  x=x*gate[:,None];offset=logit(np.clip(calibrated_p,1e-6,1-1e-6))
  def objective(w):
   z=offset+x@w;return np.mean(np.logaddexp(0,z)-y*z)+.1/2*np.dot(w,w),x.T@(expit(z)-y)/len(y)+.1*w
  r=minimize(objective,self.coef_,jac=True,method='L-BFGS-B')
  if not r.success:raise RuntimeError(r.message)
  self.coef_=r.x;self.status_='FIT';return self
 def predict(self,calibrated_p,x,gate):
  p=np.array(calibrated_p,copy=True,dtype=float);g=np.asarray(gate,dtype=bool)
  # Preserve no-article outputs exactly, without round-trip through logit/sigmoid.
  p[g]=expit(logit(np.clip(p[g],1e-6,1-1e-6))+np.asarray(x)[g]@self.coef_);return p

class UniqueArticlePCA:
 def fit(self,keys,embeddings):
  keys=np.asarray(keys);_,idx=np.unique(keys,return_index=True)
  if len(idx)<8:raise ValueError('Need at least eight distinct training articles')
  self.training_keys_=keys[idx].tolist();self.pca_=PCA(n_components=8,svd_solver='full').fit(np.asarray(embeddings)[idx]);return self
 def transform(self,embeddings):return self.pca_.transform(embeddings)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);a=p.parse_args();print(json.dumps(require_independent_review(a.review),indent=2))
