"""Equivalent binary LINEAR SVC decision computation; solver/fit unchanged."""
import numpy as np
from scipy import sparse
from sklearn.svm import SVC
ORIGINAL=SVC.decision_function

def linear_decision(self,x):
 if self.kernel!='linear' or len(self.classes_)!=2:return ORIGINAL(self,x)
 w=self.coef_;out=x@w.T
 if sparse.issparse(out):out=out.toarray()
 return np.asarray(out).reshape(-1)+self.intercept_[0]
def install():SVC.decision_function=linear_decision

def test_equivalence():
 from sklearn.calibration import CalibratedClassifierCV
 from sklearn.model_selection import StratifiedKFold
 rng=np.random.default_rng(573);x=rng.normal(size=(160,128));x[rng.random(x.shape)<.6]=0;x=sparse.csr_matrix(x);y=rng.integers(0,2,160);errors=[]
 for c in [.01,.1,1.]:
  m=SVC(kernel='linear',C=c,random_state=573).fit(x[:120],y[:120]);errors.append(float(np.max(abs(ORIGINAL(m,x[120:])-linear_decision(m,x[120:])))))
 assert max(errors)<1e-10,errors
 return dict(status='PASS',maximum_margin_error=max(errors),C=[.01,.1,1.],training_solver_changed=False,synthetic_only=True)
if __name__=='__main__':
 import json;print(json.dumps(test_equivalence()))

class DenseLinearSVC(SVC):
 """Same libsvm linear objective; dense BLAS training and exact linear margins."""
 def fit(self,x,y,sample_weight=None):
  return super().fit(x.toarray() if sparse.issparse(x) else x,y,sample_weight=sample_weight)
 def decision_function(self,x):return linear_decision(self,x)
