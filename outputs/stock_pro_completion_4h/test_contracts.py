"""Synthetic tests only; no course labels fitted."""
import unittest,tempfile
from models import *
class Contracts(unittest.TestCase):
 def test_offset_exact_fallback(self):
  z=np.array([[1.,0.],[0.,1.],[-1.,1.],[0.,0.]])*2;p=np.array([.2,.8,.6,.4]);model=Offset().fit(z,[0,1,0,1],p,.1);g=np.array([1,1,0,0],bool);q=model.predict(z,p,g);self.assertTrue(np.array_equal(q[~g],p[~g]))
 def test_offset_reload(self):
  z=np.arange(24).reshape(12,2)/24;p=np.linspace(.2,.8,12);model=Offset().fit(z,np.arange(12)%2,p,.1)
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'m.joblib';joblib.dump(model,path);self.assertTrue(np.array_equal(model.predict(z,p,np.ones(12,bool)),joblib.load(path).predict(z,p,np.ones(12,bool))))
 def test_combined_kernel(self):
  rng=np.random.default_rng(1);s=rng.normal(size=(8,4));v=rng.normal(size=(8,3));t=np.einsum('ij,ik->ijk',s,v).reshape(8,-1);np.testing.assert_allclose(t@t.T,(s@s.T)*(v@v.T),atol=1e-12)
 def test_nb_only_training_counts(self):
  x=sparse.csr_matrix([[1,0],[1,1],[0,1],[0,1]]);n=NBRatio().fit(x,[1,1,0,0]);before=n.r.copy();n.transform(sparse.csr_matrix([[999,222]]));self.assertTrue(np.array_equal(before,n.r))
 def test_tabpfn_synthetic_serialization(self):
  rng=np.random.default_rng(573);x=rng.normal(size=(40,6));y=(x[:,0]>0).astype(int)
  with tempfile.TemporaryDirectory() as t:
   model=TabPFNRemote(Path(t)/'model.bin',573).fit(x[:30],y[:30]);p=model.predict_proba(x[30:]);q=model.predict_proba(x[30:]);self.assertEqual(model.last_execution['fits'],0);np.testing.assert_allclose(p,q,atol=1e-5,rtol=0)
if __name__=='__main__':unittest.main()
