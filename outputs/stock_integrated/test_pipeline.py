import unittest,tempfile
import numpy as np,pandas as pd
from pathlib import Path
from run import combine,select_fusion,weights_grid,main
class PipelineTests(unittest.TestCase):
 def test_simplex(self):
  w=list(weights_grid());self.assertEqual(len(w),35)
  for x in w:self.assertEqual(x.sum(),1);self.assertTrue((x>=0).all())
 def test_exact_no_news(self):
  d=pd.DataFrame(dict(price=[.123456789,.9],title=[.8,.8],body=[.9,.1],semantic=[.1,.2],has_news=[0,1]));p=combine(d,np.array([0,0,0,1]));self.assertEqual(p[0],d.price[0]);self.assertEqual(p[1],.2)
 def test_ablation(self):
  for w in weights_grid('semantic'):self.assertEqual(w[3],0)
 def test_existing_run_rejected(self):
  with tempfile.TemporaryDirectory() as p:
   with self.assertRaises(FileExistsError):main(Path(p))
 def test_selection_deterministic(self):
  d=pd.DataFrame(dict(label=[0,1]*8,price=[.4,.6]*8,title=[.3,.7]*8,body=[.8,.2]*8,semantic=[.2,.8]*8,has_news=[1]*16,month=['2018-03']*8+['2018-04']*8))
  a,grid=select_fusion(d);b,_=select_fusion(d.iloc[::-1]);self.assertEqual(a,b);self.assertTrue(a['eligible'])
if __name__=='__main__':unittest.main()
