"""Fit-free contract and corruption tests."""
import unittest,tempfile
from unittest.mock import patch
from core import *
class Contracts(unittest.TestCase):
 def test_text_information(self):
  parts,_=clean('Apple will not cut 10%.','Apple gains gains $200.50. Amazon has no cut.\nNor a 5% decline.')
  u=features(parts,False);b=features(parts,True)
  self.assertEqual(u[:5],['apple','will','not','cut','10%']);self.assertEqual(u.count('gains'),2)
  self.assertIn('$200.50',u);self.assertTrue({'no','nor','not','5%'}<=set(u))
  self.assertNotIn('10% apple',b);self.assertNotIn('$200.50 amazon',b);self.assertNotIn('cut nor',b)
  self.assertIn('gains gains',b)
 def test_shrink_fallback_direction(self):
  p=np.array([.1,.499999999,.5,.8]);h=np.array([False,True,True,True])
  for a in [.25,.5,.75,1.]:
   q=shrink(p,h,a);self.assertEqual(q[0],p[0]);self.assertTrue(np.array_equal(p>=.5,q>=.5))
 def test_incomplete_refusal(self):
  with tempfile.TemporaryDirectory() as x:
   with self.assertRaises(RuntimeError):block_guard(Path(x),'seal')
 def test_cache_corruption_refusal(self):
  with patch('core.sha',return_value='new'):
   for key in ['protocol','inputs','protected','code']:
    stale={'protocol':'new','inputs':'new','protected':'new','code':{'core.py':'new','train.py':'new'}};stale[key]='changed'
    with self.assertRaises(ValueError):cache_check(stale)
 def test_completed_hash_corruption(self):
  with tempfile.TemporaryDirectory() as x:
   root=Path(x);(root/'p.csv').write_text('original');dump(root/'complete.json',{'seal':'s','files':{'p.csv':sha(root/'p.csv')}})
   self.assertFalse(block_guard(root,'s'));(root/'p.csv').write_text('modified')
   with self.assertRaises(AssertionError):block_guard(root,'s')
if __name__=='__main__':unittest.main()
