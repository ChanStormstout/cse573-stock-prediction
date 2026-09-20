"""Synthetic tests: no real predictive fit and no hardcoded PASS predicates."""
import unittest
import numpy as np,pandas as pd
from common import Features,apply_fallback

def allowed_analogy(query,candidate,allowed):
 if candidate['index'] not in allowed:raise ValueError('held-out outcome')
 if pd.Timestamp(candidate['end'])>=pd.Timestamp(query['cutoff']):raise ValueError('unmatured outcome')
 if pd.Timestamp(candidate['available'])>pd.Timestamp(query['cutoff']):raise ValueError('future news')
 return True

class Contracts(unittest.TestCase):
 def test_exact_fallback(self):
  p=np.array([.2,.9]);base=np.array([.123456789012345,.4]);out=apply_fallback('FINBERT',p,np.array([True,False]),base)
  self.assertEqual(out[0],base[0]);self.assertEqual(out[1],p[1]);np.testing.assert_array_equal(p,[.2,.9])
 def test_paper_not_overridden(self):
  p=np.array([.2]);np.testing.assert_array_equal(apply_fallback('PAPER',p,[True],[.7]),p)
 def test_heldout_label_rejected(self):
  with self.assertRaisesRegex(ValueError,'held-out'):allowed_analogy({'cutoff':'2018-05-02Z'.replace('Z','T00:00Z')},{'index':3,'end':'2018-05-01T00:00Z','available':'2018-04-30T00:00Z'},{2})
 def test_unfinished_return_rejected(self):
  with self.assertRaisesRegex(ValueError,'unmatured'):allowed_analogy({'cutoff':'2018-05-01T00:00Z'},{'index':3,'end':'2018-05-02T00:00Z','available':'2018-04-30T00:00Z'},{3})
 def test_future_article_rejected(self):
  with self.assertRaisesRegex(ValueError,'future news'):allowed_analogy({'cutoff':'2018-05-02T00:00Z'},{'index':3,'end':'2018-05-01T00:00Z','available':'2018-05-03T00:00Z'},{3})
 def test_valid_past(self):
  self.assertTrue(allowed_analogy({'cutoff':'2018-05-02T00:00Z'},{'index':3,'end':'2018-05-01T00:00Z','available':'2018-04-30T00:00Z'},{3}))

if __name__=='__main__':unittest.main(verbosity=2)
