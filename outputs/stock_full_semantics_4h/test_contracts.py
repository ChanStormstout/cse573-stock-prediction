"""Regression checks for match/fallback/calibration boundaries."""
import unittest
import numpy as np
from train_semantics import blend
from classical import model
from prepare_semantics import PAT
class Contracts(unittest.TestCase):
 def test_zero_weight_bitwise_identity(self):
  x=np.array([0.,1.,.1234567890123456]);self.assertTrue(np.array_equal(blend(x,np.array([1.,0.,.9]),np.ones(3,bool),0.),x))
 def test_unknown_evidence_exact_full(self):
  x=np.array([.1234567890123456,.9]);z=blend(x,np.array([.99,.1]),np.array([False,True]),1.);self.assertEqual(z[0],x[0]);self.assertEqual(z[1],.1)
 def test_svm_calibration_refits_text_inside_splits(self):
  m=model('TFIDF_SVM',.1,573);self.assertEqual(m.cv.n_splits,3);self.assertIn('tfidfvectorizer',m.estimator.named_steps)
 def test_target_name_boundaries(self):
  self.assertIsNone(PAT['AAPL'].search('pineapple'));self.assertIsNone(PAT['AMZN'].search('Amazonian'));self.assertIsNotNone(PAT['AMZN'].search('Amazon maintained Buy but lowered its target.'))
if __name__=='__main__':unittest.main()
