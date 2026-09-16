import unittest
import numpy as np,pandas as pd
from features_price import *
from run import ROOT,prefix,metrics
from text_vectorizer import DeterministicTfidf
from config import Config
from sklearn.linear_model import LogisticRegression
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.b,cls.c=load_sources(ROOT/'work/stock-data','AAPL')
 def snapshot(self,b=None,cut='2018-06-04 14:25Z',start='2018-06-04 14:30Z'):
  return build_price_snapshot('AAPL',pd.Timestamp(start),pd.Timestamp(cut),self.b if b is None else b,self.c)
 def test_future_poison(self):
  b=self.b.copy();b.loc[b.end>pd.Timestamp('2018-06-04 14:25Z'),['open','high','low','close']]=999999
  self.assertEqual(self.snapshot().values,self.snapshot(b).values)
 def test_open_unknown(self):
  f=self.snapshot(cut='2018-06-04 13:25Z',start='2018-06-04 13:30Z')
  self.assertEqual(f.masks['opening_gap_valid'],0);self.assertEqual(f.masks['session_return_valid'],0)
 def test_missing_bar(self):
  b=self.b.drop(pd.Timestamp('2018-06-04 14:15Z'));f=self.snapshot(b)
  self.assertEqual(f.masks['recent15_valid'],0);self.assertEqual(f.masks['session_return_valid'],0)
 def test_future_asof(self):
  with self.assertRaises(ValueError):build_price_snapshot('AAPL',pd.Timestamp('2018-06-04 14:30Z'),pd.Timestamp('2018-06-04 14:25Z'),self.b,self.c,asof_end=pd.Timestamp('2018-06-04 14:30Z'))
 def test_vocabulary_ties(self):
  x=['apple banana','banana carrot','carrot apple'];a=DeterministicTfidf(2).fit(x);b=DeterministicTfidf(2).fit(x[::-1]);self.assertEqual(a.vocabulary_,b.vocabulary_);self.assertEqual(a.transform(['futureunknown']).nnz,0)
 def test_config(self):
  with self.assertRaises(TypeError):Config(unknown=True)
  with self.assertRaises(ValueError):Config(quote_scope='other')
 def test_future_labels(self):
  d=pd.DataFrame({'month':['2018-05','2018-06'],'end_utc':pd.to_datetime(['2018-06-05','2018-06-04'],utc=True),'cutoff_utc':pd.to_datetime(['2018-05-01','2018-06-01'],utc=True)})
  with self.assertRaises(ValueError):prefix(d,'2018-06')
 def test_average_loss(self):
  x=np.arange(20).reshape(-1,1)/20;y=np.array([0,1]*10)
  def f(x,y):return LogisticRegression(solver='liblinear',C=1/(.02*len(y)),tol=1e-12).fit(x,y).predict_proba([[.2],[.8]])
  np.testing.assert_allclose(f(x,y),f(np.tile(x,(3,1)),np.tile(y,3)),atol=1e-8)
class SemanticChecks(unittest.TestCase):
 def test_offset_exact_fallback(self):
  from semantic_components import FixedBaseOffset
  p=np.array([.123456789,.654321987]);m=FixedBaseOffset();m.coef_=np.array([100.]);q=m.predict(p,[[1.],[1.]],[False,False]);np.testing.assert_array_equal(p,q)
 def test_offset_insufficient_dates(self):
  from semantic_components import FixedBaseOffset
  m=FixedBaseOffset().fit([.5]*10,np.ones((10,2)),[0,1]*5,[True]*10,list(range(10)));self.assertEqual(m.status_,'INSUFFICIENT_DATES');np.testing.assert_array_equal(m.coef_,[0,0])
 def test_calibration_monotone(self):
  from semantic_components import ForwardCalibrator
  m=ForwardCalibrator().fit(np.linspace(.1,.9,40),[1,0]*20);self.assertGreaterEqual(m.coef_[1],0)
 def test_review_resume_stops(self):
  from semantic_components import require_independent_review
  from pathlib import Path
  for _ in range(2):
   with self.assertRaises(ValueError):require_independent_review(Path(__file__).parent/'review/blind_review_40.csv')
if __name__=='__main__':unittest.main()
