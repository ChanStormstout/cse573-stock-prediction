import unittest,tempfile
from pathlib import Path
import numpy as np
from common import validate,new_run
from forecast import FactCorrection,review_gate,aggregate,validate_oof_ledger
class Tests(unittest.TestCase):
 def test_schema_and_fabricated_number(self):
  r={'sentences':{'S0':'Apple target raised from $100 to $120.'}}
  e={'kind':'target_price','action':'raise','old':'100','new':'120','unit':'USD','evidence_ids':['S0']}
  self.assertTrue(validate({'events':[e]},r)[0]);e['new']='150';self.assertFalse(validate({'events':[e]},r)[0])
 def test_wrong_numeric_direction(self):
  e={'kind':'target_price','action':'lower','old':'100','new':'120','unit':'USD','evidence_ids':['S0']}
  self.assertFalse(validate({'events':[e]},{'sentences':{'S0':'100 120'}})[0])
 def test_invalid_evidence(self):
  e={'kind':'rating','action':'maintain','old':None,'new':'Buy','unit':'rating','evidence_ids':['S99']}
  self.assertFalse(validate({'events':[e]},{'sentences':{}})[0])
 def test_unknown_is_empty(self):self.assertTrue(validate({'events':[]},{'sentences':{}})[0])
 def test_no_event_exact_fallback(self):
  m=FactCorrection();p=np.array([0.,.2,1.]);out=m.predict(np.zeros((3,12)),p,[False]*3);np.testing.assert_array_equal(out,p);self.assertIsNot(out,p)
 def test_weight_zero_fallback(self):
  p=np.array([.2,.7]);np.testing.assert_array_equal(FactCorrection().predict(np.ones((2,12)),p,[True]*2,0),p)
 def test_future_rejected(self):
  r={'symbol':'AAPL','record_key':'a','available_utc':'2018-03-02T00:00:00+00:00'}
  with self.assertRaisesRegex(ValueError,'Future news'):aggregate([r],'AAPL','2018-03-01T00:00:00+00:00',{'a'},[], '2018-02-01T00:00:00+00:00')
 def test_future_adapter_rejected(self):
  with self.assertRaisesRegex(ValueError,'Extractor'):aggregate([],'AAPL','2018-03-01T00:00:00+00:00',set(),[], '2018-04-01T00:00:00+00:00')
 def test_independent_gate_restart(self):
  reviews=[{'id':str(i),'input_sha256':str(i),'kind':'rating','reviewer':'assistant','event_group':str(i),'critical_correct':1,'missed':0,'uncertain':0} for i in range(40)]
  expected={str(i):str(i) for i in range(40)}
  for _ in range(2):self.assertEqual(review_gate(reviews,expected,('rating',))['accepted_types'],[])
 def test_duplicate_review_does_not_pass(self):
  rs=[{'id':str(i),'input_sha256':str(i),'kind':'rating','reviewer':'reviewer_1','event_group':'same','critical_correct':1,'missed':0,'uncertain':0} for i in range(40)]
  self.assertEqual(review_gate(rs,{str(i):str(i) for i in range(40)},('rating',))['status'],'WAITING_OR_FAILED')
 def test_fingerprint(self):
  with self.assertRaisesRegex(ValueError,'fingerprint'):review_gate([{'id':'a','input_sha256':'bad'}],{'a':'good'})
 def test_training_gate(self):
  with self.assertRaisesRegex(ValueError,'quality'):FactCorrection().fit(None,None,None,None,None,None,{'status':'WAITING'},100)
 def test_immutable_run(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'run';new_run(p)
   with self.assertRaises(FileExistsError):new_run(p)
 def test_in_sample_rejected(self):
  r={'cutoff_utc':'2018-03-01T00:00:00+00:00','base_training_end':'2018-03-02T00:00:00+00:00'}
  with self.assertRaisesRegex(ValueError,'OOF'):validate_oof_ledger([r],'2018-04-01T00:00:00+00:00')
 def test_duplicate_events_do_not_double_weight(self):
  e={'kind':'target_price','action':'raise','old':'100','new':'120','unit':'USD','evidence_ids':['S0']}
  r={'symbol':'AAPL','record_key':'a','event_group':'same','available_utc':'2018-03-01T00:00:00+00:00','published_utc':'2018-03-01T00:00:00+00:00','event':e}
  args=('AAPL','2018-03-01T01:00:00+00:00',{'a'},['target_price'],'2018-02-01T00:00:00+00:00')
  x,g=aggregate([r],*args);xx,gg=aggregate([r,r],*args);np.testing.assert_array_equal(x,xx);self.assertEqual(g,gg)
 def test_old_news_shrinks(self):
  e={'kind':'rating','action':'raise','old':'Hold','new':'Buy','unit':'rating','evidence_ids':['S0']}
  r={'symbol':'AAPL','record_key':'a','event_group':'one','available_utc':'2018-03-01T00:00:00+00:00','published_utc':'2018-03-01T00:00:00+00:00','event':e}
  x,_=aggregate([r],'AAPL','2018-03-01T01:00:00+00:00',{'a'},['rating'],'2018-02-01T00:00:00+00:00')
  old,_=aggregate([r],'AAPL','2018-03-02T01:00:00+00:00',{'a'},['rating'],'2018-02-01T00:00:00+00:00')
  self.assertLess(np.linalg.norm(old),np.linalg.norm(x)/10)
 def test_input_trim_preserves_current_evidence(self):
  from current_context import trim
  r={'sentences':{'S0':'Apple target raised','S1':'Apple target raised from 100 to 120.','S2':'Several other analysts also commented.','S3':'Old rating Buy.'},'spans':{k:{} for k in ['S0','S1','S2','S3']}}
  self.assertEqual(list(trim(r)['sentences']),['S0','S1'])
if __name__=='__main__':unittest.main()
