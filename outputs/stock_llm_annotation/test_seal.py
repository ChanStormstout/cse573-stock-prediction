import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from seal import seal
from labels import dump,write_rows
class SealChecks(unittest.TestCase):
 def fixture(self,p):
  inp={'id':'X','input_sha256':'h','split':'train','symbol':'AAPL','group':'g','available_utc':'2018-01-01T00:00:00+00:00','published_utc':'2018-01-01T00:00:00+00:00','sentences':{'S0':'No event.'}}
  label={'id':'X','status':'none','events':[],'reason':'No event','context_sufficient':True}
  d=p/'comparisons/b';d.mkdir(parents=True);dump(d/'summary.json',{'agreement_audit_ids':['X']});write_rows(d/'comparison.jsonl',[{'id':'X','state':'AGREED_PROVISIONAL','a':label,'b':label}]);decision={'id':'X','input_sha256':'h','label':label,'reviewer':'assistant','rationale':'Read evidence'}
  write_rows(p/'decisions.jsonl',[decision]);return inp,decision
 def run_seal(self,p,inp):
  with patch('seal.load_panel',return_value=([inp],{'inputs_sha256':'ih','extractor_earliest_freeze':'2018-05-01T00:00:00+00:00'})):
   seal(p,p/'comparisons',p/'decisions.jsonl',p/'out')
 def test_missing_agreement_audit_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);inp,_=self.fixture(p);write_rows(p/'decisions.jsonl',[])
   with self.assertRaisesRegex(ValueError,'Unresolved'):self.run_seal(p,inp)
 def test_duplicate_adjudication_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);inp,d=self.fixture(p);write_rows(p/'decisions.jsonl',[d,d])
   with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_seal(p,inp)
 def test_stale_adjudication_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);inp,d=self.fixture(p);d['input_sha256']='wrong';write_rows(p/'decisions.jsonl',[d])
   with self.assertRaisesRegex(ValueError,'provenance'):self.run_seal(p,inp)
 def test_missing_comparison_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);inp,d=self.fixture(p);write_rows(p/'comparisons/b/comparison.jsonl',[])
   with self.assertRaisesRegex(ValueError,'Incomplete'):self.run_seal(p,inp)
if __name__=='__main__':unittest.main()
