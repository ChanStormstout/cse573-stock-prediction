"""Synthetic tests: no real predictive fit and no hardcoded PASS predicates."""
import unittest
import tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from common import Features,apply_fallback
from retrieval import allowed_analogy
from analogy_evaluate import correction

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
 def test_zero_alpha_is_exact_identity(self):
  p=np.array([.123456789012345,.999999999999999]);np.testing.assert_array_equal(correction(p,[.9,.1],[True,True],0),p)
 def test_closed_gate_is_exact_identity(self):
  p=np.array([.123456789012345,.999999999999999]);np.testing.assert_array_equal(correction(p,[.9,.1],[False,False],.5),p)
 def test_correction_logit_bound(self):
  from scipy.special import logit
  p=np.array([.1,.5,.9]);out=correction(p,[.0001,.9,.9999],[True,True,True],.25)
  self.assertTrue(np.all(np.abs(logit(out)-logit(p))<=.25+1e-12))
 def test_whitelist_excludes_historical_model_answers(self):
  from common import load
  d,_,_,_=load()
  self.assertFalse(set(d.columns)&{'F0','F1','F2','F6','R1','target_return','has_qualified_event'})
 def test_fingerprint_corruption_is_rejected(self):
  import common
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'manifest.json').write_text(json.dumps({'sources':{'frozen':'old'},'artifacts':{}}))
   with patch.object(common,'PRIVATE',p),patch.object(common,'hashes',lambda:{'frozen':'changed'}):
    with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):common.check_sources()
 def test_prepared_artifact_corruption_is_rejected(self):
  import common
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'rows').write_text('original');h=common.sha(p/'rows');(p/'rows').write_text('corrupted')
   (p/'manifest.json').write_text(json.dumps({'sources':{},'artifacts':{'rows':h}}))
   with patch.object(common,'PRIVATE',p),patch.object(common,'hashes',lambda:{}):
    with self.assertRaisesRegex(ValueError,'artifact mismatch'):common.check_sources()
 def test_reader_rejects_invented_quote_and_number(self):
  from fact_pilot_infer import validate
  fact=dict(target='AMZN',kind='target_price',action='raise',actor=None,old=100,new=200,unit='USD',current_event=True)
  answer=dict(current_fact=fact,prior_fact=None,relation='unknown',current_evidence='fabricated quotation 100',prior_evidence=None)
  errors=validate(answer,dict(symbol='AMZN',current_passage='Amazon is discussed in this real passage.',prior_passage='Earlier unrelated text.'))
  self.assertIn('current_unsupported_quote',errors);self.assertIn('current_new_unsupported_number',errors)

if __name__=='__main__':unittest.main(verbosity=2)
