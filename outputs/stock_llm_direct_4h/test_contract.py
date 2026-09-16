import copy,json,unittest
from common import messages,parse_forecast,validate_boundary

def row():
 return dict(symbol='AAPL',cutoff='2018-09-04T13:25:00+00:00',interval_start='2018-09-04T13:30:00+00:00',interval_end='2018-09-04T17:30:00+00:00',timezone='UTC',price_rows=[dict(id=f'P{i}',start='2018-08-31T13:30:00+00:00',end='2018-08-31T14:30:00+00:00') for i in range(1,7)],price_summary={},news=[dict(id='N1',record_key='private',published_at='2018-09-04T10:00:00+00:00',available_at='2018-09-04T11:00:00+00:00',title='source')],news_summary={})
class Tests(unittest.TestCase):
 def test_whitelist_no_labels_or_target_prices(self):
  x=row();before=messages(x,'joint');x.update(label=1,target_return=999,opening=777,close=888,split='test',key='key')
  self.assertEqual(messages(x,'joint'),before);self.assertNotIn('private',json.dumps(before))
 def test_price_only(self):
  obj=json.loads(messages(row(),'price')[-1]['content']);self.assertNotIn('news',obj);self.assertIn('price_rows',obj)
 def test_news_only(self):
  obj=json.loads(messages(row(),'news')[-1]['content']);self.assertNotIn('price_rows',obj);self.assertIn('news',obj)
 def test_valid_and_evidence(self):
  x=parse_forecast('{"p_up":0.6,"direction":"UP","evidence_ids":["N1"]}',row(),'joint');self.assertTrue(x['valid']);self.assertTrue(x['evidence_valid'])
 def test_inconsistent_direction_falls_back(self):
  x=parse_forecast('{"p_up":0.6,"direction":"DOWN"}',row(),'joint');self.assertFalse(x['valid']);self.assertEqual(x['p'],.5)
 def test_bad_probability(self):
  for p in ['true','NaN','1.1','-0.1','"0.8"']:
   self.assertFalse(parse_forecast('{"p_up":'+p+',"direction":"UP"}',row(),'joint')['valid'])
 def test_no_partial_json_salvage(self):
  self.assertFalse(parse_forecast('x {"p_up":0.6,"direction":"UP"}',row(),'joint')['valid'])
 def test_citation_separate(self):
  x=parse_forecast('{"p_up":0.6,"direction":"UP","evidence_ids":["N1"]}',row(),'price');self.assertTrue(x['valid']);self.assertFalse(x['evidence_valid'])
 def test_boundary(self):validate_boundary(row())
 def test_future_price(self):
  x=row();x['price_rows'][0]['end']='2018-09-04T14:30:00+00:00'
  with self.assertRaises(ValueError):validate_boundary(x)
 def test_future_available(self):
  x=row();x['news'][0]['available_at']='2018-09-04T13:26:00+00:00'
  with self.assertRaises(ValueError):validate_boundary(x)
 def test_published_after_crawled(self):
  x=row();x['news'][0]['published_at']='2018-09-04T12:00:00+00:00'
  with self.assertRaises(ValueError):validate_boundary(x)
 def test_outside_news_window(self):
  x=row();x['news'][0]['available_at']='2018-09-04T09:25:00+00:00'
  with self.assertRaises(ValueError):validate_boundary(x)
 def test_wrong_target_definition(self):
  x=row();x['interval_end']='2018-09-04T18:30:00+00:00'
  with self.assertRaises(ValueError):validate_boundary(x)
class ChoiceTests(unittest.TestCase):
 def test_choice_has_no_verbal_probability_example(self):
  from choice import choice_messages
  m=choice_messages(row(),'joint');self.assertNotIn('p_up',m[0]['content']);self.assertNotIn('0.5',m[0]['content'])
 def test_choice_user_information_identical(self):
  from choice import choice_messages
  for variant in ['price','news','joint']:
   self.assertEqual(choice_messages(row(),variant)[-1],messages(row(),variant)[-1])
 def test_choice_no_outcome_leak(self):
  from choice import choice_messages
  r=row();a=choice_messages(r,'joint');r.update(label=1,target_return=888,opening_price=999)
  self.assertEqual(a,choice_messages(r,'joint'))
if __name__=='__main__':unittest.main()
