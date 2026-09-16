import unittest
from copy import deepcopy
from deduplicate import group_labels

class Grouping(unittest.TestCase):
 def setup_case(self):
  e={'kind':'target_price','action':'raise','old':'180','new':'200','unit':'USD','evidence_ids':['S0']}
  inputs={k:{'symbol':'AAPL','published_utc':t,'available_utc':t,'sentences':{'S0':'UBS raised Apple target from $180 to $200.'}} for k,t in [('A','2018-02-28T12:00:00+00:00'),('B','2018-03-01T12:00:00+00:00')]}
  labels=[{'id':k,'split':s,'answer':{'events':[deepcopy(e)]}} for k,s in [('A','train'),('B','valid')]]
  return labels,inputs
 def test_cross_split_keeps_past(self):
  ls,ins=self.setup_case();keep,drop,links=group_labels(ls,ins)
  self.assertEqual([r['id'] for r in keep],['A']);self.assertEqual(drop[0]['kept_split'],'train');self.assertTrue(links)
 def test_distinct_broker_not_merged(self):
  ls,ins=self.setup_case();ins['B']['sentences']['S0']=ins['B']['sentences']['S0'].replace('UBS','Barclays')
  self.assertEqual(len(group_labels(ls,ins)[0]),2)
 def test_distinct_amount_not_merged(self):
  ls,ins=self.setup_case();ls[1]['answer']['events'][0]['new']='210'
  self.assertEqual(len(group_labels(ls,ins)[0]),2)
 def test_later_repeat_outside_window_not_merged(self):
  ls,ins=self.setup_case();ins['B']['published_utc']='2018-04-01T12:00:00+00:00'
  self.assertEqual(len(group_labels(ls,ins)[0]),2)
 def test_unknown_broker_not_guessed(self):
  ls,ins=self.setup_case()
  for v in ins.values():v['sentences']['S0']='Apple target raised from $180 to $200.'
  self.assertEqual(len(group_labels(ls,ins)[0]),2)
 def test_no_events_remain(self):
  ls,ins=self.setup_case()
  for r in ls:r['answer']['events']=[]
  self.assertEqual(len(group_labels(ls,ins)[0]),2)
if __name__=='__main__':unittest.main()
