import json
import unittest
from labels import validate_label, parse_reply

class Tests(unittest.TestCase):
    def setUp(self):
        self.inp={'id':'X','sentences':{'S0':'Amazon maintained at Buy; target cut from $200 to $180.'}}
        self.label={'id':'X','status':'events','events':[{'kind':'target_price','action':'lower','old':'200','new':'180','unit':'USD','evidence_ids':['S0']}],'reason':'Explicit current change','context_sufficient':True}
    def test_supported_numbers(self):
        self.assertEqual(validate_label(self.label,self.inp),[])
    def test_invented_number(self):
        self.label['events'][0]['new']='170'
        self.assertIn('unmatched_value',validate_label(self.label,self.inp))
    def test_wrong_direction(self):
        self.label['events'][0]['action']='raise'
        self.assertIn('numeric_direction',validate_label(self.label,self.inp))
    def test_foreign_evidence(self):
        self.label['events'][0]['evidence_ids']=['S9']
        self.assertIn('evidence_id',validate_label(self.label,self.inp))
    def test_uncertainty_not_negative(self):
        self.label.update(status='none',events=[],context_sufficient=False)
        self.assertIn('missing_context_is_not_negative',validate_label(self.label,self.inp))
    def test_uncertain_valid(self):
        self.label.update(status='uncertain',events=[],context_sufficient=False)
        self.assertEqual(validate_label(self.label,self.inp),[])
    def test_events_status(self):
        self.label['status']='none'
        self.assertIn('events_status',validate_label(self.label,self.inp))
    def test_incomplete_json_rejected(self):
        with self.assertRaises(json.JSONDecodeError): parse_reply('{"id":"X"')
    def test_fenced_jsonl(self):
        self.assertEqual(parse_reply('```jsonl\n'+json.dumps(self.label)+'\n```'),[self.label])
    def test_prose_not_silently_stripped(self):
        with self.assertRaises(json.JSONDecodeError): parse_reply('These are good\n'+json.dumps(self.label))

if __name__=='__main__':unittest.main()
