import unittest
from contracts import values,compile_items,validate_gate,messages

class ContractTests(unittest.TestCase):
    def test_prices_forward(self): self.assertEqual(values('cut price target from $240 to $225','target_price'),('240','225'))
    def test_prices_reverse(self): self.assertEqual(values('raised target to $225 from $210','target_price'),('210','225'))
    def test_single_new(self): self.assertEqual(values('set a target of $164','target_price'),(None,'164'))
    def test_no_currency(self):
        with self.assertRaises(ValueError): values('raised target to 225','target_price')
    def test_stock_price(self):
        with self.assertRaises(ValueError): values('shares closed at $164','target_price')
    def test_mixed_values(self):
        with self.assertRaises(ValueError): values('target $164 and share price $150','target_price')
    def test_rating_pair(self): self.assertEqual(values('downgraded from "Strong Buy" to "Buy"','rating'),('Strong Buy','Buy'))
    def test_maintain_missing_old(self): self.assertEqual(values('reiterated Buy','rating'),(None,'Buy'))
    def test_negation(self):
        with self.assertRaises(ValueError): values('did not raise price target to $225','target_price')
    def test_quote_and_gate(self):
        row={'symbol':'AAPL','sentences':{'S0':'set target $164'}}
        with self.assertRaises(ValueError): validate_gate({'status':'none','evidence_ids':['S0']},row)
        obj={'items':[dict(kind='target_price',action='unknown',evidence_id='S0',quote='set target $165')]}
        answer,rejected=compile_items(obj,row,['S0']);self.assertEqual(answer,{'events':[]});self.assertEqual(len(rejected),1)
    def test_unknown_preserved(self):
        row={'sentences':{'S0':'set target $164'}}
        obj={'items':[dict(kind='target_price',action='unknown',evidence_id='S0',quote='set target $164')]}
        answer,rejected=compile_items(obj,row,['S0']);self.assertFalse(rejected);self.assertEqual(answer['events'][0]['action'],'unknown')
    def test_original_unchanged(self):
        from model_contract import messages as old
        row={'symbol':'AAPL','sentences':{'S0':'news'}}
        self.assertEqual(messages(row,'original'),old(row))

    def test_delta_rule_no_inferred_old(self):
        from rules_v2 import compile_v2
        row={'sentences':{'S0':'lowered price target by $2 to $168'}}
        obj={'items':[dict(kind='target_price',action='lower',evidence_id='S0',quote=row['sentences']['S0'])]}
        result,rejected,repairs=compile_v2(obj,row,['S0'])
        self.assertEqual(result['events'][0]['old'],None);self.assertEqual(result['events'][0]['new'],'168')
        self.assertFalse(rejected);self.assertEqual(len(repairs),1)

    def test_delta_rule_three_values_rejected(self):
        from rules_v2 import compile_v2
        row={'sentences':{'S0':'lowered target by $2 to $168; shares $155'}}
        obj={'items':[dict(kind='target_price',action='lower',evidence_id='S0',quote=row['sentences']['S0'])]}
        result,rejected,repairs=compile_v2(obj,row,['S0']);self.assertFalse(result['events']);self.assertTrue(rejected)

    def test_partial_rating_pair_never_reversed(self):
        from safety_repair import repair
        event=dict(kind='rating',action='lower',old=None,new='Positive',unit='rating',evidence_ids=['S0'])
        pred={'valid':True,'parsed':{'events':[event]},'calls':[{}, {'parsed':{'items':[dict(kind='rating',action='lower',evidence_id='S0',quote='cut to Mixed from Positive')]}}],'rejected':[]}
        fixed,drops=repair(pred);self.assertEqual(fixed['parsed']['events'],[]);self.assertEqual(len(drops),1)
        self.assertEqual(len(pred['parsed']['events']),1)

    def test_numeric_pair_not_a_rating_pair(self):
        from safety_repair import repair
        event=dict(kind='rating',action='maintain',old=None,new='overweight',unit='rating',evidence_ids=['S0'])
        pred={'valid':True,'parsed':{'events':[event]},'calls':[{}, {'parsed':{'items':[dict(kind='rating',action='maintain',evidence_id='S0',quote='reiterated overweight but trimmed her price target to 200 from 203')]}}],'rejected':[]}
        fixed,drops=repair(pred);self.assertEqual(fixed['parsed']['events'],[event]);self.assertFalse(drops)

if __name__=='__main__':unittest.main()
