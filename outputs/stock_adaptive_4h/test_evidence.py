import unittest,tempfile
from evidence import *

class EvidenceTests(unittest.TestCase):
    def test_separate_actions(self):
        s='RBC maintained a Buy rating on Apple and lowered its price target from $290 to $280.'
        f=extract('AAPL','x',s,s)
        target=[x for x in f if x['kind']=='analyst_target'];rating=[x for x in f if x['kind']=='analyst_rating']
        self.assertTrue(target);self.assertTrue(rating);self.assertEqual(target[0]['action'],'down');self.assertEqual(rating[0]['action'],'maintain')
    def test_coverage_comparison_not_direct(self):
        s='Currently, Apple’s market capitalization stands at $919.9 billion while Amazon’s is at $767.5 billion.'
        self.assertNotEqual(role('AMZN','Microsoft valuation',s)[0],'direct_rule_candidate')
    def test_coverage_wrong_rating_object(self):
        s='DA Davidson initiated coverage of Apple with a buy rating, despite the continued rise of Amazon and Google.'
        self.assertNotEqual(role('AMZN','Apple target',s)[0],'direct_rule_candidate')
    def test_coverage_real_missed_rating(self):
        s='Amazon (NASDAQ:AMZN) was rated outperform in new coverage from Telsey Advisory Group.'
        self.assertEqual(role('AMZN','Nightly Business Report',s)[0],'direct_rule_candidate')
    def test_unknown_actor(self):
        s='Apple had its price target lowered from $290 to $280 by researchers based in a large institution in London.'
        f=extract('AAPL','x',s,s);self.assertIsNone(f[0]['actor'])
    def test_conflict(self):
        t='RBC lowered Apple price target from $290 to $280.';b='RBC lowered Apple price target from $290 to $270.'
        f=extract('AAPL','x',t,b);self.assertTrue(all(x['conflict'] for x in f))
    def test_history_not_accepted(self):
        s='Previously, RBC lowered Apple price target from $290 to $280.'
        f=extract('AAPL','x','Apple market update',s);self.assertEqual(f[0]['temporal_role'],'historical');self.assertFalse(f[0]['prediction_accepted'])
    def test_multiple_objects_abstain(self):
        s='RBC lowered Apple and Amazon price targets from $290 to $280.'
        self.assertEqual(extract('AAPL','x',s,s),[])
    def test_currency_conflict(self):
        s='Apple price target was lowered from C$290 to $280.'
        f=extract('AAPL','x',s,s);self.assertTrue(all(x['conflict'] for x in f))
    def test_guidance_period_and_unit(self):
        s='Amazon expects revenue of $50 billion to $60 billion for the fourth quarter of 2018.'
        f=extract('AMZN','x','Amazon outlook',s);g=[x for x in f if x['kind']=='revenue_guidance'][0]
        self.assertEqual(g['lower'],50e9);self.assertEqual(g['upper'],60e9);self.assertIsNone(g['old_value']);self.assertIn('2018',g['period'])
    def test_exact_span(self):
        s='Introduction.\nRBC lowered Apple price target from $290 to $280. Other material.'
        for f in extract('AAPL','x','Apple outlook',s):self.assertEqual((s if f['source']=='body' else 'Apple outlook')[f['start']:f['end']],f['evidence'])
    def test_resume_requires_same_review(self):
        with tempfile.TemporaryDirectory() as td:
            t=Path(td);proposal=json.dumps({'event_group':'g','symbol':'AAPL','kind':'analyst_target'});h=hashlib.sha256(proposal.encode()).hexdigest();p=t/'r.csv';e=t/'e.json';e.write_text(json.dumps({'a':h}))
            d=pd.DataFrame([{'sample_id':'a','input_sha256':h,'proposed_fields':proposal,'symbol':'AAPL','kind':'analyst_target','event_group':'g','reviewer':'','critical_fields_correct':'','missed_target':'','uncertain':''}]);d.to_csv(p,index=False)
            self.assertEqual(quality_gate(p,e)['status'],'WAITING_INDEPENDENT_REVIEW');self.assertEqual(quality_gate(p,e)['status'],'WAITING_INDEPENDENT_REVIEW')
            d.loc[0,'proposed_fields']='{}';d.to_csv(p,index=False)
            with self.assertRaises(ValueError):quality_gate(p,e)

if __name__=='__main__':unittest.main()
