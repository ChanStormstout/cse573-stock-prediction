import unittest
from engine import *
class Contracts(unittest.TestCase):
 def test_single_changes(self):
  v={'a':article_views('Apple prices prices','Apple is not raising prices from $180 to $200.')}
  old=set(assemble(v,['a'],'CONTROL').split())
  self.assertEqual(set(assemble(v,['a'],'FREQUENCY').split()),old)
  self.assertGreater(len(assemble(v,['a'],'FREQUENCY').split()),len(old))
  self.assertEqual(set(assemble(v,['a'],'NEGATION').split())-old,{'not'})
  self.assertEqual({x for x in assemble(v,['a'],'NUMERIC').split() if not x.startswith('num_')},old)
  self.assertTrue(any(x.startswith('num_') for x in assemble(v,['a'],'NUMERIC').split()))
  self.assertIn('raising',assemble(v,['a'],'NO_STEM').split())
 def test_weighted_sets(self):
  a=np.array([[-1.],[1.]]);b=np.zeros((2,1));self.assertEqual(float(a.mean()),float(b.mean()));self.assertNotEqual(float((a*a).mean()),float((b*b).mean()))
 def test_same_article_dedup_not_added(self):
  v={'a':article_views('Apple buys','Apple buys')};self.assertEqual(assemble(v,['a','a'],'CONTROL'),assemble(v,['a'],'CONTROL'));self.assertEqual(len(assemble(v,['a','a'],'FREQUENCY').split()),2*len(assemble(v,['a'],'FREQUENCY').split()))
 def test_calibration_groups(self):
  d=pd.DataFrame({'label':[0,1]*30});g=np.repeat(np.arange(30),2);cv=calibration(np.arange(60),d,573,g)
  for a,b in cv:self.assertFalse(set(g[a])&set(g[b]))
 def test_selection_tie(self):
  rows=[dict(C=c,BA=.5,Brier=.25) for c in CS];self.assertEqual(select(rows),.01)
 def test_heldout_chunk_cannot_change_fitted_geometry(self):
  rng=np.random.default_rng(99);v=rng.normal(size=(30,768));data={'vectors':v,'window_chunks':[np.array([i]) for i in range(30)],'weights':[np.array([1.]) for i in range(30)]}
  d=pd.DataFrame({'stem_body':['apple earnings rise']*30,**{c:np.arange(30,dtype=float) for c in OLD}})
  first=SetTransform().fit(d,list(range(20)),data);changed={**data,'vectors':v.copy()};changed['vectors'][20:]+=10000;second=SetTransform().fit(d,list(range(20)),changed)
  np.testing.assert_array_equal(first.center,second.center);np.testing.assert_array_equal(first.omega,second.omega)
  self.assertEqual(first.sigma,second.sigma)
 def test_seal_corruption_rejected(self):
  import tempfile,engine
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);f=t/'source.txt';f.write_text('original');dump(t/'seal.json',{'files':{str(f):sha(f)}});old=engine.WORK;engine.WORK=t
   try:
    engine.seal_check();f.write_text('changed')
    with self.assertRaises(AssertionError):engine.seal_check()
   finally:engine.WORK=old
 def test_no_news_exact_fallback(self):
  raw=np.array([.9,.2,.6]);price=np.array([.371234567890123,.59,.72]);news=np.array([False,True,False]);stored=np.where(news,raw,price)
  np.testing.assert_array_equal(stored[~news],price[~news]);damaged=stored.copy();damaged[0]+=.01
  with self.assertRaises(AssertionError):np.testing.assert_array_equal(damaged[~news],price[~news])
 def test_synthetic_aggregation_save_reload(self):
  import tempfile
  rng=np.random.default_rng(11);data={'vectors':rng.normal(size=(30,768)),'window_chunks':[np.array([i]) for i in range(30)],'weights':[np.array([1.]) for i in range(30)]}
  d=pd.DataFrame({'stem_body':['apple gains change']*30,**{c:rng.normal(size=30) for c in OLD}});tf=SetTransform().fit(d,list(range(20)),data);reps=tf.representations(d,data)
  with tempfile.TemporaryDirectory() as tmp:
   for name,z in reps.items():
    x=sparse.hstack([tf.base(d,np.arange(20)),z[:20]]).tocsr();y=sparse.hstack([tf.base(d,np.arange(20,30)),z[20:]]).tocsr();m=classifier('FULL',.1).fit(x,np.arange(20)%2)
    p=m.predict_proba(y);f=Path(tmp)/(name+'.joblib');joblib.dump({'model':m,'transform':tf},f);saved=joblib.load(f);np.testing.assert_array_equal(saved['model'].predict_proba(y),p)
if __name__=='__main__':unittest.main()
