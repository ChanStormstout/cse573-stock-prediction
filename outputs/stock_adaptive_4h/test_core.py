import unittest,tempfile,json
from core import *
from scipy.optimize import check_grad

class Mechanisms(unittest.TestCase):
    def test_exact_absence(self):
        q=np.array([.123456789,.5,.9]);p=add_delta(q,[1,0,-1],[False,True,False]);np.testing.assert_array_equal(p[[0,2]],q[[0,2]])
    def test_zero_correction(self):
        q=np.array([.2,.6]);np.testing.assert_allclose(add_delta(q,[0,0],[1,1]),q,atol=1e-15)
    def test_series_not_mutated(self):
        q=pd.Series([.2,.6]);original=q.to_numpy().copy()
        add_delta(q,[1,-1],[1,1]);np.testing.assert_array_equal(q.to_numpy(),original)
    def test_static_zero_always_eligible(self):
        d=pd.DataFrame({'label':[0,1,0,1],'month':['2018-04']*4,'price':[.3,.8,.4,.9],'db':[.1]*4,'ds':[-.1]*4,'has_news':[1,1,0,1]})
        old=d.price.to_numpy().copy();w,grid=select_static(d);np.testing.assert_array_equal(d.price.to_numpy(),old)
        self.assertTrue(next(r['eligible'] for r in grid if r['weights']==[1.,0.,0.]))
    def test_offset_gradient(self):
        rng=np.random.default_rng(573);m=OffsetModel(.1);a=rng.normal(size=(15,4));y=rng.integers(0,2,15);o=rng.normal(size=15);w=rng.normal(size=4);pen=np.array([1,1,1,0])
        self.assertLess(check_grad(lambda x:m.objective(x,a,y,o,pen)[0],lambda x:m.objective(x,a,y,o,pen)[1],w),1e-5)
    def test_offset_intercept(self):
        y=np.array([0,0,1,1,1,1]);m=OffsetModel().fit(np.zeros((6,1)),np.ones(6),np.full(6,.5),y)
        dd,raw=m.correction(np.zeros((6,1)),np.ones(6));np.testing.assert_allclose(expit(raw),np.full(6,4/6),atol=1e-5)
    def test_fixed_news_intercept(self):
        m=OffsetModel(news_bias='zero').fit(np.zeros((6,1)),np.ones(6),np.full(6,.5),[0,0,1,1,1,1]);dd,_=m.correction(np.zeros((6,1)),np.ones(6));np.testing.assert_array_equal(dd,np.zeros(6))
    def test_future_refusal(self):
        t=pd.DataFrame({'end_utc':[pd.Timestamp('2018-07-01',tz='UTC')]});e=pd.DataFrame({'cutoff_utc':[pd.Timestamp('2018-07-01',tz='UTC')]})
        with self.assertRaises(ValueError):temporal_check(t,e)
    def test_cache_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cache';p.write_text('first');h={str(p):sha(p)};p.write_text('changed')
            with self.assertRaises(ValueError):check_hashes(h)
    def test_simplex(self):
        w=static_grid();self.assertEqual(len(w),15);self.assertTrue(all((x>=0).all() and x.sum()==1 for x in w))
    def test_router_insufficient(self):
        d=pd.DataFrame({'symbol':['AMZN'],'day':['x'],'has_news':[1],'key':['a']});r=Router().fit(d);w,reason=r.weights(d,[.5,.25,.25],.5)
        np.testing.assert_array_equal(w,[[.5,.25,.25]]);self.assertEqual(reason[0],'STATIC_INSUFFICIENT_HISTORY')
    def test_router_context_and_stock_support(self):
        d=pd.DataFrame([{'symbol':s,'day':str(day),'key':f'{s}-{day}-{h}',
            'has_news':int(s=='AAPL'),'news_count':2,'age_median':30.,'age_unknown':0.,
            'return_std':.01,'price':.55,'db':.1,'ds':-.1,'ny_hour':h+10,
            'label':day%2,'body':.57,'semantic':.53}
            for s in ['AAPL','AMZN'] for day in range(70) for h in range(3)])
        r=Router().fit(d);q=d.iloc[[0,1,210]].copy();q.loc[q.index[-1],'has_news']=1
        q.loc[q.index[1],['news_count','age_median','return_std']]=[1000,10000,2]
        w,why=r.weights(q,[.5,.25,.25],.25)
        self.assertEqual(list(why),['CONDITIONAL','STATIC_CONTEXT_OUTSIDE_RANGE','STATIC_INSUFFICIENT_HISTORY'])
        np.testing.assert_array_equal(w[1:],[[.5,.25,.25],[.5,.25,.25]])
        self.assertTrue(np.max(np.abs(w[0]-[.5,.25,.25]))<=.25)
    def test_calibration_monotone(self):
        p=np.linspace(.1,.9,20);y=np.array([0,1]*10);c=Calibration().fit(p,y);self.assertTrue((np.diff(c.predict(p))>0).all());self.assertTrue(.5<=c.coef_[0]<=2)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileExistsError):Path(td).mkdir(exist_ok=False)
    def test_json_booleans_are_booleans(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'status.json';dump(p,{'passed':np.bool_(False),'n':np.int64(3)});r=json.loads(p.read_text());self.assertIs(r['passed'],False);self.assertEqual(r['n'],3)

if __name__=='__main__':unittest.main()
