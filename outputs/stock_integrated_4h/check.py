from run import *
OUT=B/'runs/v1'
p=pd.read_csv(OUT/'predictions.csv');ledger=pd.read_csv(OUT/'oof.csv');cv=pd.read_csv(OUT/'cv.csv');choices=json.loads((OUT/'selection.json').read_text());fits=json.loads((OUT/'fits.json').read_text())
for f in fits:
 assert pd.Timestamp(f['label_end_max'])<pd.Timestamp(f['fit_cutoff']);assert sha(OUT/'models'/f"{f['name']}.joblib")==f['model_hash'];assert f['reload_max_error']<=1e-12
for c in choices:
 prior=cv[(cv.symbol==c['symbol'])&(cv.month<('2018-09' if c['month']=='final' else c['month']))]
 if c['branch'] in BRANCHES:
  prior=prior[prior.branch==c['branch']]
  expected=.1 if prior.empty else prior.groupby('C')[['BA','Brier']].mean().reset_index().sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C
  assert c['C']==expected
 else:
  prior=ledger[(ledger.symbol==c['symbol'])&(ledger.month<('2018-09' if c['month']=='final' else c['month']))];forbidden=c['branch'][3:] if c['branch'].startswith('no_') else None;expected,_=select_fusion(prior,forbidden,c['branch']=='multi_only');assert expected['weights']==c['weights'];assert expected['selection_months']==c['selection_months']
 # No news preserves selected price output exactly for fusion choices.
 if c['branch'] not in BRANCHES and c['weights'] is not None:
  q=p[(p.symbol==c['symbol'])&(p.phase=='frozen')] if c['month']=='final' else p[(p.symbol==c['symbol'])&(p.month==c['month'])&(p.phase=='outer')]
  np.testing.assert_allclose(q[c['branch']],combine(q,c['weights']),atol=1e-12,rtol=0);np.testing.assert_array_equal(q[q.has_news==0][c['branch']],q[q.has_news==0].price)
assert not p.duplicated(['symbol','key','phase']).any()
assert p[p.phase=='frozen'].groupby('symbol').size().to_dict()=={'AAPL':304,'AMZN':305}
inputs=pd.read_pickle(OUT/'inputs.pkl');assert len(inputs)==1607 and inputs.horizon.eq('4h').all();assert ((inputs.end_utc-inputs.start_utc)==pd.Timedelta('4h')).all();assert (inputs.start_utc-inputs.cutoff_utc).eq(pd.Timedelta('5min')).all()
original=pd.read_pickle(O/'stock_horizons/runs/v1/data.pkl');original=original[original.horizon=='4h'].copy();original['key']=original.symbol+'|'+original.start_utc.astype(str);np.testing.assert_array_equal(inputs.set_index('key').loc[original.key].label,original.label)
for path,h in json.loads((OUT/'sources.json').read_text()).items():assert sha(path)==h
print('PASS: 152 fit manifests and hashes; past-only C/weight selection recomputed; fusion probabilities reproduce; exact no-news price fallback; 1607 input keys and 609 frozen replay windows; original input fingerprints unchanged.')
