"""Independent protocol checks for saved runs, labels, and future-safe features."""
from core import *
def run():
 d=data();checks={};checked=0
 for s in ('AAPL','AMZN'):
  bars=load_bars(s)
  for r in d[d.symbol==s].itertuples():
   g=bars.reindex(pd.date_range(r.start_utc,r.end_utc-pd.Timedelta('5min'),freq='5min'));assert len(g)==48 and not g[['open','close']].isna().any().any();assert int(g.close.iloc[-1]>g.open.iloc[0])==r.label;checked+=1
 checks['raw_four_hour_labels']=checked
 for p in PRIVATE.glob('*/done.json'):
  if p.parent.name=='pre_fingerprint_hardening':continue
  obj=json.loads(p.read_text())
  for path,h in obj['fingerprint'].items():assert sha(ROOT/path)==h,(p,path)
  for fit in obj['fits']:assert pd.Timestamp(fit['train_end'])<pd.Timestamp(fit['eval_cutoff'])
  for choice in obj['choices']:
   boundary='2018-09' if choice['month']=='final' else choice['month'];assert all(m<boundary for m in choice['selection_months'])
  checks[p.parent.name]=dict(fits=len(obj['fits']),past_only=True,fingerprint=True)
 # Reject altered cache fingerprint without modifying real source files.
 from unittest.mock import patch
 item=PRIVATE/'P_own_lr/done.json';a=json.loads(item.read_text())['fingerprint'];b=dict(a);b[next(iter(b))]='modified'
 with patch('core.run_fingerprint',return_value=b):
  try:run_model('P_own_lr',lambda *args: (_ for _ in ()).throw(RuntimeError('must not fit')))
  except AssertionError as e:assert 'resume mismatch' in str(e)
  else:raise AssertionError('altered cache was accepted')
 checks['fingerprint_change_rejected_before_fit']=True
 audit=json.loads((PRIVATE/'cross_audit.json').read_text());assert all(r['latest_used'] is None or pd.Timestamp(r['latest_used'])<=pd.Timestamp(r['cutoff']) for r in audit);checks['cross_time_rows']=len(audit)
 fresh,_=cross_features(d);saved=pd.read_pickle(PRIVATE/'cross.pkl');assert np.allclose(fresh.to_numpy(),saved.to_numpy(),equal_nan=True);checks['cross_features_recomputed']=True
 # Direct feature mutation: no future row may change a completed window.
 bars=load_bars('AAPL');r=d[d.symbol.eq('AAPL')&d.recent_missing_60.eq(0)].iloc[0];before=window(bars,r.cutoff_utc-pd.Timedelta('60min'),r.cutoff_utc);changed=bars.copy();changed.loc[changed.index>=r.cutoff_utc,['open','close','high','low']]*=17;after=window(changed,r.cutoff_utc-pd.Timedelta('60min'),r.cutoff_utc);assert before==after;checks['future_bar_perturbation']=True
 fallback=pd.read_csv(PRIVATE/'P_own_lr/predictions.csv',float_precision='round_trip').p.to_numpy();news=d.has_original_news.to_numpy().astype(bool);evalmask=d.month.ge('2018-03').to_numpy()
 for p in PRIVATE.glob('*/predictions.csv'):
  if p.parent.name.startswith(('F1','F2','T_')):
   pred=pd.read_csv(p,float_precision='round_trip').p.to_numpy();assert np.array_equal(pred[~news&evalmask],fallback[~news&evalmask]);checks[p.parent.name]['exact_fallback']=True
 members=json.loads((PRIVATE/'history_members.json').read_text())
 for method in ('H0','H1','H2'):
  empty=np.array([not m['current'] and (method=='H0' or not m['old']) for m in members]);pred=pd.read_csv(PRIVATE/method/'predictions.csv',float_precision='round_trip').p.to_numpy();assert np.array_equal(pred[empty&evalmask],fallback[empty&evalmask]);checks[method]['exact_fallback']=True
 manifests=[json.loads((PRIVATE/f'dense_{name}.json').read_text()) for name in ('original','A1_573','A1_574','A1_575')];assert len({a['fingerprint']['tokens'] for a in manifests})==1;checks['identical_encoder_tokens']=True
 from text_models import TextTransform
 import __main__
 __main__.TextTransform=TextTransform
 transforms=[]
 for method in ('F2_new','T_mix'):
  choice=next(c for c in json.loads((PRIVATE/method/'done.json').read_text())['choices'] if c['month']=='final');tr,ev=split(d,'AMZN','final');bundle=joblib.load(PRIVATE/method/f"model_{tr[-1]}_{choice['C']}_573.joblib");assert bundle['transform'].pricecols==OLD;transforms.append(bundle['transform'].transform(d,ev)[:,:len(OLD)])
 assert np.array_equal(*transforms);checks['mix_vs_F2_identical_price_input']=True
 dump(OUT/'checks.json',checks);print('checked',len(checks),flush=True)
if __name__=='__main__':run()
