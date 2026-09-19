"""Fit-free, independently reconstructed verifier for Market Context v1."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from .market_core import R1_FEATURES, METHODS, C_GRID, SEED, SOLVER, TOL

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key_hash(keys): return hashlib.sha256(json.dumps(sorted(map(str,keys)),separators=(',',':')).encode()).hexdigest()
def probability(frame,cols,model):
 z=(frame[cols].to_numpy(float)-model['mean'])/model['scale'];z=np.where(np.isfinite(z),z,0.)
 return 1/(1+np.exp(-np.clip(z@model['coef'].ravel()+float(model['intercept'][0]),-700,700)))
def metric(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,float);q=p>=.5;up=float(q[y==1].mean()) if (y==1).any() else np.nan;down=float((~q)[y==0].mean()) if (y==0).any() else np.nan
 return {'n':int(len(y)),'BA':float(np.nanmean([up,down])),'MCC':float(matthews_corrcoef(y,q)) if len(np.unique(y))>1 and len(np.unique(q))>1 else 0.,'Brier':float(np.mean((p-y)**2)),'AUC':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,'up_recall':up,'down_recall':down,'pred_up':float(q.mean()),'constant':bool(q.min()==q.max())}
def build_gate(pred):
 cells=[]
 for (s,m),g in pred[pred.month.isin(['2018-06','2018-07','2018-08'])].groupby(['symbol','month']):
  a,b=metric(g.label,g.M0),metric(g.label,g.M1);cells.append({'symbol':s,'month':m,'delta_BA':b['BA']-a['BA'],'delta_Brier':b['Brier']-a['Brier']})
 x=pd.DataFrame(cells);by=x.groupby('symbol').mean(numeric_only=True);macro=x.groupby('month').delta_BA.mean();constant=any(metric(g.label,g.M1)['constant'] for _,g in pred[pred.month.isin(['2018-06','2018-07','2018-08'])].groupby(['symbol','month']))
 out={'outer_cells':int(len(x)),'cells':cells,'AAPL_mean_delta_BA':float(by.loc['AAPL','delta_BA']),'AMZN_mean_delta_BA':float(by.loc['AMZN','delta_BA']),'positive_macro_months':int((macro>0).sum()),'AAPL_mean_delta_Brier':float(by.loc['AAPL','delta_Brier']),'AMZN_mean_delta_Brier':float(by.loc['AMZN','delta_Brier']),'constant_outer_M1':bool(constant)}
 out['passes']=bool(out['outer_cells']==6 and out['AAPL_mean_delta_BA']>=.01 and out['AMZN_mean_delta_BA']>=.01 and out['positive_macro_months']>=2 and out['AAPL_mean_delta_Brier']<=.002 and out['AMZN_mean_delta_Brier']<=.002 and not out['constant_outer_M1']);return out
def build_attribution(pred):
 cells=[]
 for (s,m),g in pred[pred.month.isin(['2018-06','2018-07','2018-08'])].groupby(['symbol','month']):cells.append({'symbol':s,'month':m,'delta_BA':metric(g.label,g.M1)['BA']-metric(g.label,g.Mmeta)['BA'],'delta_Brier':metric(g.label,g.M1)['Brier']-metric(g.label,g.Mmeta)['Brier']})
 return {'comparison':'M1_vs_Mmeta','cells':cells,'macro_delta_BA':float(np.mean([x['delta_BA'] for x in cells]))}

def failcheck(checks,name,ok,detail=''): checks.append({'check':name,'pass':bool(ok),'detail':detail})
def grid_independent(schedule):
 rows=[]
 for r in schedule.itertuples(index=False):
  starts=pd.date_range(r.open,r.close-pd.Timedelta(minutes=5),freq='5min')
  rows.extend({'bar_start_utc':x,'bar_end_utc':x+pd.Timedelta(minutes=5),'session_id':str(r.open.date())} for x in starts)
 return pd.DataFrame(rows)
def bars(path):
 d=pd.read_csv(path);d.bar_start_utc=pd.to_datetime(d.bar_start_utc,utc=True);return d.set_index('bar_start_utc').sort_index()
def independent_feature(row,grid,etf):
 cutoff=pd.Timestamp(row.cutoff_utc); eligible=grid[grid.bar_end_utc+pd.Timedelta(minutes=1)<=cutoff].tail(12)
 if len(eligible)!=12:return {'market_window_missing':1}
 starts=pd.DatetimeIndex(eligible.bar_start_utc); out={'market_age_log1p_hours':float(np.log1p((cutoff-eligible.bar_end_utc.iloc[-1]).total_seconds()/3600)),'market_prior_session_fraction':float((eligible.session_id!=str(row.session_id)).mean()),'market_window_missing':0,'used_starts_utc':','.join(x.isoformat() for x in starts)}
 for sym in ('SPY','QQQ'):
  q=etf[sym].reindex(starts); o=q.open.to_numpy(float); c=q.close.to_numpy(float)
  if not(np.isfinite(o).all() and np.isfinite(c).all() and (o>0).all() and (c>0).all()): out['market_window_missing']=1;break
  r=np.log(c/o);out[f'{sym.lower()}_intrabar_return_60tm']=float(r.sum());out[f'{sym.lower()}_rv_60tm']=float(np.sqrt(np.square(r).sum()))
 if out['market_window_missing']:
  for f in ('spy_intrabar_return_60tm','spy_rv_60tm','qqq_intrabar_return_60tm','qqq_rv_60tm'):out[f]=np.nan
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);checks=[]
 req=['predictions.csv','metrics.csv','monthly_metrics.csv','m0_cv.csv','training_evidence.json','market_coverage.csv','protocol_fingerprint.json','advancement.json','attribution.json','market_features.csv']
 failcheck(checks,'required_artifacts',all((root/x).exists() for x in req)); source=None
 if checks[-1]['pass']:
  fp=json.loads((root/'protocol_fingerprint.json').read_text()); source=Path(fp.get('source_dir',''))
  failcheck(checks,'source_exists',source.exists());
 if source is not None and source.exists():
  d=pd.read_csv(source/'r1_rows.csv');d.cutoff_utc=pd.to_datetime(d.cutoff_utc,utc=True);sched=pd.read_csv(source/'schedule.csv');sched.open=pd.to_datetime(sched.open,utc=True);sched.close=pd.to_datetime(sched.close,utc=True);etf={'SPY':bars(source/'bars_SPY.csv'),'QQQ':bars(source/'bars_QQQ.csv')};expected=json.loads((source/'expected_keys.json').read_text());grid=grid_independent(sched);manifest=json.loads((source/'source_manifest.json').read_text())
  here=Path(__file__).resolve().parent
  r1a=here/'audit_v2/r1_independent_refit.json';r1j=json.loads(r1a.read_text())
  actual_hashes={s:sha(source/f'bars_{s}.csv') for s in ('SPY','QQQ')}
  expected_fp={'canonical_r1_input_sha256':sha(source/'r1_rows.csv'),'market_source_manifest_sha256':sha(source/'source_manifest.json'),'calendar_schedule_sha256':sha(source/'schedule.csv'),'normalized_etf_bar_hashes':actual_hashes,'r1_parity_artifact_sha256':sha(r1a),'market_feature_code_sha256':sha(here/'market_features.py'),'runner_code_sha256':sha(here/'run_market.py'),'market_preregistration_sha256':sha(here/'MARKET_PREREGISTRATION.md')}
  failcheck(checks,'protocol_fingerprint',all(fp.get(k)==v for k,v in expected_fp.items()))
  failcheck(checks,'actual_etf_hashes_and_raw_timestamp_contract',manifest.get('bar_hashes')==actual_hashes and int(manifest.get('duplicate_timestamp_count',-1))==sum(int(etf[s].index.duplicated().sum()) for s in etf) and int(manifest.get('unexpected_timestamp_count',-1))==sum(int((~etf[s].index.isin(pd.DatetimeIndex(grid.bar_start_utc))).sum()) for s in etf))
  failcheck(checks,'accepted_r1_parity_artifact',r1j.get('status')=='PASS' and r1j.get('n')==1374 and r1j.get('max_abs_probability_error',np.inf)<=1e-12 and r1j.get('direction_parity') is True)
  if fp.get('mode')=='real':
   canonical=pd.read_csv(here.parents[1]/'stock_goal60_4h/v1/predictions.csv')[['key','R1']].dropna();realpred=pd.read_csv(root/'predictions.csv');j=realpred.merge(canonical,on='key',validate='one_to_one');err=float(np.max(np.abs(j.M0-j.R1))) if len(j) else np.inf;failcheck(checks,'real_canonical_schedule_parity',len(sched)==len(sched.drop_duplicates(['open','close'])));failcheck(checks,'real_M0_canonical_probability_parity',len(j)==1374 and err<=1e-12 and bool(np.array_equal(j.M0>=.5,j.R1>=.5)),f'max_error={err}')
  else: failcheck(checks,'synthetic_schedule_contract',len(sched)==len(sched.drop_duplicates(['open','close'])))
  pred=pd.read_csv(root/'predictions.csv'); mf=pd.read_csv(root/'market_features.csv'); d=d.merge(mf[['key']+list(METHODS['M1'][len(R1_FEATURES):])],on='key',how='left',validate='one_to_one'); ev=json.loads((root/'training_evidence.json').read_text()); cv=pd.read_csv(root/'m0_cv.csv'); monthly=pd.read_csv(root/'monthly_metrics.csv'); aggregate=pd.read_csv(root/'metrics.csv')
  failcheck(checks,'canonical_feature_contract',fp.get('feature_columns',{}).get('M0')==R1_FEATURES and 'p0' not in fp.get('feature_columns',{}).get('M0',[]))
  evaluated=d[d.month>='2018-03']
  failcheck(checks,'complete_expected_keys',set(pred['key'])==set(expected['keys']) and not pred['key'].duplicated().any() and set(pred['key'])==set(evaluated['key']),f'{len(pred)} rows')
  pred_safe=pred.drop_duplicates('key',keep='first')
  failcheck(checks,'labels_exact',pred_safe.set_index('key').loc[evaluated.key,'label'].astype(int).tolist()==evaluated.label.astype(int).tolist())
  # Rebuild all seven feature fields directly from raw bars and independent calendar grid.
  rebuilt=[]
  for r in d.itertuples(index=False): rebuilt.append({'key':r.key,**independent_feature(r,grid,etf)})
  rebuilt=pd.DataFrame(rebuilt).set_index('key'); saved=mf.set_index('key')
  field=['market_age_log1p_hours','market_prior_session_fraction','market_window_missing','spy_intrabar_return_60tm','spy_rv_60tm','qqq_intrabar_return_60tm','qqq_rv_60tm']
  maxerr=0.; same=True
  for f in field:
   x=rebuilt.loc[saved.index,f].to_numpy(float);y=saved[f].to_numpy(float);same &= bool(np.all((np.isnan(x)&np.isnan(y))|np.isclose(x,y,atol=1e-12,rtol=0)));maxerr=max(maxerr,float(np.nanmax(np.abs(x-y))) if np.isfinite(x-y).any() else 0.)
  failcheck(checks,'raw_bar_time_safe_feature_reconstruction',same,f'max_error={maxerr}')
  cov=rebuilt.assign(SPY_valid=lambda x:(x.market_window_missing==0).astype(float),QQQ_valid=lambda x:(x.market_window_missing==0).astype(float)).join(d.set_index('key')[['symbol','month']]).groupby(['symbol','month'])[['SPY_valid','QQQ_valid']].mean().reset_index();scov=pd.read_csv(root/'market_coverage.csv')
  failcheck(checks,'independent_coverage',cov.merge(scov,on=['symbol','month'],suffixes=('_x','_y')).filter(regex='_').pipe(lambda x:np.allclose(x.iloc[:,0:2],x.iloc[:,2:4],atol=0)) and (cov[['SPY_valid','QQQ_valid']]>=.95).all().all())
  # Validate every model independently, including hash, scaler evidence, keys, C and probabilities.
  long=[]; allok=True; errs={m:0. for m in METHODS}; selections=[]
  for x in ev:
   name=f"{x['method']}_{x['symbol']}_{x['fold']}";npz=root/'models'/f'{name}.npz';meta=json.loads((root/'models'/f'{name}.json').read_text()) if (root/'models'/f'{name}.json').exists() else {}
   ok=npz.exists() and sha(npz)==x.get('npz_sha256') and meta==x and x['columns']==METHODS[x['method']]
   train=d[(d.symbol==x['symbol']) & d.month.isin(x['training_months'])];test=d[(d.symbol==x['symbol']) & (d.month==x['fold'])]
   mean=np.nanmean(train[x['columns']].to_numpy(float),axis=0);mean=np.where(np.isfinite(mean),mean,0);scale=np.nanstd(train[x['columns']].to_numpy(float),axis=0);scale=np.where(np.isfinite(scale)&(scale>1e-12),scale,1)
   q=np.load(npz,allow_pickle=False) if npz.exists() else None
   ok &= q is not None and np.allclose(q['mean'],mean,atol=1e-12) and np.allclose(q['scale'],scale,atol=1e-12) and x['train_key_hash']==key_hash(train.key) and x['eval_key_hash']==key_hash(test.key) and x['train_n']==len(train) and x['eval_n']==len(test)
   if q is not None:
    model={k:q[k] for k in ('mean','scale','coef','intercept')}; pp=probability(test,x['columns'],model); observed=pred_safe.set_index('key').loc[test.key,x['method']].to_numpy(float); errs[x['method']]=max(errs[x['method']],float(np.max(np.abs(pp-observed))));
   allok &= ok; selections.append(x)
  failcheck(checks,'model_hash_columns_and_scaler',allok);failcheck(checks,'all_three_probability_reconstruction',all(v<=1e-12 for v in errs.values()),json.dumps(errs))
  # Reconstruct C chronology from M0 CV, and enforce its inheritance in all methods.
  cok=True
  for (s,fold),g in cv.groupby(['symbol','fold']):
   if fold=='final':continue
   prior=cv[(cv.symbol==s)&(cv.fold<fold)]; chosen=.1 if fold=='2018-03' else sorted([(c,prior[prior.C==c].BA.mean(),prior[prior.C==c].Brier.mean()) for c in C_GRID],key=lambda z:(-z[1],z[2],z[0]))[0][0]
   vals=[x['C'] for x in ev if x['symbol']==s and x['fold']==fold]; cok &= len(vals)==3 and all(float(v)==float(chosen) for v in vals) and set(g.C)==set(C_GRID)
  for (s,fold),g in pd.DataFrame(ev).groupby(['symbol','fold']): cok &= len(set(g.C))==1 and (fold<'2018-09' or all(m<'2018-09' for m in g.training_months.iloc[0]))
  failcheck(checks,'chronological_C_and_no_later_leakage',cok)
  failcheck(checks,'shared_C_inheritance',all(len(set(g.C))==1 for _,g in pd.DataFrame(ev).groupby(['symbol','fold'])))
  failcheck(checks,'no_post_august_training',all(all(m<'2018-09' for m in g.training_months.iloc[0]) for (_,fold),g in pd.DataFrame(ev).groupby(['symbol','fold']) if fold>='2018-09'))
  # Independent saved-metric and gate/attribution reconstruction.
  mrows=[];arows=[]
  for method in METHODS:
   for (s,m),g in pred_safe.groupby(['symbol','month']):mrows.append({'symbol':s,'month':m,'method':method,**metric(g.label,g[method])})
   for (phase,s),g in pred_safe.groupby(['phase','symbol']):arows.append({'phase':phase,'symbol':s,'method':method,**metric(g.label,g[method])})
  def equal_frames(a,b,keys):
   a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
   return list(a[keys].itertuples(index=False,name=None))==list(b[keys].itertuples(index=False,name=None)) and all(np.allclose(a[c].fillna(-999),b[c].fillna(-999),atol=1e-12) for c in a.columns if c not in keys and a[c].dtype.kind in 'fiu')
  failcheck(checks,'independent_metrics',equal_frames(pd.DataFrame(mrows),monthly,['symbol','month','method']) and equal_frames(pd.DataFrame(arows),aggregate,['phase','symbol','method']))
  gate=build_gate(pred_safe); attr=build_attribution(pred_safe);sg=json.loads((root/'advancement.json').read_text());sa=json.loads((root/'attribution.json').read_text())
  failcheck(checks,'primary_gate_exact_six_cells',gate==sg and gate['outer_cells']==6)
  failcheck(checks,'attribution_exact_six_cells',attr==sa and len(attr['cells'])==6)
 status='PASS' if checks and all(x['pass'] for x in checks) else 'FAIL';(root/'verification.json').write_text(json.dumps({'status':status,'fit_free':True,'checks':checks,'check_count':len(checks)},indent=2)+'\n')
 if status!='PASS': raise SystemExit('verification failed')
if __name__=='__main__':main()
