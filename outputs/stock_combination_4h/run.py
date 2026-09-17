"""Past-only temperature combination; frozen-model intervention; offline delivery."""
from pathlib import Path
import sys,json,time,hashlib,copy
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/stock_paper_methods_4h'))
from run import Transform,calibration_fit,metric,sha,dump,scores
from report import intervals
PARENT=ROOT/'outputs/stock_paper_methods_4h/v1';PRIVATE=ROOT/'work/stock-data/paper_methods_4h/v1'
OUT=HERE/'v1';WORK=ROOT/'work/stock-data/combination_4h/v1'

def table(d,methods):
 rows=[];monthly=[]
 for (phase,sym),g in d[d.phase!='warmup'].groupby(['phase','symbol']):
  for method in methods:
   rows.append(dict(phase=phase,symbol=sym,method=method,**metric(g.label,g[method])))
   for month,x in g.groupby('month'):monthly.append(dict(phase=phase,symbol=sym,month=month,method=method,**metric(x.label,x[method])))
 return pd.DataFrame(rows),pd.DataFrame(monthly)

def gate(d):
 g=d[d.phase=='train_forward_oof'];a=scores(g,'B3').set_index(['symbol','month']);b=scores(g,'B1').set_index(['symbol','month']);diff=a[['BA','Brier']]-b[['BA','Brier']];monthly=diff.groupby('month').BA.mean();stock=diff.groupby('symbol').mean()
 passed=len(monthly)>=3 and (monthly>0).sum()>=2 and diff.BA.mean()>=.01 and stock.BA.min()>=-.01 and stock.Brier.max()<=.002 and diff.Brier.mean()<=.002
 return dict(primary='B3 vs B1',months=len(monthly),positive_months=int((monthly>0).sum()),BA_delta=float(diff.BA.mean()),Brier_delta=float(diff.Brier.mean()),stocks=stock.to_dict('index'),passed=bool(passed))

def calibrate(d):
 records=[]
 d['B0']=d.F2;d['B1']=np.nan;d['B2']=d.J3;d['B3']=np.nan
 for base,method in [('F2','B1'),('J3','B3')]:
  for month in list(pd.period_range('2018-03','2018-08',freq='M').astype(str))+['final']:
   boundary='2018-09' if month=='final' else month
   for sym in ('AAPL','AMZN'):
    fitting=d[(d.month>='2018-03')&(d.month<boundary)&(d.symbol==sym)&(d.has_original_news==1)]
    tick=time.monotonic();a,b=calibration_fit(fitting.label,fitting[base],'temperature');assert b==0 and a>0
    artifact=dict(slope=a,intercept=b,training_keys=fitting.key.tolist(),last_month=fitting.month.max() if len(fitting) else None)
    path=WORK/f'{method}_{sym}_{month}.joblib';joblib.dump(artifact,path);loaded=joblib.load(path)
    mask=(d.symbol==sym)&((d.month>='2018-09') if month=='final' else (d.month==month));p=d.loc[mask,base].to_numpy().copy();yes=d.loc[mask,'has_original_news'].to_numpy()==1
    p[yes]=expit(a*logit(np.clip(p[yes],1e-6,1-1e-6)))
    q=d.loc[mask,base].to_numpy().copy();q[yes]=expit(loaded['slope']*logit(np.clip(q[yes],1e-6,1-1e-6)));assert np.array_equal(p,q)
    d.loc[mask,method]=p
    records.append(dict(method=method,symbol=sym,month=month,n=len(fitting),slope=a,temperature=1/a,fit_last_month=artifact['last_month'],seconds=time.monotonic()-tick,sha256=sha(path)))
 usable=d.phase!='warmup';assert np.allclose(d.loc[usable,'B1'],d.loc[usable,'F2_temperature'],atol=1e-12,rtol=0)
 for base,method in [('B0','B1'),('B2','B3')]:
  assert np.array_equal(d.loc[usable,base]>=.5,d.loc[usable,method]>=.5)
  no=usable&(d.has_original_news==0);assert np.array_equal(d.loc[no,method],d.loc[no,'R1'])
 return records

def attribution(d):
 x=pd.read_pickle(PRIVATE/'inputs.pkl');z=np.load(PRIVATE/'aggregates.npz');means=z['article_mean'];events=z['event_mean'];saved=json.loads((PARENT/'selection.json').read_text())['joint']
 rows=[]
 for sym in ('AAPL','AMZN'):
  ix=np.flatnonzero((x.symbol==sym)&(x.month>='2018-09'));subset=x.iloc[ix]
  result=subset[['key','symbol','month','label']].copy();result['phase']=np.where(subset.month<'2018-11','development','later')
  result['has_news']=subset.has_original_news;result['current_vector_changed']=subset.vector_delta>1e-10
  for trained in ['N0M','N1M']:
   c=next(v['C'] for v in saved if v['symbol']==sym and v['month']=='final' and v['method']==trained)
   artifact=joblib.load(PRIVATE/'models'/f'{sym}_final_{trained}_{c}.joblib')
   for aggregation,name in [('N0M','article'),('N1M','event')]:
    tf=copy.deepcopy(artifact['transform']);tf.method=aggregation
    p=artifact['model'].predict_proba(tf.transform(x,ix,means,events))[:,1]
    p[subset.has_original_news.to_numpy()==0]=subset.R1.to_numpy()[subset.has_original_news.to_numpy()==0]
    result[f'{trained}_{name}']=p
  check=d.set_index('key').loc[result.key]
  assert np.max(np.abs(result.N0M_article.to_numpy()-check.N0M.to_numpy()))<1e-12
  assert np.max(np.abs(result.N1M_event.to_numpy()-check.N1M.to_numpy()))<1e-12
  rows.append(result)
 return pd.concat(rows,ignore_index=True)

def diagnosis(d,a):
 rows=[];days=[];cases=[]
 for phase,g in d[(d.symbol=='AMZN')&(d.phase!='warmup')].groupby('phase'):
  positive=(g.label==1).sum();negative=(g.label==0).sum();correct0=(g.N0M>=.5)==g.label;correct1=(g.N1M>=.5)==g.label
  g=g.copy();g['BA_contribution']=(correct1.astype(float)-correct0.astype(float))/np.where(g.label==1,2*positive,2*negative)
  g['Brier_contribution']=((g.N1M-g.label)**2-(g.N0M-g.label)**2)/len(g)
  for name,mask in [('all',np.ones(len(g),bool)),('news',g.has_original_news==1),('no_news',g.has_original_news==0),('vector_changed',g.vector_delta>1e-10),('vector_unchanged',g.vector_delta<=1e-10)]:
   h=g[mask];r=dict(phase=phase,subset=name,n=len(h),BA_contribution=float(h.BA_contribution.sum()),Brier_contribution=float(h.Brier_contribution.sum()))
   for m in ['N0M','N1M']:
    if len(h):r.update({f'{m}_{k}':v for k,v in metric(h.label,h[m]).items()})
   rows.append(r)
  for day,h in g.groupby('day'):days.append(dict(phase=phase,day=day,n=len(h),BA_contribution=float(h.BA_contribution.sum()),Brier_contribution=float(h.Brier_contribution.sum())))
 g=d[(d.symbol=='AMZN')&(d.phase=='later')].copy();old=(g.N0M>=.5)==g.label;new=(g.N1M>=.5)==g.label
 for category,mask in [('changed_right',new&~old),('changed_wrong',old&~new),('unchanged_vector_changed_decision',(g.vector_delta<=1e-10)&((g.N0M>=.5)!=(g.N1M>=.5))),('no_news',g.has_original_news==0)]:
  h=g[mask].copy();h['hash']=h.key.map(lambda k:hashlib.sha256(k.encode()).hexdigest())
  for r in h.sort_values('hash').head(2).itertuples():cases.append(dict(category=category,key=r.key,articles=int(r.articles),clusters=int(r.clusters),vector_delta=float(r.vector_delta),p_article=r.N0M,p_event=r.N1M,label=int(r.label)))
 pd.DataFrame(rows).to_csv(OUT/'amzn_subgroups.csv',index=False);pd.DataFrame(days).to_csv(OUT/'amzn_day_contributions.csv',index=False);dump(OUT/'cases.json',cases)
 return pd.DataFrame(rows),pd.DataFrame(days),cases

def main():
 if WORK.exists():raise FileExistsError('Preserve run; choose a new version')
 WORK.mkdir(parents=True);tick=time.monotonic()
 source=PARENT/'predictions.csv';d=pd.read_csv(source);records=calibrate(d)
 m,monthly=table(d,['F0','R1','F1','B0','B1','B2','B3','N0M','N1M']);m.to_csv(OUT/'metrics.csv',index=False);monthly.to_csv(OUT/'monthly_metrics.csv',index=False)
 a=attribution(d);a.to_csv(OUT/'aggregation_interventions.csv',index=False);am,_=table(a,['N0M_article','N0M_event','N1M_article','N1M_event']);am.to_csv(OUT/'intervention_metrics.csv',index=False)
 diagnosis(d,a);g=gate(d)
 # Reuse the existing paired-block routine with this experiment's fixed contrasts.
 import report as prior_report
 prior_report.PAIRS=[('B3','B1'),('B3','B0'),('B1','B0'),('N1M','N0M')]
 prior_report.intervals(d).to_csv(OUT/'paired_intervals.csv',index=False)
 cols=['key','symbol','day','month','phase','label','has_original_news','has_usable_text','has_qualified_event','articles','clusters','vector_delta','F0','R1','F1','B0','B1','B2','B3','N0M','N1M'];d[cols].to_csv(OUT/'predictions.csv',index=False)
 pd.DataFrame(records).to_csv(OUT/'calibrators.csv',index=False)
 dump(OUT/'execution.json',dict(status='COMPLETE',calibration_records=len(records),optimized_calibrators=sum(r['n']>=30 for r in records),new_classifier_fits=0,frozen_intervention_passes=8,rows=len(d),source_sha256=sha(source),protocol_sha256=sha(HERE/'PRE_REGISTRATION.md'),code_sha256=sha(__file__),seconds=time.monotonic()-tick,gate=g,checks=['F2-temperature replay','strict no-news fallback','positive-temperature direction preservation','saved calibrator reload','intervention endpoints reproduce N0M/N1M'],all_periods_exposed=True))
 print(json.dumps(g,indent=2));print(m[m.phase.isin(['development','later'])][['phase','symbol','method','BA','Brier']].to_string(index=False))
if __name__=='__main__':main()
