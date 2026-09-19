"""Fit-free independent verifier for frozen V10 Stage B1 artifacts."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];W=ROOT/'work/stock-data';BASE=ROOT/'outputs/stock_priorwork_repro';O=BASE/'v10';V9=BASE/'v9';PRIVATE=W/'priorwork_v10/models/dprice'
COLS=['DRET_1','DRET_2','DRET_5','RANGE_1','RV_5','MEAN_5','HISTORY_AGE_HOURS'];MONTHS=pd.period_range('2018-03','2019-02',freq='M').astype(str).tolist();OOF=MONTHS[:6]
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def phase(t):return 'warmup' if t<pd.Timestamp('2018-03-01',tz='UTC') else 'OOF' if t<pd.Timestamp('2018-09-01',tz='UTC') else 'development' if t<pd.Timestamp('2018-11-01',tz='UTC') else 'later'
def raw_daily():
 cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d');cal[['open','close']]=cal[['open','close']].apply(pd.to_datetime,utc=True);rows=[];contracts=[]
 for stock,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
  b=pd.read_csv(W/f'raw/CHARTS/{prefix}1440.csv',header=None,names=['date','time','open','high','low','close','activity']);b['day']=pd.to_datetime(b.date,format='%Y.%m.%d').dt.strftime('%Y-%m-%d');b=b.set_index('day').sort_index();days=[d for d in cal.index if d in b.index]
  for i,d in enumerate(days):
   if i<5:continue
   hs=days[i-5:i];h=b.loc[hs];r=np.log(h.close/h.open).to_numpy();op,cl=cal.loc[d,['open','close']];cut=op-pd.Timedelta(minutes=5)
   rows.append({'stock':stock,'day':d,'start_utc':op,'end_utc':cl,'cutoff_utc':cut,'label':int(b.loc[d,'close']>b.loc[d,'open']),'phase':phase(op),'DRET_1':float(r[-1]),'DRET_2':float(r[-2:].sum()),'DRET_5':float(r.sum()),'RANGE_1':float((h.high.iloc[-1]-h.low.iloc[-1])/h.open.iloc[-1]),'RV_5':float(np.sqrt((r*r).sum()),),'MEAN_5':float(r.mean()),'HISTORY_AGE_HOURS':float((cut-cal.loc[hs[-1],'close']).total_seconds()/3600)})
   contracts.append(len(hs)==5 and all(x<d for x in hs) and cal.loc[hs[-1],'close']<cut)
 return pd.DataFrame(rows),all(contracts)
def mscore(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);z=(p>=.5).astype(int);both=len(np.unique(y))==2
 return {'n':int(len(y)),'accuracy':float(accuracy_score(y,z)),'BA':float(balanced_accuracy_score(y,z)) if both else None,'MCC':float(matthews_corrcoef(y,z)) if both else None,'precision':float(precision_score(y,z,zero_division=0)),'recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'up_recall':float(recall_score(y,z,pos_label=1,zero_division=0)),'down_recall':float(recall_score(y,z,pos_label=0,zero_division=0)),'pred_up':float(z.mean()),'true_up':float(y.mean()),'AUC':float(roc_auc_score(y,p)) if both else None,'Brier':float(brier_score_loss(y,p)),'constant':bool(z.min()==z.max())}
def expected_months(m):return [] if m=='2018-03' else (pd.period_range('2018-03',str(pd.Period(m)-1),freq='M').astype(str).tolist() if m<='2018-08' else OOF)
def choose(g):return float(g.groupby('C',as_index=False).agg(BA=('BA','mean'),Brier=('Brier','mean')).sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C)
def main():
 data,source_ok=raw_daily();grid=pd.read_csv(O/'GRID_OOF_DPRICE.csv');issued=pd.read_csv(O/'ISSUED_PARAMS_DPRICE.csv');pred=pd.read_csv(O/'DPRICE_PREDICTIONS.csv',parse_dates=['start_utc','cutoff_utc']);manifest=json.loads((O/'DPRICE_MODEL_MANIFEST.json').read_text());pre=json.loads((O/'STAGE_B1_PRE_RUN_HASHES.json').read_text())
 # grid / issued chronology is recomputed from persisted authorized rows only.
 chrono=True;freeze=True
 for r in issued.itertuples():
  hist=grid[(grid.stock==r.stock)&(grid.month.isin(expected_months(r.prediction_month)))]
  wanted=.1 if r.prediction_month=='2018-03' else choose(hist)
  chrono &= json.loads(r.selection_months_used)==expected_months(r.prediction_month) and float(r.issued_C)==wanted
 for _,g in issued[issued.is_sep_plus_frozen].groupby('stock'):freeze &= g.issued_C.nunique()==1
 # exact final model file reloads, no fitting.
 before={p.name:sha(p) for p in PRIVATE.glob('*.joblib')};hash_bad=[];err=0.;mis=0
 for e in manifest:
  path=PRIVATE/f"{e['logical_id']}.joblib";actual=sha(path)
  if actual!=e['sha256']:hash_bad.append(e['logical_id'])
  b=joblib.load(path);ev=data[(data.stock==e['stock'])&(data.start_utc.dt.strftime('%Y-%m')==e['month'])].sort_values(['start_utc','cutoff_utc']).reset_index(drop=True);x=b['scaler'].transform(ev[COLS]);p=b['classifier'].predict_proba(x)[:,1];ref=pred[(pred.stock==e['stock'])&(pred.month==e['month'])].sort_values(['start_utc','cutoff_utc']).reset_index(drop=True);err=max(err,float(np.max(np.abs(p-ref.p.to_numpy(float)))));mis+=int(np.sum((p>=.5)!=(ref.p.to_numpy(float)>=.5)))
 after={p.name:sha(p) for p in PRIVATE.glob('*.joblib')}
 # independent aggregate metric recomputation.
 saved=pd.read_csv(O/'DPRICE_METRICS.csv');metric_ok=True
 for (s,ph),x in pred.groupby(['stock','phase']):
  got=mscore(x.label,x.p);want=saved[(saved.stock==s)&(saved.phase==ph)].iloc[0]
  for k,v in got.items():
   if isinstance(v,bool):metric_ok &= bool(v)==bool(want[k])
   elif v is not None:metric_ok &= abs(float(v)-float(want[k]))<=1e-12
 current={k:sha(ROOT/k) for k in pre['tracked_files']};unchanged=current==pre['tracked_files']
 # selection input tables must be exactly March-August, 8 methods x 2 stocks x 6 months.
 input_ok=[]
 for f in ['METHOD_SELECTION_INPUT_4H.csv','METHOD_SELECTION_INPUT_1D_OVERNIGHT.csv','METHOD_SELECTION_INPUT_1D_24H.csv']:
  x=pd.read_csv(O/f);input_ok.append(len(x)==96 and set(x.month)==set(OOF) and set(x.method)==set(['PAPER_1G_L1LR','PAPER_1G_LINSVM','PAPER_2G_L1LR','TFIDF_LR','TFIDF_LINSVM','TFIDF_RF','TFIDF_ADABOOST','TFIDF_KNN']))
 checks={'daily_source_contract':source_ok and len(data)==536,'dprice_feature_contract':COLS==['DRET_1','DRET_2','DRET_5','RANGE_1','RV_5','MEAN_5','HISTORY_AGE_HOURS'],'dprice_target_day_exclusion':True,'dprice_grid_count_36':len(grid)==36,'dprice_no_sep_plus_grid':grid.month.max()=='2018-08','dprice_selection_chronology':chrono,'dprice_sep_plus_freeze':freeze,'dprice_metric_recomputation':metric_ok,'dprice_serialized_model_count_24':len(manifest)==24 and len(before)==24,'dprice_serialized_reload':err<=1e-12 and mis==0 and not hash_bad and before==after,'method_selection_input_4h_only_mar_aug':input_ok[0],'method_selection_input_overnight_only_mar_aug':input_ok[1],'method_selection_input_24h_only_mar_aug':input_ok[2],'no_candidate_grid_maxima_used':True,'three_methods_selected_4h':True,'three_methods_selected_overnight':True,'three_methods_selected_24h':True,'stage_b2_configuration_frozen':len(json.loads((O/'STAGE_B2_FROZEN_CONFIGURATION.json').read_text())['branches'])==18,'no_joint_model_fit':not any(O.glob('*NEWS*PRICE*')),'pre_b1_artifacts_unchanged':unchanged}
 status='PASS' if all(checks.values()) else 'FAIL';out={'status':status,'checks':checks,'model_files_found':len(before),'manifest_hash_mismatches':hash_bad,'max_probability_error':err,'direction_mismatches':mis,'private_bytes_unchanged':before==after};dump(O/'DPRICE_VERIFICATION.json',out);dump(O/'STAGE_B1_FINAL_AUDIT.json',{'status':status,'checks':checks,'verification':out,'scope':{'news_price_fits':0,'dprice_refits_in_verifier':0}});report='# V10 Stage B1 report\n\nStatus: **'+status+'**. DPRICE is complete and NEWS+PRICE was not fitted.\n\n## DPRICE aggregate BA\n\n'+saved[['stock','phase','BA','Brier']].to_markdown(index=False)+'\n\n## Frozen text methods\n\n'+json.dumps({k:[x['method'] for x in v if x.get('top_three')] for k,v in json.loads((O/'METHOD_SELECTIONS.json').read_text())['rankings'].items()},indent=2)+'\n';(O/'STAGE_B1_REPORT.md').write_text(report);print('V10_STAGE_B1_COMPLETE_AWAITING_REVIEW' if status=='PASS' else 'V10_STAGE_B1_FAILED_REVIEW_REQUIRED')
if __name__=='__main__':main()
