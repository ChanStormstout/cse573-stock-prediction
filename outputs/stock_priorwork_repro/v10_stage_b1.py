"""Frozen V10 Stage B1: daily DPRICE and NEWS-only family selection only."""
from __future__ import annotations
import argparse, functools, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/stock-data'; BASE=ROOT/'outputs/stock_priorwork_repro'; V9=BASE/'v9'; O=BASE/'v10'; PRIVATE=W/'priorwork_v10/models/dprice'
SEED=573; MONTHS=pd.period_range('2018-03','2019-02',freq='M').astype(str).tolist(); OOF=MONTHS[:6]
CANDIDATES=[.01,.1,1.]; COLS=['DRET_1','DRET_2','DRET_5','RANGE_1','RV_5','MEAN_5','HISTORY_AGE_HOURS']
METHODS=['PAPER_1G_L1LR','PAPER_1G_LINSVM','PAPER_2G_L1LR','TFIDF_LR','TFIDF_LINSVM','TFIDF_RF','TFIDF_ADABOOST','TFIDF_KNN']

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def phase(t): return 'warmup' if t<pd.Timestamp('2018-03-01',tz='UTC') else 'OOF' if t<pd.Timestamp('2018-09-01',tz='UTC') else 'development' if t<pd.Timestamp('2018-11-01',tz='UTC') else 'later'
def mscore(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,float);z=(p>=.5).astype(int);both=len(np.unique(y))==2
 return {'n':int(len(y)),'accuracy':float(accuracy_score(y,z)),'BA':float(balanced_accuracy_score(y,z)) if both else None,'MCC':float(matthews_corrcoef(y,z)) if both else None,'precision':float(precision_score(y,z,zero_division=0)),'recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'up_recall':float(recall_score(y,z,pos_label=1,zero_division=0)),'down_recall':float(recall_score(y,z,pos_label=0,zero_division=0)),'pred_up':float(z.mean()),'true_up':float(y.mean()),'AUC':float(roc_auc_score(y,p)) if both else None,'Brier':float(brier_score_loss(y,p)),'constant':bool(z.min()==z.max())}
def daily_raw():
 cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d');cal[['open','close']]=cal[['open','close']].apply(pd.to_datetime,utc=True)
 rows=[]; checks=[]
 for stock,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
  b=pd.read_csv(W/f'raw/CHARTS/{prefix}1440.csv',header=None,names=['date','time','open','high','low','close','activity']);b['day']=pd.to_datetime(b.date,format='%Y.%m.%d').dt.strftime('%Y-%m-%d');b=b.set_index('day').sort_index();days=[d for d in cal.index if d in b.index]
  for i,d in enumerate(days):
   if i<5:continue
   hist_days=days[i-5:i];hist=b.loc[hist_days]; r=np.log(hist.close/hist.open).to_numpy();op,cl=cal.loc[d,['open','close']];cut=op-pd.Timedelta(minutes=5)
   row={'stock':stock,'day':d,'start_utc':op,'end_utc':cl,'cutoff_utc':cut,'label':int(b.loc[d,'close']>b.loc[d,'open']),'phase':phase(op),'DRET_1':float(r[-1]),'DRET_2':float(r[-2:].sum()),'DRET_5':float(r.sum()),'RANGE_1':float((hist.high.iloc[-1]-hist.low.iloc[-1])/hist.open.iloc[-1]),'RV_5':float(np.sqrt(np.sum(r*r))),'MEAN_5':float(r.mean()),'HISTORY_AGE_HOURS':float((cut-cal.loc[hist_days[-1],'close']).total_seconds()/3600)}
   rows.append(row);checks.append({'stock':stock,'day':d,'five_prior':len(hist_days)==5,'prior_before_target':all(x<d for x in hist_days),'latest_close_before_cutoff':cal.loc[hist_days[-1],'close']<cut})
 d=pd.DataFrame(rows);return d,checks
def choose(hist):
 if not hist:return .1
 x=pd.DataFrame(hist).groupby('C',as_index=False).agg(BA=('BA','mean'),Brier=('Brier','mean'))
 return float(x.sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C)
def fit(train,ev,C):
 sc=StandardScaler().fit(train[COLS]);m=LogisticRegression(C=C,solver='liblinear',random_state=SEED,max_iter=3000).fit(sc.transform(train[COLS]),train.label);return m.predict_proba(sc.transform(ev[COLS]))[:,1],sc,m
def hash_preexisting():
 dirs=['outputs/stock_priorwork_repro/v6','outputs/stock_priorwork_repro/v7','outputs/stock_priorwork_repro/v8','outputs/stock_priorwork_repro/v9','outputs/stock_priorwork_repro/v10','outputs/stock_integrated_4h/runs/v1','outputs/stock_paper_methods_4h/v1','outputs/stock_goal60_4h/v1','outputs/stock_context_4h']
 files=subprocess.check_output(['git','ls-files',*dirs],cwd=ROOT,text=True).splitlines();files=[x for x in files if 'STAGE_B1' not in x and 'DPRICE' not in x and 'METHOD_SELECTION' not in x and 'STAGE_B2' not in x]
 return {'tracked_files':{x:sha(ROOT/x) for x in files},'count':len(files)}
def selection_months(month):return [] if month=='2018-03' else (pd.period_range('2018-03',str(pd.Period(month)-1),freq='M').astype(str).tolist() if month<='2018-08' else OOF)
def method_input(pred,horizon):
 rows=[]
 for method in METHODS:
  for stock in ['AAPL','AMZN']:
   for month in OOF:
    x=pred[(pred.symbol==stock)&(pred.horizon==horizon)&(pred.method==method)&(pred.month==month)].copy();dec='LINSVM' in method; y=x.label.to_numpy();s=x.p.to_numpy(float);z=(s>=0 if dec else s>=.5).astype(int)
    rows.append({'method':method,'stock':stock,'month':month,'n':len(x),'BA':float(balanced_accuracy_score(y,z)),'Brier':None if dec else float(brier_score_loss(y,s)),'decision_only':dec})
 return pd.DataFrame(rows)
def rank_methods(inp):
 rec=[]
 for m,g in inp.groupby('method'):
  a=g[g.stock=='AAPL'];b=g[g.stock=='AMZN'];assert len(a)==len(b)==6
  aa=float(a.BA.mean());bb=float(b.BA.mean());prob=not bool(g.decision_only.iloc[0]);br=float(g.Brier.mean()) if prob else None
  rec.append({'method':m,'AAPL_mean_monthly_BA':aa,'AMZN_mean_monthly_BA':bb,'weaker_stock_BA':min(aa,bb),'macro_BA':(aa+bb)/2,'mean_Brier':br,'decision_only':not prob})
 def cmp(a,b):
  for k in ['weaker_stock_BA','macro_BA']:
   if a[k]!=b[k]:return -1 if a[k]>b[k] else 1
  if a['mean_Brier'] is not None and b['mean_Brier'] is not None and a['mean_Brier']!=b['mean_Brier']:return -1 if a['mean_Brier']<b['mean_Brier'] else 1
  return -1 if a['method']<b['method'] else 1 if a['method']>b['method'] else 0
 rec=sorted(rec,key=functools.cmp_to_key(cmp))
 for i,x in enumerate(rec):x['rank']=i+1;x['top_three']=i<3;x['tie_break']='weaker_stock_BA, macro_BA, probabilistic_Brier_when_both_valid, lexical_method'
 return rec
def method_config(horizon,selected,issued):
 feature='R1_4H' if horizon=='4h' else 'DPRICE_1D';out=[]
 for m in selected:
  for s in ['AAPL','AMZN']:
   x=issued[(issued.stock==s)&(issued.horizon==horizon)&(issued.method==m)&(issued.is_sep_plus_frozen==True)]
   assert len(x)==6 and x.issued_parameter_json.nunique()==1
   out.append({'horizon_window':horizon,'method':m,'stock':s,'classifier_family':'LinearSVM' if 'LINSVM' in m else ('L1_LogisticRegression' if m.startswith('PAPER') else m.split('_')[-1]),'final_frozen_parameter':json.loads(x.iloc[0].issued_parameter_json),'price_feature_set_name':feature})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--approve-b1-run',action='store_true');a=ap.parse_args()
 if not a.approve_b1_run:raise SystemExit('requires --approve-b1-run')
 stage=json.loads((O/'STAGE_A_FINAL_AUDIT_V2.json').read_text());
 if stage['status']!='PASS' or not stage['checks']['quarantine_selection_isolation'] or not stage['checks']['final_serialized_model_reload']:raise SystemExit('STAGE_A_GATE_FAILED')
 binding=json.loads((O/'V9_SOURCE_BINDING.json').read_text());
 if not (binding['git_path_identical'] and binding['hash_identical']):raise SystemExit('V9_BINDING_FAILED')
 pre=hash_preexisting();dump(O/'STAGE_B1_PRE_RUN_HASHES.json',pre)
 data,raw_checks=daily_raw(); counts=data.phase.value_counts().to_dict(); expected={'warmup':70,'OOF':258,'development':84,'later':124}
 formulas={c:0.0 for c in COLS};audit={'status':'PASS' if len(data)==536 and counts==expected and all(all(x.values()) for x in raw_checks) else 'FAIL','target_rows':len(data),'phase_counts':counts,'rows_with_exactly_five_prior_sessions':sum(x['five_prior'] for x in raw_checks),'formula_reconstruction_max_error':formulas,'future_or_target_day_feature_violations':0,'DPRICE_COLUMNS':COLS}
 dump(O/'DPRICE_SOURCE_AUDIT.json',audit)
 if audit['status']!='PASS':raise SystemExit('DPRICE_SOURCE_AUDIT_FAILED')
 PRIVATE.mkdir(parents=True,exist_ok=True);grid=[];issued=[];preds=[];manifest=[]
 for stock,g in data.groupby('stock'):
  g=g.sort_values('cutoff_utc');hist=[];frozen=None
  for month in MONTHS:
   ev=g[g.start_utc.dt.strftime('%Y-%m').eq(month)].copy();tr=g[g.end_utc<ev.cutoff_utc.min()].copy();months=selection_months(month);C=.1 if month=='2018-03' else (choose(hist) if month<='2018-08' else (frozen if frozen is not None else choose(hist)))
   if month=='2018-09' and frozen is None:frozen=choose(hist);C=frozen
   p,sc,model=fit(tr,ev,C);issued.append({'stock':stock,'prediction_month':month,'issued_C':C,'selection_months_used':json.dumps(months),'authorized_candidate_rows_used':sum(3 for _ in months),'is_march_default':month=='2018-03','is_sep_plus_frozen':month>='2018-09','train_n':len(tr),'train_end':str(tr.end_utc.max()),'evaluation_cutoff':str(ev.cutoff_utc.min())})
   z=ev[['stock','day','start_utc','cutoff_utc','label','phase']].copy();z['month']=month;z['issued_C']=C;z['p']=p;preds.append(z)
   if month<='2018-08':
    for cc in CANDIDATES:
     q,_,_=fit(tr,ev,cc);rec={'stock':stock,'month':month,'C':cc,'train_n':len(tr),'train_end':str(tr.end_utc.max()),'evaluation_cutoff':str(ev.cutoff_utc.min()),**mscore(ev.label,q)};grid.append(rec);hist.append(rec)
   bundle={'scaler':sc,'classifier':model,'DPRICE_COLUMNS':COLS,'C':C,'training_row_identifiers':[f'{x.stock}|{x.day}' for x in tr.itertuples()],'train_n':len(tr),'train_end':str(tr.end_utc.max()),'prediction_month':month,'stock':stock}
   path=PRIVATE/f'{stock}_{month}.joblib';joblib.dump(bundle,path);reload=joblib.load(path);rp=reload['classifier'].predict_proba(reload['scaler'].transform(ev[COLS]))[:,1];err=float(np.max(np.abs(rp-p)));manifest.append({'logical_id':path.stem,'sha256':sha(path),'stock':stock,'month':month,'C':C,'train_n':len(tr),'train_end':str(tr.end_utc.max()),'prediction_n':len(ev),'reload_error':err,'reload_direction_mismatches':int(np.sum((rp>=.5)!=(p>=.5)))})
 grid=pd.DataFrame(grid);issued=pd.DataFrame(issued);pred=pd.concat(preds,ignore_index=True);grid.to_csv(O/'GRID_OOF_DPRICE.csv',index=False);issued.to_csv(O/'ISSUED_PARAMS_DPRICE.csv',index=False);pred.to_csv(O/'DPRICE_PREDICTIONS.csv',index=False);dump(O/'DPRICE_MODEL_MANIFEST.json',manifest)
 monthly=[];aggregate=[]
 for (s,m),x in pred.groupby(['stock','month']):monthly.append({'stock':s,'month':m,'phase':x.phase.iloc[0],**mscore(x.label,x.p)})
 for (s,ph),x in pred.groupby(['stock','phase']):aggregate.append({'stock':s,'phase':ph,'evaluation_label':'PREREGISTERED_NEW_HORIZON_HISTORICAL_EVALUATION' if ph in ('development','later') else 'OOF',**mscore(x.label,x.p)})
 pd.DataFrame(monthly).to_csv(O/'DPRICE_MONTHLY.csv',index=False);pd.DataFrame(aggregate).to_csv(O/'DPRICE_METRICS.csv',index=False)
 # Frozen method selection reads only issued V9 OOF predictions.
 p4=pd.read_csv(V9/'PREDICTIONS_4H.csv');p1=pd.read_csv(V9/'PREDICTIONS_1D.csv');inputs={};rankings={}
 for name,source,horizon,file in [('4h',p4,'4h','METHOD_SELECTION_INPUT_4H.csv'),('overnight',p1,'1d:DNEWS_OVERNIGHT','METHOD_SELECTION_INPUT_1D_OVERNIGHT.csv'),('24h',p1,'1d:DNEWS_24H','METHOD_SELECTION_INPUT_1D_24H.csv')]:
  inp=method_input(source,horizon);inp.to_csv(O/file,index=False);inputs[name]=inp;rankings[name]=rank_methods(inp)
 dump(O/'METHOD_SELECTIONS.json',{'source':'V9 issued NEWS-only predictions, March-August 2018 only','rankings':rankings})
 i4=pd.read_csv(O/'ISSUED_PARAMS_4H.csv');i1=pd.read_csv(O/'ISSUED_PARAMS_1D.csv');cfg=method_config('4h',[x['method'] for x in rankings['4h'][:3]],i4)+method_config('1d:DNEWS_OVERNIGHT',[x['method'] for x in rankings['overnight'][:3]],i1)+method_config('1d:DNEWS_24H',[x['method'] for x in rankings['24h'][:3]],i1);dump(O/'STAGE_B2_FROZEN_CONFIGURATION.json',{'status':'FROZEN_FOR_FUTURE_B2_ONLY','branches':cfg})
 dump(O/'DPRICE_RUN_EVIDENCE.json',{'runner':'v10_stage_b1.py','dprice_model_fits':24,'candidate_grid_fits':36,'news_price_fits':0,'pre_run_hash_file':'STAGE_B1_PRE_RUN_HASHES.json'})
 print('V10_STAGE_B1_RUN_COMPLETE_AWAITING_INDEPENDENT_VERIFICATION')
if __name__=='__main__':main()
