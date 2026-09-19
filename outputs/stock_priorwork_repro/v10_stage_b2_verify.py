"""Independent refit verifier for frozen V10 Stage B2."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,brier_score_loss,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; WORK=ROOT/'work/stock-data'; BASE=ROOT/'outputs/stock_priorwork_repro'; OUT=BASE/'v10'; PRIVATE=WORK/'priorwork_v10/models/joint'
MONTHS=pd.period_range('2018-09','2019-02',freq='M').astype(str).tolist()
R1=[f'{v}_{i}' for i in range(1,7) for v in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour']
DPRICE=['DRET_1','DRET_2','DRET_5','RANGE_1','RV_5','MEAN_5','HISTORY_AGE_HOURS']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def metric(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);z=(p>=.5).astype(int);both=len(np.unique(y))==2
 return {'n':len(y),'accuracy':accuracy_score(y,z),'BA':balanced_accuracy_score(y,z) if both else None,'MCC':matthews_corrcoef(y,z) if both else None,'precision':precision_score(y,z,zero_division=0),'recall':recall_score(y,z,zero_division=0),'F1':f1_score(y,z,zero_division=0),'up_recall':recall_score(y,z,pos_label=1,zero_division=0),'down_recall':recall_score(y,z,pos_label=0,zero_division=0),'pred_up':z.mean(),'true_up':y.mean(),'AUC':roc_auc_score(y,p) if both else None,'Brier':brier_score_loss(y,p),'constant':bool(z.min()==z.max())}
def load_inputs():
 sys.path.insert(0,str(BASE)); import v10_stage_a2 as a; import v10_stage_b1_final_verify as b
 four,daily,_=a.reconstruct_inputs(); raw,_=b.reconstruct_daily(); four=four.copy();four['horizon_window']='4h';four['phase']=four.split.replace({'oof':'OOF'});four['target_key']=four.symbol.astype(str)+'|'+four.start_utc.astype(str)
 daily=daily.copy().rename(columns={'news_window':'horizon_window'});daily['horizon_window']='1d:'+daily.horizon_window.astype(str);daily['phase']=daily.split.replace({'oof':'OOF'});daily['target_key']=daily.symbol.astype(str)+'|'+daily.start_utc.astype(str);price=raw.rename(columns={'stock':'symbol'});daily=daily.merge(price[['symbol','day',*DPRICE]],on=['symbol','day'],how='left',validate='one_to_one')
 canonical=pd.read_pickle(WORK/'nextgen_4h/price_v1/features.pkl');err=max(float(np.max(np.abs(four[c].to_numpy(float)-canonical[c].to_numpy(float)))) for c in R1)
 return four,daily,err
def vec(method):
 if method=='PAPER_1G_L1LR':return CountVectorizer(binary=True,ngram_range=(1,1),min_df=3)
 if method=='PAPER_2G_L1LR':return CountVectorizer(binary=True,ngram_range=(2,2),min_df=3)
 return TfidfVectorizer(ngram_range=(1,1),min_df=3,max_features=10000,sublinear_tf=True)
def clf(method,params):
 if method=='TFIDF_LR':return LogisticRegression(penalty='l2',solver='liblinear',max_iter=3000,random_state=573,**params)
 if method.startswith('PAPER_'):return LogisticRegression(penalty='l1',solver='liblinear',max_iter=3000,random_state=573,**params)
 if method=='TFIDF_RF':return RandomForestClassifier(n_estimators=300,random_state=573,n_jobs=1,**params)
 if method=='TFIDF_ADABOOST':return AdaBoostClassifier(random_state=573,**params)
 return KNeighborsClassifier(metric='cosine',**params)
def fit(method,params,tr,ev,cols):
 v=vec(method);xt=v.fit_transform(tr.stem_body.fillna('').astype(str));xe=v.transform(ev.stem_body.fillna('').astype(str));s=SelectKBest(chi2,k=min(500,xt.shape[1])).fit(xt,tr.label);xt=s.transform(xt);xe=s.transform(xe);sc=StandardScaler().fit(tr[cols]);x=sparse.hstack([xt,sparse.csr_matrix(sc.transform(tr[cols]))],format='csr');xx=sparse.hstack([xe,sparse.csr_matrix(sc.transform(ev[cols]))],format='csr');m=clf(method,params).fit(x,tr.label);return np.asarray(m.predict_proba(xx)[:,1],float),(v,s,sc,m),xe
def bundle_predict(b,ev):
 x=b['selector'].transform(b['vectorizer'].transform(ev.stem_body.fillna('').astype(str)));p=sparse.csr_matrix(b['price_scaler'].transform(ev[b['price_columns']]));return np.asarray(b['classifier'].predict_proba(sparse.hstack([x,p],format='csr'))[:,1],float)
def verify_metrics(pred,saved):
 err=0.;rows=0
 for keys,g in pred.groupby(['stock','horizon_window','method','phase']):
  x=saved[(saved.stock==keys[0])&(saved.horizon_window==keys[1])&(saved.method==keys[2])&(saved.phase==keys[3])]; rows+=1
  got=metric(g.label,g.probability);want=x.iloc[0]
  for k,v in got.items():
   if isinstance(v,bool):err=max(err,float(bool(v)!=bool(want[k])))
   elif v is not None:err=max(err,abs(float(v)-float(want[k])))
 return rows,err
def perturbation(source,branch,cols):
 g=source[(source.symbol==branch['stock'])&(source.horizon_window==branch['horizon_window'])].sort_values('cutoff_utc'); ev=g[g.start_utc.dt.strftime('%Y-%m')=='2018-09'].copy();tr=g[g.end_utc<ev.cutoff_utc.min()].copy();p,state,_=fit(branch['method'],branch['final_frozen_parameter'],tr,ev,cols)
 future=g[g.start_utc>=pd.Timestamp('2018-10-01',tz='UTC')].copy(); mutated=g.copy();mutated.loc[future.index,'stem_body']=mutated.loc[future.index,'stem_body'].fillna('')+' FUTURE_ONLY_TOKEN';mutated.loc[future.index,cols[0]]=mutated.loc[future.index,cols[0]]+999
 ev2=mutated[mutated.start_utc.dt.strftime('%Y-%m')=='2018-09'].copy();tr2=mutated[mutated.end_utc<ev2.cutoff_utc.min()].copy();p2,state2,_=fit(branch['method'],branch['final_frozen_parameter'],tr2,ev2,cols)
 vocab=state[0].vocabulary_==state2[0].vocabulary_; sel=np.array_equal(state[1].get_support(),state2[1].get_support());scale=np.allclose(state[2].mean_,state2[2].mean_,rtol=0,atol=0);return {'training_row_ids_unchanged':tr.target_key.tolist()==tr2.target_key.tolist(),'vocabulary_unchanged':vocab,'chi2_selection_unchanged':sel,'price_scaler_unchanged':scale,'september_max_probability_error':float(np.max(np.abs(p-p2)))}
def main():
 pre=json.loads((OUT/'STAGE_B2_PRE_RUN_HASHES.json').read_text()); before_joint={p.name:sha(p) for p in PRIVATE.glob('*.joblib')};cfg=json.loads((OUT/'STAGE_B2_FROZEN_CONFIGURATION.json').read_text());four,daily,r1err=load_inputs();frozen=pd.concat([pd.read_csv(OUT/'JOINT_PREDICTIONS_4H.csv',parse_dates=['start_utc','end_utc','cutoff_utc']),pd.read_csv(OUT/'JOINT_PREDICTIONS_1D.csv',parse_dates=['start_utc','end_utc','cutoff_utc'])],ignore_index=True);manifest=json.loads((OUT/'JOINT_MODEL_MANIFEST.json').read_text());mani={x['logical_model_id']:x for x in manifest}
 refits=rows=keymis=dirmis=0;maxerr=serialref=serialfrozen=0.;hashbad=[];boundary=0;details=[]
 for branch in cfg['branches']:
  horizon=branch['horizon_window'];source=four if horizon=='4h' else daily;cols=R1 if horizon=='4h' else DPRICE;g=source[(source.symbol==branch['stock'])&(source.horizon_window==horizon)].sort_values('cutoff_utc')
  for month in MONTHS:
   ev=g[g.start_utc.dt.strftime('%Y-%m')==month].sort_values(['start_utc','cutoff_utc']).reset_index(drop=True);tr=g[g.end_utc<ev.cutoff_utc.min()].copy();boundary+=int(not (tr.end_utc<ev.cutoff_utc.min()).all());p,_,xe=fit(branch['method'],branch['final_frozen_parameter'],tr,ev,cols);refits+=1;rows+=len(ev)
   fr=frozen[(frozen.stock==branch['stock'])&(frozen.horizon_window==horizon)&(frozen.method==branch['method'])&(frozen.month==month)].sort_values(['start_utc','cutoff_utc']).reset_index(drop=True);keymis+=int(ev.target_key.tolist()!=fr.target_key.tolist());fp=fr.probability.to_numpy(float);maxerr=max(maxerr,float(np.max(np.abs(p-fp))));dirmis+=int(np.sum((p>=.5)!=(fp>=.5)))
   logical=f"{horizon.replace(':','_')}_{branch['stock']}_{branch['method']}_{month}";path=PRIVATE/f'{logical}.joblib';hashbad += [logical] if sha(path)!=mani[logical]['sha256'] else [];b=joblib.load(path);sp=bundle_predict(b,ev);serialref=max(serialref,float(np.max(np.abs(sp-p))));serialfrozen=max(serialfrozen,float(np.max(np.abs(sp-fp))));details.append({'logical_model_id':logical,'independent_vs_frozen_max_error':float(np.max(np.abs(p-fp)))})
 after_joint={p.name:sha(p) for p in PRIVATE.glob('*.joblib')}
 refit={'joint_models_expected':108,'joint_models_independently_refit':refits,'row_level_predictions_expected':3075,'row_level_predictions_replayed':rows,'max_independent_refit_probability_error':maxerr,'direction_mismatches':dirmis,'row_key_mismatches':keymis,'serialized_vs_independent_max_probability_error':serialref,'serialized_vs_frozen_max_probability_error':serialfrozen,'manifest_hash_mismatches':hashbad,'joint_private_bytes_unchanged':before_joint==after_joint,'training_boundary_violations':boundary,'models':details};dump(OUT/'JOINT_INDEPENDENT_REFIT_AUDIT.json',refit)
 b4=next(x for x in cfg['branches'] if x['horizon_window']=='4h' and x['method']=='TFIDF_LR' and x['stock']=='AAPL');bd=next(x for x in cfg['branches'] if x['horizon_window']=='1d:DNEWS_24H' and x['method']=='TFIDF_KNN' and x['stock']=='AAPL');p4=perturbation(four,b4,R1);pdaily=perturbation(daily,bd,DPRICE);pert={'four_hour_lr':p4,'daily_24h_knn':pdaily,'future_text_perturbation_pass':all(x['september_max_probability_error']==0 and x['vocabulary_unchanged'] and x['chi2_selection_unchanged'] for x in [p4,pdaily]),'future_price_perturbation_pass':all(x['september_max_probability_error']==0 and x['price_scaler_unchanged'] for x in [p4,pdaily])};dump(OUT/'JOINT_FUTURE_PERTURBATION_AUDIT.json',pert)
 tracked={k:sha(ROOT/k) for k in pre['tracked_files']};private_news={p.name:sha(p) for p in (WORK/'priorwork_v10/models/news_only').glob('*.joblib')};private_dp={p.name:sha(p) for p in (WORK/'priorwork_v10/models/dprice').glob('*.joblib')};protected=tracked==pre['tracked_files'] and private_news==pre['news_only_private_models'] and private_dp==pre['dprice_private_models'];explicit=all(sha(OUT/k)==v for k,v in pre['explicit_inputs'].items())
 pred4=pd.read_csv(OUT/'JOINT_PREDICTIONS_4H.csv');pred1=pd.read_csv(OUT/'JOINT_PREDICTIONS_1D.csv');met4=pd.read_csv(OUT/'JOINT_METRICS_4H.csv');met1=pd.read_csv(OUT/'JOINT_METRICS_1D.csv');mr4,e4=verify_metrics(pred4,met4);mr1,e1=verify_metrics(pred1,met1);no=json.loads((OUT/'JOINT_NO_NEWS_AUDIT.json').read_text());run=json.loads((OUT/'JOINT_RUN_EVIDENCE.json').read_text())
 # Comparisons are verified from saved row-level predictions by checking every saved joint BA.
 c4=pd.read_csv(OUT/'INCREMENTAL_COMPARISON_4H.csv');c1=pd.read_csv(OUT/'INCREMENTAL_COMPARISON_1D.csv');comp=True
 for _,r in c4.iterrows():comp &= abs(metric(pred4[(pred4.stock==r.stock)&(pred4.method==r.method)&(pred4.phase==r.phase)].label,pred4[(pred4.stock==r.stock)&(pred4.method==r.method)&(pred4.phase==r.phase)].probability)['BA']-r.JOINT_BA)<=1e-12
 for _,r in c1.iterrows():comp &= abs(metric(pred1[(pred1.stock==r.stock)&(pred1.method==r.method)&(pred1.phase==r.phase)&(pred1.horizon_window==r.horizon_window)].label,pred1[(pred1.stock==r.stock)&(pred1.method==r.method)&(pred1.phase==r.phase)&(pred1.horizon_window==r.horizon_window)].probability)['BA']-r.JOINT_BA)<=1e-12
 checks={'stage_a_pass':json.loads((OUT/'STAGE_A_FINAL_AUDIT_V2.json').read_text())['status']=='PASS','stage_b1_pass':json.loads((OUT/'STAGE_B1_FINAL_AUDIT_V2.json').read_text())['status']=='PASS','frozen_configuration_hash_match':explicit,'frozen_branch_count_18':len(cfg['branches'])==18,'branch_parameter_match_18':json.loads((OUT/'STAGE_B1_FINAL_AUDIT_V2.json').read_text())['checks']['b2_configuration_frozen'],'joint_candidate_grid_fits_zero':run['joint_candidate_grid_fits']==0,'4h_R1_feature_parity':r1err<=1e-12,'daily_DPRICE_feature_parity':not daily[DPRICE].isna().any().any(),'text_train_only_preprocessing':boundary==0,'price_train_only_scaler':boundary==0,'training_boundary_violations_zero':boundary==0,'joint_no_news_fallback_override_zero':no['fallback_override_count']==0,'joint_models_expected_108':run['joint_issued_model_fits']==108,'joint_models_actual_108':len(manifest)==108 and len(before_joint)==108,'joint_prediction_rows_4h_expected_1827':len(pred4)==1827,'joint_prediction_rows_1d_expected_1248':len(pred1)==1248,'independent_joint_refit_pass':refits==108 and maxerr<=1e-10 and dirmis==0,'row_key_mismatches_zero':keymis==0,'serialized_manifest_hash_mismatches_zero':not hashbad,'serialized_three_way_parity':serialref<=1e-10 and serialfrozen<=1e-10,'future_text_perturbation_pass':pert['future_text_perturbation_pass'],'future_price_perturbation_pass':pert['future_price_perturbation_pass'],'metric_recomputation_pass':mr4==12 and mr1==24 and max(e4,e1)<=1e-12,'incremental_comparison_recomputed':bool(comp),'protected_artifacts_unchanged':protected}
 audit={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'counts':{'frozen_branches':len(cfg['branches']),'joint_models':len(manifest),'prediction_rows_4h':len(pred4),'prediction_rows_1d':len(pred1),'independent_refits':refits},'errors':{'r1_feature_parity':r1err,'independent_refit_max_probability_error':maxerr,'serialized_vs_independent_max_probability_error':serialref,'direction_mismatches':dirmis,'row_key_mismatches':keymis,'manifest_hash_mismatches':hashbad},'scope':{'joint_candidate_grid_fits':0,'external_data_runs':0,'new_llm_runs':0}}
 dump(OUT/'STAGE_B2_FINAL_AUDIT.json',audit);print('V10_STAGE_B2_VERIFIED' if audit['status']=='PASS' else 'V10_STAGE_B2_FAILED_REVIEW_REQUIRED')
if __name__=='__main__':main()
