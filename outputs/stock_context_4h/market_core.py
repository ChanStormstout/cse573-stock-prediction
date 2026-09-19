"""Shared, frozen Market Context v1 production core.

Synthetic fixtures and the future authenticated Alpaca branch deliberately use
the same data preparation, market-bar feature construction, chronological
selection, fitting, serialisation and metric functions.  This module does not
perform network access.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from .market_features import MARKET_CONTROLS, MARKET_NUMERIC, build_market_row, expected_regular_grid

OLD=[f'{x}_{i}' for i in range(1,7) for x in ('return','range')]+['history_age_hours','return_mean','return_std','ny_hour']
REC=[f'{x}_{m}' for m in (5,15,30,60) for x in ('recent_return','recent_range','recent_rv','recent_missing')]
R1_FEATURES=OLD+REC+['overnight_gap','overnight_gap_missing','minutes_from_open','minutes_to_close']
C_GRID=(.01,.1,1.)
META=MARKET_CONTROLS
METHODS={'M0':R1_FEATURES,'Mmeta':R1_FEATURES+META,'M1':R1_FEATURES+META+MARKET_NUMERIC}
SEED=573; SOLVER='liblinear'; TOL=1e-8

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_sha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def key_hash(keys): return json_sha(sorted(map(str,keys)))
def sigmoid(v): return 1/(1+np.exp(-np.clip(v,-700,700)))

def standard_fit(frame, cols):
    a=frame[cols].to_numpy(float); mean=np.nanmean(a,axis=0); mean=np.where(np.isfinite(mean),mean,0.)
    scale=np.nanstd(a,axis=0); scale=np.where(np.isfinite(scale)&(scale>1e-12),scale,1.)
    return mean,scale
def standard_apply(frame, cols, mean, scale):
    z=(frame[cols].to_numpy(float)-mean)/scale
    return np.where(np.isfinite(z),z,0.)
def probability(frame, cols, model):
    return sigmoid(standard_apply(frame,cols,model['mean'],model['scale']) @ model['coef'].ravel()+float(model['intercept'][0]))
def metric(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,float); q=p>=.5
    up=float(q[y==1].mean()) if (y==1).any() else np.nan; down=float((~q)[y==0].mean()) if (y==0).any() else np.nan
    return {'n':int(len(y)),'BA':float(np.nanmean([up,down])),'MCC':float(matthews_corrcoef(y,q)) if len(np.unique(y))>1 and len(np.unique(q))>1 else 0.,'Brier':float(np.mean((p-y)**2)),'AUC':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,'up_recall':up,'down_recall':down,'pred_up':float(q.mean()),'constant':bool(q.min()==q.max())}

def normalize_bars(path):
    d=pd.read_csv(path); d['bar_start_utc']=pd.to_datetime(d['bar_start_utc'],utc=True)
    return d.set_index('bar_start_utc').sort_index()[['open','close']]
def load_input(root):
    root=Path(root); d=pd.read_csv(root/'r1_rows.csv'); d['cutoff_utc']=pd.to_datetime(d['cutoff_utc'],utc=True)
    schedule=pd.read_csv(root/'schedule.csv'); schedule['open']=pd.to_datetime(schedule['open'],utc=True); schedule['close']=pd.to_datetime(schedule['close'],utc=True)
    manifest=json.loads((root/'source_manifest.json').read_text()); expected=json.loads((root/'expected_keys.json').read_text())
    return d,schedule,{'SPY':normalize_bars(root/'bars_SPY.csv'),'QQQ':normalize_bars(root/'bars_QQQ.csv')},manifest,expected,root

def add_market_features(d,schedule,bars):
    grid=expected_regular_grid(schedule); rows=[]
    for r in d.itertuples(index=False):
        z=build_market_row(r.cutoff_utc,str(r.session_id),grid,bars)
        rows.append(z)
    f=pd.DataFrame(rows,index=d.index)
    out=pd.concat([d,f],axis=1); out['SPY_valid']=(out.market_window_missing==0).astype(float); out['QQQ_valid']=out.SPY_valid
    return out,grid

def select_c(records, month):
    if month=='2018-03': return .1
    prior=[x for x in records if x['fold']<month]
    if not prior: return .1
    ranks=[]
    for c in C_GRID:
        q=[x for x in prior if float(x['C'])==c]
        ranks.append((c,float(np.mean([x['BA'] for x in q])),float(np.mean([x['Brier'] for x in q]))))
    return sorted(ranks,key=lambda x:(-x[1],x[2],x[0]))[0][0]
def final_c(records):
    ranks=[]
    for c in C_GRID:
        q=[x for x in records if float(x['C'])==c]
        ranks.append((c,float(np.mean([x['BA'] for x in q])),float(np.mean([x['Brier'] for x in q]))))
    return sorted(ranks,key=lambda x:(-x[1],x[2],x[0]))[0][0]
def fit_one(train, ev, cols, c):
    mean,scale=standard_fit(train,cols); z=standard_apply(train,cols,mean,scale)
    lr=LogisticRegression(C=c,solver=SOLVER,random_state=SEED,max_iter=3000,tol=TOL).fit(z,train.label.astype(int))
    return {'mean':mean,'scale':scale,'coef':lr.coef_,'intercept':lr.intercept_,'classes':lr.classes_}, probability(ev,cols,{'mean':mean,'scale':scale,'coef':lr.coef_,'intercept':lr.intercept_})
def save_model(models,name,model,meta):
    p=models/(name+'.npz'); np.savez(p,mean=model['mean'],scale=model['scale'],coef=model['coef'],intercept=model['intercept'],classes=model['classes'])
    meta['npz_sha256']=sha(p); (models/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')

def build_gate(pred):
    cells=[]
    for (s,m),g in pred[pred.month.isin(['2018-06','2018-07','2018-08'])].groupby(['symbol','month']):
        a=metric(g.label,g.M0); b=metric(g.label,g.M1); cells.append({'symbol':s,'month':m,'delta_BA':b['BA']-a['BA'],'delta_Brier':b['Brier']-a['Brier']})
    x=pd.DataFrame(cells); by=x.groupby('symbol').mean(numeric_only=True) if len(x) else pd.DataFrame()
    macro=x.groupby('month').delta_BA.mean() if len(x) else pd.Series(dtype=float)
    constant=any(metric(g.label,g.M1)['constant'] for _,g in pred[pred.month.isin(['2018-06','2018-07','2018-08'])].groupby(['symbol','month']))
    out={'outer_cells':int(len(x)),'cells':cells,'AAPL_mean_delta_BA':float(by.loc['AAPL','delta_BA']) if 'AAPL' in by.index else None,'AMZN_mean_delta_BA':float(by.loc['AMZN','delta_BA']) if 'AMZN' in by.index else None,'positive_macro_months':int((macro>0).sum()),'AAPL_mean_delta_Brier':float(by.loc['AAPL','delta_Brier']) if 'AAPL' in by.index else None,'AMZN_mean_delta_Brier':float(by.loc['AMZN','delta_Brier']) if 'AMZN' in by.index else None,'constant_outer_M1':bool(constant)}
    out['passes']=bool(out['outer_cells']==6 and out['AAPL_mean_delta_BA']>=.01 and out['AMZN_mean_delta_BA']>=.01 and out['positive_macro_months']>=2 and out['AAPL_mean_delta_Brier']<=.002 and out['AMZN_mean_delta_Brier']<=.002 and not out['constant_outer_M1'])
    return out
def build_attribution(pred):
    cells=[]
    for (s,m),g in pred[pred.month.isin(['2018-06','2018-07','2018-08'])].groupby(['symbol','month']): cells.append({'symbol':s,'month':m,'delta_BA':metric(g.label,g.M1)['BA']-metric(g.label,g.Mmeta)['BA'],'delta_Brier':metric(g.label,g.M1)['Brier']-metric(g.label,g.Mmeta)['Brier']})
    return {'comparison':'M1_vs_Mmeta','cells':cells,'macro_delta_BA':float(np.mean([x['delta_BA'] for x in cells])) if cells else None}

def run_pipeline(source_dir,out,mode='synthetic'):
    d,schedule,bars,manifest,expected,source=load_input(source_dir); out=Path(out); out.mkdir(parents=True,exist_ok=False); models=out/'models';models.mkdir()
    # Source contract is validated before any training, including synthetic inputs.
    required={'provider','feed','adjustment','timeframe','symbols','bar_hashes','request_range','acquired_at','duplicate_timestamp_count','unexpected_timestamp_count'}
    if mode=='real' and not {'provider':'Alpaca','feed':'SIP','adjustment':'raw','timeframe':'5Min','symbols':['SPY','QQQ']}.items() <= manifest.items(): raise ValueError('invalid real source manifest')
    if not required <= set(manifest): raise ValueError('incomplete source manifest')
    d,grid=add_market_features(d,schedule,bars); d['month']=d.month.astype(str); d['phase']=np.where(d.month<'2018-09','forward_oof',np.where(d.month<='2018-10','development','later'))
    if set(expected['keys'])!=set(d.loc[d.month>='2018-03','key']): raise ValueError('expected key manifest mismatch')
    d[['key']+MARKET_CONTROLS+MARKET_NUMERIC+['SPY_valid','QQQ_valid']].to_csv(out/'market_features.csv',index=False)
    coverage=d.groupby(['symbol','month'])[['SPY_valid','QQQ_valid']].mean().reset_index(); coverage.to_csv(out/'market_coverage.csv',index=False)
    if (coverage[['SPY_valid','QQQ_valid']]<.95).any().any(): raise ValueError('coverage gate failed')
    preds=[]; cv=[]; evidence=[]
    for symbol,g in d.groupby('symbol'):
        m0records=[]; months=sorted(g.month.unique())
        for month in months:
            if month<'2018-03': continue
            ev=g[g.month==month]; tr=g[g.month<month] if month<'2018-09' else g[g.month<'2018-09']
            if not len(tr) or len(ev)==0: continue
            c=select_c(m0records,month) if month<'2018-09' else final_c(m0records)
            if month<'2018-09':
                for cand in C_GRID:
                    temp,p=fit_one(tr,ev,METHODS['M0'],cand); r=metric(ev.label,p); m0records.append({'symbol':symbol,'fold':month,'C':cand,**r,'train_months':sorted(tr.month.unique())})
            for method,cols in METHODS.items():
                model,p=fit_one(tr,ev,cols,c); name=f'{method}_{symbol}_{month}'
                meta={'method':method,'symbol':symbol,'fold':month,'C':c,'columns':cols,'training_months':sorted(tr.month.unique()),'train_n':int(len(tr)),'eval_n':int(len(ev)),'train_key_hash':key_hash(tr.key),'eval_key_hash':key_hash(ev.key),'solver':SOLVER,'tol':TOL,'seed':SEED}
                save_model(models,name,model,meta); evidence.append(meta)
                for key,label,v in zip(ev.key,ev.label,p): preds.append({'key':key,'symbol':symbol,'month':month,'phase':ev.phase.iloc[0],'label':int(label),'method':method,'probability':float(v)})
        cv.extend(m0records)
    long=pd.DataFrame(preds); wide=long.pivot(index=['key','symbol','month','phase','label'],columns='method',values='probability').reset_index(); wide.to_csv(out/'predictions.csv',index=False)
    monthly=[]; aggregate=[]
    for (s,m,method),g in long.groupby(['symbol','month','method']): monthly.append({'symbol':s,'month':m,'method':method,**metric(g.label,g.probability)})
    for (phase,s,method),g in long.groupby(['phase','symbol','method']): aggregate.append({'phase':phase,'symbol':s,'method':method,**metric(g.label,g.probability)})
    pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False); pd.DataFrame(aggregate).to_csv(out/'metrics.csv',index=False); pd.DataFrame(cv).to_csv(out/'m0_cv.csv',index=False)
    (out/'training_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n'); (out/'advancement.json').write_text(json.dumps(build_gate(wide),indent=2)+'\n'); (out/'attribution.json').write_text(json.dumps(build_attribution(wide),indent=2)+'\n')
    fp={'mode':mode,'source_dir':str(source.resolve()),'methods':METHODS,'market_preregistration_sha256':sha(Path(__file__).with_name('MARKET_PREREGISTRATION.md')),'market_feature_code_sha256':sha(Path(__file__).with_name('market_features.py')),'runner_code_sha256':sha(Path(__file__).with_name('run_market.py')),'canonical_r1_input_sha256':sha(source/'r1_rows.csv'),'market_source_manifest_sha256':sha(source/'source_manifest.json'),'normalized_etf_bar_hashes':manifest['bar_hashes'],'calendar_schedule_sha256':sha(source/'schedule.csv'),'feature_columns':METHODS,'C_grid':list(C_GRID),'seed':SEED,'solver':SOLVER,'tol':TOL,'cutoff_rule':'bar_end_utc + 1 minute <= cutoff_utc','expected_key_hash':key_hash(expected['keys'])}
    (out/'protocol_fingerprint.json').write_text(json.dumps(fp,indent=2)+'\n')
    return out
