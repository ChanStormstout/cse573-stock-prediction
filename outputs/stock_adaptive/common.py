"""Past-only monthly replay utilities. No training at import."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(O/'stock_robust'))
spec=importlib.util.spec_from_file_location('robust_common_e13',O/'stock_robust/common.py')
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
baseline,PRICE,scores,sha,verify_files,require_empty=[getattr(robust,k) for k in ['baseline','PRICE','scores','sha','verify_files','require_empty']]

def load_data():
    d=pd.concat([pd.read_pickle(O/'stock_robust/results/data.pkl'),pd.read_pickle(O/'stock_final_test/results/test_inputs.pkl')],ignore_index=True)
    d=d.sort_values(['symbol','start_utc']).reset_index(drop=True)
    for c in ['start_utc','end_utc','cutoff_utc']:d[c]=pd.to_datetime(d[c],utc=True,format='mixed')
    assert len(d)==3233 and not d.duplicated(['symbol','start_utc']).any()
    assert np.isfinite(d[PRICE].to_numpy()).all()
    return d

def train_rows(d,month,policy):
    hi=pd.Timestamp(month+'-01',tz='UTC');lo=hi-pd.DateOffset(months=6) if policy=='rolling6' else pd.Timestamp('2018-01-01',tz='UTC')
    return d[(d.start_utc>=lo)&(d.end_utc<hi)].copy()

def inner_folds(train,month,policy):
    hi=pd.Timestamp(month+'-01',tz='UTC')
    for offset in [3,2,1]:
        v=hi-pd.DateOffset(months=offset)
        a=train[train.end_utc<v].copy();b=train[(train.start_utc>=v)&(train.start_utc<v+pd.DateOffset(months=1))].copy()
        assert len(a)>=100 and len(b)>=20 and a.label.nunique()==2 and b.label.nunique()==2
        assert a.end_utc.max()<b.cutoff_utc.min()
        yield v.strftime('%Y-%m'),a,b

def metrics(d,p):
    return dict(n=len(d),**scores(d.label.to_numpy(),np.asarray(p)))

def row_keys(d):return (d.symbol+'|'+d.start_utc.astype(str)).tolist()

def snapshot(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_sha256.json'}

def select(train,month,policy,factory,candidates):
    grid=[]
    for config in candidates:
        for fold,a,b in inner_folds(train,month,policy):
            model=factory(config).fit(a,a.label)
            grid.append(dict(config=config,fold=fold,n_train=len(a),n_valid=len(b),train_end=str(a.end_utc.max()),valid_cutoff=str(b.cutoff_utc.min()),**scores(b.label.to_numpy(),model.predict_proba(b)[:,1])))
    g=pd.DataFrame(grid)
    best=g.groupby('config')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','config'],ascending=[False,True,False]).iloc[0]
    return float(best.config),g
