"""M01: immutable, past-only experiment infrastructure."""
from pathlib import Path
import json,hashlib,importlib.util,sys,time,platform
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
B=Path(__file__).resolve().parent; O=B.parent; ROOT=O.parent; W=ROOT/'work/stock-data'
spec=importlib.util.spec_from_file_location('adaptive_reuse',O/'stock_adaptive/common.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
PRICE=old.PRICE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
def keys(d):return (d.symbol+'|'+d.start_utc.astype(str)).tolist()
def verify(manifest):
    for p,h in manifest.items():
        if sha(p)!=h:raise RuntimeError('Fingerprint mismatch: '+p)
def temporal(a,b):
    if len(a)==0 or len(b)==0 or a.end_utc.max()>=b.cutoff_utc.min():raise ValueError('Future/overlapping training labels')
def folds(d):
    for mo in [6,7,8]:
        lo=pd.Timestamp(f'2018-{mo:02}-01',tz='UTC');hi=lo+pd.DateOffset(months=1)
        a=d[d.end_utc<lo];b=d[(d.start_utc>=lo)&(d.start_utc<hi)];temporal(a,b);yield str(lo)[:7],a,b

def metrics(y,p):
    p=np.asarray(p);pred=p>=.5
    return dict(n=len(p),**old.scores(np.asarray(y),p),predicted_up=float(pred.mean()),constant_prediction=bool(np.all(pred==pred[0])),p_equal_half=float(np.mean(p==.5)))
def model(k=100,penalty='l2',C=.1):
    m=old.baseline('paper_stem_body',C)
    return m.set_params(features__words__selectkbest__k=k,model__l1_ratio=int(penalty=='l1'))
def price(C=.1):return old.baseline('price',C)
def pred_rows(d,p,method,**extra):
    z=d[['symbol','start_utc','end_utc','cutoff_utc','split','label','has_news']].copy();z['p']=p;z['method']=method
    for k,v in extra.items():z[k]=v
    return z

def save_model(m,a,b,folder,p=None):
    folder.mkdir(parents=True,exist_ok=False);joblib.dump(m,folder/'model.joblib')
    p=m.predict_proba(b)[:,1] if p is None else p
    np.testing.assert_allclose(joblib.load(folder/'model.joblib').predict_proba(b)[:,1],p,atol=1e-10,rtol=0)
    dump(folder/'manifest.json',{'train_keys':keys(a),'prediction_keys':keys(b),'train_end':a.end_utc.max(),'predict_start':b.cutoff_utc.min(),'model_sha256':sha(folder/'model.joblib')})

def prepare(run,config):
    if run.exists():raise FileExistsError('Refuse existing experiment directory: '+str(run))
    d=old.load_data();run.mkdir(parents=True)
    src=[O/'stock_robust/results/data.pkl',O/'stock_final_test/results/test_inputs.pkl',O/'stock_adaptive/common.py',O/'stock_robust/common.py',O/'stock_baseline/run_baseline.py']
    src+=list(B.glob('*.py'))
    dump(run/'sources.json',{str(p):sha(p) for p in src});dump(run/'config.json',config)
    d.to_pickle(run/'data.pkl');dump(run/'input_hash.json',{str(run/'data.pkl'):sha(run/'data.pkl'),str(run/'config.json'):sha(run/'config.json')})
    d[['symbol','start_utc','end_utc','cutoff_utc','split','label']].to_csv(run/'samples.csv',index=False)
    import importlib.metadata as md
    dump(run/'environment.json',{'python':sys.version,'platform':platform.platform(),'packages':{x:md.version(x) for x in ['numpy','pandas','scipy','scikit-learn','joblib','torch','transformers']}})

def load(run):
    verify(json.loads((run/'sources.json').read_text()));verify(json.loads((run/'input_hash.json').read_text()));return pd.read_pickle(run/'data.pkl')

def stage(run,name,func):
    load(run);folder=run/name
    if folder.exists():raise FileExistsError('Stage already exists; preserve failed/complete run: '+str(folder))
    folder.mkdir();t=time.monotonic();dump(folder/'status.json',{'state':'running'})
    try:
        func(folder);dump(folder/'status.json',{'state':'complete','elapsed_seconds':time.monotonic()-t})
    except Exception as e:
        dump(folder/'status.json',{'state':'failed','error':repr(e),'elapsed_seconds':time.monotonic()-t});raise
    dump(folder/'hashes.json',{str(p):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='hashes.json'})
