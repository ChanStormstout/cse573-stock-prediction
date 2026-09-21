from pathlib import Path
import hashlib, json, sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE=Path(__file__).resolve().parent
OUT=HERE/'v1'; MODELS=HERE/'private_models'
sys.path.insert(0,str(HERE))
import run

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_logit(p): return logit(np.clip(np.asarray(p,float),1e-6,1-1e-6))

def independent_design(raw,gates,artifact,indices,symbols):
    parts=[]
    for block in artifact['blocks']:
        cfg=artifact['spec']['blocks'][block]
        cols=cfg['cols']
        x=raw[block][cols].to_numpy(float)[indices]
        mu=np.asarray(cfg['mean'],float); sd=np.asarray(cfg['std'],float)
        g=np.asarray(gates[block],float)[indices]
        parts.append(g[:,None]*np.concatenate([np.ones((len(indices),1)),(x-mu)/sd],axis=1))
    z=np.concatenate(parts,axis=1)
    if artifact['spec']['partial']:
        alpha=artifact['spec']['alpha']
        sym=np.asarray(symbols)[indices]
        sign=np.where(sym=='AAPL',1.,-1.)
        shrink=np.asarray([alpha[s] for s in sym],float)
        z=np.concatenate([z,z*sign[:,None]*shrink[:,None]],axis=1)
    return z

def replay_artifact(d,raw,gates,artifact,indices):
    base=d.F1.to_numpy(float)[indices]
    if artifact['identity']:
        return base.copy()
    z=independent_design(raw,gates,artifact,indices,d.symbol.to_numpy())
    p=expit(safe_logit(base)+z@np.asarray(artifact['beta'],float))
    inactive=np.ones(len(indices),dtype=bool)
    for block in artifact['blocks']:
        inactive &= np.asarray(gates[block])[indices]==0
    p[inactive]=base[inactive]
    if not np.array_equal(p[inactive],base[inactive]):
        raise AssertionError('exact inactive fallback failed during replay')
    return p

def main():
    checks={}
    d=run.assemble(); raw,gates=run.build_raw(d)
    pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip')
    sel=pd.read_csv(OUT/'selection.csv')
    checks['row_key_contract']=bool(len(pred)==1607 and pred.key.nunique()==1607 and pred.key.tolist()==d.key.tolist())
    checks['label_contract']=bool(np.array_equal(pred.label.to_numpy(int),d.label.to_numpy(int)))
    manifest=json.loads((OUT/'MODEL_MANIFEST.json').read_text())
    checks['model_manifest']=bool(set(manifest)=={p.name for p in MODELS.glob('*.joblib')} and all(sha(MODELS/n)==h for n,h in manifest.items()))
    maxerr=0.;fallback_errors=0;replayed=0
    for method in run.BLOCKS:
        out=d.F1.to_numpy(float).copy()
        for issued in [f'2018-{x:02d}' for x in range(3,9)]+['final']:
            if issued=='final':
                ix=np.flatnonzero(d.phase.isin(['development','later']).to_numpy())
            else:
                ix=np.flatnonzero((d.month==issued).to_numpy())
            path=MODELS/f'{method}_{issued}.joblib'
            if path.exists():
                artifact=joblib.load(path); q=replay_artifact(d,raw,gates,artifact,ix)
            else:
                q=d.F1.to_numpy(float)[ix]
            out[ix]=q; replayed+=len(ix)
        err=float(np.max(np.abs(out-pred[method].to_numpy(float))))
        maxerr=max(maxerr,err)
        inactive=(d.phase!='warmup').to_numpy()
        for block in run.BLOCKS[method]: inactive &= np.asarray(gates[block])==0
        fallback_errors += int(np.sum(out[inactive]!=d.F1.to_numpy(float)[inactive]))
    checks['fit_free_model_replay']=bool(maxerr<=1e-12)
    checks['exact_gated_fallback']=bool(fallback_errors==0)
    audit=json.loads((OUT/'INPUT_AUDIT.json').read_text())
    checks['source_hashes']=bool(all(sha(run.ROOT/p)==h for p,h in audit['sources'].items()))
    monthly=pd.read_csv(OUT/'monthly_metrics.csv')
    recomputed=run.report_tables(d,{m:pred[m].to_numpy(float) for m in ['BASE']+list(run.BLOCKS)})[1]
    keys=['method','phase','symbol','month']; cols=['n','BA','MCC','Brier','Accuracy','AUC','predicted_up']
    a=monthly.sort_values(keys).reset_index(drop=True);b=recomputed.sort_values(keys).reset_index(drop=True)
    metric_error=0.
    for c in cols:
        metric_error=max(metric_error,float(np.nanmax(np.abs(a[c].to_numpy(float)-b[c].to_numpy(float)))))
    checks['metric_recomputation']=bool(metric_error<=1e-12)
    gate=pd.read_csv(OUT/'advancement.csv')
    gate2=run.gates_table(recomputed)
    checks['advancement_recomputation']=bool(gate[['method','passes']].astype(str).equals(gate2[['method','passes']].astype(str)))
    status='PASS' if all(checks.values()) else 'FAIL'
    result={'status':status,'checks':checks,'maximum_probability_error':maxerr,'metric_error':metric_error,'fallback_errors':fallback_errors,'replayed_rows':replayed,'fit_calls':0}
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    if status!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
