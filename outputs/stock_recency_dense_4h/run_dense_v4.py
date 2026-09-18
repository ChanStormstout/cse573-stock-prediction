"""Fair dense-window controls with a corrected June--August gate."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import INPUTS, R1_COLS, dump, load_data, metric_rows, scores, sha, standardize_fit, standardize_transform


def fit(train, evaluate, weights, name, private):
    if train.empty or train.end_utc.max() >= evaluate.cutoff_utc.min():
        raise AssertionError("dense model saw future label")
    mean, scale = standardize_fit(train, R1_COLS); x = standardize_transform(train, R1_COLS, mean, scale); z = standardize_transform(evaluate, R1_COLS, mean, scale)
    model = LogisticRegression(C=0.1, solver="liblinear", max_iter=3000, tol=1e-8, random_state=573)
    model.fit(x, train.label.to_numpy(int), sample_weight=np.asarray(weights, float)); p = model.predict_proba(z)[:, 1]
    path = private / "models" / f"{name}.joblib"; path.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"mean":mean,"scale":scale,"columns":R1_COLS,"model":model}, path)
    q = joblib.load(path)["model"].predict_proba(z)[:,1]; assert np.max(np.abs(p-q)) <= 1e-12
    return p, {"name":name,"train_n":len(train),"eval_n":len(evaluate),"weight_sum":float(np.sum(weights)),"model_sha256":sha(path),"reload_max_abs_error":float(np.max(np.abs(p-q)))}


def main(public_path, private_path):
    t0=time.monotonic(); public=Path(public_path); private=Path(private_path); public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    dense=pd.read_pickle(private/"dense_windows.pkl"); official=load_data().set_index("key")
    # Per-column parity against canonical official rows; this is diagnostic,
    # never silently mixed into a model if it fails.
    off=dense[dense.official==1].set_index("key"); rows=[]
    keys=sorted(set(off.index)&set(official.index))
    for c in R1_COLS:
        a=off.loc[keys,c].to_numpy(float); b=official.loc[keys,c].to_numpy(float); both=np.isfinite(a)&np.isfinite(b); dif=np.abs(a[ both]-b[ both]) if both.any() else np.array([])
        rows.append({"column":c,"rows_compared":len(keys),"finite_pairs":int(both.sum()),"max_abs_diff":float(dif.max()) if len(dif) else None,"unequal_at_1e-12":int(np.sum(dif>1e-12)) if len(dif) else 0,"dense_missing_only":int((~np.isfinite(a)&np.isfinite(b)).sum()),"official_missing_only":int((np.isfinite(a)&~np.isfinite(b)).sum())})
    parity=pd.DataFrame(rows); parity.to_csv(public/"dense_feature_parity.csv",index=False)
    parity_pass=bool((parity["unequal_at_1e-12"]==0).all() and (parity.dense_missing_only==0).all() and (parity.official_missing_only==0).all())
    # Even if parity passes, using one generator for every row keeps the
    # augmented and control models matched.  If it fails this is mandatory.
    feature_mode="reconstructed_all_official_and_augmented"
    fits=[]; chunks=[]; outer=[]
    for month in [f"2018-{m:02d}" for m in range(3,9)]:
        for symbol in ("AAPL","AMZN"):
            ev=dense[(dense.symbol==symbol)&(dense.month==month)&(dense.official==1)].copy()
            off_train=dense[(dense.symbol==symbol)&(dense.month<month)&(dense.official==1)].copy(); dense_train=dense[(dense.symbol==symbol)&(dense.month<month)].copy()
            if ev.empty: continue
            row=ev[["key","symbol","day","month","phase","label","target_return"]].copy().reset_index(drop=True)
            p0,e0=fit(off_train,ev,np.ones(len(off_train)),f"{symbol}_{month}_D0_equal",private);p0d,e0d=fit(off_train,ev,off_train.official_day_weight.to_numpy(),f"{symbol}_{month}_D0_day",private);p1,e1=fit(dense_train,ev,dense_train.day_weight.to_numpy(),f"{symbol}_{month}_D1",private)
            fits += [{**e0,"method":"D0_equal","symbol":symbol,"month":month},{**e0d,"method":"D0_day","symbol":symbol,"month":month},{**e1,"method":"D1","symbol":symbol,"month":month}]
            row["D0_equal"]=p0;row["D0_day"]=p0d;row["D1"]=p1;chunks.append(row)
            for m in ("D0_equal","D0_day","D1"): outer.append({"method":m,"symbol":symbol,"month":month,**scores(ev.label,row[m])})
    for symbol in ("AAPL","AMZN"):
        ev=dense[(dense.symbol==symbol)&(dense.month>="2018-09")&(dense.official==1)].copy(); off_train=dense[(dense.symbol==symbol)&(dense.month<"2018-09")&(dense.official==1)].copy();dense_train=dense[(dense.symbol==symbol)&(dense.month<"2018-09")].copy()
        row=ev[["key","symbol","day","month","phase","label","target_return"]].copy().reset_index(drop=True)
        p0,e0=fit(off_train,ev,np.ones(len(off_train)),f"{symbol}_final_D0_equal",private);p0d,e0d=fit(off_train,ev,off_train.official_day_weight.to_numpy(),f"{symbol}_final_D0_day",private);p1,e1=fit(dense_train,ev,dense_train.day_weight.to_numpy(),f"{symbol}_final_D1",private)
        fits += [{**e0,"method":"D0_equal","symbol":symbol,"month":"final"},{**e0d,"method":"D0_day","symbol":symbol,"month":"final"},{**e1,"method":"D1","symbol":symbol,"month":"final"}]
        row["D0_equal"]=p0;row["D0_day"]=p0d;row["D1"]=p1;chunks.append(row)
    pred=pd.concat(chunks,ignore_index=True);pred.to_csv(public/"dense_predictions.csv",index=False);m,mm=metric_rows(pred,["D0_equal","D0_day","D1"]);m.to_csv(public/"dense_metrics.csv",index=False);mm.to_csv(public/"dense_monthly_metrics.csv",index=False)
    out=pd.DataFrame(outer);out.to_csv(public/"dense_outer_scores.csv",index=False); gate_rows=out[out.month.isin(["2018-06","2018-07","2018-08"])]; assert len(gate_rows[gate_rows.method=="D1"])==6
    q=out[out.method=="D0_day"].merge(out[out.method=="D1"],on=["symbol","month"],suffixes=("_base","_new"));q["delta_BA"]=q.BA_new-q.BA_base;stock=q.groupby("symbol").delta_BA.mean();months=q.groupby("month").delta_BA.mean();gate={"baseline":"D0_day","candidate":"D1","months":["2018-06","2018-07","2018-08"],"n_month_rows":int(len(gate_rows[gate_rows.method=="D1"])),"n_month_stock_rows":int(len(q)),"AAPL_delta_BA":float(stock.get("AAPL",np.nan)),"AMZN_delta_BA":float(stock.get("AMZN",np.nan)),"macro_delta_BA":float(q.delta_BA.mean()),"positive_outer_months":int((months>0).sum()),"passes":bool(stock.min()>=.01 and stock.max()>=-.01 and (months>0).sum()>=2)}
    rsel=public.parent/"stock_recency_dense_4h"/"v4"/"recency_selection.json"; raw=json.loads(rsel.read_text()) if rsel.exists() else {}; r1_pass=any(x.get("method")=="R1" and x.get("passes") for x in raw.get("gate",[]))
    summary={"status":"COMPLETE","feature_mode":feature_mode,"canonical_parity_pass":parity_pass,"dense_gate_corrected":gate,"recency_R1_gate":r1_pass,"D2_status":"ELIGIBLE_TO_RUN" if r1_pass and gate["passes"] else "STOPPED_BY_GATE","D2_reason":"both recency R1 and dense gates required" if not (r1_pass and gate["passes"]) else "eligible","rows":int(len(dense)),"official_rows":int(dense.official.sum()),"runtime_seconds":time.monotonic()-t0}
    dump(public/"dense_gate_corrected.json",summary);dump(public/"dense_training_evidence.json",{"fits":fits,"summary":summary});dump(public/"dense_training_summary.json",summary);print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--public",default="outputs/stock_recency_dense_4h/v4");p.add_argument("--private",default="work/stock-data/recency_dense_4h/v4_dense");a=p.parse_args();main(a.public,a.private)
