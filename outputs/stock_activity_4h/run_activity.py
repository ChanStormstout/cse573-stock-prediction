"""Guarded A1 runner. Importing this file for A0 parity never scores A1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import ACTIVITY_FEATURES, CS, HERE, PRIVATE, metric, sha
from prepare import ensure

R1 = [*(f"{kind}_{i}" for i in range(1,7) for kind in ("return","range")),"history_age_hours","return_mean","return_std","ny_hour",*(x for m in (5,15,30,60) for x in (f"recent_return_{m}",f"recent_range_{m}",f"recent_rv_{m}",f"recent_missing_{m}")),"overnight_gap","overnight_gap_missing","minutes_from_open","minutes_to_close"]

class StandardizeMissing:
    def fit(self, frame, columns):
        values=frame[columns].to_numpy(float); self.mean=np.nanmean(values,axis=0); self.mean=np.where(np.isfinite(self.mean),self.mean,0.0); self.scale=np.nanstd(values,axis=0); self.scale=np.where(np.isfinite(self.scale)&(self.scale>1e-12),self.scale,1.0); return self
    def transform(self,frame,columns):
        return np.where(np.isfinite((frame[columns].to_numpy(float)-self.mean)/self.scale),(frame[columns].to_numpy(float)-self.mean)/self.scale,0.0)

def fit_probability(train, evaluate, columns, C, name, model_dir):
    if train.end_utc.max() >= evaluate.cutoff_utc.min(): raise AssertionError("training chronology")
    scaler=StandardizeMissing().fit(train,columns); x=scaler.transform(train,columns); z=scaler.transform(evaluate,columns)
    model=LogisticRegression(C=C,solver="liblinear",max_iter=3000,random_state=573,tol=1e-8).fit(x,train.label)
    if model.n_iter_.max()>=3000: raise RuntimeError("no convergence")
    path=model_dir/f"{name}.joblib"; joblib.dump({"columns":columns,"scaler":scaler,"model":model},path); loaded=joblib.load(path); p=model.predict_proba(z)[:,1]; q=loaded["model"].predict_proba(loaded["scaler"].transform(evaluate,columns))[:,1]
    np.testing.assert_allclose(p,q,atol=1e-12,rtol=0)
    return p,{"C":C,"train_end_max":train.end_utc.max().isoformat(),"eval_cutoff_min":evaluate.cutoff_utc.min().isoformat(),"model_sha256":sha(path),"reload_max_abs_error":float(np.abs(p-q).max()),"iterations":int(model.n_iter_.max())}

def choose(records):
    return sorted([(-np.mean([x["BA"] for x in records if x["C"]==c]),np.mean([x["Brier"] for x in records if x["C"]==c]),c) for c in CS])[0][2]

def reproduce_a0(data=None):
    """Allowed canonical R1 reproduction; returns only A0 evidence/predictions."""
    data=ensure() if data is None else data; models=PRIVATE/"a0_models"; models.mkdir(parents=True,exist_ok=True); records=[]; out=[]; evidence=[]
    for symbol, group in data.groupby("symbol"):
        cv=[]
        for month in [f"2018-{m:02d}" for m in range(3,9)]:
            train=group[(group.month<month)&(group.end_utc < group.loc[group.month.eq(month),"cutoff_utc"].min())]; test=group[group.month.eq(month)]; chosen=.1 if not cv else choose(cv)
            allp={}
            for c in CS:
                p,e=fit_probability(train,test,R1,c,f"a0_{symbol}_{month}_{c}",models); allp[c]=p; r={"symbol":symbol,"month":month,"C":c,**metric(test.label,p)}; cv.append(r); records.append(r); evidence.append({"phase":"forward",**r,**e})
            out.append(test[["key","symbol","month","label"]].assign(phase="train_forward_oof",A0=allp[chosen])); evidence.append({"phase":"selected", "symbol":symbol,"month":month,"A0_C":chosen})
        train=group[group.month<"2018-09"]; test=group[group.month>="2018-09"]; chosen=choose(cv); p,e=fit_probability(train,test,R1,chosen,f"a0_{symbol}_frozen",models); phase=np.where(test.month.isin(["2018-09","2018-10"]),"development","later"); out.append(test[["key","symbol","month","label"]].assign(phase=phase,A0=p)); evidence.append({"phase":"frozen","symbol":symbol,"A0_C":chosen,**e})
    prediction=pd.concat(out).sort_values(["key"]); PRIVATE.mkdir(parents=True,exist_ok=True); prediction.to_csv(PRIVATE/"a0_predictions_private.csv",index=False); (PRIVATE/"a0_evidence.json").write_text(json.dumps({"cv":records,"evidence":evidence},indent=2)+"\n")
    return prediction,records,evidence

def run(output):
    if output.exists(): raise FileExistsError(output)
    data=ensure(); model_dir=PRIVATE/"v1_models"; model_dir.mkdir(parents=True,exist_ok=True); a0,cv,evidence=reproduce_a0(data); chunks=[]
    for symbol, group in data.groupby("symbol"):
        a1cv=[]
        for month in [f"2018-{m:02d}" for m in range(3,9)]:
            test=group[group.month.eq(month)]; train=group[(group.month<month)&(group.end_utc<test.cutoff_utc.min())]; c0=[x for x in evidence if x.get("phase")=="selected" and x.get("symbol")==symbol and x.get("month")==month][0]["A0_C"]; c1=.1 if not a1cv else choose(a1cv); row=test[["key","symbol","month","label"]].copy()
            for name,c in (("A1",c1),("A1_matchedC",c0)):
                p,e=fit_probability(train,test,R1+ACTIVITY_FEATURES,c,f"{name}_{symbol}_{month}",model_dir); row[name]=p; evidence.append({"phase":"forward","method":name,"symbol":symbol,"month":month,**e})
                if name=="A1":
                    for cc in CS:
                        pp,_=fit_probability(train,test,R1+ACTIVITY_FEATURES,cc,f"A1_{symbol}_{month}_{cc}",model_dir); a1cv.append({"symbol":symbol,"month":month,"C":cc,**metric(test.label,pp)})
            chunks.append(row.assign(phase="train_forward_oof"))
        test=group[group.month>="2018-09"]; train=group[group.month<"2018-09"]; c1=choose(a1cv); c0=[x for x in evidence if x.get("phase")=="frozen" and x.get("symbol")==symbol][0]["A0_C"]; row=test[["key","symbol","month","label"]].copy()
        for name,c in (("A1",c1),("A1_matchedC",c0)): row[name]=fit_probability(train,test,R1+ACTIVITY_FEATURES,c,f"{name}_{symbol}_frozen",model_dir)[0]
        chunks.append(row.assign(phase=np.where(test.month.isin(["2018-09","2018-10"]),"development","later")))
    pred=a0.merge(pd.concat(chunks),on=["key","symbol","month","label","phase"],validate="one_to_one"); output.mkdir(parents=True); pred.to_csv(output/"predictions.csv",index=False)
    raise RuntimeError("Post-run aggregation is intentionally reserved for verifier-reviewed Activity v1 execution.")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--approve-activity-run",action="store_true"); p.add_argument("--output",type=Path,default=HERE/"v1"); a=p.parse_args()
    if not a.approve_activity_run: raise SystemExit("Activity v1 is preregistered but paused. Run only after explicit APPROVE_ACTIVITY_RUN.")
    run(a.output)
