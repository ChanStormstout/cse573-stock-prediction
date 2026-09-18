"""Guarded A1 runner. Importing this file for A0 parity never scores A1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import ACTIVITY_FEATURES, CS, HERE, PRIVATE, metric, sha
from prepare import ensure

R1 = [*(f"{kind}_{i}" for i in range(1,7) for kind in ("return","range")),"history_age_hours","return_mean","return_std","ny_hour",*(x for m in (5,15,30,60) for x in (f"recent_return_{m}",f"recent_range_{m}",f"recent_rv_{m}",f"recent_missing_{m}")),"overnight_gap","overnight_gap_missing","minutes_from_open","minutes_to_close"]
MONTHS = [f"2018-{m:02d}" for m in range(3,9)]

def selected_evidence_c(evidence, *, method, symbol, phase, month=None):
    rows=[r for r in evidence if r.get("method")==method and r.get("symbol")==symbol and r.get("phase")==phase and (month is None or r.get("month")==month)]
    if len(rows)!=1: raise AssertionError(f"expected one selected evidence row: {method=} {symbol=} {phase=} {month=}")
    value=rows[0].get("C")
    if value not in CS: raise AssertionError(f"invalid selected C: {value}")
    return float(value)

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
    return p,{"C":C,"train_n":int(len(train)),"eval_n":int(len(evaluate)),"training_months":sorted(train.month.unique().tolist()),"train_target_end_max":train.end_utc.max().isoformat(),"eval_cutoff_min":evaluate.cutoff_utc.min().isoformat(),"time_safe":bool(train.end_utc.max()<evaluate.cutoff_utc.min()),"feature_columns":list(columns),"model_path":str(path),"model_sha256":sha(path),"reload_max_abs_error":float(np.abs(p-q).max()),"iterations":int(model.n_iter_.max())}

def choose(records):
    return sorted([(-np.mean([x["BA"] for x in records if x["C"]==c]),np.mean([x["Brier"] for x in records if x["C"]==c]),c) for c in CS])[0][2]

def reproduce_a0(data=None):
    """Allowed canonical R1 reproduction; returns only A0 evidence/predictions."""
    data=ensure() if data is None else data; models=PRIVATE/"a0_models"; models.mkdir(parents=True,exist_ok=True); records=[]; out=[]; evidence=[]
    for symbol, group in data.groupby("symbol"):
        cv=[]; selected_candidates={}
        for month in [f"2018-{m:02d}" for m in range(3,9)]:
            train=group[(group.month<month)&(group.end_utc < group.loc[group.month.eq(month),"cutoff_utc"].min())]; test=group[group.month.eq(month)]; chosen=.1 if not cv else choose(cv)
            allp={}
            for c in CS:
                p,e=fit_probability(train,test,R1,c,f"a0_{symbol}_{month}_{c}",models); allp[c]=p; r={"symbol":symbol,"month":month,"method":"A0","C":c,**metric(test.label,p)}; cv.append(r); records.append(r); evidence.append({"phase":"candidate",**r,**e}); selected_candidates[(symbol,month,c)]=e
            out.append(test[["key","symbol","month","label"]].assign(phase="train_forward_oof",A0=allp[chosen])); evidence.append({"phase":"selected","symbol":symbol,"month":month,"method":"A0",**selected_candidates[(symbol,month,chosen)]})
        train=group[group.month<"2018-09"]; test=group[group.month>="2018-09"]; chosen=choose(cv); p,e=fit_probability(train,test,R1,chosen,f"a0_{symbol}_frozen",models); phase=np.where(test.month.isin(["2018-09","2018-10"]),"development","later"); out.append(test[["key","symbol","month","label"]].assign(phase=phase,A0=p)); evidence.append({"phase":"frozen","symbol":symbol,"method":"A0",**e})
    prediction=pd.concat(out).sort_values(["key"]); PRIVATE.mkdir(parents=True,exist_ok=True); prediction.to_csv(PRIVATE/"a0_predictions_private.csv",index=False); (PRIVATE/"a0_evidence.json").write_text(json.dumps({"cv":records,"evidence":evidence},indent=2)+"\n")
    return prediction,records,evidence

def run(output):
    if output.exists(): raise FileExistsError(output)
    data=ensure(); model_dir=PRIVATE/"v1_models"; model_dir.mkdir(parents=True,exist_ok=False); a0,cv,evidence=reproduce_a0(data); all_cv=list(cv); chunks=[]; selection=[]
    for symbol, group in data.groupby("symbol"):
        a1cv=[]; a1_candidates={}
        for month in [f"2018-{m:02d}" for m in range(3,9)]:
            test=group[group.month.eq(month)]; train=group[(group.month<month)&(group.end_utc<test.cutoff_utc.min())]; c0=selected_evidence_c(evidence,method="A0",symbol=symbol,phase="selected",month=month); c1=.1 if not a1cv else choose(a1cv); row=test[["key","symbol","month","label"]].copy(); selection += [{"symbol":symbol,"month":month,"method":"A0","chosen_C":c0,"eligible_prior_months":[m for m in MONTHS if m<month],"selection_rule":"past forward BA, Brier, smaller C"},{"symbol":symbol,"month":month,"method":"A1","chosen_C":c1,"eligible_prior_months":[m for m in MONTHS if m<month],"selection_rule":"past forward BA, Brier, smaller C"},{"symbol":symbol,"month":month,"method":"A1_matchedC","source_method":"A0","chosen_C":c0,"eligible_prior_months":[m for m in MONTHS if m<month],"selection_rule":"same-fold A0 C"}]
            for name,c in (("A1",c1),("A1_matchedC",c0)):
                p,e=fit_probability(train,test,R1+ACTIVITY_FEATURES,c,f"{name}_{symbol}_{month}",model_dir); row[name]=p; evidence.append({"phase":"selected","method":name,"symbol":symbol,"month":month,**e})
                if name=="A1":
                    for cc in CS:
                        pp,ee=fit_probability(train,test,R1+ACTIVITY_FEATURES,cc,f"A1_{symbol}_{month}_{cc}",model_dir); record={"symbol":symbol,"month":month,"method":"A1","C":cc,**metric(test.label,pp)}; a1cv.append(record); all_cv.append(record); a1_candidates[(month,cc)]=ee
            chunks.append(row.assign(phase="train_forward_oof"))
        test=group[group.month>="2018-09"]; train=group[group.month<"2018-09"]; c1=choose(a1cv); c0=selected_evidence_c(evidence,method="A0",symbol=symbol,phase="frozen"); row=test[["key","symbol","month","label"]].copy(); selection += [{"symbol":symbol,"month":"final","method":"A0","chosen_C":c0,"eligible_prior_months":MONTHS},{"symbol":symbol,"month":"final","method":"A1","chosen_C":c1,"eligible_prior_months":MONTHS},{"symbol":symbol,"month":"final","method":"A1_matchedC","source_method":"A0","chosen_C":c0,"eligible_prior_months":MONTHS,"selection_rule":"same-fold A0 C"}]
        for name,c in (("A1",c1),("A1_matchedC",c0)):
            pp,ee=fit_probability(train,test,R1+ACTIVITY_FEATURES,c,f"{name}_{symbol}_frozen",model_dir); row[name]=pp; evidence.append({"phase":"frozen","method":name,"symbol":symbol,"month":"final",**ee})
        chunks.append(row.assign(phase=np.where(test.month.isin(["2018-09","2018-10"]),"development","later")))
    pred=a0.merge(pd.concat(chunks),on=["key","symbol","month","label","phase"],validate="one_to_one"); output.mkdir(parents=True); pred.to_csv(output/"predictions.csv",index=False)
    def rows(frame,keys):
        result=[]
        for values,g in frame.groupby(keys):
            values=(values,) if not isinstance(values,tuple) else values; base=g.A0.to_numpy(float)>=.5
            for method in ("A0","A1","A1_matchedC"):
                q=g[method].to_numpy(float)>=.5; result.append(dict(zip(keys,values))|{"method":method,**metric(g.label,g[method]),"changed_direction_count":int((q!=base).sum()),"repaired_A0_errors":int(((q==g.label.to_numpy())&(base!=g.label.to_numpy())).sum()),"introduced_errors":int(((q!=g.label.to_numpy())&(base==g.label.to_numpy())).sum())})
        return pd.DataFrame(result)
    monthly=rows(pred,["symbol","month","phase"]); monthly.to_csv(output/"monthly_metrics.csv",index=False); aggregate=rows(pred,["symbol","phase"]);aggregate.to_csv(output/"metrics.csv",index=False);pd.DataFrame(all_cv).to_csv(output/"cv.csv",index=False);(output/"selection.json").write_text(json.dumps(selection,indent=2)+"\n");(output/"training_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    outer=monthly[monthly.month.isin(["2018-06","2018-07","2018-08"])];
    def contrast(method):
        x=outer[outer.method.eq(method)].set_index(["symbol","month"]);b=outer[outer.method.eq("A0")].set_index(["symbol","month"]);d=x.BA-b.BA;db=x.Brier-b.Brier;stock=d.groupby(level=0).mean();month=d.groupby(level=1).mean();brier=db.groupby(level=0).mean();constant=x.groupby(level=0).constant.any();return {"contrast":f"{method}_vs_A0","AAPL_mean_delta_BA":float(stock["AAPL"]),"AMZN_mean_delta_BA":float(stock["AMZN"]),"weaker_stock_mean_delta_BA":float(stock.min()),"macro_delta_BA":float(d.mean()),"positive_macro_months":int((month>0).sum()),"AAPL_mean_delta_Brier":float(brier["AAPL"]),"AMZN_mean_delta_Brier":float(brier["AMZN"]),"max_stock_mean_delta_Brier":float(brier.max()),"constant_any_stock":bool(constant.any()),"outer_cells":int(len(d)),"passes":bool(len(d)==6 and stock["AAPL"]>=.01 and stock["AMZN"]>=.01 and (month>0).sum()>=2 and brier["AAPL"]<=.002 and brier["AMZN"]<=.002 and not constant.any())}
    advancement=contrast("A1");attribution=contrast("A1_matchedC");(output/"advancement.json").write_text(json.dumps(advancement,indent=2)+"\n");(output/"attribution_control.json").write_text(json.dumps(attribution,indent=2)+"\n");data.groupby("symbol")[ACTIVITY_FEATURES].agg(["count","mean","std"]).to_csv(output/"activity_feature_summary.csv");files=[HERE/"PRE_REGISTRATION.md",HERE/"common.py",HERE/"prepare.py",HERE/"run_activity.py",HERE.parents[1]/"work/stock-data/activity_4h/features.pkl",HERE.parents[1]/"work/stock-data/activity_4h/sources.json",HERE.parent/"stock_goal60_4h/v1/predictions.csv",HERE.parent/"stock_nextgen_4h/runs/v1/PREDICTIONS.csv"];(output/"protocol_fingerprint.json").write_text(json.dumps({"source_hashes":{str(x):sha(x) for x in files},"r1":R1,"activity_features":ACTIVITY_FEATURES,"cs":CS,"registered_forward_months":MONTHS,"history_sessions":20,"min_history_sessions":10},indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--approve-activity-run",action="store_true"); p.add_argument("--output",type=Path,default=HERE/"v1"); a=p.parse_args()
    if not a.approve_activity_run: raise SystemExit("Activity v1 is preregistered but paused. Run only after explicit APPROVE_ACTIVITY_RUN.")
    run(a.output)
