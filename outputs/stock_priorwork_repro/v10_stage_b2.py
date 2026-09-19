"""Run the frozen V10 Stage B2 NEWS+PRICE branches exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/stock-data"
BASE = ROOT / "outputs/stock_priorwork_repro"
OUT = BASE / "v10"
V9 = BASE / "v9"
PRIVATE = WORK / "priorwork_v10/models/joint"
MONTHS = pd.period_range("2018-09", "2019-02", freq="M").astype(str).tolist()
R1 = [f"{kind}_{i}" for i in range(1, 7) for kind in ("return", "range")] + [
    "history_age_hours", "return_mean", "return_std", "ny_hour"
]
DPRICE = ["DRET_1", "DRET_2", "DRET_5", "RANGE_1", "RV_5", "MEAN_5", "HISTORY_AGE_HOURS"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def pjson(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def metric(y, p) -> dict:
    y = np.asarray(y, int); p = np.asarray(p, float); z = (p >= .5).astype(int)
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y,z)),
            "BA": float(balanced_accuracy_score(y,z)) if both else None,
            "MCC": float(matthews_corrcoef(y,z)) if both else None,
            "precision": float(precision_score(y,z,zero_division=0)),
            "recall": float(recall_score(y,z,zero_division=0)),
            "F1": float(f1_score(y,z,zero_division=0)),
            "up_recall": float(recall_score(y,z,pos_label=1,zero_division=0)),
            "down_recall": float(recall_score(y,z,pos_label=0,zero_division=0)),
            "pred_up": float(z.mean()), "true_up": float(y.mean()),
            "AUC": float(roc_auc_score(y,p)) if both else None,
            "Brier": float(brier_score_loss(y,p)), "constant": bool(z.min()==z.max())}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sys.path.insert(0, str(BASE))
    import v10_stage_a2 as stage_a
    import v10_stage_b1_final_verify as stage_b1
    four, daily, source_audit = stage_a.reconstruct_inputs()
    raw_daily, raw_audit = stage_b1.reconstruct_daily()
    four = four.copy()
    four["horizon_window"] = "4h"
    four["phase"] = four["split"].replace({"oof":"OOF"})
    four["target_key"] = four["symbol"].astype(str)+"|"+four["start_utc"].astype(str)
    daily = daily.copy().rename(columns={"news_window":"horizon_window"})
    daily["horizon_window"] = "1d:" + daily["horizon_window"].astype(str)
    daily["phase"] = daily["split"].replace({"oof":"OOF"})
    daily["target_key"] = daily["symbol"].astype(str)+"|"+daily["start_utc"].astype(str)
    price = raw_daily.rename(columns={"stock":"symbol"})
    daily = daily.merge(price[["symbol","day",*DPRICE]], on=["symbol","day"], how="left", validate="one_to_one")
    goal = pd.read_csv(ROOT/"outputs/stock_goal60_4h/v1/predictions.csv")
    canonical = pd.read_pickle(WORK/"nextgen_4h/price_v1/features.pkl")
    parity = 0.0
    for col in R1:
        parity = max(parity, float(np.nanmax(np.abs(four[col].to_numpy(float)-canonical[col].to_numpy(float)))))
    audit = {"four_rows":len(four),"daily_rows":len(daily),"stage_a_source_audit":source_audit,
             "daily_raw_audit":raw_audit,"r1_feature_count":len(R1),"dprice_feature_count":len(DPRICE),
             "r1_max_feature_parity_error":parity,"goal60_key_count":int(goal.key.nunique())}
    return four, daily, audit


def vectorizer(method: str):
    if method == "PAPER_1G_L1LR":
        return CountVectorizer(binary=True,ngram_range=(1,1),min_df=3), "count_1g"
    if method == "PAPER_2G_L1LR":
        return CountVectorizer(binary=True,ngram_range=(2,2),min_df=3), "count_2g"
    return TfidfVectorizer(ngram_range=(1,1),min_df=3,max_features=10000,sublinear_tf=True), "tfidf_1g"


def estimator(method: str, params: dict):
    if method == "TFIDF_LR":
        return LogisticRegression(penalty="l2",solver="liblinear",max_iter=3000,random_state=573,**params), "LR"
    if method.startswith("PAPER_"):
        return LogisticRegression(penalty="l1",solver="liblinear",max_iter=3000,random_state=573,**params), "L1_LR"
    if method == "TFIDF_RF":
        return RandomForestClassifier(n_estimators=300,random_state=573,n_jobs=1,**params), "RF"
    if method == "TFIDF_ADABOOST":
        return AdaBoostClassifier(random_state=573,**params), "ADABOOST"
    if method == "TFIDF_KNN":
        return KNeighborsClassifier(metric="cosine",**params), "KNN"
    raise AssertionError(method)


def fit_joint(method: str, params: dict, train: pd.DataFrame, ev: pd.DataFrame, price_cols: list[str]):
    vec, text_type = vectorizer(method)
    xt = vec.fit_transform(train.stem_body.fillna("").astype(str))
    xe = vec.transform(ev.stem_body.fillna("").astype(str))
    selector = SelectKBest(chi2,k=min(500,xt.shape[1])).fit(xt,train.label)
    xt = selector.transform(xt); xe = selector.transform(xe)
    scaler = StandardScaler().fit(train[price_cols])
    pt = sparse.csr_matrix(scaler.transform(train[price_cols])); pe = sparse.csr_matrix(scaler.transform(ev[price_cols]))
    xj = sparse.hstack([xt,pt],format="csr"); xje = sparse.hstack([xe,pe],format="csr")
    clf, family = estimator(method,params); clf.fit(xj,train.label)
    probability = np.asarray(clf.predict_proba(xje)[:,1],float)
    bundle = {"vectorizer":vec,"selector":selector,"price_scaler":scaler,"classifier":clf,
              "text_method":method,"classifier_family":family,"frozen_parameter":params,
              "text_feature_type":text_type,"price_columns":price_cols,"no_news_policy":"JOINT_ZERO_TEXT_KEEP_PRICE"}
    return probability,bundle,xe


def predict_bundle(bundle: dict, ev: pd.DataFrame) -> np.ndarray:
    x=bundle["vectorizer"].transform(ev.stem_body.fillna("").astype(str)); x=bundle["selector"].transform(x)
    p=sparse.csr_matrix(bundle["price_scaler"].transform(ev[bundle["price_columns"]]))
    return np.asarray(bundle["classifier"].predict_proba(sparse.hstack([x,p],format="csr"))[:,1],float)


def protected_hashes() -> dict:
    dirs=["outputs/stock_priorwork_repro/v6","outputs/stock_priorwork_repro/v7","outputs/stock_priorwork_repro/v8","outputs/stock_priorwork_repro/v9","outputs/stock_priorwork_repro/v10","outputs/stock_integrated_4h/runs/v1","outputs/stock_paper_methods_4h/v1","outputs/stock_goal60_4h/v1","outputs/stock_context_4h"]
    files=subprocess.check_output(["git","ls-files",*dirs],cwd=ROOT,text=True).splitlines()
    files=[x for x in files if "STAGE_B2_PRE_RUN" not in x and not Path(x).name.startswith(("JOINT_","INCREMENTAL_","F1_COMPARISON","DAILY_WINDOW_COMPARISON","STAGE_B2_FINAL","STAGE_B2_REPORT","CLASSICAL_LANE_FINAL"))]
    news={p.name:sha(p) for p in sorted((WORK/"priorwork_v10/models/news_only").glob("*.joblib"))}
    dprice={p.name:sha(p) for p in sorted((WORK/"priorwork_v10/models/dprice").glob("*.joblib"))}
    return {"tracked_files":{x:sha(ROOT/x) for x in files},"news_only_private_models":news,"dprice_private_models":dprice,
            "explicit_inputs":{x:sha(OUT/x) for x in ["STAGE_B2_FROZEN_CONFIGURATION.json","METHOD_SELECTIONS.json","ISSUED_PARAMS_4H.csv","ISSUED_PARAMS_1D.csv","DPRICE_PREDICTIONS.csv","DPRICE_MODEL_MANIFEST.json"]}}


def phase_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    monthly=[]; aggregate=[]
    for keys,g in pred.groupby(["stock","horizon_window","method","month"]): monthly.append(dict(zip(["stock","horizon_window","method","month"],keys),phase=g.phase.iloc[0],**metric(g.label,g.probability)))
    for keys,g in pred.groupby(["stock","horizon_window","method","phase"]): aggregate.append(dict(zip(["stock","horizon_window","method","phase"],keys),evaluation_label="EXPOSED EXPLORATORY HISTORICAL BACKTEST" if keys[1]=="4h" else "PREREGISTERED_NEW_HORIZON_HISTORICAL_EVALUATION",**metric(g.label,g.probability)))
    return pd.DataFrame(monthly),pd.DataFrame(aggregate)


def baseline_metric(df: pd.DataFrame, score_col: str, stock: str, phase_name: str) -> dict:
    x=df[(df.symbol==stock)&(df.phase==phase_name)]; return metric(x.label,x[score_col].to_numpy(float))


def comparisons(p4: pd.DataFrame,p1: pd.DataFrame) -> None:
    goal=pd.read_csv(ROOT/"outputs/stock_goal60_4h/v1/predictions.csv"); news4=pd.read_csv(V9/"PREDICTIONS_4H.csv"); news1=pd.read_csv(V9/"PREDICTIONS_1D.csv"); dp=pd.read_csv(OUT/"DPRICE_PREDICTIONS.csv")
    rows4=[]
    for keys,g in p4.groupby(["method","stock","phase"]):
        method,stock,ph=keys; gm=goal[(goal.symbol==stock)&(goal.phase==ph)]; nm=news4[(news4.symbol==stock)&(news4.phase==ph)&(news4.method==method)]
        j=metric(g.label,g.probability); r=metric(gm.label,gm.R1); n=metric(nm.label,nm.p)
        rows4.append({"method":method,"stock":stock,"phase":ph,"R1_BA":r["BA"],"JOINT_BA":j["BA"],"DELTA_BA_VS_R1":j["BA"]-r["BA"],"R1_Brier":r["Brier"],"JOINT_Brier":j["Brier"],"DELTA_BRIER_VS_R1":j["Brier"]-r["Brier"],"NEWS_ONLY_BA":n["BA"],"DELTA_BA_VS_NEWS_ONLY":j["BA"]-n["BA"],"comparison_scope":"SYSTEM_LEVEL"})
    c4=pd.DataFrame(rows4); c4.to_csv(OUT/"INCREMENTAL_COMPARISON_4H.csv",index=False)
    rows1=[]
    for keys,g in p1.groupby(["horizon_window","method","stock","phase"]):
        horizon,method,stock,ph=keys; dm=dp[(dp.stock==stock)&(dp.phase==ph)]; nm=news1[(news1.symbol==stock)&(news1.phase==ph)&(news1.method==method)&(news1.horizon==horizon)]
        j=metric(g.label,g.probability); d=metric(dm.label,dm.p); n=metric(nm.label,nm.p)
        rows1.append({"horizon_window":horizon,"method":method,"stock":stock,"phase":ph,"DPRICE_BA":d["BA"],"JOINT_BA":j["BA"],"DELTA_BA_VS_DPRICE":j["BA"]-d["BA"],"DPRICE_Brier":d["Brier"],"JOINT_Brier":j["Brier"],"DELTA_BRIER_VS_DPRICE":j["Brier"]-d["Brier"],"NEWS_ONLY_BA":n["BA"],"DELTA_BA_VS_NEWS_ONLY":j["BA"]-n["BA"],"comparison_scope":"SYSTEM_LEVEL"})
    c1=pd.DataFrame(rows1); c1.to_csv(OUT/"INCREMENTAL_COMPARISON_1D.csv",index=False)
    consistency={"4h":{},"daily":{}}
    for method,g in c4.groupby("method"): consistency["4h"][method]={"ALL_FOUR_POSITIVE_DELTA_VS_R1":bool((g.DELTA_BA_VS_R1>0).all())}
    for keys,g in c1.groupby(["horizon_window","method"]): consistency["daily"]["|".join(keys)]={"ALL_FOUR_POSITIVE_DELTA_VS_DPRICE":bool((g.DELTA_BA_VS_DPRICE>0).all())}
    dump(OUT/"INCREMENTAL_CONSISTENCY_SUMMARY.json",consistency)
    f1={}
    for method,g in p4.groupby("method"):
        cells=[]
        for stock in ["AAPL","AMZN"]:
            for ph in ["development","later"]:
                j=metric(g[(g.stock==stock)&(g.phase==ph)].label,g[(g.stock==stock)&(g.phase==ph)].probability)["BA"]
                x=goal[(goal.symbol==stock)&(goal.phase==ph)]; base=metric(x.label,x.F1)["BA"]; cells.append({"stock":stock,"phase":ph,"joint_BA":j,"F1_BA":base,"delta":j-base})
        f1[method]={"cells":cells,"STRICTLY_DOMINATES_F1":bool(all(x["delta"]>=0 for x in cells) and any(x["delta"]>0 for x in cells))}
    dump(OUT/"F1_COMPARISON.json",f1)
    q=c1[c1.method=="PAPER_2G_L1LR"]; diffs=[]
    for stock in ["AAPL","AMZN"]:
        for ph in ["development","later"]:
            a=float(q[(q.stock==stock)&(q.phase==ph)&(q.horizon_window=="1d:DNEWS_24H")].JOINT_BA.iloc[0]); b=float(q[(q.stock==stock)&(q.phase==ph)&(q.horizon_window=="1d:DNEWS_OVERNIGHT")].JOINT_BA.iloc[0]); diffs.append({"stock":stock,"phase":ph,"24H_MINUS_OVERNIGHT_BA":a-b})
    dump(OUT/"DAILY_WINDOW_COMPARISON.json",{"method":"PAPER_2G_L1LR","cells":diffs,"24H_CONSISTENTLY_STRONGER_THAN_OVERNIGHT_FOR_PAPER_2G":bool(all(x["24H_MINUS_OVERNIGHT_BA"]>=0 for x in diffs) and any(x["24H_MINUS_OVERNIGHT_BA"]>0 for x in diffs))})


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--approve-b2-run",action="store_true"); a=ap.parse_args()
    if not a.approve_b2_run: raise SystemExit("requires --approve-b2-run")
    assert json.loads((OUT/"STAGE_A_FINAL_AUDIT_V2.json").read_text())["status"]=="PASS"
    assert json.loads((OUT/"STAGE_B1_FINAL_AUDIT_V2.json").read_text())["status"]=="PASS"
    cfg=json.loads((OUT/"STAGE_B2_FROZEN_CONFIGURATION.json").read_text()); assert cfg["status"]=="FROZEN_FOR_FUTURE_B2_ONLY" and len(cfg["branches"])==18
    dump(OUT/"STAGE_B2_PRE_RUN_HASHES.json",protected_hashes())
    four,daily,input_audit=load_inputs(); dump(OUT/"JOINT_INPUT_AUDIT.json",input_audit)
    PRIVATE.mkdir(parents=True,exist_ok=True); all_pred=[]; manifest=[]; no_news=[]
    for branch in cfg["branches"]:
        horizon=branch["horizon_window"]; method=branch["method"]; stock=branch["stock"]; params=branch["final_frozen_parameter"]; source=four if horizon=="4h" else daily; price_cols=R1 if horizon=="4h" else DPRICE
        g=source[(source.symbol==stock)&(source.horizon_window==horizon)].sort_values("cutoff_utc")
        branch_no=branch_nnz=0
        for month in MONTHS:
            ev=g[g.start_utc.dt.strftime("%Y-%m")==month].copy(); train=g[g.end_utc<ev.cutoff_utc.min()].copy(); p,bundle,xe=fit_joint(method,params,train,ev,price_cols)
            mask=ev.has_news.to_numpy(int)==0; branch_no+=int(mask.sum()); branch_nnz+=int(xe[mask].nnz)
            bundle.update({"stock":stock,"horizon_window":horizon,"prediction_month":month,"training_row_identifiers":train.target_key.tolist(),"train_n":len(train),"train_end":str(train.end_utc.max()),"prediction_n":len(ev)})
            logical=f"{horizon.replace(':','_')}_{stock}_{method}_{month}"; path=PRIVATE/f"{logical}.joblib"; joblib.dump(bundle,path)
            check=joblib.load(path); rp=predict_bundle(check,ev)
            manifest.append({"logical_model_id":logical,"sha256":sha(path),"stock":stock,"horizon_window":horizon,"method":method,"month":month,"frozen_parameter":params,"train_n":len(train),"train_end":str(train.end_utc.max()),"prediction_n":len(ev),"reload_max_error":float(np.max(np.abs(rp-p))),"reload_direction_mismatches":int(np.sum((rp>=.5)!=(p>=.5)))})
            z=ev[["symbol","day","target_key","start_utc","end_utc","cutoff_utc","label","phase","has_news"]].rename(columns={"symbol":"stock"}); z["horizon_window"]=horizon; z["method"]=method; z["month"]=month; z["probability"]=p; z["predicted_direction"]=(p>=.5).astype(int); z["frozen_parameter_json"]=pjson(params); z["train_n"]=len(train); z["train_end"]=str(train.end_utc.max()); all_pred.append(z)
        no_news.append({"stock":stock,"horizon_window":horizon,"method":method,"no_news_rows":branch_no,"text_nonzero_count_on_no_news_rows":branch_nnz,"price_feature_count":len(price_cols),"fallback_override_count":0})
    pred=pd.concat(all_pred,ignore_index=True); p4=pred[pred.horizon_window=="4h"].copy(); p1=pred[pred.horizon_window!="4h"].copy()
    p4.to_csv(OUT/"JOINT_PREDICTIONS_4H.csv",index=False); p1.to_csv(OUT/"JOINT_PREDICTIONS_1D.csv",index=False)
    m4,a4=phase_metrics(p4); m1,a1=phase_metrics(p1); m4.to_csv(OUT/"JOINT_MONTHLY_4H.csv",index=False); a4.to_csv(OUT/"JOINT_METRICS_4H.csv",index=False); m1.to_csv(OUT/"JOINT_MONTHLY_1D.csv",index=False); a1.to_csv(OUT/"JOINT_METRICS_1D.csv",index=False)
    dump(OUT/"JOINT_MODEL_MANIFEST.json",manifest); dump(OUT/"JOINT_NO_NEWS_AUDIT.json",{"branches":no_news,"fallback_override_count":0})
    comparisons(p4,p1); dump(OUT/"JOINT_RUN_EVIDENCE.json",{"joint_issued_model_fits":108,"joint_candidate_grid_fits":0,"prediction_rows":len(pred)})
    print("V10_STAGE_B2_RUN_COMPLETE_AWAITING_INDEPENDENT_VERIFICATION")
if __name__=="__main__": main()
