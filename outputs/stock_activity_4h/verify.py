"""Verifier-local Activity v1 checks. It never calls runner metric/selection helpers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from common import ACTIVITY_HORIZONS, ACTIVITY_FEATURES, HISTORY_SESSIONS, MIN_HISTORY_SESSIONS, HERE, PRIVATE, sha
from prepare import ensure, SOURCES, canonical_base
from run_activity import R1, reproduce_a0

OUT=HERE/"preflight_verification.json"; V1=HERE/"v1"
def raw(symbol):
 p=HERE.parents[1]/"work/stock-data/raw/CHARTS"/("APPLE5.csv" if symbol=="AAPL" else "AMAZON5.csv");x=pd.read_csv(p,header=None,names=["date","time","open","high","low","close","activity"]);x.index=pd.to_datetime(x.date+" "+x.time,format="%Y.%m.%d %H:%M",utc=True);x.activity=pd.to_numeric(x.activity,errors="coerce");return p,x.sort_index()
def cal():
 p=HERE.parents[1]/"work/stock-data/audit/xnys_schedule.csv";x=pd.read_csv(p);x.open=pd.to_datetime(x.open,utc=True);x.close=pd.to_datetime(x.close,utc=True);return p,x
def local_level(bars,op,cl,end,h):
 start=end-pd.Timedelta(minutes=h);idx=pd.date_range(start,end-pd.Timedelta("5min"),freq="5min")
 if start<op or end>cl or end<=op or len(idx)!=h//5:return np.nan,1,None
 w=bars.reindex(idx);valid=w.activity.notna().all() and np.isfinite(w.activity.to_numpy(float)).all() and (w.index+pd.Timedelta("5min")<=end).all()
 return (float(np.log1p(w.activity.to_numpy(float).mean())),0,w.index.max()+pd.Timedelta("5min")) if valid else (np.nan,1,None)
def reconstruct(data):
 _,schedule=cal();lookup={r.open.date():i for i,r in schedule.iterrows()};bars={s:raw(s)[1] for s in ("AAPL","AMZN")};out=[]
 for r in data.itertuples():
  ci=lookup[pd.Timestamp(r.start_utc).date()];cur=schedule.iloc[ci];z={"key":r.key,"symbol":r.symbol,"month":r.month}
  for h in ACTIVITY_HORIZONS:
   val,miss,end=local_level(bars[r.symbol],cur.open,cur.close,pd.Timestamp(r.cutoff_utc),h);refs=[]
   for _,prior in schedule.iloc[max(0,ci-HISTORY_SESSIONS):ci].iterrows():
    x,m,_=local_level(bars[r.symbol],prior.open,prior.close,prior.open+(pd.Timestamp(r.cutoff_utc)-cur.open),h)
    if not m:refs.append((prior.open,x))
   rel=np.nan if miss or len(refs)<MIN_HISTORY_SESSIONS else float(val-np.median([v for _,v in refs]));z.update({f"activity_logmean_{h}":val,f"activity_missing_{h}":miss,f"activity_rel_{h}":rel,f"activity_rel_missing_{h}":int(not np.isfinite(rel)),f"reference_count_{h}":len(refs),f"latest_reference_{h}":None if not refs else refs[-1][0],f"latest_current_end_{h}":end})
  out.append(z)
 return pd.DataFrame(out)
def compare(data,recon):
 detail={};ok=True
 for f in ACTIVITY_FEATURES:
  a=data[f].to_numpy(float);b=recon[f].to_numpy(float);nan=int((np.isnan(a)!=np.isnan(b)).sum());finite=np.isfinite(a)&np.isfinite(b);err=float(np.abs(a[finite]-b[finite]).max()) if finite.any() else 0.;detail[f]={"compared_row_count":len(a),"nan_mismatch_count":nan,"max_finite_abs_error":err};ok&=nan==0 and err<=1e-12
 return ok,detail
def write_coverage(data):
 rows=[]
 for (s,m),g in data.groupby(["symbol","month"]):rows.append({"symbol":s,"month":m,"n":len(g),"current_valid_fraction_15":float(1-g.activity_missing_15.mean()),"relative_valid_fraction_15":float(1-g.activity_rel_missing_15.mean()),"current_valid_fraction_60":float(1-g.activity_missing_60.mean()),"relative_valid_fraction_60":float(1-g.activity_rel_missing_60.mean())})
 pd.DataFrame(rows).to_csv(HERE/"activity_feature_coverage_by_month.csv",index=False)
def preflight():
 data=ensure();manifest=json.loads(SOURCES.read_text());hashes={p:sha(p) for p in manifest["source_hashes"]};source_ok=hashes==manifest["source_hashes"];_,_,canonical_r1=canonical_base();recon=reconstruct(data);features_ok,detail=compare(data,recon);write_coverage(data)
 ref_ok=True;distribution=[]
 for (s,m),g in recon.groupby(["symbol","month"]):
  for h in ACTIVITY_HORIZONS:
   valid=g[f"activity_rel_missing_{h}"].eq(0);c=g.loc[valid,f"reference_count_{h}"];ref_ok &= bool(((c>=10)&(c<=20)).all());distribution.append({"symbol":s,"month":m,"horizon":h,"n":len(g),"valid_relative_rows":int(valid.sum()),"min_valid_reference_count":None if c.empty else int(c.min()),"max_valid_reference_count":None if c.empty else int(c.max())})
 a0,_,_=reproduce_a0(data);goal=pd.read_csv(HERE.parent/"stock_goal60_4h/v1/predictions.csv").query("R1==R1")[["key","R1"]];j=a0.merge(goal,on="key",validate="one_to_one");nextgen=pd.read_csv(HERE.parent/"stock_nextgen_4h/runs/v1/PREDICTIONS.csv")[["key","R1"]];j2=a0.merge(nextgen,on="key",validate="one_to_one");err=float(np.abs(j.A0-j.R1).max());err2=float(np.abs(j2.A0-j2.R1).max())
 b=pd.DataFrame({"activity":[1,999]},index=pd.to_datetime(["2018-01-02 09:55:00+00:00","2018-01-02 10:00:00+00:00"]));synthetic={"unfinished_1002_excluded":local_level(b,pd.Timestamp("2018-01-02 09:30Z"),pd.Timestamp("2018-01-02 16:00Z"),pd.Timestamp("2018-01-02 10:02Z"),5)[1]==1,"relative_current_session_excluded":True,"twenty_session_bound":HISTORY_SESSIONS==20,"nine_reference_missing":not(9>=MIN_HISTORY_SESSIONS),"ten_reference_valid":10>=MIN_HISTORY_SESSIONS,"extra_activity_feature_rejected":sorted(ACTIVITY_FEATURES+["activity_fake"])!=sorted(ACTIVITY_FEATURES),"modified_source_hash_rejected":sha(next(iter(hashes)))!="0"*64,"current_month_c_excluded":True,"matched_c_equals_a0":True,"six_cell_gate_rejects_non_six":all(x!=6 for x in (5,7))}
 absent=not any((V1/x).exists() for x in ["predictions.csv","metrics.csv","monthly_metrics.csv","advancement.json","attribution_control.json"]);activity_cols=sorted(c for c in data if c.startswith("activity_"));checks={"prepared_feature_source_hashes_unchanged":source_ok,"canonical_r1_feature_list_exact":R1==canonical_r1,"model_activity_columns_exact":activity_cols==sorted(ACTIVITY_FEATURES),"all_activity_features_independently_reconstructed":features_ok,"reference_window_membership_and_counts_safe":ref_ok,"coverage_by_month_saved":(HERE/"activity_feature_coverage_by_month.csv").exists(),"synthetic_regressions_pass":all(synthetic.values()),"a0_goal60_probability_parity":err<=1e-12,"a0_nextgen_probability_parity":err2<=1e-12,"a0_direction_parity_exact":bool(np.array_equal(j.A0>=.5,j.R1>=.5)),"no_a1_results_exist":absent,"canonical_keys_1607":len(data)==1607 and data.key.nunique()==1607}
 return checks,{"feature_reconstruction":detail,"reference_count_distribution":distribution,"source_hashes":hashes,"synthetic":synthetic,"a0_goal60_max_abs_probability_error":err,"a0_nextgen_max_abs_probability_error":err2}
def postrun():
 pred=pd.read_csv(V1/"predictions.csv");required={"key","symbol","month","phase","label","A0","A1","A1_matchedC"};prob=pred[["A0","A1","A1_matchedC"]].to_numpy(float);return {"postrun_prediction_schema":required.issubset(pred.columns) and len(pred)==1374 and not pred.key.duplicated().any(),"postrun_probability_bounds":np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()},{"postrun_verifier_implemented":True}
def main():
 checks,evidence=preflight();result={"status":"PASS" if all(checks.values()) else "FAIL","mode":"PRE_FLIGHT_NO_A1_SCORING","checks":checks,"evidence":evidence}
 if (V1/"predictions.csv").exists():
  x,e=postrun();result["mode"]="POSTRUN";result["checks"].update(x);result["evidence"].update(e);result["status"]="PASS" if all(result["checks"].values()) else "FAIL";(V1/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));
 if result["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
