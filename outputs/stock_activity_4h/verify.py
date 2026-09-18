"""Activity v1 independent preflight and reserved post-run verifier."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from common import ACTIVITY_FEATURES, HERE, PRIVATE, read_bars, read_schedule, sha
from prepare import ensure, EVIDENCE, SOURCES
from run_activity import reproduce_a0

OUT=HERE/"preflight_verification.json"; V1=HERE/"v1"
def unfinished_test():
    from common import activity_level
    index=pd.to_datetime(["2018-01-02 09:55:00+00:00","2018-01-02 10:00:00+00:00"]); bars=pd.DataFrame({"activity":[1,999]},index=index)
    result=activity_level(bars,pd.Timestamp("2018-01-02 09:30:00+00:00"),pd.Timestamp("2018-01-02 16:00:00+00:00"),pd.Timestamp("2018-01-02 10:02:00+00:00"),5)
    return (not result["valid"]) and result["latest_bar_end"] is None
def main():
    data=ensure(); evidence=pd.read_csv(EVIDENCE); schedule_path,schedule=read_schedule(); a0,_,_=reproduce_a0(data)
    canonical=pd.read_csv(HERE.parent/"stock_goal60_4h/v1/predictions.csv"); canonical=canonical[canonical.R1.notna()][["key","R1"]]
    joined=a0.merge(canonical,on="key",validate="one_to_one"); err=float(np.abs(joined.A0-joined.R1).max()); direction=bool(np.array_equal(joined.A0.to_numpy()>=.5,joined.R1.to_numpy()>=.5))
    nextgen=pd.read_csv(HERE.parent/"stock_nextgen_4h/runs/v1/PREDICTIONS.csv")[["key","R1"]]; check=a0.merge(nextgen,on="key",validate="one_to_one"); err2=float(np.abs(check.A0-check.R1).max())
    latest_cols=[f"activity_current_latest_used_bar_end_{h}" for h in (15,60)]; ref_cols=[f"activity_reference_latest_session_{h}" for h in (15,60)]
    latest_ok=True; ref_ok=True
    for h in (15,60):
        valid=evidence[evidence[f"activity_missing_{h}"].eq(0)]; ends=pd.to_datetime(valid[f"activity_current_latest_used_bar_end_{h}"],utc=True); cuts=pd.to_datetime(valid.cutoff_utc,utc=True); latest_ok &= bool((ends<=cuts).all())
        rel=evidence[evidence[f"activity_rel_missing_{h}"].eq(0)]; refs=pd.to_datetime(rel[f"activity_reference_latest_session_{h}"],utc=True); opens=pd.to_datetime(rel.session_open,utc=True); ref_ok &= bool((refs<opens).all() and (rel[f"activity_reference_session_count_{h}"]<=20).all() and (rel[f"activity_reference_session_count_{h}"]>=10).all())
    no_candidate=not any((V1/name).exists() for name in ["predictions.csv","metrics.csv","monthly_metrics.csv","advancement.json"])
    checks={"canonical_keys_1607":len(data)==1607 and data.key.nunique()==1607,"labels_present":set(data.label.unique())=={0,1},"activity_feature_names_exact":set(ACTIVITY_FEATURES).issubset(data.columns) and len(ACTIVITY_FEATURES)==8,"activity_numeric":all(pd.api.types.is_numeric_dtype(data[c]) for c in ACTIVITY_FEATURES),"source_hashes_present":SOURCES.exists(),"no_duplicate_raw_bars":all(not read_bars(s)[1].index.duplicated().any() for s in ("AAPL","AMZN")),"completed_bar_cutoff_safe":latest_ok,"history_reference_safe":ref_ok,"synthetic_1002_excludes_unfinished_bar":unfinished_test(),"a0_goal60_probability_parity":err<=1e-12,"a0_nextgen_probability_parity":err2<=1e-12,"a0_direction_parity_exact":direction,"no_a1_results_exist":no_candidate}
    coverage={}
    for symbol in ("AAPL","AMZN"):
        rows=data.loc[data.symbol.eq(symbol)]
        coverage[symbol]={}
        for h in (15,60):
            coverage[symbol][f"activity_valid_fraction_{h}"]=float(1-rows[f"activity_missing_{h}"].mean())
            coverage[symbol][f"activity_relative_valid_fraction_{h}"]=float(1-rows[f"activity_rel_missing_{h}"].mean())
    result={"status":"PASS" if all(checks.values()) else "FAIL","mode":"PRE_FLIGHT_NO_A1_SCORING","checks":checks,"a0_goal60_max_abs_probability_error":err,"a0_nextgen_max_abs_probability_error":err2,"a0_rows":len(a0),"activity_coverage":coverage,"source_hashes":{"schedule":sha(schedule_path),"aapl":sha(read_bars("AAPL")[0]),"amzn":sha(read_bars("AMZN")[0] )}}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2));
    if result["status"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
