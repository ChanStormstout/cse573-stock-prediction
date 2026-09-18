"""Label-free raw activity audit for Activity v1."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common import HERE, read_bars, read_schedule, sha

def stats(symbol, bars, source):
    x=bars.activity.to_numpy(float); absret=np.abs(np.log(bars.close.to_numpy(float)/bars.open.to_numpy(float))); rng=(bars.high.to_numpy(float)-bars.low.to_numpy(float))/bars.open.to_numpy(float)
    q=np.quantile(x,[.01,.05,.25,.75,.95,.99])
    return {"symbol":symbol,"raw_rows":len(bars),"source_sha256":sha(source),"dtype":str(bars.activity.dtype),"finite_count":int(np.isfinite(x).sum()),"missing_count":int(np.isnan(x).sum()),"negative_count":int((x<0).sum()),"zero_count":int((x==0).sum()),"min":float(np.nanmin(x)),"max":float(np.nanmax(x)),"mean":float(np.nanmean(x)),"median":float(np.nanmedian(x)),"q01":float(q[0]),"q05":float(q[1]),"q25":float(q[2]),"q75":float(q[3]),"q95":float(q[4]),"q99":float(q[5]),"unique_count":int(pd.Series(x).nunique()),"corr_abs_return":float(np.corrcoef(x,absret)[0,1]),"corr_high_low_range":float(np.corrcoef(x,rng)[0,1])}

def main():
    schedule_path,schedule=read_schedule(); rows=[]; profiles=[]; gran=[]
    for symbol in ("AAPL","AMZN"):
        source,bars=read_bars(symbol); rows.append(stats(symbol,bars,source))
        for session in schedule.itertuples():
            regular=bars[(bars.index>=session.open)&(bars.index+pd.Timedelta("5min")<=session.close)].copy()
            regular["slot"]=(regular.index-session.open).total_seconds()/60
            profiles.append(regular.assign(symbol=symbol)[["symbol","slot","activity"]])
        # Same-source 15/30/60 files exist; audit only descriptive aggregation agreement.
        for minutes in (15,30,60):
            path=source.with_name(("APPLE" if symbol=="AAPL" else "AMAZON")+f"{minutes}.csv")
            other=pd.read_csv(path,header=None,names=["date","time","open","high","low","close","activity"]); other.index=pd.to_datetime(other.date+" "+other.time,format="%Y.%m.%d %H:%M",utc=True)
            checks=[]
            for stamp,row in other.iterrows():
                w=bars.reindex(pd.date_range(stamp,stamp+pd.Timedelta(minutes=minutes-5),freq="5min"))
                if len(w)==minutes//5 and w.activity.notna().all(): checks.append((float(row.activity),float(w.activity.mean()),float(w.activity.sum())))
            a=np.asarray(checks)
            diff=np.abs(a[:,0]-a[:,2]) if len(a) else np.array([])
            gran.append({"symbol":symbol,"minutes":minutes,"complete_windows":int(len(a)),"exact_sum_match_count":int((diff==0).sum()),"exact_sum_match_fraction":float((diff==0).mean()) if len(diff) else None,"max_abs_sum_discrepancy":float(diff.max()) if len(diff) else None,"mean_abs_sum_discrepancy":float(diff.mean()) if len(diff) else None,"mean_abs_error_vs_5m_mean":float(np.abs(a[:,0]-a[:,1]).mean()) if len(a) else None,"status":"DESCRIPTIVE_ONLY_NOT_SEMANTIC_PROOF"})
    summary=pd.DataFrame(rows); summary.to_csv(HERE/"activity_summary.csv",index=False)
    profile=pd.concat(profiles).groupby(["symbol","slot"]).activity.agg(median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),count="size").reset_index(); profile.to_csv(HERE/"activity_intraday_profile.csv",index=False)
    result={"status":"SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY","schedule_sha256":sha(schedule_path),"sources":summary.to_dict("records"),"intraday_slots":int(len(profile)),"cross_granularity":gran,"authoritative_definition_found":False,"definition_search":"Local course metadata names the seventh field activity but does not define vendor semantics."}
    (HERE/"activity_data_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Activity v1 label-free audit","","Status: **SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY**. No authoritative vendor definition was found locally. Descriptive correlations do not establish field meaning or predictive value.","","## Raw field summary","",summary.to_markdown(index=False),"","## Cross-granularity check","",pd.DataFrame(gran).to_markdown(index=False),"","The raw 15/30/60 files exist, but their aggregation comparison is descriptive only and does not identify the opaque field as volume or any other semantic quantity."]
    (HERE/"ACTIVITY_AUDIT.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
