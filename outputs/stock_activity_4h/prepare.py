"""Private cutoff-safe Activity v1 feature generation; no candidate scoring."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import ACTIVITY_HORIZONS, ACTIVITY_FEATURES, HERE, PRIVATE, ROOT, dump, read_bars, read_schedule, relative_activity, sha, activity_level

FEATURES = PRIVATE / "features.pkl"; EVIDENCE = PRIVATE / "feature_evidence.csv"; SOURCES = PRIVATE / "sources.json"

def canonical_base():
    """Call the reviewed R1 builder without copying or changing its feature logic."""
    src = ROOT / "outputs/stock_nextgen_4h/recent_price.py"; parent = src.parent
    saved_common = sys.modules.pop("common", None); sys.path.insert(0, str(parent))
    try:
        spec = importlib.util.spec_from_file_location("activity_canonical_recent_price", src)
        module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
        data, _audit, sources = module.prepare_features()
        return data, sources, module.R1
    finally:
        sys.path.remove(str(parent)); sys.modules.pop("common", None)
        if saved_common is not None: sys.modules["common"] = saved_common

def build():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    data, base_sources, r1 = canonical_base()
    schedule_path, schedule = read_schedule(); sessions = {row.open.date(): i for i,row in schedule.iterrows()}
    evidence=[]; source_paths=list(base_sources)+[schedule_path]
    for symbol, ids in data.groupby("symbol").groups.items():
        bar_path, bars = read_bars(symbol); source_paths.append(bar_path)
        for idx in ids:
            row=data.loc[idx]; cutoff=pd.Timestamp(row.cutoff_utc); current_index=sessions.get(pd.Timestamp(row.start_utc).date())
            if current_index is None: raise AssertionError("missing current XNYS session")
            session=schedule.iloc[current_index]; record={"key":row.key,"symbol":symbol,"cutoff_utc":cutoff.isoformat(),"session_open":session.open.isoformat(),"minutes_from_open":float((cutoff-session.open).total_seconds()/60)}
            for h in ACTIVITY_HORIZONS:
                cur=activity_level(bars,session.open,session.close,cutoff,h)
                rel=relative_activity(bars,schedule,current_index,cutoff,h,cur["value"])
                data.loc[idx,f"activity_logmean_{h}"]=cur["value"]; data.loc[idx,f"activity_missing_{h}"]=int(not cur["valid"])
                data.loc[idx,f"activity_rel_{h}"]=rel["value"]; data.loc[idx,f"activity_rel_missing_{h}"]=int(not rel["valid"])
                record.update({f"activity_logmean_{h}":cur["value"],f"activity_rel_{h}":rel["value"],f"activity_missing_{h}":int(not cur["valid"]),f"activity_rel_missing_{h}":int(not rel["valid"]),f"activity_current_latest_used_bar_end_{h}":None if cur["latest_bar_end"] is None else cur["latest_bar_end"].isoformat(),f"activity_reference_session_count_{h}":rel["reference_count"],f"activity_reference_latest_session_{h}":None if rel["latest_reference_session"] is None else rel["latest_reference_session"].isoformat()})
                if cur["valid"] and cur["latest_bar_end"] > cutoff: raise AssertionError("unfinished current bar used")
                if rel["valid"] and not rel["latest_reference_session"] < session.open: raise AssertionError("future reference used")
            evidence.append(record)
    if set(ACTIVITY_FEATURES) != {c for c in data.columns if c in ACTIVITY_FEATURES}: raise AssertionError("activity feature contract changed")
    data.to_pickle(FEATURES); pd.DataFrame(evidence).to_csv(EVIDENCE,index=False)
    frozen_code=[ROOT/"outputs/stock_nextgen_4h/recent_price.py",ROOT/"outputs/stock_nextgen_4h/common.py",Path(__file__),HERE/"common.py"]
    dump(SOURCES,{"source_hashes":{str(p):sha(p) for p in dict.fromkeys(source_paths+frozen_code)},"r1_columns":r1,"rows":len(data),"feature_names":ACTIVITY_FEATURES,"semantics":"SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY"})
    return data

def ensure(): return pd.read_pickle(FEATURES) if FEATURES.exists() else build()
if __name__ == "__main__":
    x=build(); print({"status":"PREPARED_NO_A1_SCORING","rows":len(x),"private":str(PRIVATE)})
