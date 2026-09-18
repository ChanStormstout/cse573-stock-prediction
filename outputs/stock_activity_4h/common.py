"""Frozen contracts for the isolated Activity v1 experiment."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work" / "stock-data"
PRIVATE = WORK / "activity_4h"
ACTIVITY_HORIZONS = (15, 60)
ACTIVITY_FEATURES = ["activity_logmean_15", "activity_logmean_60", "activity_rel_15", "activity_rel_60", "activity_missing_15", "activity_missing_60", "activity_rel_missing_15", "activity_rel_missing_60"]
HISTORY_SESSIONS, MIN_HISTORY_SESSIONS = 20, 10
CS = (0.01, 0.1, 1.0)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b"") : h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")

def read_bars(symbol):
    path = WORK / "raw/CHARTS" / ("APPLE5.csv" if symbol == "AAPL" else "AMAZON5.csv")
    x = pd.read_csv(path, header=None, names=["date","time","open","high","low","close","activity"])
    x.index = pd.to_datetime(x.date + " " + x.time, format="%Y.%m.%d %H:%M", utc=True)
    if x.index.duplicated().any(): raise AssertionError(f"duplicate raw bars: {symbol}")
    x["activity"] = pd.to_numeric(x.activity, errors="coerce")
    return path, x.sort_index()

def read_schedule():
    path = WORK / "audit/xnys_schedule.csv"
    s = pd.read_csv(path)
    s["open"] = pd.to_datetime(s.open, utc=True); s["close"] = pd.to_datetime(s.close, utc=True)
    return path, s

def activity_level(bars, session_open, session_close, end, minutes):
    n = minutes // 5; start = end - pd.Timedelta(minutes=minutes)
    if end <= session_open or start < session_open or end > session_close: return {"valid":False,"value":np.nan,"latest_bar_end":None}
    idx = pd.date_range(start, end - pd.Timedelta("5min"), freq="5min")
    if len(idx) != n: raise AssertionError("unexpected activity index length")
    window = bars.reindex(idx)
    valid = (not window.activity.isna().any() and np.isfinite(window.activity.to_numpy(float)).all() and (window.index + pd.Timedelta("5min") <= end).all())
    if not valid: return {"valid":False,"value":np.nan,"latest_bar_end":None}
    return {"valid":True,"value":float(np.log1p(window.activity.to_numpy(float).mean())),"latest_bar_end":window.index.max()+pd.Timedelta("5min")}

def relative_activity(bars, schedule, current_index, cutoff, minutes, current_value):
    if not np.isfinite(current_value): return {"valid":False,"value":np.nan,"reference_count":0,"latest_reference_session":None}
    cur = schedule.iloc[current_index]; offset = cutoff-cur.open; refs=[]
    for _, prior in schedule.iloc[max(0,current_index-HISTORY_SESSIONS):current_index].iterrows():
        result = activity_level(bars, prior.open, prior.close, prior.open+offset, minutes)
        if result["valid"]: refs.append((prior.open,result["value"]))
    latest = refs[-1][0] if refs else None
    if len(refs) < MIN_HISTORY_SESSIONS: return {"valid":False,"value":np.nan,"reference_count":len(refs),"latest_reference_session":latest}
    if not latest < cur.open: raise AssertionError("future reference session")
    return {"valid":True,"value":float(current_value-np.median([v for _,v in refs])),"reference_count":len(refs),"latest_reference_session":latest}

def metric(y,p):
    from sklearn.metrics import balanced_accuracy_score,brier_score_loss,matthews_corrcoef,roc_auc_score
    y=np.asarray(y,int); p=np.asarray(p,float); q=p>=.5
    return {"n":int(len(y)),"BA":float(balanced_accuracy_score(y,q)),"MCC":float(matthews_corrcoef(y,q)),"Brier":float(brier_score_loss(y,p)),"AUC":float(roc_auc_score(y,p)),"up_recall":float(q[y==1].mean()),"down_recall":float((~q[y==0]).mean()),"pred_up":float(q.mean()),"constant":bool(q.min()==q.max())}
