from __future__ import annotations
import numpy as np
import pandas as pd

MARKET_NUMERIC=["spy_intrabar_return_60tm","spy_rv_60tm","qqq_intrabar_return_60tm","qqq_rv_60tm"]
MARKET_CONTROLS=["market_age_log1p_hours","market_prior_session_fraction","market_window_missing"]
MARKET_FIELDS=MARKET_NUMERIC+MARKET_CONTROLS

def build_market_row(cutoff,current_session_id,expected_grid,etf_bars):
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None: raise ValueError("cutoff must be UTC-aware")
    allowed=expected_grid.loc[expected_grid.bar_end_utc+pd.Timedelta(minutes=1)<=cutoff]
    window=allowed.tail(12)
    if len(window)!=12: raise ValueError("Insufficient calendar warmup; extend intake, not prices")
    starts=pd.DatetimeIndex(window.bar_start_utc)
    out={"market_age_log1p_hours":float(np.log1p((cutoff-window.bar_end_utc.iloc[-1]).total_seconds()/3600)),"market_prior_session_fraction":float((window.session_id!=current_session_id).mean()),"market_window_missing":0,"used_starts_utc":[x.isoformat() for x in starts]}
    values={}
    for symbol in ("SPY","QQQ"):
        selected=etf_bars[symbol].reindex(starts);o=selected.open.to_numpy(float);c=selected.close.to_numpy(float)
        if not(np.isfinite(o).all() and np.isfinite(c).all() and (o>0).all() and (c>0).all()): out["market_window_missing"]=1;break
        r=np.log(c/o);values[f"{symbol.lower()}_intrabar_return_60tm"]=float(r.sum());values[f"{symbol.lower()}_rv_60tm"]=float(np.sqrt(np.square(r).sum()))
    out.update({x:np.nan for x in MARKET_NUMERIC} if out["market_window_missing"] else values)
    return out

def expected_regular_grid(schedule):
    rows=[]
    for r in schedule.itertuples():
        starts=pd.date_range(r.open,r.close-pd.Timedelta(minutes=5),freq="5min")
        rows += [{"bar_start_utc":x,"bar_end_utc":x+pd.Timedelta(minutes=5),"session_id":str(r.open.date())} for x in starts]
    return pd.DataFrame(rows)
