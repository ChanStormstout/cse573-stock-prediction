"""V9 full-grid chronological runner; reuses frozen representations only."""
from pathlib import Path
import sys,json,joblib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/stock-data';B=ROOT/'outputs/stock_priorwork_repro';O=B/'v9';PRIVATE=W/'priorwork_v9/models'
sys.path.insert(0,str(B));import run as core
METHODS=core.METHODS
def phase(t):return 'warmup' if t<pd.Timestamp('2018-03-01',tz='UTC') else 'oof' if t<pd.Timestamp('2018-09-01',tz='UTC') else 'development' if t<pd.Timestamp('2018-11-01',tz='UTC') else 'later'
def default(n):return core.choose([],n)
def rank(hist,n):
 x=pd.DataFrame(hist);g=x.groupby('params',as_index=False).agg(ba=('ba','mean'),brier=('brier','mean'))
 return json.loads(g.sort_values(['ba','brier','params'],ascending=[False,True,True]).iloc[0].params) if len(g) else default(n)
def full(d,h):
 issued=[];grid=[];pred=[];manifest=[]
 for n in METHODS:
  for (s,w),g in d.groupby(['symbol','news_window']):
   g=g.sort_values('cutoff_utc');hist=[];final=None
   for mo in pd.period_range('2018-03','2019-02',freq='M').astype(str):
    ev=g[g.start_utc.dt.strftime('%Y-%m').eq(mo)];tr=g[g.end_utc<ev.cutoff_utc.min()]
    if ev.empty or tr.label.nunique()<2:continue
    if mo<'2018-09': prm=default(n) if mo=='2018-03' else rank(hist,n)
    else:
     if final is None:final=rank(hist,n)
     prm=final
    p,dec,meta=core.fit_pred(n,prm,tr,ev,False);r=dict(stock=s,horizon=h if h=='4h' else h+':'+w,method=n,month=mo,phase=phase(ev.start_utc.iloc[0]),params=json.dumps(prm,sort_keys=True),issued=True,**core.mscore(ev.label,p,dec));issued.append(r)
    z=ev[['symbol','day','start_utc','cutoff_utc','label','has_news','news_count']].copy();z['horizon']=r['horizon'];z['method']=n;z['month']=mo;z['phase']=r['phase'];z['p']=p;pred.append(z)
    # After issue, score every frozen candidate; only stored for later selection.
    for cand in core.grid(core.spec(n)[2]):
     cp,cd,cm=core.fit_pred(n,cand,tr,ev,False);grid.append(dict(stock=s,horizon=r['horizon'],method=n,month=mo,params=json.dumps(cand,sort_keys=True),train_n=len(tr),train_end=str(tr.end_utc.max()),evaluation_cutoff=str(ev.cutoff_utc.min()),**core.mscore(ev.label,cp,cd)))
    if mo<'2018-09':hist += [x for x in grid if x['stock']==s and x['horizon']==r['horizon'] and x['method']==n and x['month']==mo]
    PRIVATE.mkdir(parents=True,exist_ok=True);f=PRIVATE/f'{h}_{w}_{s}_{n}_{mo}.json';f.write_text(json.dumps({'params':prm,'train_n':len(tr),'train_end':str(tr.end_utc.max())}));manifest.append({'id':str(f.relative_to(ROOT)),'month':mo,'stock':s,'method':n,'params':prm})
 return pd.DataFrame(issued),pd.DataFrame(grid),pd.concat(pred),manifest
def main():
 if O.exists():raise SystemExit('immutable')
 O.mkdir();(O/'PREREGISTRATION_SNAPSHOT.md').write_text((B/'V9_FULL_GRID_ADDENDUM.md').read_text())
 four=pd.read_pickle(W/'nextgen_4h/price_v1/features.pkl');four['news_window']='4H';four['split']=four.start_utc.map(phase)
 daily=pd.read_pickle(B/'v8/private_daily.pkl');daily['split']=daily.start_utc.map(phase)
 a,ga,pa,ma=full(four,'4h');b,gb,pb,mb=full(daily,'1d')
 a.to_csv(O/'METRICS_4H.csv',index=False);ga.to_csv(O/'GRID_OOF_4H.csv',index=False);pa.to_csv(O/'PREDICTIONS_4H.csv',index=False);b.to_csv(O/'METRICS_1D.csv',index=False);gb.to_csv(O/'GRID_OOF_1D.csv',index=False);pb.to_csv(O/'PREDICTIONS_1D.csv',index=False);(O/'MODEL_MANIFEST.json').write_text(json.dumps(ma+mb,indent=2))
if __name__=='__main__':main()
