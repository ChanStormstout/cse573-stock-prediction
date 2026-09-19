"""Frozen V8 joint follow-up using only OOF-selected global text methods."""
from pathlib import Path
import sys,json,pandas as pd
ROOT=Path(__file__).resolve().parents[2]; O=ROOT/'outputs/stock_priorwork_repro/v8';sys.path.insert(0,str(ROOT/'outputs/stock_priorwork_repro'));import run as core
def main():
 d=pd.read_pickle(O/'private_daily.pkl');m=pd.read_csv(O/'METRICS_1D.csv');sel=json.loads((O/'SELECTIONS_1D.json').read_text());rows=[];pred=[]
 for w in ['DNEWS_OVERNIGHT','DNEWS_24H']:
  h='1d:'+w;chosen=core.global_selected(m,h)
  for n in chosen:
   for s,g in d[d.news_window.eq(w)].groupby('symbol'):
    g=g.sort_values('cutoff_utc');tr=g[g.start_utc<pd.Timestamp('2018-09-01',tz='UTC')]; prm=sel[f'{h}:{s}:{n}']
    for ph in ['development','later']:
     ev=g[g.split.eq(ph)];p,dec,_=core.fit_pred(n,prm,tr,ev,True);rows.append(dict(news_window=w,stock=s,method='NEWS_PLUS_PRICE_'+n,phase=ph,**core.mscore(ev.label,p,dec)))
     z=ev[['symbol','day','start_utc','label']].copy();z['news_window']=w;z['method']='NEWS_PLUS_PRICE_'+n;z['phase']=ph;z['p']=p;pred.append(z)
 pd.DataFrame(rows).to_csv(O/'JOINT_METRICS_1D.csv',index=False);pd.concat(pred).to_csv(O/'JOINT_PREDICTIONS_1D.csv',index=False)
if __name__=='__main__':main()
