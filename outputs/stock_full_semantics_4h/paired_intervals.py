"""Descriptive day/block-paired sensitivity, averaging seed-specific deltas."""
import json
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent/'v1'
def ba(y,q,idx):
 yy=y[idx];qq=q[:,idx];up=yy==1;down=~up
 if not up.any() or not down.any():return np.nan
 return .5*(qq[:,up].mean(axis=1)+(~qq[:,down]).mean(axis=1))
def interval(frame,method,stock):
 seeds=sorted(frame.seed.unique());g=frame[frame.seed==seeds[0]].sort_values('row_id');keys=g.row_id.tolist();days=g.day.to_numpy();y=g.label.to_numpy();p=[];b=[]
 for seed in seeds:
  v=frame[frame.seed==seed].set_index('row_id').loc[keys];assert v.label.tolist()==y.tolist();p.append(v[method].to_numpy()>=.5);b.append(v.FULL.to_numpy()>=.5)
 p=np.array(p);b=np.array(b);unique=sorted(set(days));members={day:np.flatnonzero(days==day) for day in unique};rng=np.random.default_rng(573);result=[]
 for length in [1,5]:
  samples=[]
  for _ in range(1000):
   starts=rng.integers(0,len(unique),size=int(np.ceil(len(unique)/length)));sample=[unique[(start+j)%len(unique)] for start in starts for j in range(length)][:len(unique)];ix=np.concatenate([members[t] for t in sample]);samples.append(float(np.mean(ba(y,p,ix)-ba(y,b,ix))))
  result.append(dict(symbol=stock,method=method,block_days=length,replicates=1000,delta_BA=float(np.mean(ba(y,p,np.arange(len(y)))-ba(y,b,np.arange(len(y))))),lo=float(np.nanquantile(samples,.025)),hi=float(np.nanquantile(samples,.975))))
 return result
def main():
 rows=[];assert json.loads((HERE/'CLASSICAL_VERIFICATION.json').read_text())['status']=='PASS';d=pd.read_csv(HERE/'CLASSICAL_PREDICTIONS.csv')
 for stock,g in d.groupby('symbol'):
  for m in ['TFIDF_LR','TFIDF_SVM','TFIDF_RF']:rows.extend(interval(g,m,stock))
 if (HERE/'SEMANTIC_VERIFICATION.json').exists():
  assert json.loads((HERE/'SEMANTIC_VERIFICATION.json').read_text())['status']=='PASS';s=pd.read_csv(HERE/'SEMANTIC_PREDICTIONS.csv')
  for (encoder,stock),g in s.groupby(['encoder','symbol']):
   for r in interval(g,'p',stock):r['method']='FULL_'+encoder;rows.append(r)
 pd.DataFrame(rows).to_csv(HERE/'PAIRED_INTERVALS.csv',index=False)
 (HERE/'INTERVAL_NOTE.md').write_text('Intervals resample the same trading-day clusters for each method and its FULL comparator, with one-day and circular five-day blocks. Each replicate averages three seed-specific BA differences; seeds are not counted as independent extra market samples. These are descriptive sensitivity intervals after repeated exploratory analysis, not confirmatory significance tests or evidence of future-market generalization.\n')
if __name__=='__main__':main()
