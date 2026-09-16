"""Build same-row, as-of temporal snapshots; historical outcomes never copied."""
import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
B=Path(__file__).resolve().parent;OUT=B/'results';base=json.loads((B.parent/'stock_baseline/results/results.json').read_text())
extra=['finbert_positive','finbert_negative','finbert_neutral','sentiment_std','log_news_count','has_news'];manifest={'stocks':{},'source_hashes':{}}
for s in ['AAPL','AMZN']:
 source=B.parent/f'stock_finbert/results/{s}/development_features.csv';d=pd.read_csv(source).fillna({'news_record_keys':''});assert d.split.isin(['train','validation']).all();assert pd.to_datetime(d.cutoff_utc,utc=True).is_monotonic_increasing
 features=base['stocks'][s]['features']+extra;x=d[features].to_numpy(dtype=np.float64);cut=pd.to_datetime(d.cutoff_utc,utc=True).astype('int64').to_numpy();ends=pd.to_datetime(d.end_utc,utc=True).astype('int64').to_numpy()
 tensor=np.zeros((len(d),6,len(features)+2));source_ix=np.full((len(d),6),-1,dtype=np.int64);lengths=[]
 for i in range(len(d)):
  previous=np.flatnonzero((np.arange(len(d))<i)&(ends<=cut[i]))[-5:];chosen=np.append(previous,i);n=len(chosen);lengths.append(n);source_ix[i,:n]=chosen;tensor[i,:n,:len(features)]=x[chosen];tensor[i,:n,-2]=np.log1p((cut[i]-cut[chosen])/3.6e12);tensor[i,:n,-1]=1
  assert (cut[chosen]<=cut[i]).all();assert (ends[previous]<=cut[i]).all()
 folder=OUT/s;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'data.npz',X=tensor,current=x,y=d.label.to_numpy(),lengths=np.array(lengths),source_indices=source_ix)
 d[['symbol','start_utc','end_utc','cutoff_utc','split','label','news_record_keys']].to_csv(folder/'manifest.csv',index=False)
 ages=np.expm1(tensor[:,:,-2]);longest=[ages[i,0] for i in range(len(d))]
 manifest['stocks'][s]={'n':len(d),'split':d.split.value_counts().to_dict(),'base_features':features,'step_features':features+['log_elapsed_hours','valid_step'],'shape':list(tensor.shape),'padded_rows':int(np.sum(np.array(lengths)<6)),'oldest_snapshot_age_hours_median':float(np.median(longest)),'oldest_snapshot_age_hours_max':float(np.max(longest)),'current_news_only_rows':int((d.has_news==0).sum())};manifest['source_hashes'][s]=hashlib.sha256(source.read_bytes()).hexdigest()
(OUT/'data_summary.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
