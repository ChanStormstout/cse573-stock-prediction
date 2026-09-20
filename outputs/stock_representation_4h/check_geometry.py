"""Fit-free check of every future aggregation partition's bandwidth support."""
from engine import *
def main():
 seal_check();inp=joblib.load(WORK/'inputs.joblib');data=joblib.load(WORK/'chunks.joblib');rows=[]
 with threadpool_limits(2):
  for b in inp['random']:
   for scope,a in [('outer',b['train'])]+[(str(j),v[0]) for j,v in enumerate(b['inner'])]:
    ix=sorted({int(c) for i in a for c in data['window_chunks'][i]});v=data['vectors'][ix];center=v.mean(0);std=v.std(0);std=np.where(std>1e-12,std,1)
    rng=np.random.default_rng(573);sample=rng.choice(ix,min(1024,len(ix)),replace=False);z=(data['vectors'][sample]-center)/std;perm=rng.permutation(len(sample));dd=np.sum((z-z[perm])**2,axis=1);valid=dd[dd>0]
    assert len(valid)>0 and np.isfinite(valid).all();sigma=np.sqrt(np.median(valid));assert np.isfinite(sigma) and sigma>0
    rows.append(dict(block=b['name'],scope=scope,unique_train_chunks=len(ix),positive_distances=len(valid),sigma=float(sigma)))
 pd.DataFrame(rows).to_csv(PUB/'geometry_preflight.csv',index=False)
 dump(PUB/'GEOMETRY_PREFLIGHT.json',dict(status='PASS',partitions=len(rows),min_sigma=min(r['sigma'] for r in rows),min_positive_distances=min(r['positive_distances'] for r in rows),model_fits=0))
 print('All240 aggregation geometry partitions valid',flush=True)
if __name__=='__main__':main()
