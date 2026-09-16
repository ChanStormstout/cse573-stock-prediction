"""Run a frozen integrated bundle on prepared historical inputs, without labels."""
from run import Transform,combine,BRANCHES,sha
from pathlib import Path
import argparse,joblib,numpy as np,pandas as pd

def predict(bundle_path,input_path,embedding_path):
 bundle=joblib.load(bundle_path)
 if sha(embedding_path)!=bundle['embedding_sha256']:raise ValueError('Embedding cache mismatch')
 d=pd.read_pickle(input_path);d=d[d.symbol==bundle['symbol']].copy();z=np.load(embedding_path);keys=z['keys'];emb=z['embeddings'].astype(np.float64)
 out=d[['symbol','start_utc','has_news']].copy()
 for kind,obj in bundle['models'].items():out[kind]=obj['classifier'].predict_proba(obj['transform'].transform(d,keys,emb))[:,1]
 w=bundle['choice']['weights'];out['integrated']=out.title.to_numpy() if w is None else combine(out,w)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--embeddings',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 predict(a.bundle,a.input,a.embeddings).to_csv(a.output,index=False)
