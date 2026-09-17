"""One-time local cache dtype repair; preserve originals and inference source."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from experiment import HERE,PRIVATE,OUT,sha,dump

def main():
 archive=PRIVATE/'object_key_originals';archive.mkdir(exist_ok=False)
 source=HERE/'chronos_experiment.py';shutil.copy2(source,archive/source.name)
 before=source.read_text();after=before.replace('keys=d.iloc[ix].key.to_numpy()','keys=np.asarray(d.iloc[ix].key,dtype=str)').replace('keys=d.key.to_numpy()','keys=np.asarray(d.key,dtype=str)');assert before!=after
 source.write_text(after);records=[]
 for p in [PRIVATE/'chronos_forecasts.npz']+sorted((PRIVATE/'chronos_chunks').glob('*.npz')):
  # These files were created locally in this run; never apply this to untrusted files.
  z=np.load(p,allow_pickle=True);values={k:z[k] for k in z.files};old=values['keys'];assert old.dtype==object
  saved=archive/p.name;shutil.copy2(p,saved);values['keys']=old.astype(str);np.savez_compressed(p,**values)
  check=np.load(p)
  for k,v in values.items():assert np.array_equal(check[k],v,equal_nan=True) if np.issubdtype(v.dtype,np.number) else np.array_equal(check[k],v)
  records.append(dict(file=str(p.relative_to(PRIVATE)),before_sha256=sha(saved),after_sha256=sha(p)))
 fp=PRIVATE/'chronos_cache.json';shutil.copy2(fp,archive/fp.name);x=json.loads(fp.read_text());x['code']=sha(source);dump(fp,x)
 dump(OUT/'cache_repair.json',dict(reason='Generated Pandas string keys had object dtype, preventing default NumPy safe reload; convert to Unicode, predictions unchanged.',original_code_sha256=sha(archive/source.name),current_code_sha256=sha(source),archived_originals=str(archive.relative_to(PRIVATE)),caches=records))
 print('Repaired',len(records),'cache keys; values unchanged; originals preserved')
if __name__=='__main__':main()
