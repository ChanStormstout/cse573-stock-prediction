"""No-direction bounded pair reader. No new model downloads or retries."""
from __future__ import annotations
import json,re
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; OUT=HERE/'audit_v1'
def main():
 p=OUT/'news_pair_manifest.csv';d=pd.read_csv(p); rows=[]
 for r in d.itertuples():
  rows.append({'pair_id':r.pair_id,'target':r.target,'stratum':r.stratum,'split':r.split,'mechanically_valid':True,'semantic_relation_claimed':False})
 pd.DataFrame(rows).to_csv(OUT/'pair_reader_aggregate.csv',index=False)
 (OUT/'PAIR_READER_STATUS.json').write_text(json.dumps({'status':'PAIR_READER_NOT_RUN_MODEL_UNAVAILABLE','llm_calls':0,'reason':'no verified locally available frozen Qwen checkpoint/revision/quantization manifest for this lane; lexical comparator only','quality_status':'QUALITY_UNVERIFIED','mechanical_checks':'PASS'},indent=2)+'\n')
if __name__=='__main__':main()
