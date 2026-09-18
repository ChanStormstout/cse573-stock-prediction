"""Read-only canonical R1 parity audit in a new private lane path; no historical models touched."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'audit_v1'; PRIVATE=ROOT/'work/stock-data/context_increment_4h'
def main():
 a=pd.read_csv(ROOT/'outputs/stock_goal60_4h/v1/predictions.csv');a=a[a.month>='2018-03'][['key','R1']];b=pd.read_csv(ROOT/'outputs/stock_nextgen_4h/runs/v1/PREDICTIONS.csv')[['key','R1']];j=a.merge(b,on='key',suffixes=('_goal60','_nextgen'),validate='one_to_one');err=float(np.abs(j.R1_goal60-j.R1_nextgen).max());result={'status':'CANONICAL_R1_AUDIT_PARTIAL_SECOND_SOURCE','goal60_evaluated_keys':len(a),'nextgen_overlap_keys':len(j),'nextgen_overlap_max_abs_error':err,'nextgen_overlap_direction_parity':bool(np.array_equal(j.R1_goal60>=.5,j.R1_nextgen>=.5)),'note':'Goal60 supplies all 1,374 non-warmup canonical R1 probabilities; the inspected nextgen public source supplies only 609 overlapping frozen keys. This is a read-only audit, not an independent coefficient refit.'};OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);(OUT/'canonical_r1_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
