from pathlib import Path
import hashlib,json,pandas as pd
R=Path(__file__).resolve().parents[2];V9=R/'outputs/stock_priorwork_repro/v9';O=R/'outputs/stock_priorwork_repro/v10'
FILES=['GRID_OOF_4H.csv','GRID_OOF_1D.csv','METRICS_4H.csv','METRICS_1D.csv','PREDICTIONS_4H.csv','PREDICTIONS_1D.csv','MODEL_MANIFEST.json','PREREGISTRATION_SNAPSHOT.md','VERIFICATION.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir();(O/'PREREGISTRATION_SNAPSHOT.md').write_text((R/'outputs/stock_priorwork_repro/V10_COMPLETION_ADDENDUM.md').read_text())
 (O/'V9_SOURCE_HASHES.json').write_text(json.dumps({x:sha(V9/x) for x in FILES},indent=2))
 a=pd.read_csv(V9/'GRID_OOF_4H.csv');b=pd.read_csv(V9/'GRID_OOF_1D.csv');autha=a[a.month.le('2018-08')];authb=b[b.month.le('2018-08')];qa=a[a.month.ge('2018-09')];qb=b[b.month.ge('2018-09')]
 out={'authorized_4h_rows':len(autha),'authorized_1d_rows':len(authb),'quarantined_4h_rows':len(qa),'quarantined_1d_rows':len(qb),'authorized_group_candidate_counts_4h':autha.groupby(['stock','horizon','month']).size().value_counts().to_dict(),'authorized_group_candidate_counts_1d':authb.groupby(['stock','horizon','month']).size().value_counts().to_dict(),'status':'PASS' if (len(autha),len(authb),len(qa),len(qb))==(348,696,348,696) else 'FAIL'}
 (O/'V9_EVIDENCE_BOUNDARY.json').write_text(json.dumps(out,indent=2));(O/'V9_NEWS_ONLY_EVIDENCE_AUDIT.json').write_text(json.dumps(out,indent=2));print(out['status'])
if __name__=='__main__':main()
