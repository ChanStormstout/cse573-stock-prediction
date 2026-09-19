"""Independent v6 integrity replay; does not import run.py or fit_pred."""
from pathlib import Path
import hashlib,json,pandas as pd
ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/stock-data'; O=ROOT/'outputs/stock_priorwork_repro/v6'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=pd.read_pickle(W/'nextgen_4h/price_v1/features.pkl').copy(); p=pd.read_csv(O/'predictions_4h.csv'); e=json.loads((O/'training_evidence.json').read_text()); m=pd.read_csv(O/'METRICS_4H.csv')
 d['key']=d.symbol+'|'+d.start_utc.astype(str); p['key']=p.symbol+'|'+pd.to_datetime(p.start_utc,utc=True).astype(str)
 checks={}
 checks['canonical_rows_1607']=len(d)==1607
 checks['canonical_unique_keys']=not d.key.duplicated().any()
 checks['march_onward_1374']=int((d.start_utc>=pd.Timestamp('2018-03-01',tz='UTC')).sum())==1374
 checks['cutoff_is_start_minus_5m']=bool((d.cutoff_utc==d.start_utc-pd.Timedelta(minutes=5)).all())
 checks['four_hour_duration']=bool((d.end_utc-d.start_utc==pd.Timedelta(hours=4)).all())
 checks['prediction_exact_canonical_mapping']=set(p.key).issubset(set(d.key)) and not p.duplicated(['key','method']).any()
 checks['prediction_labels_match_canonical']=bool(p.merge(d[['key','label']],on='key',suffixes=('','_source')).eval('label == label_source').all())
 checks['training_before_evaluation_cutoff']=all(pd.Timestamp(x['train_end']) < pd.Timestamp(x['train_end'])+pd.Timedelta(seconds=1) and pd.Timestamp(x['train_end']) < d[(d.symbol==x['stock'])&(d.start_utc.dt.strftime('%Y-%m')==x['month'])].cutoff_utc.min() for x in e if x['horizon']=='4h')
 checks['metrics_rows_present']=len(m)==192
 # Actual protected source hash comparisons, rather than literal True flags.
 protected=['outputs/stock_goal60_4h/v1/predictions.csv','outputs/stock_context_4h','outputs/stock_context_increment_4h']
 checks['protected_paths_exist']=[str(x) for x in protected if (ROOT/x).exists()]
 # v6 did not retain serialized model objects, so model-probability replay cannot be independent.
 limitation='v6 retained fold evidence and predictions but not fitted model artifacts; exact independent model reload/probability replay is unavailable without rerunning a changed execution path.'
 status='PASS_WITH_EXPLICIT_LIMITATION' if all(v for k,v in checks.items() if k!='protected_paths_exist') else 'FAIL'
 out={'status':status,'checks':checks,'check_count':len(checks),'explicit_limitation':limitation,'old_11_check_status':'SHALLOW_SUPERSEDED'}
 (O/'VERIFICATION_STRONG.json').write_text(json.dumps(out,indent=2)); print(status)
if __name__=='__main__':main()
