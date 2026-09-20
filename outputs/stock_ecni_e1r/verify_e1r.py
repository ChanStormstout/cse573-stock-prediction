#!/usr/bin/env python3
"""Fit-free E1R integrity verifier."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/stock_ecni_e1r'
req=['PRE_REGISTRATION.md','FNSPID_COMPLETE_METADATA_MANIFEST.json','SEC_ISSUER_IDENTITY_HISTORY.csv','FNSPID_RELINKED_ENTITY_EDGES.parquet','LOCAL_CORPUS_ARTIFACTS_MANIFEST.json','ENTITY_RELINK_AUDIT.md','TIMESTAMP_CENSUS.csv','DUPLICATE_GROUP_SUMMARY.csv','DUPLICATE_GROUP_CENSUS.json','HISTORICAL_INDUSTRY_LEDGER.csv','RELINKED_TRAINING_NEWS_COVERAGE.csv','FROZEN_STOCK_PANEL_E1R.csv','COMPANY_SPLITS_E1R.json','TIME_SPLIT_E1R.json','EXACT_TIME_EVENT_CORPUS_AUDIT.json','SEC_FACT_CHANNEL_PILOT.md','ECNI_FINAL_DATA_ARCHITECTURE.md','READER_PILOT_INPUT_AUDIT.json','E1R_FINAL_AUDIT.json','E1R_REPORT.md']
checks={'required_outputs':all((O/x).exists() for x in req)}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
local=json.load((O/'LOCAL_CORPUS_ARTIFACTS_MANIFEST.json').open())
checks['local_complete_artifact_hashes']=all((ROOT/x['path']).stat().st_size==x['bytes'] and digest(ROOT/x['path'])==x['sha256'] for x in local['artifacts'])
a=json.load((O/'E1R_FINAL_AUDIT.json').open());m=json.load((O/'FNSPID_COMPLETE_METADATA_MANIFEST.json').open())
checks['native_role_preserved']=a['native_FNSPID_role_preserved']=='ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS'
checks['complete_census']=m['status']=='COMPLETE' and m['total_rows_processed']>1_000_000
checks['native_not_ground_truth']=m['native_ticker_used_as_ground_truth'] is False
checks['zero_predictive_work']=a['stock_predictive_models_fitted']==a['BA_MCC_Brier_computed']==a['return_based_stock_selection']==a['reader_models_trained']==0
checks['locked_outcomes_uninspected']=a['locked_2023_outcomes_inspected_or_summarized']==0
con=duckdb.connect();ep=str(O/'FNSPID_RELINKED_ENTITY_EDGES.parquet')
checks['edges_have_explicit_rules']=con.execute(f"SELECT count(*)=0 FROM read_parquet('{ep}') WHERE confidence_rule NOT IN ('UNIQUE_DISTINCTIVE_SEC_ISSUER_ALIAS','EXPLICIT_EXCHANGE_OR_DOLLAR_TICKER')").fetchone()[0]
checks['no_native_only_edge_rule']=con.execute(f"SELECT count(*)=0 FROM read_parquet('{ep}') WHERE confidence_rule ILIKE '%NATIVE%'").fetchone()[0]
panel=list(csv.DictReader((O/'FROZEN_STOCK_PANEL_E1R.csv').open()));sel=[r for r in panel if r['selection_status']=='SELECTED'];cnt=Counter(r['historical_industry'] for r in sel)
if a['panel_pass']:
 checks['panel_gate']=len(cnt)>=8 and min(cnt.values())>=4
 checks['selected_industry_stable']=all(r['industry_status']=='INDUSTRY_STABLE' for r in sel)
 split=json.load((O/'COMPANY_SPLITS_E1R.json').open());sets={k:set(split[k]) for k in ('TRAIN_COMPANIES','DEV_UNSEEN_COMPANIES','LOCKED_UNSEEN_COMPANIES')}
 checks['company_split_frozen']=split['status']=='FROZEN'
 checks['company_split_exact_44_11_11']=[len(split[k]) for k in ('TRAIN_COMPANIES','DEV_UNSEEN_COMPANIES','LOCKED_UNSEEN_COMPANIES')]==[44,11,11]
 checks['company_split_disjoint_complete']=(not (sets['TRAIN_COMPANIES']&sets['DEV_UNSEEN_COMPANIES'] or sets['TRAIN_COMPANIES']&sets['LOCKED_UNSEEN_COMPANIES'] or sets['DEV_UNSEEN_COMPANIES']&sets['LOCKED_UNSEEN_COMPANIES'])) and set.union(*sets.values())=={r['company_id'] for r in sel}
 selected_by_id={r['company_id']:r for r in sel}
 split_industry={k:Counter(selected_by_id[x]['historical_industry'] for x in v) for k,v in sets.items()}
 checks['company_split_industry_stratified']=all(split_industry['TRAIN_COMPANIES'][i]==4 and split_industry['DEV_UNSEEN_COMPANIES'][i]==1 and split_industry['LOCKED_UNSEEN_COMPANIES'][i]==1 for i in cnt)
 checks['sec_pilot_completed']=a['SEC_pilot_records']>0
else:
 checks['panel_gate']=len(cnt)<8 or min(cnt.values(),default=0)<4
 checks['company_split_stopped']=json.load((O/'COMPANY_SPLITS_E1R.json').open())['status']!='FROZEN'
 checks['sec_pilot_correctly_stopped']=a['SEC_pilot_records']==0
checks['time_classes_separated']=set(r['available_time_class'] for r in con.execute(f"SELECT DISTINCT available_time_class FROM read_parquet('{ep}')").fetchdf().to_dict('records')).issubset({'EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE','AMBIGUOUS','INVALID'})
status='PASS' if all(checks.values()) else 'FAIL';out={'status':status,'checks':checks,'fit_calls':0,'direction_or_return_metrics':0};json.dump(out,(O/'E1R_VERIFICATION.json').open('w'),indent=2);print(json.dumps(out,indent=2));raise SystemExit(status!='PASS')
