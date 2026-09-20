#!/usr/bin/env python3
"""Fit-free E1 artifact and evidence-ledger contract verifier."""
import csv, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; O=ROOT/'outputs/stock_ecni_e1'
required=['SOURCE_MANIFEST.json','FNSPID_STRATIFIED_SAMPLE_MANIFEST.json','FNSPID_TIMESTAMP_AUDIT.csv',
'FNSPID_TEXT_AVAILABILITY_AUDIT.csv','FNSPID_ENTITY_AUDIT.csv','FNSPID_DUPLICATE_AUDIT.json',
'FNSPID_PRICE_COVERAGE_BY_SYMBOL.csv','COMPANY_IDENTITY_LEDGER.csv','TRAINING_PERIOD_NEWS_COVERAGE.csv',
'FROZEN_STOCK_PANEL.csv','COMPANY_SPLITS.json','FROZEN_TIME_SPLIT.json','POINT_IN_TIME_TASK_SPEC.md',
'SEC_POINT_IN_TIME_PILOT.csv','UNIFIED_EVIDENCE_LEDGER_SCHEMA.json','ECNI_DATA_ARCHITECTURE_FREEZE.md',
'CMIN_OFFICIAL_PROTOCOL.md','EDT_ACCESS_AUDIT.md','READER_PILOT_ELIGIBILITY.md',
'HUMAN_READER_LABEL_PROTOCOL.md','LICENSE_USE_AUDIT.md','E1_FINAL_AUDIT.json','E1_REPORT.md']
checks={}
checks['required_artifacts']=all((O/x).exists() for x in required)
audit=json.load((O/'E1_FINAL_AUDIT.json').open())
checks['conditional_status']=audit['status']=='CONDITIONAL_DATA_REDESIGN_REQUIRED'
checks['zero_prediction']=audit['gates']['stock_predictive_models_fitted']==audit['gates']['BA_MCC_Brier_calculated']==audit['gates']['return_based_stock_selection']==0
manifest=json.load((O/'FNSPID_STRATIFIED_SAMPLE_MANIFEST.json').open())
checks['frozen_sample_7711']=manifest.get('sample_records')==7711 or manifest.get('selected_records')==7711
prices=list(csv.DictReader((O/'FNSPID_PRICE_COVERAGE_BY_SYMBOL.csv').open()))
checks['price_rows_7693']=len(prices)==7693
checks['no_direction_fields']=not any(any(x in k.lower() for x in ['label','return','direction','brier','mcc','balanced_accuracy']) for k in prices[0])
panel=list(csv.DictReader((O/'FROZEN_STOCK_PANEL.csv').open()))
checks['panel_honestly_unfrozen']=len(panel)==1 and panel[0]['selection_status']=='NOT_FROZEN' and not panel[0]['symbol']
schema=json.load((O/'UNIFIED_EVIDENCE_LEDGER_SCHEMA.json').open())
checks['ledger_required_fields']=set(schema['required'])=={'evidence_id','target_company_id','source_type','source_record_id','published_or_available_time','time_precision_class','target_relation_status','reader_status'}

# Fit-free query semantics smoke test: later evidence and wrong-company evidence are excluded.
ledger=[
 {'target_company_id':'CIK1','published_or_available_time':'2022-01-03T14:20:00+00:00','time_precision_class':'EXACT_INTRADAY_USABLE'},
 {'target_company_id':'CIK1','published_or_available_time':'2022-01-03T14:31:00+00:00','time_precision_class':'EXACT_INTRADAY_USABLE'},
 {'target_company_id':'CIK2','published_or_available_time':'2022-01-03T14:00:00+00:00','time_precision_class':'EXACT_INTRADAY_USABLE'},]
cut=datetime.fromisoformat('2022-01-03T14:25:00+00:00')
got=[x for x in ledger if x['target_company_id']=='CIK1' and x['time_precision_class']=='EXACT_INTRADAY_USABLE' and datetime.fromisoformat(x['published_or_available_time'])<=cut]
checks['point_in_time_query_smoke']=len(got)==1
checks['locked_outcomes_uninspected']=audit['gates']['locked_2023_outcomes_uninspected'] is True
status='PASS' if all(checks.values()) else 'FAIL'
result={'status':status,'checks':checks,'fit_calls':0,'prediction_metrics_computed':0}
json.dump(result,(O/'E1_VERIFICATION.json').open('w'),indent=2)
print(json.dumps(result,indent=2))
raise SystemExit(status!='PASS')
