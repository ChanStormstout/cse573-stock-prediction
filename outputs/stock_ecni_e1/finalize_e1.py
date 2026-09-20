#!/usr/bin/env python3
"""Materialize non-predictive E1 ledgers and gate artifacts."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/stock_ecni_e1'; PRIV=ROOT/'work/stock-data/ecni_e1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def wc(path,rows,fields):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

sample=[json.loads(x) for x in (PRIV/'audit/fnspid_stratified_sample.jsonl').open()]
prices={r['symbol']:r for r in csv.DictReader((OUT/'FNSPID_PRICE_COVERAGE_BY_SYMBOL.csv').open())}
sec=json.load((PRIV/'raw/company_tickers_exchange.json').open())
secrows=[dict(zip(sec['fields'],r)) for r in sec['data']]

# Public identity ledger: modern SEC identity only; no historical-sector claim.
ident=[]
for r in secrows:
 t=str(r['ticker']).upper()
 ident.append({'company_id':f"CIK{int(r['cik']):010d}",'ticker':t,'issuer_name':r['name'],'CIK':int(r['cik']),
  'exchange':r['exchange'],'sector':'UNAVAILABLE','industry':'UNAVAILABLE','effective_date_range':'UNRESOLVED',
  'identity_confidence':'CURRENT_SEC_MAPPING_ONLY','identity_source':'SEC company_tickers_exchange.json current snapshot',
  'history_limitation':'Modern mapping; ticker history and survivorship are not established.',
  'price_eligible_2018_2023':prices.get(t,{}).get('price_eligible_2018_2023','0')})
wc(OUT/'COMPANY_IDENTITY_LEDGER.csv',ident,list(ident[0]))

# Sample-derived coverage is explicitly non-census and unusable for ranking.
c=defaultdict(lambda:Counter())
for r in sample:
 if '2018'<=str(r['year'])<='2021' and r.get('ticker'):
  x=c[r['ticker'].upper()];x['sample_articles']+=1;x['title']+=bool(r.get('title'));x['body']+=bool(r.get('body'));x['classifiable']+=bool(r.get('date'))
  if r.get('source'):x['publishers']+=0
  x['publisher_set_'+str(r.get('source'))]=1
cov=[]
for t,x in sorted(c.items()):
 n=x['sample_articles']; pubs=sum(v for k,v in x.items() if k.startswith('publisher_set_'))
 cov.append({'symbol':t,'period':'2018-01-01/2021-12-31','trading_days':'NOT_COMPUTED_WITHOUT_PANEL',
  'days_with_eligible_pre_cutoff_news':'UNAVAILABLE_FROM_RANGE_SAMPLE','news_coverage_fraction':'UNAVAILABLE_FROM_RANGE_SAMPLE',
  'sample_article_count':n,'sample_publisher_count':pubs,'sample_timestamp_classifiable_fraction':x['classifiable']/n,
  'sample_title_available_fraction':x['title']/n,'sample_body_available_fraction':x['body']/n,
  'entity_audit_status':'PROVISIONAL_SAMPLE_ONLY','panel_selection_eligible':False,
  'reason':'Bounded byte-range sample is not a complete stock/day census.'})
wc(OUT/'TRAINING_PERIOD_NEWS_COVERAGE.csv',cov,list(cov[0]))

panel_fields=['selection_status','symbol','company_id','sector','coverage_stratum','reason','deterministic_tie_break']
wc(OUT/'FROZEN_STOCK_PANEL.csv',[{'selection_status':'NOT_FROZEN','symbol':'','company_id':'','sector':'','coverage_stratum':'',
 'reason':'Exact 2018-2021 per-stock news census and defensible sector taxonomy unavailable from bounded acquisition.',
 'deterministic_tie_break':'sha256(company_id|57320260919) registered but not applied'}],panel_fields)
json.dump({'status':'NOT_FROZEN_PANEL_PREREQUISITE_FAILED','hash_rule':'sha256(company_id|57320260919)',
 'target_proportions':{'train': '2/3','dev_unseen':'1/6','locked_unseen':'1/6'},'assignments':[],
 'reason':'No deterministic named panel was defensibly frozen.'},(OUT/'COMPANY_SPLITS.json').open('w'),indent=2)
json.dump({'status':'FROZEN_CONDITIONAL_ON_PANEL','train_forward_validation':['2018-01-01','2021-12-31'],
 'development':['2022-01-01','2022-12-31'],'locked_test':['2023-01-01','2023-12-31'],
 'locked_label_status':'NOT_INSPECTED_OR_SUMMARIZED_IN_E1','coverage_basis':'price/news existence only'},(OUT/'FROZEN_TIME_SPLIT.json').open('w'),indent=2)

# SEC pilot is intentionally blocked by the missing frozen panel.
wc(OUT/'SEC_POINT_IN_TIME_PILOT.csv',[],['CIK','ticker','accession','form','filing_date','acceptance_datetime','period_end','document_reference','availability_time','amendment_status'])

schema={'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'UNIFIED_EVIDENCE_LEDGER_SCHEMA.json',
 'title':'Unified point-in-time evidence ledger','type':'object','additionalProperties':False,
 'required':['evidence_id','target_company_id','source_type','source_record_id','published_or_available_time','time_precision_class','target_relation_status','reader_status'],
 'properties':{
  'evidence_id':{'type':'string','minLength':1},'target_company_id':{'type':'string','minLength':1},
  'source_type':{'enum':['NEWS','SEC']},'source_record_id':{'type':'string','minLength':1},
  'published_or_available_time':{'type':'string','format':'date-time'},
  'time_precision_class':{'enum':['EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE','AMBIGUOUS','INVALID']},
  'source_or_publisher':{'type':['string','null']},'title':{'type':['string','null']},
  'body_private_reference':{'type':['string','null']},'url_or_accession':{'type':['string','null']},
  'target_relation_status':{'enum':['DIRECT_TARGET','MULTI_COMPANY_INCLUDES_TARGET','INDIRECT_OR_COMPETITOR','NO_EVIDENCE_TARGET','UNDETERMINED']},
  'duplicate_group_id':{'type':['string','null']},'content_hash':{'type':['string','null'],'pattern':'^[0-9a-f]{64}$'},
  'event_period':{'type':['string','null']},'reader_status':{'type':'string'},
  'availability_policy':{'type':'string'},'provenance_revision':{'type':['string','null']}}}
json.dump(schema,(OUT/'UNIFIED_EVIDENCE_LEDGER_SCHEMA.json').open('w'),indent=2)

sources=[
 {'dataset':'FNSPID','artifact':'Stock_news/All_external.csv','revision':'bf9189c41527198897d1af3e17b1a0095279fc45','remote_size':5731397037,'remote_sha256':'5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3','acquisition':'12 deterministic 2 MiB HTTP ranges; private'},
 {'dataset':'FNSPID','artifact':'Stock_news/nasdaq_exteral_data.csv','revision':'bf9189c41527198897d1af3e17b1a0095279fc45','remote_size':23232979597,'remote_sha256':'1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da','acquisition':'12 deterministic 2 MiB HTTP ranges; private'},
 {'dataset':'FNSPID','artifact':'Stock_price/full_history.zip','revision':'bf9189c41527198897d1af3e17b1a0095279fc45','remote_size':589525596,'remote_sha256':'03da4fce7ebea90d5715ba3501773d410ae663b617027b338ef000a9955dab91','local_sha256':sha(PRIV/'raw/full_history.zip'),'acquisition':'complete official file; private'},
 {'dataset':'SEC','artifact':'company_tickers_exchange.json','revision':'downloaded_current_2026-09-19','local_size':(PRIV/'raw/company_tickers_exchange.json').stat().st_size,'local_sha256':sha(PRIV/'raw/company_tickers_exchange.json'),'acquisition':'official SEC current mapping'},
 {'dataset':'Nasdaq','artifact':'screener current snapshot','revision':'downloaded_current_2026-09-19','local_size':(PRIV/'raw/nasdaq_screener.json').stat().st_size,'local_sha256':sha(PRIV/'raw/nasdaq_screener.json'),'acquisition':'current identity candidate; no sector fields'}]
json.dump({'generated':'2026-09-19','sources':sources,'private_sample_sha256':sha(PRIV/'audit/fnspid_stratified_sample.jsonl'),
 'copyright_boundary':'No article body is committed.'},(OUT/'SOURCE_MANIFEST.json').open('w'),indent=2)

audit={'status':'CONDITIONAL_DATA_REDESIGN_REQUIRED','fnspid_role':'ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS',
 'architecture':'ARCH_C_CONDITIONAL','gates':{
  'broader_fnspid_audit_completed':True,'timestamp_role_frozen':True,'text_role_frozen':True,'price_corpus_audited':True,
  'deterministic_named_panel_frozen':False,'company_split_frozen':False,'time_split_frozen':True,
  'locked_2023_outcomes_uninspected':True,'evidence_ledger_implemented':True,'architecture_frozen':True,
  'reader_label_protocol_frozen':True,'stock_predictive_models_fitted':0,'BA_MCC_Brier_calculated':0,'return_based_stock_selection':0},
 'blocking_reasons':['Bounded range sample cannot yield exact 2018-2021 stock/day news coverage.','Current identity sources lack a defensible sector taxonomy and historical membership.','Provisional entity-association lower confidence bound is below 0.80.'],
 'next_required_action':'Acquire a metadata-complete news index and a versioned sector/issuer mapping, then rerun panel freeze without outcomes.'}
json.dump(audit,(OUT/'E1_FINAL_AUDIT.json').open('w'),indent=2)
print(json.dumps(audit,indent=2))
