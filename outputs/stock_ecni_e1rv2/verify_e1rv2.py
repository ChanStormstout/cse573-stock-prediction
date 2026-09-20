#!/usr/bin/env python3
"""Fit-free verifier for the outcome-blind E1R-V2 semantic audit package."""
import csv,hashlib,json,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/stock_ecni_e1rv2';P=ROOT/'work/stock-data/ecni_e1rv2/private';E=ROOT/'outputs/stock_ecni_e1r'
PROTECTED={'E1R_FINAL_AUDIT.json':'b28114b895ecbe7ef3ace8e42279e9271e5e94165922ac168796b9381c669705','FROZEN_STOCK_PANEL_E1R.csv':'7c5a1a1d618f76a9801cd57a05f531c1a2885674500e11c93e1697d903559948','COMPANY_SPLITS_E1R.json':'7ae03321d0ede4e444a90b89cede0d26291eac756eac2aaeb757f9035fb28f34','TIME_SPLIT_E1R.json':'53ed35ad21ff462fa6913db1d1acaa41126771d6c355e9e4f77c572c40acba3c'}
REQ=['PRE_REGISTRATION.md','ALIAS_RISK_CENSUS.csv','NDAQ_SENTINEL_AUDIT.json','ENTITY_SEMANTIC_REVIEW_MANIFEST.json','ENTITY_SEMANTIC_LABELING_GUIDE.md','MECHANICAL_ALIAS_CONTAMINATION_REPORT.md','SEC_FORM_COVERAGE_BY_COMPANY.csv','SEC_FACT_CHANNEL_COVERAGE.md','SEC_FACTUAL_EVENT_CANDIDATES.jsonl','SEC_SUPPLEMENTAL_SOURCE_MANIFEST.json','READER_PAIR_SAMPLING_MANIFEST.json','READER_PAIR_LABELING_GUIDE.md','E1RV2_FINAL_AUDIT.json','E1RV2_REPORT.md','run_e1rv2.py','verify_e1rv2.py']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
checks={'required_public_outputs':all((O/x).is_file() for x in REQ),'historical_E1R_preserved':all(digest(E/n)==v for n,v in PROTECTED.items())}
a=json.load((O/'E1RV2_FINAL_AUDIT.json').open());m=json.load((O/'ENTITY_SEMANTIC_REVIEW_MANIFEST.json').open());n=json.load((O/'NDAQ_SENTINEL_AUDIT.json').open());rp=json.load((O/'READER_PAIR_SAMPLING_MANIFEST.json').open())
cards=[json.loads(x) for x in (P/'ENTITY_SEMANTIC_REVIEW_CARDS.jsonl').open()]
checks['private_entity_cards_hash']=digest(P/'ENTITY_SEMANTIC_REVIEW_CARDS.jsonl')==m['private_sha256']==n['private_cards_sha256']
checks['entity_card_ids_exact']=len(cards)==len({x['card_id'] for x in cards})==m['total_unique_cards']==515 and {x['card_id'] for x in cards}==set(m['card_ids'])
checks['no_semantic_labels_assigned']=all(x['semantic_label'] is None and x['independent_reviewer_id'] is None for x in cards)
panel=[x for x in cards if 'PANEL_BASE' in x['sample_roles']];pc=Counter(x['company_id'] for x in panel)
checks['panel_review_330_all_66']=len(panel)==330 and len(pc)==66 and set(pc.values())=={5}
sentinel=[x for x in cards if 'NDAQ_SENTINEL' in x['sample_roles']]
checks['NDAQ_sentinel_exact_100']=len(sentinel)==100 and all(x['target_ticker']=='NDAQ' and x['CIK']==1120193 for x in sentinel) and n['sentinel_cards']==100
risk=[x for x in cards if 'ALIAS_RISK_ENRICHED' in x['sample_roles']]
checks['risk_enriched_exact_100']=len(risk)==100
aliases=list(csv.DictReader((O/'ALIAS_RISK_CENSUS.csv').open()))
checks['alias_census_complete_unique']=len(aliases)==3129 and len({(x['CIK'],x['matched_alias']) for x in aliases})==3129
nd=[x for x in aliases if x['CIK']=='1120193' and x['matched_alias'].lower()=='nasdaq']
checks['NDAQ_alias_flagged']=len(nd)==1 and 'DUAL_ROLE_EXCHANGE_COMPANY_SENTINEL' in nd[0]['risk_flags'] and float(nd[0]['training_day_coverage'])==1.0
checks['independent_review_pending']=a['status']=='AWAITING_INDEPENDENT_ENTITY_REVIEW' and m['independent_labels_received']==0 and a['semantic_precision'] is None and not a['semantic_precision_gate_evaluated']
checks['no_relink_or_panel_rebuild']=not a['complete_relinking_rerun'] and not a['panel_changed'] and a['final_panel_size_historical_pending_validation']==66 and not (O/'FROZEN_STOCK_PANEL_E1RV2.csv').exists()
sec=list(csv.DictReader((O/'SEC_FORM_COVERAGE_BY_COMPANY.csv').open()));events=[json.loads(x) for x in (O/'SEC_FACTUAL_EVENT_CANDIDATES.jsonl').open()]
checks['SEC_all_66_companies']=len(sec)==66 and len({x['company_id'] for x in sec})==66
checks['SEC_filer_classes']=Counter(x['filer_class'] for x in sec)==Counter(a['filer_class_counts'])
forms=['8-K','10-Q','10-K','6-K','20-F'];computed={f:sum(int(x[f'{f}_count']) for x in sec) for f in forms}
checks['SEC_form_counts']=computed==a['SEC_form_counts'] and sum(computed.values())==len(events)==a['SEC_factual_event_candidates']
checks['SEC_event_fields']=all(set(['CIK','accession','form','acceptance_time','period_end','document_reference','filer_class'])<=set(x) and '2018-01-01'<=x['filing_date']<='2021-12-31' for x in events)
sm=json.load((O/'SEC_SUPPLEMENTAL_SOURCE_MANIFEST.json').open());supp={x['file']:x for x in sm['overlapping_supplemental_files']}
z=zipfile.ZipFile(ROOT/'work/stock-data/ecni_e1r/raw/submissions.zip');selected=[x for x in csv.DictReader((E/'FROZEN_STOCK_PANEL_E1R.csv').open()) if x['selection_status']=='SELECTED'];required=set()
for p in selected:
 sx=json.loads(z.read(f"CIK{int(p['CIK']):010d}.json"))
 required|={x['name'] for x in sx.get('filings',{}).get('files',[]) if x.get('filingTo','')>='2018-01-01' and x.get('filingFrom','')<='2021-12-31'}
checks['SEC_supplemental_coverage_complete']=required==set(supp) and all(digest(P/'sec_supplemental_submissions'/name)==x['sha256'] for name,x in supp.items())
checks['reader_pairs_correctly_blocked']=rp['status']=='BLOCKED_PENDING_INDEPENDENT_ENTITY_REVIEW' and rp['reader_review_cards']==0 and rp['news_pair_candidates']==0 and (P/'READER_PAIR_REVIEW_CARDS.jsonl').stat().st_size==0
checks['zero_outcomes_and_models']=a['returns_or_predictive_outcomes_inspected']==a['predictive_models_fitted']==a['reader_models_fitted']==0
checks['public_manifest_has_no_card_text']=not any(k in m for k in ('title','evidence_context','article','body')) and not (O/'ENTITY_SEMANTIC_REVIEW_CARDS.jsonl').exists()
status='PASS' if all(checks.values()) else 'FAIL';out={'status':status,'checks':checks,'fit_calls':0,'return_or_direction_metrics':0,'independent_semantic_labels':0}
(O/'E1RV2_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(status!='PASS')
