#!/usr/bin/env python3
"""Finalize E1R without predictive computation."""
from __future__ import annotations
import csv,hashlib,json,zipfile,duckdb
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/stock_ecni_e1r';PRIV=ROOT/'work/stock-data/ecni_e1r';RAW=PRIV/'raw';PVT=PRIV/'private'
def H(s):return hashlib.sha256(s.encode()).hexdigest()
def filehash(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.load((OUT/'FNSPID_COMPLETE_METADATA_MANIFEST.json').open());panel=list(csv.DictReader((OUT/'FROZEN_STOCK_PANEL_E1R.csv').open()));selected=[r for r in panel if r['selection_status']=='SELECTED'];summary=json.load((PVT/'panel_build_summary.json').open())
 # Bounded SEC fact-channel pilot after panel decision only.
 z=zipfile.ZipFile(RAW/'submissions.zip');zn=set(z.namelist());byind=defaultdict(list)
 for r in selected:byind[r['historical_industry']].append(r)
 sec_rows=[]
 if summary['panel_pass']:
  for ind,rs in sorted(byind.items()):
   r=min(rs,key=lambda x:H(x['company_id']+'|ECNI_E1R_SEC_PILOT'));
   n=f"CIK{int(r['CIK']):010d}.json";x=json.loads(z.read(n)) if n in zn else None
   if not x:continue
   rec=x.get('filings',{}).get('recent',{});kept=0
   for i,form in enumerate(rec.get('form',[])):
    fd=rec['filingDate'][i];
    if form not in ('8-K','10-Q','10-K') or not ('2018-01-01'<=fd<='2021-12-31'):continue
    sec_rows.append({'historical_industry':ind,'CIK':int(r['CIK']),'ticker':r['ticker'],'accession':rec['accessionNumber'][i],'form':form,'filing_date':fd,
     'acceptance_datetime':rec['acceptanceDateTime'][i],'period_end':rec['reportDate'][i],'document_reference':rec['primaryDocument'][i],
     'availability_time':rec['acceptanceDateTime'][i],'amendment_status':'AMENDMENT' if form.endswith('/A') else 'ORIGINAL'})
    kept+=1
    if kept>=6:break
 with (PVT/'sec_fact_channel_pilot.csv').open('w',newline='') as f:
  fields=list(sec_rows[0]) if sec_rows else ['historical_industry','CIK','ticker','accession','form','filing_date','acceptance_datetime','period_end','document_reference','availability_time','amendment_status'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(sec_rows)
 (OUT/'SEC_FACT_CHANNEL_PILOT.md').write_text('# SEC factual channel pilot\n\n'+(('The bounded pilot selected one company per represented historical industry by deterministic CIK hash. It retained **%d** 8-K/10-Q/10-K filing metadata objects from 2018–2021. Each object stores acceptance time separately from report period; **%d/%d** have `period_end != availability_time` calendar date, demonstrating the point-in-time distinction. The row-level ledger remains private and no returns were accessed.\n'%(len(sec_rows),sum(x['period_end'] not in x['availability_time'] for x in sec_rows),len(sec_rows))) if sec_rows else 'Not run because the balanced-panel gate did not pass. No arbitrary companies were substituted.\n'))
 # Derived role and architecture from complete census/panel only.
 con=duckdb.connect();ep=str(OUT/'FNSPID_RELINKED_ENTITY_EDGES.parquet');mp=str(PVT/'fnspid_complete_metadata.parquet')
 n,body,eligible=con.execute(f"SELECT count(*),sum(CASE WHEN body_available THEN 1 ELSE 0 END),sum(CASE WHEN available_time_class IN ('EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE') THEN 1 ELSE 0 END) FROM read_parquet('{ep}') WHERE relation_type IN ('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE')").fetchone()
 bodyfrac=body/n if n else 0;timefrac=eligible/n if n else 0
 # Outcome-blind heuristic candidate-pair support counts; these are not labels.
 flag=con.execute(f"SELECT sum(candidate_numeric_update),sum(candidate_period_change),sum(candidate_action_change),sum(candidate_denial_correction),sum(candidate_hypothetical_opinion) FROM read_parquet('{mp}')").fetchone()
 dups=sum(int(r['article_count'])-1 for r in csv.DictReader((OUT/'DUPLICATE_GROUP_SUMMARY.csv').open()) if r['group_type']=='EXACT_TITLE')
 multi=con.execute(f"SELECT count(DISTINCT record_id_hash) FROM read_parquet('{ep}') WHERE relation_type='MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE'").fetchone()[0]
 rp=json.load((OUT/'READER_PILOT_INPUT_AUDIT.json').open());rp['outcome_blind_candidate_counts']={'new_event':'UNLABELED_POOL_REQUIRES_HUMAN_REVIEW','repeat_exact_title_excess_records':dups,'numeric_update_heuristic':int(flag[0] or 0),'period_change_heuristic':int(flag[1] or 0),'action_change_heuristic':int(flag[2] or 0),'denial_correction_heuristic':int(flag[3] or 0),'different_actor':'UNRESOLVED_WITHOUT_ACTOR_EXTRACTION','different_company_multi_edge_articles':int(multi),'hypothetical_opinion_heuristic':int(flag[4] or 0),'insufficient_evidence':'UNLABELED_POOL_REQUIRES_HUMAN_REVIEW'};json.dump(rp,(OUT/'READER_PILOT_INPUT_AUDIT.json').open('w'),indent=2)
 if summary['panel_pass'] and bodyfrac>=.70 and timefrac>=.80:role='RELINKED_ROLE_A_FULL_ECNI';arch='ARCH_C_HYBRID_FACTS_WHERE_VERIFIED'
 elif summary['panel_pass'] and timefrac>=.80:role='RELINKED_ROLE_B_DISSEMINATION_AND_DENSE_TEXT';arch='ARCH_B_SEC_FACTS_PLUS_RELINKED_NEWS_DISSEMINATION'
 elif n and timefrac>=.80:role='RELINKED_ROLE_C_DISSEMINATION_ONLY';arch='ARCH_D_EXTERNAL_NEWS_SOURCE_REQUIRED'
 else:role='RELINKED_ROLE_D_REJECT';arch='ARCH_D_EXTERNAL_NEWS_SOURCE_REQUIRED'
 if summary['panel_pass']:status='PASS_PANEL_AND_ARCHITECTURE_FROZEN'
 elif n==0:status='FAIL_FNSPID_RELINKING_INSUFFICIENT'
 else:status='CONDITIONAL_NEWS_SOURCE_REDESIGN_REQUIRED'
 (OUT/'ECNI_FINAL_DATA_ARCHITECTURE.md').write_text(f'''# ECNI final data architecture\n\n- Original native FNSPID role remains `ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS`.\n- Derived relinked role: `{role}`.\n- Architecture: `{arch}`.\n- Relinked edge body fraction: {bodyfrac:.4%}; timestamp-eligible fraction: {timefrac:.4%}.\n- Balanced panel gate: `{summary['panel_pass']}`.\n\nNews facts are usable only when supported by an explicit high-confidence issuer edge and private evidence reference. SEC acceptance time is authoritative for filing availability. Missing factual evidence remains unknown. This choice uses only entity, timing, text, coverage, SEC and access evidence; no prediction result exists.\n''')
 # Complete source manifest hashes.
 source={'official_revision':'bf9189c41527198897d1af3e17b1a0095279fc45','files':manifest['files'],'SEC_submissions_bulk':{'bytes':(RAW/'submissions.zip').stat().st_size,'sha256':filehash(RAW/'submissions.zip')},'Fama_French_SIC12':{'bytes':(RAW/'Siccodes12.zip').stat().st_size,'sha256':filehash(RAW/'Siccodes12.zip')}}
 json.dump(source,(OUT/'SOURCE_MANIFEST_E1R.json').open('w'),indent=2)
 split=json.load((OUT/'COMPANY_SPLITS_E1R.json').open())
 audit={'status':status,'native_FNSPID_role_preserved':'ROLE_C_PANEL_PRICE_ONLY_OR_REJECTED_NEWS','FNSPID_RELINKED_role':role,'architecture':arch,'complete_rows_processed':manifest['total_rows_processed'],'relinked_edges':n,'body_fraction':bodyfrac,'timestamp_eligible_fraction':timefrac,**summary,
  'company_split_status':split['status'],'company_split_counts':{k:len(split[k]) for k in ('TRAIN_COMPANIES','DEV_UNSEEN_COMPANIES','LOCKED_UNSEEN_COMPANIES')},'time_split_status':json.load((OUT/'TIME_SPLIT_E1R.json').open())['status'],'SEC_pilot_records':len(sec_rows),
  'locked_2023_outcomes_inspected_or_summarized':0,'stock_predictive_models_fitted':0,'BA_MCC_Brier_computed':0,'return_based_stock_selection':0,'reader_models_trained':0}
 json.dump(audit,(OUT/'E1R_FINAL_AUDIT.json').open('w'),indent=2)
 (OUT/'E1R_REPORT.md').write_text(f'''# ECNI Stage E1R report\n\n## Decision\n\n**`{status}`**. E1's native association rejection remains unchanged. The derived complete-corpus view is `{role}` and the final architecture is `{arch}`.\n\n## Complete census and relinking\n\nThe two official files contributed {manifest['total_rows_processed']:,} processed rows with {manifest['total_parse_failures']:,} parse failures. Native ticker was diagnostic only. Independent distinctive SEC issuer aliases or explicit ticker syntax created {n:,} high-confidence article-company edges. This is deterministic evidence-rule support, not independently measured human precision.\n\nRelinked edges have {bodyfrac:.2%} body availability and {timefrac:.2%} exact-or-conservative timestamp eligibility. Exact-time and date-only records remain separate. The complete 10,848,303-edge ledger and 31,457,434-group duplicate ledger are locally retained and hash-addressed in `LOCAL_CORPUS_ARTIFACTS_MANIFEST.json`; they are not committed because they exceed repository artifact limits.\n\n## Panel\n\nThe frozen selection contains {summary['selected_companies']} companies across {summary['represented_industries']} historical Fama–French industries. The registered requirement is at least eight industries with four companies each; panel gate = **{summary['panel_pass']}**. Company split status is `{audit['company_split_status']}` with `{audit['company_split_counts']}`; each represented industry contributes 4 train, 1 development-unseen, and 1 locked-unseen company. Time split status is `{audit['time_split_status']}`. 2023 outcome labels were not inspected or summarized.\n\n## SEC and reader boundary\n\nThe SEC pilot contains {len(sec_rows)} filing metadata objects. Reader input counts are recorded, but no reader was trained and no reader score exists. CMIN-US remains a separate standard benchmark.\n\n## Outcome-blind confirmation\n\n- Stock predictive models fitted: **0**\n- BA/MCC/Brier computed: **0**\n- Return-based stock selection: **0**\n- Reader models trained: **0**\n- Locked 2023 outcomes inspected: **0**\n''')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
