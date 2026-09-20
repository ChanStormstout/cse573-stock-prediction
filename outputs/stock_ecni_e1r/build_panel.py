#!/usr/bin/env python3
"""Outcome-blind E1R coverage, historical-SIC validation, panel, and audits."""
from __future__ import annotations
import bisect,csv,hashlib,io,json,math,re,time,urllib.request,zipfile
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/stock_ecni_e1r';PRIV=ROOT/'work/stock-data/ecni_e1r';RAW=PRIV/'raw';PVT=PRIV/'private'
EDGES=OUT/'FNSPID_RELINKED_ENTITY_EDGES.parquet'; IDENT=OUT/'SEC_ISSUER_IDENTITY_HISTORY.csv'; PRICEZIP=ROOT/'work/stock-data/ecni_e1/raw/full_history.zip';SEED='57320260919'

def H(s):return hashlib.sha256(s.encode()).hexdigest()
def write_csv(p,rows,fields=None):
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def parse_ff12():
 text=zipfile.ZipFile(RAW/'Siccodes12.zip').read(zipfile.ZipFile(RAW/'Siccodes12.zip').namelist()[0]).decode()
 groups={};cur=None
 for line in text.splitlines():
  m=re.match(r'\s*(\d+)\s+(\w+)\s+(.*)',line)
  if m:cur=(int(m.group(1)),m.group(2),m.group(3).strip());groups[cur]=[];continue
  m=re.match(r'\s*(\d{4})-(\d{4})',line)
  if m and cur:groups[cur].append((int(m.group(1)),int(m.group(2))))
 def map_sic(s):
  try:s=int(s)
  except:return ('UNKNOWN','Unknown')
  for (num,code,desc),ranges in groups.items():
   if any(a<=s<=b for a,b in ranges):return (code,desc)
  return ('Other','Other')
 return map_sic,groups

def price_sessions(tickers):
 out={}
 z=zipfile.ZipFile(PRICEZIP); names=set(z.namelist())
 for t in tickers:
  n=f'full_history/{t}.csv'
  if n not in names:continue
  dates=[]
  for r in csv.DictReader(io.TextIOWrapper(z.open(n),encoding='utf-8-sig',errors='replace')):
   d=r.get('date','')
   if '2018-01-01'<=d<='2023-12-31':dates.append(d)
  out[t]=sorted(set(dates))
 return out

def effective_session(recorded,tc,sessions):
 if not recorded or not sessions:return None
 d=recorded[:10]
 if tc=='DATE_ONLY_CONSERVATIVE':
  i=bisect.bisect_right(sessions,d);return sessions[i] if i<len(sessions) else None
 if tc!='EXACT_INTRADAY_USABLE':return None
 try:dt=datetime.fromisoformat(recorded.replace(' UTC','+00:00').replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York'))
 except:return None
 ld=dt.date().isoformat(); cutoff=(9,25)
 if ld in sessions and (dt.hour,dt.minute)<=cutoff:return ld
 i=bisect.bisect_right(sessions,ld);return sessions[i] if i<len(sessions) else None

def sec_json(cik,z,zn):
 n=f'CIK{cik:010d}.json';return json.loads(z.read(n)) if n in zn else None

def filing_sic(cik,acc,cache):
 p=cache/f'{acc}.head';
 if not p.exists():
  compact=acc.replace('-','');url=f'https://www.sec.gov/Archives/edgar/data/{cik}/{compact}/{acc}.txt'
  req=urllib.request.Request(url,headers={'User-Agent':'ASU CSE573 academic audit victor@example.edu','Range':'bytes=0-131071'})
  try:
   with urllib.request.urlopen(req,timeout=30) as r:p.write_bytes(r.read(131072))
  except Exception:return None
  time.sleep(.11)
 text=p.read_text(errors='replace')
 m=re.search(r'STANDARD INDUSTRIAL CLASSIFICATION:.*?\[(\d{4})\]',text,re.I)
 return int(m.group(1)) if m else None

def main():
 con=duckdb.connect();ep=str(EDGES)
 # Safe aggregates. Distinct evidence dates are remapped below.
 agg=con.execute(f"""SELECT CIK, resolved_ticker,
 count(*) article_count, count(DISTINCT publisher_hash) FILTER (WHERE publisher_hash<>'') publisher_count,
 sum(CASE WHEN body_available THEN 1 ELSE 0 END) body_available_record_count,
 sum(CASE WHEN NOT body_available THEN 1 ELSE 0 END) title_only_record_count,
 sum(CASE WHEN available_time_class='EXACT_INTRADAY_USABLE' THEN 1 ELSE 0 END) exact_records,
 sum(CASE WHEN available_time_class='DATE_ONLY_CONSERVATIVE' THEN 1 ELSE 0 END) date_only_records,
 count(DISTINCT coalesce(nullif(url_hash,''),nullif(title_hash,''),record_id_hash)) unique_duplicate_group_count,
 min(substr(recorded_time,1,10)) news_first_date,max(substr(recorded_time,1,10)) news_last_date
 FROM read_parquet('{ep}') WHERE relation_type IN ('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE') AND substr(recorded_time,1,10) BETWEEN '2018-01-01' AND '2021-12-31'
 AND available_time_class IN ('EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE') GROUP BY CIK,resolved_ticker""").fetchdf()
 drows=con.execute(f"""SELECT DISTINCT CIK,resolved_ticker,recorded_time,available_time_class FROM read_parquet('{ep}')
 WHERE relation_type IN ('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE') AND substr(recorded_time,1,10) BETWEEN '2018-01-01' AND '2021-12-31' AND available_time_class IN ('EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE')""").fetchall()
 ids={int(r['CIK']):r for r in csv.DictReader(IDENT.open())}; tickers={str(x) for x in agg['resolved_ticker'].tolist()};sessions=price_sessions(tickers)
 days=defaultdict(set); exactdays=defaultdict(set); datedays=defaultdict(set)
 for cik,t,recorded,tc in drows:
  e=effective_session(recorded,tc,sessions.get(t,[]))
  if e and e<='2021-12-31':days[cik].add(e);(exactdays if tc=='EXACT_INTRADAY_USABLE' else datedays)[cik].add(e)
 map_sic,groups=parse_ff12();coverage=[]
 for x in agg.to_dict('records'):
  cik=int(x['CIK']);t=x['resolved_ticker']; ident=ids.get(cik,{});sic=ident.get('SEC_current_SIC','');industry,idesc=map_sic(sic);td=sum('2018-01-01'<=d<='2021-12-31' for d in sessions.get(t,[]))
  n=int(x['article_count']);coverage.append({'CIK':cik,'company_id':f'CIK{cik:010d}','ticker':t,'issuer_name':ident.get('canonical_issuer_name',''),'provisional_FF12_industry':industry,'provisional_FF12_description':idesc,
   'training_days':td,'days_with_any_eligible_news':len(days[cik]),'days_with_exact_time_news':len(exactdays[cik]),'days_with_date_only_news':len(datedays[cik]),
   'news_coverage_fraction':len(days[cik])/td if td else 0,'article_count':n,'unique_duplicate_group_count':int(x['unique_duplicate_group_count']),
   'publisher_count':int(x['publisher_count']),'body_available_record_count':int(x['body_available_record_count']),'title_only_record_count':int(x['title_only_record_count']),
   'exact_time_fraction':int(x['exact_records'])/n if n else 0,'date_only_fraction':int(x['date_only_records'])/n if n else 0,'news_first_date':x['news_first_date'],'news_last_date':x['news_last_date']})
 # Provisional tiers, then filing-time SIC validation for six candidates per tier/industry.
 byind=defaultdict(list)
 for r in coverage:
  if r['training_days']>=1000 and r['provisional_FF12_industry'] not in ('UNKNOWN','Other'):byind[r['provisional_FF12_industry']].append(r)
 candidates=[]
 for ind,rs in byind.items():
  rs.sort(key=lambda r:(r['news_coverage_fraction'],r['CIK']))
  n=len(rs)
  for i,r in enumerate(rs):r['coverage_tier']='LOW' if i<n/3 else ('MEDIUM' if i<2*n/3 else 'HIGH')
  for tier in ['LOW','MEDIUM','HIGH']:
   tierrows=[r for r in rs if r['coverage_tier']==tier]
   tierrows.sort(key=lambda r:(-r['training_days'],-(datetime.fromisoformat(r['news_last_date'])-datetime.fromisoformat(r['news_first_date'])).days,H(str(r['CIK'])+'|ECNI_E1R_PANEL|'+SEED)))
   candidates.extend(tierrows[:6])
 z=zipfile.ZipFile(RAW/'submissions.zip');zn=set(z.namelist());cache=PVT/'sec_headers';cache.mkdir(exist_ok=True)
 industry_rows=[];validated={}
 for r in candidates:
  cik=r['CIK'];x=sec_json(cik,z,zn);obs=[];filings=[]
  if x:
   rec=x.get('filings',{}).get('recent',{});forms=rec.get('form',[])
   for idx,form in enumerate(forms):
    fd=rec['filingDate'][idx] if idx<len(rec.get('filingDate',[])) else ''
    if form in ('10-K','10-Q','20-F','40-F') and '2018-01-01'<=fd<='2021-12-31':filings.append((fd,rec['accessionNumber'][idx],form))
  picks=[]
  if filings:
   filings.sort();picks=[filings[0],filings[-1]] if len(filings)>1 else [filings[0]]
  for fd,acc,form in picks:
   sic=filing_sic(cik,acc,cache);ind,desc=map_sic(sic)
   obs.append({'filing_date':fd,'accession':acc,'form':form,'SIC':sic,'FF12':ind})
  inds={q['FF12'] for q in obs if q['SIC'] is not None};stable=len(obs)>=1 and len(inds)==1 and 'UNKNOWN' not in inds
  status='INDUSTRY_STABLE' if stable else ('INDUSTRY_UNSTABLE' if len(inds)>1 else 'INDUSTRY_UNRESOLVED')
  final=next(iter(inds)) if stable else 'UNRESOLVED';validated[cik]=(status,final,obs)
  industry_rows.append({'CIK':cik,'ticker':r['ticker'],'issuer_name':r['issuer_name'],'training_period_SIC_evidence':json.dumps(obs,separators=(',',':')),
   'derived_FF12_industry':final,'industry_status':status,'source':'SEC filing submission headers accepted 2018-2021','mapping_source':'Ken French SICcodes12.zip','mapping_sha256':'d801141acf039f2e06e6d4d9ba2b3992e9747a1d82fabd53ef21da4a3af79fff'})
 write_csv(OUT/'HISTORICAL_INDUSTRY_LEDGER.csv',industry_rows)
 # Select 2/tier where validated filing industry equals provisional, fallback 4 industry minimum.
 selected=[];allpanel=[]
 for ind,rs in byind.items():
  val=[]
  for r in rs:
   st,final,obs=validated.get(r['CIK'],('NOT_VALIDATED','',[]));r['industry_status']=st;r['historical_industry']=final
   eligible=st=='INDUSTRY_STABLE' and final==ind
   r['panel_eligible']=eligible
   if eligible:val.append(r)
  picks=[]
  for tier in ['LOW','MEDIUM','HIGH']:
   tr=[r for r in val if r['coverage_tier']==tier];tr.sort(key=lambda r:(-r['training_days'],-(datetime.fromisoformat(r['news_last_date'])-datetime.fromisoformat(r['news_first_date'])).days,H(str(r['CIK'])+'|ECNI_E1R_PANEL|'+SEED)))
   picks+=tr[:2]
  if len(picks)<4:
   rest=[r for r in val if r not in picks];rest.sort(key=lambda r:H(str(r['CIK'])+'|ECNI_E1R_PANEL|'+SEED));picks+=(rest[:4-len(picks)])
  if len(picks)>=4:selected+=picks
  pset={r['CIK'] for r in picks}
  for r in rs:
   allpanel.append({**r,'selection_status':'SELECTED' if r['CIK'] in pset else 'EXCLUDED','selection_reason':'deterministic industry/tier rule' if r['CIK'] in pset else ('industry validation/eligibility failed' if not r.get('panel_eligible') else 'not highest deterministic priority within tier')})
 # Public coverage with historical fields.
 for r in coverage:
  rr=next((p for p in allpanel if p['CIK']==r['CIK']),None);r['coverage_tier']=rr.get('coverage_tier','UNASSIGNED') if rr else 'UNASSIGNED';r['historical_industry']=rr.get('historical_industry','UNRESOLVED') if rr else 'UNRESOLVED';r['industry_status']=rr.get('industry_status','NOT_VALIDATED') if rr else 'NOT_VALIDATED'
 write_csv(OUT/'RELINKED_TRAINING_NEWS_COVERAGE.csv',coverage)
 write_csv(OUT/'FROZEN_STOCK_PANEL_E1R.csv',allpanel)
 represented=Counter(r['historical_industry'] for r in selected);panel_pass=len(represented)>=8 and min(represented.values(),default=0)>=4
 # Company split with 4/1/1 companies per represented industry.  The held-out
 # tiers rotate deterministically across industries so coverage tiers remain
 # balanced globally even though each industry/tier cell contains only two
 # companies (and therefore cannot itself contain all three split roles).
 split={'TRAIN_COMPANIES':[],'DEV_UNSEEN_COMPANIES':[],'LOCKED_UNSEEN_COMPANIES':[]}
 by_selected_industry=defaultdict(list)
 for r in selected:by_selected_industry[r['historical_industry']].append(r)
 tiers=['LOW','MEDIUM','HIGH']
 for industry,rs in sorted(by_selected_industry.items()):
  tier_rows={t:sorted((r for r in rs if r['coverage_tier']==t),key=lambda r:H(str(r['CIK'])+'|ECNI_E1R_COMPANY_SPLIT|'+SEED)) for t in tiers}
  rotation=int(H(industry+'|ECNI_E1R_SPLIT_TIER|'+SEED),16)%len(tiers)
  dev_tier=tiers[rotation];locked_tier=tiers[(rotation+1)%len(tiers)]
  dev=tier_rows[dev_tier][0];locked=tier_rows[locked_tier][0]
  for r in rs:
   bucket='DEV_UNSEEN_COMPANIES' if r is dev else ('LOCKED_UNSEEN_COMPANIES' if r is locked else 'TRAIN_COMPANIES')
   split[bucket].append(r['company_id'])
 json.dump({'status':'FROZEN' if panel_pass else 'NOT_FROZEN_PANEL_GATE_FAILED','hash_rule':'Per-industry 4/1/1; held-out coverage tiers rotate by SHA256(industry|ECNI_E1R_SPLIT_TIER|57320260919); company within tier by SHA256(CIK|ECNI_E1R_COMPANY_SPLIT|57320260919)','stratified_by':['historical_industry','coverage_tier'],'expected_counts':{'TRAIN_COMPANIES':44,'DEV_UNSEEN_COMPANIES':11,'LOCKED_UNSEEN_COMPANIES':11},**split},(OUT/'COMPANY_SPLITS_E1R.json').open('w'),indent=2)
 json.dump({'train_forward_validation':['2018-01-01','2021-12-31'],'development':['2022-01-01','2022-12-31'],'locked_test':['2023-01-01','2023-12-31'],'locked_outcome_status':'NOT_INSPECTED_OR_SUMMARIZED','status':'FROZEN' if panel_pass else 'CONDITIONAL_ON_PANEL'},(OUT/'TIME_SPLIT_E1R.json').open('w'),indent=2)
 # Deterministic audit sample, no claim of human precision.
 audit=con.execute(f"""WITH x AS (SELECT *,substr(recorded_time,1,4) audit_year,(native_ticker=resolved_ticker) native_agreement,
 row_number() OVER (PARTITION BY source_file,substr(recorded_time,1,4),relation_type,(native_ticker=resolved_ticker),body_available ORDER BY sha256(record_id_hash||company_id||relation_type||'{SEED}')) rn
 FROM read_parquet('{ep}')), s AS (SELECT * EXCLUDE(rn) FROM x WHERE rn<=5)
 SELECT * FROM s ORDER BY sha256(record_id_hash||company_id||relation_type||'{SEED}') LIMIT 600""").fetchdf()
 audit.to_csv(PVT/'entity_relink_locked_review.csv',index=False)
 ac=Counter(audit['relation_type']);rules=Counter(audit['confidence_rule']);agree=sum(audit['native_ticker']==audit['resolved_ticker'])
 high=sum(ac[x] for x in ('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE'))
 (OUT/'ENTITY_RELINK_AUDIT.md').write_text(f'''# Entity relink audit\n\nA deterministic locked set of 600 edge candidates was frozen at `{H((PVT/'entity_relink_locked_review.csv').read_text())}`. It spans the complete-census output and is stored privately without article bodies.\n\n- Relation counts: `{dict(ac)}`\n- High-confidence candidates: `{high}/600`; indirect/competitor controls: `{600-high}/600`.\n- Rule counts: `{dict(rules)}`\n- Native-tag agreement: `{agree}/600`; disagreement: `{600-agree}/600`.\n- Every sampled candidate has an explicit distinctive SEC alias or explicit exchange/dollar-ticker rule; only the two registered high-confidence relation classes enter panel coverage.\n\nThis is an **algorithmic evidence-rule support audit**, not independent human gold and not measured precision. Native FNSPID association quality remains the original E1 result.\n''')
 # Duplicate summaries from complete private metadata.
 mp=str(PVT/'fnspid_complete_metadata.parquet')
 q=f"""WITH base AS (SELECT * FROM read_parquet('{mp}')), x AS (
 SELECT 'EXACT_URL' typ,url_hash val,recorded_time,publisher_hash FROM base WHERE url_hash<>'' UNION ALL
 SELECT 'EXACT_TITLE',normalized_title_hash,recorded_time,publisher_hash FROM base WHERE normalized_title_hash<>'' UNION ALL
 SELECT 'EXACT_BODY',body_hash,recorded_time,publisher_hash FROM base WHERE body_hash<>'' UNION ALL
 SELECT 'NEAR_TITLE_CANDIDATE',near_title_fingerprint,recorded_time,publisher_hash FROM base WHERE near_title_fingerprint<>'')
 SELECT typ group_type,sha256(typ||':'||val) duplicate_group_id,count(*) article_count,count(DISTINCT publisher_hash) independent_publisher_count,min(recorded_time) first_observation,max(recorded_time) latest_observation,date_diff('day',try_cast(substr(min(recorded_time),1,10) AS DATE),try_cast(substr(max(recorded_time),1,10) AS DATE)) time_spread_days,'RECONSTRUCT_BY_RECORD_HASH_JOIN_TO_EDGE_DATASET' company_membership_reference FROM x GROUP BY typ,val HAVING count(*)>1"""
 dup=con.execute(q).fetchdf();dup.to_csv(OUT/'DUPLICATE_GROUP_SUMMARY.csv',index=False,lineterminator='\n')
 # Exact event eligibility audit.
 exact=con.execute(f"SELECT count(*) records,count(DISTINCT record_id_hash) articles,count(DISTINCT CIK) companies,sum(CASE WHEN body_available THEN 1 ELSE 0 END) body_edges FROM read_parquet('{ep}') WHERE relation_type IN ('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE') AND available_time_class='EXACT_INTRADAY_USABLE'").fetchone()
 json.dump({'task':'EXACT_TIME_EVENT_CONDITIONED','horizon':'NOT_SELECTED_IN_E1R','edge_records':exact[0],'unique_articles':exact[1],'companies':exact[2],'body_edges':exact[3],'outcome_statistics_computed':0},(OUT/'EXACT_TIME_EVENT_CORPUS_AUDIT.json').open('w'),indent=2)
 # Reader counts are purely candidate/support counts.
 bodies=con.execute(f"SELECT count(*),sum(CASE WHEN body_available THEN 1 ELSE 0 END),count(DISTINCT record_id_hash) FROM read_parquet('{ep}') WHERE relation_type IN ('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE')").fetchone()
 json.dump({'high_confidence_edges':bodies[0],'body_edges':bodies[1],'unique_news_articles':bodies[2],'SEC_backed_pairs':0,'news_only_edges':bodies[0],
  'category_support_status':'NOT_MEASURED_WITHOUT_OUTCOME_BLIND_HUMAN_RELATION_LABELS','reader_training_authorized':False,'reader_scores_computed':0},(OUT/'READER_PILOT_INPUT_AUDIT.json').open('w'),indent=2)
 # Outcome status.
 result={'panel_pass':panel_pass,'selected_companies':len(selected),'represented_industries':dict(represented),'minimum_companies_per_industry':min(represented.values(),default=0),
  'complete_coverage_companies':len(coverage),'historical_industry_candidates_checked':len(industry_rows)}
 json.dump(result,(PVT/'panel_build_summary.json').open('w'),indent=2);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
