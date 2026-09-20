#!/usr/bin/env python3
"""Outcome-blind E1R-V2 semantic review pack and SEC coverage audit."""
from __future__ import annotations
import csv, hashlib, json, math, re, sys, time, urllib.request, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import duckdb

csv.field_size_limit(sys.maxsize)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/stock_ecni_e1rv2'; E1R=ROOT/'outputs/stock_ecni_e1r'
PRIV=ROOT/'work/stock-data/ecni_e1rv2'; PVT=PRIV/'private'; RAW=ROOT/'work/stock-data/ecni_e1r/raw'
EDGE=E1R/'FNSPID_RELINKED_ENTITY_EDGES.parquet'; META=ROOT/'work/stock-data/ecni_e1r/private/fnspid_complete_metadata.parquet'
SEED='ECNI_E1RV2_ENTITY_REVIEW|57320260919'
HIGH=('DIRECT_TARGET_HIGH_CONFIDENCE','MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE')
FILES={'all_external':RAW/'All_external.csv','nasdaq':RAW/'nasdaq_exteral_data.csv'}
PROTECTED={
 'E1R_FINAL_AUDIT.json':'b28114b895ecbe7ef3ace8e42279e9271e5e94165922ac168796b9381c669705',
 'FROZEN_STOCK_PANEL_E1R.csv':'7c5a1a1d618f76a9801cd57a05f531c1a2885674500e11c93e1697d903559948',
 'COMPANY_SPLITS_E1R.json':'7ae03321d0ede4e444a90b89cede0d26291eac756eac2aaeb757f9035fb28f34',
 'TIME_SPLIT_E1R.json':'53ed35ad21ff462fa6913db1d1acaa41126771d6c355e9e4f77c572c40acba3c'}

KNOWN_EXCHANGE={'nasdaq','new york stock exchange','nyse','cboe','chicago mercantile exchange','intercontinental exchange','american stock exchange'}
KNOWN_INDEX_MARKET={'dow','dow jones','s p','s p global','russell','market','markets','stock market'}
GEOGRAPHIC={'china','america','american','national','united states','europe','asia','africa','canada','mexico','brazil','india','japan'}
COMMON_DUAL={'apple','target','oracle','gap','block','square','snap','zoom','visa','discover','ally','unity','affirm','toast','mosaic','monster','pioneer','frontier','crown','diamond','first','general','global','national'}
LISTING=re.compile(r'\b(trades?|traded|trading|listed|listing)\s+(?:on|at)\s+(?:the\s+)?nasdaq\b|\bnasdaq[- ]listed\b|\bnasdaq\s+(?:exchange|market|composite|index)\b',re.I)
CORPORATE_NDAQ=re.compile(r'\bnasdaq\s*,?\s+inc\b|\bnasdaq(?:\'s)?\s+(?:announc|report|acquir|launch|appoint|revenue|earnings)',re.I)

def H(s): return hashlib.sha256(str(s).encode('utf-8','replace')).hexdigest()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def norm(s): return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()
def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def write_csv(path,rows,fields=None):
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields: fields.append(k)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows)
def jsonl(path,rows):
 with path.open('w',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def protected_ok(): return all(digest(E1R/n)==v for n,v in PROTECTED.items())

def base_data():
 panel=list(csv.DictReader((E1R/'FROZEN_STOCK_PANEL_E1R.csv').open()))
 selected=[r for r in panel if r['selection_status']=='SELECTED']
 split=json.load((E1R/'COMPANY_SPLITS_E1R.json').open())
 split_of={x:k for k in ('TRAIN_COMPANIES','DEV_UNSEEN_COMPANIES','LOCKED_UNSEEN_COMPANIES') for x in split[k]}
 identities={int(r['CIK']):r for r in csv.DictReader((E1R/'SEC_ISSUER_IDENTITY_HISTORY.csv').open())}
 coverage={int(r['CIK']):r for r in csv.DictReader((E1R/'RELINKED_TRAINING_NEWS_COVERAGE.csv').open())}
 return selected,split,split_of,identities,coverage

def alias_aggregates(con,coverage):
 ep=str(EDGE)
 q=f"""WITH b AS (SELECT *, substr(recorded_time,1,10) record_date FROM read_parquet('{ep}'))
 SELECT CIK,resolved_ticker,matched_alias,
 count(*) matched_articles,
 sum(CASE WHEN matched_field='title' THEN 1 ELSE 0 END) title_matches,
 sum(CASE WHEN matched_field='body' THEN 1 ELSE 0 END) body_matches,
 sum(CASE WHEN matched_field='explicit_ticker' THEN 1 ELSE 0 END) exact_ticker_matches,
 sum(CASE WHEN relation_type='DIRECT_TARGET_HIGH_CONFIDENCE' THEN 1 ELSE 0 END) direct_high_confidence_edges,
 sum(CASE WHEN relation_type='MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE' THEN 1 ELSE 0 END) multi_company_high_confidence_edges,
 sum(CASE WHEN relation_type IN {HIGH} AND record_date BETWEEN '2018-01-01' AND '2021-12-31' THEN 1 ELSE 0 END) training_high_confidence_edges,
 count(DISTINCT CASE WHEN relation_type IN {HIGH} AND record_date BETWEEN '2018-01-01' AND '2021-12-31' THEN record_date END) training_distinct_recorded_dates,
 count(DISTINCT publisher_hash) FILTER (WHERE publisher_hash<>'') publisher_count,
 avg(CASE WHEN native_ticker=resolved_ticker THEN 1.0 ELSE 0.0 END) native_tag_agreement_rate
 FROM b GROUP BY CIK,resolved_ticker,matched_alias"""
 rows=con.execute(q).fetchdf().to_dict('records')
 pubs=con.execute(f"""WITH p AS (SELECT CIK,matched_alias,publisher_hash,count(*) n FROM read_parquet('{ep}') WHERE publisher_hash<>'' GROUP BY ALL),
 t AS (SELECT CIK,matched_alias,sum(n) total,max(n) top_n,arg_max(publisher_hash,n) top_hash FROM p GROUP BY CIK,matched_alias)
 SELECT p.CIK,p.matched_alias,t.top_hash,t.top_n*1.0/t.total top_publisher_fraction,
 -sum((p.n*1.0/t.total)*ln(p.n*1.0/t.total)) publisher_entropy FROM p JOIN t USING(CIK,matched_alias) GROUP BY p.CIK,p.matched_alias,t.top_hash,t.top_n,t.total""").fetchall()
 pmap={(int(c),a):(h,float(fr),float(en)) for c,a,h,fr,en in pubs}
 vals=sorted(int(r['matched_articles']) for r in rows); q99=vals[max(0,math.ceil(.99*len(vals))-1)]
 for r in rows:
  cik=int(r['CIK']);r['CIK']=cik
  ph,pf,pe=pmap.get((cik,r['matched_alias']),('',0.0,0.0));r['top_publisher_hash']=ph;r['top_publisher_fraction']=pf;r['publisher_entropy']=pe
  td=int(coverage.get(cik,{}).get('training_days') or 0);days=int(r['training_distinct_recorded_dates'])
  r['training_days_denominator']=td;r['training_day_coverage']=min(days,td)/td if td else 0.0
  r['_q99']=q99
 return rows,q99

def initial_flags(row,q99):
 a=norm(row['matched_alias']);flags=[]
 if a in KNOWN_EXCHANGE or any(x in a for x in ('stock exchange','nasdaq')):flags.append('EXCHANGE_OR_VENUE_NAME')
 if a in KNOWN_INDEX_MARKET:flags.append('INDEX_OR_MARKET_NAME')
 if a in GEOGRAPHIC:flags.append('GEOGRAPHIC_ENTITY')
 if len(a.split())==1:flags.append('SINGLE_TOKEN_SEMANTIC_RISK')
 if a in COMMON_DUAL:flags.append('COMMON_WORD_PRODUCT_OR_CATEGORY')
 if int(row['matched_articles'])>=q99:flags.append('EDGE_COUNT_TOP_1_PERCENT')
 if row['training_day_coverage']>=.90 and int(row['matched_articles'])>=1000:flags.append('IMPLAUSIBLY_HIGH_TRAINING_DAY_COVERAGE')
 if row['top_publisher_fraction']>=.80 and int(row['matched_articles'])>=100:flags.append('PUBLISHER_CONCENTRATION_GE_80_PERCENT')
 if a=='nasdaq':flags.append('DUAL_ROLE_EXCHANGE_COMPANY_SENTINEL')
 return flags

def candidate_rows(con,selected):
 ep=str(EDGE);mp=str(META);ciks=','.join(str(int(r['CIK'])) for r in selected)
 common=f"""SELECT e.*,m.row_number,m.year,(e.native_ticker=e.resolved_ticker) native_agreement,
 sha256(e.record_id_hash||e.company_id||e.matched_alias||'{SEED}') sample_hash
 FROM read_parquet('{ep}') e JOIN read_parquet('{mp}') m USING(record_id_hash)
 WHERE e.relation_type IN {HIGH}"""
 panel=con.execute(f"""WITH x AS ({common} AND e.CIK IN ({ciks})), y AS
 (SELECT *,row_number() OVER(PARTITION BY CIK,year,source_file,matched_field,native_agreement,relation_type ORDER BY sample_hash) rn FROM x)
 SELECT * EXCLUDE(rn) FROM y WHERE rn<=2""").fetchdf().to_dict('records')
 sentinel=con.execute(f"""WITH x AS ({common} AND e.resolved_ticker='NDAQ'), y AS
 (SELECT *,row_number() OVER(PARTITION BY year,source_file,matched_field,native_agreement,relation_type ORDER BY sample_hash) rn FROM x)
 SELECT * EXCLUDE(rn) FROM y WHERE rn<=5""").fetchdf().to_dict('records')
 alias_one=con.execute(f"""WITH x AS ({common}), y AS
 (SELECT *,row_number() OVER(PARTITION BY CIK,matched_alias ORDER BY sample_hash) rn FROM x)
 SELECT * EXCLUDE(rn) FROM y WHERE rn=1""").fetchdf().to_dict('records')
 return panel,sentinel,alias_one

def choose_panel(candidates,selected):
 by=defaultdict(list)
 for x in candidates:by[int(x['CIK'])].append(x)
 out=[]
 for company in selected:
  cik=int(company['CIK']);pool=sorted(by[cik],key=lambda x:x['sample_hash']);chosen=[];covered=set()
  while pool and len(chosen)<5:
   def tokens(x):return {f"field:{x['matched_field']}",f"agree:{bool(x['native_agreement'])}",f"relation:{x['relation_type']}"}
   best=min(pool,key=lambda x:(-len(tokens(x)-covered),x['sample_hash']))
   chosen.append(best);covered|=tokens(best);pool=[x for x in pool if x['record_id_hash']!=best['record_id_hash']]
  for x in chosen:x['_sample_roles']={'PANEL_BASE'}
  out+=chosen
 return out

def choose_sentinel(candidates,n=100):
 pool=sorted(candidates,key=lambda x:x['sample_hash']);chosen=[];seen=set()
 for x in pool:
  st=(int(x['year']),x['source_file'],x['matched_field'],bool(x['native_agreement']))
  if st not in seen:chosen.append(x);seen.add(st)
  if len(chosen)>=n:break
 used={x['record_id_hash'] for x in chosen}
 for x in pool:
  if x['record_id_hash'] not in used:chosen.append(x);used.add(x['record_id_hash'])
  if len(chosen)>=n:break
 chosen=chosen[:n]
 for x in chosen:x.setdefault('_sample_roles',set()).add('NDAQ_SENTINEL')
 return chosen

def scan_raw(wanted):
 # wanted maps (source,row_number) to one or more edge candidates.
 cache=PVT/'selected_raw_context_cache.jsonl';pcache=PVT/'normalized_publisher_names.json'
 if cache.exists() and pcache.exists():
  found={}
  for line in cache.open():
   r=json.loads(line);found[(r.pop('record_id_hash'),r.pop('company_id'),r.pop('matched_alias'))]=r
  return found,set(json.load(pcache.open())),{}
 found={};publisher_names=set();publisher_hash_to_name={}
 for tag,path in FILES.items():
  target={i for (s,i) in wanted if s==tag}; maxrow=max(target) if target else 0
  with path.open(encoding='utf-8',errors='replace',newline='') as f:
   for i,r in enumerate(csv.DictReader(f),1):
    pub=r.get('Publisher') or '';pn=norm(pub)
    if pn:publisher_names.add(pn);publisher_hash_to_name.setdefault(H(pub),pub)
    if i in target:
     for x in wanted[(tag,i)]:
      field=x['matched_field'];alias=str(x['matched_alias']);title=r.get('Article_title') or '';body=r.get('Article') or ''
      txt=title if field=='title' else body
      nt=norm(txt);pos=nt.find(norm(alias));context=nt[max(0,pos-260):pos+len(norm(alias))+260] if pos>=0 else nt[:520]
      key=(x['record_id_hash'],x['company_id'],alias)
      found[key]={'title':title,'evidence_context_normalized':context,'publisher':pub,'recorded_time':r.get('Date') or ''}
    if i%1000000==0:print('raw_scan',tag,i,'wanted_found',len(found),flush=True)
    # Continue the full file to collect publisher names even after target rows.
  print('raw_scan_complete',tag,'cards',len(found),'publishers',len(publisher_names),flush=True)
 with cache.open('w') as f:
  for (record,company,alias),r in found.items():f.write(json.dumps({'record_id_hash':record,'company_id':company,'matched_alias':alias,**r},ensure_ascii=False,separators=(',',':'))+'\n')
 write_json(pcache,sorted(publisher_names))
 return found,publisher_names,publisher_hash_to_name

def make_cards(panel,sentinel,alias_one,alias_rows,q99):
 allc={}
 for x in panel+sentinel+alias_one:
  key=(x['record_id_hash'],x['company_id'],str(x['matched_alias']))
  if key not in allc:allc[key]=x
  allc[key].setdefault('_sample_roles',set()).update(x.get('_sample_roles',set()))
 wanted=defaultdict(list)
 for x in allc.values():wanted[(x['source_file'],int(x['row_number']))].append(x)
 raw,publisher_names,pubmap=scan_raw(wanted)
 # Finalize risk flags now that publisher/source-name overlap is known.
 for r in alias_rows:
  flags=initial_flags(r,q99)
  if norm(r['matched_alias']) in publisher_names:flags.append('PUBLISHER_OR_SOURCE_NAME_OVERLAP')
  r['risk_flags']='|'.join(sorted(set(flags))) if flags else 'NONE'
  r['risk_flag_count']=len(set(flags))
 risk_aliases=sorted(alias_rows,key=lambda r:(-r['risk_flag_count'],-r['training_day_coverage'],-int(r['matched_articles']),H(f"{r['CIK']}|{r['matched_alias']}|{SEED}")))
 top=risk_aliases[:100]
 if not any(int(r['CIK'])==1120193 and norm(r['matched_alias'])=='nasdaq' for r in top):
  nd=next(r for r in alias_rows if int(r['CIK'])==1120193 and norm(r['matched_alias'])=='nasdaq');top[-1]=nd
 topkeys={(int(r['CIK']),str(r['matched_alias'])) for r in top}
 amap={(int(x['CIK']),str(x['matched_alias'])):x for x in alias_one}
 risk=[]
 for k in topkeys:
  x=amap[k];x.setdefault('_sample_roles',set()).add('ALIAS_RISK_ENRICHED');risk.append(x)
 final={}
 for x in panel+sentinel+risk:
  k=(x['record_id_hash'],x['company_id'],str(x['matched_alias']))
  if k not in final:final[k]=x
  final[k].setdefault('_sample_roles',set()).update(x.get('_sample_roles',set()))
 cards=[]
 for x in sorted(final.values(),key=lambda z:(z['resolved_ticker'],z['sample_hash'])):
  key=(x['record_id_hash'],x['company_id'],str(x['matched_alias'])); rr=raw[key]
  cards.append({'card_id':H('|'.join(key)+'|'+SEED),'record_id_hash':x['record_id_hash'],'company_id':x['company_id'],'CIK':int(x['CIK']),'target_ticker':x['resolved_ticker'],'target_company':next((r['canonical_issuer_name'] for r in csv.DictReader((E1R/'SEC_ISSUER_IDENTITY_HISTORY.csv').open()) if int(r['CIK'])==int(x['CIK'])),x['resolved_ticker']),
   'title':rr['title'],'evidence_context_normalized':rr['evidence_context_normalized'],'matched_alias':x['matched_alias'],'matched_field':x['matched_field'],'native_ticker':x['native_ticker'],'publisher':rr['publisher'],'date':rr['recorded_time'],'source_file':x['source_file'],'relation_claim_under_review':x['relation_type'],'sample_roles':sorted(x['_sample_roles']),'semantic_label':None,'independent_reviewer_id':None})
 return cards,alias_rows,top

def sec_audit(selected,split_of):
 z=zipfile.ZipFile(RAW/'submissions.zip');rows=[];events=[];supp_manifest=[];cache=PVT/'sec_supplemental_submissions';cache.mkdir(exist_ok=True)
 forms=['8-K','10-Q','10-K','6-K','20-F']
 for p in selected:
  cik=int(p['CIK']);x=json.loads(z.read(f'CIK{cik:010d}.json'));records=[('SEC_BULK_RECENT',x.get('filings',{}).get('recent',{}))]
  for fm in x.get('filings',{}).get('files',[]):
   if fm.get('filingTo','')<'2018-01-01' or fm.get('filingFrom','')>'2021-12-31':continue
   name=fm['name'];cp=cache/name
   if not cp.exists():
    req=urllib.request.Request('https://data.sec.gov/submissions/'+name,headers={'User-Agent':'ASU CSE573 academic audit victor@example.edu'})
    with urllib.request.urlopen(req,timeout=60) as response:cp.write_bytes(response.read())
    time.sleep(.12)
   supp_manifest.append({'CIK':cik,'ticker':p['ticker'],'file':name,'url':'https://data.sec.gov/submissions/'+name,'bytes':cp.stat().st_size,'sha256':digest(cp),'filingFrom':fm.get('filingFrom'),'filingTo':fm.get('filingTo')})
   records.append((name,json.loads(cp.read_text())))
  filings=[];seen=set()
  for source_name,rec in records:
   for i,form in enumerate(rec.get('form',[])):
    fd=rec.get('filingDate',[])[i] if i<len(rec.get('filingDate',[])) else '';base=form[:-2] if form.endswith('/A') else form
    accession=rec.get('accessionNumber',[])[i] if i<len(rec.get('accessionNumber',[])) else ''
    if base in forms and '2018-01-01'<=fd<='2021-12-31' and accession not in seen:
     seen.add(accession);filings.append({'CIK':cik,'company_id':p['company_id'],'ticker':p['ticker'],'historical_industry':p['historical_industry'],'company_split':split_of[p['company_id']],
      'accession':accession,'form':base,'filed_form':form,'filing_date':fd,'acceptance_time':rec.get('acceptanceDateTime',[])[i] if i<len(rec.get('acceptanceDateTime',[])) else '',
      'period_end':rec.get('reportDate',[])[i] if i<len(rec.get('reportDate',[])) else '', 'document_reference':rec.get('primaryDocument',[])[i] if i<len(rec.get('primaryDocument',[])) else '',
      'document_description':rec.get('primaryDocDescription',[])[i] if i<len(rec.get('primaryDocDescription',[])) else '', 'amendment':form.endswith('/A'),'submission_history_source':source_name})
  fc=Counter(r['form'] for r in filings)
  filer='FOREIGN_PRIVATE_ISSUER' if fc['20-F'] or fc['6-K'] else ('DOMESTIC_REGISTRANT' if fc['10-K'] or fc['10-Q'] or fc['8-K'] else 'OTHER_OR_UNCERTAIN')
  for e in filings:e['filer_class']=filer;e['priority']='EVENT_DISCLOSURE' if e['form'] in ('8-K','6-K') else 'PERIODIC_COMPARABLE_FACTS';events.append(e)
  rows.append({'CIK':cik,'company_id':p['company_id'],'ticker':p['ticker'],'issuer_name':p['issuer_name'],'historical_industry':p['historical_industry'],'company_split':split_of[p['company_id']],'filer_class':filer,
   '8-K_count':fc['8-K'],'10-Q_count':fc['10-Q'],'10-K_count':fc['10-K'],'6-K_count':fc['6-K'],'20-F_count':fc['20-F'],'total_factual_forms':sum(fc.values()),'training_period':'2018-01-01/2021-12-31','source':'SEC submissions bulk recent arrays plus official overlapping supplemental histories','missingness_policy':'ZERO_MEANS_NO_MATCHING_FORM_IN_FROZEN_SOURCE_NOT_NO_FACTUAL_DISCLOSURE'})
 return rows,events,supp_manifest

def main():
 OUT.mkdir(exist_ok=True);PVT.mkdir(parents=True,exist_ok=True)
 if not protected_ok():raise SystemExit('Historical E1R protected artifact hash mismatch')
 selected,split,split_of,identities,coverage=base_data();assert len(selected)==66
 con=duckdb.connect();alias_rows,q99=alias_aggregates(con,coverage)
 panel_candidates,sentinel_candidates,alias_one=candidate_rows(con,selected)
 panel=choose_panel(panel_candidates,selected);assert len(panel)==330
 sentinel=choose_sentinel(sentinel_candidates,100);assert len(sentinel)==100
 cards,alias_rows,toprisk=make_cards(panel,sentinel,alias_one,alias_rows,q99)
 private_cards=PVT/'ENTITY_SEMANTIC_REVIEW_CARDS.jsonl';jsonl(private_cards,cards)
 write_csv(OUT/'ALIAS_RISK_CENSUS.csv',alias_rows,[k for k in alias_rows[0] if not k.startswith('_')])
 panel_counts=Counter(c['target_ticker'] for c in cards if 'PANEL_BASE' in c['sample_roles'])
 sentinel_cards=[c for c in cards if 'NDAQ_SENTINEL' in c['sample_roles']]
 listing=sum(bool(LISTING.search(c['title']+' '+c['evidence_context_normalized'])) for c in sentinel_cards)
 corporate=sum(bool(CORPORATE_NDAQ.search(c['title']+' '+c['evidence_context_normalized'])) for c in sentinel_cards)
 nd=next(r for r in alias_rows if int(r['CIK'])==1120193 and norm(r['matched_alias'])=='nasdaq')
 write_json(OUT/'NDAQ_SENTINEL_AUDIT.json',{'status':'AWAITING_INDEPENDENT_LABELS','CIK':1120193,'ticker':'NDAQ','alias':'nasdaq','complete_edge_diagnostics':{k:(int(nd[k]) if k.endswith(('articles','matches','edges','dates','count','denominator')) else nd[k]) for k in ['matched_articles','title_matches','body_matches','exact_ticker_matches','direct_high_confidence_edges','multi_company_high_confidence_edges','training_high_confidence_edges','training_distinct_recorded_dates','training_days_denominator','training_day_coverage','publisher_count','top_publisher_fraction','native_tag_agreement_rate']},
  'all_NDAQ_explicit_ticker_edges':sum(int(r['exact_ticker_matches']) for r in alias_rows if int(r['CIK'])==1120193),
  'sentinel_cards':len(sentinel_cards),'private_cards_sha256':digest(private_cards),'mechanical_listing_phrase_cards':listing,'mechanical_corporate_context_cards':corporate,'semantic_precision':None,'independent_labels_available':False,'interpretation':'Phrase diagnostics flag possible contamination but do not establish semantic correctness.'})
 manifest={'status':'FROZEN_AWAITING_INDEPENDENT_ENTITY_REVIEW','seed':SEED,'private_path':'work/stock-data/ecni_e1rv2/private/ENTITY_SEMANTIC_REVIEW_CARDS.jsonl','private_sha256':digest(private_cards),'total_unique_cards':len(cards),'panel_base_cards':sum('PANEL_BASE' in c['sample_roles'] for c in cards),'panel_companies_covered':len(panel_counts),'panel_cards_per_company_min':min(panel_counts.values()),'panel_cards_per_company_max':max(panel_counts.values()),'NDAQ_sentinel_cards':len(sentinel_cards),'alias_risk_enriched_cards':sum('ALIAS_RISK_ENRICHED' in c['sample_roles'] for c in cards),'overlap_cards':sum(len(c['sample_roles'])>1 for c in cards),'independent_labels_received':0,'semantic_precision_computed':False,'copyrighted_text_committed':False,
  'sampling_dimensions':['year','source_file','matched_field','native_agreement','single_vs_multi_company_relation'],'card_ids':[c['card_id'] for c in cards]}
 write_json(OUT/'ENTITY_SEMANTIC_REVIEW_MANIFEST.json',manifest)
 (OUT/'ENTITY_SEMANTIC_LABELING_GUIDE.md').write_text('''# Entity semantic labeling guide\n\nReview only the target company, title, evidence context, alias, field, publisher, native tag, and date shown on each blinded card. Do not consult stock prices or outcomes. Assign exactly one label:\n\n- `DIRECT_TARGET`: target company is itself a substantive actor, object, or subject.\n- `MULTI_COMPANY_DIRECT`: target is substantive and the report directly covers multiple companies.\n- `INDIRECT_COMPETITOR_OR_COUNTERPARTY`: target appears only through another entity's event or comparison.\n- `VENUE_OR_MARKET_NAME_ONLY`: alias denotes an exchange, index, listing venue, or market description.\n- `PUBLISHER_OR_SOURCE_NAME_ONLY`: alias occurs only as a publisher/source identity.\n- `INCIDENTAL_MENTION`: target is named without a substantive target event or statement.\n- `NO_TARGET_EVIDENCE`: displayed evidence does not support the target relation.\n- `UNCERTAIN`: available context is insufficient to decide.\n\nFor NDAQ, “shares trade on Nasdaq,” “Nasdaq-listed,” or a Nasdaq index reference is venue/market use, not NASDAQ, Inc. corporate evidence. Do not accept the relinker's prior class as gold. Record reviewer identity independently. The frozen gate is ≥0.90 selected-panel point precision, Wilson lower bound ≥0.85, no company below 0.60 with ≥5 reviews, and NDAQ ≥0.90.\n''')
 # Mechanical report contains only diagnostics, never semantic precision.
 risky=[r for r in alias_rows if r['risk_flag_count']]
 top_lines='\n'.join(f"- `{r['resolved_ticker']}` / `{r['matched_alias']}`: {int(r['matched_articles']):,} matches; `{r['risk_flags']}`" for r in sorted(risky,key=lambda r:(-r['risk_flag_count'],-r['training_day_coverage'],-int(r['matched_articles'])))[:15])
 (OUT/'MECHANICAL_ALIAS_CONTAMINATION_REPORT.md').write_text(f'''# Mechanical alias contamination diagnostics\n\nThese source-only flags are not semantic labels or measured precision. The census contains **{len(alias_rows):,}** alias/CIK rows; **{len(risky):,}** have at least one mechanical risk flag. The global top-1% edge-count threshold is **{q99:,}**. `training_day_coverage` is a bounded source diagnostic: distinct 2018–2021 recorded dates divided by that stock's trading-day denominator and capped at one; the historical panel retains the stricter effective-session coverage calculation.\n\nNDAQ's `nasdaq` alias produced **{int(nd['matched_articles']):,}** all-years matched edges, including **{int(nd['direct_high_confidence_edges'])+int(nd['multi_company_high_confidence_edges']):,}** all-years currently high-confidence edges and **{int(nd['training_high_confidence_edges']):,}** high-confidence edges during 2018–2021. Its bounded training-day coverage diagnostic is **{nd['training_day_coverage']:.3f}**, publisher count is **{int(nd['publisher_count'])}**, and native-tag agreement is **{nd['native_tag_agreement_rate']:.4f}**. In the 100-card sentinel sample, **{listing}** cards match a listing/venue phrase and **{corporate}** match a narrow corporate-context phrase; overlap is possible. This confirms a material contamination risk but cannot establish precision.\n\n## Highest mechanical-risk aliases\n\n{top_lines}\n\nTop risk aliases are recorded in `ALIAS_RISK_CENSUS.csv`. Risk ranking uses only alias form, source concentration, edge volume, and training-period coverage. No market outcome was accessed. The `nasdaq` alias remains pending independent review; no alias rule or complete relation ledger was changed in this stage.\n''')
 # SEC form coverage and factual event metadata.
 sec_rows,events,sec_supp=sec_audit(selected,split_of);write_csv(OUT/'SEC_FORM_COVERAGE_BY_COMPANY.csv',sec_rows);jsonl(OUT/'SEC_FACTUAL_EVENT_CANDIDATES.jsonl',events)
 write_json(OUT/'SEC_SUPPLEMENTAL_SOURCE_MANIFEST.json',{'status':'COMPLETE','official_endpoint':'https://data.sec.gov/submissions/','overlapping_supplemental_files':sec_supp,'file_count':len(sec_supp),'outcomes_accessed':0})
 filer=Counter(r['filer_class'] for r in sec_rows);formtot={f:sum(int(r[f'{f}_count']) for r in sec_rows) for f in ['8-K','10-Q','10-K','6-K','20-F']}
 zeros=sum(int(r['total_factual_forms'])==0 for r in sec_rows)
 totals=[int(r['total_factual_forms']) for r in sec_rows];coverage_ratio=max(totals)/max(1,min(totals));severely_unequal=bool(zeros or coverage_ratio>=5)
 byclass=defaultdict(Counter)
 for r in sec_rows:
  for f in formtot:byclass[r['filer_class']][f]+=int(r[f'{f}_count'])
 (OUT/'SEC_FACT_CHANNEL_COVERAGE.md').write_text('# SEC fact-channel coverage\n\nTraining-period counts use SEC submission metadata only and no outcomes.\n\n'+f'- Filer classes: `{dict(filer)}`\n- Form totals: `{formtot}`\n- Companies with zero matching frozen-source forms: **{zeros}/66**\n- Counts by filer class: `{ {k:dict(v) for k,v in byclass.items()} }`\n- Per-company total-form range: **{min(totals)}–{max(totals)}**; max/min ratio **{coverage_ratio:.2f}**.\n- Mechanical severe-inequality warning: **{severely_unequal}**, defined as any zero-coverage company or a max/min ratio of at least 5.\n\nForeign-private issuers are evaluated with 6-K and 20-F in addition to the domestic 8-K/10-Q/10-K channel. A zero is explicit missingness in the frozen SEC source and is not interpreted as absence of factual disclosure. Unequal form volume is retained as a future missingness/coverage variable; facts are never fabricated.\n')
 # Entity validation has not passed, so news/SEC reader pairs are not constructed.
 private_pairs=PVT/'READER_PAIR_REVIEW_CARDS.jsonl';private_pairs.write_text('')
 pair_manifest={'status':'BLOCKED_PENDING_INDEPENDENT_ENTITY_REVIEW','candidate_pair_construction_authorized':False,'SEC_factual_event_candidates':len(events),'news_pair_candidates':0,'reader_review_cards':0,'private_path':'work/stock-data/ecni_e1rv2/private/READER_PAIR_REVIEW_CARDS.jsonl','private_sha256':digest(private_pairs),'reason':'Frozen protocol requires semantic validation or rule-level repair before current/predecessor pair construction.','future_target_cards':300,'returns_or_direction_labels_used':0}
 write_json(OUT/'READER_PAIR_SAMPLING_MANIFEST.json',pair_manifest)
 (OUT/'READER_PAIR_LABELING_GUIDE.md').write_text('''# Reader-pair labeling guide (deferred)\n\nPair-card construction is blocked until independent entity review passes or failed aliases are repaired and the complete corpus is relinked. The future pack will label exactly one of `REPEAT_SAME_FACT`, `NEW_EVENT`, `NUMERIC_UPDATE`, `ACTION_CHANGE`, `PERIOD_CHANGE`, `DENIAL_OR_CORRECTION`, `DIFFERENT_ACTOR`, `DIFFERENT_COMPANY`, `HYPOTHETICAL_OR_OPINION`, or `INSUFFICIENT_EVIDENCE`. Heuristic categories are candidates, never gold. Near-duplicate families stay in one split, SEC-backed and news-only evidence remain identifiable, and no market outcome is shown.\n''')
 audit={'status':'AWAITING_INDEPENDENT_ENTITY_REVIEW','E1RV2_preregistration_sha':'8f69790578820edb146565a3c6c487c9fbeac04d','historical_E1R_protected_hashes_match':protected_ok(),'selected_panel_before_semantic_audit':66,'entity_review_cards':len(cards),'independent_entity_labels_available':False,'semantic_precision':None,'semantic_precision_gate_evaluated':False,'alias_rules_requiring_repair':'PENDING_INDEPENDENT_REVIEW_NDAQ_SENTINEL_FLAGGED','complete_relinking_rerun':False,'panel_changed':False,'final_panel_size_historical_pending_validation':66,'company_split_counts':{k:len(split[k]) for k in ('TRAIN_COMPANIES','DEV_UNSEEN_COMPANIES','LOCKED_UNSEEN_COMPANIES')},'filer_class_counts':dict(filer),'SEC_form_counts':formtot,'SEC_fact_channel_severely_unequal':severely_unequal,'SEC_company_form_count_range':[min(totals),max(totals)],'SEC_factual_event_candidates':len(events),'reader_pair_candidates':0,'reader_review_cards':0,'returns_or_predictive_outcomes_inspected':0,'predictive_models_fitted':0,'reader_models_fitted':0,'next_required_action':'Independent blinded semantic labeling of the frozen entity cards, including the 100-card NDAQ sentinel.'}
 write_json(OUT/'E1RV2_FINAL_AUDIT.json',audit)
 (OUT/'E1RV2_REPORT.md').write_text(f'''# ECNI Stage E1R-V2 report\n\n## Status\n\n**`AWAITING_INDEPENDENT_ENTITY_REVIEW`**. Historical E1R remains preserved. No independent semantic labels were supplied, so semantic precision and the registered gate are not measurable yet.\n\n## Mechanical findings\n\nThe complete alias census contains {len(alias_rows):,} alias/CIK rows. NDAQ's `nasdaq` alias has {int(nd['matched_articles']):,} all-years matches, {int(nd['direct_high_confidence_edges'])+int(nd['multi_company_high_confidence_edges']):,} all-years high-confidence edges, and {int(nd['training_high_confidence_edges']):,} 2018–2021 high-confidence edges. Its historical E1R panel coverage was 1.0. The dedicated 100-card sentinel contains {listing} mechanical listing/venue phrase matches and {corporate} narrow corporate-context matches. These are contamination diagnostics, not semantic labels.\n\nThe private blinded entity pack contains {len(cards)} unique cards: 330 panel-base cards covering all 66 companies, 100 NDAQ sentinel cards, and 100 alias-risk-enriched cards before overlap. No copyrighted card text is committed; the public manifest records the private hash and card IDs.\n\n## Panel and repair boundary\n\nNo alias rule was changed, complete relinking was not rerun, and no E1RV2 replacement panel was created before independent labels. The historical panel remains 66 companies with 44/11/11 company splits, pending semantic validation. A failed alias must trigger a rule-level repair and full relinking; individual rows cannot be deleted.\n\n## SEC channel\n\nFiler classes are `{dict(filer)}`. Training-period form totals are `{formtot}` and {len(events):,} outcome-blind SEC filing candidates were retained. Foreign-private issuers use 6-K/20-F coverage rather than being judged by domestic forms alone. Per-company form counts range from {min(totals)} to {max(totals)}, so the frozen mechanical criterion flags severe coverage inequality; missingness must remain explicit.\n\n## Reader boundary\n\nReader-pair construction and reader training are blocked. The next required action is independent blinded semantic labeling, including the frozen NDAQ sentinel. Returns/outcomes inspected: **0**. Predictive models fitted: **0**.\n''')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
