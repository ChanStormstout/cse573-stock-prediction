#!/usr/bin/env python3
"""Complete, outcome-blind FNSPID metadata census and independent SEC-CIK relinking."""
from __future__ import annotations
import csv, hashlib, io, json, re, statistics, sys, zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import ahocorasick
import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(sys.maxsize)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/stock_ecni_e1r'; PRIV=ROOT/'work/stock-data/ecni_e1r'; RAW=PRIV/'raw'; PVT=PRIV/'private'
PRICE_AUDIT=ROOT/'outputs/stock_ecni_e1/FNSPID_PRICE_COVERAGE_BY_SYMBOL.csv'
SEC_CURRENT=ROOT/'work/stock-data/ecni_e1/raw/company_tickers_exchange.json'
REV='bf9189c41527198897d1af3e17b1a0095279fc45'; SEED='57320260919'
FILES=[('all_external',RAW/'All_external.csv','5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3'),('nasdaq',RAW/'nasdaq_exteral_data.csv','1a7a3eb8e6b97ec19f286f2cfca3371542bddb272ab1eb8f36e33ad98fa5c4da')]
SUFFIX=re.compile(r'\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|holdings?|group|common stock|ordinary shares?)\b')
COMMON={'all','cat','it','a','an','the','one','first','best','new','general','global','national','international','united','american','digital','energy','capital','financial','bank','trust','market','markets','stock','stocks','system','systems','technology','technologies','data','service','services','health','care','communications','media','realty','resources','partners','management','enterprise','enterprises','industries','industrial'}
EVENT_CUE=re.compile(r'\b(announce|announced|report|reported|said|filed|filing|launch|launched|acquire|acquired|acquisition|merge|merger|earnings|revenue|profit|loss|guidance|forecast|rating|upgrade|downgrade|target price|lawsuit|sue|settle|agreement|appoint|resign|dividend|buyback|offering|contract|approve|approved|reject|recall|invest|investment|stake|shares)\b')

def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def H(s):return hashlib.sha256((s or '').encode('utf-8','replace')).hexdigest()
def norm(s):return re.sub(r'[^a-z0-9]+',' ',(s or '').lower()).strip()
def alias_norm(s):return re.sub(r'\s+',' ',SUFFIX.sub(' ',norm(s))).strip()
def timeclass(s):
 if not s:return 'INVALID'
 try:d=datetime.fromisoformat(s.replace(' UTC','+00:00').replace('Z','+00:00'))
 except:return 'INVALID'
 if d.hour==d.minute==d.second==0:return 'DATE_ONLY_CONSERVATIVE'
 return 'EXACT_INTRADAY_USABLE' if ('UTC' in s or 'Z' in s or '+' in s[-7:]) else 'AMBIGUOUS'
def nearfp(s):
 toks=[x for x in norm(s).split() if len(x)>2]
 if not toks:return ''
 # Token-order preserving coarse fingerprint for cutoff-safe near-duplicate candidates.
 return H(' '.join(toks[:12]))[:16]
def safe_alias(a):
 toks=a.split()
 if not a or a in COMMON:return False
 if len(toks)==1:return len(a)>=5 and a not in COMMON
 return len(a)>=6 and any(len(x)>=4 and x not in COMMON for x in toks)

def load_identities():
 price={r['symbol'] for r in csv.DictReader(PRICE_AUDIT.open()) if r['price_eligible_2018_2023']=='1'}
 cur=json.load(SEC_CURRENT.open()); current=[dict(zip(cur['fields'],x)) for x in cur['data']]
 z=zipfile.ZipFile(RAW/'submissions.zip'); znames=set(z.namelist())
 identities=[]; rejected=[]
 aliasmap=defaultdict(set); ticker_to={}
 for r in current:
  t=str(r['ticker']).upper(); cik=int(r['cik'])
  if t not in price:continue
  name=f'CIK{cik:010d}.json'
  if name not in znames:continue
  try:x=json.loads(z.read(name))
  except:continue
  names=[x.get('name','')]+[q.get('name','') for q in x.get('formerNames',[])]
  aliases=[]
  for n in names:
   for a in {norm(n),alias_norm(n)}:
    if safe_alias(a):aliases.append(a);aliasmap[a].add(cik)
    elif a:rejected.append({'normalized_alias':a,'CIK':cik,'reason':'UNSAFE_SHORT_OR_COMMON_ALIAS'})
  ticker_to[t]=cik
  identities.append({'CIK':cik,'company_id':f'CIK{cik:010d}','canonical_issuer_name':x.get('name') or r['name'],'ticker':t,'exchange':r['exchange'],
   'former_names':json.dumps(x.get('formerNames',[]),separators=(',',':')),'effective_information_date':'2026-09-19_CURRENT_SNAPSHOT',
   'SEC_current_SIC':str(x.get('sic') or ''),'SEC_current_SIC_description':x.get('sicDescription') or '',
   'identity_source':'SEC submissions bulk + company_tickers_exchange','identity_confidence':'CURRENT_LISTED_IDENTITY_VERIFIED',
   'aliases':json.dumps(sorted(set(aliases)),separators=(',',':'))})
 # Drop aliases shared by multiple CIKs.
 unique={a:next(iter(cs)) for a,cs in aliasmap.items() if len(cs)==1}
 for a,cs in aliasmap.items():
  if len(cs)>1:
   for cik in sorted(cs):rejected.append({'normalized_alias':a,'CIK':cik,'reason':'ALIAS_SHARED_BY_MULTIPLE_CIKS'})
 A=ahocorasick.Automaton()
 for a,cik in unique.items():A.add_word(' '+a+' ',(a,cik))
 A.make_automaton()
 return identities,rejected,unique,ticker_to,A

def schemas():
 md=pa.schema([('source_file',pa.string()),('row_number',pa.int64()),('record_id_hash',pa.string()),('recorded_time',pa.string()),('year',pa.int16()),('publisher_hash',pa.string()),('native_candidate_tag',pa.string()),('title_hash',pa.string()),('normalized_title_hash',pa.string()),('url_hash',pa.string()),('body_available',pa.bool_()),('body_length',pa.int32()),('body_hash',pa.string()),('summary_available',pa.bool_()),('timestamp_class',pa.string()),('near_title_fingerprint',pa.string()),('candidate_numeric_update',pa.bool_()),('candidate_period_change',pa.bool_()),('candidate_action_change',pa.bool_()),('candidate_denial_correction',pa.bool_()),('candidate_hypothetical_opinion',pa.bool_())])
 ed=pa.schema([('record_id_hash',pa.string()),('company_id',pa.string()),('CIK',pa.int64()),('native_ticker',pa.string()),('resolved_ticker',pa.string()),('relation_type',pa.string()),('matched_alias',pa.string()),('matched_field',pa.string()),('evidence_start_normalized',pa.int32()),('evidence_end_normalized',pa.int32()),('confidence_rule',pa.string()),('available_time_class',pa.string()),('recorded_time',pa.string()),('source_file',pa.string()),('publisher_hash',pa.string()),('body_available',pa.bool_()),('title_hash',pa.string()),('url_hash',pa.string())])
 return md,ed

def main():
 OUT.mkdir(exist_ok=True);PVT.mkdir(parents=True,exist_ok=True)
 for _,p,expected in FILES:
  if not p.exists() or sha_file(p)!=expected:raise SystemExit(f'Source hash mismatch: {p}')
 identities,rejected,aliasmap,ticker_to,A=load_identities()
 # Public identity history contains no bodies.
 fields=list(identities[0]);
 with (OUT/'SEC_ISSUER_IDENTITY_HISTORY.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(identities)
 with (OUT/'AMBIGUOUS_ALIAS_LEDGER.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['normalized_alias','CIK','reason'],lineterminator='\n');w.writeheader();w.writerows(rejected)
 cik_ticker={x['CIK']:x['ticker'] for x in identities}
 mdsch,edsch=schemas(); mw=pq.ParquetWriter(PVT/'fnspid_complete_metadata.parquet',mdsch,compression='zstd',use_dictionary=True); ew=pq.ParquetWriter(OUT/'FNSPID_RELINKED_ENTITY_EDGES.parquet',edsch,compression='zstd',use_dictionary=True)
 total=Counter(); years=defaultdict(Counter); timecnt=defaultdict(Counter); edgecnt=Counter(); native_agree=Counter(); mdrows=[];edrows=[]
 def flush():
  nonlocal mdrows,edrows
  if mdrows:mw.write_table(pa.Table.from_pylist(mdrows,schema=mdsch));mdrows=[]
  if edrows:ew.write_table(pa.Table.from_pylist(edrows,schema=edsch));edrows=[]
 for filetag,path,_ in FILES:
  with path.open(encoding='utf-8',errors='replace',newline='') as f:
   reader=csv.DictReader(f)
   for i,r in enumerate(reader,1):
    total[(filetag,'rows')]+=1
    try:
     date=(r.get('Date') or '').strip();title=r.get('Article_title') or '';body=r.get('Article') or '';url=r.get('Url') or '';pub=r.get('Publisher') or ''
     native=(r.get('Stock_symbol') or '').strip().upper(); tc=timeclass(date)
     try:y=int(date[:4])
     except:y=0
     record=H(filetag+'|'+str(i)+'|'+url+'|'+date+'|'+title)
     nt=norm(title); nb=norm(body)
     lowtext=(title+' '+body[:10000]).lower()
     mdrows.append({'source_file':filetag,'row_number':i,'record_id_hash':record,'recorded_time':date,'year':y,'publisher_hash':H(pub) if pub else '',
      'native_candidate_tag':native,'title_hash':H(title) if title else '','normalized_title_hash':H(nt) if nt else '','url_hash':H(norm(url)) if url else '',
      'body_available':bool(body.strip()),'body_length':len(body),'body_hash':H(body) if body else '',
      'summary_available':any((r.get(k) or '').strip() for k in ['Lsa_summary','Luhn_summary','Textrank_summary','Lexrank_summary']),
      'timestamp_class':tc,'near_title_fingerprint':nearfp(title),
      'candidate_numeric_update':bool(re.search(r'\d',lowtext) and re.search(r'\b(raise|raised|lower|lowered|increase|increased|decrease|decreased|from|to)\b',lowtext)),
      'candidate_period_change':bool(re.search(r'\b(quarter|quarterly|fiscal|annual|year|month|guidance)\b',lowtext)),
      'candidate_action_change':bool(re.search(r'\b(upgrade|upgraded|downgrade|downgraded|maintain|maintained|initiate|initiated|raise|lower)\b',lowtext)),
      'candidate_denial_correction':bool(re.search(r'\b(deny|denied|denies|correct|corrected|correction|retract|retracted|clarif(?:y|ied|ication))\b',lowtext)),
      'candidate_hypothetical_opinion':bool(re.search(r'\b(could|would|may|might|opinion|believe|expects?|possible|potential)\b',lowtext))})
     years[filetag][y]+=1;timecnt[(filetag,y)][tc]+=1
     found={}
     for field,text in [('title',nt),('body',nb)]:
      if not text:continue
      padded=' '+text+' '
      for end,(a,cik) in A.iter(padded):
       start=end-len(a)+1
       old=found.get(cik)
       if old is None or (old[0]=='body' and field=='title'):found[cik]=(field,a,max(0,start-1),max(0,end-1))
     # Explicit ticker syntax only; native tag never creates an edge.
     original=title+'\n'+body
     for t in re.findall(r'(?i)(?:\$|NASDAQ\s*:\s*|NYSE\s*:\s*)([A-Z][A-Z0-9.\-]{0,9})\b',original):
      t=t.upper();cik=ticker_to.get(t)
      if cik is not None and cik not in found:found[cik]=('explicit_ticker',t,-1,-1)
     high={}
     for cik,(field,a,st,en) in found.items():
      if field in ('title','explicit_ticker'):high[cik]=True
      else:
       padded=' '+nb+' ';context=padded[max(0,st-180):min(len(padded),en+181)]
       high[cik]=bool(EVENT_CUE.search(context))
     high_count=sum(high.values())
     for cik,(field,a,st,en) in found.items():
      rt=cik_ticker[cik]; rule='EXPLICIT_EXCHANGE_OR_DOLLAR_TICKER' if field=='explicit_ticker' else 'UNIQUE_DISTINCTIVE_SEC_ISSUER_ALIAS'
      rel=('MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE' if high_count>1 else 'DIRECT_TARGET_HIGH_CONFIDENCE') if high[cik] else 'INDIRECT_OR_COMPETITOR'
      edrows.append({'record_id_hash':record,'company_id':f'CIK{cik:010d}','CIK':cik,'native_ticker':native,'resolved_ticker':rt,'relation_type':rel,
       'matched_alias':a,'matched_field':field,'evidence_start_normalized':st,'evidence_end_normalized':en,'confidence_rule':rule,
       'available_time_class':tc,'recorded_time':date,'source_file':filetag,'publisher_hash':H(pub) if pub else '',
       'body_available':bool(body.strip()),'title_hash':H(title) if title else '','url_hash':H(norm(url)) if url else ''})
      edgecnt[(filetag,rel)]+=1;native_agree['agree' if native==rt else 'disagree']+=1
     total[(filetag,'parsed')]+=1
    except Exception:
     total[(filetag,'parse_failures')]+=1
    if len(mdrows)>=50000 or len(edrows)>=50000:flush()
    if i%500000==0:print(filetag,i,'edges',sum(edgecnt.values()),flush=True)
  flush()
 mw.close();ew.close()
 # Census outputs.
 ts=[]
 for (ft,y),c in sorted(timecnt.items()):
  n=sum(c.values());ts.append({'source_file':ft,'year':y,'records':n,**{k:c[k] for k in ['EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE','AMBIGUOUS','INVALID']}})
 with (OUT/'TIMESTAMP_CENSUS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(ts[0]),lineterminator='\n');w.writeheader();w.writerows(ts)
 manifests=[]
 for ft,p,expected in FILES:
  manifests.append({'source_file':ft,'path':p.name,'bytes':p.stat().st_size,'sha256':expected,'rows_processed':total[(ft,'rows')],'rows_parsed':total[(ft,'parsed')],'parse_failures':total[(ft,'parse_failures')],'year_min':min(years[ft]),'year_max':max(years[ft])})
 out={'status':'COMPLETE','official_revision':REV,'files':manifests,'total_rows_processed':sum(x['rows_processed'] for x in manifests),
  'total_parse_failures':sum(x['parse_failures'] for x in manifests),'private_metadata_index':{'format':'parquet','sha256':sha_file(PVT/'fnspid_complete_metadata.parquet'),'committed':False},
  'relinked_edges':{'rows':sum(edgecnt.values()),'native_agreement':dict(native_agree)},'native_ticker_used_as_ground_truth':False,'copyrighted_bodies_retained_in_public_outputs':False}
 json.dump(out,(OUT/'FNSPID_COMPLETE_METADATA_MANIFEST.json').open('w'),indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
