#!/usr/bin/env python3
"""Outcome-blind ECNI E1 data qualification.

This script reads private bounded FNSPID samples and the official price archive.
It never constructs direction labels, returns, or predictive features.
"""
from __future__ import annotations

import csv, hashlib, io, json, math, re, statistics, zipfile
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stock_ecni_e1"
PRIVATE = ROOT / "work/stock-data/ecni_e1"
SAMPLE = PRIVATE / "audit/fnspid_stratified_sample.jsonl"
PRICE_ZIP = PRIVATE / "raw/full_history.zip"
SEC = PRIVATE / "raw/company_tickers_exchange.json"
SEED = "57320260919"

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def norm(s): return re.sub(r"[^a-z0-9]+"," ",(s or "").lower()).strip()
def present(s): return bool((s or "").strip())
def q(vals,p):
    if not vals:return 0
    vals=sorted(vals); x=(len(vals)-1)*p; a=int(x); b=min(a+1,len(vals)-1)
    return vals[a]*(b-x)+vals[b]*(x-a)
def wilson(k,n,z=1.96):
    if not n:return (0,0)
    ph=k/n; den=1+z*z/n; c=(ph+z*z/(2*n))/den
    d=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return c-d,c+d
def dt_class(s):
    if not present(s): return "INVALID"
    try:
        d=datetime.fromisoformat(s.replace(" UTC","+00:00").replace("Z","+00:00"))
    except Exception:return "INVALID"
    if d.hour==d.minute==d.second==0:return "DATE_ONLY_CONSERVATIVE"
    return "EXACT_INTRADAY_USABLE" if ('UTC' in s or '+' in s or s.endswith('Z')) else "AMBIGUOUS"
def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)

def news_audits(records):
    ts=defaultdict(Counter); txt={}
    groups=defaultdict(list)
    for r in records:
        keys=[(r['file'],r['year'],r.get('source') or 'UNKNOWN'),(r['file'],'ALL','ALL')]
        cls=dt_class(r.get('date',''))
        for key in keys:
            ts[key][cls]+=1; ts[key]['records']+=1
            a=txt.setdefault(key,{'records':0,'title':0,'body':0,'summary':0,'publisher':0,'author':0,'url':0,'body_lengths':[]})
            a['records']+=1
            for fld in ['title','body','publisher','author','url']: a[fld]+=present(r.get(fld,''))
            a['summary']+=any(present(x) for x in r.get('summaries',[]))
            if present(r.get('body','')): a['body_lengths'].append(len(r['body']))
        groups[(r['file'],r['year'],r.get('source') or 'UNKNOWN')].append(r)
    tsrows=[]
    for key,c in sorted(ts.items()):
        n=c['records']; tsrows.append({'file':key[0],'year':key[1],'source':key[2],'records':n,
          **{x:c[x] for x in ['EXACT_INTRADAY_USABLE','DATE_ONLY_CONSERVATIVE','AMBIGUOUS','INVALID']},
          'classifiable_fraction':(c['EXACT_INTRADAY_USABLE']+c['DATE_ONLY_CONSERVATIVE'])/n})
    write_csv(OUT/'FNSPID_TIMESTAMP_AUDIT.csv',tsrows,list(tsrows[0]))
    txrows=[]
    for key,a in sorted(txt.items()):
        n=a['records']; lens=a.pop('body_lengths')
        txrows.append({'file':key[0],'year':key[1],'source':key[2],'records':n,
          **{f'{x}_available_fraction':a[x]/n for x in ['title','body','summary','publisher','author','url']},
          'body_length_p25':q(lens,.25),'body_length_median':q(lens,.5),'body_length_p75':q(lens,.75)})
    write_csv(OUT/'FNSPID_TEXT_AVAILABILITY_AUDIT.csv',txrows,list(txrows[0]))
    return tsrows,txrows

def entity_audit(records):
    sec=json.load(SEC.open())
    rows=sec['data'][1:] if isinstance(sec,dict) and 'data' in sec else []
    aliases={str(x[2]).upper():[str(x[1]),str(x[2])] for x in rows if len(x)>=4}
    eligible=[r for r in records if present(r.get('ticker',''))]
    chosen=sorted(eligible,key=lambda r:hashlib.sha256((SEED+'|entity|'+r['record_id']).encode()).hexdigest())[:600]
    out=[]; counts=Counter()
    for r in chosen:
        ticker=r['ticker'].upper().strip(); title=r.get('title',''); body=r.get('body',''); text=' '.join([title,body]); ntext=' '+norm(text)+' '
        names=aliases.get(ticker,[ticker]); terms={norm(ticker)}
        for name in names:
            nn=norm(name)
            if nn: terms.add(nn)
            for suffix in [' common stock',' corporation',' incorporated',' inc',' plc',' limited',' ltd',' company',' corp']:
                if nn.endswith(suffix):terms.add(nn[:-len(suffix)].strip())
        hit=any(len(t)>1 and (' '+t+' ') in ntext for t in terms)
        other=0
        for t,ns in aliases.items():
            if t==ticker:continue
            if (' '+norm(t)+' ') in ntext and len(t)>1:other+=1
        if not (present(title) or present(body)): status='UNDETERMINED'
        elif hit and other: status='MULTI_COMPANY_INCLUDES_TARGET'
        elif hit: status='DIRECT_TARGET'
        elif any(x in ntext for x in [' competitor ',' rival ',' versus ',' vs ']): status='INDIRECT_OR_COMPETITOR'
        else: status='NO_EVIDENCE_TARGET'
        counts[status]+=1
        out.append({'record_id_hash':hashlib.sha256(r['record_id'].encode()).hexdigest(),'file':r['file'],'year':r['year'],'ticker':ticker,
          'relationship':status,'evidence_field':'title+body' if present(body) else ('title' if present(title) else 'none'),
          'target_term_detected':hit,'provisional_method':'deterministic lexical identity audit; not human gold'})
    write_csv(OUT/'FNSPID_ENTITY_AUDIT.csv',out,list(out[0]))
    good=counts['DIRECT_TARGET']+counts['MULTI_COMPANY_INCLUDES_TARGET']; lo,hi=wilson(good,len(out))
    return {'records':len(out),'counts':dict(counts),'association_rate':good/len(out),'wilson95':[lo,hi],
            'limitation':'Provisional deterministic lexical audit, not independent human gold.'}

def duplicate_audit(records):
    def group(field):
        g=defaultdict(list)
        for r in records:
            v=r.get(field,''); v=norm(v) if field=='title' else v.strip()
            if v:g[v].append(r)
        return [v for v in g.values() if len(v)>1]
    ug=group('url'); tg=group('title')
    # bounded near-title candidates sharing an informative prefix token
    buckets=defaultdict(list)
    for r in records:
        t=norm(r.get('title','')); toks=[x for x in t.split() if len(x)>=5]
        if t and toks:buckets[toks[0]].append((t,r))
    edges=[]
    for vals in buckets.values():
        vals=vals[:250]
        for i in range(len(vals)):
            for j in range(i+1,len(vals)):
                if vals[i][0]!=vals[j][0] and SequenceMatcher(None,vals[i][0],vals[j][0]).ratio()>=.9:
                    edges.append((vals[i][1]['record_id'],vals[j][1]['record_id']))
    def stats(gs):
        return {'groups':len(gs),'records_in_groups':sum(map(len,gs)),'max_group_size':max(map(len,gs),default=0),
          'multi_publisher_groups':sum(len({x.get('source') for x in g})>1 for g in gs)}
    out={'sample_records':len(records),'exact_url':stats(ug),'exact_normalized_title':stats(tg),
      'near_title_similarity_threshold':0.9,'near_title_edges':len(edges),
      'body_similarity':'not computed when bodies unavailable; no copyrighted text is exported',
      'interpretation':'Candidate dissemination groups only; no records collapsed.'}
    json.dump(out,(OUT/'FNSPID_DUPLICATE_AUDIT.json').open('w'),indent=2)
    return out

def price_audit():
    rows=[]
    with zipfile.ZipFile(PRICE_ZIP) as z:
        for name in z.namelist():
            if not name.startswith('full_history/') or not name.endswith('.csv'):continue
            sym=Path(name).stem
            try: recs=list(csv.DictReader(io.TextIOWrapper(z.open(name),encoding='utf-8-sig',errors='replace')))
            except Exception:continue
            dates=[]; missing=dup=invalid=0; adjusted=0
            for r in recs:
                ds=(r.get('date') or '').strip(); dates.append(ds)
                vals={k:(r.get(k) or '').strip() for k in ['open','high','low','close','adj close','volume']}
                missing+=any(not v for v in vals.values()); adjusted+=bool(vals['adj close'])
                try:
                    o,h,l,c=map(float,[vals['open'],vals['high'],vals['low'],vals['close']])
                    invalid+=not(l<=min(o,c)<=max(o,c)<=h)
                except Exception: invalid+=1
            dup=len(dates)-len(set(dates)); mono=dates not in [sorted(dates),sorted(dates,reverse=True)]
            target=[d for d in dates if '2018-01-01'<=d<='2023-12-31']
            rows.append({'symbol':sym,'first_date':min(dates,default=''),'last_date':max(dates,default=''),'trading_rows':len(recs),
              'rows_2018_2023':len(target),'missing_ohlcv_rows':missing,'missing_ohlcv_rate':missing/len(recs) if recs else 1,
              'duplicate_dates':dup,'non_monotonic_dates':int(mono),'invalid_ohlc_rows':invalid,
              'adjusted_close_available_fraction':adjusted/len(recs) if recs else 0,
              'price_eligible_2018_2023':int(min(dates,default='9999')<='2018-01-02' and max(dates,default='')>='2023-12-28' and not missing and not dup and not mono and not invalid)})
    write_csv(OUT/'FNSPID_PRICE_COVERAGE_BY_SYMBOL.csv',rows,list(rows[0]))
    return rows

def main():
    OUT.mkdir(exist_ok=True)
    records=[json.loads(x) for x in SAMPLE.open()]
    ts,tx=news_audits(records)
    ent=entity_audit(records); dup=duplicate_audit(records); prices=price_audit()
    summary={'sample_records':len(records),'sample_sha256':sha256(SAMPLE),'price_zip_sha256':sha256(PRICE_ZIP),
      'entity':ent,'duplicates':dup,'price_symbols':len(prices),'price_eligible_symbols':sum(int(r['price_eligible_2018_2023']) for r in prices)}
    json.dump(summary,(PRIVATE/'audit/computed_summary.json').open('w'),indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
