"""Independent exact source-offset and complete selected-paragraph coverage audit."""
import sys,json,re,html,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import ROOT,sha,digest,dump,check_sources,load
import pandas as pd
HERE=Path(__file__).resolve().parent;SRC=ROOT/'work/stock-data/full_semantics_4h/v1/semantic'
def main():
 check_sources();d,*_=load();mf=json.loads((SRC/'manifest.json').read_text());a=[json.loads(s) for s in (SRC/'articles.jsonl').read_text().splitlines()];c=[json.loads(s) for s in (SRC/'chunks.jsonl').read_text().splitlines()]
 for n,h in mf['artifacts'].items():assert sha(SRC/n)==h
 lookup={r['chunk_id']:r for r in c};assert len(lookup)==len(c)
 expected={(r.symbol,k) for r in d.itertuples() for k in r.news_record_keys.split('|') if k};assert expected=={(r['symbol'],r['record_key']) for r in a};assert len(expected)==len(a)
 raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');bodies={}
 for archive,g in raw.loc[sorted({k for _,k in expected})].groupby('archive'):
  p=ROOT/'work/stock-data/raw/news'/archive;assert sha(p)==mf['archives'][archive]
  with zipfile.ZipFile(p) as z:
   for row in g.itertuples():bodies[row.Index]=str(json.loads(z.read(row.member)).get('text',''))
 for r in a:
  body=bodies[r['record_key']];assert digest(body)==r['body_sha256'];normalized=html.unescape(re.sub(r'<[^>]+>',' ',body));assert digest(normalized)==r['normalized_body_sha256']
  pattern=r'\b(?:Apple|AAPL)\b' if r['symbol']=='AAPL' else r'\b(?:Amazon|AMZN)\b'
  selected=[(i,m.start(),m.end(),m.group()) for i,m in enumerate(re.finditer(r'[^\r\n]+',normalized)) if re.search(pattern,m.group(),re.I)]
  assert selected==[(p['paragraph_id'],p['start'],p['end'],p['text']) for p in r['paragraphs']]
  for pid,start,end,text in selected:
   chunks=sorted([lookup[k] for k in r['chunks'] if lookup[k]['paragraph_id']==pid],key=lambda q:q['start']);joined=[]
   for x in chunks:
    assert start<=x['start']<x['end']<=end;assert x['article_id']==r['article_id'];assert x['finbert_tokens']<=512 and x['modern_tokens']<=512
    excerpt=normalized[x['start']:x['end']];assert x['text']=='TARGET='+r['symbol']+'\nTITLE: '+str(raw.loc[r['record_key'],'title'])+'\nPARAGRAPH: '+excerpt;joined.extend(excerpt.split())
   assert joined==text.split(),'missing/duplicated/reordered paragraph words'
 dump(HERE/'v1/SEMANTIC_INPUT_VERIFICATION.json',dict(status='PASS',article_pairs=len(a),chunks=len(c),source_membership=True,source_hashes=True,exact_offsets=True,complete_paragraph_word_coverage=True,matched_chunk_budget=True,fit_calls=0,semantic_role_quality='NOT_INDEPENDENTLY_REVIEWED'))
 print('Input verification PASS')
if __name__=='__main__':main()
