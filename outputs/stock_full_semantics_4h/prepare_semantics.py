"""Outcome-blind complete target paragraphs; common chunks fit both tokenizers."""
import sys,json,re,zipfile,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import load,check_sources,ROOT,sha,dump,digest
import pandas as pd
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;OUT=HERE/'v1';DEST=ROOT/'work/stock-data/full_semantics_4h/v1/semantic'
PAT={'AAPL':re.compile(r'\b(?:Apple|AAPL)\b',re.I),'AMZN':re.compile(r'\b(?:Amazon|AMZN)\b',re.I)}
def main():
 check_sources();assert not DEST.exists();DEST.mkdir(parents=True)
 d,*_=load();raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');raw['record_key']=raw.archive+'::'+raw.member;raw=raw.set_index('record_key')
 finger=json.loads((ROOT/'outputs/stock_integrated_4h/prepared/embedding_inputs.json').read_text())
 fb=AutoTokenizer.from_pretrained('ProsusAI/finbert',revision=finger['revision'],cache_dir=str(ROOT/'work/stock-data/finbert-cache'),local_files_only=True)
 modern=AutoTokenizer.from_pretrained(ROOT/'work/stock-data/foundation_4h/models/modern',local_files_only=True)
 pairs=sorted({(r.symbol,k) for r in d.itertuples() for k in r.news_record_keys.split('|') if k});bodies={};archive_hash={}
 needed=sorted({k for _,k in pairs})
 for archive,g in raw.loc[needed].groupby('archive'):
  path=ROOT/'work/stock-data/raw/news'/archive;archive_hash[archive]=sha(path)
  with zipfile.ZipFile(path) as z:
   for row in g.itertuples():bodies[row.Index]=str(json.loads(z.read(row.member)).get('text',''))
 articles=[];chunks=[];counts={}
 for symbol,key in pairs:
  body=html.unescape(re.sub(r'<[^>]+>',' ',bodies[key]));title=str(raw.loc[key,'title']);selected=[]
  for j,m in enumerate(re.finditer(r'[^\r\n]+',body)):
   if PAT[symbol].search(m.group()):selected.append(dict(paragraph_id=j,start=m.start(),end=m.end(),text=m.group()))
  aid=digest(symbol+'|'+key);article_chunks=[]
  # Preserve every selected paragraph, including negation and all numbers. No fixed-character truncation.
  for para in selected:
   pieces=list(re.finditer(r'\S+',para['text']));start=0
   while start<len(pieces):
    end=start+1
    def payload(n):return 'TARGET='+symbol+'\nTITLE: '+title+'\nPARAGRAPH: '+para['text'][pieces[start].start():pieces[n-1].end()]
    if max(len(t.encode(payload(end))) for t in [fb,modern])>512:raise ValueError('single word/title cannot fit common 512 budget; no silent truncation')
    lo=end;hi=len(pieces)
    while lo<hi:
     mid=(lo+hi+1)//2
     if max(len(t.encode(payload(mid))) for t in [fb,modern])<=512:lo=mid
     else:hi=mid-1
    end=lo
    assert max(len(t.encode(payload(end))) for t in [fb,modern])<=512
    cid=digest(aid+'|'+str(para['paragraph_id'])+'|'+str(start));text=payload(end)
    chunks.append(dict(chunk_id=cid,article_id=aid,symbol=symbol,record_key=key,paragraph_id=para['paragraph_id'],start=para['start']+pieces[start].start(),end=para['start']+pieces[end-1].end(),text=text,finbert_tokens=len(fb.encode(text)),modern_tokens=len(modern.encode(text))))
    article_chunks.append(cid);start=end
  articles.append(dict(article_id=aid,symbol=symbol,record_key=key,chunks=article_chunks,paragraphs=selected,body_sha256=digest(bodies[key]),normalized_body_sha256=digest(body),target_evidence=bool(selected)))
  counts[symbol]=counts.get(symbol,0)+bool(selected)
 for name,rows in [('articles.jsonl',articles),('chunks.jsonl',chunks)]:
  with (DEST/name).open('x') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 dump(DEST/'manifest.json',dict(code=sha(__file__),protocol=sha(OUT/'SEMANTIC_PROTOCOL.json'),archives=archive_hash,finbert_revision=finger['revision'],artifacts={n:sha(DEST/n) for n in ['articles.jsonl','chunks.jsonl']}))
 dump(OUT/'SEMANTIC_INPUT_AUDIT.json',dict(status='PREPARED_OUTCOME_BLIND',article_target_pairs=len(pairs),paragraphs=sum(len(r['paragraphs']) for r in articles),chunks=len(chunks),with_explicit_target_paragraph_by_stock=counts,max_finbert_tokens=max(r['finbert_tokens'] for r in chunks),max_modern_tokens=max(r['modern_tokens'] for r in chunks),discarded_selected_paragraphs=0,source='canonical title-filtered article membership unchanged; target paragraph lexical match is not independent semantic validation',offset_space='HTML-unescaped tag-stripped body; raw and normalized hashes retained'))
if __name__=='__main__':main()
