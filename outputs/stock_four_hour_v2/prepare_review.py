"""Create label-blind, deterministic company-paragraph review material."""
import json,re,hashlib,zipfile
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from run import ROOT
B=Path(__file__).parent
ALIASES={'AAPL':re.compile(r'\b(?:AAPL|Apple)\b',re.I),'AMZN':re.compile(r'\b(?:AMZN|Amazon(?:\.com)?)\b',re.I)}
def extract(body,symbol,tok):
 pieces=[];spans=[]
 for m in re.finditer(r'[^\r\n]+',body):
  if ALIASES[symbol].search(m.group()):pieces.append(m.group());spans.append([m.start(),m.end()])
  if len(pieces)==2:break
 text='\n'.join(pieces)
 enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
 if len(enc['input_ids'])>254:text=text[:enc['offset_mapping'][253][1]]
 return text,spans

def main():
 out=B/'review';out.mkdir(exist_ok=False)
 tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',revision='4556d13015211d73dccd3fdd39d39232506f3e43',cache_dir=str(ROOT/'work/stock-data/finbert-cache'),local_files_only=True)
 d=pd.read_pickle(ROOT/'outputs/stock_horizons/runs/v1/data.pkl');d=d[(d.horizon=='4h')&(d.split=='train')];r=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');r['key']=r.archive+'::'+r.member;r=r.set_index('key');selected=[];used=set()
 for sym in ['AAPL','AMZN']:
  for month in ['2018-03','2018-04','2018-05','2018-06','2018-07']:
   keys={k for x in d[(d.symbol==sym)&d.day.str.startswith(month)].article_keys for k in x.split('|') if k}
   keys=sorted(keys,key=lambda k:hashlib.sha256((sym+'|'+k).encode()).hexdigest());n=0
   for key in keys:
    row=r.loc[key]
    with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/row.archive) as z:body=json.loads(z.read(row.member)).get('text','')
    # Exact-content dedup avoids filling check strata with identical syndicated copies.
    ident=hashlib.sha256((sym+'|'+re.sub(r'\s+',' ',body).strip()).encode()).hexdigest()
    if ident in used:continue
    used.add(ident);text,spans=extract(body,sym,tok);selected.append(dict(sample_id=f'{sym}_{month}_{n+1}',symbol=sym,month=month,record_key=key,title=row.title,body=body,extracted=text,source_spans=json.dumps(spans),available_utc=str(row.available_utc),published_utc=str(row.published_utc),body_sha256=hashlib.sha256(body.encode()).hexdigest(),subject_correct='',template_or_mixed='',missed_target='',reviewer='',review_note=''));n+=1
    if n==4:break
 review=pd.DataFrame(selected);review.to_csv(out/'blind_review_40.csv',index=False)
 manifest={'status':'WAITING_INDEPENDENT_REVIEW','n':len(review),'strata':review.groupby(['symbol','month']).size().to_dict().__str__(),'rule':'All 40 checked by independent human; >=90% subject-correct, missing and template/mixed explicitly recorded. No assistant labels substitute. No future price labels displayed. Exact content dedup only; reviewer must flag near-duplicate event groups.','version':'paragraph_v1_max2_254_content_tokens','sha256':hashlib.sha256((out/'blind_review_40.csv').read_bytes()).hexdigest()}
 (out/'status.json').write_text(json.dumps(manifest,indent=2));print(manifest)
if __name__=='__main__':main()
