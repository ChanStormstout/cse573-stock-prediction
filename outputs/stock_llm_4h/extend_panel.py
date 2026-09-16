"""Pre-inference panel repair: add three late-Feb analyst articles outside trading windows."""
from common import *
from prepare import select_input
import pandas as pd,zipfile

def main():
 source=B/'data/pilot_v2/inputs.jsonl';out=new_run(B/'data/pilot_v3');rs=rows(source)
 raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl')
 members=['news_0003249.json','news_0003561.json','news_0002582.json']
 archive='2018_02_d157b48c57be246ec7dd80e7af4388a2.zip'
 with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive) as z:
  for j,m in enumerate(members,40):
   r=raw[(raw.archive==archive)&(raw.member==m)].iloc[0];body=json.loads(z.read(m)).get('text','');sentences,spans=select_input('AAPL',r.title,body)
   rs.append({'id':f'L{j}','symbol':'AAPL','record_key':archive+'::'+m,'group':f'late_feb_supplement_{j}','split':'check','stratum':'analyst_supplement','available_utc':str(r.available_utc),'published_utc':str(r.published_utc),'title':r.title,'sentences':sentences,'spans':spans,'body':body,'body_sha256':digest(body.encode())})
 write_rows(out/'inputs.jsonl',rs)
 dump(out/'manifest.json',{'n':len(rs),'source_sha256':file_sha(source),'input_sha256':file_sha(out/'inputs.jsonl'),'repair':'v1 dropped AMZN via record-only dedup; v2 repaired. v2 late check has no clear analyst-action positives. Before any frozen fact inference add first three distinct late-Feb rating/target candidates from all raw articles, including outside trading windows. Supplement reported separately. No target stock labels read.','quality':'assistant annotations pending; not independent; not representative sample'})
 for r in rs[-3:]:
  print(r['id'],r['title']);print('\n'.join(k+': '+v for k,v in r['sentences'].items()))
if __name__=='__main__':main()
