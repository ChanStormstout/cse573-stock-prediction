"""Pre-existing evidence rules on the SAME selected input, no new tuned rules."""
import sys,re,argparse
from common import *
sys.path.insert(0,str(ROOT/'outputs/stock_adaptive_4h'))
from evidence import extract

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=B/'runs/rule_v1');p.add_argument('--data',type=Path,default=B/'data/pilot_v3');a=p.parse_args();out=new_run(a.out);check_panel(a.data);rs=rows(a.data/'inputs.jsonl');labels=rows(a.data/'labels.jsonl');preds=[]
 for r in rs:
  pieces=[];spans=[];cursor=0
  for key,text in r['sentences'].items():
   if key=='S0':continue
   spans.append((cursor,cursor+len(text),key));pieces.append(text);cursor+=len(text)+1
  body='\n'.join(pieces);es=extract(r['symbol'],r['record_key'],r['sentences']['S0'],body,r['published_utc'],r['available_utc']);events=[];seen=set()
  for e in es:
   if e['kind'] not in {'analyst_target','analyst_rating'} or e['temporal_role']=='historical' or e['conflict']:continue
   ids=['S0'] if e['source']=='title' else [k for a,b,k in spans if a<e['end'] and b>e['start']]
   evidence=' '.join(r['sentences'][k] for k in ids)
   kind='rating' if e['kind']=='analyst_rating' else 'target_price'
   def num(v):
    if v is None:return None
    return next((s for s in re.findall(r'\d[\d,]*(?:\.\d+)?',evidence) if float(s.replace(',',''))==v),None)
   old=e.get('old_rating') if kind=='rating' else num(e.get('old_value'));new=e.get('new_rating') if kind=='rating' else num(e.get('new_value'))
   ev={'kind':kind,'action':{'up':'raise','down':'lower','upgrade':'raise','downgrade':'lower','maintain':'maintain'}.get(e['action'],'unknown'),'old':old,'new':new,'unit':'rating' if kind=='rating' else 'USD','evidence_ids':ids}
   sig=signature(ev)
   if sig in seen:continue
   seen.add(sig);events.append(ev)
  obj={'events':events};valid,errors=validate(obj,r);preds.append({'id':r['id'],'parsed':obj,'valid':valid,'errors':errors})
 write_rows(out/'predictions.jsonl',preds);dump(out/'metrics.json',evaluate(preds,labels));dump(out/'manifest.json',{'extractor_sha256':file_sha(ROOT/'outputs/stock_adaptive_4h/evidence.py'),'inputs_sha256':file_sha(a.data/'inputs.jsonl'),'no_new_rule_fitting':True})
if __name__=='__main__':main()
