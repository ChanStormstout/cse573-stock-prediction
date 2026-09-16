"""Post-check safety fix: never reverse a partially recognized rating pair."""
import argparse
import json
import re
import sys
from pathlib import Path
from contracts import ROOT, values
sys.path.insert(0,str(ROOT/'outputs/stock_llm_4h'))
from common import rows,dump,evaluate,file_sha,check_panel

WORD = r'(?:a\s+)?[\"“”\x27]*[A-Za-z][\w-]*(?:\s+(?:buy|sell|perform|weight))?[\"“”\x27]*(?:\s+rating)?'
PAIR = re.compile(r'\bfrom\s+'+WORD+r'\s+to\s+'+WORD+r'|\bto\s+'+WORD+r'\s+from\s+'+WORD,re.I)

def repair(pred):
    result=json.loads(json.dumps(pred));dropped=[]
    if not pred['valid'] or len(pred['calls'])!=2:return result,dropped
    obj=pred['calls'][1]['parsed']
    if not isinstance(obj,dict) or not isinstance(obj.get('items'),list):return result,dropped
    for item in obj['items']:
        if not isinstance(item,dict) or item.get('kind')!='rating':continue
        quote=item.get('quote')
        if not isinstance(quote,str) or not PAIR.search(quote):continue
        try:old,new=values(quote,'rating')
        except (ValueError,TypeError):continue
        if old is not None:continue
        event=dict(kind='rating',action=item.get('action'),old=None,new=new,unit='rating',evidence_ids=[item.get('evidence_id')])
        if event in result['parsed']['events']:
            result['parsed']['events'].remove(event);dropped.append({'item':item,'reason':'unparsed_rating_pair'})
    result['rejected']+=dropped;result['safety_drops']=len(dropped)
    return result,dropped

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=ROOT/'work/stock-data/annotation/student_v3';check_panel(data);labels=rows(data/'labels.jsonl')
    source=a.input/'predictions.jsonl';original=rows(source);expected={r['id'] for r in labels if r['split'] in {'valid','check'}}
    assert len(original)==len(expected) and {r['id'] for r in original}==expected
    a.out.mkdir(parents=True,exist_ok=False);fixed=[];changes=[]
    for pred in original:
        row,drops=repair(pred);fixed.append(row)
        if drops:changes.append({'id':row['id'],'count':len(drops)})
    with (a.out/'predictions.jsonl').open('w') as f:
        for row in fixed:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(a.out/'metrics.json',evaluate(fixed,labels))
    dump(a.out/'manifest.json',{'status':'POST_CHECK_SAFETY_FIX_NOT_INDEPENDENT_IMPROVEMENT',
         'source_sha256':file_sha(source),'code_sha256':file_sha(__file__),'parser_sha256':file_sha(Path(__file__).with_name('contracts.py')),
         'changes':changes,'additional_model_calls':0,'rule':'Refuse an unresolved from/to rating pair instead of treating its sole recognized word as the new rating.'})

if __name__=='__main__':main()
