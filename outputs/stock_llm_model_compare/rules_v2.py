"""Single registered delta-to-target repair on already validated literal items."""
import argparse
import json
from pathlib import Path
import re
import sys
from contracts import ROOT, PRICE, compile_items, validate_gate
sys.path.insert(0,str(ROOT/'outputs/stock_llm_4h'))
from common import rows,dump,validate,evaluate,file_sha,check_panel

def compile_v2(obj,row,ids):
    answer,rejected=compile_items(obj,row,ids)
    remain=[];repairs=[]
    for rec in rejected:
        item=rec['item'];quote=item['quote'] if isinstance(item,dict) and 'quote' in item else ''
        match=re.search(r'\bby\s+'+PRICE+r'\s+to\s+'+PRICE,quote,re.I) if isinstance(quote,str) else None
        if rec['reason']=='ambiguous_target_values' and item['kind']=='target_price' and match and len(re.findall(PRICE,quote,re.I))==2:
            event=dict(kind='target_price',action=item['action'],old=None,new=match.group(2),unit='USD',evidence_ids=[item['evidence_id']])
            if event not in answer['events']:answer['events'].append(event)
            repairs.append({'reason':'change_magnitude_not_old_value','event':event})
        else:remain.append(rec)
    return answer,remain,repairs

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'work/stock-data/annotation/student_v3';check_panel(data)
    assert json.loads((a.run/'summary.json').read_text())['completed']
    manifest=json.loads((a.run/'manifest.json').read_text())
    assert manifest['input_sha256']==file_sha(data/'inputs.jsonl') and manifest['labels_sha256']==file_sha(data/'labels.jsonl')
    inputs={r['id']:r for r in rows(data/'inputs.jsonl')};labels=rows(data/'labels.jsonl')
    source=a.run/'staged/predictions.jsonl';preds=[]
    source_rows=rows(source);expected={r['id'] for r in labels if r['split'] in {'valid','check'}}
    assert len(source_rows)==len(expected) and {r['id'] for r in source_rows}==expected
    for original in source_rows:
        pred=dict(original);pred['repairs']=[]
        if len(original['calls'])==2:
            try:
                ids=validate_gate(original['calls'][0]['parsed'],inputs[original['id']])
                obj,rejected,repairs=compile_v2(original['calls'][1]['parsed'],inputs[original['id']],ids)
                ok,errors=validate(obj,inputs[original['id']])
                pred.update(parsed=obj,valid=ok,errors=errors,rejected=rejected,repairs=repairs)
            except (ValueError,TypeError):pass
        preds.append(pred)
    with (a.out/'predictions.jsonl').open('w') as f:
        for pred in preds:f.write(json.dumps(pred,ensure_ascii=False)+'\n')
    dump(a.out/'metrics.json',evaluate(preds,labels))
    dump(a.out/'manifest.json',{'source_predictions_sha256':file_sha(source),'code_sha256':file_sha(__file__),
                              'amendment_sha256':file_sha(Path(__file__).with_name('RULE_AMENDMENT.md')),
                              'additional_model_calls':0,'repaired_articles':sum(bool(r['repairs']) for r in preds),
                              'training':False})

if __name__=='__main__':main()
