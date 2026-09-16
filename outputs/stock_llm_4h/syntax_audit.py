"""Offline frozen-output format audit; never changes primary metrics or labels."""
from common import *

def main():
 out=new_run(B/'runs/syntax_audit_v1');inputs={r['id']:r for r in rows(B/'data/pilot_v3/inputs.jsonl')};preds=rows(B/'runs/frozen_v1/predictions.jsonl');labels=rows(B/'data/pilot_v3/labels.jsonl');repaired=[];changes=[]
 for p in preds:
  obj=p.get('parsed')
  if obj is None:
   try:
    candidate,end=json.JSONDecoder().raw_decode(p['raw'].strip());trailing=p['raw'].strip()[end:]
    # Permit only trailing quote/brace punctuation, not alternative prose/second answers.
    if trailing and all(c in '"} \n' for c in trailing):obj=candidate;changes.append(p['id'])
   except (ValueError,TypeError):pass
  ok,why=validate(obj,inputs[p['id']]);repaired.append({'id':p['id'],'parsed':obj,'valid':ok,'errors':why})
 dump(out/'summary.json',{'punctuation_only_recovered_ids':changes,'n_recovered_json':len(changes),'unchanged_primary_results':True,'metrics':evaluate(repaired,labels),'conclusion_boundary':'Post-hoc format diagnostic; no invented or corrected financial facts. Not a new forecasting result.'})
if __name__=='__main__':main()
