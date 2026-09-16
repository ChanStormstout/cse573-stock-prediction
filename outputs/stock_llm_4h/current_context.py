"""Deterministic removal of explicit historical/other-broker-list tails.
Registered from TRAIN-only oracle diagnosis; no labels used for selection.
"""
from common import *
MARKER=re.compile(r'(?:a number of|several|other|a number of other)\s+(?:other\s+)?(?:research\s+|equities\s+)?(?:analysts|brokerages|research firms|research analysts)|among\s+\d+\s+analysts\s+covering',re.I)

def trim(row):
 r=dict(row);keep=[]
 for k,text in r['sentences'].items():
  if k!='S0' and MARKER.search(text):break
  keep.append(k)
 r['sentences']={k:r['sentences'][k] for k in keep};r['spans']={k:r['spans'][k] for k in keep}
 return r

def main():
 src=B/'data/pilot_v3';out=new_run(B/'data/pilot_current_v1');original=rows(src/'inputs.jsonl');new=[trim(r) for r in original];write_rows(out/'inputs.jsonl',new)
 # Copy labels unchanged except input fingerprints, then audit evidence deletion.
 lookup={r['id']:r for r in new};labels=rows(src/'labels.jsonl');missing=[]
 for l in labels:
  kept=lookup[l['id']]['sentences'];lost=sorted({k for e in l['answer']['events'] for k in e['evidence_ids']} - set(kept))
  if lost:missing.append({'id':l['id'],'removed_gold_ids':lost})
  l['input_sha256']=digest(json.dumps(kept,sort_keys=True).encode())
 write_rows(out/'labels.jsonl',labels)
 dump(out/'manifest.json',{'source_sha256':file_sha(src/'inputs.jsonl'),'input_sha256':file_sha(out/'inputs.jsonl'),'labels_sha256':file_sha(out/'labels.jsonl'),'code_sha256':file_sha(__file__),'changed_articles':sum(a['sentences']!=b['sentences'] for a,b in zip(original,new)),'removed_sentences':sum(len(a['sentences'])-len(b['sentences']) for a,b in zip(original,new)),'removed_gold_evidence':missing,'label_free_selection':True,'formal_quality_passed':False})
 assert not missing,'Keep deletion report; do not run inference claiming equivalent target coverage'
 print((out/'manifest.json').read_text())
if __name__=='__main__':main()
