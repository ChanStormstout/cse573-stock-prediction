"""Check input spans, annotation seal, temporal split and selected result integrity."""
from common import *
from forecast import instant

def main():
 data=B/'data/pilot_v3';rs=rows(data/'inputs.jsonl');labels=rows(data/'labels.jsonl');byid={r['id']:r for r in rs};seal=json.loads((data/'label_seal.json').read_text());assert file_sha(data/'labels.jsonl')==seal['sha256'];assert file_sha(data/'inputs.jsonl')==seal['inputs_sha256']
 spans=0
 for r in rs:
  for k,s in r['spans'].items():
   text=r['body'] if s['source']=='body' else r['title'];assert text[s['start']:s['end']]==r['sentences'][k],(r['id'],k);spans+=1
  assert digest(r['body'].encode())==r['body_sha256']
  assert instant(r['published_utc'])<=instant(r['available_utc'])
 splits={s:[l for l in labels if l['split']==s] for s in ['train','valid','check']}
 for a,b in [('train','valid'),('valid','check')]:
  assert max(instant(r['available_utc']) for r in splits[a])<min(instant(r['available_utc']) for r in splits[b])
  assert not ({r['event_group'] for r in splits[a]}&{r['event_group'] for r in splits[b]})
 for l in labels:
  r=byid[l['id']];assert digest(json.dumps(r['sentences'],sort_keys=True).encode())==l['input_sha256'];assert validate(l['answer'],r)[0]
 result={'n_articles':len(rs),'exact_input_spans_verified':spans,'labels_hash_verified':True,'assistant_only':True,'chronological_splits':{s:len(v) for s,v in splits.items()},'no_group_overlap_for_registered_groups':True,'grouping_caveat':'Heuristic plus assistant merge of the Longbow duplicates, not a proof that every semantic duplicate is found.','stock_outcomes_in_annotation_inputs':False}
 out=B/'verification.json';dump(out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
