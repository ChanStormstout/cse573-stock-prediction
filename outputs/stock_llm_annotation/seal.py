"""Export resolved model-provisional labels for SFT, never a human acceptance seal."""
import argparse
import collections
import json
from pathlib import Path
from deduplicate import group_labels
from labels import load_panel, rows, sha, dump, write_rows, validate_label, PROMPT

def seal(data, comparisons, decisions, out):
    inputs, manifest=load_panel(data); lookup={r['id']:r for r in inputs}
    merged={}
    audits=set()
    for directory in sorted(Path(comparisons).iterdir()):
        if not (directory/'comparison.jsonl').exists(): continue
        summary=json.loads((directory/'summary.json').read_text());audits.update(summary['agreement_audit_ids'])
        for r in rows(directory/'comparison.jsonl'):
            if r['id'] in merged: raise ValueError('Duplicate comparison ID')
            merged[r['id']]=r
    decision_path=Path(decisions)
    decision_rows=rows(decision_path)
    decisions={r['id']:r for r in decision_rows}
    if len(decisions)!=len(decision_rows): raise ValueError('Duplicate adjudication ID')
    if set(decisions)-set(lookup): raise ValueError('Foreign adjudication ID')
    if set(merged)-set(lookup): raise ValueError('Foreign comparison ID')
    accepted=[]; excluded=[]
    for key,r in merged.items():
        decision=decisions.get(key)
        if decision:
            if decision.get('input_sha256')!=lookup[key]['input_sha256'] or not decision.get('reviewer') or not decision.get('rationale'): raise ValueError('Invalid adjudication provenance')
            label=decision['label']
        elif r['state']=='AGREED_PROVISIONAL' and key not in audits:
            label=r['a']
        else:
            excluded.append({'id':key,'reason':'pending_adjudication_or_agreement_audit'});continue
        problems=validate_label(label,lookup[key])
        if problems: raise ValueError((key,problems))
        if label['status'] not in {'none','events'}:
            excluded.append({'id':key,'reason':label['status']});continue
        inp=lookup[key]
        accepted.append({'id':key,'split':inp['split'],'symbol':inp['symbol'],'event_group':inp['group'],'available_utc':inp['available_utc'],'answer':{'events':label['events']},'input_sha256':inp['input_sha256'],'annotation_status':'DUAL_GPT_PROVISIONAL_NOT_HUMAN_GOLD','independent_reviewer':None})
    missing=set(lookup)-set(merged)
    if missing: raise ValueError(f'Incomplete annotation coverage: {len(missing)} missing inputs; no premature training seal')
    if any(r['reason']=='pending_adjudication_or_agreement_audit' for r in excluded):
        raise ValueError('Unresolved adjudications or agreement audits; no premature training seal')
    # Check rows are retained for later evaluation, never consumed by the SFT loader.
    accepted, duplicates, links=group_labels(accepted,lookup)
    excluded.extend(duplicates)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_rows(out/'duplicate_links.jsonl',links)
    write_rows(out/'inputs.jsonl',[lookup[r['id']] for r in accepted]);write_rows(out/'labels.jsonl',accepted)
    dump(out/'label_seal.json',{'sha256':sha((out/'labels.jsonl').read_bytes()),'inputs_sha256':sha((out/'inputs.jsonl').read_bytes()),'source_inputs_sha256':manifest['inputs_sha256'],'status':'MODEL_PROVISIONAL_NOT_INDEPENDENT','extractor_frozen_at':manifest['extractor_earliest_freeze'],'excluded':excluded,'n':len(accepted),'counts':dict(collections.Counter(r['split']+'|'+r['symbol']+'|'+('events' if r['answer']['events'] else 'none') for r in accepted)),'decisions_sha256':sha(decision_path.read_bytes()),'comparison_hashes':{str(p.relative_to(comparisons)):sha(p.read_bytes()) for p in sorted(Path(comparisons).glob('*/*.json*'))},'independent_review_passed':False,'prompt_sha256':sha(PROMPT.encode())})
    print(json.dumps({'accepted':len(accepted),'excluded':len(excluded),'independent_review_passed':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['data','comparisons','decisions','out']:p.add_argument('--'+k,required=True)
    seal(**vars(p.parse_args()))
