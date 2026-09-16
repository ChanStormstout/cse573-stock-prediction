"""V3 narrows v2 to explicit rating transitions and cleans issuer spans.
Guidance saved for coverage diagnostics only. No predictor run here.
"""
import json,re,sys,datetime
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import *
sys.path.insert(0,str(O/'stock_event_facts_v2'))
from extract import sentences,COMP
RATE=r'(?:strong[ -]buy|market perform|equal[ -]weight|outperform|underperform|overweight|underweight|neutral|hold|buy|sell)'
FROMTO=re.compile(r'from\s+(?:a\s+)?[“"\']?('+RATE+r')[”"\']?\s*(?:rating\s+)?to\s+(?:a\s+)?[“"\']?('+RATE+r')',re.I)
TOFROM=re.compile(r'to\s+(?:a\s+)?[“"\']?('+RATE+r')[”"\']?\s*(?:rating\s+)?from\s+(?:a\s+)?[“"\']?('+RATE+r')',re.I)
def refine(f,body):
    f=f.copy()
    if f.get('actor'):
        f['actor']=re.sub(r'^Analysts at\s+','',f['actor'],flags=re.I)
        f['actor']=re.sub(r'\s+to\s+'+RATE+r'$','',f['actor'],flags=re.I)
    if f['kind']=='analyst_rating' and f['action'] in ['up','down']:
        found=None
        for sentence in sentences(body):
            if not re.search(COMP[f['symbol']],sentence,re.I):continue
            a=FROMTO.search(sentence);b=TOFROM.search(sentence)
            if (a or b) and re.search(r'upgrad|downgrad',sentence,re.I) and not re.search(r'\b(?:last year|previously|correction)\b',sentence,re.I):
                old,new=a.groups() if a else b.groups()[::-1]
                if old.lower()==new.lower():continue
                found=(sentence,old,new);break
        if found is None:return None
        f['evidence'],f['old_rating'],f['new_rating']=found;f['q']=.5
    return f
if __name__=='__main__':
    R=B/'results';require_empty(R);R.mkdir(exist_ok=True);paths=[B/'prepare.py',O/'stock_event_facts_v2/results/events.jsonl',O/'stock_event_facts_v2/results/assistant_review.json',O/'stock_event_facts_v2/extract.py',O/'stock_event_facts/extract.py',O/'stock_event_facts/results/bodies.json'];P=dict(id='E14_V3',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rule='v2 frames plus issuer cleanup; upgrade/downgrade rating requires explicit old/new rating in body sentence naming company; otherwise abstain',predictor_types=['analyst_target','analyst_rating'],guidance='diagnostic only; insufficient unused review cases and few distinct disclosures',review='assistant provisional checks on previously unreviewed training record keys; >=90% per supported type, at least 10 new records combined; AMZN low coverage reported and not claimed independently validated; human review pending',source_sha256={str(p.relative_to(O.parent)):sha(p) for p in paths});(B/'pilot_protocol.json').write_text(json.dumps(P,indent=2));bodies=json.loads((O/'stock_event_facts/results/bodies.json').read_text());records=[]
    for line in (O/'stock_event_facts_v2/results/events.jsonl').read_text().splitlines():
        x=json.loads(line);f=refine(x,bodies[x['record_key']])
        if f:records.append(f)
    (R/'events.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in records))
    seen={x['record_key'] for path in [O/'stock_event_facts/results/assistant_review_v1.json',O/'stock_event_facts_v2/results/assistant_review.json'] for x in json.loads(path.read_text())};new=[x for x in records if x['split']=='train' and x['kind'] in P['predictor_types'] and x['record_key'] not in seen];rng=np.random.default_rng(577);rng.shuffle(new);new=new[:40]
    for i,x in enumerate(new):x.update(review_id=i,assistant_key_facts_correct=None,human_key_facts_correct=None,review_note='')
    (R/'review_input.json').write_text(json.dumps(new,indent=2,ensure_ascii=False,allow_nan=False));verify_files(O.parent,P['source_sha256']);print(pd.DataFrame(records).groupby(['symbol','split','kind']).size().to_string());print('new checks',len(new))
