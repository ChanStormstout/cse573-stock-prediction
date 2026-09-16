"""M06 full schema and evidence graph; quality is deliberately not self-approved."""
from core import *
import re
from news_diagnostics import groups_online
sp=importlib.util.spec_from_file_location('old_event_v2',O/'stock_event_facts_v2/extract.py');v2=importlib.util.module_from_spec(sp);sp.loader.exec_module(v2)
RATE=r'(?:strong[ -]buy|market perform|equal[ -]weight|outperform|underperform|overweight|underweight|neutral|hold|buy|sell)'
PAIR=re.compile(r'from\s+(?:a\s+)?[“\"\']?('+RATE+r')[”\"\']?\s*(?:rating\s+)?to\s+(?:a\s+)?[“\"\']?('+RATE+r')',re.I)
REVERSE=re.compile(r'to\s+(?:a\s+)?[“\"\']?('+RATE+r')[”\"\']?\s*(?:rating\s+)?from\s+(?:a\s+)?[“\"\']?('+RATE+r')',re.I)
# Conservative canonical spans. Unknown issuer stays None, not string guessed from headline tail.
ISSUERS=['Wells Fargo & Co','Wells Fargo','Morgan Stanley','Atlantic Equities','Goldman Sachs','Bank of America','Deutsche Bank','Credit Suisse','JPMorgan','JP Morgan','Barclays','Citigroup','UBS','Jefferies','Piper Jaffray','BMO Capital Markets','RBC Capital Markets','Royal Bank of Canada','Mizuho','KeyBanc Capital Markets','Stifel Nicolaus','Loop Capital','Oppenheimer','Nomura','Canaccord Genuity','Raymond James','Cowen']
def canonical_actor(span):
    if not span:return None
    for name in ISSUERS:
        if re.match(re.escape(name)+r'(?:\b|\.)',span,re.I):return name
    return None

def normalize(s):return ' '.join(v2.clean(s).split())
def extract(symbol,key,title,body,published=None,available=None):
    fs=v2.extract(symbol,key,title,body);out=[];cp=v2.COMP[symbol];normalized=normalize(body)
    for f in fs:
        f=f.copy();f['unknown_fields']=[];f['extraction_status']='provisional'
        # Explicit corrections override ambiguous headline actions: abstain and retain diagnostic record.
        if re.search(r'\bcorrection\b|not\s+(?:a\s+)?downgrade|only\s+(?:cut|lowered)\s+(?:its\s+)?(?:price\s+)?target',body,re.I) and f['kind']=='analyst_rating':
            f['extraction_status']='conflict_abstain';f['action']='unknown';f['rating_action']='unknown'
        if f['kind']=='analyst_rating' and f['action'] in ['up','down']:
            candidates=[]
            for sent in v2.sentences(body):
                if not re.search(cp,sent,re.I) or re.search(r'previously|last year|other analysts|correction',sent,re.I):continue
                a=PAIR.search(sent);b=REVERSE.search(sent)
                if (a or b) and re.search(r'upgrad|downgrad',sent,re.I):
                    old,new=a.groups() if a else b.groups()[::-1]
                    if old.lower()!=new.lower():candidates.append((sent,old,new))
            if len(candidates)==1:f['evidence'],f['old_rating'],f['new_rating']=candidates[0]
            else:f['action']='unknown';f['rating_action']='unknown';f['extraction_status']='ambiguous_rating_abstain'
        oldactor=f.get('actor');f['actor']=canonical_actor(re.sub(r'^Analysts at\s+','',oldactor or '',flags=re.I)) if f['kind']!='revenue_guidance' else symbol
        if f['actor'] is None:f['unknown_fields'].append('actor')
        if f['kind']=='analyst_target':
            f['unit']='USD per share' if '$' in f['evidence'] else None
            if re.search(r'previously|last year|other analysts|historical',f['evidence'],re.I):f['extraction_status']='historical_abstain'
            if f.get('old_value') is not None and f.get('new_value') is not None:
                direction='up' if f['new_value']>f['old_value'] else 'down' if f['new_value']<f['old_value'] else 'maintain'
                if f['action'] not in [direction,'set','unknown']:f['extraction_status']='numeric_direction_conflict'
            f['period']=None # price target horizon not inferred
        if f['kind']=='revenue_guidance':
            if not f.get('period') or not re.search(r'20\d\d',f['period']):f['unknown_fields'].append('fiscal_year');f['extraction_status']='period_unknown'
            if f.get('lower',0)>f.get('upper',0):f['extraction_status']='range_invalid'
        ev=normalize(f['evidence']);pos=normalized.find(ev)
        if pos>=0:source='normalized_body';source_text=normalized
        elif normalize(title).find(ev)>=0:source='normalized_title';source_text=normalize(title);pos=source_text.find(ev)
        else:source='unresolved';source_text='';pos=-1;f['extraction_status']='evidence_unresolved'
        f.update(target_company=symbol,evidence=ev,evidence_source=source,evidence_start=pos,evidence_end=pos+len(ev) if pos>=0 else -1,evidence_text_sha256=hashlib.sha256(source_text.encode()).hexdigest(),published_utc=str(published),available_utc=str(available),first_disclosure_time=None)
        f['unknown_fields'].append('first_disclosure_time');f['formal_quality_status']='pending_independent_review';out.append(f)
    return out

def run(d,out):
    bodies=json.loads((O/'stock_event_facts/results/bodies.json').read_text());n=pd.read_pickle(W/'audit/news_index.pkl');n['key']=n.archive+'::'+n.member;n=n.set_index('key',drop=False);allframes=[];reviewed=set()
    for fn in [O/'stock_event_facts/results/assistant_review_v1.json',O/'stock_event_facts_v2/results/assistant_review.json',O/'stock_event_facts_v3/results/assistant_review.json']:
        reviewed.update(x['record_key'] for x in json.loads(fn.read_text()))
    for stock,s in d.groupby('symbol'):
        used=sorted({k for x in s.news_record_keys.fillna('') for k in x.split('|') if k});q=n.loc[used].reset_index(drop=True).sort_values(['available_utc','key']);q['group']=groups_online(q);prior=set(q.loc[q.key.isin(reviewed),'group'])
        for r in q.itertuples(index=False):
            subset='development' if r.group in prior or int(hashlib.sha256((stock+r.group).encode()).hexdigest()[:8],16)%3 else 'locked_check'
            for f in extract(stock,r.key,r.title,bodies[r.key],r.published_utc,r.available_utc):
                f.update(event_group=stock+'|'+r.group,quality_split=subset,title=r.title);allframes.append(f)
    (out/'events.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allframes));a=pd.DataFrame(allframes);a.to_csv(out/'all_frames.csv',index=False)
    review=a.drop_duplicates(['event_group','kind']).copy();review['human_reviewer']='';review['human_key_fields_correct']='';review['human_missed_events']='';review['human_notes']='';review.to_csv(out/'human_review.csv',index=False)
    # Relationship storage is inspectable evidence, not a trained GNN.
    graph=[]
    for i,f in enumerate(allframes):
        eid=f'event:{i}';graph.extend([dict(source='article:'+f['record_key'],relation='evidence_for',target=eid),dict(source=eid,relation='targets',target='company:'+f['symbol'])])
        if f['actor']:graph.append(dict(source='actor:'+f['actor'],relation='issued',target=eid))
    dump(out/'evidence_graph.json',{'nodes':allframes,'edges':graph,'quality':'unapproved; do not infer causal relations'})
    counts=review.groupby(['symbol','kind','quality_split']).size().rename('n').reset_index();counts.to_csv(out/'quality_counts.csv',index=False)
    dump(out/'quality_gate.json',{'independent_reviews':0,'full_event_approved':False,'graph_approved':False,'M07_training':'blocked by missing independent field approval; code/test only','required':'at least30 nonduplicate checked cases per used type,>=90% key field correctness plus coverage/misses','assistant_labels':'provisional only','known_error_fixes':'conservative unknown issuer, explicit old/new ratings, correction abstention, unit/period checks; coverage changes not proof of full quality'})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();d=load(a.run.resolve());stage(a.run.resolve(),'M06',lambda out:run(d,out))
