"""Evidence-bearing conservative rules. Schema success is NOT human acceptance."""
import re,hashlib,json
from pathlib import Path
import pandas as pd
COMP={'AAPL':r'\b(?:Apple|AAPL)\b','AMZN':r'\b(?:Amazon(?:\.com)?|AMZN)\b'}
PT=r'(?:price\s+(?:target|objective)|(?:target|objective)\s+price)'
MONEY=r'\$\s*([\d,]+(?:\.\d+)?)'
RATE=r'(?:strong[ -]buy|market perform|equal[ -]weight|outperform|underperform|overweight|underweight|neutral|hold|buy|sell)'
HIST=re.compile(r'\b(?:previously|last year|other analysts|in other analyst news|historical ratings|also reiterated|also reaffirmed)\b',re.I)
PROMO=re.compile(r'InvestorsObserver|(?:click|read|view).{0,30}(?:full report|options report)|trade ideas.{0,40}https?://|sign up.{0,40}newsletter',re.I)
ISSUERS=['DA Davidson','D.A. Davidson','RBC Capital Markets','RBC','Citigroup','Citi','Bank of America','Goldman Sachs','Morgan Stanley','JPMorgan','J.P. Morgan','Barclays','Jefferies','Piper Jaffray','Credit Suisse','UBS','Deutsche Bank','Wells Fargo','BMO Capital Markets','Oppenheimer','Raymond James','Canaccord Genuity','Stifel']

def spans(text):
    start=0
    for m in re.finditer(r'\n+|(?<=[.!?])\s+(?=[A-Z])',text):
        if '\n' not in m.group() and re.search(r'\b(?:Inc|Corp|Co|Mr|Dr|[A-Z])\.$',text[max(start,m.start()-12):m.start()]):continue
        a,b=start,m.start()
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if b>a:yield a,b,text[a:b]
        start=m.end()
    a,b=start,len(text)
    while a<b and text[a].isspace():a+=1
    while b>a and text[b-1].isspace():b-=1
    if b>a:yield a,b,text[a:b]

def role(symbol,title,body):
    cp=COMP[symbol]; evidence=[];comparison=False
    for a,b,s in spans(body):
        if not re.search(cp,s,re.I):continue
        if re.search(r'\b(?:versus|competes?|competing|competitor|rival|compared|comparison|catch up|ahead of|market capitali[sz]ation|ratio of|non-risk-adjusted ranking)\b',s,re.I):comparison=True
        # Target must govern a factual action, not generic "is/has/stock" co-occurrence.
        direct=re.search(cp+r"(?:\s*\([^)]*\))?(?:['’]s)?\s+(?:(?:now|also|reportedly)\s+)?(?:said|announced|reported|expects?|forecasts?|projects?|plans?|launched|introduced|acquired|agreed|hired|cut|raised|reduced|increased|will\s+(?:open|launch|invest|hire)|has\s+(?:announced|launched|agreed|acquired)|was\s+(?:rated|upgraded|downgraded)|had\s+its\s+(?:rating|price target))\b",s,re.I)
        obj=re.search(r'(?:'+PT+r'|rating)\s+(?:on|for|of)\s+(?:shares of\s+)?'+cp,s,re.I) or re.search(r'(?:upgraded|downgraded)\s+(?:shares of\s+)?'+cp,s,re.I)
        if direct or obj:evidence.append({'start':a,'end':b,'text':s})
    marketing=bool(PROMO.search(title+'\n'+body))
    if evidence and not marketing:return 'direct_rule_candidate',evidence[:3],marketing
    if comparison:return 'comparison_candidate',evidence[:3],marketing
    if re.search(cp,body,re.I) or re.search(cp,title,re.I):return 'mention_or_promotion',evidence[:3],marketing
    return 'no_target',[],marketing

def number(x):return float(x.replace(',',''))
def actor(s):
    hits=[v for v in ISSUERS if re.search(r'(?<!\w)'+re.escape(v)+r'(?!\w)',s,re.I)]
    hits=[v for v in hits if not any(v!=w and v.lower() in w.lower() for w in hits)]
    return hits[0] if len(hits)==1 else None

def extract(symbol,key,title,body,published=None,available=None):
    cp=COMP[symbol];other=COMP['AMZN' if symbol=='AAPL' else 'AAPL'];events=[]
    for source,text in [('title',title),('body',body)]:
        for a,b,s in spans(text):
            if not re.search(cp,s,re.I) or len(s)>1800:continue
            # Abstain on unresolved multiple target objects; retain the raw span for review.
            if re.search(other,s,re.I):continue
            history=bool(HIST.search(s));issuer=actor(s)
            common=dict(symbol=symbol,record_key=key,actor=issuer,published_utc=published,available_utc=available,first_disclosure_utc=None,period=None,unit=None,old_value=None,new_value=None,source=source,start=a,end=b,evidence=s,evidence_sha256=hashlib.sha256(s.encode()).hexdigest(),document_sha256=hashlib.sha256(text.encode()).hexdigest(),temporal_role='historical' if history else 'unknown',quality_status='UNREVIEWED',prediction_accepted=False,conflict=False,unknown_fields=['first_disclosure_utc'])
            if issuer is None:common['unknown_fields'].append('actor')
            target_link=bool(re.search(cp+r"(?:['’]s)?[^.;]{0,90}"+PT,s,re.I) or re.search(PT+r'[^.;]{0,40}(?:on|for|of)\s+(?:shares of\s+)?'+cp,s,re.I))
            if re.search(PT,s,re.I) and target_link:
                fromto=re.search(r'from\s+'+MONEY+r'\s+to\s+'+MONEY,s,re.I);tofrom=re.search(r'to\s+'+MONEY+r'\s+from\s+'+MONEY,s,re.I)
                pair=fromto.groups() if fromto else tofrom.groups()[::-1] if tofrom else None
                new=None
                if pair:old,new=map(number,pair)
                else:
                    old=None;m=re.search(PT+r'[^.;]{0,70}?\bto\s+'+MONEY,s,re.I) or re.search(PT+r'\s+(?:of|at)\s+'+MONEY,s,re.I)
                    if m:new=number(m.group(1))
                if new is not None and new>0:
                    action='up' if old and new>old else 'down' if old and new<old else 'maintain' if old and new==old else 'unknown'
                    # Currency/percentage ambiguity is an explicit rejection, not a guess.
                    currency_conflict=bool(re.search(r'(?:C\$|A\$|HK\$|£|€|CAD|AUD|HKD|EUR|GBP)',s))
                    f={**common,'unknown_fields':common['unknown_fields'].copy(),'kind':'analyst_target','action':action,'old_value':old,'new_value':new,'unit':'USD per share','change':new/old-1 if old else None,'numeric_valid':not currency_conflict}
                    if old is None:f['unknown_fields'].append('old_value')
                    if currency_conflict:f['conflict']=True
                    events.append(f)
            if re.search(r'\brating\b',s,re.I):
                transition=re.search(r'from\s+(?:a\s+)?[“"\']?('+RATE+r')[”"\']?\s*(?:rating\s+)?to\s+(?:a\s+)?[“"\']?('+RATE+r')',s,re.I)
                maintain=re.search(r'\b(?:maintain\w*|reiterat\w*|reaffirm\w*|reissu\w*)\b[^.;]{0,45}\b('+RATE+r')\b',s,re.I)
                directed=re.search(r'(?:rating\s+(?:on|for)|upgrad\w*|downgrad\w*)\s+(?:shares of\s+)?'+cp,s,re.I) or re.match(cp,s,re.I)
                if (transition or maintain) and directed:
                    action='maintain' if maintain and not transition else 'upgrade' if re.search(r'upgrad',s,re.I) else 'downgrade' if re.search(r'downgrad',s,re.I) else 'transition'
                    events.append({**common,'kind':'analyst_rating','action':action,'old_rating':transition.group(1) if transition else None,'new_rating':transition.group(2) if transition else maintain.group(1)})
            direct=re.search(cp+r'\s+(?:now\s+)?(?:expects?|forecasts?|projects?|guides?)\b',s,re.I)
            if direct and re.search(r'\brevenue\b',s,re.I) and not re.search(r'\b(?:per share|analysts expect|might|could)\b',s,re.I):
                m=re.search(r'\$([\d.]+)\s*(billion|million|bn)?\s*(?:and|to|[-–])\s*\$([\d.]+)\s*(billion|million|bn)\b',s,re.I)
                per=re.search(r'\b(?:fiscal\s+20\d\d\s+(?:first|second|third|fourth)\s+quarter|(?:first|second|third|fourth|next|current)\s+quarter(?:\s+of\s+(?:fiscal\s+)?20\d\d)?|Q[1-4](?:\s+20\d\d)?)\b',s,re.I)
                if m:
                    lo,u1,hi,u2=m.groups();scale={'billion':1e9,'bn':1e9,'million':1e6};lo=float(lo)*scale[(u1 or u2).lower()];hi=float(hi)*scale[u2.lower()]
                    if lo<=hi:events.append({**common,'kind':'revenue_guidance','action':'issued_range','actor':symbol,'lower':lo,'upper':hi,'new_value':(lo+hi)/2,'unit':'USD revenue','period':per.group(0) if per else None,'numeric_valid':True})
    headline=[f for f in events if f['source']=='title']
    for f in events:
        same=[h for h in headline if h['kind']==f['kind'] and h.get('actor')==f.get('actor')]
        if f['source']=='body' and same:
            if any((h.get('new_value') is not None and f.get('new_value') is not None and h['new_value']!=f['new_value']) or (h.get('new_rating') and f.get('new_rating') and h['new_rating'].lower()!=f['new_rating'].lower()) for h in same):
                f['conflict']=True
                for h in same:h['conflict']=True
            elif not f['temporal_role']=='historical':f['temporal_role']='headline_supported_not_first_disclosure'
        if re.search(r'\b(?:correction|corrected|incorrectly|erroneously)\b',title+' '+f['evidence'],re.I):f['conflict']=True
        f['prediction_accepted']=False
        f['event_id']=hashlib.sha256((key+'|'+symbol+'|'+f['kind']+'|'+f['source']+'|'+str(f['start'])).encode()).hexdigest()[:20]
    return events

def quality_gate(review_path,expected_path):
    """Called afresh on every invocation; a restart cannot bypass a failed review."""
    expected=json.loads(Path(expected_path).read_text());d=pd.read_csv(review_path,keep_default_na=False)
    if set(d.sample_id)!=set(expected):raise ValueError('REVIEW_INPUT_MISMATCH')
    for row in d.itertuples():
        if row.input_sha256!=expected[row.sample_id]:raise ValueError('REVIEW_FINGERPRINT_MISMATCH')
        if hashlib.sha256(row.proposed_fields.encode()).hexdigest()!=row.input_sha256:raise ValueError('REVIEW_CONTENT_MISMATCH')
        proposal=json.loads(row.proposed_fields)
        if proposal['event_group']!=row.event_group or proposal['symbol']!=row.symbol or proposal['kind']!=row.kind:raise ValueError('REVIEW_GROUP_MISMATCH')
    reviewed=d[d.reviewer.str.strip().ne('')]
    if len(reviewed)!=len(d):return {'status':'WAITING_INDEPENDENT_REVIEW','reviewed':len(reviewed),'required':len(d),'accepted_types':[]}
    if reviewed.reviewer.str.contains('assistant|codex|chatgpt|gpt',case=False).any():raise ValueError('ASSISTANT_NOT_INDEPENDENT')
    results={};accepted=[]
    for kind,g in reviewed.groupby('kind'):
        values=pd.to_numeric(g.critical_fields_correct,errors='coerce')
        if not values.isin([0,1]).all() or g.missed_target.eq('').any() or g.uncertain.eq('').any():raise ValueError('INCOMPLETE_REVIEW_FIELDS')
        n=g.event_group.nunique();acc=float(values.mean());results[kind]={'nonduplicate_groups':n,'accuracy':acc}
        if n>=30 and acc>=.9:accepted.append(kind)
    return {'status':'PASSED_LIMITED_TYPES' if accepted else 'FAILED_OR_INSUFFICIENT_PER_TYPE','accepted_types':accepted,'by_type':results,'caveat':'Declared reviewer identity, not externally authenticated; accuracy is separate from forecast utility.'}
