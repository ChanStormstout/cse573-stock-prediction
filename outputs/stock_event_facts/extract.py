"""Conservative evidence-bearing rating/target and company revenue-guidance frames.
Unknown is explicit. Publication is an age proxy, never claimed first disclosure.
"""
import re,html
COMP={'AAPL':r'\b(?:Apple(?:\s+Inc\.)?|AAPL)\b','AMZN':r'\b(?:Amazon(?:\.com)?(?:\s+Inc\.)?|AMZN)\b'}
PT=r'(?:price\s+(?:target|objective)|(?:target|objective)\s+price)'
MONEY=r'\$\s*([\d,]+(?:\.\d+)?)'

def clean(text):
    text=re.sub(r'</(?:p|div|li|h\d)>|<br\s*/?>','\n',text,flags=re.I)
    return html.unescape(re.sub(r'<[^>]+>',' ',text)).replace('\xa0',' ')

def sentences(text):
    for para in clean(text).splitlines():
        para=' '.join(para.split())
        if not para:continue
        start=0
        for m in re.finditer(r'[.!?]\s+(?=[A-Z])',para):
            prev=re.search(r'([A-Za-z]+)$',para[:m.start()]);word=prev.group(1) if prev else ''
            if word.lower() in ['inc','corp','co','com','mr','dr','mrs'] or (len(word)==1 and word.isupper()):continue
            yield para[start:m.start()+1];start=m.end()
        if start<len(para):yield para[start:]

def money(x):return float(x.replace(',',''))

def frame(symbol,key,kind,evidence):
    return dict(symbol=symbol,record_key=key,kind=kind,evidence=evidence,actor=None,action='unknown',old_value=None,new_value=None,change=None,rating_action='unknown',period=None,unit=None,current_disclosure='unknown',first_disclosure_time=None,q=.5)

def extract(symbol,key,title,body):
    cp=COMP[symbol];out=[];ss=list(sentences(body))
    # Restrict analyst frames to analyst-topic headlines, not boilerplate in holdings articles.
    analyst_topic=bool(re.search(PT+r'|\b(?:analyst|upgrades?|downgrades?|ratings?|reiterates?|reaffirms?)\b',title,re.I))
    if analyst_topic:
        for evidence in ss:
            if len(evidence)>1600 or not re.search(cp,evidence,re.I) or not re.search(PT,evidence,re.I):continue
            if re.search(r'\b(?:previously|last year|other analysts|also reiterated|also reaffirmed)\b',evidence,re.I):continue
            # Explicit target-as-object templates, not mere co-occurrence.
            direct=re.search(cp+r'(?:\s*\([^)]*\))?\s+(?:had its|has been|was|is|price target|stock price target)',evidence,re.I)
            reverse=re.search(PT+r'(?:\s+\w+){0,3}\s+(?:of|on|for)\s+(?:shares of\s+)?'+cp,evidence,re.I)
            if not (direct or reverse):continue
            f=frame(symbol,key,'analyst_target',evidence)
            a=re.search(r'from\s+'+MONEY+r'\s+to\s+'+MONEY,evidence,re.I)
            b=re.search(r'to\s+'+MONEY+r'\s+from\s+'+MONEY,evidence,re.I)
            if a:f['old_value'],f['new_value']=map(money,a.groups())
            elif b:f['new_value'],f['old_value']=map(money,b.groups())
            else:
                m=re.search(r'(?:to|given a|assigned a|issued a|set a)\s+'+MONEY,evidence,re.I)
                if m:f['new_value']=money(m.group(1))
            if f['old_value'] and f['new_value']:
                f['change']=f['new_value']/f['old_value']-1
                f['action']='up' if f['change']>0 else 'down' if f['change']<0 else 'maintain'
                f['q']=1.
            else:
                tail=evidence[re.search(PT,evidence,re.I).start():]
                f['action']='up' if re.search(r'\b(?:raised|boosted|increased|lifted)\b',tail[:140],re.I) else 'down' if re.search(r'\b(?:cut|lowered|trimmed|reduced)\b',tail[:140],re.I) else 'set' if f['new_value'] else 'unknown'
            if f['action']=='unknown' and f['new_value'] is None:continue
            f['unit']='USD per share'
            actor=re.search(r'\bby\s+(?:equities researchers at\s+|stock analysts at\s+|analysts at\s+)?([A-Z][A-Za-z.&\s-]{1,65}?)(?=\s+(?:from|to|in|on)\b|\.$|$)',evidence)
            if actor:f['actor']=actor.group(1).strip()
            if re.search(r'\b(?:reiterated|reaffirmed|maintained|reissued)\b',evidence,re.I):f['rating_action']='maintain'
            # Reliable title-level agreement is recorded, not inferred first disclosure.
            title_money=[money(x) for x in re.findall(MONEY,title)]
            if f['new_value'] in title_money:f['current_disclosure']='headline_supported'
            if re.search(r'\b(?:upgrades and downgrades|price target changes)\b',title,re.I):f['current_disclosure']='roundup_current_unknown'
            out.append(f);break
    # Guidance: company itself must explicitly forecast revenue, with a numeric range.
    for evidence in ss:
        if len(evidence)>1600:continue
        m=re.search(cp+r'(?:\s*\([^)]*\))?\s+(?:(?:now|currently|also)\s+)?(?:expects?|forecasts?|projects?|guides?|guided|forecasted)\b',evidence,re.I)
        if not m or re.search(r'\b(?:might|could|would|analysts expect|expected to)\b',evidence[:m.end()],re.I):continue
        tail=evidence[m.start():]
        if not re.search(r'\brevenue\b',tail,re.I):continue
        rg=re.search(r'(?:between|of|to|from)\s+\$(?:US)?\s*([\d.]+)\s*(billion|million|bn|b)?\s*(?:and|to|-)\s*\$(?:US)?\s*([\d.]+)\s*(billion|million|bn|b)\b',tail,re.I)
        if not rg:continue
        lo,u1,hi,u2=rg.groups();u1=(u1 or u2).lower();u2=u2.lower()
        factors={'billion':1e9,'bn':1e9,'b':1e9,'million':1e6};lo=float(lo)*factors[u1];hi=float(hi)*factors[u2]
        if lo>hi:continue
        f=frame(symbol,key,'revenue_guidance',evidence);f.update(action='issued_range',new_value=(lo+hi)/2,lower=lo,upper=hi,unit='USD revenue',actor=symbol)
        per=re.search(r'\b(?:Q[1-4](?:\s+20\d\d)?|(?:first|second|third|fourth|final|current|next)[ -]quarter(?:\s+of\s+20\d\d)?|fiscal\s+20\d\d)\b',evidence,re.I)
        if per:f['period']=per.group(0)
        # No old guidance detected: do not invent revision or surprise from new level.
        out.append(f);break
    return out
