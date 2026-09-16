"""E14 v2: headline event anchor, matched body enrichment, conservative unknowns."""
from pathlib import Path
import importlib.util,re
sp=importlib.util.spec_from_file_location('event_rules_v1',Path(__file__).resolve().parent.parent/'stock_event_facts/extract.py');v1=importlib.util.module_from_spec(sp);sp.loader.exec_module(v1)
v1.COMP={'AAPL':r'\b(?:Apple|AAPL)\b(?:\s+Inc\.?)?','AMZN':r'\b(?:Amazon(?:\.com)?|AMZN)\b(?:\s+Inc\.?)?'}
COMP=v1.COMP;PT=v1.PT;MONEY=v1.MONEY;clean=v1.clean;sentences=v1.sentences
UP=r'\b(?:rais(?:e|es|ed)|increas(?:e|es|ed)|boost(?:s|ed)?|lift(?:s|ed)?|upped|hoisted|hikes?|hiked|revises up)\b'
DOWN=r'\b(?:cuts?|lower(?:s|ed)?|trim(?:s|med)?|reduc(?:es|ed))\b'
def direction(text):return 'up' if re.search(UP,text,re.I) else 'down' if re.search(DOWN,text,re.I) else None

def title_actor(title,symbol):
    cp=COMP[symbol]
    m=re.match(r'^(.{2,65}?)\s+(?:'+UP+'|'+DOWN+r'|Reiterates?|Reaffirms?)\s+'+cp,title,re.I)
    if m:return m.group(1).strip()
    m=re.search(r'\b(?:at|by|from)\s+([A-Z][\w.& ]{1,60}?)(?:\s+Analysts)?(?:\s+-.*)?$',title)
    return m.group(1).strip() if m else None

def extract(symbol,key,title,body):
    out=[];cp=COMP[symbol]
    roundup=bool(re.search(r'(?:price target changes|top analyst upgrades and downgrades)',title,re.I))
    # Body can enrich the current headline event only; avoid unrelated historical analyst lists.
    target_topic=bool(re.search(PT,title,re.I))
    current=v1.extract(symbol,key,title,title) if target_topic and not roundup else []
    current=[f for f in current if f['kind']=='analyst_target']
    candidates=[]
    if target_topic or roundup:
        for sentence in sentences(body):
            if re.search(r'Posted by|\b\d{1,2}/\d{1,2}/20\d\d\b|\b(?:previously|finally|last year|other analysts)\b',sentence,re.I):continue
            # One explicitly named action per evidence span prevents mixed analyst attribution.
            if len(re.findall(cp,sentence,re.I))>2:continue
            fs=v1.extract(symbol,key,title,sentence)
            for f in fs:
                if f['kind']!='analyst_target':continue
                di=direction(sentence)
                if f['change'] is None and di:f['action']=di
                if 'investment analysts at ' in sentence and f['actor'] is None:
                    m=re.search(r'by investment analysts at (.{2,50}?)(?=\s+(?:from|to|in)\b)',sentence)
                    if m:f['actor']=m.group(1)
                candidates.append(f)
    if current:
        f=current[0];di=direction(title)
        if f['change'] is None and di:f['action']=di
        f['actor']=title_actor(title,symbol);f['current_disclosure']='headline_supported'
        # Require target value + direction + known actor agreement; otherwise retain title facts.
        good=[a for a in candidates if a['new_value']==f['new_value'] and (f['action']=='set' or a['action']==f['action']) and (f['actor'] is None or (a['actor'] is not None and a['actor'].lower()==f['actor'].lower()))]
        good.sort(key=lambda x:x['old_value'] is not None,reverse=True)
        if good:
            g=good[0];g['actor']=g['actor'] or f['actor'];g['current_disclosure']='headline_supported';f=g
        f['q']=1. if f['old_value'] is not None else .5
        out.append(f)
    elif roundup and candidates:
        f=candidates[0];f['current_disclosure']='roundup_current_unknown';f['q']=.5;out.append(f)
    elif re.search(cp,title,re.I) and not target_topic:
        # Rating-only headline: never enrich it with another analyst's old target price.
        action='up' if re.search(r'\bupgrad(?:e|es|ed)\b',title,re.I) else 'down' if re.search(r'\bdowngrad(?:e|es|ed)\b',title,re.I) else 'maintain' if re.search(r'\b(?:reiterated|reiterates|reaffirmed|reaffirms|maintains|maintained)\b',title,re.I) and re.search(r'\brating\b',title,re.I) else None
        if action and not re.search(r';|weekly|recent ratings|updates|software|computers|mac pro',title,re.I):
            # Require target next to action, or target-led rating headline.
            if re.search(r'(?:upgrad\w*|downgrad\w*|reiterat\w*|reaffirm\w*)\s+(?:shares of\s+)?'+cp,title,re.I) or re.match(cp,title,re.I):
                f=v1.frame(symbol,key,'analyst_rating',title);f.update(action=action,rating_action=action,current_disclosure='headline_supported',actor=title_actor(title,symbol));out.append(f)
    for f in v1.extract(symbol,key,title,body):
        if f['kind']!='revenue_guidance':continue
        ev=f['evidence']
        # Keep a complete period span when both fiscal year and quarter are given.
        per=re.search(r'(?:fiscal\s+20\d\d\s+(?:first|second|third|fourth)\s+quarter|(?:first|second|third|fourth|final)\s+quarter\s+of\s+(?:fiscal\s+year\s+)?20\d\d)',ev,re.I)
        if per:f['period']=per.group(0)
        out.append(f)
    return out
