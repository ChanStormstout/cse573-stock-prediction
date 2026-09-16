"""Conservative post-annotation event grouping, for exclusion only, not live features.

Broker names are matched only inside supplied event evidence. This small fixed
dictionary does not certify complete entity extraction. Same explicit fact from
the same named broker within seven publication days is grouped; unknown/multiple
brokers are left unmerged. All exclusions are retained for coverage reporting.
"""
import datetime as dt
import re
from labels import signature

BROKERS = {
    'bmo':r'\bBMO\b', 'bernstein':r'\bBernstein\b|\bSacconaghi\b',
    'keybanc':r'\bKeyBanc\b|\bKeyCorp\b', 'atlantic':r'\bAtlantic (?:Equities|Securities)\b',
    'longbow':r'\bLongbow\b', 'zacks':r'\bZacks\b', 'bidaskclub':r'\bBidaskClub\b',
    'valuengine':r'\bValuEngine\b', 'vetr':r'\bVetr\b', 'cascend':r'\bCascend\b',
    'loop':r'\bLoop Capital\b', 'morgan_stanley':r'\bMorgan Stanley\b',
    'rbc':r'\bRoyal Bank of Canada\b|\bRBC\b', 'ubs':r'\bUBS\b',
    'piper':r'\bPiper Jaffray\b', 'hsbc':r'\bHSBC\b', 'barclays':r'\bBarclays\b',
    'mizuho':r'\bMizuho\b', 'maxim':r'\bMaxim\b', 'canaccord':r'\bCanaccord\b',
    'william_blair':r'\bWilliam Blair\b', 'oppenheimer':r'\bOppenheimer\b',
    'rosenblatt':r'\bRosenblatt\b', 'bank_america':r'\bBank of America\b',
    'morningstar':r'\bMorningstar\b', 'baird':r'\bBaird\b',
    'suntrust':r'\bSunTrust\b', 'stifel':r'\bStifel\b',
    'nomura':r'\bNomura\b|\bInstinet\b', 'gbh':r'\bGBH\b|\bIves\b',
    'citi':r'\bCiti(?:group)?\b', 'davidson':r'\bD\.?A\.? Davidson\b',
    'mkm':r'\bMKM\b', 'deutsche':r'\bDeutsche Bank\b', 'macquarie':r'\bMacquarie\b',
}

def group_labels(labels, inputs):
    parent={r['id']:r['id'] for r in labels};seen={};links=[]
    def root(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    for r in sorted(labels,key=lambda r:(inputs[r['id']]['published_utc'],r['id'])):
        inp=inputs[r['id']];t=dt.datetime.fromisoformat(inp['published_utc'])
        for e in r['answer']['events']:
            text=' '.join(inp['sentences'][s] for s in e['evidence_ids'])
            brokers=[k for k,regex in BROKERS.items() if re.search(regex,text,re.I)]
            if len(brokers)!=1:continue
            key=(inp['symbol'],brokers[0],signature(e))
            for old,time in seen.get(key,[]):
                if abs((t-time).total_seconds())<=7*86400:
                    parent[root(r['id'])]=root(old)
                    links.append({'left':old,'right':r['id'],'broker':brokers[0],'kind':e['kind'],'reason':'Same broker/action/values within seven publication days; conservative grouping'})
            seen.setdefault(key,[]).append((r['id'],t))
    groups={}
    for r in labels:groups.setdefault(root(r['id']),[]).append(r)
    retained=[];excluded=[]
    for rs in groups.values():
        # Preserve earliest available observation, never move future evidence backwards.
        rs.sort(key=lambda r:(inputs[r['id']]['available_utc'],r['id']))
        chosen=rs[0];retained.append(chosen)
        for r in rs[1:]:excluded.append({'id':r['id'],'kept':chosen['id'],'split':r['split'],'kept_split':chosen['split'],'reason':'post_annotation_event_duplicate'})
    return retained, excluded, links
