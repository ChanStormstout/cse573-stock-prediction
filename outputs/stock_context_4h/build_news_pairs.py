"""Outcome-blind, body-backed target association and family-safe pair sampling."""
from __future__ import annotations
import hashlib,json,re,zipfile
from collections import Counter,defaultdict,deque
from pathlib import Path
import pandas as pd
from outputs.stock_paper_methods_4h.run import compatible,signature

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'audit_v2'; PRIVATE=ROOT/'work/stock-data/context_increment_4h/audit_v2'; RAW=ROOT/'work/stock-data/raw/news'
TICKER={'AAPL':re.compile(r'(?<![A-Z0-9])AAPL(?![A-Z0-9])',re.I),'AMZN':re.compile(r'(?<![A-Z0-9])AMZN(?![A-Z0-9])',re.I)}
LEGAL={'AAPL':re.compile(r'\bApple\s*(?:,?\s*Inc\.?|Corporation)\b',re.I),'AMZN':re.compile(r'\bAmazon(?:\.com)?\s*(?:,?\s*Inc\.?|Corporation)\b',re.I)}
COMMON={'AAPL':re.compile(r'\bApple\b',re.I),'AMZN':re.compile(r'\bAmazon\b',re.I)}
EX={'AAPL':('apple pie','apple valley','apple sauce','apple picking','apple orchard','apple tree'),'AMZN':('amazon rainforest','amazon river','amazonian','amazon jungle')}
PHYS=re.compile(r'american\s+association\s+for\s+physician\s+leadership',re.I); APPLE_IND=re.compile(r'\bApple\s*(?:,?\s*(?:Inc\.?|Corporation)|stock|shares|earnings|iPhone|Mac|supplier|investors?)\b|\b(?:NASDAQ|NYSE)\s*:\s*AAPL\b|\bApple\s*\(\s*AAPL\s*\)',re.I)
STRATA=('near_duplicate_candidate','moderate_overlap_numeric_difference','moderate_overlap_same_numeric_tokens','weak_overlap_candidate')

def missing_hash(value):
    if value is None or pd.isna(value): return None
    value=str(value).strip()
    return None if not value or value.lower() in {'nan','none'} else value
def node(target,key): return f'{target}|{key}'
def pair_id(target,current,past): return hashlib.sha256(f'{target}|{current}|{past}'.encode()).hexdigest()[:16]
def split_sentences(body):
    """Deterministic sentence/chunk split; pathological unbroken text is chunked, never dropped."""
    out=[]
    for sentence in [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+',body or '') if x.strip()]:
        while len(sentence)>1600:
            cut=sentence.rfind(' ',0,1601)
            if cut<=0: cut=1600
            out.append(sentence[:cut].strip());sentence=sentence[cut:].strip()
        if sentence:out.append(sentence)
    return out
def association(target,title,body):
    title=title or '';body=body or '';both=title+'\n'+body;ss=[title]+split_sentences(body)
    if target=='AAPL' and PHYS.search(both) and TICKER[target].search(both) and not LEGAL[target].search(both) and not APPLE_IND.search(both): return None
    for i,text in enumerate(ss):
        if TICKER[target].search(text) or LEGAL[target].search(text): return (0,'ticker_or_legal',i,[i])
    if COMMON[target].search(title) and not any(x in both.casefold() for x in EX[target]): return (1,'title_company_name',0,[0])
    for i,text in enumerate(ss[1:],1):
        if COMMON[target].search(text) and not any(x in text.casefold() for x in EX[target]): return (2,'body_company_sentence',i,list(range(max(1,i-1),min(len(ss),i+2))))
    return None
def title_tokens(title): return set(re.findall(r'[a-z]+|\d+(?:\.\d+)?',str(title).lower()))

def load_associated():
    index=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl'); zips={}; rows=[]; errors=Counter()
    try:
      for row in index.itertuples(index=False):
        try:
            if row.archive not in zips: zips[row.archive]=zipfile.ZipFile(RAW/row.archive)
            obj=json.loads(zips[row.archive].read(row.member)); title=str(obj.get('title') or row.title or ''); body=str(obj.get('text') or '')
        except Exception:
            errors['json_read_error']+=1; continue
        key=f'{row.archive}::{row.member}'
        for target in ('AAPL','AMZN'):
            e=association(target,title,body)
            if e:
                rows.append({'target':target,'article_key':key,'title':title,'available_utc':row.available_utc,'exact_hash':missing_hash(row.exact_hash),'normalized_hash':missing_hash(row.normalized_hash),'association_tier':e[0],'association_evidence_type':e[1],'association_sentence_id':e[2],'context_sentence_ids':e[3]})
    finally:
      for z in zips.values(): z.close()
    out=pd.DataFrame(rows);out.available_utc=pd.to_datetime(out.available_utc,utc=True); return out.sort_values(['target','available_utc','article_key']).reset_index(drop=True),dict(errors)

def families(full):
    """Target-local DSU over hashes and seven-day title-compatible neighborhoods."""
    keys=[node(r.target,r.article_key) for r in full.itertuples()]; parent={k:k for k in keys}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for target,g in full.groupby('target',sort=False):
        title_by_key=dict(zip(g.article_key,g.title)); sig_by_key={k:signature(v) for k,v in title_by_key.items()}
        # exact / normalized hashes cannot union when missing.
        for col in ('exact_hash','normalized_hash'):
            for _,bucket in g[g[col].notna()].groupby(col):
                ns=[node(target,x) for x in bucket.article_key]
                for n in ns[1:]: union(ns[0],n)
        # Inverted title-token index restricts compatible comparisons to a 7-day neighborhood.
        active=defaultdict(lambda: defaultdict(set)); queue=deque()
        for pos,r in enumerate(g.itertuples(index=False)):
            cutoff=r.available_utc-pd.Timedelta(days=7)
            while queue and queue[0][0] < cutoff:
                _,old,old_tokens,old_guard=queue.popleft()
                for t in old_tokens: active[old_guard][t].discard(old)
            tokens,guard=signature(r.title); candidates=set()
            for t in tokens: candidates.update(active[guard][t])
            for old in candidates:
                if guard==sig_by_key[old][1] and compatible(r.title,title_by_key[old]): union(node(target,r.article_key),node(target,old))
            queue.append((r.available_utc,r.article_key,tokens,guard))
            for t in tokens: active[guard][t].add(r.article_key)
    root={k:find(k) for k in parent}; ids={k:hashlib.sha256(v.encode()).hexdigest()[:16] for k,v in root.items()}; sizes=Counter(ids.values());return ids,sizes

def build_pairs(full):
    eligible=full[full.available_utc.dt.strftime('%Y-%m').between('2018-03','2018-08')].copy(); rows=[]
    for target,current in eligible.groupby('target',sort=False):
        base=full[full.target==target].reset_index(drop=True); values=base.available_utc.astype('int64').to_numpy()
        for r in current.itertuples(index=False):
            right=values.searchsorted(r.available_utc.value,side='left');left=values.searchsorted((r.available_utc-pd.Timedelta(days=7)).value,side='left');prior=base.iloc[left:right]
            if prior.empty:continue
            tt=title_tokens(r.title)
            sim=lambda x:len(tt&title_tokens(x))/max(1,len(tt|title_tokens(x)))
            p=prior.assign(_sim=prior.title.map(sim)).sort_values(['_sim','available_utc','article_key'],ascending=[False,False,True]).iloc[0]
            nums=lambda x:set(re.findall(r'\d+(?:\.\d+)?',x)); equal=nums(r.title)==nums(p.title);s=float(p._sim)
            st='near_duplicate_candidate' if s>=.8 else ('moderate_overlap_numeric_difference' if s>=.25 and not equal else ('moderate_overlap_same_numeric_tokens' if s>=.25 else 'weak_overlap_candidate'))
            rows.append({'pair_id':pair_id(target,r.article_key,p.article_key),'target':target,'current_key':r.article_key,'past_key':p.article_key,'current_available_utc':r.available_utc.isoformat(),'past_available_utc':p.available_utc.isoformat(),'title_similarity':s,'stratum':st,'same_exact_hash':bool(r.exact_hash and p.exact_hash and r.exact_hash==p.exact_hash),'same_normalized_hash':bool(r.normalized_hash and p.normalized_hash and r.normalized_hash==p.normalized_hash),'numeric_sets_equal':equal,'association_tier_current':r.association_tier,'association_tier_past':p.association_tier,'association_sentence_id_current':r.association_sentence_id,'association_sentence_id_past':p.association_sentence_id,'context_sentence_ids_current':r.context_sentence_ids,'context_sentence_ids_past':p.context_sentence_ids})
    candidates=pd.DataFrame(rows).drop_duplicates(['target','current_key','past_key']).sort_values(['stratum','target','pair_id']); selected=pd.concat([candidates[(candidates.stratum==s)&(candidates.target==t)].head(8) for s in STRATA for t in ('AAPL','AMZN')],ignore_index=True); return selected,{'eligible_current_AAPL':int((eligible.target=='AAPL').sum()),'eligible_current_AMZN':int((eligible.target=='AMZN').sum())}

def main():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    if (PRIVATE/'news_pairs_private.jsonl').exists(): raise FileExistsError('refuse to overwrite pair build')
    full,errors=load_associated(); selected,candidates=build_pairs(full); fam,sizes=families(full); selected['current_family_id']=[fam[node(r.target,r.current_key)] for r in selected.itertuples()];selected['past_family_id']=[fam[node(r.target,r.past_key)] for r in selected.itertuples()];selected['pair_family_key']=selected.apply(lambda r:hashlib.sha256('|'.join(sorted([r.current_family_id,r.past_family_id])).encode()).hexdigest()[:16],axis=1)
    # Pair components share either endpoint family.
    pp={x:x for x in selected.pair_id}
    def pf(x):
        while pp[x]!=x:pp[x]=pp[pp[x]];x=pp[x]
        return x
    for i,a in selected.iterrows():
        af={a.current_family_id,a.past_family_id}
        for j in range(i):
            b=selected.iloc[j]
            if af & {b.current_family_id,b.past_family_id}:
                x,y=pf(a.pair_id),pf(b.pair_id)
                if x!=y:pp[max(x,y)]=min(x,y)
    selected['pair_component_id']=selected.pair_id.map(pf);selected['split']=selected.pair_component_id.map(lambda x:'locked_check' if int(hashlib.sha256(x.encode()).hexdigest()[:2],16)%2 else 'pilot')
    allfam={q:set(selected.loc[selected.split==q,'current_family_id'])|set(selected.loc[selected.split==q,'past_family_id']) for q in ('pilot','locked_check')};cross=len(allfam['pilot']&allfam['locked_check']);assert cross==0
    selected.to_json(PRIVATE/'news_pairs_private.jsonl',orient='records',lines=True);selected.drop(columns=['current_key','past_key']).to_csv(OUT/'news_pair_manifest.csv',index=False)
    counts=selected.groupby(['target','stratum','split']).size().reset_index(name='count').to_dict('records');audit={'selected_pair_count':len(selected),'unique_selected_article_target_nodes':len({node(r.target,r.current_key) for r in selected.itertuples()}|{node(r.target,r.past_key) for r in selected.itertuples()}),'full_family_graph_node_count':len(fam),'unique_family_count':len(sizes),'pilot_pair_count':int((selected.split=='pilot').sum()),'locked_pair_count':int((selected.split=='locked_check').sum()),'cross_split_family_count':cross,'largest_family_size':max(sizes.values()),'largest_pair_component_size':int(selected.pair_component_id.value_counts().max()),'pair_component_count':int(selected.pair_component_id.nunique()),'stock_stratum_split_counts':counts,'candidate_counts':candidates,'json_read_error':errors.get('json_read_error',0)};(OUT/'family_split_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
