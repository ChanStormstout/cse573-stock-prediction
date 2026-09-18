"""Prepare outcome-blind retrieval, then attach only matured training outcomes."""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from common import *

def main():
    if (PRIVATE/'manifest.json').exists():
        check_prepared();print('Matching preparation already complete');return
    if PRIVATE.exists(): raise FileExistsError('Incomplete preparation; archive explicitly before retry')
    PRIVATE.mkdir(parents=True);OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'protocol.json',dict(registration_sha256=sha(HERE/'PRE_REGISTRATION.md'),amendment_sha256=sha(HERE/'V2_REGISTRATION.md'),
                                inference='frozen Qwen3.5-9B',selection='fixed; no label-tuned hyperparameters',
                                evaluation='exposed exploratory history',seed=573))
    data=pd.read_pickle(W/'paper_methods_4h/v1/inputs.pkl').reset_index(drop=True)
    packs={r['key']:r for r in jsonl(PACKS)}
    assert len(data)==len(packs)==1607
    docs=[]; seen=[];groups=[]
    # Group chronologically, never rewrite old groups based on later reports.
    for i,r in data.iterrows():
        inp,key=pack_current(packs[r.key])
        docs.append(None if inp is None else dict(input=inp,record_key=key,
                    category=category(inp['news']['title'])))
    order=sorted(range(len(data)),key=lambda i:(data.iloc[i].cutoff_utc,data.iloc[i].key))
    groups=[None]*len(data)
    for i in order:
        if docs[i] is None:continue
        inp=docs[i]['input'];g=None
        for j in seen:
            other=docs[j]['input']
            if inp['symbol']==other['symbol'] and (docs[i]['record_key']==docs[j]['record_key'] or
                inp['news']['evidence']==other['news']['evidence'] or
                jac(inp['news']['title'],other['news']['title'])>=.65):
                g=groups[j];break
        if g is None:g=f'G{len(seen):04d}'
        groups[i]=g;seen.append(i)
    import re
    texts=['' if d is None else re.sub(r'\b(?:apple|aapl|amazon|amzn|inc|nasdaq|com|stock|stocks|share|shares|company|market)\b',' ',
           (d['input']['news']['title']+' '+d['input']['news']['evidence']).lower()) for d in docs]
    vec=HashingVectorizer(n_features=16384,alternate_sign=False,ngram_range=(1,2),norm='l2',stop_words='english')
    x=vec.transform(texts);sim=(x@x.T).toarray()
    # Original outcomes, calculated by program only after input representation is fixed.
    returns=[]
    bars={}
    for s,name in [('AAPL','APPLE'),('AMZN','AMAZON')]:
        b=pd.read_csv(W/f'raw/CHARTS/{name}5.csv',header=None,names=['date','time','open','high','low','close','activity'])
        b.index=pd.to_datetime(b.date+' '+b.time,format='%Y.%m.%d %H:%M',utc=True);bars[s]=b
    for r in data.itertuples():
        ix=pd.date_range(r.start_utc,r.end_utc-pd.Timedelta('5min'),freq='5min');v=bars[r.symbol].reindex(ix)
        assert len(v)==48 and v[['open','high','low','close']].notna().all().all()
        ret=float(100*(v.close.iloc[-1]/v.open.iloc[0]-1));assert int(ret>0)==r.label
        returns.append(ret)
    base=pd.read_csv(ROOT/'outputs/stock_foundation_4h/v1/all_predictions.csv',float_precision='round_trip').set_index('key')
    records=[];audit=[];prompts=[]
    for i,r in data.iterrows():
        phase='warmup' if r.month<'2018-03' else ('train_oof' if r.month<'2018-09' else ('development' if r.month<'2018-11' else 'later'))
        a=select_cases(i,data,docs,sim[i],groups)
        fallback=None if phase=='warmup' else float(base.loc[r.key,'R1'])
        historical=[]
        for c in a:
            j=c['index']; rr=data.iloc[j]
            assert rr.month<min(r.month,'2018-09') and rr.end_utc<r.cutoff_utc
            historical.append(dict(case_id=f'H{j:04d}',input=docs[j]['input'],similarity=round(c['score'],5),
                                   outcome=dict(return_pct=round(returns[j],6),direction='UP' if returns[j]>0 else 'DOWN')))
        vote=(1+sum(c['score']*int(returns[c['index']]>0) for c in a))/(2+sum(c['score'] for c in a)) if a else fallback
        records.append(dict(key=r.key,symbol=r.symbol,day=str(r.day),month=r.month,phase=phase,label=int(r.label),
                            has_original_news=int(base.loc[r.key,'has_original_news']),has_excerpt=int(docs[i] is not None),
                            n_cases=len(a),R1=fallback,P1=vote,**{m:None if phase=='warmup' else float(base.loc[r.key,m]) for m in ['F0','F1','F2','F6']}))
        audit.append(dict(index=i,key=r.key,group=groups[i],query_cutoff=str(r.cutoff_utc),case_indices=[c['index'] for c in a],
                          case_groups=[groups[c['index']] for c in a],case_ends=[str(data.iloc[c['index']].end_utc) for c in a],
                          case_months=[data.iloc[c['index']].month for c in a],scores=[c['score'] for c in a]))
        if phase=='warmup' or docs[i] is None:continue
        for variant in ['P0','P2','P3']:
            if variant!='P0' and not a:continue
            payload={'CURRENT':docs[i]['input']}
            if variant!='P0':
                payload['HISTORICAL_CASES']=[{k:v for k,v in h.items() if variant=='P3' or k!='outcome'} for h in historical]
            prompts.append(dict(key=r.key,variant=variant,messages=[{'role':'system','content':SYSTEM},
                            {'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]))
    pd.DataFrame(records).to_csv(PRIVATE/'prepared.csv',index=False)
    write_jsonl(PRIVATE/'prompts.jsonl',prompts);dump(PRIVATE/'retrieval.json',audit)
    dump(PRIVATE/'documents.json',docs);data[['key','symbol','day','month','label','start_utc','end_utc','cutoff_utc']].to_pickle(PRIVATE/'rows.pkl')
    np.savez_compressed(PRIVATE/'retrieval_arrays.npz',similarity=sim,returns=np.array(returns))
    eligible=pd.DataFrame(records).query('phase != "warmup"')
    cases=[]
    for _,g in eligible.assign(has_analog=lambda x:x.n_cases.gt(0)).groupby(['symbol','phase','has_analog']):
        cases.append(min(g.key,key=digest))
    dump(PRIVATE/'case_selection.json',cases)
    summary=eligible.groupby(['symbol','phase']).agg(windows=('key','size'),current_excerpt=('has_excerpt','sum'),
        with_analog=('n_cases',lambda x:int((x>0).sum())),mean_cases=('n_cases','mean')).reset_index()
    summary.to_csv(OUT/'coverage.csv',index=False)
    dump(OUT/'input_audit.json',dict(windows=len(data),labels_match_original=True,
        eligible_windows=len(eligible),unique_groups=len(set(g for g in groups if g is not None)),
        inference_counts=pd.Series([p['variant'] for p in prompts]).value_counts().to_dict(),
        source_text_public=False,case_selection_sha256=sha(PRIVATE/'case_selection.json')))
    files=['prepared.csv','prompts.jsonl','retrieval.json','documents.json','rows.pkl','retrieval_arrays.npz','case_selection.json']
    dump(PRIVATE/'manifest.json',dict(sources=fingerprint(),artifacts={f:sha(PRIVATE/f) for f in files}))
    print(summary.to_string(index=False)); print('prompts',len(prompts))

if __name__=='__main__':main()
