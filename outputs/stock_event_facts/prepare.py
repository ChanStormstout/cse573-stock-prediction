"""Register rule pilot, extract train-only review material before predictive use."""
from pathlib import Path
import sys,json,datetime,zipfile,re
import pandas as pd,numpy as np
import extract
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import load_data,sha,require_empty,verify_files
if __name__=='__main__':
    R=B/'results';require_empty(R);R.mkdir(exist_ok=True)
    source=[B/'extract.py',B/'prepare.py',O/'stock_adaptive/common.py',O/'stock_robust/results/data.pkl',O/'stock_final_test/results/test_inputs.pkl',O.parent/'work/stock-data/audit/news_index.pkl']
    P=dict(id='E14_PILOT',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),types=['analyst_target','revenue_guidance'],rule='explicit company/object templates; unknowns retained; no forecasts from other analysts as company guidance',review='training-period accepted and rejected candidates; assistant evidence review is provisional, not independent human gold; >=90% correct key facts per supported type needed for exploratory expansion; human signoff remains required for formal quality claim',quality_keys=['company/object','event action','old/new number pairing','evidence supports known fields'],first_disclosure='unknown; published age proxy only',source_sha256={str(p.relative_to(O.parent)):sha(p) for p in source})
    (B/'pilot_protocol.json').write_text(json.dumps(P,indent=2));D=load_data();news=pd.read_pickle(O.parent/'work/stock-data/audit/news_index.pkl');news['record_key']=news.archive+'::'+news.member;news=news.set_index('record_key');pairs=[]
    for (s,split),g in D.groupby(['symbol','split']):
        keys=sorted(set(k for seq in g.news_record_keys for k in seq.split('|') if k));pairs.extend([dict(symbol=s,split=split,record_key=k) for k in keys])
    pairs=pd.DataFrame(pairs);assert not pairs.duplicated(['symbol','record_key']).any()
    keys=sorted(set(pairs.record_key));bodies={};rawhash={}
    for archive,g in news.loc[keys].groupby('archive'):
        path=O.parent/'work/stock-data/raw/news'/archive;rawhash[str(path.relative_to(O.parent))]=sha(path)
        with zipfile.ZipFile(path) as z:
            for k,row in g.iterrows():bodies[k]=json.loads(z.read(row.member)).get('text','')
    (R/'bodies.json').write_text(json.dumps(bodies));(R/'raw_sha256.json').write_text(json.dumps(rawhash,indent=2))
    records=[];candidates=[]
    for r in pairs.itertuples():
        row=news.loc[r.record_key];found=extract.extract(r.symbol,r.record_key,row.title,bodies[r.record_key])
        for f in found:f.update(split=r.split,title=row.title,published_utc=str(row.published_utc),available_utc=str(row.available_utc),lag_hours=float(row.lag_hours));records.append(f)
        if r.split=='train' and (found or re.search(r'price target|target price|guidance|outlook|forecast|upgrades and downgrades',row.title,re.I)):
            candidates.append(dict(symbol=r.symbol,record_key=r.record_key,title=row.title,accepted=bool(found),frames=found))
    (R/'events.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));pairs.to_csv(R/'article_pairs.csv',index=False)
    # One accepted event per article in review; all available guidance, random analyst cases.
    c=pd.DataFrame(candidates);accepted=c[c.accepted];rejected=c[~c.accepted]
    accepted=accepted.sample(frac=1,random_state=573)
    checks=[]
    for s in ['AAPL','AMZN']:
        checks+=accepted[accepted.symbol.eq(s)].head(30).to_dict('records')
        checks+=rejected[rejected.symbol.eq(s)].sample(min(10,len(rejected[rejected.symbol.eq(s)])),random_state=574).to_dict('records')
    existing={r['record_key'] for r in checks}
    checks += [r for r in candidates if any(f['kind']=='revenue_guidance' for f in r['frames']) and r['record_key'] not in existing]
    for i,r in enumerate(checks):r.update(review_id=i,assistant_key_facts_correct=None,human_key_facts_correct=None,review_note='')
    (R/'review_input.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False));c.drop(columns='frames').to_csv(R/'training_candidates.csv',index=False)
    verify_files(O.parent,P['source_sha256']);summary=pd.DataFrame(records).groupby(['symbol','split','kind']).size().to_dict();print(summary,flush=True);print('review cases',len(checks),flush=True)
