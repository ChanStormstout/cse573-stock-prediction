"""Second rule pilot, new article review sample; no predictive scores."""
import sys,json,datetime
from pathlib import Path
import pandas as pd
import extract
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import *
if __name__=='__main__':
    R=B/'results';require_empty(R);R.mkdir(exist_ok=True);old=O/'stock_event_facts/results'
    paths=[B/'extract.py',B/'prepare.py',O/'stock_event_facts/extract.py',old/'bodies.json',old/'article_pairs.csv',old/'assistant_review_v1.json',O.parent/'work/stock-data/audit/news_index.pkl']
    P=dict(id='E14_V2_PILOT',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),changes='headline anchored event; matching body price/action/actor; rating-only headline does not acquire historical PT; action morphology fixed; Inc suffix fixed; complete fiscal period when present',review='exclude v1 review article keys; random 30 AAPL accepted, all AMZN accepted; unseen training candidates only; >=90% key-fact correctness for provisional predictor use; independent human review pending; guidance too few fresh checks remains excluded from downstream main experiment',source_sha256={str(p.relative_to(O.parent)):sha(p) for p in paths});(B/'pilot_protocol.json').write_text(json.dumps(P,indent=2))
    pairs=pd.read_csv(old/'article_pairs.csv');bodies=json.loads((old/'bodies.json').read_text());n=pd.read_pickle(O.parent/'work/stock-data/audit/news_index.pkl');n['record_key']=n.archive+'::'+n.member;n=n.set_index('record_key');records=[]
    for r in pairs.itertuples():
        row=n.loc[r.record_key]
        for f in extract.extract(r.symbol,r.record_key,row.title,bodies[r.record_key]):
            f.update(split=r.split,title=row.title,published_utc=str(row.published_utc),available_utc=str(row.available_utc),lag_hours=float(row.lag_hours));records.append(f)
    (R/'events.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));seen={x['record_key'] for x in json.loads((old/'review_input.json').read_text())};v=pd.DataFrame(records);new=v[v.split.eq('train')&~v.record_key.isin(seen)&v.kind.ne('revenue_guidance')];review=[]
    for s,g in new.groupby('symbol'):
        review+=g.sample(min(30,len(g)),random_state=575).to_dict('records')
    for i,r in enumerate(review):r.update(review_id=i,assistant_key_facts_correct=None,human_key_facts_correct=None,review_note='')
    (R/'review_input.json').write_text(json.dumps(review,indent=2,ensure_ascii=False));verify_files(O.parent,P['source_sha256']);print(v.groupby(['symbol','split','kind']).size().to_string());print('new checks',len(review))
