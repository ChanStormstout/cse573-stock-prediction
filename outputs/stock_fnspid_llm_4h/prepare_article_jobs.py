#!/usr/bin/env python3
"""Prepare outcome-blind binary LLM jobs for added FNSPID articles."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
PUBLIC=HERE/'v1'; PRIVATE=ROOT/'work/stock-data/fnspid_llm_4h/v1'
SOURCE=ROOT/'work/stock-data/fnspid_augmented_4h/v1'
MODEL=ROOT/'work/stock-data/model_compare/model'
ALIASES={'AAPL':('apple','aapl','iphone','ipad','macbook'),'AMZN':('amazon','amzn','aws')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def sentences(text):return [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+',str(text)) if x.strip()]
def snippet(row):
    ss=sentences(row.body); aliases=ALIASES[row.symbol]
    hit=[s for s in ss if any(a in s.lower() for a in aliases)]
    chosen=(hit[:5] if hit else ss[:4]); value=('TITLE: '+str(row.title)+'\nEVIDENCE:\n'+'\n'.join(chosen))[:2600]
    return value

SYSTEM=("You are checking financial news for a four-hour forecasting system. "
        "Use only the supplied text and earlier reports. Do not predict the stock. "
        "Return exactly A or B with no explanation.")

def main():
    if PUBLIC.exists() or PRIVATE.exists():raise FileExistsError('v1 exists; immutable lane')
    PUBLIC.mkdir(parents=True);PRIVATE.mkdir(parents=True)
    articles=pd.read_parquet(SOURCE/'fnspid_articles.parquet').sort_values(['symbol','available_at_utc','article_key']).reset_index(drop=True)
    z=np.load(SOURCE/'combined_embeddings.npz');lookup={k:i for i,k in enumerate(z['keys'].astype(str))};emb=z['finbert'].astype(float)
    vectors=emb[[lookup[k] for k in articles.article_key]];vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    rows=[];max_prior=0
    for i,r in enumerate(articles.itertuples(index=False)):
        available=pd.Timestamp(r.available_at_utc)
        prior_mask=(articles.symbol.eq(r.symbol)&(pd.to_datetime(articles.available_at_utc,utc=True)<available)).to_numpy()
        pool=np.flatnonzero(prior_mask)
        if len(pool):
            sim=vectors[pool]@vectors[i];order=pool[np.argsort(-sim,kind='stable')[:3]]
        else:order=[]
        prior=[{'date':str(pd.Timestamp(articles.iloc[j].available_at_utc).date()),'title':str(articles.iloc[j].title)[:300],'similarity':float(vectors[j]@vectors[i])} for j in order]
        max_prior=max(max_prior,len(prior));past='\n'.join(f"P{n+1} [{x['date']}]: {x['title']}" for n,x in enumerate(prior)) or 'NO EARLIER MATCH'
        current=snippet(r)
        prompts={
          'quality':f"TARGET={r.symbol}\nCURRENT\n{current}\nEARLIER REPORTS\n{past}\n\nA = the current article contains a specific factual development about TARGET that is useful as current company information.\nB = it is mainly a repeat, historical/background item, market-price recap, ranking, holdings/portfolio template, unrelated mention, or uncertain.\nANSWER:",
          'novelty':f"TARGET={r.symbol}\nCURRENT\n{current}\nEARLIER REPORTS\n{past}\n\nA = current supplies a new or changed target-company fact relative to the earlier reports.\nB = it repeats the same fact, supplies background/recap only, or novelty cannot be supported.\nANSWER:",
          'polarity':f"TARGET={r.symbol}\nCURRENT\n{current}\n\nJudge only explicit target-company changes, not stock-price movement.\nA = the supported change is more positive than negative for TARGET.\nB = the supported change is more negative than positive for TARGET. If neutral or unclear, choose the closer option conservatively.\nANSWER:",
        }
        for task,prompt in prompts.items():rows.append({'job_id':f'{r.article_key}::{task}','article_key':r.article_key,'symbol':r.symbol,'task':task,'system':SYSTEM,'prompt':prompt,'prior_count':len(prior)})
    with (PRIVATE/'article_jobs.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    model_files={p.name:sha(p) for p in MODEL.glob('*') if p.is_file()}
    manifest={'status':'PREPARED','articles':len(articles),'jobs':len(rows),'tasks':['quality','novelty','polarity'],'labels_or_returns_loaded':False,'max_prior':max_prior,'source_hashes':{str((SOURCE/'fnspid_articles.parquet').relative_to(ROOT)):sha(SOURCE/'fnspid_articles.parquet'),str((SOURCE/'combined_embeddings.npz').relative_to(ROOT)):sha(SOURCE/'combined_embeddings.npz'),str((HERE/'PRE_REGISTRATION.md').relative_to(ROOT)):sha(HERE/'PRE_REGISTRATION.md')},'private_jobs_sha256':sha(PRIVATE/'article_jobs.jsonl'),'model_files':model_files,'model_revision':'8b2b98c00a6b4d291155e4890773ca8f769aee53'}
    dump(PUBLIC/'ARTICLE_JOB_AUDIT.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='model_files'},indent=2))
if __name__=='__main__':main()
